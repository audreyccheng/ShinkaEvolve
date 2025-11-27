# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements a completely different rearrangement algorithm:

- Replica allocation via hybrid divisor apportionment:
  D’Hondt for the bulk seats, then a micro-AB tail between
  Huntington–Hill and Sainte–Laguë to minimize the post-allocation peak.

- Group-to-node placement by balanced round-robin (BRR) with strict capacity.

- GPU placement within nodes by Latin dispersion per label (logical expert),
  followed by a tiny capacity-fixing move from overfull to underfull GPUs
  prioritizing heavy items, and a single guarded 1x1 swap.

This keeps the same public APIs and tensor shapes, but follows a different
algorithmic strategy from the previous implementations.
"""

import math
import torch
from collections import defaultdict
from typing import Tuple


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64,
                     device=perm.device).expand(perm.shape),
    )
    return inv


def _balanced_groups_round_robin(tokens_per_group: torch.Tensor,
                                 num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced round-robin (BRR) assignment of groups to nodes with strict capacity.

    Parameters:
        tokens_per_group: [L, G], total tokens per group (float)
        num_nodes: number of nodes

    Returns:
        group_pack_index: [L, G] -> node id for each group
        group_rank_in_pack: [L, G] -> rank within node's group slots [0..groups_per_node-1]
    """
    L, G = tokens_per_group.shape
    assert G % num_nodes == 0
    cap = G // num_nodes

    # Sort groups by tokens descending for each layer, then cyclically assign to nodes,
    # skipping nodes that hit capacity.
    order = tokens_per_group.argsort(dim=-1, descending=True)  # [L, G]
    group_pack_index = torch.full((L, G), -1, dtype=torch.int64, device=tokens_per_group.device)
    group_rank_in_pack = torch.full_like(group_pack_index, -1)

    for i in range(L):
        node_counts = [0] * num_nodes
        ptr = 0  # current node pointer
        for gg in order[i].tolist():
            # find next node with available capacity
            start = ptr
            while node_counts[ptr] >= cap:
                ptr = (ptr + 1) % num_nodes
                if ptr == start:
                    break  # all full (should not happen)
            # assign
            node = ptr
            rank = node_counts[node]
            group_pack_index[i, gg] = node
            group_rank_in_pack[i, gg] = rank
            node_counts[node] += 1
            # advance ptr cyclically
            ptr = (ptr + 1) % num_nodes

    return group_pack_index, group_rank_in_pack


def _counts_divisor_tail_choice(
    w: torch.Tensor,
    base_counts: torch.Tensor,
    tail: int,
) -> torch.Tensor:
    """
    Given base_counts (after bulk seats), simulate two tail strategies for T seats:
      - Huntington–Hill (HH): priority = w / sqrt(c*(c+1))
      - Sainte–Laguë (SL):   priority = w / (c + 0.5)
    Pick the tail that yields lower predicted peak (ties by second-highest).

    Returns counts for the chosen tail.
    """
    n = w.numel()
    if tail <= 0 or n == 0:
        return base_counts

    def simulate_tail(c_init: torch.Tensor, mode: str) -> torch.Tensor:
        c = c_init.clone()
        for _ in range(tail):
            if mode == "HH":
                prio = w / torch.sqrt((c.to(w.dtype)) * (c.to(w.dtype) + 1.0))
            else:  # "SL"
                prio = w / (c.to(w.dtype) + 0.5)
            j = int(torch.argmax(prio).item())
            c[j] += 1
        return c

    c_hh = simulate_tail(base_counts, "HH")
    c_sl = simulate_tail(base_counts, "SL")

    avg_hh = w / c_hh.to(w.dtype)
    avg_sl = w / c_sl.to(w.dtype)
    max_hh = float(avg_hh.max().item())
    max_sl = float(avg_sl.max().item())

    if abs(max_hh - max_sl) > 1e-12:
        return c_hh if max_hh < max_sl else c_sl

    # tie-break by second-highest
    def second_highest(x: torch.Tensor) -> float:
        vals, _ = torch.sort(x, descending=True)
        if vals.numel() >= 2:
            return float(vals[1].item())
        return float(vals[0].item()) if vals.numel() == 1 else 0.0

    sh_hh = second_highest(avg_hh)
    sh_sl = second_highest(avg_sl)
    return c_hh if sh_hh < sh_sl else c_sl


def _apportion_counts_row_hybrid(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Hybrid divisor apportionment per row with:
      - D’Hondt (Jefferson) for the bulk seats for speed
      - Adaptive tail: micro AB between Huntington–Hill and Sainte–Laguë
    to minimize the predicted peak average.

    Parameters:
        w: [n] float CPU
        target_total: total seats (replicas)

    Returns:
        counts: [n] int64, sum == target_total, each >= 1
    """
    n = w.numel()
    assert target_total >= n
    if n == 0:
        return torch.zeros(0, dtype=torch.int64, device=w.device)

    # Start with one seat per item
    counts = torch.ones(n, dtype=torch.int64, device=w.device)
    extras = target_total - n
    if extras == 0:
        return counts

    # Adaptive tail length based on dispersion (clamped)
    mean = float(w.mean().item()) if n > 0 else 0.0
    std = float(w.std(unbiased=False).item()) if n > 0 else 0.0
    cv = (std / max(mean, 1e-12)) if mean > 0 else 1.0
    s = min(1.3, max(0.7, cv))
    alpha = 0.10
    tail = int(max(1, min(extras, round(alpha * extras * s))))
    bulk = extras - tail

    # Bulk by D’Hondt: priority = w / (c+1)
    if bulk > 0:
        # Use a simple loop with argmax; n is typically small to medium per node.
        for _ in range(bulk):
            prio = w / (counts.to(w.dtype) + 1.0)
            j = int(torch.argmax(prio).item())
            counts[j] += 1

    # Tail decision between HH and SL
    counts = _counts_divisor_tail_choice(w, counts, tail)
    return counts


def _replicate_experts_apportion(
    weight: torch.Tensor,
    num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replica allocation via hybrid divisor apportionment per row.

    Parameters:
        weight: [R, num_log] float CPU
        num_phy: integer seats per row

    Returns:
        phy2log: [R, num_phy] int64
        rank:    [R, num_phy] int64
        logcnt:  [R, num_log] int64
    """
    R, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device

    phy2log_rows = []
    rank_rows = []
    cnt_rows = []
    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)

    for i in range(R):
        w = weight[i]
        counts = _apportion_counts_row_hybrid(w, num_phy)
        cnt_rows.append(counts)

        # Build phy2log and rank
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)
        phy2log_rows.append(phy2log_i)
        rank_rows.append(rank_i)

    phy2log = torch.stack(phy2log_rows, dim=0)
    rank = torch.stack(rank_rows, dim=0)
    logcnt = torch.stack(cnt_rows, dim=0)
    return phy2log, rank, logcnt


def _latin_spread_pack_to_gpus(
    weights: torch.Tensor,  # [R, N] per-replica loads
    labels: torch.Tensor,   # [R, N] label per replica (logical id within node)
    gpus_per_node: int,
    cap_per_gpu: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Latin dispersion: spread replicas of the same label across GPUs by a
    circulant assignment, then fix capacities by moving heaviest items from
    overfull GPUs to the most underfull GPUs. Optionally apply a single
    heaviest↔lightest 1x1 swap when strictly improving the global peak.

    Returns:
        pack_index: [R, N] gpu id in [0..g-1]
        rank_in_pack: [R, N] position within that gpu
    """
    R, N = weights.shape
    device = weights.device

    pack_index = torch.full((R, N), -1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full_like(pack_index, -1)

    if gpus_per_node == 1:
        # Trivial
        pack_index.fill_(0)
        rank_in_pack.copy_(torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(R, N))
        return pack_index, rank_in_pack

    for r in range(R):
        row_w = weights[r]
        row_lab = labels[r]

        # Group items by label
        by_label = defaultdict(list)
        for idx in range(N):
            lab = int(row_lab[idx].item())
            by_label[lab].append(idx)

        # Initial Latin dispersion
        gpu_lists = [[] for _ in range(gpus_per_node)]
        for lab, items in by_label.items():
            k = len(items)
            # Deterministic offsets/steps derived from label
            if gpus_per_node > 1:
                start = (lab * 1315423911) % gpus_per_node
                step = 1 + ((lab ^ (lab >> 16)) % (gpus_per_node - 1))
            else:
                start, step = 0, 0
            for t, item in enumerate(items):
                gpu = (start + t * step) % gpus_per_node
                gpu_lists[gpu].append(item)

        # Capacity fixing: move heaviest from overfull to most underfull
        def gpu_load(p):
            if not gpu_lists[p]:
                return 0.0
            idx_tensor = torch.tensor(gpu_lists[p], dtype=torch.int64, device=device)
            return float(row_w[idx_tensor].sum().item())

        # Precompute loads
        loads = [gpu_load(p) for p in range(gpus_per_node)]

        def recompute_load(p):
            idx_tensor = torch.tensor(gpu_lists[p], dtype=torch.int64, device=device) if gpu_lists[p] else None
            loads[p] = float(row_w[idx_tensor].sum().item()) if idx_tensor is not None else 0.0

        while True:
            over = [(p, len(gpu_lists[p]) - cap_per_gpu) for p in range(gpus_per_node) if len(gpu_lists[p]) > cap_per_gpu]
            under = [(p, cap_per_gpu - len(gpu_lists[p])) for p in range(gpus_per_node) if len(gpu_lists[p]) < cap_per_gpu]
            if not over or not under:
                break
            # pick most overfull and most underfull
            o = max(over, key=lambda x: x[1])[0]
            u = max(under, key=lambda x: x[1])[0]
            m = min(len(gpu_lists[o]) - cap_per_gpu, cap_per_gpu - len(gpu_lists[u]))
            if m <= 0:
                break
            # pick m heaviest from o
            if gpu_lists[o]:
                o_idx = gpu_lists[o]
                o_w = row_w[torch.tensor(o_idx, dtype=torch.int64, device=device)]
                order = torch.argsort(o_w, descending=True).tolist()
                to_move = [o_idx[j] for j in order[:m]]
            else:
                to_move = []
            # move
            if to_move:
                # remove from o
                set_move = set(to_move)
                gpu_lists[o] = [x for x in gpu_lists[o] if x not in set_move]
                gpu_lists[u].extend(to_move)
                recompute_load(o)
                recompute_load(u)
            else:
                break

        # Optional single 1x1 swap between heaviest and lightest GPUs
        h = max(range(gpus_per_node), key=lambda p: loads[p])
        l = min(range(gpus_per_node), key=lambda p: loads[p])
        cur_max = max(loads)
        if gpu_lists[h] and gpu_lists[l] and cur_max > min(loads):
            # candidates: top-2 in heavy, bottom-2 in light
            h_idx_tensor = torch.tensor(gpu_lists[h], dtype=torch.int64, device=device)
            l_idx_tensor = torch.tensor(gpu_lists[l], dtype=torch.int64, device=device)
            h_w = row_w[h_idx_tensor]
            l_w = row_w[l_idx_tensor]
            kh = min(2, h_w.numel())
            kl = min(2, l_w.numel())
            if kh > 0 and kl > 0:
                h_top = torch.topk(h_w, kh).indices.tolist()
                l_bot = torch.topk(l_w, kl, largest=False).indices.tolist()
                other_max = max([loads[p] for p in range(gpus_per_node) if p != h and p != l], default=float("-inf"))
                applied = False
                best = None  # (new_peak, d_ai, r_bi, a_item, b_item, wa, wb)
                for ai in h_top:
                    a_item = int(h_idx_tensor[ai].item())
                    wa = float(h_w[ai].item())
                    for bi in l_bot:
                        b_item = int(l_idx_tensor[bi].item())
                        wb = float(l_w[bi].item())
                        new_h = loads[h] - wa + wb
                        new_l = loads[l] - wb + wa
                        new_peak = max(new_h, new_l, other_max)
                        if best is None or new_peak + 1e-9 < best[0]:
                            best = (new_peak, ai, bi, a_item, b_item, wa, wb)
                if best is not None and best[0] + 1e-9 < cur_max:
                    _, ai, bi, a_item, b_item, wa, wb = best
                    # apply swap
                    gpu_lists[h][ai] = b_item
                    gpu_lists[l][bi] = a_item
                    loads[h] = loads[h] - wa + wb
                    loads[l] = loads[l] - wb + wa
                    applied = True
                _ = applied  # keep for readability

        # Emit pack_index and ranks
        for p in range(gpus_per_node):
            for rnk, idx in enumerate(gpu_lists[p]):
                pack_index[r, idx] = p
                rank_in_pack[r, idx] = rnk

    return pack_index, rank_in_pack


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical policy with new internals:
      1) Groups -> nodes by balanced round-robin (BRR) with strict capacity.
      2) Within each (layer, node): replicas via hybrid divisor apportionment.
      3) Latin dispersion of replicas to GPUs with tiny capacity-fixing + 1x1 swap.

    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: total number of GPUs, multiple of num_nodes

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: BRR packing of groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, G]
    group_pack_index, group_rank_in_pack = _balanced_groups_round_robin(tokens_per_group, num_nodes)

    # Build mapping log <-> meta-logical within nodes
    log2mlog = (((group_pack_index * (groups_per_node) + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = _inverse(log2mlog)

    # Step 2: Replica allocation within nodes via hybrid apportionment
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy_per_node = num_physical_experts // num_nodes
    phy2mlog, phyrank, mlogcnt = _replicate_experts_apportion(tokens_per_mlog, phy_per_node)

    # Step 3: Latin dispersion to GPUs within nodes
    # Compute per-replica loads
    per_replica_load = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)  # [L*num_nodes, phy_per_node]
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = _latin_spread_pack_to_gpus(
        per_replica_load, phy2mlog, gpus_per_node, phy_experts_per_gpu
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)

    # Convert back to global logical indices
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [L*num_nodes, phy_per_node]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer with apportionment + Latin spread.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Fallback: treat as single node (preserve interface)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    # Build logical->physical map with padded 3rd dim
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # scatter physical indices into logical->physical map
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_eplb(weight: torch.Tensor, num_replicas: int, num_groups: int,
             num_nodes: int, num_gpus: int):
    """Run the expert parallelism load balancer"""
    phy2log, log2phy, logcnt = rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts", "run_eplb"]