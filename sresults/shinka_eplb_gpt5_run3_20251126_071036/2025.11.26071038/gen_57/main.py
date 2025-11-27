# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses a fundamentally different approach:
- Cyclic group-to-node assignment after sorting groups by load.
- Replica allocation via Huntington–Hill divisor apportionment within nodes,
  with a tiny guarded donor→receiver fix-up to reduce the peak.
- Deterministic power-of-two choices (Po2C) packing with exact capacities
  and label-duplicate penalty to spread replicas across GPUs.
"""

import torch
import heapq
from collections import defaultdict


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64,
                     device=perm.device).expand(perm.shape),
    )
    return inv


def _pack_groups_cyclic(tokens_per_group: torch.Tensor,
                        num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assign groups to nodes after sorting by descending load:
    group j in sorted order -> node (j % num_nodes).
    Ensures exactly groups_per_node groups per node and good balance
    without per-step LPT comparisons.
    """
    num_layers, num_groups = tokens_per_group.shape
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    # Sort groups per row by descending total tokens
    sorted_idx = tokens_per_group.float().sort(dim=-1, descending=True).indices  # [L, G]
    pack_index = torch.full_like(tokens_per_group, -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(tokens_per_group, -1, dtype=torch.int64)

    for i in range(num_layers):
        counts = [0] * num_nodes
        for j, g in enumerate(sorted_idx[i].tolist()):
            node = j % num_nodes
            pack_index[i, g] = node
            rank_in_pack[i, g] = counts[node]
            counts[node] += 1
        # sanity: exact capacity
        assert all(c == groups_per_node for c in counts), "Cyclic packing must fill exactly."

    return pack_index, rank_in_pack


def _hh_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Huntington–Hill (equal proportions) apportionment:
    Start with c_i = 1; give each next replica to argmax w_i / sqrt(c_i (c_i + 1))
    until sum c_i = target_total. Deterministic tie-breaking by lower current avg then index.
    """
    num_log = w.numel()
    assert target_total >= num_log
    device = w.device

    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=device)
    if num_log == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    counts = torch.ones(num_log, dtype=torch.int64, device=device)
    extras = int(target_total - num_log)
    # Priority queue of (-benefit, tie1, tie2, idx)
    # benefit = w / sqrt(c(c+1)); tie1 = current avg (w/c) ascending; tie2 = idx ascending
    heap = []
    w_np = w.tolist()
    c_np = counts.tolist()
    for i in range(num_log):
        c = c_np[i]
        benefit = w_np[i] / ((c * (c + 1)) ** 0.5) if w_np[i] > 0.0 else 0.0
        avg = w_np[i] / c if c > 0 else float("inf")
        heapq.heappush(heap, (-benefit, avg, i, i))  # final i used as deterministic tie

    for _ in range(extras):
        if not heap:
            break
        neg_b, _, _, idx = heapq.heappop(heap)
        c_np[idx] += 1
        c = c_np[idx]
        counts[idx] = c
        # compute updated benefit
        benefit = w_np[idx] / ((c * (c + 1)) ** 0.5) if w_np[idx] > 0.0 else 0.0
        avg = w_np[idx] / c if c > 0 else float("inf")
        heapq.heappush(heap, (-benefit, avg, idx, idx))

    return counts


def _replicate_hh_with_fixup(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Row-wise Huntington–Hill replica allocation with a tiny guarded
    donor→receiver fix-up over a bounded 2×2 candidate set.

    Parameters:
        weight: [R, num_log], float CPU
        num_phy: total replicas per row

    Returns:
        phy2log: [R, num_phy], logical expert id for each physical expert
        rank:    [R, num_phy], replica rank per logical expert
        logcnt:  [R, num_log], replica counts per logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device

    phy2log_list = []
    rank_list = []
    logcnt_list = []
    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)

    for i in range(n):
        w = weight[i]
        counts = _hh_counts_row(w, num_phy)

        # Guarded fix-up among donors {top-2 by avg with count>1} and receivers {bottom-2 by avg}
        if num_log > 1:
            counts_safe = torch.clamp(counts, min=1)
            avg = w / counts_safe.to(w.dtype)
            cur_peak = float(avg.max().item())

            can_donate = (counts > 1)
            if bool(can_donate.any()):
                kd = int(min(2, int(can_donate.sum().item())))
                avg_mask = avg.clone()
                avg_mask[~can_donate] = float("-inf")
                donors = torch.topk(avg_mask, k=kd).indices.tolist() if kd > 0 else []

                kr = int(min(2, num_log))
                receivers = torch.topk(-avg, k=kr).indices.tolist() if kr > 0 else []

                best = None  # (new_peak, new_second, d, r)
                for d in donors:
                    for r in receivers:
                        if d == r:
                            continue
                        c_try = counts.clone()
                        c_try[d] -= 1
                        c_try[r] += 1
                        avg_try = w / c_try.to(w.dtype)
                        new_peak = float(avg_try.max().item())
                        if new_peak + 1e-12 < cur_peak:
                            # tie-break by second-highest
                            if num_log >= 2:
                                top2 = torch.topk(avg_try, k=min(2, num_log)).values
                                new_second = float(top2[-1].item()) if top2.numel() >= 2 else float("-inf")
                            else:
                                new_second = float("-inf")
                            cand = (new_peak, new_second, d, r)
                            if best is None or cand < best:
                                best = cand
                if best is not None:
                    _, _, d, r = best
                    counts[d] -= 1
                    counts[r] += 1

        logcnt_list.append(counts)

        # Build phy2log and rank by contiguous blocks per expert id
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    logcnt = torch.stack(logcnt_list, dim=0)
    return phy2log, rank, logcnt


# 64-bit deterministic hash utilities (Murmur-inspired mix)
_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    x &= _MASK64
    x ^= (x >> 33)
    x = (x * 0xff51afd7ed558ccd) & _MASK64
    x ^= (x >> 33)
    x = (x * 0xc4ceb9fe1a85ec53) & _MASK64
    x ^= (x >> 33)
    return x & _MASK64


def _hash3(a: int, b: int, c: int) -> int:
    x = (a * 0x9e3779b97f4a7c15) & _MASK64
    x ^= _mix64(b ^ 0x243f6a8885a308d3)
    x = _mix64(x + (c ^ 0x13198a2e03707344))
    return x & _MASK64


def _pack_power2_choices(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic power-of-two choices (Po2C) packing with exact capacities:
    - For each item (sorted by descending weight), choose two candidate packs
      via hashing (label, replica-rank, row-seed).
    - Place the item into the candidate with lower projected load; penalize
      duplicate labels to spread replicas. If both are full, fall back to the
      lightest non-full pack.
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    # Trivial: each pack gets exactly one
    if cap == 1 or num_packs == 1:
        idx = torch.arange(n_items, dtype=torch.int64, device=weights.device)
        pack_index = (idx % num_packs).expand(num_layers, n_items).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=weights.device)
    rank_in_pack = torch.full_like(pack_index, -1)

    # Process items in descending weight per row
    sorted_idx_all = weights.sort(dim=-1, descending=True).indices  # [L, n]

    for i in range(num_layers):
        row_w = weights[i]
        row_lab = labels[i]

        # projected loads and counts per pack
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        # label presence (penalty if duplicate)
        label_in_pack = [defaultdict(int) for _ in range(num_packs)]
        pack_groups = [[] for _ in range(num_packs)]

        # tiny penalty to discourage duplicates; scale by mean weight
        mean_w = float(row_w.mean().item() if row_w.numel() > 0 else 1.0)
        dup_pen = 1e-6 * mean_w

        for pos, g in enumerate(sorted_idx_all[i].tolist()):
            wv = float(row_w[g].item())
            lab = int(row_lab[g].item())
            # Two candidate packs from deterministic hash:
            # seed uses row index; use g as the "replica id" within row
            seed0 = _hash3(i + 0x9e37, lab, g)
            seed1 = _hash3(i + 0x85eb, lab ^ g, g + 1)
            p1 = int(seed0 % num_packs)
            p2 = int(seed1 % num_packs)
            if p2 == p1:
                p2 = (p2 + 1) % num_packs

            # Cost = projected load + small penalty if label already present
            def cost_of(p: int) -> float:
                return loads[p] + wv + (dup_pen if label_in_pack[p].get(lab, 0) > 0 else 0.0)

            # Choose best among candidates that still have capacity
            cand = []
            if counts[p1] < cap:
                cand.append((cost_of(p1), p1))
            if counts[p2] < cap:
                cand.append((cost_of(p2), p2))

            if cand:
                cand.sort(key=lambda x: (x[0], counts[x[1]]))  # tie-break by fewer items
                chosen = cand[0][1]
            else:
                # Both candidates are full; choose lightest non-full pack
                # by projected load (without penalty to ensure feasibility)
                chosen = None
                best_load = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    proj = loads[p] + wv
                    if (best_load is None or proj < best_load or
                        (abs(proj - (best_load if best_load is not None else 0.0)) <= 1e-12 and counts[p] < (best_cnt if best_cnt is not None else 1 << 30))):
                        best_load = proj
                        best_cnt = counts[p]
                        chosen = p
                if chosen is None:
                    # Should not happen if capacities are correct
                    chosen = min(range(num_packs), key=lambda p: loads[p])

            # Place item
            pack_index[i, g] = chosen
            rank_in_pack[i, g] = counts[chosen]
            counts[chosen] += 1
            loads[chosen] += wv
            pack_groups[chosen].append(g)
            label_in_pack[chosen][lab] += 1

        # Optional single-pass micro 1x1 swap: heaviest ↔ lightest
        if num_packs >= 2:
            h = max(range(num_packs), key=lambda p: loads[p])
            l = min(range(num_packs), key=lambda p: loads[p])
            if loads[h] > loads[l] and pack_groups[h] and pack_groups[l]:
                # nearest-match swap by searching top-1 in h vs bottom-1 in l
                h_idx = torch.tensor(pack_groups[h], dtype=torch.int64, device=row_w.device)
                l_idx = torch.tensor(pack_groups[l], dtype=torch.int64, device=row_w.device)
                h_w = row_w[h_idx]
                l_w = row_w[l_idx]
                ai = int(torch.topk(h_w, 1).indices.item())
                bi = int(torch.topk(l_w, 1, largest=False).indices.item())
                a_item = int(h_idx[ai].item())
                b_item = int(l_idx[bi].item())
                wa = float(h_w[ai].item())
                wb = float(l_w[bi].item())

                other_max = max([loads[p] for p in range(num_packs) if p != h and p != l], default=float("-inf"))
                other_min = min([loads[p] for p in range(num_packs) if p != h and p != l], default=float("inf"))
                new_h = loads[h] - wa + wb
                new_l = loads[l] - wb + wa
                new_peak = max(new_h, new_l, other_max)
                old_peak = max(loads)
                if new_peak + 1e-12 < old_peak:
                    # apply swap
                    loads[h] = new_h
                    loads[l] = new_l
                    pack_groups[h][ai] = b_item
                    pack_groups[l][bi] = a_item
                    pack_index[i, a_item] = l
                    pack_index[i, b_item] = h
                    for rr, gg in enumerate(pack_groups[h]):
                        rank_in_pack[i, gg] = rr
                    for rr, gg in enumerate(pack_groups[l]):
                        rank_in_pack[i, gg] = rr

    return pack_index, rank_in_pack


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

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

    # Step 1: pack groups to nodes using cyclic after sorting by load
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _pack_groups_cyclic(tokens_per_group, num_nodes)

    # meta-logical remapping
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = _inverse(log2mlog)

    # Step 2: replica allocation within nodes via Huntington–Hill
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = _replicate_hh_with_fixup(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node via Po2C packing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = _pack_power2_choices(
        tokens_per_phy, phy2mlog, gpus_per_node
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [num_layers * num_nodes, num_phy_per_node]
    # convert back to global logical indexing
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
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
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # fallback: treat all as a single node
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

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