# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements a fundamentally different rearrangement algorithm:

- Replica counts via Huntington–Hill apportionment (divisor method):
  allocate seats to experts by priority w_i / sqrt(c_i (c_i+1)), starting
  from c_i=1, batched to keep O(n) amortized passes.

- Deterministic power-of-two GPU packing:
  each physical expert (label, replica_rank) chooses two candidate GPUs via a
  stable 64-bit hash; assign to the lighter candidate under capacity; if both
  are full, fall back to the lightest GPU with capacity, preferring those that
  don't already host the same label.

These choices improve peak smoothing and diversity with very low CPU overhead.
"""

import torch
from collections import defaultdict


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Longest-processing-time greedy with capacity constraints
    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for g in indices[i].tolist():
            # choose pack with capacity and minimal load
            best_p = None
            best_load = None
            best_cnt = None
            for p in range(num_packs):
                if pack_items[p] >= groups_per_pack:
                    continue
                if (best_load is None or pack_weights[p] < best_load or
                    (abs(pack_weights[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and pack_items[p] < (best_cnt if best_cnt is not None else 1 << 30))):
                    best_p = p
                    best_load = pack_weights[p]
                    best_cnt = pack_items[p]
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = pack_items[best_p]
            pack_items[best_p] += 1
            pack_weights[best_p] += float(weight[i, g].item())
    return pack_index, rank_in_pack


def _hh_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Huntington–Hill apportionment for integer replica counts c_i >= 1 with sum c_i == target_total.

    Priority: p_i = w_i / sqrt(c_i * (c_i + 1))
    Start with c_i = 1 and allocate remaining replicas by repeatedly choosing
    the i with the largest priority. We batch allocate up to n seats per pass.

    Parameters:
        w: [num_log], float CPU tensor
        target_total: total replicas to allocate

    Returns:
        counts: [num_log], int64 CPU tensor
    """
    num_log = w.numel()
    assert target_total >= num_log
    if num_log == 0:
        return torch.empty(0, dtype=torch.int64, device=w.device)

    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=w.device)

    # Initialize one seat per expert
    counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
    extras = target_total - num_log

    # Fast path for all-zero weights: uniform distribution
    maxw = float(w.max().item()) if num_log > 0 else 0.0
    if maxw == 0.0:
        if extras > 0:
            base = extras // num_log
            rem = extras % num_log
            if base > 0:
                counts += base
            if rem > 0:
                counts[:rem] += 1
        return counts

    # Batch HH assignment: up to n allocations per pass
    # We recompute full priorities each batch for simplicity and determinism
    # Cost is low because extras is typically O(n).
    while extras > 0:
        k = min(extras, num_log)
        # p_i = w_i / sqrt(c_i * (c_i + 1))
        denom = (counts.to(w.dtype) * (counts.to(w.dtype) + 1.0)).sqrt_().clamp_min_(1e-12)
        prio = w / denom
        topk = torch.topk(prio, k=k, largest=True).indices
        counts[topk] += 1
        extras -= k
    return counts


def replicate_experts_hh(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replica allocation using Huntington–Hill apportionment.

    Parameters:
        weight: [X, num_log] (CPU float)
        num_phy: total number of replicas per row

    Returns:
        phy2log: [X, num_phy], logical id per physical expert (CPU int64)
        rank:    [X, num_phy], rank within its logical expert (CPU int64)
        logcnt:  [X, num_log], replica counts per logical expert (CPU int64)
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
        counts = _hh_counts_row(w, num_phy)  # int64
        logcnt_list.append(counts)

        # Make contiguous blocks per expert; GPU placement will spread with hashing
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


def _stable_hash64(x: int) -> int:
    """
    A fast, deterministic 64-bit integer hash (splitmix64-like).
    """
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF


def pack_power2_hash(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic power-of-two choices packing with exact capacity per pack.

    For each item g (sorted by descending weight):
      - compute its per-replica rank among items with the same label
      - choose two candidate packs via stable 64-bit hashing of (label, rank)
      - place into the lighter candidate under capacity; if both full, fall
        back to the lightest pack with capacity, preferring packs without the
        same label, then fewer items.

    Parameters:
        weights: [X, n], CPU float tensor (per-physical loads)
        labels:  [X, n], CPU int64 tensor (logical id per physical)
        num_packs: number of packs (e.g., GPUs within a node)

    Returns:
        pack_index: [X, n], assigned pack id per item
        rank_in_pack: [X, n], rank inside the pack
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    # Trivial fast path
    if cap == 1 or num_packs == 1:
        idx = torch.arange(n_items, dtype=torch.int64, device=weights.device)
        pack_index = (idx % num_packs).expand(num_layers, n_items).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Sort each row by descending weight
    sorted_idx_all = weights.sort(dim=-1, descending=True).indices

    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=weights.device)
    rank_in_pack = torch.full_like(pack_index, -1)

    for i in range(num_layers):
        row_w = weights[i]
        row_labels = labels[i]
        sorted_idx = sorted_idx_all[i].tolist()

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        # label occurrence per pack for tie-breaking
        label_in_pack = [defaultdict(int) for _ in range(num_packs)]
        # current replica rank per label (as we assign along sorted order)
        label_next_rank = defaultdict(int)

        for g in sorted_idx:
            lab = int(row_labels[g].item())
            wv = float(row_w[g].item())
            rep_rank = label_next_rank[lab]
            label_next_rank[lab] = rep_rank + 1

            # two deterministic candidates
            h1 = _stable_hash64(lab)
            c1 = int(h1 % num_packs)
            h2 = _stable_hash64(lab ^ 0x9e3779b97f4a7c15 ^ rep_rank)
            c2 = int(h2 % num_packs)
            if c2 == c1 and num_packs > 1:
                c2 = (c2 + 1) % num_packs  # avoid identical candidates deterministically

            # choose the better candidate under capacity
            cand = []
            if counts[c1] < cap:
                cand.append(c1)
            if counts[c2] < cap and c2 != c1:
                cand.append(c2)

            pick = None
            if cand:
                # choose by projected load, then fewer same-label, then fewer items
                best_key = None
                for p in cand:
                    key = (loads[p] + wv, label_in_pack[p].get(lab, 0), counts[p])
                    if best_key is None or key < best_key:
                        best_key = key
                        pick = p

            if pick is None:
                # Fallback to any pack with capacity:
                # choose minimal projected load; tie-break: no same label, then fewer items
                best_key = None
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    key = (loads[p] + wv, label_in_pack[p].get(lab, 0), counts[p])
                    if best_key is None or key < best_key:
                        best_key = key
                        pick = p

            # Apply assignment
            pack_index[i, g] = pick
            rank_in_pack[i, g] = counts[pick]
            counts[pick] += 1
            loads[pick] += wv
            label_in_pack[pick][lab] += 1

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy greedy replica construction kept for API compatibility if needed,
    but not used in the hierarchical pipeline. Retained to preserve exports.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


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

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes (capacity-aware LPT)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate within nodes using Huntington–Hill apportionment
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts_hh(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node with two-choice hashing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = pack_power2_hash(
        tokens_per_phy, phy2mlog, gpus_per_node
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

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
        # hierarchical policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global policy (treat as single node)
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