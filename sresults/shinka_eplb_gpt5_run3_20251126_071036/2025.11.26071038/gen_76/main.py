# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses:
- Hybrid apportionment for replica allocation (D’Hondt bulk + peak-aware
  Sainte–Laguë/D’Hondt tail) to minimize the maximum per-replica load.
- Slack-first label-spread placement for GPU packing within nodes to reduce
  hotspots and align loads to per-GPU targets.
- A tiny, bounded peak-capping 1x1 swap for final smoothing.

Inputs/outputs remain identical to the original program.
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
            # choose pack with capacity and minimal load (tie: fewer items)
            best_p = None
            best_load = None
            best_cnt = None
            for p in range(num_packs):
                if pack_items[p] >= groups_per_pack:
                    continue
                if (best_load is None or pack_weights[p] < best_load or
                    (abs(pack_weights[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and
                     pack_items[p] < (best_cnt if best_cnt is not None else 1 << 30))):
                    best_p = p
                    best_load = pack_weights[p]
                    best_cnt = pack_items[p]
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = pack_items[best_p]
            pack_items[best_p] += 1
            pack_weights[best_p] += float(weight[i, g].item())
    return pack_index, rank_in_pack


def _apportion_counts_row_hybrid(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Allocate integer replica counts >= 1 via hybrid apportionment:
    - Start with one seat for each expert.
    - Bulk seats: D’Hondt priorities p = w / (c + 1).
    - Tail seats (last T): pick at each step between D’Hondt and Sainte–Laguë
      (p = w / (2c + 1)) by simulating the new peak average and committing the
      choice that yields a lower new peak (tie by lower second-highest avg).

    Parameters:
        w: [num_log], float tensor (CPU)
        target_total: total replicas to allocate

    Returns:
        counts: [num_log], int64, counts >= 1, sum == target_total.
    """
    num_log = int(w.numel())
    assert target_total >= num_log
    device = w.device

    # All-zero quick path: distribute evenly
    if num_log == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    maxw = float(w.max().item())
    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=device)
    if maxw == 0.0:
        counts = torch.ones(num_log, dtype=torch.int64, device=device)
        extras = target_total - num_log
        if extras > 0:
            base_add = extras // num_log
            rem = extras % num_log
            if base_add > 0:
                counts += base_add
            if rem > 0:
                counts[:rem] += 1
        return counts

    # Start with one per expert
    counts = torch.ones(num_log, dtype=torch.int64, device=device)
    extras = target_total - num_log
    if extras <= 0:
        return counts

    # Bulk D’Hondt distribution in chunks for speed
    # leave small adaptive tail T
    T = max(1, int(round(0.1 * extras)))  # per-row adaptive tail size
    bulk = max(0, extras - T)

    if bulk > 0:
        remaining = bulk
        while remaining > 0:
            step = min(remaining, num_log)
            # D’Hondt priority for next seat from baseline c: w / (c + 1)
            denom = (counts + 1).to(w.dtype)
            priorities = w / denom
            top_idx = torch.topk(priorities, k=step).indices
            counts[top_idx] += 1
            remaining -= step

    # Tail: adaptive picks between D’Hondt and Sainte–Laguë
    # Helper: compute peak and second-highest average for tie-breaking
    def _peak_and_second(avg: torch.Tensor) -> tuple[float, float]:
        # avg: float tensor
        if avg.numel() == 0:
            return 0.0, 0.0
        # top-2
        k = 2 if avg.numel() >= 2 else 1
        topk = torch.topk(avg, k=k).values
        peak = float(topk[0].item())
        second = float(topk[1].item()) if k == 2 else -float("inf")
        return peak, second

    for _ in range(T):
        # D’Hondt pick
        dh_prior = w / (counts + 1).to(w.dtype)
        d_idx = int(torch.argmax(dh_prior).item())

        # Sainte–Laguë pick: w / (2c + 1)
        sl_prior = w / (2 * counts + 1).to(w.dtype)
        s_idx = int(torch.argmax(sl_prior).item())

        if d_idx == s_idx:
            counts[d_idx] += 1
            continue

        # Evaluate peak if assign using D’Hondt
        c_d = counts.clone()
        c_d[d_idx] += 1
        avg_d = w / c_d.to(w.dtype)
        peak_d, second_d = _peak_and_second(avg_d)

        # Evaluate peak if assign using Sainte–Laguë
        c_s = counts.clone()
        c_s[s_idx] += 1
        avg_s = w / c_s.to(w.dtype)
        peak_s, second_s = _peak_and_second(avg_s)

        # Commit the assignment that yields lower new peak (tie by lower second highest)
        if peak_d + 1e-12 < peak_s or (abs(peak_d - peak_s) <= 1e-12 and second_d + 1e-12 < second_s):
            counts = c_d
        else:
            counts = c_s

    return counts


def replicate_experts_apportion(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid apportionment-based replication:
    - Allocate counts per logical expert via _apportion_counts_row_hybrid.
    - Build physical-to-logical and rank arrays using contiguous blocks.

    Parameters:
        weight: [X, num_log] (CPU float)
        num_phy: total number of replicas

    Returns:
        phy2log: [X, num_phy], logical expert id for each physical expert
        rank:    [X, num_phy], replica rank per logical expert
        logcnt:  [X, num_log], replica counts per logical expert
    """
    n, num_log = weight.shape
    device = weight.device
    assert num_phy >= num_log

    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)
    phy2log_list = []
    rank_list = []
    logcnt_list = []

    for i in range(n):
        w = weight[i]  # [num_log], float
        counts = _apportion_counts_row_hybrid(w, num_phy)
        logcnt_list.append(counts)

        # Construct physical mapping and ranks: contiguous by logical expert
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


def pack_slack_spread_label_rounds(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slack-first label-spread packing:
    - Compute per-GPU target = total_load / num_packs.
    - Place replicas round-by-round per label: at each mini-step choose the GPU
      with the largest slack (target - current_load) that avoids label
      duplication and respects capacity; fallback to largest slack regardless
      of duplication if necessary.
    - Apply one bounded peak-capping 1x1 swap between the heaviest and
      lightest GPUs if it strictly reduces the global peak.

    Parameters:
        weights: [X, n], float CPU (per-physical-replica loads)
        labels:  [X, n], int64 CPU (logical expert id for each physical replica)
        num_packs: number of packs (GPUs within node)

    Returns:
        pack_index: [X, n], assigned pack id per item
        rank_in_pack: [X, n], position within the pack
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    device = weights.device
    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full_like(pack_index, -1)

    if cap == 1 or num_packs == 1:
        # Simple round-robin when each pack has capacity 1
        idx = torch.arange(n_items, dtype=torch.int64, device=device)
        pack_idx_row = (idx % num_packs)
        pack_index.copy_(pack_idx_row.expand_as(pack_index))
        rank_in_pack.zero_()
        return pack_index, rank_in_pack

    for i in range(num_layers):
        row_w = weights[i]
        row_lab = labels[i]

        # Group items by label (replica sets). Use order by label total weight (equal per item for a label).
        lab2items = defaultdict(list)
        for it in range(n_items):
            lab2items[int(row_lab[it].item())].append(it)

        # Sort labels by their per-item weight descending (same across a label's items; grab first as representative)
        label_order = sorted(lab2items.keys(),
                             key=lambda L: float(row_w[lab2items[L][0]].item()),
                             reverse=True)

        total_load = float(row_w.sum().item())
        target = total_load / max(1, num_packs)

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)]
        pack_groups = [[] for _ in range(num_packs)]

        # Compute max round length across labels
        max_len = max((len(lab2items[L]) for L in label_order), default=0)

        # Place items round-by-round across labels
        for r in range(max_len):
            for lab in label_order:
                items = lab2items[lab]
                if r >= len(items):
                    continue
                g = items[r]
                wv = float(row_w[g].item())

                # candidate GPUs with capacity
                cap_candidates = [p for p in range(num_packs) if counts[p] < cap]
                if not cap_candidates:
                    continue  # should not happen

                # First, prefer GPUs without this label
                no_dup = [p for p in cap_candidates if label_counts[p].get(lab, 0) == 0]
                cand_list = no_dup if no_dup else cap_candidates

                # Choose by largest slack (target - load), tie by fewer items
                best_p = None
                best_slack = None
                best_cnt = None
                for p in cand_list:
                    slack = target - loads[p]
                    if (best_p is None or slack > best_slack + 1e-12 or
                        (abs(slack - (best_slack if best_slack is not None else 0.0)) <= 1e-12 and counts[p] < best_cnt)):
                        best_p = p
                        best_slack = slack
                        best_cnt = counts[p]

                # Assign
                p = best_p if best_p is not None else cand_list[0]
                pack_index[i, g] = p
                rank_in_pack[i, g] = counts[p]
                counts[p] += 1
                loads[p] += wv
                label_counts[p][lab] += 1
                pack_groups[p].append(g)

        # One bounded peak-capping 1x1 swap between heaviest and lightest
        if num_packs >= 2:
            h = max(range(num_packs), key=lambda k: loads[k])
            l = min(range(num_packs), key=lambda k: loads[k])
            if pack_groups[h] and pack_groups[l]:
                # Take heaviest from h, lightest from l
                h_idx_tensor = torch.tensor(pack_groups[h], dtype=torch.int64, device=device)
                l_idx_tensor = torch.tensor(pack_groups[l], dtype=torch.int64, device=device)
                h_w = row_w[h_idx_tensor]
                l_w = row_w[l_idx_tensor]
                ai = int(torch.argmax(h_w).item())
                bi = int(torch.argmin(l_w).item())
                a_item = int(h_idx_tensor[ai].item())
                b_item = int(l_idx_tensor[bi].item())
                wa = float(row_w[a_item].item())
                wb = float(row_w[b_item].item())

                # Check label duplication constraint (soft): prefer swaps that don't create new duplicates
                la = int(row_lab[a_item].item())
                lb = int(row_lab[b_item].item())
                dup_penalty = 0
                dup_penalty += 1 if label_counts[h].get(lb, 0) > 0 else 0
                dup_penalty += 1 if label_counts[l].get(la, 0) > 0 else 0

                # Evaluate peak after swap
                new_h = loads[h] - wa + wb
                new_l = loads[l] - wb + wa
                other_max = max([loads[p] for p in range(num_packs) if p != h and p != l], default=float("-inf"))
                new_peak = max(new_h, new_l, other_max)
                cur_peak = max(loads)

                # Apply only if strictly reduces peak or reduces peak with no duplicate penalty increase
                if new_peak + 1e-9 < cur_peak or (new_peak + 1e-9 < cur_peak + 1e-12 and dup_penalty == 0):
                    loads[h] = new_h
                    loads[l] = new_l
                    pack_groups[h][ai] = b_item
                    pack_groups[l][bi] = a_item
                    pack_index[i, a_item] = l
                    pack_index[i, b_item] = h
                    # Ranks only for affected packs
                    for r, g in enumerate(pack_groups[h]):
                        rank_in_pack[i, g] = r
                    for r, g in enumerate(pack_groups[l]):
                        rank_in_pack[i, g] = r

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

    # Step 2: replicate within nodes using hybrid apportionment
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts_apportion(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node via slack-first label-spread
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = pack_slack_spread_label_rounds(
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