# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

import torch
import random
import bisect
import math


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

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # CPU-based solver
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    pack_index = torch.empty(weight.shape, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty(weight.shape, dtype=torch.int64, device="cpu")

    # Use a fixed seed for deterministic behavior across runs if needed,
    # but local randomness helps exploration.
    rng = random.Random(42)

    for i in range(num_layers):
        row_weights = weight_cpu[i].tolist()

        # --- Internal Solver Logic ---
        num_items = len(row_weights)

        # 1. Initial Greedy LPT
        # Sort items by weight descending
        sorted_indices = sorted(range(num_items), key=lambda k: row_weights[k], reverse=True)

        packs = [[] for _ in range(num_packs)]
        pack_weights = [0.0] * num_packs

        for idx in sorted_indices:
            w = row_weights[idx]
            # Find pack with min weight that has space
            best_p = -1
            min_val = float('inf')
            for p in range(num_packs):
                if len(packs[p]) < groups_per_pack:
                    if pack_weights[p] < min_val:
                        min_val = pack_weights[p]
                        best_p = p
            packs[best_p].append(idx)
            pack_weights[best_p] += w

        # Helper: Optimized Pairwise Swap with 3-Way Fallback
        def refine_swaps(curr_packs, curr_weights, max_iter=20):
            for _ in range(max_iter):
                # Identify min and max packs
                min_p, max_p = 0, 0
                min_val, max_val = curr_weights[0], curr_weights[0]

                for p in range(1, num_packs):
                    w = curr_weights[p]
                    if w < min_val:
                        min_val = w
                        min_p = p
                    elif w > max_val:
                        max_val = w
                        max_p = p

                diff = max_val - min_val
                if diff < 1e-6:
                    break

                target_delta = diff / 2.0
                best_swap = None
                best_gap = diff
                found_swap = False

                # Prepare light pack items for binary search: (weight, index_in_pack)
                min_items = []
                for idx_in_pack, global_idx in enumerate(curr_packs[min_p]):
                    min_items.append((row_weights[global_idx], idx_in_pack))
                min_items.sort(key=lambda x: x[0])
                min_ws = [x[0] for x in min_items]

                # 1. Try Pairwise Swap
                for idx_u, u in enumerate(curr_packs[max_p]):
                    w_u = row_weights[u]
                    target_v = w_u - target_delta

                    pos = bisect.bisect_left(min_ws, target_v)
                    candidates = []
                    if pos < len(min_ws): candidates.append(pos)
                    if pos > 0: candidates.append(pos - 1)

                    for c_idx in candidates:
                        w_v = min_ws[c_idx]
                        delta = w_u - w_v

                        if 0 < delta < diff:
                            gap = abs(delta - target_delta)
                            if gap < best_gap:
                                best_gap = gap
                                idx_v = min_items[c_idx][1]
                                best_swap = (idx_u, idx_v, delta)
                                if gap < 1e-6:
                                    found_swap = True
                                    break
                    if found_swap: break

                if best_swap:
                    idx_u, idx_v, delta = best_swap
                    u = curr_packs[max_p][idx_u]
                    v = curr_packs[min_p][idx_v]

                    curr_packs[max_p][idx_u] = v
                    curr_packs[min_p][idx_v] = u
                    curr_weights[max_p] -= delta
                    curr_weights[min_p] += delta
                    continue

                # 2. Try 3-Way Cyclic Swap: Max -> Mid -> Min -> Max
                found_3way = False
                best_3way = None

                max_items = [(row_weights[u], idx) for idx, u in enumerate(curr_packs[max_p])]
                max_items.sort(key=lambda x: x[0], reverse=True)

                mid_candidates = [p for p in range(num_packs) if p != max_p and p != min_p]
                # Randomize to avoid stuck loops if multiple identical mid packs
                if len(mid_candidates) > 5:
                    rng.shuffle(mid_candidates)
                    mid_candidates = mid_candidates[:5]

                for mid_p in mid_candidates:
                    w_mid = curr_weights[mid_p]
                    mid_pack_items = [(row_weights[x], idx) for idx, x in enumerate(curr_packs[mid_p])]
                    mid_pack_items.sort(key=lambda x: x[0])
                    mid_ws = [x[0] for x in mid_pack_items]

                    # Maximize improvement: iterate items from max and min
                    for w_u, idx_u in max_items:
                        for w_w, idx_w in min_items:
                            if w_u <= w_w: break # Max must give more than it takes

                            # We want to maintain Mid: w_v approx w_u
                            pos = bisect.bisect_left(mid_ws, w_u)
                            cands = []
                            if pos < len(mid_ws): cands.append(pos)
                            if pos > 0: cands.append(pos - 1)

                            for c_idx in cands:
                                w_v = mid_ws[c_idx]
                                idx_v = mid_pack_items[c_idx][1]

                                new_max_w = max_val - w_u + w_w
                                new_min_w = min_val - w_w + w_v
                                new_mid_w = w_mid - w_v + w_u

                                # Check if spread reduces without creating new outliers
                                if (new_max_w < max_val - 1e-6 and
                                    new_min_w > min_val + 1e-6 and
                                    new_mid_w < max_val and
                                    new_mid_w > min_val):
                                    best_3way = (idx_u, idx_v, idx_w, mid_p)
                                    found_3way = True
                                    break
                            if found_3way: break
                        if found_3way: break
                    if found_3way: break

                if found_3way:
                    idx_u, idx_v, idx_w, mid_p = best_3way
                    u = curr_packs[max_p][idx_u]
                    v = curr_packs[mid_p][idx_v]
                    w = curr_packs[min_p][idx_w]

                    curr_packs[max_p][idx_u] = w
                    curr_packs[mid_p][idx_v] = u
                    curr_packs[min_p][idx_w] = v

                    w_u, w_v, w_w = row_weights[u], row_weights[v], row_weights[w]
                    curr_weights[max_p] += (w_w - w_u)
                    curr_weights[mid_p] += (w_u - w_v)
                    curr_weights[min_p] += (w_v - w_w)
                else:
                    break

        # Initial Refinement
        refine_swaps(packs, pack_weights, max_iter=20)

        # 2. Large Neighborhood Search (LNS) / Ruin & Recreate
        # Iteratively destroy and repair the solution to escape local optima
        best_diff = max(pack_weights) - min(pack_weights)
        best_packs = [p[:] for p in packs] # Deep copy

        lns_iters = 50

        for _ in range(lns_iters):
            if best_diff < 1e-6: break

            # Select K packs to ruin
            # Ideally: Heaviest, Lightest, and some Randoms
            # Sort packs by weight
            sorted_pack_indices = sorted(range(num_packs), key=pack_weights.__getitem__)

            candidates = set()
            candidates.add(sorted_pack_indices[0]) # Lightest
            candidates.add(sorted_pack_indices[-1]) # Heaviest

            # Add random packs until we have up to 4 (or num_packs)
            while len(candidates) < min(4, num_packs):
                candidates.add(rng.randint(0, num_packs - 1))

            candidate_list = list(candidates)

            # Remove items
            removed_items = []
            for p in candidate_list:
                removed_items.extend(packs[p])
                # Subtract weights
                pack_weights[p] -= sum(row_weights[x] for x in packs[p])
                packs[p] = []

            # Perturbed LPT Re-insertion
            # Sort items by weight with multiplicative noise to vary the greedy choice
            # Noise between 0.9 and 1.1
            removed_items.sort(key=lambda x: row_weights[x] * (0.9 + rng.random() * 0.2),
                               reverse=True)

            # Re-pack greedily into the selected candidates
            # We track local weights of candidates
            cand_weights = [pack_weights[p] for p in candidate_list]

            for item_idx in removed_items:
                w = row_weights[item_idx]

                # Find best candidate pack (min weight)
                best_c = -1
                min_w = float('inf')
                for c_i, p_idx in enumerate(candidate_list):
                    if len(packs[p_idx]) < groups_per_pack:
                        if cand_weights[c_i] < min_w:
                            min_w = cand_weights[c_i]
                            best_c = c_i

                p_real = candidate_list[best_c]
                packs[p_real].append(item_idx)
                cand_weights[best_c] += w
                pack_weights[p_real] += w

            # Polish with swaps
            refine_swaps(packs, pack_weights, max_iter=10)

            # Check acceptance
            curr_diff = max(pack_weights) - min(pack_weights)
            if curr_diff < best_diff:
                best_diff = curr_diff
                best_packs = [p[:] for p in packs]
            else:
                # Revert
                packs = [p[:] for p in best_packs]
                # Recompute weights (safer than tracking deltas)
                for p in range(num_packs):
                    pack_weights[p] = sum(row_weights[x] for x in packs[p])

        # Fill output tensors
        for p in range(num_packs):
            for r, g in enumerate(best_packs[p]):
                pack_index[i, g] = p
                rank_in_pack[i, g] = r

    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
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

    # Maintain current average load to avoid recomputing division fully
    # score = weight / count
    current_scores = weight.clone()

    for i in range(num_log, num_phy):
        # Find index with max score
        redundant_indices = current_scores.max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Increment count
        logcnt[arangen, redundant_indices] += 1

        # Update scores efficiently using gather/scatter logic (or indexing)
        # We only need to update the scores for the experts that got a replica
        # new_score = weight / new_count
        selected_weights = weight[arangen, redundant_indices]
        selected_counts = logcnt[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = selected_weights / selected_counts

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

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
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
        # use global load-balance policy
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