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
import itertools
import math

def solve_partition_exact(pool_weights: list[float], K: int, target_avg: float) -> tuple[float, list[int]]:
    """
    Exact solver for partitioning pool_weights (size 2K) into two sets of size K
    to minimize max(sum(A), sum(B)).
    Returns (best_max_val, indices_A).
    """
    n = len(pool_weights)
    # Fix first element to A to reduce symmetry
    # We want to choose K indices. We pick 0, then choose K-1 from remaining n-1.
    
    best_max = float('inf')
    best_A = []
    
    total_sum = sum(pool_weights)
    w0 = pool_weights[0]
    indices_rest = list(range(1, n))
    
    for comb in itertools.combinations(indices_rest, K - 1):
        # comb is tuple of K-1 indices
        sum_A = w0
        for idx in comb:
            sum_A += pool_weights[idx]
        
        sum_B = total_sum - sum_A
        current_max = max(sum_A, sum_B)
        
        if current_max < best_max:
            best_max = current_max
            best_A = [0] + list(comb)
            if abs(current_max - target_avg) < 1e-6:
                return best_max, best_A
                
    return best_max, best_A

def solve_partition_heuristic(pool_weights: list[float], K: int, trials: int) -> tuple[float, list[int]]:
    """
    Heuristic solver for partitioning pool_weights into two sets of size K.
    """
    n = len(pool_weights)
    indices = list(range(n))
    best_max = float('inf')
    best_A = list(range(K))
    
    # 1. Randomized Greedy + Swap
    for _ in range(trials):
        # Randomize order
        random.shuffle(indices)
        
        # Initial split
        idx_A = indices[:K]
        idx_B = indices[K:]
        
        sum_A = sum(pool_weights[i] for i in idx_A)
        sum_B = sum(pool_weights[i] for i in idx_B)
        
        # Local Swap Optimization (Hill Climbing)
        improved = True
        while improved:
            improved = False
            curr_max = max(sum_A, sum_B)
            
            # Simple O(K^2) search for best swap
            best_swap = None
            best_swap_gain = 0.0
            
            # Optimization: Only look for swaps that reduce the max
            for i in range(K):
                u = idx_A[i]
                w_u = pool_weights[u]
                for j in range(K):
                    v = idx_B[j]
                    w_v = pool_weights[v]
                    
                    delta = w_u - w_v
                    
                    # New sums
                    new_A = sum_A - delta
                    new_B = sum_B + delta
                    new_max = max(new_A, new_B)
                    
                    if new_max < curr_max - 1e-6:
                        gain = curr_max - new_max
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j, delta)
            
            if best_swap:
                i, j, delta = best_swap
                # Swap indices
                tmp = idx_A[i]
                idx_A[i] = idx_B[j]
                idx_B[j] = tmp
                sum_A -= delta
                sum_B += delta
                improved = True
        
        final_max = max(sum_A, sum_B)
        if final_max < best_max:
            best_max = final_max
            best_A = list(idx_A)
            
    return best_max, best_A

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Handle trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # CPU processing
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    # Parameters
    num_restarts = 8  # Increased restarts
    max_refine_steps = 20
    
    for i in range(num_layers):
        row_weight = weight_cpu[i]
        row_weight_list = row_weight.tolist()
        
        best_diff = float('inf')
        best_packs = None
        
        # Base LPT indices
        base_indices = torch.argsort(row_weight, descending=True).tolist()
        
        for attempt in range(num_restarts):
            # 1. Initialization
            current_packs = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs
            
            if attempt == 0:
                indices = base_indices
            else:
                # Noisy LPT
                # Noise range [0.9, 1.1]
                noise = torch.rand(num_groups, device="cpu") * 0.2 + 0.9
                indices = torch.argsort(row_weight * noise, descending=True).tolist()
            
            # Greedy Packing
            for group_idx in indices:
                w = row_weight_list[group_idx]
                # Find pack with min weight that has space
                best_p = -1
                min_val = float('inf')
                for p in range(num_packs):
                    if len(current_packs[p]) < groups_per_pack:
                        if pack_weights[p] < min_val:
                            min_val = pack_weights[p]
                            best_p = p
                current_packs[best_p].append(group_idx)
                pack_weights[best_p] += w
                
            # 2. Refinement Loop
            for _ in range(max_refine_steps):
                # Identify Max Pack
                max_p = 0
                max_val = pack_weights[0]
                min_val = pack_weights[0]
                
                for p in range(1, num_packs):
                    if pack_weights[p] > max_val:
                        max_val = pack_weights[p]
                        max_p = p
                    if pack_weights[p] < min_val:
                        min_val = pack_weights[p]
                
                if max_val - min_val < 1e-6:
                    break
                    
                best_move = None
                # We want to reduce max_val.
                # Current best improvement starts at 0 (or slightly negative tolerance)
                best_new_max = max_val
                
                # Try pairing max_p with every other pack
                # Sorting candidates by weight (lightest first) is a good heuristic
                candidates = sorted([p for p in range(num_packs) if p != max_p], 
                                    key=pack_weights.__getitem__)
                                    
                for other_p in candidates:
                    other_val = pack_weights[other_p]
                    target_avg = (max_val + other_val) / 2.0
                    
                    # If perfect balancing of this pair doesn't improve upon our best found move, skip.
                    # We want new_max < best_new_max.
                    # Best possible new_max for this pair is target_avg.
                    if target_avg >= best_new_max - 1e-6:
                        continue
                        
                    # Prepare Pool
                    pool_indices = current_packs[max_p] + current_packs[other_p]
                    pool_weights = [row_weight_list[x] for x in pool_indices]
                    pool_len = len(pool_indices)
                    K = groups_per_pack
                    
                    # Solve Partition
                    if pool_len <= 14:
                        local_max, idx_A_local = solve_partition_exact(pool_weights, K, target_avg)
                    else:
                        local_max, idx_A_local = solve_partition_heuristic(pool_weights, K, trials=20)
                    
                    # Check if this result improves the global situation (reduces max_p)
                    # Note: local_max is the max(new_p1, new_p2). 
                    # We are guaranteed local_max <= max_val because we can always keep original.
                    if local_max < best_new_max - 1e-6:
                        best_new_max = local_max
                        
                        # Reconstruct items
                        # idx_A_local are indices into pool_indices
                        items_A = [pool_indices[i] for i in idx_A_local]
                        
                        # items_B are the rest
                        set_A = set(idx_A_local)
                        items_B = [pool_indices[i] for i in range(pool_len) if i not in set_A]
                        
                        # Recalculate exact sums to avoid float drift
                        w_A = sum(row_weight_list[x] for x in items_A)
                        w_B = sum(row_weight_list[x] for x in items_B)
                        
                        best_move = (other_p, items_A, items_B, w_A, w_B)
                        
                        # If we hit the theoretical limit, stop searching
                        if abs(local_max - target_avg) < 1e-6:
                            break
                            
                if best_move:
                    other_p, items_A, items_B, w_A, w_B = best_move
                    current_packs[max_p] = items_A
                    current_packs[other_p] = items_B
                    pack_weights[max_p] = w_A
                    pack_weights[other_p] = w_B
                else:
                    # No improvement possible for max_p
                    break
            
            # Check Result
            current_max = max(pack_weights)
            current_min = min(pack_weights)
            current_diff = current_max - current_min
            
            if current_diff < best_diff:
                best_diff = current_diff
                best_packs = [list(p) for p in current_packs]
                if best_diff < 1e-6:
                    break
        
        # Fill Output
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
    
    # Track current load per replica: weight / count
    current_scores = weight.clone()

    for i in range(num_log, num_phy):
        # Greedily split the expert with the highest average load per replica
        redundant_indices = current_scores.max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        logcnt[arangen, redundant_indices] += 1
        
        # Update scores
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
    Hierarchical rebalancing strategy.
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
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)
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