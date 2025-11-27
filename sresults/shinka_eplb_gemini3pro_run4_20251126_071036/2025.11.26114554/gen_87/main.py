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

    # Optimization: fast path for trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # CPU-side processing for complex logic to avoid GPU kernel launch overheads
    # for sequential heuristics.
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    # Heuristic parameters
    # 3 Restarts: 1 deterministic, 2 randomized
    num_restarts = 3 
    # Max refinement iterations per restart
    max_refine_steps = 20

    for i in range(num_layers):
        row_weight = weight_cpu[i]
        row_weight_list = row_weight.tolist()

        best_score = float('inf')
        best_packs = None
        
        # Pre-compute sorted indices for deterministic run
        # We process (index, weight) tuples for easier sorting
        base_indices = torch.argsort(row_weight, descending=True).tolist()

        for attempt in range(num_restarts):
            # 1. Initialization
            if attempt == 0:
                # Deterministic LPT (Longest Processing Time)
                indices = base_indices
            else:
                # Randomized LPT: perturb weights to explore different greedy packings
                # Noise range [0.9, 1.1] works well to shuffle items with similar weights
                noise = torch.rand(num_groups, device="cpu") * 0.2 + 0.9
                indices = torch.argsort(row_weight * noise, descending=True).tolist()

            current_packs = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs

            # Greedy Assignment
            for idx in indices:
                w = row_weight_list[idx]
                
                # Assign to the pack with minimum current weight that has capacity
                # Since num_packs is small, linear scan is fast enough
                best_p = -1
                min_w = float('inf')
                for p in range(num_packs):
                    if len(current_packs[p]) < groups_per_pack:
                        if pack_weights[p] < min_w:
                            min_w = pack_weights[p]
                            best_p = p
                
                current_packs[best_p].append(idx)
                pack_weights[best_p] += w

            # 2. Refinement (Local Search)
            # Try to improve balance by swapping items between packs.
            for _ in range(max_refine_steps):
                # Identify packing order
                sorted_packs = sorted(range(num_packs), key=pack_weights.__getitem__, reverse=True)
                
                swap_found = False
                
                # Attempt to move weight from heavier packs to lighter packs
                # Outer loop: Heavier packs
                for i1 in range(num_packs):
                    p1 = sorted_packs[i1]
                    # Inner loop: Lighter packs (start from lightest)
                    for i2 in range(num_packs - 1, i1, -1):
                        p2 = sorted_packs[i2]
                        
                        diff = pack_weights[p1] - pack_weights[p2]
                        if diff < 1e-6:
                            # If the difference between current heavy and light is negligible,
                            # no swap between closer packs will help much either.
                            break
                        
                        target = diff / 2.0
                        best_swap = None
                        best_gap = diff
                        
                        # Find best swap (u, v)
                        # We want (w_u - w_v) approx target
                        # Optimization: if lists are short, nested loop is fine.
                        p1_items = current_packs[p1]
                        p2_items = current_packs[p2]
                        
                        for idx_u, u in enumerate(p1_items):
                            w_u = row_weight_list[u]
                            for idx_v, v in enumerate(p2_items):
                                w_v = row_weight_list[v]
                                
                                delta = w_u - w_v
                                
                                # Valid swap must reduce the weight of p1 (delta > 0)
                                # and not overshoot so much that p2 becomes heavier than p1 was (delta < diff)
                                if 0 < delta < diff:
                                    gap = abs(delta - target)
                                    if gap < best_gap:
                                        best_gap = gap
                                        best_swap = (idx_u, idx_v, delta)
                                        if gap < 1e-5: break # Sufficiently good
                            if best_swap and best_gap < 1e-5: break
                            
                        if best_swap:
                            # Execute swap
                            idx_u, idx_v, delta = best_swap
                            u = current_packs[p1][idx_u]
                            v = current_packs[p2][idx_v]
                            
                            current_packs[p1][idx_u] = v
                            current_packs[p2][idx_v] = u
                            pack_weights[p1] -= delta
                            pack_weights[p2] += delta
                            
                            swap_found = True
                            break # Break to re-sort packs
                    
                    if swap_found: break
                
                if not swap_found:
                    break

            # 3. Evaluation
            current_spread = max(pack_weights) - min(pack_weights)
            if current_spread < best_score:
                best_score = current_spread
                best_packs = [list(p) for p in current_packs]
                if best_score < 1e-6: break # Optimal

        # Store results
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
    
    # Track current average load (weight / count)
    # Using float for division
    current_scores = weight.float()
    
    for i in range(num_log, num_phy):
        # Greedily assign next replica to the expert with highest current average load
        redundant_indices = current_scores.max(dim=-1).indices
        
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        
        logcnt[arangen, redundant_indices] += 1
        
        # Update scores only for affected experts to minimize overhead
        selected_weights = weight[arangen, redundant_indices].float()
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