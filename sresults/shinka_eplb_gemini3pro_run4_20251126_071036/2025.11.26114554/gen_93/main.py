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
import heapq

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Algorithm: Randomized Greedy Initialization + Local Beam Search Refinement

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
    device = weight.device

    # Trivial case: 1 group per pack (1-to-1 mapping)
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=device).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros((num_layers, num_groups), dtype=torch.int64, device=device)
        return pack_index, rank_in_pack

    # Work on CPU for efficiency with scalar logic
    weight_cpu = weight.cpu()
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64)

    # Helper: Beam Search Solver for a subset of items/packs
    def solve_subset_beam(items, w_map, capacity, k_packs, width=32):
        """
        Optimally pack 'items' into 'k_packs' each with 'capacity' using Beam Search.
        """
        # Sort items Descending (LPT)
        items_sorted = sorted(items, key=lambda x: w_map[x], reverse=True)
        
        # Beam State: (max_load, sum_sq, loads_tuple, counts_tuple, assignment_tuple)
        # We start with empty packs.
        start_node = (0.0, 0.0, tuple([0.0]*k_packs), tuple([0]*k_packs), ())
        beam = [start_node]
        
        for item_idx in items_sorted:
            w = w_map[item_idx]
            next_beam = []
            
            for b_max, b_ss, loads, counts, assign in beam:
                # Symmetry breaking: Bins with same (load, count) are equivalent.
                # Only try to put item in one of the equivalent bins.
                seen_signatures = set()
                
                for p in range(k_packs):
                    if counts[p] < capacity:
                        # Signature for symmetry breaking
                        sig = (round(loads[p], 5), counts[p])
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)
                        
                        # Create new state
                        nl = list(loads)
                        nl[p] += w
                        nc = list(counts)
                        nc[p] += 1
                        
                        # Update metrics
                        # Delta Sum Squares = (L+w)^2 - L^2 = 2Lw + w^2
                        delta_ss = 2 * loads[p] * w + w * w
                        new_ss = b_ss + delta_ss
                        new_max = b_max if b_max > nl[p] else nl[p]
                        
                        next_beam.append((new_max, new_ss, tuple(nl), tuple(nc), assign + (p,)))
            
            # Prune beam
            if len(next_beam) > width:
                # Primary key: Max Load, Secondary: Sum Squares
                beam = heapq.nsmallest(width, next_beam, key=lambda x: (x[0], x[1]))
            else:
                beam = next_beam
        
        # Best solution found
        best = beam[0]
        assign_indices = best[4]
        final_weights = best[2]
        
        # Reconstruct packs
        res_packs = [[] for _ in range(k_packs)]
        for i, p in enumerate(assign_indices):
            res_packs[p].append(items_sorted[i])
            
        return res_packs, list(final_weights)

    # Main Loop over Layers
    for i in range(num_layers):
        layer_w_tensor = weight_cpu[i]
        layer_w_list = layer_w_tensor.tolist()
        
        # --- 1. Initialization: Randomized Greedy ---
        best_assignment = None
        best_pack_weights = None
        best_global_max = float('inf')
        best_global_ss = float('inf')
        
        # Number of restarts
        NUM_RESTARTS = 10
        base_indices = sorted(range(num_groups), key=lambda x: layer_w_list[x], reverse=True)
        
        for attempt in range(NUM_RESTARTS):
            if attempt == 0:
                indices = base_indices
            else:
                # Add noise to weights to change sort order (Perturbed LPT)
                # Noise range [0.9, 1.1]
                noise = [random.uniform(0.9, 1.1) for _ in range(num_groups)]
                indices = sorted(range(num_groups), key=lambda x: layer_w_list[x] * noise[x], reverse=True)
            
            curr_packs = [[] for _ in range(num_packs)]
            curr_weights = [0.0] * num_packs
            curr_counts = [0] * num_packs
            
            # Greedy Fill (Best Fit)
            for idx in indices:
                w = layer_w_list[idx]
                best_p = -1
                min_load = float('inf')
                
                for p in range(num_packs):
                    if curr_counts[p] < groups_per_pack:
                        if curr_weights[p] < min_load:
                            min_load = curr_weights[p]
                            best_p = p
                
                curr_packs[best_p].append(idx)
                curr_weights[best_p] += w
                curr_counts[best_p] += 1
            
            # Check Metrics
            c_max = max(curr_weights)
            c_ss = sum(x*x for x in curr_weights)
            
            if c_max < best_global_max - 1e-6:
                best_global_max = c_max
                best_global_ss = c_ss
                best_assignment = curr_packs
                best_pack_weights = curr_weights
            elif abs(c_max - best_global_max) < 1e-6 and c_ss < best_global_ss - 1e-6:
                best_global_ss = c_ss
                best_assignment = curr_packs
                best_pack_weights = curr_weights
        
        packs = best_assignment
        pack_weights = best_pack_weights
        
        # --- 2. Refinement: LNS with Beam Search ---
        # Focus on Max and Min packs to reduce spread
        MAX_LNS_STEPS = 50
        consecutive_fail = 0
        limit_fail = 8
        
        for step in range(MAX_LNS_STEPS):
            # Identify extremes
            max_p = -1; max_val = -1.0
            min_p = -1; min_val = float('inf')
            
            for p in range(num_packs):
                val = pack_weights[p]
                if val > max_val: max_val = val; max_p = p
                if val < min_val: min_val = val; min_p = p
            
            if max_val - min_val < 1e-6:
                break
                
            # Select subset of packs to optimize
            candidates = [max_p, min_p]
            if num_packs > 2:
                # Add a random third pack to allow 3-way exchanges
                # Avoid max_p and min_p
                r = random.randint(0, num_packs - 1)
                while r == max_p or r == min_p:
                    r = random.randint(0, num_packs - 1)
                candidates.append(r)
            
            # Collect items from candidates
            sub_items = []
            for p in candidates:
                sub_items.extend(packs[p])
            
            # Solve subset optimally(ish) with Beam Search
            # Beam width 32 is sufficient for small subset (2 or 3 packs)
            new_sub_packs, new_sub_weights = solve_subset_beam(
                sub_items, layer_w_list, groups_per_pack, len(candidates), width=32
            )
            
            # Check for improvement
            old_sub_max = max(pack_weights[p] for p in candidates)
            old_sub_ss = sum(pack_weights[p]**2 for p in candidates)
            
            new_sub_max = max(new_sub_weights)
            new_sub_ss = sum(x*x for x in new_sub_weights)
            
            improved = False
            # Accept if Max load reduces OR Max load stays same but Variance (Sum Sq) reduces
            if new_sub_max < old_sub_max - 1e-6:
                improved = True
            elif abs(new_sub_max - old_sub_max) < 1e-6 and new_sub_ss < old_sub_ss - 1e-6:
                improved = True
            
            if improved:
                for local_idx, p in enumerate(candidates):
                    packs[p] = new_sub_packs[local_idx]
                    pack_weights[p] = new_sub_weights[local_idx]
                consecutive_fail = 0
            else:
                consecutive_fail += 1
                if consecutive_fail >= limit_fail:
                    break

        # --- 3. Output ---
        for p in range(num_packs):
            for r, g_idx in enumerate(packs[p]):
                pack_index[i, g_idx] = p
                rank_in_pack[i, g_idx] = r

    return pack_index.to(device), rank_in_pack.to(device)


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