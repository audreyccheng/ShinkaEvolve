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
import math

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.
    
    Algorithm: Iterative Subset Refinement Balancer
    1. Diversified Randomized Greedy Initialization (Min-Load & Target-Aware)
    2. Hybrid Refinement:
       - Exhaustive Best-Pair Swap (Max vs Others)
       - 3-Way Randomized Ruin & Recreate (LNS)

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

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    weight_cpu = weight.cpu()
    
    # Pre-allocate output tensors on CPU to allow direct assignment
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64)

    # Tuning Configuration
    NUM_INIT_RESTARTS = 12
    MAX_REFINE_STEPS = 50
    LNS_SUB_RESTARTS = 5

    for i in range(num_layers):
        w_tensor = weight_cpu[i]
        w_list = w_tensor.tolist()
        
        # ---------------------------------------------------------
        # 1. Initialization: Randomized Greedy with Target Awareness
        # ---------------------------------------------------------
        
        best_assignment = None # List[List[int]]
        best_loads = None      # List[float]
        best_max_load = float('inf')
        best_ss = float('inf') # Sum of squares
        
        # Base sort indices (LPT)
        sorted_indices_base = sorted(range(num_groups), key=lambda x: w_list[x], reverse=True)
        avg_load = sum(w_list) / num_packs
        
        for attempt in range(NUM_INIT_RESTARTS):
            # Candidate generation: Perturb weights to diversify greedy choices
            if attempt == 0:
                indices = sorted_indices_base
            else:
                # Add noise (+/- 5%) and resort
                noise_scale = 0.05
                # Create (weight, index) tuples for efficient sorting
                perturbed = []
                for idx in sorted_indices_base:
                    # random.random() is [0.0, 1.0) -> -0.5 to 0.5
                    p_w = w_list[idx] * (1.0 + (random.random() - 0.5) * noise_scale)
                    perturbed.append((p_w, idx))
                perturbed.sort(key=lambda x: x[0], reverse=True)
                indices = [x[1] for x in perturbed]

            # Greedy Strategy Selection
            # First half: Standard "Min Current Load" (Best Fit)
            # Second half: "Target Driven" (minimize deviation from avg)
            use_target = (attempt >= NUM_INIT_RESTARTS // 2)

            curr_assign = [[] for _ in range(num_packs)]
            curr_loads = [0.0] * num_packs
            curr_counts = [0] * num_packs
            
            for item_idx in indices:
                item_w = w_list[item_idx]
                
                best_p = -1
                best_score = float('inf')
                
                for p in range(num_packs):
                    if curr_counts[p] < groups_per_pack:
                        if use_target:
                            # Objective: minimize (new_load - average)^2
                            new_load = curr_loads[p] + item_w
                            score = (new_load - avg_load) ** 2
                        else:
                            # Objective: minimize current load (standard Best Fit)
                            score = curr_loads[p]
                        
                        if score < best_score:
                            best_score = score
                            best_p = p
                
                curr_assign[best_p].append(item_idx)
                curr_loads[best_p] += item_w
                curr_counts[best_p] += 1
            
            # Evaluate candidate
            c_max = max(curr_loads)
            c_ss = sum(l*l for l in curr_loads)
            
            if c_max < best_max_load - 1e-6:
                best_max_load = c_max
                best_ss = c_ss
                best_assignment = curr_assign
                best_loads = curr_loads
            elif abs(c_max - best_max_load) < 1e-6 and c_ss < best_ss - 1e-6:
                best_ss = c_ss
                best_assignment = curr_assign
                best_loads = curr_loads

        # Load best initialization
        packs = best_assignment
        pack_loads = best_loads

        # ---------------------------------------------------------
        # 2. Refinement Loop
        # ---------------------------------------------------------
        
        for step in range(MAX_REFINE_STEPS):
            # Find stats
            max_p = 0
            min_p = 0
            max_l = pack_loads[0]
            min_l = pack_loads[0]
            
            for p_idx in range(1, num_packs):
                l = pack_loads[p_idx]
                if l > max_l:
                    max_l = l
                    max_p = p_idx
                if l < min_l:
                    min_l = l
                    min_p = p_idx
            
            # Convergence check
            if max_l - min_l < 1e-6:
                break
            
            improved = False
            
            # --- Strategy A: Exhaustive Best-Pair Swap ---
            # Search for the optimal single swap between Max Pack and any other pack.
            # We explicitly check all item pairs to find the one that minimizes the resulting max load.
            
            # Prioritize looking at Min Pack, then others.
            candidates = [min_p]
            others = [p for p in range(num_packs) if p != max_p and p != min_p]
            # Shuffle others to avoid bias, check a subset if too many
            random.shuffle(others)
            candidates.extend(others[:4]) 
            
            best_swap_move = None # (p_other, idx_in_max, idx_in_other, resulting_local_max)

            # Iterate items in Max Pack
            for i_u, u in enumerate(packs[max_p]):
                w_u = w_list[u]
                
                for p_other in candidates:
                    l_other = pack_loads[p_other]
                    
                    # Optimization: We need w_u > w_v for Max to decrease.
                    # Max decreases by delta = w_u - w_v.
                    # Other increases by delta.
                    # Condition: new_other_load < current_max_load (approx)
                    # i.e., l_other + (w_u - w_v) < max_l
                    
                    for i_v, v in enumerate(packs[p_other]):
                        w_v = w_list[v]
                        delta = w_u - w_v
                        
                        # Only consider swaps that reduce max pack weight
                        if delta > 1e-6:
                            # Calculate exact new loads involved
                            new_max_val = max_l - delta
                            new_other_val = l_other + delta
                            
                            # The new local max among these two packs
                            local_max = max(new_max_val, new_other_val)
                            
                            # Valid strictly improving move?
                            # We want the resulting worst case (local_max) to be as small as possible.
                            # And it MUST be better than the current global max (max_l).
                            
                            if local_max < max_l - 1e-6:
                                if best_swap_move is None or local_max < best_swap_move[3] - 1e-6:
                                    best_swap_move = (p_other, i_u, i_v, local_max)
                                elif abs(local_max - best_swap_move[3]) < 1e-6:
                                    # Tie-breaker: choose swap with larger variance reduction (lower SS)
                                    # Compare (new_max^2 + new_other^2)
                                    current_best_ss = best_swap_move[3]**2 + (max_l + pack_loads[best_swap_move[0]] - best_swap_move[3])**2 # Approx logic
                                    # Actually just picking one is fine for speed.
                                    pass

            if best_swap_move:
                p_dest, i_u, i_v, _ = best_swap_move
                # Execute Swap
                u = packs[max_p][i_u]
                v = packs[p_dest][i_v]
                
                packs[max_p][i_u] = v
                packs[p_dest][i_v] = u
                
                delta = w_list[u] - w_list[v]
                pack_loads[max_p] -= delta
                pack_loads[p_dest] += delta
                improved = True
            
            # --- Strategy B: 3-Way Ruin & Recreate (LNS) ---
            # If pairwise swap fails to improve max load (local optimum), try 3-way reshuffle.
            # Select Max, Min, and a Random pack.
            if not improved and num_packs >= 3:
                lns_indices = {max_p, min_p}
                while len(lns_indices) < 3:
                    lns_indices.add(random.randint(0, num_packs - 1))
                lns_packs = list(lns_indices)
                
                # Gather items
                pool_items = []
                for p in lns_packs:
                    pool_items.extend(packs[p])
                
                # Stats before LNS
                cur_sub_max = max(pack_loads[p] for p in lns_packs)
                cur_sub_ss = sum(pack_loads[p]**2 for p in lns_packs)
                
                best_sub_result = None
                
                # Sort LPT
                pool_items.sort(key=lambda x: w_list[x], reverse=True)
                
                # Run randomized greedy on this subset
                for sub_att in range(LNS_SUB_RESTARTS):
                    # Slight perturbation to break ties differently
                    if sub_att > 0:
                        temp_items = sorted(pool_items, key=lambda x: w_list[x] * (0.95 + random.random()*0.1), reverse=True)
                    else:
                        temp_items = pool_items
                    
                    sub_bins = {p: [] for p in lns_packs}
                    sub_w = {p: 0.0 for p in lns_packs}
                    sub_c = {p: 0 for p in lns_packs}
                    
                    possible = True
                    for itm in temp_items:
                        w = w_list[itm]
                        # Best fit
                        b_best = -1
                        min_val = float('inf')
                        for p in lns_packs:
                            if sub_c[p] < groups_per_pack:
                                if sub_w[p] < min_val:
                                    min_val = sub_w[p]
                                    b_best = p
                        
                        if b_best == -1:
                            possible = False; break
                        
                        sub_bins[b_best].append(itm)
                        sub_w[b_best] += w
                        sub_c[b_best] += 1
                    
                    if possible:
                        new_sub_max = max(sub_w.values())
                        new_sub_ss = sum(v**2 for v in sub_w.values())
                        
                        # Accept if better max, or same max and better variance
                        if new_sub_max < cur_sub_max - 1e-6:
                            cur_sub_max = new_sub_max
                            cur_sub_ss = new_sub_ss
                            best_sub_result = (sub_bins, sub_w)
                            improved = True
                        elif abs(new_sub_max - cur_sub_max) < 1e-6 and new_sub_ss < cur_sub_ss - 1e-6:
                            cur_sub_ss = new_sub_ss
                            best_sub_result = (sub_bins, sub_w)
                            improved = True
                            
                if best_sub_result:
                    new_bins, new_ws = best_sub_result
                    for p in lns_packs:
                        packs[p] = new_bins[p]
                        pack_loads[p] = new_ws[p]

        # ---------------------------------------------------------
        # 3. Final Write
        # ---------------------------------------------------------
        for p_idx in range(num_packs):
            for r_idx, g_idx in enumerate(packs[p_idx]):
                pack_index[i, g_idx] = p_idx
                rank_in_pack[i, g_idx] = r_idx

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