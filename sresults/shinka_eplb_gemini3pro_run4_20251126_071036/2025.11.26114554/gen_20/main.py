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

    # Handle trivial case where each pack gets exactly one item
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # Optimization: Process on CPU using standard Python lists for flexibility and speed in scalar logic
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    
    # Pre-allocate output tensors
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    for i in range(num_layers):
        row_w = weight_cpu[i]
        total_weight = row_w.sum().item()
        target_avg = total_weight / num_packs
        
        # Prepare items: list of (original_index, weight)
        items = []
        for idx in range(num_groups):
            items.append((idx, row_w[idx].item()))
        
        # Sort items descending by weight (common for all heuristics)
        items_desc = sorted(items, key=lambda x: x[1], reverse=True)
        
        candidates = []

        # --- Strategy 1: ZigZag (Snake) Packing ---
        # Distribute items 0..M-1, then M-1..0, etc.
        # This naturally pairs large with small items.
        packs_zz = [[] for _ in range(num_packs)]
        loads_zz = [0.0] * num_packs
        for k, (idx, w) in enumerate(items_desc):
            row = k // num_packs
            col = k % num_packs
            # Zig-zag mapping
            bin_idx = col if (row % 2 == 0) else (num_packs - 1 - col)
            packs_zz[bin_idx].append((idx, w))
            loads_zz[bin_idx] += w
        candidates.append((packs_zz, loads_zz))

        # --- Strategy 2: Projected Best-Fit LPT ---
        # Greedy allocation that accounts for remaining empty slots.
        # We try to keep the "Projected Load" of all bins close to Target.
        packs_plpt = [[] for _ in range(num_packs)]
        loads_plpt = [0.0] * num_packs
        caps_plpt = [groups_per_pack] * num_packs
        
        current_rem_weight = total_weight
        current_rem_count = num_groups
        
        for idx, w in items_desc:
            # Update remaining stats (excluding current item)
            current_rem_weight -= w
            current_rem_count -= 1
            avg_rem = current_rem_weight / current_rem_count if current_rem_count > 0 else 0.0
            
            best_p = -1
            min_score = float('inf')
            
            for p in range(num_packs):
                if caps_plpt[p] > 0:
                    # Score is deviation of (current + new + future_fill) from Target
                    # Future fill assumption: remaining slots filled with average remaining weight
                    proj_load = loads_plpt[p] + w + (caps_plpt[p] - 1) * avg_rem
                    score = abs(proj_load - target_avg)
                    if score < min_score:
                        min_score = score
                        best_p = p
            
            packs_plpt[best_p].append((idx, w))
            loads_plpt[best_p] += w
            caps_plpt[best_p] -= 1
        
        candidates.append((packs_plpt, loads_plpt))

        # --- Evaluation and Refinement ---
        best_diff_global = float('inf')
        best_packing_global = None

        for packs_init, loads_init in candidates:
            # Create a working copy
            current_packs = [list(p) for p in packs_init]
            current_loads = list(loads_init)
            
            # Iterative Improvement
            # We specifically target the MAX load pack to reduce it.
            max_iter = 20
            for _ in range(max_iter):
                # Identify Min and Max packs
                min_p = 0
                max_p = 0
                min_v = current_loads[0]
                max_v = current_loads[0]
                
                for p in range(1, num_packs):
                    v = current_loads[p]
                    if v < min_v:
                        min_v = v
                        min_p = p
                    if v > max_v:
                        max_v = v
                        max_p = p
                
                diff = max_v - min_v
                if diff < 1e-6:
                    break
                
                # We want to swap u (from max_p) with v (from other_p)
                # Goal: reduce max_v.
                # Ideally, we swap with min_p to also raise min_v (reduce range both ends).
                
                best_swap = None
                
                # 1. Try swapping with Min Pack (Most efficient for range reduction)
                target_delta = diff / 2.0
                best_gap_sq = diff * diff
                
                p1 = max_p
                p2 = min_p
                
                # Check items to find u in max_p and v in min_p
                for i1, (u, w_u) in enumerate(current_packs[p1]):
                    for i2, (v, w_v) in enumerate(current_packs[p2]):
                        delta = w_u - w_v
                        # We need w_u > w_v (delta > 0) to reduce max
                        if 0 < delta < diff:
                            gap = abs(delta - target_delta)
                            if gap * gap < best_gap_sq:
                                best_gap_sq = gap * gap
                                best_swap = (p1, p2, i1, i2, delta, u, w_u, v, w_v)
                                if gap < 1e-5: break # Perfect swap
                    if best_swap and best_gap_sq < 1e-10: break
                
                # 2. If no good swap with Min, try Any Pack that allows reduction of Max
                # A good swap with Min is usually one that brings us close to target. 
                # If the best swap with min is still poor (e.g. barely reduces max), look elsewhere.
                if best_swap is None:
                    # We iterate all other packs. We just want ANY valid reduction of Max.
                    # Sort others by load ascending (lighter first)
                    other_packs = sorted([p for p in range(num_packs) if p != max_p], 
                                         key=lambda k: current_loads[k])
                    
                    for p_other in other_packs:
                        # Maximum allowed weight increase for p_other is strictly less than what makes it reach old max_v
                        # We want: current_loads[p_other] + delta < max_v
                        # So: delta < max_v - current_loads[p_other]
                        limit = max_v - current_loads[p_other]
                        if limit < 1e-6: continue

                        local_best = None
                        local_max_delta = -1.0 # We want max delta to reduce max_p as much as possible

                        for i1, (u, w_u) in enumerate(current_packs[max_p]):
                            for i2, (v, w_v) in enumerate(current_packs[p_other]):
                                delta = w_u - w_v
                                if 0 < delta < limit:
                                    # Valid swap. Does it reduce max_p more?
                                    if delta > local_max_delta:
                                        local_max_delta = delta
                                        local_best = (max_p, p_other, i1, i2, delta, u, w_u, v, w_v)
                        
                        if local_best:
                            best_swap = local_best
                            break # Take the first valid swap with a lighter bin (Greedy on bin order)
                
                if best_swap:
                    p_from, p_to, i_from, i_to, delta, u_idx, u_w, v_idx, v_w = best_swap
                    current_packs[p_from][i_from] = (v_idx, v_w)
                    current_packs[p_to][i_to] = (u_idx, u_w)
                    current_loads[p_from] -= delta
                    current_loads[p_to] += delta
                else:
                    break # No moves possible to alleviate Max Load
            
            # Check final score for this candidate
            final_max = max(current_loads)
            final_min = min(current_loads)
            final_diff = final_max - final_min
            
            if final_diff < best_diff_global:
                best_diff_global = final_diff
                best_packing_global = current_packs
                if best_diff_global < 1e-6:
                    break # Optimal

        # Write results
        for p in range(num_packs):
            for r, (idx, _) in enumerate(best_packing_global[p]):
                pack_index[i, idx] = p
                rank_in_pack[i, idx] = r

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
    
    # Optimization: Maintain current scores to avoid repetitive division
    # Initial score is weight / 1
    current_scores = weight.clone().float()

    for i in range(num_log, num_phy):
        # Find expert with highest load per replica
        redundant_indices = current_scores.max(dim=-1).indices
        
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        
        # Increment counts
        logcnt[arangen, redundant_indices] += 1
        
        # Update scores only for changed experts
        # score = weight / count
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