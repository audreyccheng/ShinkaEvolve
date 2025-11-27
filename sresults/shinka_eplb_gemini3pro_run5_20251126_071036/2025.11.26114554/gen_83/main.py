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


def _refine_1vs1(weights: torch.Tensor,
                 pack_indices: torch.Tensor,
                 pack_weights: torch.Tensor,
                 max_iters: int = 20) -> None:
    """
    Refines packing by swapping 1 item from the heaviest pack with 1 item from any other pack.
    Updates pack_indices and pack_weights in-place.
    Vectorized over the batch dimension (num_layers).
    """
    L, M, G = pack_indices.shape
    batch_idx = torch.arange(L, device=weights.device)

    for _ in range(max_iters):
        # 1. Identify max load pack
        max_load, max_pid = torch.max(pack_weights, dim=1) # [L], [L]

        # 2. Gather weights
        # max_pid: [L] -> [L, 1, 1] -> [L, 1, G]
        max_indices_gather = max_pid.view(L, 1, 1).expand(-1, 1, G)
        items_max = pack_indices.gather(1, max_indices_gather).squeeze(1) # [L, G]
        w_max = weights.gather(1, items_max) # [L, G]

        # All items: [L, M, G]
        w_all = weights.gather(1, pack_indices.view(L, -1)).view(L, M, G)

        # 3. Calculate Deltas: w_max[i] - w_all[k, j]
        # [L, 1, 1, G] - [L, M, G, 1] -> [L, M, G, G]
        # i corresponds to last dim, j corresponds to 2nd to last dim
        deltas = w_max.unsqueeze(1).unsqueeze(2) - w_all.unsqueeze(3)

        # 4. Prospective Loads
        # New Max = Max - delta
        # New Other = Other + delta
        new_max = max_load.view(L, 1, 1, 1) - deltas
        new_other = pack_weights.view(L, M, 1, 1) + deltas

        # Objective: minimize max(new_max, new_other)
        metrics = torch.max(new_max, new_other)

        # Mask self-swaps (k == max_pid)
        mask = (torch.arange(M, device=weights.device).unsqueeze(0) == max_pid.unsqueeze(1))
        metrics.masked_fill_(mask.view(L, M, 1, 1), float('inf'))

        # 5. Best Swap
        flat_metrics = metrics.view(L, -1)
        best_val, best_idx = torch.min(flat_metrics, dim=1)

        # 6. Check improvement
        improve = best_val < (max_load - 1e-6)
        if not improve.any():
            break

        active = batch_idx[improve]
        
        # Decode indices
        # idx in 0..M*G*G-1
        idx = best_idx[improve]
        k = idx // (G * G)          # pack index
        rem = idx % (G * G)
        j = rem // G                # item index in other pack
        i = rem % G                 # item index in max pack
        
        p_max = max_pid[active]
        p_other = k
        
        # Execute Swap
        val_max = pack_indices[active, p_max, i]
        val_other = pack_indices[active, p_other, j]
        
        pack_indices[active, p_max, i] = val_other
        pack_indices[active, p_other, j] = val_max
        
        # Update weights
        d_val = deltas.view(L, -1)[active, idx]
        pack_weights[active, p_max] -= d_val
        pack_weights[active, p_other] += d_val


def _refine_2vs2(weights: torch.Tensor,
                 pack_indices: torch.Tensor,
                 pack_weights: torch.Tensor,
                 max_iters: int = 10) -> None:
    """
    Refines packing by swapping 2 items from Max Pack with 2 items from Min Pack.
    Updates pack_indices and pack_weights in-place.
    """
    L, M, G = pack_indices.shape
    if G < 2: return # Cannot swap 2 items if group size < 2
    
    batch_idx = torch.arange(L, device=weights.device)
    
    # Pre-compute valid pair mask (i != j)
    eye = torch.eye(G, device=weights.device, dtype=torch.bool)
    
    for _ in range(max_iters):
        # Identify Max and Min packs
        max_load, max_pid = torch.max(pack_weights, dim=1)
        min_load, min_pid = torch.min(pack_weights, dim=1)
        
        # If packs are balanced or same pack, stop
        diff = max_load - min_load
        if (diff < 1e-6).all():
            break
            
        # Gather items
        idx_max = pack_indices[batch_idx, max_pid] # [L, G]
        idx_min = pack_indices[batch_idx, min_pid] # [L, G]
        
        w_max = weights.gather(1, idx_max) # [L, G]
        w_min = weights.gather(1, idx_min) # [L, G]
        
        # Calculate Pair Sums: S[i, j] = w[i] + w[j]
        # [L, G, G]
        S_max = w_max.unsqueeze(2) + w_max.unsqueeze(1)
        S_min = w_min.unsqueeze(2) + w_min.unsqueeze(1)
        
        # Calculate Delta: PairMax - PairMin
        # We want to swap (i,j) from max with (k,l) from min
        # New Max = Max - (S_max - S_min)
        # Dimensions: L, i, j, k, l
        # [L, G, G, 1, 1] - [L, 1, 1, G, G]
        deltas = S_max.unsqueeze(3).unsqueeze(4) - S_min.unsqueeze(1).unsqueeze(2)
        
        new_max = max_load.view(L, 1, 1, 1, 1) - deltas
        new_min = min_load.view(L, 1, 1, 1, 1) + deltas
        
        # Optimization Goal: Reduce max load
        metrics = torch.max(new_max, new_min)
        
        # Mask diagonal (cannot pick same item twice)
        # Mask if i==j or k==l
        mask_max = eye.view(1, G, G, 1, 1)
        mask_min = eye.view(1, 1, 1, G, G)
        mask = mask_max | mask_min
        metrics.masked_fill_(mask, float('inf'))
        
        # Find Best
        flat = metrics.view(L, -1)
        best_val, best_idx = torch.min(flat, dim=1)
        
        improve = best_val < (max_load - 1e-6)
        if not improve.any():
            break
            
        active = batch_idx[improve]
        
        # Decode indices
        idx = best_idx[improve]
        G2 = G*G
        G3 = G*G*G
        
        # i, j (max); k, l (min)
        i = idx // G3
        rem = idx % G3
        j = rem // G2
        rem = rem % G2
        k = rem // G
        l = rem % G
        
        p_max = max_pid[active]
        p_min = min_pid[active]
        
        # Swap
        v_max_i = pack_indices[active, p_max, i]
        v_max_j = pack_indices[active, p_max, j]
        v_min_k = pack_indices[active, p_min, k]
        v_min_l = pack_indices[active, p_min, l]
        
        pack_indices[active, p_max, i] = v_min_k
        pack_indices[active, p_max, j] = v_min_l
        pack_indices[active, p_min, k] = v_max_i
        pack_indices[active, p_min, l] = v_max_j
        
        d_val = deltas.view(L, -1)[active, idx]
        pack_weights[active, p_max] -= d_val
        pack_weights[active, p_min] += d_val


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_attempts: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Greedy LPT + Re-weighting + 
    Multi-Stage Refinement (1-vs-1 and 2-vs-2).
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    
    # Validation
    if num_packs == num_groups:
        pack_index = torch.arange(num_groups, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    groups_per_pack = num_groups // num_packs
    
    # State tracking
    best_pack_index = torch.zeros_like(weight, dtype=torch.int64)
    best_rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
    best_max_loads = torch.full((num_layers,), float('inf'), device=device)

    # Virtual weights
    virtual_weight = weight.clone().float()

    # Pre-computed indices
    batch_idx = torch.arange(num_layers, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device)
    
    p_ids = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack)
    r_ids = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1)

    for attempt in range(num_attempts):
        # 1. Greedy Initialization (LPT)
        sorted_indices = torch.argsort(virtual_weight, dim=1, descending=True)
        sorted_real_weights = weight.gather(1, sorted_indices)
        
        pack_weights = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
        pack_counts = torch.zeros(num_layers, num_packs, device=device, dtype=torch.int64)
        pack_assignment = torch.zeros(num_layers, num_packs, groups_per_pack, dtype=torch.int64, device=device)
        
        for i in range(num_groups):
            w = sorted_real_weights[:, i]
            idx = sorted_indices[:, i]
            
            is_full = (pack_counts >= groups_per_pack)
            cand_weights = torch.where(is_full, inf_tensor, pack_weights)
            best_pack = torch.argmin(cand_weights, dim=1)
            
            slots = pack_counts[batch_idx, best_pack]
            pack_assignment[batch_idx, best_pack, slots] = idx
            pack_weights[batch_idx, best_pack] += w
            pack_counts[batch_idx, best_pack] += 1
            
        # 2. Refinement Stage 1: 1-vs-1 Swaps (Max vs All)
        _refine_1vs1(weight, pack_assignment, pack_weights, max_iters=20)
        
        # 3. Refinement Stage 2: 2-vs-2 Swaps (Max vs Min)
        # Only run if attempts > 0 or for fine-tuning
        _refine_2vs2(weight, pack_assignment, pack_weights, max_iters=10)
        
        # 4. Optional: One more pass of 1-vs-1 to clean up
        _refine_1vs1(weight, pack_assignment, pack_weights, max_iters=10)
        
        # 5. Check Improvement
        curr_max_loads, curr_max_pids = torch.max(pack_weights, dim=1)
        improved = curr_max_loads < best_max_loads
        
        if improved.any():
            flat_assignment = pack_assignment.view(num_layers, -1)
            flat_p = p_ids.reshape(num_layers, -1)
            flat_r = r_ids.reshape(num_layers, -1)
            
            curr_pidx = torch.empty_like(best_pack_index)
            curr_rank = torch.empty_like(best_rank_in_pack)
            curr_pidx.scatter_(1, flat_assignment, flat_p)
            curr_rank.scatter_(1, flat_assignment, flat_r)
            
            mask = improved.unsqueeze(1).expand_as(best_pack_index)
            best_max_loads = torch.where(improved, curr_max_loads, best_max_loads)
            best_pack_index = torch.where(mask, curr_pidx, best_pack_index)
            best_rank_in_pack = torch.where(mask, curr_rank, best_rank_in_pack)
            
        # 6. Adaptive Re-weighting
        if attempt < num_attempts - 1:
            # Boost weights of items in max pack to prioritize them next time
            items_in_max = pack_assignment[batch_idx, curr_max_pids] # [L, G]
            
            mults = torch.ones_like(virtual_weight)
            src = torch.full_like(items_in_max, 1.05, dtype=virtual_weight.dtype)
            mults.scatter_(1, items_in_max, src)
            
            virtual_weight = virtual_weight * mults

    return best_pack_index, best_rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas using vectorized greedy selection.
    """
    n, num_log = weight.shape
    device = weight.device

    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Initial mapping (first copy)
    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    
    # Vectorized assignment for remaining replicas
    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        best_expert = torch.argmax(scores, dim=1)
        logcnt[torch.arange(n, device=device), best_expert] += 1

    # Reconstruction
    for i in range(n):
        c = logcnt[i]
        phy2log[i] = torch.repeat_interleave(torch.arange(num_log, device=device), c)
        
        curr = 0
        for idx in range(num_log):
             cnt = c[idx].item()
             rank[i, curr:curr+cnt] = torch.arange(cnt, device=device)
             curr += cnt

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
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
        inv.scatter_(1, perm, torch.arange(perm.size(1), device=perm.device).expand(perm.shape))
        return inv

    # Step 1: Groups -> Nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: Replicas -> Nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: Replicas -> GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)

    # Adjust offsets
    node_offsets = torch.arange(0, num_logical_experts, num_logical_experts // num_nodes, device=weight.device)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets.view(1, -1, 1)).flatten(-2)

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
    weight = weight.float().cpu() # Ensure CPU

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1

    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64, device=logcnt.device)

    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1),
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