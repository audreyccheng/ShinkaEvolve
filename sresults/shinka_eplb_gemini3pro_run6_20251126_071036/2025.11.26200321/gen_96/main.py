# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using 
Big-Rock Round-Robin (BRRR) initialization and Targeted Max-Reduction Refinement.
"""

import torch

def _refine_packing_max_reduction(weights: torch.Tensor,
                                  pack_ids: torch.Tensor,
                                  pack_loads: torch.Tensor,
                                  ranks: torch.Tensor,
                                  num_packs: int,
                                  num_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines packing by attempting pairwise swaps between the heaviest pack and 
    any other pack to strictly reduce the maximum load.
    """
    batch_size, num_items = weights.shape
    device = weights.device
    
    # Precompute pairwise weight differences: [B, N, N]
    w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)
    
    batch_range = torch.arange(batch_size, device=device)

    for _ in range(num_iters):
        # Identify Max Pack
        max_vals, max_indices = pack_loads.max(dim=1) # [B]
        
        # State to track best swap: [Gain, Flat_Idx, Target_P]
        best_gain = torch.full((batch_size,), -1.0, device=device, dtype=weights.dtype)
        best_flat_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        best_target_p = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        
        mask_max = (pack_ids == max_indices.unsqueeze(1))
        
        # We iterate packs to find the best target
        for p in range(num_packs):
            mask_target = (pack_ids == p)
            valid_batch = (max_indices != p)
            
            if not valid_batch.any():
                continue
            
            target_vals = pack_loads[:, p]
            load_diff = max_vals - target_vals
            
            # Gain = min(delta, diff - delta)
            # We want to move i (Max) -> j (Target)
            # Delta = w_i - w_j
            
            delta = w_diff
            
            valid_swap = mask_max.unsqueeze(2) & mask_target.unsqueeze(1)
            valid_swap = valid_swap & valid_batch.view(-1, 1, 1)
            
            # Compute Gain
            gap = load_diff.view(-1, 1, 1)
            gain = torch.min(delta, gap - delta)
            
            # Improvement check
            valid_swap = valid_swap & (gain > 1e-5)
            
            gain = torch.where(valid_swap, gain, torch.tensor(-1.0, device=device, dtype=weights.dtype))
            
            # Best for this pack p
            flat_gain = gain.view(batch_size, -1)
            p_max_gain, p_max_idx = flat_gain.max(dim=1)
            
            improve = p_max_gain > best_gain
            if improve.any():
                best_gain = torch.where(improve, p_max_gain, best_gain)
                best_flat_idx = torch.where(improve, p_max_idx, best_flat_idx)
                best_target_p = torch.where(improve, torch.tensor(p, device=device), best_target_p)
        
        # Apply best swaps
        active = best_gain > 1e-5
        if not active.any():
            break
            
        active_batch = batch_range[active]
        active_flat = best_flat_idx[active]
        p_target = best_target_p[active]
        p_max = max_indices[active]
        
        i_idx = active_flat // num_items
        j_idx = active_flat % num_items
        
        w_i = weights[active_batch, i_idx]
        w_j = weights[active_batch, j_idx]
        delta_val = w_i - w_j
        
        pack_loads[active_batch, p_max] -= delta_val
        pack_loads[active_batch, p_target] += delta_val
        
        pack_ids[active_batch, i_idx] = p_target
        pack_ids[active_batch, j_idx] = p_max
        
        r_i = ranks[active_batch, i_idx]
        r_j = ranks[active_batch, j_idx]
        ranks[active_batch, i_idx] = r_j
        ranks[active_batch, j_idx] = r_i
        
    return pack_ids, ranks, pack_loads

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using BRRR (Big-Rock Round-Robin) Ensemble.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs
    
    # Configuration
    num_candidates = 128
    num_refine = 8
    
    # 1. Base LPT Sort
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)
    c_lpt = lpt_idx.unsqueeze(1)
    
    # 2. Candidate Generation
    # Slot 0: Pure LPT
    # Slot 1..10: LPT with Round-Robin forced prefix (Depth 1..10*M)
    # Slot 11..20: LPT with Mirror forced prefix (Depth 1..10*M)
    # Slot 21..127: Noisy LPT
    
    # Pattern candidates reuse the exact LPT order
    c_pattern = c_lpt.expand(-1, 20, -1)
    
    # Noisy candidates
    num_noisy = 107
    noise = torch.rand(num_layers, num_noisy, num_items, device=device) * 0.4 + 0.8
    noisy_w = weight.unsqueeze(1) * noise
    _, c_noisy = noisy_w.sort(dim=-1, descending=True)
    
    all_indices = torch.cat([c_lpt, c_pattern, c_noisy], dim=1) # [L, 128, N]
    
    # Gather weights [L, 128, N]
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)
    
    # Flatten
    batch_size = num_layers * num_candidates
    flat_weights = ordered_weights.view(batch_size, num_items)
    
    # 3. Fixed Pattern Initialization
    # Initialize fixed_ids with -1 (no constraint)
    fixed_ids = torch.full((batch_size, num_items), -1, device=device, dtype=torch.long)
    
    # Base Patterns
    rr_pattern = torch.arange(num_items, device=device) % num_packs
    
    # Mirror Pattern: 0..M-1, M-1..0, ...
    pat_mirror_base = torch.cat([torch.arange(num_packs, device=device), 
                                 torch.arange(num_packs-1, -1, -1, device=device)]) # length 2M
    
    # Fill constraints for candidates 1..20
    for k in range(1, 11):
        depth = min(k * num_packs, num_items)
        
        # Round Robin (Candidates 1..10)
        cand_rr = k
        indices_rr = torch.arange(cand_rr, batch_size, num_candidates, device=device)
        fixed_ids[indices_rr, :depth] = rr_pattern[:depth]
        
        # Mirror (Candidates 11..20)
        cand_mirror = 10 + k
        indices_mirror = torch.arange(cand_mirror, batch_size, num_candidates, device=device)
        
        reps = (depth // pat_mirror_base.numel()) + 1
        full_pat = pat_mirror_base.repeat(reps)[:depth]
        fixed_ids[indices_mirror, :depth] = full_pat

    # 4. Vectorized Greedy with Fixed Constraints
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    flat_ids = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    flat_ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    
    batch_range = torch.arange(batch_size, device=device)
    inf = torch.tensor(float('inf'), device=device, dtype=weight.dtype)
    
    # Main Loop
    for i in range(num_items):
        w = flat_weights[:, i]
        
        # Check constraints
        f_id = fixed_ids[:, i]
        has_fixed = (f_id != -1)
        
        # Greedy logic
        valid_mask = pack_counts < capacity
        temp_loads = torch.where(valid_mask, pack_loads, inf)
        greedy_choice = temp_loads.argmin(dim=1)
        
        # Apply constraint
        chosen_pack = torch.where(has_fixed, f_id, greedy_choice)
        
        # Update
        flat_ids[:, i] = chosen_pack
        flat_ranks[:, i] = pack_counts[batch_range, chosen_pack]
        
        pack_loads[batch_range, chosen_pack] += w
        pack_counts[batch_range, chosen_pack] += 1
        
    # 5. Top-K Selection
    loads = pack_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values
    
    _, best_k_indices = imbalance.topk(num_refine, dim=1, largest=False)
    
    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    flat_selected = (layer_offsets + best_k_indices).flatten()
    
    ref_weights = flat_weights[flat_selected]
    ref_ids = flat_ids[flat_selected]
    ref_ranks = flat_ranks[flat_selected]
    ref_loads = pack_loads[flat_selected]
    
    # 6. Targeted Refinement
    ref_ids, ref_ranks, ref_loads = _refine_packing_max_reduction(
        ref_weights, ref_ids, ref_loads, ref_ranks, num_packs, num_iters=20
    )
    
    # 7. Final Selection (Imbalance + L2 Tie-Break)
    final_loads_view = ref_loads.view(num_layers, num_refine, num_packs)
    final_imb = final_loads_view.max(dim=-1).values - final_loads_view.min(dim=-1).values
    
    # L2 Norm: sum(load^2)
    l2_norm = (final_loads_view ** 2).sum(dim=-1)
    
    # Combined Score
    score = final_imb + 1e-6 * l2_norm
    best_in_k = score.argmin(dim=1)
    
    # 8. Scatter Back
    winner_flat_idx = (torch.arange(num_layers, device=device) * num_refine) + best_in_k
    
    final_aligned_ids = ref_ids[winner_flat_idx]
    final_aligned_ranks = ref_ranks[winner_flat_idx]
    
    # Map back to original indices
    winner_cand_idx = best_k_indices.gather(1, best_in_k.unsqueeze(1)).squeeze(1)
    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_perm = all_indices.gather(1, idx_view).squeeze(1)
    
    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    
    pack_index.scatter_(1, final_perm, final_aligned_ids)
    rank_in_pack.scatter_(1, final_perm, final_aligned_ranks)
    
    return pack_index, rank_in_pack

def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Binary Search
    low = weight.sum(dim=-1, keepdim=True) / num_phy
    high = weight.max(dim=-1, keepdim=True).values
    low = torch.max(low, torch.tensor(1e-6, device=device))

    for _ in range(15):
        mid = (low + high) * 0.5
        counts = torch.ceil(weight / mid)
        total = counts.sum(dim=-1, keepdim=True)
        mask = total <= num_phy
        high = torch.where(mask, mid, high)
        low = torch.where(mask, low, mid)

    logcnt = torch.ceil(weight / high).long().clamp(min=1)

    # Correct sum
    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    # Under-allocation
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break
            
            density = weight / logcnt.float()
            density[~active] = -1.0
            target_idx = density.argmax(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] += 1

    # Over-allocation
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break
            
            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')
            cost[~active] = float('inf')
            
            target_idx = cost.argmin(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(-1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] -= 1

    # Construct maps
    flat_log_ids = torch.arange(num_log, device=device).repeat(num_layers)
    flat_counts = logcnt.flatten()
    flat_phy2log = torch.repeat_interleave(flat_log_ids, flat_counts)
    
    target_size = num_layers * num_phy
    if flat_phy2log.numel() != target_size:
        if flat_phy2log.numel() < target_size:
            flat_phy2log = torch.cat([flat_phy2log, torch.zeros(target_size - flat_phy2log.numel(), device=device, dtype=torch.long)])
        else:
            flat_phy2log = flat_phy2log[:target_size]
            
    phy2log = flat_phy2log.view(num_layers, num_phy)
    
    offsets = torch.zeros_like(logcnt)
    offsets[:, 1:] = logcnt[:, :-1].cumsum(dim=1)
    mapped_offsets = offsets.gather(1, phy2log)
    phy_indices = torch.arange(num_phy, device=device).expand(num_layers, -1)
    rank = phy_indices - mapped_offsets

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing.
    """
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
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

    # 1. Groups -> Nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
        
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # 2. Replicate within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # 3. Physical -> GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    
    node_offsets = torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)
    
    pphy2mlog_restored = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets).flatten(-2)
    
    pphy2log = mlog2log.gather(-1, pphy2mlog_restored)
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
    Entry point.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
            
    max_replicas = int(logcnt.max().item())
    
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    
    flat_layer_idx = torch.arange(num_layers, device=logcnt.device).unsqueeze(-1).expand(-1, num_replicas).flatten()
    flat_log_idx = phy2log.flatten()
    flat_rank_idx = phyrank.flatten()
    flat_phy_ids = torch.arange(num_replicas, dtype=torch.int64, device=logcnt.device).expand(num_layers, -1).flatten()
    
    indices = (flat_layer_idx * num_logical_experts * max_replicas) + \
              (flat_log_idx * max_replicas) + \
              flat_rank_idx
              
    log2phy.view(-1).scatter_(0, indices, flat_phy_ids)
    
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