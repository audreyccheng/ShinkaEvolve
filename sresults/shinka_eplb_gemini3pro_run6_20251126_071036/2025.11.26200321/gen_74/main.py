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

    Uses a Hybrid Stochastic Search:
    1. Multi-Strategy Initialization: Noisy LPT, Noisy Snake (Z-Curve), and Soft-Greedy.
    2. Variance-Aware Local Search: Vectorized swaps minimizing Max Load with L2 Norm tie-breaking.

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # --- 1. Candidate Generation ---
    num_candidates = 256
    # Split candidates strategies
    # 0-95: Noisy LPT + Deterministic Greedy
    # 96-159: Noisy Snake (Z-curve) direct assignment
    # 160-255: LPT Sort + Soft Greedy (Randomness in pack selection)
    
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [L*C, N]
    num_problems = num_layers * num_candidates

    # Base Noise for sorting
    # We apply varying degrees of noise to weights to induce different sort orders
    scales = torch.linspace(0, 0.4, num_candidates, device=device).unsqueeze(1) # [C, 1]
    noise_matrix = torch.rand_like(w_expanded) * w_expanded * scales.repeat(num_layers, 1)
    
    # Ensure Candidate 0 is pure LPT (no noise)
    noise_matrix.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    
    # Sort keys
    sort_keys = w_expanded + noise_matrix
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True) # [L*C, N]
    
    # Re-gather weights in sorted order
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)
    
    # Arrays to store packing configuration
    # We will reconstruct pack_contents [L*C, M, K] from different strategies
    pack_contents = torch.zeros(num_problems, num_packs, groups_per_pack, dtype=torch.long, device=device)

    # Strategy Masking
    c_ids = torch.arange(num_candidates, device=device).repeat_interleave(num_layers)
    
    mask_greedy = (c_ids < 96) | (c_ids >= 160)
    mask_snake = (c_ids >= 96) & (c_ids < 160)
    
    # --- Strategy A & C: Greedy Approaches ---
    if mask_greedy.any():
        idx_greedy = torch.nonzero(mask_greedy).squeeze()
        w_greedy = sorted_weight[idx_greedy] # [Subset, N]
        if w_greedy.ndim == 1: w_greedy = w_greedy.unsqueeze(0)
        
        # Soft Greedy Noise
        # For candidates >= 160, we add noise to pack loads
        use_soft = (c_ids[idx_greedy] >= 160)
        
        # Initialize packs
        # Ensure correct subset size
        subset_size = len(idx_greedy) if idx_greedy.ndim > 0 else 1
        p_weights = torch.zeros(subset_size, num_packs, device=device, dtype=weight.dtype)
        p_counts = torch.zeros(subset_size, num_packs, device=device, dtype=torch.int64)
        p_assgn = torch.zeros(subset_size, num_groups, device=device, dtype=torch.int64)
        
        inf_val = torch.tensor(float('inf'), device=device)
        
        # Precompute soft noise scales
        avg_w = w_greedy.mean(dim=1, keepdim=True)
        soft_scale = torch.zeros(subset_size, 1, device=device, dtype=weight.dtype)
        soft_scale[use_soft] = avg_w[use_soft] * 1.0 # Significant noise
        
        for i in range(num_groups):
            w_item = w_greedy[:, i:i+1]
            
            # Mask full
            is_full = (p_counts >= groups_per_pack)
            costs = torch.where(is_full, inf_val, p_weights)
            
            # Add dynamic noise for Soft Greedy
            if use_soft.any():
                noise = torch.rand_like(costs) * soft_scale
                costs = costs + noise
                
            chosen = costs.argmin(dim=1, keepdim=True)
            p_assgn[:, i:i+1] = chosen
            p_weights.scatter_add_(1, chosen, w_item)
            p_counts.scatter_add_(1, chosen, torch.ones_like(chosen))
            
        # Convert assignment to pack_contents
        _, sort_p = p_assgn.sort(dim=1, stable=True)
        pack_contents[idx_greedy] = sort_p.view(-1, num_packs, groups_per_pack)

    # --- Strategy B: Snake / Z-Curve ---
    if mask_snake.any():
        idx_snake = torch.nonzero(mask_snake).squeeze()
        subset_size = len(idx_snake) if idx_snake.ndim > 0 else 1
        
        # Snake pattern generation
        range_n = torch.arange(num_groups, device=device)
        rows = range_n // num_packs
        cols = range_n % num_packs
        flip_mask = (rows % 2 == 1)
        cols[flip_mask] = (num_packs - 1) - cols[flip_mask]
        
        pack_ids_snake = cols
        
        # Group indices by pack_id
        _, perm_snake = pack_ids_snake.sort() # [N]
        
        perm_expanded = perm_snake.view(1, num_packs, groups_per_pack).expand(subset_size, -1, -1)
        pack_contents[idx_snake] = perm_expanded

    # --- 2. Refinement: Variance-Aware Swap ---
    # We iterate and swap items from Max Pack to others.
    # Metric: Lexicographical (MaxLoad, L2Norm)
    
    num_iters = 30
    for _ in range(num_iters):
        flat_c = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, groups_per_pack)
        
        pack_sums = curr_w.sum(dim=2) # [Batch, M]
        
        # Find Max Pack (with random tie-break)
        tie_break = torch.rand_like(pack_sums) * 1e-5
        _, p_max = (pack_sums + tie_break).max(dim=1)
        val_max = pack_sums.gather(1, p_max.unsqueeze(1)).squeeze(1)
        
        # Get Max Pack Items
        gather_max = p_max.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        w_max_items = torch.gather(curr_w, 1, gather_max).squeeze(1) # [Batch, K]
        
        # Compute Swap Deltas: w_max - w_other
        diffs = w_max_items.view(num_problems, 1, groups_per_pack, 1) - curr_w.view(num_problems, num_packs, 1, groups_per_pack)
        
        # Optimization: Minimize Pair Max
        val_max_exp = val_max.view(num_problems, 1, 1, 1)
        pack_sums_exp = pack_sums.view(num_problems, num_packs, 1, 1)
        
        new_pair_max = torch.max(val_max_exp - diffs, pack_sums_exp + diffs)
        imp_max = val_max_exp - new_pair_max
        
        # Variance Reduction (L2 Norm Proxy)
        # Minimize (max-diff)^2 + (other+diff)^2
        current_sq = val_max_exp.square() + pack_sums_exp.square()
        new_sq = (val_max_exp - diffs).square() + (pack_sums_exp + diffs).square()
        imp_l2 = current_sq - new_sq
        
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == p_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        # Criteria:
        # 1. imp_max > 1e-6 (Max Load Decreases)
        # 2. imp_max > -1e-6 (Max Load constant) AND imp_l2 > 1e-4 (Variance Decreases) AND diffs > 0
        valid_reduce = (imp_max > 1e-6)
        valid_variance = (imp_max > -1e-6) & (imp_l2 > 1e-4) & (diffs > 0)
        
        valid_move = (valid_reduce | valid_variance) & (~mask_self)
        
        # Score: Prioritize Max Reduction
        score = imp_max * 1000.0 + imp_l2
        score = torch.where(valid_move, score, torch.tensor(float('-inf'), device=device))
        
        flat_score = score.view(num_problems, -1)
        best_val, best_idx = flat_score.max(dim=1)
        
        if not (best_val > float('-inf')).any():
            break
            
        active = torch.nonzero(best_val > float('-inf')).squeeze(1)
        if len(active) == 0: break
        
        sel = best_idx[active]
        p_max_act = p_max[active]
        
        K = groups_per_pack
        K2 = K*K
        
        p_other = sel // K2
        rem = sel % K2
        idx_in_max = rem // K
        idx_in_other = rem % K
        
        # Swap
        val_max_ptr = pack_contents[active, p_max_act, idx_in_max].clone()
        val_oth_ptr = pack_contents[active, p_other, idx_in_other].clone()
        
        pack_contents[active, p_max_act, idx_in_max] = val_oth_ptr
        pack_contents[active, p_other, idx_in_other] = val_max_ptr

    # --- 3. Final Selection ---
    flat_c = pack_contents.view(num_problems, -1)
    final_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, groups_per_pack)
    final_max = final_w.sum(dim=2).max(dim=1).values
    
    final_max = final_max.view(num_layers, num_candidates)
    best_cand = final_max.argmin(dim=1)
    
    best_offset = torch.arange(num_layers, device=device) * num_candidates + best_cand
    
    best_pack_c = pack_contents[best_offset]
    best_sort_w_idx = sorted_indices[best_offset]
    
    flat_best_c = best_pack_c.view(num_layers, -1)
    original_idx = torch.gather(best_sort_w_idx, 1, flat_best_c)
    
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    
    p_pat = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    r_pat = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index.scatter_(1, original_idx, p_pat)
    rank_in_pack.scatter_(1, original_idx, r_pat)
    
    return pack_index, rank_in_pack


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

    # Initialize with 1 replica per expert
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Greedily add replicas to the expert with highest current load per replica
    for i in range(num_log, num_phy):
        # Metric: current load per replica = weight / count
        # Finding the expert where adding a replica reduces the max load the most
        # is equivalent to finding the expert with the current maximum load per replica.
        metrics = weight / logcnt
        redundant_indices = metrics.max(dim=-1).indices

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
    # Ensure weight is float for calculations. Keep on original device for speed.
    weight = weight.float()

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