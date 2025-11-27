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

    Uses a Massive Parallel Hybrid Initialization (Spectrum LPT, Interleaved, Random)
    followed by a Vectorized Max-Any Swap local search.

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

    # --- Massive Parallel Initialization ---
    # 256 candidates per layer to maximize probability of finding good packing
    num_candidates = 256
    num_problems = num_layers * num_candidates

    # Expand weights: [L*C, N]
    w_expanded = weight.repeat_interleave(num_candidates, dim=0)

    # Generate Candidate IDs: 0..255 for each layer repeated
    cand_ids = torch.arange(num_candidates, device=device).repeat(num_layers)

    # Prepare keys for sorting
    keys = w_expanded.clone()

    # Strategy 1: Spectrum Randomized LPT (Candidates 0-127)
    # Scale noise from 0.0 to 0.4. Candidate 0 is pure LPT (scale 0).
    mask_spectrum = (cand_ids < 128)
    if mask_spectrum.any():
        scales = torch.linspace(0, 0.4, 128, device=device)
        # Map global candidate indices to 0..127
        current_ids = cand_ids[mask_spectrum]
        scale_vals = scales[current_ids]
        
        noise = torch.rand_like(w_expanded[mask_spectrum]) * w_expanded[mask_spectrum] * scale_vals.unsqueeze(1)
        keys[mask_spectrum] += noise
        
        # Ensure Candidate 0 (and multiples if layers > 1) is exactly pure LPT
        mask_pure = (cand_ids == 0)
        keys[mask_pure] = w_expanded[mask_pure]

    # Strategy 3: Random Shuffle (Candidates 192-255)
    # Use random keys
    mask_random = (cand_ids >= 192)
    if mask_random.any():
        keys[mask_random] = torch.rand_like(w_expanded[mask_random])
    
    # Strategy 2: Interleaved (Candidates 128-191)
    # We will handle this by deriving from the sorted indices of the Pure LPT candidate later,
    # or we can just sort keys here (they are currently copies of w_expanded, i.e., Pure LPT keys)
    # and permute the result.
    # Note: mask_inter keys are currently w_expanded (Pure LPT).
    
    # Perform Sort to determine order for Spectrum and Random (and base for Interleaved)
    _, sorted_indices = keys.sort(dim=-1, descending=True)

    # Implement Strategy 2: Interleaved LPT
    mask_inter = (cand_ids >= 128) & (cand_ids < 192)
    if mask_inter.any():
        # Ideally, we want the sorted indices from Candidate 0 (Pure LPT)
        # But `sorted_indices[mask_inter]` currently holds sorted indices of Pure LPT (since keys were w_expanded)
        # So we just need to permute them.
        
        # Construct interleave map [0, N-1, 1, N-2, ...]
        imap = torch.empty(num_groups, dtype=torch.long, device=device)
        imap[0::2] = torch.arange((num_groups + 1) // 2, device=device)
        imap[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        
        # Apply map
        subset = sorted_indices[mask_inter]
        sorted_indices[mask_inter] = subset[:, imap]

    # Gather actual weights in the determined order
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- Phase 2: Vectorized Greedy Assignment ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    sorted_pack_index = torch.zeros_like(sorted_indices)

    # Infinity for masking
    inf_val = torch.tensor(float('inf'), device=device)

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]

        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        candidates = torch.where(is_full, inf_val, pack_weights)

        # Choose pack with min weight
        chosen_pack = candidates.argmin(dim=1, keepdim=True)

        sorted_pack_index[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 3: Vectorized Max-Any Swap Refinement ---
    # Prepare pack contents for fast swapping
    # We sort sorted_pack_index to group items by pack ID
    # This effectively creates the `pack_contents` structure
    _, pack_sort_idx = sorted_pack_index.sort(dim=1, stable=True)
    pack_contents = pack_sort_idx.view(num_problems, num_packs, groups_per_pack)

    K = groups_per_pack
    num_iters = 50

    for _ in range(num_iters):
        # 1. Compute current pack weights from contents
        flat_contents = pack_contents.view(num_problems, -1)
        current_items = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
        pack_sums = current_items.sum(dim=2)

        # 2. Find Max Pack (Bottleneck)
        val_max, idx_max_pack = pack_sums.max(dim=1) # [LC], [LC]

        # 3. Gather items from Max Pack
        gather_idx = idx_max_pack.view(-1, 1, 1).expand(-1, 1, K)
        w_max_items = torch.gather(current_items, 1, gather_idx).squeeze(1) # [LC, K]

        # 4. Calculate Improvement
        # Try swapping item 'u' in Max Pack with item 'v' in Any Pack 'p'
        # diff = w_u - w_v
        # New Max Load = val_max - diff (assuming val_max was the unique max or we reduce it)
        # New Other Load = pack_sums[p] + diff
        # Improvement condition: New Max < Old Max AND New Other < Old Max
        # We want to maximize the reduction of the bottleneck.
        # Reduction = val_max - max(val_max - diff, pack_sums[p] + diff)
        #           = min(diff, val_max - pack_sums[p] - diff)
        
        w_max_exp = w_max_items.view(num_problems, 1, K, 1)
        w_curr_exp = current_items.view(num_problems, num_packs, 1, K)
        
        diffs = w_max_exp - w_curr_exp # [LC, M, K, K]
        
        val_max_exp = val_max.view(num_problems, 1, 1, 1)
        pack_sums_exp = pack_sums.view(num_problems, num_packs, 1, 1)
        
        # Improvement metric
        improvement = torch.min(diffs, val_max_exp - pack_sums_exp - diffs)

        # 5. Masking
        # Mask self-swaps (Max Pack with itself)
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max_pack.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        # Valid if strictly improving (> epsilon) and diff > 0
        valid_mask = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        
        scores = torch.where(valid_mask, improvement, torch.tensor(float('-inf'), device=device))

        # 6. Select Best Swap per Problem
        flat_scores = scores.view(num_problems, -1)
        best_imp, flat_idx = flat_scores.max(dim=1)

        if not (best_imp > float('-inf')).any():
            break

        # 7. Execute Swaps
        valid_layers = torch.nonzero(best_imp > float('-inf')).squeeze(1)
        
        if len(valid_layers) > 0:
            f_idx = flat_idx[valid_layers]
            
            # Decode indices: M * K * K
            K2 = K * K
            p_other = f_idx // K2
            rem = f_idx % K2
            idx_in_max = rem // K
            idx_in_other = rem % K
            
            p_max = idx_max_pack[valid_layers]
            
            # Swap indices in pack_contents
            val_max_idx = pack_contents[valid_layers, p_max, idx_in_max]
            val_other_idx = pack_contents[valid_layers, p_other, idx_in_other]
            
            pack_contents[valid_layers, p_max, idx_in_max] = val_other_idx
            pack_contents[valid_layers, p_other, idx_in_other] = val_max_idx

    # --- Phase 4: Selection ---
    # Recompute final loads
    flat_contents = pack_contents.view(num_problems, -1)
    final_items = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
    final_max_loads = final_items.sum(dim=2).max(dim=1).values

    # Find best candidate per layer
    reshaped_loads = final_max_loads.view(num_layers, num_candidates)
    best_candidate_idx = reshaped_loads.argmin(dim=1)

    # Gather best solution
    best_indices = torch.arange(num_layers, device=device) * num_candidates + best_candidate_idx
    best_contents = pack_contents[best_indices] # [L, M, K]
    best_sorted_indices = sorted_indices[best_indices] # [L, N]

    # Map back to original indices
    flat_best_contents = best_contents.view(num_layers, -1)
    original_item_indices = torch.gather(best_sorted_indices, 1, flat_best_contents)

    # Output tensors
    pack_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    rank_ids = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index = torch.empty_like(pack_ids)
    rank_in_pack = torch.empty_like(rank_ids)

    pack_index.scatter_(1, original_item_indices, pack_ids)
    rank_in_pack.scatter_(1, original_item_indices, rank_ids)

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