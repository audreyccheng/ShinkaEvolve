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


def _refine_1swap(weights: torch.Tensor,
                  pack_indices: torch.Tensor,
                  pack_weights: torch.Tensor,
                  num_packs: int,
                  groups_per_pack: int,
                  max_iters: int = 50) -> torch.Tensor:
    """
    Vectorized 1-item swap refinement.
    Swaps one item from the heaviest pack with one item from any other pack
    to minimize the maximum load.
    """
    B = weights.shape[0]
    device = weights.device
    batch_idx = torch.arange(B, device=device)

    for _ in range(max_iters):
        # 1. Identify max load pack
        max_vals, max_pids = torch.max(pack_weights, dim=1)

        # 2. Gather items
        # Items in max pack: [B, G]
        gather_indices = max_pids.view(B, 1, 1).expand(B, 1, groups_per_pack)
        items_max = torch.gather(pack_indices, 1, gather_indices).squeeze(1)
        w_max = torch.gather(weights, 1, items_max)

        # Items in all packs: [B, M, G]
        w_all = torch.gather(weights, 1, pack_indices.view(B, -1)).view(B, num_packs, groups_per_pack)

        # 3. Deltas: w_max[i] - w_all[k, j]
        # [B, G, 1] - [B, 1, M*G] -> [B, G, M*G] optimized?
        # Let's keep dimensions explicit: [B, 1, 1, G] - [B, M, G, 1] -> [B, M, G, G]
        deltas = w_max.unsqueeze(1).unsqueeze(2) - w_all.unsqueeze(3)

        # 4. Prospective loads
        new_max = max_vals.view(B, 1, 1, 1) - deltas
        new_other = pack_weights.view(B, num_packs, 1, 1) + deltas

        # Objective: max(new_max, new_other)
        obj = torch.max(new_max, new_other)

        # Mask self-swaps
        mask_k = (torch.arange(num_packs, device=device).unsqueeze(0) == max_pids.unsqueeze(1))
        mask = mask_k.view(B, num_packs, 1, 1).expand(B, num_packs, groups_per_pack, groups_per_pack)
        obj.masked_fill_(mask, float('inf'))

        # 5. Find best swap
        flat_obj = obj.view(B, -1)
        min_obj, best_idx = torch.min(flat_obj, dim=1)

        # 6. Convergence check
        improve = min_obj < (max_vals - 1e-5)
        if not improve.any():
            break

        active = batch_idx[improve]
        if len(active) == 0:
            break
        
        # 7. Execute swap
        # indices into [M, G, G]
        idx_flat = best_idx[improve]
        G = groups_per_pack
        G2 = G * G

        k_idx = idx_flat // G2
        rem = idx_flat % G2
        j_idx = rem // G
        i_idx = rem % G

        p_max = max_pids[active]
        
        # Indices in pack_indices
        # max pack: [active, p_max, i_idx]
        # other pack: [active, k_idx, j_idx]
        
        val_max = pack_indices[active, p_max, i_idx]
        val_other = pack_indices[active, k_idx, j_idx]

        pack_indices[active, p_max, i_idx] = val_other
        pack_indices[active, k_idx, j_idx] = val_max

        # Update weights
        d = deltas.view(B, -1)[active, idx_flat]
        pack_weights[active, p_max] -= d
        pack_weights[active, k_idx] += d

    return pack_indices


def _refine_2swap(weights: torch.Tensor,
                  pack_indices: torch.Tensor,
                  pack_weights: torch.Tensor,
                  num_packs: int,
                  groups_per_pack: int,
                  max_iters: int = 10) -> torch.Tensor:
    """
    Vectorized 2-item swap refinement.
    Swaps a pair of items from the heaviest pack with a pair from another pack.
    """
    L, M, G = pack_indices.shape
    device = weights.device
    
    if G < 2:
        return pack_indices

    # Pre-compute pair indices
    triu = torch.triu_indices(G, G, offset=1, device=device)
    u_idx, v_idx = triu[0], triu[1]
    num_pairs = u_idx.shape[0]
    
    batch_idx = torch.arange(L, device=device)

    for _ in range(max_iters):
        max_vals, max_pids = torch.max(pack_weights, dim=1)

        # Get item weights in max pack
        items_max = pack_indices[batch_idx, max_pids] # [L, G]
        w_max = torch.gather(weights, 1, items_max)
        
        # Pair sums in max pack: [L, P]
        w_max_pairs = w_max[:, u_idx] + w_max[:, v_idx]

        # Get item weights in all packs
        w_all = torch.gather(weights, 1, pack_indices.view(L, -1)).view(L, M, G)
        
        # Pair sums in all packs: [L, M, P]
        w_all_pairs = w_all[:, :, u_idx] + w_all[:, :, v_idx]

        # Deltas: [L, M, P_other, P_max]
        deltas = w_max_pairs.unsqueeze(1).unsqueeze(2) - w_all_pairs.unsqueeze(3)

        new_max = max_vals.view(L, 1, 1, 1) - deltas
        new_other = pack_weights.view(L, M, 1, 1) + deltas

        obj = torch.max(new_max, new_other)

        mask_k = (torch.arange(M, device=device).unsqueeze(0) == max_pids.unsqueeze(1))
        mask = mask_k.view(L, M, 1, 1).expand(L, M, num_pairs, num_pairs)
        obj.masked_fill_(mask, float('inf'))

        flat_obj = obj.view(L, -1)
        min_obj, best_idx = torch.min(flat_obj, dim=1)

        improve = min_obj < (max_vals - 1e-5)
        if not improve.any():
            break
            
        active = batch_idx[improve]
        if len(active) == 0:
            break

        idx_flat = best_idx[improve]
        
        P = num_pairs
        P2 = P * P
        
        k_idx = idx_flat // P2
        rem = idx_flat % P2
        p_other_idx = rem // P
        p_max_idx = rem % P

        p_max = max_pids[active]
        
        # Indices to swap
        u_m, v_m = u_idx[p_max_idx], v_idx[p_max_idx]
        u_o, v_o = u_idx[p_other_idx], v_idx[p_other_idx]

        # Values
        val_m_u = pack_indices[active, p_max, u_m]
        val_m_v = pack_indices[active, p_max, v_m]
        val_o_u = pack_indices[active, k_idx, u_o]
        val_o_v = pack_indices[active, k_idx, v_o]

        pack_indices[active, p_max, u_m] = val_o_u
        pack_indices[active, p_max, v_m] = val_o_v
        pack_indices[active, k_idx, u_o] = val_m_u
        pack_indices[active, k_idx, v_o] = val_m_v
        
        d = deltas.view(L, -1)[active, idx_flat]
        pack_weights[active, p_max] -= d
        pack_weights[active, k_idx] += d

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel Randomized Greedy LPT + Hybrid Refinement (1-swap & 2-swap).
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    
    if num_groups % num_packs != 0:
        # Fallback or strict requirement handled elsewhere, but assert for safety
        assert num_groups % num_packs == 0
        
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Expand candidates
    B = num_layers * num_candidates
    w_exp = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

    # Randomized LPT
    noise = torch.rand(B, num_groups, device=device) * 0.1
    # Ensure first candidate is pure LPT (noise=0) for stability
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0.0
    
    sort_keys = w_exp * (1.0 + noise)
    sorted_idx = torch.argsort(sort_keys, dim=1, descending=True)
    sorted_w = torch.gather(w_exp, 1, sorted_idx)

    # Greedy Packing
    pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.int64)
    pack_assignment = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.int64)
    
    batch_idx = torch.arange(B, device=device)
    inf_val = float('inf')

    for i in range(num_groups):
        w_item = sorted_w[:, i]
        idx_item = sorted_idx[:, i]
        
        is_full = pack_counts >= groups_per_pack
        cand = torch.where(is_full, inf_val, pack_weights)
        best_p = torch.argmin(cand, dim=1)
        
        slot = pack_counts[batch_idx, best_p]
        pack_assignment[batch_idx, best_p, slot] = idx_item
        pack_weights[batch_idx, best_p] += w_item
        pack_counts[batch_idx, best_p] += 1

    # Refine 1-swap on all candidates
    pack_assignment = _refine_1swap(w_exp, pack_assignment, pack_weights, num_packs, groups_per_pack)

    # Select Best
    max_loads = torch.max(pack_weights, dim=1).values.view(num_layers, num_candidates)
    best_cand_idx = torch.argmin(max_loads, dim=1)
    
    best_flat_idx = torch.arange(num_layers, device=device) * num_candidates + best_cand_idx
    
    final_assignment = pack_assignment[best_flat_idx] # [L, M, G]
    
    # Re-calculate weights for final refinement (floating point drift safety)
    final_w_flat = torch.gather(weight, 1, final_assignment.view(num_layers, -1)).view(num_layers, num_packs, groups_per_pack)
    final_pack_weights = final_w_flat.sum(dim=2)

    # Refine 2-swap on best candidate
    final_assignment = _refine_2swap(weight, final_assignment, final_pack_weights, num_packs, groups_per_pack)

    # Output construction
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)
    
    flat_assign = final_assignment.view(num_layers, -1)
    p_ids = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    r_ids = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index.scatter_(1, flat_assign, p_ids)
    rank_in_pack.scatter_(1, flat_assign, r_ids)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int,
        boost: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas.
    Optional boost tensor scales weights for allocation logic.
    """
    n, num_log = weight.shape
    device = weight.device

    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)
    
    # Effective weight for selection
    if boost is not None:
        eff_weight = weight * boost
    else:
        eff_weight = weight

    rows = torch.arange(n, device=device)
    for _ in range(num_log, num_phy):
        scores = eff_weight / logcnt
        best_expert = torch.argmax(scores, dim=1)
        logcnt[rows, best_expert] += 1

    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    for i in range(n):
        counts = logcnt[i]
        phy2log[i] = torch.repeat_interleave(torch.arange(num_log, device=device), counts)
        curr = 0
        for idx in range(num_log):
             cnt = counts[idx].item()
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
    """
    Hierarchical rebalancing with Feedback Loop.
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus
    num_replicas_per_node = num_physical_experts // num_nodes
    num_gpus_per_node = num_gpus // num_nodes

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

    # Step 2: Feedback Loop for Replica Allocation
    # Gather tokens per logical expert within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    
    # 2a. Initial Replica Allocation (No Boost)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_replicas_per_node)
    
    # 2b. Initial GPU Packing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus_per_node)
    
    # 2c. Feedback: Identify Bottleneck Experts
    # Calculate load per pack
    # pack_index: [L*Node, Num_Phy] -> pack id for each physical replica
    # tokens_per_phy: [L*Node, Num_Phy]
    
    # We need to map pack_index to loads. 
    # Use efficient scatter add.
    B_flat = pack_index.shape[0]
    pack_loads = torch.zeros(B_flat, num_gpus_per_node, device=weight.device, dtype=weight.dtype)
    pack_loads.scatter_add_(1, pack_index, tokens_per_phy)
    
    # Find max load and max pack ID
    max_load, max_pids = torch.max(pack_loads, dim=1) # [B_flat]
    
    # Identify items in max pack
    # item mask: pack_index == max_pids
    in_max_pack = (pack_index == max_pids.unsqueeze(1)) # [B_flat, Num_Phy]
    
    # Identify logical experts corresponding to these items
    # phy2mlog: [B_flat, Num_Phy] -> logical ID in node
    # We want to boost the logical experts that ended up in the bottleneck GPU.
    
    # Create boost map
    boost = torch.ones_like(tokens_per_mlog) # [B_flat, Num_Mlog]
    
    # Logical IDs to boost
    # We can scatter add a count, then clamp/convert to factor
    # But phy2mlog has repeats. 
    # Just need a boolean mask: if ANY replica of logical expert L is in max pack, boost L.
    
    # Helper to create mask [B_flat, Num_Mlog]
    # scatter max pack presence to logical
    # We want boost[b, l] = 1.2 if l has a replica in max pack
    
    # logical_in_max: [B_flat, Num_Mlog]
    # We can use scatter with reduce='max' (if available for float) or just careful looping/indexing
    # Since we just want to set boost, we can gather indices.
    
    rows, cols = torch.nonzero(in_max_pack, as_tuple=True)
    # rows are batch indices, cols are phy indices
    log_ids = phy2mlog[rows, cols]
    
    # Apply boost (10% increase weight for D'Hondt)
    boost[rows, log_ids] = 1.15
    
    # 2d. Second Pass Replica Allocation (With Boost)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_replicas_per_node, boost=boost)
    
    # 3. Final GPU Packing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus_per_node)

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

    # Dispatch policy
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

    # Construct reverse mapping log2phy
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Efficient scatter
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