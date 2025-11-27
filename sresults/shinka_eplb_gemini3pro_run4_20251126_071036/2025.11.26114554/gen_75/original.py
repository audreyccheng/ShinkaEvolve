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
    device = weight.device

    # Handle trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Use CPU for processing to avoid kernel launch overhead on sequential logic
    weight_cpu = weight.cpu()
    
    # Pre-allocate output tensors
    pack_index = torch.empty(weight.shape, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(weight.shape, dtype=torch.int64, device=device)

    # Pre-allocate helper tensors for scatter
    flat_packs_base = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)
    flat_ranks_base = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(-1)

    for i in range(num_layers):
        layer_weights_t = weight_cpu[i]
        layer_weights_list = layer_weights_t.tolist()

        # --- 1. Initialization (Randomized Greedy) ---
        # Generate candidates: 1 Deterministic + 4 Randomized
        candidates = []
        
        # Deterministic LPT
        candidates.append(sorted(range(num_groups), key=lambda x: layer_weights_list[x], reverse=True))

        # Randomized LPT
        for _ in range(4):
            # Add noise to vary the sort order
            noise = torch.rand(num_groups) * 0.1 + 0.95
            indices_rand = (layer_weights_t * noise).sort(descending=True).indices.tolist()
            candidates.append(indices_rand)

        best_assignment_list = None
        best_pack_weights_list = None
        min_max_load = float('inf')
        min_ss = float('inf')

        # Evaluate candidates with fast Python Greedy
        for indices in candidates:
            current_pack_contents = [[] for _ in range(num_packs)]
            current_pack_weights = [0.0] * num_packs
            current_pack_cnt = [0] * num_packs

            for idx in indices:
                w = layer_weights_list[idx]
                
                # Find best valid pack (min load)
                best_p = -1
                min_w = float('inf')
                
                # Small loop, fast in Python for typical num_packs
                for p in range(num_packs):
                    if current_pack_cnt[p] < groups_per_pack:
                        if current_pack_weights[p] < min_w:
                            min_w = current_pack_weights[p]
                            best_p = p
                
                current_pack_contents[best_p].append(idx)
                current_pack_weights[best_p] += w
                current_pack_cnt[best_p] += 1

            max_load = max(current_pack_weights)
            ss = sum(x*x for x in current_pack_weights)

            if max_load < min_max_load - 1e-6:
                min_max_load = max_load
                min_ss = ss
                best_assignment_list = current_pack_contents
                best_pack_weights_list = current_pack_weights
            elif abs(max_load - min_max_load) < 1e-6:
                if ss < min_ss - 1e-6:
                    min_ss = ss
                    best_assignment_list = current_pack_contents
                    best_pack_weights_list = current_pack_weights

        # Convert best assignment to tensor for vectorized operations
        # [num_packs, groups_per_pack]
        pack_assignment = torch.tensor(best_assignment_list, dtype=torch.int64)
        pack_weights = torch.tensor(best_pack_weights_list, dtype=torch.float32)

        # --- 2. Refinement Loop ---
        # Combine Vectorized Local Search (efficient) and LNS (robust)
        
        NUM_GLOBAL_ITERS = 5
        
        for _ in range(NUM_GLOBAL_ITERS):
            improved_any = False
            
            # Phase A: Vectorized Swaps
            # Check swaps between Max Pack and all other packs efficiently
            for _ in range(20):
                max_pack = torch.argmax(pack_weights).item()
                max_w = pack_weights[max_pack].item()

                # Weights of items in max pack: [1, G, 1]
                u_indices = pack_assignment[max_pack]
                w_u = layer_weights_t[u_indices].view(1, groups_per_pack, 1)

                # Weights of items in all packs: [M, 1, G]
                w_v = layer_weights_t[pack_assignment].view(num_packs, 1, groups_per_pack)

                # Diffs: max_w - pack_weights [M, 1, 1]
                diffs = (max_w - pack_weights).view(num_packs, 1, 1)
                
                # Deltas: w_u - w_v [M, G, G]
                deltas = w_u - w_v

                # Valid swaps: 0 < delta < diff
                # This ensures both packs decrease in weight relative to original max_w
                mask = (deltas > 1e-6) & (deltas < diffs)
                
                if not mask.any():
                    break

                # Heuristic: maximize delta * (diff - delta)
                # This prioritizes swaps that balance the two packs (delta ~ diff/2)
                gains = deltas * (diffs - deltas)
                gains = torch.where(mask, gains, -1.0)
                
                # Find best swap
                best_flat_idx = torch.argmax(gains).item()
                max_gain = gains.view(-1)[best_flat_idx].item()
                
                if max_gain < 0:
                    break
                    
                # Decode indices
                # Flattened dim is M * G * G
                p_target = best_flat_idx // (groups_per_pack * groups_per_pack)
                rem = best_flat_idx % (groups_per_pack * groups_per_pack)
                u_idx = rem // groups_per_pack
                v_idx = rem % groups_per_pack
                
                # Execute swap
                idx_u = pack_assignment[max_pack, u_idx].item()
                idx_v = pack_assignment[p_target, v_idx].item()
                
                pack_assignment[max_pack, u_idx] = idx_v
                pack_assignment[p_target, v_idx] = idx_u
                
                delta_val = deltas.view(-1)[best_flat_idx].item()
                pack_weights[max_pack] -= delta_val
                pack_weights[p_target] += delta_val
                
                improved_any = True

            # Phase B: Large Neighborhood Search (LNS)
            # Pick Max, Min, and Random packs, destroy and recreate
            if num_packs > 1:
                lns_steps = 3 if improved_any else 5 # More LNS if swaps didn't help
                for _ in range(lns_steps):
                    max_p = torch.argmax(pack_weights).item()
                    min_p = torch.argmin(pack_weights).item()
                    
                    if max_p == min_p:
                        break
                        
                    packs_to_repack = [max_p, min_p]
                    if num_packs > 2:
                        # Add a random pack to enable cyclic improvement
                        rand_p = random.randint(0, num_packs - 1)
                        while rand_p in packs_to_repack:
                            rand_p = random.randint(0, num_packs - 1)
                        packs_to_repack.append(rand_p)
                    
                    # Collect items
                    items = []
                    for p in packs_to_repack:
                        items.extend(pack_assignment[p].tolist())
                        
                    # Prepare sub-problem data
                    current_sub_max = max(pack_weights[p].item() for p in packs_to_repack)
                    current_sub_ss = sum(pack_weights[p].item()**2 for p in packs_to_repack)
                    
                    best_sub_res = None
                    
                    # Prepare items with weights for sorting
                    item_data = [(layer_weights_list[x], x) for x in items]
                    
                    # Sub-restarts: 1 Deterministic + 3 Randomized
                    for attempt in range(4):
                        if attempt == 0:
                            # Deterministic LPT
                            sorted_items = sorted(item_data, key=lambda x: x[0], reverse=True)
                        else:
                            # Randomized LPT
                            sorted_items = sorted(item_data, key=lambda x: x[0] * (1.0 + random.random()*0.1), reverse=True)
                        
                        temp_weights = {p: 0.0 for p in packs_to_repack}
                        temp_counts = {p: 0 for p in packs_to_repack}
                        temp_contents = {p: [] for p in packs_to_repack}
                        
                        possible = True
                        for w, idx in sorted_items:
                            # Best Fit
                            best_local_p = -1
                            min_local_w = float('inf')
                            for p in packs_to_repack:
                                if temp_counts[p] < groups_per_pack:
                                    if temp_weights[p] < min_local_w:
                                        min_local_w = temp_weights[p]
                                        best_local_p = p
                            
                            if best_local_p == -1:
                                possible = False
                                break
                            
                            temp_contents[best_local_p].append(idx)
                            temp_weights[best_local_p] += w
                            temp_counts[best_local_p] += 1
                        
                        if possible:
                            new_max = max(temp_weights.values())
                            new_ss = sum(v**2 for v in temp_weights.values())
                            
                            if new_max < current_sub_max - 1e-6:
                                current_sub_max = new_max
                                current_sub_ss = new_ss
                                best_sub_res = (temp_contents, temp_weights)
                            elif abs(new_max - current_sub_max) < 1e-6 and new_ss < current_sub_ss - 1e-6:
                                current_sub_ss = new_ss
                                best_sub_res = (temp_contents, temp_weights)

                    if best_sub_res:
                        new_contents, new_weights_map = best_sub_res
                        for p in packs_to_repack:
                            pack_assignment[p] = torch.tensor(new_contents[p], dtype=torch.int64)
                            pack_weights[p] = new_weights_map[p]
                        improved_any = True

            if not improved_any:
                break

        # Final write to output tensors
        # Use scatter for efficiency
        flat_experts = pack_assignment.view(-1).to(device)
        pack_index[i].scatter_(0, flat_experts, flat_packs_base)
        rank_in_pack[i].scatter_(0, flat_experts, flat_ranks_base)

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