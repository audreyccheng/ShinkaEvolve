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

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Operations on CPU are generally faster for this sequential/iterative logic
    weight_cpu = weight.cpu()

    pack_index = torch.empty(weight.shape, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(weight.shape, dtype=torch.int64, device=device)

    # Pre-allocate helper tensors for scatter
    flat_packs_base = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)
    flat_ranks_base = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(-1)

    for i in range(num_layers):
        layer_weights_t = weight_cpu[i]
        layer_weights = layer_weights_t.tolist()

        # --- 1. Initialization (Randomized Greedy) ---
        candidates = []
        # Deterministic LPT
        candidates.append(sorted(range(num_groups), key=lambda x: layer_weights[x], reverse=True))

        # Randomized LPT (Perturbed weights)
        # Generate multiple candidates to improve diversity
        for _ in range(4):
            noise = torch.rand(num_groups) * 0.1 + 0.95  # +/- 5% noise
            indices_rand = (layer_weights_t * noise).sort(descending=True).indices.tolist()
            candidates.append(indices_rand)

        best_assignment = None
        best_pack_weights = None
        min_max_load = float('inf')
        min_ss = float('inf')

        # Evaluate candidates with fast Python Greedy
        for indices in candidates:
            current_pack_contents = [[] for _ in range(num_packs)]
            current_pack_weights = [0.0] * num_packs
            current_pack_cnt = [0] * num_packs

            # Pack
            for idx in indices:
                w = layer_weights[idx]
                # Best fit among non-full packs
                best_p = -1
                min_w = float('inf')
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
                best_assignment = current_pack_contents
                best_pack_weights = current_pack_weights
            elif abs(max_load - min_max_load) < 1e-6:
                if ss < min_ss - 1e-6:
                    min_ss = ss
                    best_assignment = current_pack_contents
                    best_pack_weights = current_pack_weights

        # Convert best assignment to tensor for vectorized refinement
        # [num_packs, groups_per_pack]
        pack_assignment = torch.tensor(best_assignment, dtype=torch.int64)
        pack_weights = torch.tensor(best_pack_weights, dtype=torch.float32)

        # --- 2. Refinement Loop ---
        MAX_OUTER_LOOPS = 5
        
        for _ in range(MAX_OUTER_LOOPS):
            improved_any = False

            # Phase 1: Vectorized Local Search (Swap)
            # Uses broadcasting to evaluate swaps between the heaviest pack and all others simultaneously
            for _ in range(10):
                max_pack = torch.argmax(pack_weights).item()
                max_w = pack_weights[max_pack].item()

                # Items in max pack: [G]
                u_indices = pack_assignment[max_pack]
                w_u = layer_weights_t[u_indices]

                # Items in all packs: [M, G]
                w_v = layer_weights_t[pack_assignment]

                # Deltas: w_u - w_v. Shape [M, G, G] (via broadcasting)
                # Gain function aims to balance the pair (max_pack, p)
                diffs = (max_w - pack_weights).view(-1, 1, 1) # [M, 1, 1]
                deltas = w_u.view(1, groups_per_pack, 1) - w_v.view(num_packs, 1, groups_per_pack) # [M, G, G]

                # Mask: valid positive delta strictly less than diff (to avoid overshooting too much)
                # Actually, strictly less than diff ensures w_p < old_max after swap.
                mask = (deltas > 1e-6) & (deltas < diffs)

                if not mask.any():
                    break

                gains = deltas * (diffs - deltas)
                gains = torch.where(mask, gains, -1.0)

                best_val, best_flat_idx = gains.view(-1).max(0)

                if best_val < 0:
                    break
                
                best_flat = best_flat_idx.item()
                p_target = best_flat // (groups_per_pack * groups_per_pack)
                rem = best_flat % (groups_per_pack * groups_per_pack)
                u_pos = rem // groups_per_pack
                v_pos = rem % groups_per_pack

                # Swap
                item_u = pack_assignment[max_pack, u_pos].item()
                item_v = pack_assignment[p_target, v_pos].item()
                delta_val = deltas.view(-1)[best_flat].item()

                pack_assignment[max_pack, u_pos] = item_v
                pack_assignment[p_target, v_pos] = item_u
                pack_weights[max_pack] -= delta_val
                pack_weights[p_target] += delta_val
                improved_any = True

            # Phase 2: K-Way LNS (Ruin & Recreate)
            # Repack max, min, and a random pack using multiple randomized attempts
            for _ in range(5):
                max_p = torch.argmax(pack_weights).item()
                min_p = torch.argmin(pack_weights).item()

                if max_p == min_p:
                    break

                lns_packs = [max_p, min_p]
                if num_packs > 2:
                    p3 = random.randint(0, num_packs - 1)
                    while p3 in lns_packs:
                        p3 = random.randint(0, num_packs - 1)
                    lns_packs.append(p3)

                # Collect items
                items_flat = []
                for p in lns_packs:
                    items_flat.extend(pack_assignment[p].tolist())
                
                # Base sort LPT
                sub_weights = [(layer_weights[x], x) for x in items_flat]
                sub_weights.sort(key=lambda x: x[0], reverse=True)

                current_sub_max = max(pack_weights[p].item() for p in lns_packs)
                current_sub_ss = sum(pack_weights[p].item()**2 for p in lns_packs)

                best_sub_assign = None
                best_sub_weights = None
                found_lns_improvement = False

                # 4 attempts: 1 deterministic, 3 randomized
                for attempt in range(4):
                    if attempt == 0:
                        order = sub_weights
                    else:
                        # Perturb weights for sorting
                        order = sorted([(w * random.uniform(0.9, 1.1), w, idx) for w, idx in sub_weights], key=lambda x: x[0], reverse=True)
                        order = [(w, idx) for _, w, idx in order]

                    temp_bins = [[] for _ in lns_packs]
                    temp_ws = [0.0] * len(lns_packs)
                    temp_cts = [0] * len(lns_packs)
                    
                    possible = True
                    for w_val, idx in order:
                        # Greedy Best Fit
                        best_b = -1
                        min_local_w = float('inf')
                        for b in range(len(lns_packs)):
                            if temp_cts[b] < groups_per_pack:
                                if temp_ws[b] < min_local_w:
                                    min_local_w = temp_ws[b]
                                    best_b = b
                        if best_b == -1:
                            possible = False
                            break
                        temp_bins[best_b].append(idx)
                        temp_ws[best_b] += w_val
                        temp_cts[best_b] += 1
                    
                    if possible:
                        new_max = max(temp_ws)
                        new_ss = sum(x*x for x in temp_ws)

                        if new_max < current_sub_max - 1e-6:
                            current_sub_max = new_max
                            current_sub_ss = new_ss
                            best_sub_assign = temp_bins
                            best_sub_weights = temp_ws
                            found_lns_improvement = True
                        elif abs(new_max - current_sub_max) < 1e-6 and new_ss < current_sub_ss - 1e-6:
                            current_sub_ss = new_ss
                            best_sub_assign = temp_bins
                            best_sub_weights = temp_ws
                            found_lns_improvement = True

                if found_lns_improvement:
                    for i_idx, p_idx in enumerate(lns_packs):
                        pack_assignment[p_idx] = torch.tensor(best_sub_assign[i_idx], dtype=torch.int64)
                        pack_weights[p_idx] = best_sub_weights[i_idx]
                    improved_any = True

            if not improved_any:
                break

        # Final write to output tensors
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