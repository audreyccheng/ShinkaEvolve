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

    Algorithm: Target-Centric Greedy + Vectorized Swap + LNS

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
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    weight_cpu = weight.cpu()
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device=device)
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device=device)

    # Pre-allocate scatter helpers
    flat_packs_base = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)
    flat_ranks_base = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(-1)

    for i in range(num_layers):
        layer_w = weight_cpu[i]
        w_list = layer_w.tolist()
        total_weight = sum(w_list)
        target_avg = total_weight / num_packs

        # 1. Randomized Greedy (Target-Centric)
        candidates = []
        # Deterministic Sort
        candidates.append(sorted(range(num_groups), key=lambda x: w_list[x], reverse=True))

        # Randomized Sorts
        for _ in range(5):
            noise = torch.rand(num_groups) * 0.1 + 0.95
            cand = (layer_w * noise).argsort(descending=True).tolist()
            candidates.append(cand)

        best_pack_contents = None
        best_pack_weights = None
        min_max_load = float('inf')
        min_ss = float('inf')

        for indices in candidates:
            current_packs = [[] for _ in range(num_packs)]
            current_weights = [0.0] * num_packs
            current_counts = [0] * num_packs

            for idx in indices:
                w = w_list[idx]

                # Target-Centric Best Fit
                best_p = -1
                best_score = float('inf')

                # Check all valid packs
                for p in range(num_packs):
                    if current_counts[p] < groups_per_pack:
                        # Score: squared deviation from target after adding w
                        # (L + w - T)^2
                        new_load = current_weights[p] + w
                        score = (new_load - target_avg) ** 2
                        if score < best_score:
                            best_score = score
                            best_p = p

                current_packs[best_p].append(idx)
                current_weights[best_p] += w
                current_counts[best_p] += 1

            max_l = max(current_weights)
            ss = sum(x*x for x in current_weights)

            if max_l < min_max_load - 1e-6:
                min_max_load = max_l
                min_ss = ss
                best_pack_contents = current_packs
                best_pack_weights = current_weights
            elif abs(max_l - min_max_load) < 1e-6:
                if ss < min_ss - 1e-6:
                    min_ss = ss
                    best_pack_contents = current_packs
                    best_pack_weights = current_weights

        # 2. Vectorized Refinement
        # Convert to tensors
        pack_assignment = torch.tensor(best_pack_contents, dtype=torch.int64)
        pack_weights = torch.tensor(best_pack_weights, dtype=torch.float32)

        # Iteration
        for _ in range(10): # Num phases
            improved_swap = False
            # Swap loop
            for _ in range(10):
                max_p = torch.argmax(pack_weights).item()
                max_w = pack_weights[max_p].item()

                u_indices = pack_assignment[max_p]
                w_u = layer_w[u_indices] # [G]

                w_v = layer_w[pack_assignment] # [M, G]

                diffs = (max_w - pack_weights).view(-1, 1, 1) # [M, 1, 1]
                deltas = w_u.view(1, groups_per_pack, 1) - w_v.view(num_packs, 1, groups_per_pack)

                mask = (deltas > 1e-6) & (deltas < diffs)
                if not mask.any(): break

                gains = deltas * (diffs - deltas)
                gains = torch.where(mask, gains, -1.0)

                best_flat = torch.argmax(gains).item()
                if gains.view(-1)[best_flat] < 0: break

                p_target = best_flat // (groups_per_pack * groups_per_pack)
                rem = best_flat % (groups_per_pack * groups_per_pack)
                u_idx = rem // groups_per_pack
                v_idx = rem % groups_per_pack

                # Swap
                val_u = pack_assignment[max_p, u_idx].item()
                val_v = pack_assignment[p_target, v_idx].item()

                pack_assignment[max_p, u_idx] = val_v
                pack_assignment[p_target, v_idx] = val_u

                d_val = deltas.view(-1)[best_flat].item()
                pack_weights[max_p] -= d_val
                pack_weights[p_target] += d_val
                improved_swap = True

            # 3. LNS Refinement (Target-Centric)
            if num_packs < 2: break

            # Identify candidates
            sorted_p = torch.argsort(pack_weights, descending=True)
            p_max = sorted_p[0].item()
            p_min = sorted_p[-1].item()

            lns_packs = [p_max, p_min]
            if num_packs > 2:
                 # Pick random
                 r = torch.randint(0, num_packs, (1,)).item()
                 while r in lns_packs:
                     r = torch.randint(0, num_packs, (1,)).item()
                 lns_packs.append(r)

            # Extract items
            flat_items = []
            for p in lns_packs:
                flat_items.extend(pack_assignment[p].tolist())

            # Sort LPT
            flat_items.sort(key=lambda x: w_list[x], reverse=True)

            # Re-Greedy (Target-Centric) on subset
            temp_bins = [[] for _ in lns_packs]
            temp_ws = [0.0] * len(lns_packs)
            temp_cnts = [0] * len(lns_packs)

            # We calculate local target avg for this subset to guide packing
            subset_total = sum(w_list[x] for x in flat_items)
            subset_target = subset_total / len(lns_packs)

            possible = True
            for item in flat_items:
                w = w_list[item]
                best_b = -1
                best_sc = float('inf')

                for b in range(len(lns_packs)):
                    if temp_cnts[b] < groups_per_pack:
                        sc = (temp_ws[b] + w - subset_target) ** 2
                        if sc < best_sc:
                            best_sc = sc
                            best_b = b

                if best_b == -1:
                    possible = False; break
                temp_bins[best_b].append(item)
                temp_ws[best_b] += w
                temp_cnts[best_b] += 1

            if possible:
                old_max = max(pack_weights[p].item() for p in lns_packs)
                new_max = max(temp_ws)
                old_ss = sum(pack_weights[p].item()**2 for p in lns_packs)
                new_ss = sum(x**2 for x in temp_ws)

                if new_max < old_max - 1e-6 or (abs(new_max - old_max) < 1e-6 and new_ss < old_ss - 1e-6):
                    for idx, p in enumerate(lns_packs):
                        pack_assignment[p] = torch.tensor(temp_bins[idx], dtype=torch.int64)
                        pack_weights[p] = temp_ws[idx]
                    improved_swap = True # Treat as improvement to continue loop

            if not improved_swap:
                break

        # Assign to output
        flat_assignment = pack_assignment.view(-1).to(device)
        pack_index[i].scatter_(0, flat_assignment, flat_packs_base)
        rank_in_pack[i].scatter_(0, flat_assignment, flat_ranks_base)

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