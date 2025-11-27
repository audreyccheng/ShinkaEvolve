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


def _refine_packing(weights: torch.Tensor,
                    pack_indices: torch.Tensor,
                    pack_weights: torch.Tensor,
                    num_packs: int,
                    groups_per_pack: int,
                    max_iters: int = 50) -> torch.Tensor:
    """
    Iteratively refines the packing by swapping items.
    Supports:
    1. 1-item swap between Max Load Pack and Any Other Pack.
    2. 2-item swap between Max Load Pack and Min Load Pack.
    """
    device = weights.device

    # Pre-compute indices for 2-item pairs if applicable
    can_do_pairs = groups_per_pack >= 2
    if can_do_pairs:
        # Upper triangle indices excluding diagonal
        pair_idxs = torch.triu_indices(groups_per_pack, groups_per_pack, offset=1, device=device)
        num_pairs = pair_idxs.shape[1]

    for _ in range(max_iters):
        # 1. Identify stats
        max_load, max_pid = torch.max(pack_weights, dim=0)

        # 2. Strategy A: 1-item swap Max vs All
        items_max = pack_indices[max_pid] # [G]
        w_max = weights[items_max]        # [G]

        all_items = pack_indices          # [M, G]
        w_all = weights[all_items]        # [M, G]

        # deltas[p, j, i] = w_max[i] - w_all[p, j]
        # [1, 1, G] - [M, G, 1] -> [M, G, G]
        deltas_1 = w_max.view(1, 1, -1) - w_all.view(num_packs, groups_per_pack, 1)

        new_max_1 = max_load - deltas_1
        new_other_1 = pack_weights.view(num_packs, 1, 1) + deltas_1

        obj_1 = torch.max(new_max_1, new_other_1)
        # Mask self-swap
        obj_1[max_pid] = float('inf')

        min_obj_1, idx_flat_1 = torch.min(obj_1.flatten(), dim=0)

        # 3. Strategy B: 2-item swap Max vs Min
        min_obj_2 = float('inf')

        if can_do_pairs:
            min_load, min_pid = torch.min(pack_weights, dim=0)
            if max_pid != min_pid:
                items_min = pack_indices[min_pid]
                w_min = weights[items_min]

                # Sums of pairs
                w_max_pairs = w_max[pair_idxs[0]] + w_max[pair_idxs[1]] # [N_pairs]
                w_min_pairs = w_min[pair_idxs[0]] + w_min[pair_idxs[1]] # [N_pairs]

                # Delta = pair_from_max - pair_from_min
                # [N_p, 1] - [1, N_p] -> [N_p, N_p]
                deltas_2 = w_max_pairs.unsqueeze(1) - w_min_pairs.unsqueeze(0)

                new_max_2 = max_load - deltas_2
                new_min_2 = min_load + deltas_2

                # Objective: minimize max(new_max, new_min)
                # We assume intermediate packs don't become new max, which is a heuristic.
                obj_2 = torch.max(new_max_2, new_min_2)

                min_obj_2, idx_flat_2 = torch.min(obj_2.flatten(), dim=0)

        # 4. Select Best Move
        best_obj = max_load - 1e-6
        move_type = 0 # 0: None, 1: 1-item, 2: 2-item

        if min_obj_1 < best_obj:
            best_obj = min_obj_1
            move_type = 1

        if min_obj_2 < best_obj:
            best_obj = min_obj_2
            move_type = 2

        if move_type == 0:
            break

        # 5. Execute Move
        if move_type == 1:
            # 1-item swap
            # idx_flat_1: p * G * G + j * G + i
            G2 = groups_per_pack * groups_per_pack
            p = idx_flat_1 // G2
            rem = idx_flat_1 % G2
            j = rem // groups_per_pack
            i = rem % groups_per_pack

            val_max = items_max[i].item()
            val_other = all_items[p, j].item()

            pack_indices[max_pid, i] = val_other
            pack_indices[p, j] = val_max

            d = deltas_1[p, j, i]
            pack_weights[max_pid] -= d
            pack_weights[p] += d

        else: # move_type == 2
            # 2-item swap
            # idx_flat_2: pair_max_idx * num_pairs + pair_min_idx
            pm_idx = idx_flat_2 // num_pairs
            pn_idx = idx_flat_2 % num_pairs

            m1, m2 = pair_idxs[0, pm_idx], pair_idxs[1, pm_idx]
            n1, n2 = pair_idxs[0, pn_idx], pair_idxs[1, pn_idx]

            val_m1 = pack_indices[max_pid, m1].item()
            val_m2 = pack_indices[max_pid, m2].item()
            val_n1 = pack_indices[min_pid, n1].item()
            val_n2 = pack_indices[min_pid, n2].item()

            pack_indices[max_pid, m1] = val_n1
            pack_indices[max_pid, m2] = val_n2
            pack_indices[min_pid, n1] = val_m1
            pack_indices[min_pid, n2] = val_m2

            d = deltas_2[pm_idx, pn_idx]
            pack_weights[max_pid] -= d
            pack_weights[min_pid] += d

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_attempts: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Implements a Greedy LPT initialization with Iterative Re-weighting and
    Swapping refinement.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        num_attempts: number of iterations for re-weighting optimization

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Optimization for trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Pre-allocate outputs
    pack_index = torch.full_like(weight, -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(weight, -1, dtype=torch.int64)

    # Pre-compute grid indices for scattering results later
    p_ids_grid = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack)
    r_ids_grid = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1)
    flat_p_ids = p_ids_grid.flatten()
    flat_r_ids = r_ids_grid.flatten()

    for i in range(num_layers):
        best_max_load = float('inf')
        best_assignment = None

        # Virtual weights for iterative re-weighting
        virtual_weight = weight[i].clone().float()

        for attempt in range(num_attempts):
            # Sort based on virtual weights to determine order
            sorted_indices = torch.argsort(virtual_weight, descending=True)

            # 1. Greedy Initialization using Virtual Weights for order, Real Weights for load
            current_pack_weights = torch.zeros(num_packs, device=device, dtype=weight.dtype)
            current_pack_counts = torch.zeros(num_packs, device=device, dtype=torch.int64)
            pack_assignment = torch.zeros((num_packs, groups_per_pack),
                                          dtype=torch.int64, device=device)

            for j in range(num_groups):
                item_idx = sorted_indices[j]
                w = weight[i, item_idx] # Real weight

                # Find valid pack with min weight
                is_full = current_pack_counts >= groups_per_pack
                masked_weights = torch.where(is_full, float('inf'), current_pack_weights)
                best_pack = torch.argmin(masked_weights)

                # Assign
                slot = current_pack_counts[best_pack]
                pack_assignment[best_pack, slot] = item_idx
                current_pack_weights[best_pack] += w
                current_pack_counts[best_pack] += 1

            # 2. Iterative Refinement (Swapping) using Real Weights
            pack_assignment = _refine_packing(
                weight[i], pack_assignment, current_pack_weights,
                num_packs, groups_per_pack
            )

            # 3. Check solution quality
            max_load, max_pid = torch.max(current_pack_weights, dim=0)

            if max_load < best_max_load:
                best_max_load = max_load
                best_assignment = pack_assignment.clone()

            # 4. Re-weighting: Boost virtual weights of items in the heaviest pack
            # This forces them to be processed earlier in the next greedy pass
            if attempt < num_attempts - 1:
                items_in_max = pack_assignment[max_pid]
                virtual_weight[items_in_max] *= 1.05

        # Store best result for this layer
        if best_assignment is not None:
            flat_items = best_assignment.flatten()
            pack_index[i, flat_items] = flat_p_ids
            rank_in_pack[i, flat_items] = flat_r_ids

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.

    Uses a vectorized greedy approach (Jefferson/D'Hondt method) which is
    optimal for the min-max load objective with discrete allocations.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Initialize phy2log with basic mapping [0, 1, ..., num_log-1] for the first part
    # We will overwrite the redundant parts in the loop
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Greedily assign remaining replicas
    # We need to assign `num_phy - num_log` replicas.
    # In each step, assign to the expert with the highest current load per replica.
    # This loop is vectorized over layers (dim 0).
    for i in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)

        # Assign the new replica
        phy2log[:, i] = indices

        # Record the rank for this new replica
        rank[:, i] = logcnt[arangen, indices]

        # Increment replica count
        logcnt[arangen, indices] += 1

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing: Groups->Nodes, then Replicas->GPUs.
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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # Use improved balanced packing with swapping
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

    # Use improved balanced packing here as well
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