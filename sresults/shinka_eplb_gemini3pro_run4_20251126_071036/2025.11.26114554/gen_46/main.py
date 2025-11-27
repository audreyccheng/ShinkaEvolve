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
        packs_zz = [[] for _ in range(num_packs)]
        loads_zz = [0.0] * num_packs
        for k, (idx, w) in enumerate(items_desc):
            row = k // num_packs
            col = k % num_packs
            bin_idx = col if (row % 2 == 0) else (num_packs - 1 - col)
            packs_zz[bin_idx].append((idx, w))
            loads_zz[bin_idx] += w
        candidates.append((packs_zz, loads_zz))

        # --- Strategy 2: Projected Best-Fit LPT ---
        packs_plpt = [[] for _ in range(num_packs)]
        loads_plpt = [0.0] * num_packs
        caps_plpt = [groups_per_pack] * num_packs
        current_rem_weight = total_weight
        current_rem_count = num_groups
        for idx, w in items_desc:
            current_rem_weight -= w
            current_rem_count -= 1
            avg_rem = current_rem_weight / current_rem_count if current_rem_count > 0 else 0.0
            best_p = -1
            min_score = float('inf')
            for p in range(num_packs):
                if caps_plpt[p] > 0:
                    proj_load = loads_plpt[p] + w + (caps_plpt[p] - 1) * avg_rem
                    score = abs(proj_load - target_avg)
                    if score < min_score:
                        min_score = score
                        best_p = p
            packs_plpt[best_p].append((idx, w))
            loads_plpt[best_p] += w
            caps_plpt[best_p] -= 1
        candidates.append((packs_plpt, loads_plpt))

        # --- Strategy 3: Randomized LPT Restarts ---
        # Reduced to 2 restarts for efficiency
        for _ in range(2):
            noise = torch.rand(num_groups, device="cpu") * 0.05 + 0.975
            noisy_items = []
            for k in range(num_groups):
                orig_idx, orig_w = items[k]
                noisy_items.append((orig_idx, orig_w, orig_w * noise[k].item()))
            noisy_items.sort(key=lambda x: x[2], reverse=True)
            packs_rnd = [[] for _ in range(num_packs)]
            loads_rnd = [0.0] * num_packs
            for idx, w, _ in noisy_items:
                best_p = -1
                min_load = float('inf')
                for p in range(num_packs):
                    if len(packs_rnd[p]) < groups_per_pack:
                        if loads_rnd[p] < min_load:
                            min_load = loads_rnd[p]
                            best_p = p
                packs_rnd[best_p].append((idx, w))
                loads_rnd[best_p] += w
            candidates.append((packs_rnd, loads_rnd))

        # --- Evaluation and Refinement ---
        best_diff_global = float('inf')
        best_packing_global = None

        for packs_init, loads_init in candidates:
            current_packs = [list(p) for p in packs_init]
            current_loads = list(loads_init)

            # Iterative Improvement
            for _ in range(30):
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

                improved = False

                # --- 1. Reshuffle (Partitioning) Max & Min ---
                pool = current_packs[max_p] + current_packs[min_p]
                pool.sort(key=lambda x: x[1], reverse=True)
                new_load_max = 0.0
                new_load_min = 0.0
                new_pack_max = []
                new_pack_min = []
                valid_reshuffle = True
                for item in pool:
                    if len(new_pack_max) < groups_per_pack and (len(new_pack_min) == groups_per_pack or new_load_max < new_load_min):
                        new_pack_max.append(item)
                        new_load_max += item[1]
                    elif len(new_pack_min) < groups_per_pack:
                        new_pack_min.append(item)
                        new_load_min += item[1]
                    else:
                        valid_reshuffle = False
                        break
                if valid_reshuffle:
                    if abs(new_load_max - new_load_min) < diff - 1e-6:
                        current_packs[max_p] = new_pack_max
                        current_packs[min_p] = new_pack_min
                        current_loads[max_p] = new_load_max
                        current_loads[min_p] = new_load_min
                        improved = True
                if improved: continue

                # --- 2. Pairwise Swap Refinement ---
                best_swap = None
                target_delta = diff / 2.0
                best_gap_sq = diff * diff
                for i1, (u, w_u) in enumerate(current_packs[max_p]):
                    for i2, (v, w_v) in enumerate(current_packs[min_p]):
                        delta = w_u - w_v
                        if 0 < delta < diff:
                            gap = abs(delta - target_delta)
                            if gap * gap < best_gap_sq:
                                best_gap_sq = gap * gap
                                best_swap = (i1, i2, delta, u, w_u, v, w_v)
                                if gap < 1e-6: break
                    if best_swap and best_gap_sq < 1e-12: break

                if best_swap:
                    i_from, i_to, delta, u_idx, u_w, v_idx, v_w = best_swap
                    current_packs[max_p][i_from] = (v_idx, v_w)
                    current_packs[min_p][i_to] = (u_idx, u_w)
                    current_loads[max_p] -= delta
                    current_loads[min_p] += delta
                    improved = True
                if improved: continue

                # --- 3. 3-Way Cyclic Swap Refinement ---
                # Max gives to Mid, Mid gives to Min, Min gives to Max
                # Swap (u from Max) -> Mid (replaces v)
                # Swap (v from Mid) -> Min (replaces w)
                # Swap (w from Min) -> Max (replaces u)
                mid_candidates = [p for p in range(num_packs) if p != max_p and p != min_p]
                for p_mid in mid_candidates:
                    for i_u, (u, w_u) in enumerate(current_packs[max_p]):
                         for i_w, (w, w_w) in enumerate(current_packs[min_p]):
                            if w_w >= w_u: continue
                            for i_v, (v, w_v) in enumerate(current_packs[p_mid]):
                                if w_v > w_w:
                                    new_max = current_loads[max_p] + w_w - w_u
                                    new_mid = current_loads[p_mid] + w_u - w_v
                                    new_min = current_loads[min_p] + w_v - w_w
                                    if new_mid < max_v and new_mid > min_v:
                                        if max(new_max, new_mid, new_min) < max_v:
                                            current_packs[max_p][i_u] = (w, w_w)
                                            current_packs[p_mid][i_v] = (u, w_u)
                                            current_packs[min_p][i_w] = (v, w_v)
                                            current_loads[max_p] = new_max
                                            current_loads[p_mid] = new_mid
                                            current_loads[min_p] = new_min
                                            improved = True
                                            break
                            if improved: break
                         if improved: break
                    if improved: break

                if not improved:
                    break

            final_diff = max(current_loads) - min(current_loads)
            if final_diff < best_diff_global:
                best_diff_global = final_diff
                best_packing_global = current_packs
                if best_diff_global < 1e-6: break

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