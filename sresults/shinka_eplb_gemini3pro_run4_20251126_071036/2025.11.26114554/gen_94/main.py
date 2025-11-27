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
import itertools

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Handle trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # CPU processing for complex sequential logic
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    # Heuristic parameters
    num_restarts = 3
    max_refine_steps = 25

    for i in range(num_layers):
        row_weight = weight_cpu[i]
        row_weight_list = row_weight.tolist()

        best_diff = float('inf')
        best_packs = None

        # Base indices for deterministic LPT
        base_indices = torch.argsort(row_weight, descending=True).tolist()

        for attempt in range(num_restarts):
            # 1. Initialization
            if attempt == 0:
                indices = base_indices
            else:
                noise = torch.rand(num_groups, device="cpu") * 0.15 + 0.925
                indices = torch.argsort(row_weight * noise, descending=True).tolist()

            current_packs = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs

            # Greedy Packing
            for group_idx in indices:
                w = row_weight_list[group_idx]
                best_p = -1
                min_val = float('inf')
                for p in range(num_packs):
                    if len(current_packs[p]) < groups_per_pack:
                        if pack_weights[p] < min_val:
                            min_val = pack_weights[p]
                            best_p = p
                current_packs[best_p].append(group_idx)
                pack_weights[best_p] += w

            # 2. Refinement Phase
            for step in range(max_refine_steps):
                # Identify max pack
                max_p = max(range(num_packs), key=pack_weights.__getitem__)
                max_val = pack_weights[max_p]

                min_val = min(pack_weights)
                if max_val - min_val < 1e-6:
                    break

                best_move = None
                best_move_improvement = -1.0

                candidate_packs = [p for p in range(num_packs) if p != max_p]
                # Sort candidates by weight: testing lightest first often helps move most weight
                candidate_packs.sort(key=pack_weights.__getitem__)

                for other_p in candidate_packs:
                    other_val = pack_weights[other_p]

                    # Target is average of the pair
                    avg = (max_val + other_val) / 2.0
                    potential_reduction = max_val - avg

                    if potential_reduction < best_move_improvement + 1e-6:
                        continue

                    pool = current_packs[max_p] + current_packs[other_p]
                    pool_weights = [row_weight_list[x] for x in pool]

                    K = groups_per_pack
                    pool_len = 2 * K

                    best_local_set_A = None
                    best_local_max = float('inf')

                    # EXACT SOLVER for small K
                    if pool_len <= 14:
                         # Fix first element of pool to Set A to reduce symmetry (13 choose 6 vs 14 choose 7)
                         indices_rest = list(range(1, pool_len))
                         w_0 = pool_weights[0]

                         for comb in itertools.combinations(indices_rest, K - 1):
                             current_w = w_0 + sum(pool_weights[idx] for idx in comb)
                             other_w = (max_val + other_val) - current_w
                             local_max = max(current_w, other_w)

                             if local_max < best_local_max:
                                 best_local_max = local_max
                                 best_local_set_A = [0] + list(comb)
                                 if abs(local_max - avg) < 1e-5:
                                     break
                    else:
                        # Heuristic Solver
                        pool_indices = list(range(pool_len))
                        sub_trials = 10
                        for sub in range(sub_trials):
                            noise_arr = [random.uniform(0.9, 1.1) for _ in range(pool_len)]
                            sorted_idx = sorted(pool_indices, key=lambda i: pool_weights[i] * noise_arr[i], reverse=True)

                            w1, w2 = 0.0, 0.0
                            c1, c2 = 0, 0
                            s1, s2 = [], []

                            for idx in sorted_idx:
                                w = pool_weights[idx]
                                if c1 < K and c2 < K:
                                    if w1 <= w2:
                                        w1 += w; c1 += 1; s1.append(idx)
                                    else:
                                        w2 += w; c2 += 1; s2.append(idx)
                                elif c1 < K:
                                    w1 += w; c1 += 1; s1.append(idx)
                                else:
                                    w2 += w; c2 += 1; s2.append(idx)

                            # Swap Local Search
                            while True:
                                improved_swap = False
                                current_max = max(w1, w2)
                                for i1, idx1 in enumerate(s1):
                                    for i2, idx2 in enumerate(s2):
                                        diff_w = pool_weights[idx1] - pool_weights[idx2]
                                        nw1, nw2 = w1 - diff_w, w2 + diff_w
                                        nmax = max(nw1, nw2)
                                        if nmax < current_max - 1e-6:
                                            w1, w2 = nw1, nw2
                                            s1[i1], s2[i2] = s2[i2], s1[i1]
                                            current_max = nmax
                                            improved_swap = True
                                            break
                                    if improved_swap: break
                                if not improved_swap: break

                            if max(w1, w2) < best_local_max:
                                best_local_max = max(w1, w2)
                                best_local_set_A = list(s1)

                    # Check improvement
                    improvement = max_val - best_local_max
                    if improvement > best_move_improvement + 1e-6:
                        best_move_improvement = improvement
                        set_A_indices = best_local_set_A
                        set_A_set = set(set_A_indices)
                        new_items_A = [pool[x] for x in set_A_indices]
                        new_items_B = [pool[x] for x in range(pool_len) if x not in set_A_set]
                        # Calculate exact new weights to avoid drift
                        w_A_exact = sum(row_weight_list[x] for x in new_items_A)
                        w_B_exact = sum(row_weight_list[x] for x in new_items_B)
                        best_move = (other_p, new_items_A, new_items_B, w_A_exact, w_B_exact)

                        if abs(best_local_max - avg) < 1e-5:
                             break

                if best_move:
                    other_p, items_A, items_B, w_A, w_B = best_move
                    current_packs[max_p] = items_A
                    current_packs[other_p] = items_B
                    pack_weights[max_p] = w_A
                    pack_weights[other_p] = w_B
                else:
                    break

            current_diff = max(pack_weights) - min(pack_weights)
            if current_diff < best_diff:
                best_diff = current_diff
                best_packs = [list(p) for p in current_packs]
                if best_diff < 1e-6:
                    break

        for p in range(num_packs):
            for r, g in enumerate(best_packs[p]):
                pack_index[i, g] = p
                rank_in_pack[i, g] = r

    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.
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

    # Efficient incremental update of scores
    current_scores = weight.clone() # weight / 1

    for i in range(num_log, num_phy):
        # Pick expert with max current load
        redundant_indices = current_scores.max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        logcnt[arangen, redundant_indices] += 1

        # Update score: weight / new_count
        selected_weights = weight[arangen, redundant_indices]
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
    Hierarchical rebalancing strategy.
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