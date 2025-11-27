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

    Uses a 2-pass strategy:
    1. Standard Greedy LPT + Refinement.
    2. Reweighted Greedy LPT (penalizing heaviest pack items) + Refinement on original weights.
    Returns the better of the two for each layer, refined further by 2-item swaps.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    original_weight = weight.float()

    def solve_pass(sort_weight, calc_weight):
        # LPT Sort
        _, sorted_indices = sort_weight.sort(dim=-1, descending=True)
        vals = torch.gather(calc_weight, 1, sorted_indices)

        # Init
        p_weights = torch.zeros(num_layers, num_packs, device=device)
        p_counts = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)
        p_contents = torch.zeros(num_layers, num_packs, groups_per_pack, device=device)
        p_ids = torch.zeros(num_layers, num_packs, groups_per_pack, dtype=torch.int64, device=device)

        row_idx = torch.arange(num_layers, device=device)

        # Greedy
        for i in range(num_groups):
            w = vals[:, i]
            mask = p_counts < groups_per_pack
            curr_w = p_weights.clone()
            curr_w[~mask] = float('inf')
            pid = torch.argmin(curr_w, dim=1)

            sid = p_counts[row_idx, pid]
            p_weights[row_idx, pid] += w
            p_counts[row_idx, pid] += 1
            p_contents[row_idx, pid, sid] = w
            p_ids[row_idx, pid, sid] = sorted_indices[:, i]

        # 1-swap Refinement
        for _ in range(20):
            pw = p_contents.sum(dim=2)
            vmax, imax = pw.max(dim=1)
            vmin, imin = pw.min(dim=1)
            diff = vmax - vmin
            active = diff > 1e-4
            if not active.any(): break

            gmax = imax.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
            gmin = imin.view(-1, 1, 1).expand(-1, 1, groups_per_pack)

            item_max = torch.gather(p_contents, 1, gmax).squeeze(1)
            item_min = torch.gather(p_contents, 1, gmin).squeeze(1)

            delta = item_max.unsqueeze(2) - item_min.unsqueeze(1)
            target = diff.view(-1, 1, 1)
            imp = target - (target - 2 * delta).abs()

            best_imp, best_idx = imp.view(num_layers, -1).max(dim=1)
            do_swap = (best_imp > 1e-5) & active
            if not do_swap.any(): break

            l = torch.where(do_swap)[0]
            s = best_idx[l]
            u = s // groups_per_pack
            v = s % groups_per_pack
            pm = imax[l]
            pn = imin[l]

            vu = p_contents[l, pm, u]
            vv = p_contents[l, pn, v]
            p_contents[l, pm, u] = vv
            p_contents[l, pn, v] = vu

            iu = p_ids[l, pm, u]
            iv = p_ids[l, pn, v]
            p_ids[l, pm, u] = iv
            p_ids[l, pn, v] = iu

        return p_contents, p_ids

    # Pass 1
    c1, id1 = solve_pass(original_weight, original_weight)

    # Pass 2: Reweight heaviest pack items
    w1 = c1.sum(dim=2)
    max_idx = w1.argmax(dim=1)
    g_idx = max_idx.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
    max_items = torch.gather(id1, 1, g_idx).squeeze(1)

    biased_weight = original_weight.clone()
    row_idx = torch.arange(num_layers, device=device).unsqueeze(1).expand(-1, groups_per_pack)
    biased_weight[row_idx, max_items] *= 1.05

    c2, id2 = solve_pass(biased_weight, original_weight)

    # Select best
    l1 = c1.sum(dim=2).max(dim=1).values
    l2 = c2.sum(dim=2).max(dim=1).values
    use2 = l2 < l1

    pack_contents = torch.where(use2.view(-1, 1, 1), c2, c1)
    pack_item_ids = torch.where(use2.view(-1, 1, 1), id2, id1)

    # 2-swap Refinement (on best result)
    if groups_per_pack >= 2:
        for _ in range(10):
            pw = pack_contents.sum(dim=2)
            vmax, imax = pw.max(dim=1)
            vmin, imin = pw.min(dim=1)
            diff = vmax - vmin
            active = diff > 1e-4
            if not active.any(): break

            gmax = imax.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
            gmin = imin.view(-1, 1, 1).expand(-1, 1, groups_per_pack)

            item_max = torch.gather(pack_contents, 1, gmax).squeeze(1)
            item_min = torch.gather(pack_contents, 1, gmin).squeeze(1)

            pair_max = item_max.unsqueeze(2) + item_max.unsqueeze(1)
            pair_min = item_min.unsqueeze(2) + item_min.unsqueeze(1)

            delta = pair_max.unsqueeze(3).unsqueeze(4) - pair_min.unsqueeze(1).unsqueeze(2)
            target = diff.view(-1, 1, 1, 1, 1)
            imp = target - (target - 2 * delta).abs()

            imp.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
            imp.diagonal(dim1=3, dim2=4).fill_(-float('inf'))

            best_imp, best_idx = imp.view(num_layers, -1).max(dim=1)
            do_swap = (best_imp > 1e-5) & active
            if not do_swap.any(): break

            l = torch.where(do_swap)[0]
            s = best_idx[l]
            G = groups_per_pack
            G2 = G*G
            G3 = G2*G

            i_idx = s // G3
            r = s % G3
            j_idx = r // G2
            r = r % G2
            k_idx = r // G
            m_idx = r % G

            pm = imax[l]
            pn = imin[l]

            vi = pack_contents[l, pm, i_idx]
            vj = pack_contents[l, pm, j_idx]
            vk = pack_contents[l, pn, k_idx]
            vm = pack_contents[l, pn, m_idx]

            pack_contents[l, pm, i_idx] = vk
            pack_contents[l, pm, j_idx] = vm
            pack_contents[l, pn, k_idx] = vi
            pack_contents[l, pn, m_idx] = vj

            ii = pack_item_ids[l, pm, i_idx]
            ij = pack_item_ids[l, pm, j_idx]
            ik = pack_item_ids[l, pn, k_idx]
            im = pack_item_ids[l, pn, m_idx]

            pack_item_ids[l, pm, i_idx] = ik
            pack_item_ids[l, pm, j_idx] = im
            pack_item_ids[l, pn, k_idx] = ii
            pack_item_ids[l, pn, m_idx] = ij

    # Final Output
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)

    flat_ids = pack_item_ids.view(num_layers, -1)
    pack_grid = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    rank_grid = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index.scatter_(1, flat_ids, pack_grid)
    rank_in_pack.scatter_(1, flat_ids, rank_grid)

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

    # Pre-compute scores
    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Update state
        logcnt[arangen, redundant_indices] += 1
        new_cnt = logcnt[arangen, redundant_indices].float()
        chosen_weight = weight[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = chosen_weight / new_cnt

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