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
                     num_packs: int,
                     refine_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        refine_steps: small bounded number of refinement swaps per layer

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Fast path: one item per pack
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Greedy assignment (CPU-friendly)
    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for g in indices[i]:
            # pick lightest pack with remaining capacity
            p = min((pi for pi in range(num_packs) if pack_items[pi] < groups_per_pack),
                    key=pack_weights.__getitem__)
            pack_index[i, g] = p
            rank_in_pack[i, g] = pack_items[p]
            pack_items[p] += 1
            pack_weights[p] += float(weight[i, g].item())

    # Bounded refinement between heaviest and lightest pack
    if groups_per_pack > 1 and refine_steps > 0:
        max_swaps = int(refine_steps)
        for i in range(num_layers):
            for _ in range(max_swaps):
                packs = pack_index[i]  # [num_groups], CPU
                w = weight[i]          # CPU
                # Compute pack loads
                pack_w = torch.zeros(num_packs, dtype=w.dtype)
                pack_w.scatter_add_(0, packs, w)
                h = int(torch.argmax(pack_w))
                l = int(torch.argmin(pack_w))
                delta = float(pack_w[h] - pack_w[l])
                if delta <= 1e-9:
                    break

                heavy_idx = torch.nonzero(packs == h, as_tuple=False).squeeze(1)
                light_idx = torch.nonzero(packs == l, as_tuple=False).squeeze(1)
                if heavy_idx.numel() == 0 or light_idx.numel() == 0:
                    break

                hw = w[heavy_idx]
                lw = w[light_idx]
                lw_sorted, lw_perm = torch.sort(lw)  # ascending
                if lw_sorted.numel() == 0 or hw.numel() == 0:
                    break

                # For each heavy item, find light item closest to target = hw - delta/2
                target = hw - (delta / 2.0)
                pos = torch.searchsorted(lw_sorted, target)
                pos = torch.clamp(pos, 0, lw_sorted.numel() - 1)
                # Consider neighbors pos and pos-1 for best approximation
                cand_pos = torch.stack([pos, torch.clamp(pos - 1, 0, lw_sorted.numel() - 1)], dim=1)
                cand_lw = lw_sorted[cand_pos]  # [H, 2]
                resid = (delta - 2.0 * (hw.unsqueeze(1) - cand_lw)).abs()
                best_flat = int(torch.argmin(resid).item())
                best_h_index = best_flat // 2
                best_option = best_flat % 2
                j_sorted_idx = int(cand_pos[best_h_index, best_option].item())

                wi = float(hw[best_h_index].item())
                wj = float(lw_sorted[j_sorted_idx].item())
                new_delta = abs(delta - 2.0 * (wi - wj))
                improved = new_delta < delta - 1e-9
                if improved:
                    # Commit best 1x1 swap
                    hi = heavy_idx[best_h_index]
                    lj = light_idx[lw_perm[j_sorted_idx]]
                    pack_index[i, hi] = l
                    pack_index[i, lj] = h
                    # Reassign ranks within affected packs to keep 0..groups_per_pack-1
                    for p in (h, l):
                        mask = pack_index[i] == p
                        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                        if idx.numel() == 0:
                            continue
                        prev_rank = rank_in_pack[i, idx]
                        order = torch.argsort(prev_rank)
                        rank_in_pack[i, idx[order]] = torch.arange(order.numel(), dtype=torch.int64)
                    continue
                else:
                    # 2x2 fallback: evaluate top-2 heavy vs bottom-2 light
                    kh = min(2, hw.numel())
                    kl = min(2, lw_sorted.numel())
                    if kh >= 2 and kl >= 2:
                        # top-2 heavy
                        top2_hw_vals, top2_hw_pos = torch.topk(hw, k=2, largest=True)
                        # bottom-2 light: use sorted ascending
                        bot2_lw_vals = lw_sorted[:2]
                        delta22 = abs(delta - 2.0 * float((top2_hw_vals.sum() - bot2_lw_vals.sum()).item()))
                        if delta22 < delta - 1e-9:
                            # Map local positions to global indices and commit both swaps
                            hi1 = heavy_idx[top2_hw_pos[0]]
                            hi2 = heavy_idx[top2_hw_pos[1]]
                            # bot2 indices in original light order via lw_perm
                            lj1 = light_idx[lw_perm[0]]
                            lj2 = light_idx[lw_perm[1]]
                            # perform two swaps
                            pack_index[i, hi1] = l
                            pack_index[i, lj1] = h
                            pack_index[i, hi2] = l
                            pack_index[i, lj2] = h
                            # Reassign ranks for affected packs
                            for p in (h, l):
                                mask = pack_index[i] == p
                                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                                if idx.numel() == 0:
                                    continue
                                prev_rank = rank_in_pack[i, idx]
                                order = torch.argsort(prev_rank)
                                rank_in_pack[i, idx[order]] = torch.arange(order.numel(), dtype=torch.int64)
                            continue
                    # No beneficial swap found; early exit
                    break

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
    dtype_i64 = torch.int64

    # Initialize base mapping (one replica per logical expert)
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    logcnt = torch.ones(n, num_log, dtype=dtype_i64, device=device)

    base = torch.arange(num_log, dtype=dtype_i64, device=device).expand(n, -1)
    phy2log[:, :num_log] = base
    rank[:, :num_log] = 0

    if num_redundant == 0:
        return phy2log, rank, logcnt

    arangen = torch.arange(n, dtype=dtype_i64, device=device)

    # Hybrid allocation with adaptive tail:
    # Use D'Hondt for bulk and Sainte-Laguë for the tail whose length scales with dispersion.
    # Tail fraction = alpha * s where s ~ coefficient of variation clamped to [0.7, 1.3]
    # This keeps tail small but more effective under skew.
    mean = weight.mean(dim=-1).clamp_min(1e-12)
    std = weight.std(dim=-1)
    cv = (std / mean).clamp_min(0)  # per-row CV
    s = float(cv.mean().item())
    s = max(0.7, min(1.3, s))
    alpha = 0.10
    tail = max(1, min(num_redundant, int(round(alpha * num_redundant * s))))
    bulk = num_redundant - tail
    col = num_log

    # Bulk phase (D'Hondt): benefit = weight / r
    for _ in range(max(0, bulk)):
        benefit = weight / logcnt
        best = benefit.max(dim=-1).indices
        phy2log[:, col] = best
        rank[:, col] = logcnt[arangen, best]
        logcnt[arangen, best] += 1
        col += 1

    # Tail phase: per-step A/B between D'Hondt and Sainte-Laguë; choose the one
    # minimizing the predicted post-pick peak per row (lexicographic tie-break on receiver count).
    if tail > 0:
        for _ in range(tail):
            r_f = logcnt.to(weight.dtype)
            avg = weight / r_f
            # current top-2 stats
            k2 = 2 if num_log >= 2 else 1
            top2_vals, top2_idx = torch.topk(avg, k=k2, dim=-1, largest=True)
            cur_max_idx = top2_idx[:, 0]
            cur_max = top2_vals[:, 0]
            cur_second = top2_vals[:, 1] if k2 == 2 else top2_vals[:, 0]

            # candidate winners under D'Hondt and Sainte-Laguë
            idx_D = (weight / r_f).argmax(dim=-1)
            denom_S = (2.0 * r_f - 1.0)
            idx_S = (weight / denom_S).argmax(dim=-1)

            # predicted peaks if allocating to each
            # For candidate c: new avg at c becomes w_c / (r_c + 1); others unchanged.
            wD = weight[arangen, idx_D]
            wS = weight[arangen, idx_S]
            cD_cnt = r_f[arangen, idx_D]
            cS_cnt = r_f[arangen, idx_S]
            newD = wD / (cD_cnt + 1.0)
            newS = wS / (cS_cnt + 1.0)

            # max_except_c is cur_second if c is current argmax else cur_max
            isDmax = (idx_D == cur_max_idx)
            isSmax = (idx_S == cur_max_idx)
            max_except_D = torch.where(isDmax, cur_second, cur_max)
            max_except_S = torch.where(isSmax, cur_second, cur_max)
            peakD = torch.maximum(max_except_D, newD)
            peakS = torch.maximum(max_except_S, newS)

            # choose better; on tie prefer lower receiver count (before increment), then lower index
            better = peakD < peakS
            tie = peakD == peakS
            cnt_tie_pref = cD_cnt < cS_cnt
            idx_tie_pref = idx_D < idx_S
            choose_D = better | (tie & (cnt_tie_pref | ((cD_cnt == cS_cnt) & idx_tie_pref)))
            chosen = torch.where(choose_D, idx_D, idx_S)

            phy2log[:, col] = chosen
            rank[:, col] = logcnt[arangen, chosen]
            logcnt[arangen, chosen] += 1
            col += 1

    # Strengthened replication fix-up per row:
    # Evaluate donors=top-2 by avg and receivers=bottom-2; allow up to 2 moves
    # if both strictly reduce the peak. Keep O(1) candidates and deterministic column choice.
    if num_log > 1 and num_redundant > 0:
        def best_move_for_row(ri: int):
            avg_row = weight[ri] / logcnt[ri].to(weight.dtype)
            kdon = min(2, num_log)
            krec = min(2, num_log)
            top_vals, top_idx = torch.topk(avg_row, k=kdon, largest=True)
            bot_vals, bot_idx = torch.topk(avg_row, k=krec, largest=False)
            cur_max = float(top_vals[0].item())
            second = float((top_vals[1].item() if kdon > 1 else top_vals[0].item()))
            best_pair = None
            best_peak = None
            for d in top_idx.tolist():
                cd = int(logcnt[ri, d].item())
                if cd <= 1:
                    continue
                for r in bot_idx.tolist():
                    if r == d:
                        continue
                    cr = int(logcnt[ri, r].item())
                    new_d = float(weight[ri, d].item()) / float(cd - 1)
                    new_r = float(weight[ri, r].item()) / float(cr + 1)
                    cand_peak = max(second, new_d, new_r)
                    if cand_peak + 1e-12 < cur_max:
                        if best_peak is None or cand_peak < best_peak:
                            best_peak = cand_peak
                            best_pair = (d, r)
            return best_pair

        for ri in range(n):
            # First move (if any)
            mv = best_move_for_row(ri)
            applied = False
            if mv is not None:
                d, r = mv
                donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() > 0:
                    maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
                    col_idx = donor_cols[maxr_idx]
                    new_rank = int(logcnt[ri, r].item())
                    phy2log[ri, col_idx] = r
                    rank[ri, col_idx] = new_rank
                    logcnt[ri, d] -= 1
                    logcnt[ri, r] += 1
                    applied = True

            # Optionally attempt a second move after recomputing with updated counts
            if applied:
                mv2 = best_move_for_row(ri)
                if mv2 is not None:
                    d2, r2 = mv2
                    donor_cols2 = torch.nonzero(phy2log[ri] == d2, as_tuple=False).squeeze(1)
                    if donor_cols2.numel() > 0:
                        maxr_idx2 = torch.argmax(rank[ri, donor_cols2]).item()
                        col_idx2 = donor_cols2[maxr_idx2]
                        new_rank2 = int(logcnt[ri, r2].item())
                        phy2log[ri, col_idx2] = r2
                        rank[ri, col_idx2] = new_rank2
                        logcnt[ri, d2] -= 1
                        logcnt[ri, r2] += 1

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
        tokens_per_group, num_nodes, refine_steps=1)
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
                                                num_gpus // num_nodes,
                                                refine_steps=2)
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