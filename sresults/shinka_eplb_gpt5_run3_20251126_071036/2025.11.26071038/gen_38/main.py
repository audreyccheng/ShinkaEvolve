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

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i
                 for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1

    # Bounded k-candidate (k=2) refinement per layer to reduce max imbalance
    if groups_per_pack > 1:
        max_swaps = 2  # keep small to preserve speed
        for i in range(num_layers):
            for _ in range(max_swaps):
                packs = pack_index[i]  # [num_groups], CPU
                w = weight[i]  # CPU
                # Compute pack loads
                pack_w = torch.zeros(num_packs, dtype=w.dtype)
                pack_w.scatter_add_(0, packs, w)
                h = int(torch.argmax(pack_w))
                # Consider the lightest and second-lightest packs
                light_order = torch.argsort(pack_w, descending=False)
                l0 = int(light_order[0].item())
                light_candidates = [l0]
                if num_packs >= 2:
                    l1 = int(light_order[1].item())
                    if l1 != l0 and l1 != h:
                        light_candidates.append(l1)

                # If heaviest is also the sole light candidate, stop
                if len(light_candidates) == 1 and l0 == h:
                    break

                heavy_idx = torch.nonzero(packs == h, as_tuple=False).squeeze(1)
                if heavy_idx.numel() == 0:
                    break
                hw_all = w[heavy_idx]
                kh = min(2, hw_all.numel())
                if kh == 0:
                    break
                h_sel_local = torch.topk(hw_all, k=kh, largest=True).indices
                h_sel = heavy_idx[h_sel_local]
                topk_hw = w[h_sel]  # [kh]

                # Evaluate best 1x1 swap across light candidates
                best_choice = None  # (new_delta, hi, lj, chosen_light)
                for l in light_candidates:
                    if l == h:
                        continue
                    light_idx = torch.nonzero(packs == l, as_tuple=False).squeeze(1)
                    if light_idx.numel() == 0:
                        continue
                    lw_all = w[light_idx]
                    kl = min(2, lw_all.numel())
                    if kl == 0:
                        continue
                    bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=kl, largest=True)
                    bottomk_lw = -bottomk_lw_vals
                    delta = float((pack_w[h] - pack_w[l]).item())
                    if delta <= 1e-9:
                        continue
                    diff = topk_hw.unsqueeze(1) - bottomk_lw.unsqueeze(0)  # [kh, kl]
                    cand_new_delta = (delta - 2.0 * diff).abs()
                    best_flat = int(torch.argmin(cand_new_delta).item())
                    ih = best_flat // kl
                    jl = best_flat % kl
                    new_delta = float(cand_new_delta[ih, jl].item())
                    if (best_choice is None) or (new_delta < best_choice[0] - 0.0):
                        best_choice = (
                            new_delta,
                            h_sel[ih],
                            light_idx[bottomk_pos_l[jl]],
                            l,
                        )

                # Optionally evaluate a 2x2 exchange (top-2 heavy vs bottom-2 from absolute lightest)
                two_two_candidate = None
                if hw_all.numel() >= 2 and l0 != h:
                    light_idx0 = torch.nonzero(packs == l0, as_tuple=False).squeeze(1)
                    if light_idx0.numel() >= 2:
                        kh2 = min(2, hw_all.numel())
                        kl2 = min(2, light_idx0.numel())
                        t_h_vals, t_h_pos = torch.topk(hw_all, k=kh2, largest=True)
                        lw0 = w[light_idx0]
                        b_l_vals, b_l_pos = torch.topk(-lw0, k=kl2, largest=True)
                        b_l_vals = -b_l_vals
                        delta0 = float((pack_w[h] - pack_w[l0]).item())
                        new_delta_22 = abs(delta0 - 2.0 * float((t_h_vals.sum() - b_l_vals.sum()).item()))
                        two_two_candidate = (new_delta_22, t_h_pos, b_l_pos, l0)

                applied = False
                # Prefer 2x2 only if strictly better than best 1x1
                if two_two_candidate is not None:
                    nd22, hpos22, lpos22, lsel22 = two_two_candidate
                    better_than_1x1 = (best_choice is None) or (nd22 + 1e-12 < best_choice[0])
                    if better_than_1x1 and nd22 + 1e-9 < float((pack_w[h] - pack_w[lsel22]).item()):
                        # Map local positions to global indices
                        hi1 = heavy_idx[hpos22[0]]
                        lj1 = torch.nonzero(packs == lsel22, as_tuple=False).squeeze(1)[lpos22[0]]
                        # If available, take second pair
                        if hpos22.numel() >= 2 and lpos22.numel() >= 2:
                            hi2 = heavy_idx[hpos22[1]]
                            lj2 = torch.nonzero(packs == lsel22, as_tuple=False).squeeze(1)[lpos22[1]]
                        else:
                            hi2 = None
                            lj2 = None

                        # Commit swaps
                        wi1 = float(w[hi1].item())
                        wj1 = float(w[lj1].item())
                        pack_index[i, hi1] = lsel22
                        pack_index[i, lj1] = h
                        pack_w[h] = pack_w[h] - wi1 + wj1
                        pack_w[lsel22] = pack_w[lsel22] - wj1 + wi1

                        if hi2 is not None and lj2 is not None:
                            wi2 = float(w[hi2].item())
                            wj2 = float(w[lj2].item())
                            pack_index[i, hi2] = lsel22
                            pack_index[i, lj2] = h
                            pack_w[h] = pack_w[h] - wi2 + wj2
                            pack_w[lsel22] = pack_w[lsel22] - wj2 + wi2

                        # Reassign ranks within affected packs
                        for p in (h, lsel22):
                            mask = pack_index[i] == p
                            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                            if idx.numel() > 0:
                                prev_rank = rank_in_pack[i, idx]
                                order = torch.argsort(prev_rank)
                                new_ranks = torch.arange(order.numel(), dtype=torch.int64)
                                rank_in_pack[i, idx[order]] = new_ranks
                        applied = True

                if not applied and best_choice is not None:
                    new_delta_11, hi, lj, lsel = best_choice
                    delta_sel = float((pack_w[h] - pack_w[lsel]).item())
                    if new_delta_11 + 1e-9 < delta_sel:
                        wi = float(w[hi].item())
                        wj = float(w[lj].item())
                        # Commit 1x1 swap
                        pack_index[i, hi] = lsel
                        pack_index[i, lj] = h
                        pack_w[h] = pack_w[h] - wi + wj
                        pack_w[lsel] = pack_w[lsel] - wj + wi
                        # Reassign ranks within affected packs
                        for p in (h, lsel):
                            mask = pack_index[i] == p
                            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                            if idx.numel() == 0:
                                continue
                            prev_rank = rank_in_pack[i, idx]
                            order = torch.argsort(prev_rank)
                            new_ranks = torch.arange(order.numel(), dtype=torch.int64)
                            rank_in_pack[i, idx[order]] = new_ranks
                        continue
                    else:
                        break
                elif applied:
                    # 2x2 applied, continue to next iteration
                    continue
                else:
                    # no improving move available
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

    # Initialize base mapping (one replica per logical expert)
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)

    if num_redundant == 0:
        return phy2log, rank, logcnt

    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Hybrid allocation with micro-tuned tail (tiny A/B/C around 10%)
    if num_redundant > 0:
        tail0 = max(1, (num_redundant + 9) // 10)
        tail_candidates = sorted(set([
            max(1, tail0 - 1),
            tail0,
            min(num_redundant, tail0 + 1),
        ]))
        best_tail = tail_candidates[0]
        best_pred = None
        # simulate only counts to choose tail; bulk uses D'Hondt, tail uses Sainte-Laguë
        for tl in tail_candidates:
            bulk_sim = max(0, num_redundant - tl)
            cnt_sim = logcnt.clone()
            # bulk
            for _ in range(bulk_sim):
                benefit = weight / cnt_sim.to(weight.dtype)
                best_idx = benefit.argmax(dim=-1)
                cnt_sim[arangen, best_idx] += 1
            # tail
            for _ in range(tl):
                denom = (2 * cnt_sim - 1).to(weight.dtype)
                benefit = weight / denom
                best_idx = benefit.argmax(dim=-1)
                cnt_sim[arangen, best_idx] += 1
            avg_sim = weight / cnt_sim.to(weight.dtype)
            pred_peak = avg_sim.max(dim=-1).values.mean().item()
            if best_pred is None or pred_peak < best_pred:
                best_pred = pred_peak
                best_tail = tl
        tail = best_tail
    else:
        tail = 0
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

    # Tail phase (Sainte-Laguë): benefit = weight / (2r - 1)
    if tail > 0:
        for _ in range(tail):
            denom = (2 * logcnt - 1).to(weight.dtype)
            benefit = weight / denom
            best = benefit.max(dim=-1).indices
            phy2log[:, col] = best
            rank[:, col] = logcnt[arangen, best]
            logcnt[arangen, best] += 1
            col += 1

    # Strengthened replication fix-up per row:
    # Evaluate moves from top-2 donors (by avg load) to bottom-2 receivers and
    # apply the single best move if it strictly reduces the peak.
    # If the first improvement is shallow (<10%), attempt one more best move.
    if num_log > 1 and num_redundant > 0:
        kd = min(2, num_log)
        kr = min(2, num_log)
        for ri in range(n):
            # helper to get best move for current counts
            def best_move_for_row() -> tuple[bool, int, int, float, float]:
                r_f = logcnt[ri].to(weight.dtype)
                avg = weight[ri] / r_f
                top_vals, top_idx = torch.topk(avg, k=kd, largest=True)
                bot_vals, bot_idx = torch.topk(avg, k=kr, largest=False)
                cur_max = float(top_vals[0].item())
                second = float((top_vals[1].item() if kd > 1 else top_vals[0].item()))
                best_pair = None
                best_new_peak = None
                for d in top_idx.tolist():
                    cd = int(logcnt[ri, d].item())
                    if cd <= 1:
                        continue
                    for r in bot_idx.tolist():
                        if d == r:
                            continue
                        cr = int(logcnt[ri, r].item())
                        new_d = float(weight[ri, d].item()) / float(cd - 1)
                        new_r = float(weight[ri, r].item()) / float(cr + 1)
                        cand_peak = max(second, new_d, new_r)
                        if cand_peak + 1e-12 < cur_max:
                            if best_new_peak is None or cand_peak < best_new_peak:
                                best_new_peak = cand_peak
                                best_pair = (d, r)
                if best_pair is None:
                    return (False, -1, -1, cur_max, cur_max)
                return (True, best_pair[0], best_pair[1], cur_max, best_new_peak if best_new_peak is not None else cur_max)

            # First move
            improved, d, r, old_peak, predicted = best_move_for_row()
            if not improved:
                continue
            # Choose donor physical column corresponding to donor's highest rank
            donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
            if donor_cols.numel() == 0:
                continue
            maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
            col_idx = donor_cols[maxr_idx]
            # Apply first move
            new_rank = int(logcnt[ri, r].item())
            phy2log[ri, col_idx] = r
            rank[ri, col_idx] = new_rank
            logcnt[ri, d] -= 1
            logcnt[ri, r] += 1

            # Check improvement; if shallow, try one more move
            avg_after = weight[ri] / logcnt[ri].to(weight.dtype)
            new_peak = float(avg_after.max().item())
            if new_peak > 0.9 * old_peak:
                improved2, d2, r2, old_peak2, predicted2 = best_move_for_row()
                if improved2:
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