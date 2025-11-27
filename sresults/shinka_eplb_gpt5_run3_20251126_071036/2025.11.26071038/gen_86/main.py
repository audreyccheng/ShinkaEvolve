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
                     labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        labels: optional [X, n] int64 labels; when provided, near-ties on load are
                broken by preferring the pack where the label is least frequent.

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
    lab_cpu = labels.long().cpu() if labels is not None else None

    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        label_cnt = [dict() for _ in range(num_packs)] if lab_cpu is not None else None
        w = weight[i].cpu()
        mean_w = float(torch.mean(w).item()) if w.numel() > 0 else 0.0
        eps = 1e-6 * max(mean_w, 1e-12)

        for group in indices[i]:
            # Choose the lightest pack with remaining capacity; near-ties broken by label diversity
            best_pack = None
            best_load = float("inf")
            best_same = float("inf")
            lbl = int(lab_cpu[i, group].item()) if lab_cpu is not None else -1
            for p in range(num_packs):
                if pack_items[p] >= groups_per_pack:
                    continue
                pl = pack_weights[p]
                if pl + eps < best_load:
                    best_pack = p
                    best_load = pl
                    best_same = float("inf") if lab_cpu is None else label_cnt[p].get(lbl, 0)
                elif abs(pl - best_load) <= eps:
                    if lab_cpu is not None:
                        same = label_cnt[p].get(lbl, 0)
                        if same < best_same:
                            best_pack = p
                            best_load = pl
                            best_same = same
                    # else keep earlier pack deterministically
            pack = best_pack
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += float(w[group].item())
            pack_items[pack] += 1
            if lab_cpu is not None:
                label_cnt[pack][lbl] = label_cnt[pack].get(lbl, 0) + 1

    # Adaptive bounded refinement:
    # - If imbalance is high, allow up to 3 swaps; else 2.
    # - Consider both lightest and second-lightest packs for 1x1 swaps (k=2 candidates).
    # - Optionally apply one 2x2 exchange (top-2 heavy vs bottom-2 lightest) if it beats the best 1x1 improvement.
    if groups_per_pack > 1:
        for i in range(num_layers):
            packs = pack_index[i]  # [num_groups], CPU
            w = weight[i]  # CPU
            # Compute current pack loads once
            pack_w = torch.zeros(num_packs, dtype=w.dtype)
            pack_w.scatter_add_(0, packs, w)
            delta0 = float((pack_w.max() - pack_w.min()).item())
            mean_ld = float(pack_w.mean().item())
            max_swaps = 3 if mean_ld > 0 and (delta0 / max(mean_ld, 1e-12)) > 0.12 else 2
            for _ in range(max_swaps):
                # Identify heaviest and candidate lightest packs
                h = int(torch.argmax(pack_w).item())
                light_order = torch.argsort(pack_w, descending=False)
                l0 = int(light_order[0].item())
                light_candidates = [l0]
                if num_packs > 2:
                    l1 = int(light_order[1].item())
                    if l1 != l0 and l1 != h:
                        light_candidates.append(l1)

                # Guard equalized case
                if h in light_candidates and len(light_candidates) == 1:
                    break

                # Precompute heavy candidates (top-2)
                heavy_idx = torch.nonzero(packs == h, as_tuple=False).squeeze(1)
                if heavy_idx.numel() == 0:
                    break
                hw_all = w[heavy_idx]
                kh = min(2, hw_all.numel())
                if kh == 0:
                    break
                topk_hw_vals, topk_pos_h = torch.topk(hw_all, k=kh, largest=True)

                # Best 1x1 across light candidates
                best_11 = None  # (new_delta, hi_global, lj_global, light_pack)
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

                    # bottom-k from light pack
                    bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=kl, largest=True)
                    bottomk_lw = -bottomk_lw_vals

                    delta = float((pack_w[h] - pack_w[l]).item())
                    if delta <= 1e-12:
                        continue

                    diff = topk_hw_vals.unsqueeze(1) - bottomk_lw.unsqueeze(0)  # [kh, kl]
                    cand_new_delta = (delta - 2.0 * diff).abs()
                    flat = int(torch.argmin(cand_new_delta).item())
                    ih = flat // kl
                    jl = flat % kl
                    nd = float(cand_new_delta[ih, jl].item())
                    if (best_11 is None) or (nd < best_11[0] - 0.0):
                        hi_global = heavy_idx[topk_pos_h[ih]]
                        lj_global = light_idx[bottomk_pos_l[jl]]
                        best_11 = (nd, hi_global, lj_global, l)

                # Optional single 2x2 vs lightest pack only
                best_22 = None  # (new_delta_22, hi1, hi2, lj1, lj2, l0)
                l = l0
                if l != h:
                    light_idx0 = torch.nonzero(packs == l, as_tuple=False).squeeze(1)
                    if hw_all.numel() >= 2 and light_idx0.numel() >= 2:
                        t_h_vals, t_h_pos = torch.topk(hw_all, k=2, largest=True)
                        lw0 = w[light_idx0]
                        b_l_vals, b_l_pos = torch.topk(-lw0, k=2, largest=True)
                        b_l_vals = -b_l_vals
                        delta_l0 = float((pack_w[h] - pack_w[l]).item())
                        nd22 = abs(delta_l0 - 2.0 * float((t_h_vals.sum() - b_l_vals.sum()).item()))
                        hi1 = heavy_idx[t_h_pos[0]]
                        hi2 = heavy_idx[t_h_pos[1]]
                        lj1 = light_idx0[b_l_pos[0]]
                        lj2 = light_idx0[b_l_pos[1]]
                        best_22 = (nd22, hi1, hi2, lj1, lj2, l)

                # Decide and apply the best improving move
                improved = False
                if best_11 is not None:
                    nd11, hi, lj, lsel = best_11
                    delta_sel = float((pack_w[h] - pack_w[lsel]).item())
                    # Compare with 2x2 if available and better
                    use_22 = False
                    if best_22 is not None:
                        nd22, hi1, hi2, lj1, lj2, l22 = best_22
                        if nd22 + 1e-12 < nd11:
                            # verify strict improvement
                            if nd22 + 1e-12 < delta_sel:
                                # apply 2x2
                                wi1 = float(w[hi1].item()); wj1 = float(w[lj1].item())
                                packs[hi1] = l22; packs[lj1] = h
                                pack_w[h] = pack_w[h] - wi1 + wj1
                                pack_w[l22] = pack_w[l22] - wj1 + wi1
                                wi2 = float(w[hi2].item()); wj2 = float(w[lj2].item())
                                packs[hi2] = l22; packs[lj2] = h
                                pack_w[h] = pack_w[h] - wi2 + wj2
                                pack_w[l22] = pack_w[l22] - wj2 + wi2
                                # Update indices tensor
                                pack_index[i] = packs
                                # Reassign ranks within affected packs
                                for p in (h, l22):
                                    mask = pack_index[i] == p
                                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                                    if idx.numel() > 0:
                                        prev_rank = rank_in_pack[i, idx]
                                        order = torch.argsort(prev_rank)
                                        new_ranks = torch.arange(order.numel(), dtype=torch.int64)
                                        rank_in_pack[i, idx[order]] = new_ranks
                                improved = True
                                use_22 = True
                    if not improved:
                        # apply 1x1 if strictly improves
                        if nd11 + 1e-12 < delta_sel:
                            wi = float(w[hi].item()); wj = float(w[lj].item())
                            packs[hi] = lsel; packs[lj] = h
                            pack_w[h] = pack_w[h] - wi + wj
                            pack_w[lsel] = pack_w[lsel] - wj + wi
                            pack_index[i] = packs
                            for p in (h, lsel):
                                mask = pack_index[i] == p
                                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                                if idx.numel() > 0:
                                    prev_rank = rank_in_pack[i, idx]
                                    order = torch.argsort(prev_rank)
                                    new_ranks = torch.arange(order.numel(), dtype=torch.int64)
                                    rank_in_pack[i, idx[order]] = new_ranks
                            improved = True

                if not improved:
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

    # Hybrid allocation: D'Hondt for bulk, Sainte-Laguë for the last ~10% (at least 1)
    tail = max(1, (num_redundant + 9) // 10)
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

    # Tail phase: per-step peak-aware chooser among D'Hondt, Sainte-Laguë, and Huntington–Hill
    if tail > 0:
        for _ in range(tail):
            r_f = logcnt.to(weight.dtype)
            avg_cur = weight / r_f
            # second-best current average per row
            top2 = torch.topk(avg_cur, k=min(2, num_log), dim=-1).values
            second = top2[:, 1] if num_log >= 2 else top2[:, 0]

            # Candidate indices for D, S, H
            idx_D = (weight / r_f).max(dim=-1).indices
            idx_S = (weight / (2.0 * r_f - 1.0)).max(dim=-1).indices
            idx_H = (weight / torch.sqrt(r_f * (r_f + 1.0))).max(dim=-1).indices

            # Stack candidates [n, 3]
            cand_idx = torch.stack([idx_D, idx_S, idx_H], dim=-1)
            # Gather weights and counts for candidates
            w_cand = weight.gather(1, cand_idx)
            c_cand = r_f.gather(1, cand_idx)
            new_recv_avg = w_cand / (c_cand + 1.0)

            sec_exp = second.unsqueeze(1).expand_as(new_recv_avg)
            pred_peak = torch.maximum(sec_exp, new_recv_avg)  # [n, 3]

            # Lexicographic tie-break: pred_peak -> new_recv_avg -> receiver_count -> method order
            peak_min = pred_peak.min(dim=1).values.unsqueeze(1)
            mask = pred_peak == peak_min
            # minimize new_recv_avg among ties
            new_min = torch.where(mask, new_recv_avg, torch.full_like(new_recv_avg, float('inf')))
            new_min_vals = new_min.min(dim=1).values.unsqueeze(1)
            mask2 = mask & (new_recv_avg == new_min_vals)
            # minimize receiver count among ties
            cnt_pref = torch.where(mask2, c_cand, torch.full_like(c_cand, float('inf')))
            cnt_min = cnt_pref.min(dim=1).values.unsqueeze(1)
            mask3 = mask2 & (c_cand == cnt_min)
            # final: prefer earlier candidate order (D, S, H)
            # convert boolean mask3 to indices
            choice = mask3.float().argmax(dim=1)  # [n]
            best = cand_idx[torch.arange(n, device=device), choice]

            phy2log[:, col] = best
            rank[:, col] = logcnt[arangen, best]
            logcnt[arangen, best] += 1
            col += 1

    # Strengthened replication fix-up per row with conditional second move:
    # Try one donor->receiver move; if peak improvement is shallow (<10%),
    # recompute and attempt one more move that strictly improves.
    if num_log > 1 and num_redundant > 0:
        avg = weight / logcnt.to(weight.dtype)  # [n, num_log]
        kd = min(2, num_log)
        kr = min(2, num_log)
        top_vals, top_idx = torch.topk(avg, k=kd, dim=-1, largest=True)
        cur_max = top_vals[:, 0]
        second = top_vals[:, 1] if kd > 1 else top_vals[:, 0]
        _, bot_idx = torch.topk(avg, k=kr, dim=-1, largest=False)

        for ri in range(n):
            # Helper to find best improving move under current counts
            def best_move_for_row():
                best_new_peak_local = None
                best_pair_local = None
                donors = top_idx[ri].tolist()
                receivers = bot_idx[ri].tolist()
                sec_val = float(second[ri].item())
                cur_peak = float(cur_max[ri].item())
                for d in donors:
                    cd = int(logcnt[ri, d].item())
                    if cd <= 1:
                        continue
                    for r in receivers:
                        if d == r:
                            continue
                        cr = int(logcnt[ri, r].item())
                        new_d = float(weight[ri, d].item()) / float(cd - 1)
                        new_r = float(weight[ri, r].item()) / float(cr + 1)
                        candidate_peak = max(sec_val, new_d, new_r)
                        if candidate_peak + 1e-12 < cur_peak:
                            if best_new_peak_local is None or candidate_peak < best_new_peak_local:
                                best_new_peak_local = candidate_peak
                                best_pair_local = (d, r)
                return best_pair_local, best_new_peak_local

            # First move
            best_pair, best_new_peak = best_move_for_row()
            if best_pair is None:
                continue
            d, r = best_pair
            donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
            if donor_cols.numel() == 0:
                continue
            maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
            col_idx = donor_cols[maxr_idx]
            old_peak = float(cur_max[ri].item())

            # Apply first move
            new_rank = int(logcnt[ri, r].item())
            phy2log[ri, col_idx] = r
            rank[ri, col_idx] = new_rank
            logcnt[ri, d] -= 1
            logcnt[ri, r] += 1

            # Evaluate improvement; if shallow, attempt a second move
            avg_after = weight[ri] / logcnt[ri].to(weight.dtype)
            new_peak = float(avg_after.max().item())
            if new_peak > 0.9 * old_peak:
                # Recompute avg, donors/receivers under updated counts
                avg2 = weight[ri] / logcnt[ri].to(weight.dtype)
                top_vals2, top_idx2 = torch.topk(avg2, k=kd, largest=True)
                second2 = float((top_vals2[1].item() if kd > 1 else top_vals2[0].item()))
                # Temporarily update globals for helper
                top_idx[ri] = top_idx2
                second[ri] = second2
                cur_max[ri] = top_vals2[0]
                # Find and apply a second improving move
                best_pair2, best_new_peak2 = best_move_for_row()
                if best_pair2 is not None:
                    d2, r2 = best_pair2
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
    # Use diversity-aware tie-breaking by passing meta-logical labels (phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes,
                                                labels=phy2mlog)
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