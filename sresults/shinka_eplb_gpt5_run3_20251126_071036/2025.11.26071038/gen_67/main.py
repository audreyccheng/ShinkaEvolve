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


class PermOps:
    @staticmethod
    def inverse(perm: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse permutation row-wise.

        perm: [L, N], a permutation for each row.
        returns inv_perm where inv_perm[g, perm[g, i]] = i
        """
        L, N = perm.shape
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(N, dtype=torch.int64, device=perm.device).expand(L, -1))
        return inv


def _rank_from_packidx_rowwise(pack_idx_row: torch.Tensor, num_packs: int) -> torch.Tensor:
    """
    Deterministic ranks within each pack by ascending original item id.
    pack_idx_row: [N]
    returns rank_row: [N]
    """
    N = pack_idx_row.numel()
    device = pack_idx_row.device
    dtype_i64 = torch.int64
    rank_row = torch.empty(N, dtype=dtype_i64, device=device)
    for p in range(num_packs):
        ids = torch.nonzero(pack_idx_row == p, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        ids_sorted = torch.sort(ids).values
        rank_row[ids_sorted] = torch.arange(ids_sorted.numel(), dtype=dtype_i64, device=device)
    return rank_row


def _kcandidate_refine_row(weights: torch.Tensor,
                           pack_idx: torch.Tensor,
                           num_packs: int,
                           k: int = 2,
                           max_swaps: int = 1,
                           adaptive_second: bool = False,
                           consider_second_light: bool = False) -> torch.Tensor:
    """
    Bounded k-candidate refinement on a single row.
    - Evaluate up to k x k pair swaps between the heaviest pack and one or two light packs.
    - When consider_second_light=False: minimize |delta - 2*(wi - wj)| for the lightest pack.
    - When consider_second_light=True: pick the single swap that minimizes the predicted new global peak,
      considering both the lightest and second-lightest packs. Apply only if it strictly improves.
    """
    if max_swaps <= 0 or num_packs <= 1:
        return pack_idx

    device = weights.device
    pack_w = torch.zeros(num_packs, dtype=weights.dtype, device=device)
    pack_w.scatter_add_(0, pack_idx, weights)

    swaps_done = 0
    added_adaptive = False
    while swaps_done < max_swaps or (adaptive_second and not added_adaptive):
        h = int(torch.argmax(pack_w).item())

        # Determine light candidates
        order = torch.argsort(pack_w, descending=False)
        l0 = int(order[0].item())
        light_candidates = [l0]
        if consider_second_light and order.numel() > 1:
            l1 = int(order[1].item())
            if l1 != l0 and l1 != h:
                light_candidates.append(l1)

        # All equal or only heaviest exists
        if len(light_candidates) == 0 or (len(light_candidates) == 1 and light_candidates[0] == h):
            break

        # Precompute heavy indices/weights
        heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
        if heavy_idx.numel() == 0:
            break
        hw_all = weights[heavy_idx]
        k_h = min(k, hw_all.numel())
        if k_h <= 0:
            break
        topk_hw, topk_pos_h = torch.topk(hw_all, k=k_h, largest=True)

        cur_peak = float(pack_w.max().item())

        # Best candidate across light packs
        best = None  # (score_peak, hi, lj, l_sel, wi, wj, aux_metric)
        for l in light_candidates:
            if l == h:
                continue
            light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
            if light_idx.numel() == 0:
                continue
            lw_all = weights[light_idx]
            k_l = min(k, lw_all.numel())
            if k_l <= 0:
                continue
            bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=k_l, largest=True)
            bottomk_lw = -bottomk_lw_vals

            delta_hl = float((pack_w[h] - pack_w[l]).item())
            if delta_hl <= 1e-12:
                continue

            # Evaluate all pairings
            diff = topk_hw.unsqueeze(1) - bottomk_lw.unsqueeze(0)  # [k_h, k_l]

            if consider_second_light:
                # Choose by minimizing predicted new global peak. Tie-break by new delta.
                # New pack loads if we swap a pair (wi, wj):
                # new_h = pack_w[h] - wi + wj; new_l = pack_w[l] - wj + wi; peak = max(other_max, new_h, new_l)
                # First compute nearest to delta/2 to get a strong candidate cheaply, but still evaluate best via argmin peak.
                new_delta = (delta_hl - 2.0 * diff).abs()
                flat_idx = int(torch.argmin(new_delta).item())
                ih = flat_idx // k_l
                jl = flat_idx % k_l
                wi = float(topk_hw[ih].item())
                wj = float(bottomk_lw[jl].item())

                # Compute other packs' max excluding h and l
                if num_packs > 2:
                    mask = torch.ones(num_packs, dtype=torch.bool, device=pack_w.device)
                    mask[h] = False
                    mask[l] = False
                    other_max = float(pack_w[mask].max().item()) if mask.any() else float('-inf')
                else:
                    other_max = float('-inf')
                new_h = float(pack_w[h].item()) - wi + wj
                new_l = float(pack_w[l].item()) - wj + wi
                cand_peak = max(other_max, new_h, new_l)
                cand_aux = float(new_delta[ih, jl].item())

                if best is None or cand_peak < best[0] - 1e-12 or (abs(cand_peak - best[0]) <= 1e-12 and cand_aux < best[6]):
                    hi = heavy_idx[topk_pos_h[ih]]
                    lj = light_idx[bottomk_pos_l[jl]]
                    best = (cand_peak, hi, lj, l, wi, wj, cand_aux)
            else:
                # Original delta-minimization on the single lightest pack
                new_delta = (delta_hl - 2.0 * diff).abs()
                flat_idx = int(torch.argmin(new_delta).item())
                ih = flat_idx // k_l
                jl = flat_idx % k_l
                wi = float(topk_hw[ih].item())
                wj = float(bottomk_lw[jl].item())
                cand_aux = float(new_delta[ih, jl].item())
                if best is None or cand_aux < best[6] - 0.0:
                    hi = heavy_idx[topk_pos_h[ih]]
                    lj = light_idx[bottomk_pos_l[jl]]
                    best = (cur_peak, hi, lj, l, wi, wj, cand_aux)

        if best is None:
            break

        cand_peak, hi, lj, l_sel, wi, wj, cand_aux = best
        # Strict improvement guard:
        if consider_second_light:
            if cand_peak + 1e-12 >= cur_peak:
                break
        else:
            # Compare against imbalance between h and chosen l
            delta_sel = float((pack_w[h] - pack_w[l_sel]).item())
            if cand_aux + 1e-12 >= delta_sel:
                break

        # Commit swap
        pack_idx[hi] = l_sel
        pack_idx[lj] = h
        pack_w[h] = pack_w[h] - wi + wj
        pack_w[l_sel] = pack_w[l_sel] - wj + wi

        swaps_done += 1
        if adaptive_second and not added_adaptive:
            if consider_second_light:
                # Allow one extra attempt only if improvement is shallow (<20%)
                new_peak_now = float(pack_w.max().item())
                if new_peak_now > 0.8 * cur_peak:
                    added_adaptive = True
            else:
                # Use delta-based ratio
                improve_ratio = 1.0 - (cand_aux / max(delta_sel, 1e-12))
                if improve_ratio < 0.20:
                    added_adaptive = True

    return pack_idx


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     refine_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Capacity-constrained greedy with bounded k-candidate refinement.
    Parameters:
        weight: [X, n]
        num_packs: M
        refine_steps: number of refinement swaps per row (small integer)
    Returns:
        pack_index: [X, n]
        rank_in_pack: [X, n]
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    capacity = num_groups // num_packs

    if capacity == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Work on CPU for light control-flow loops (keeps speed)
    w_cpu = weight.float().cpu()
    pack_index = torch.full((num_layers, num_groups), -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, -1)

    for i in range(num_layers):
        w = w_cpu[i]
        order = torch.argsort(w, descending=True)
        load = [0.0] * num_packs
        counts = [0] * num_packs
        pidx = torch.empty(num_groups, dtype=torch.int64)

        for g in order.tolist():
            # choose the lightest pack with remaining capacity
            best_p = None
            best_load = None
            for p in range(num_packs):
                if counts[p] < capacity:
                    pl = load[p]
                    if best_load is None or pl < best_load:
                        best_p = p
                        best_load = pl
            pidx[g] = best_p
            load[best_p] += float(w[g].item())
            counts[best_p] += 1

        # bounded k=2 candidate refinement
        pidx = _kcandidate_refine_row(w, pidx, num_packs, k=2, max_swaps=int(refine_steps), adaptive_second=False)

        # ranks deterministic
        rnk = _rank_from_packidx_rowwise(pidx, num_packs)
        pack_index[i] = pidx
        rank_in_pack[i] = rnk

    # Move back to original device if needed
    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


def _balanced_packing_diverse(weights: torch.Tensor,
                              labels: torch.Tensor,
                              num_packs: int,
                              refine_steps: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same as balanced_packing but with diversity-aware tie-break under near-ties:
      - If candidate packs' loads are within eps, prefer the pack with fewer same-label items.

    weights: [X, n] (CPU or CUDA)
    labels:  [X, n] int64, label per item (e.g., logical expert id)
    returns pack_index, rank_in_pack on same device as inputs
    """
    L, N = weights.shape
    assert N % num_packs == 0
    capacity = N // num_packs

    if capacity == 1:
        pack_index = torch.arange(N, dtype=torch.int64, device=weights.device).expand(weights.shape)
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    w_cpu = weights.float().cpu()
    lab_cpu = labels.long().cpu()
    pack_index = torch.full((L, N), -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, -1)

    for li in range(L):
        w = w_cpu[li]
        labs = lab_cpu[li]
        order = torch.argsort(w, descending=True)
        # per-pack loads and counts
        load = [0.0] * num_packs
        counts = [0] * num_packs
        # label_counter per pack: dict[label] -> cnt
        label_cnt = [dict() for _ in range(num_packs)]
        pidx = torch.empty(N, dtype=torch.int64)

        mean_w = float(torch.mean(w).item()) if N > 0 else 0.0
        eps = 1e-6 * max(mean_w, 1e-12)
        lam = 1e-8 * max(mean_w, 1e-12)  # negligible penalty scale

        for g in order.tolist():
            # candidate packs with remaining capacity
            cand = [p for p in range(num_packs) if counts[p] < capacity]
            # baseline minimal load
            min_load = min(load[p] for p in cand)
            # compute effective loads with diversity penalty only for near-tied packs
            eff_load = []
            for p in cand:
                ld = load[p]
                if ld - min_load <= eps:
                    same = label_cnt[p].get(int(labs[g].item()), 0)
                    eff_load.append(ld + lam * same)
                else:
                    eff_load.append(ld)
            # pick argmin
            best_idx = int(torch.argmin(torch.tensor(eff_load)).item())
            best_pack = cand[best_idx]

            # assign
            pidx[g] = best_pack
            load[best_pack] += float(w[g].item())
            counts[best_pack] += 1
            lab = int(labs[g].item())
            label_cnt[best_pack][lab] = label_cnt[best_pack].get(lab, 0) + 1

        # refinement with k=2, consider second-lightest pack and adaptive second swap
        pidx = _kcandidate_refine_row(
            w,
            pidx,
            num_packs,
            k=2,
            max_swaps=int(refine_steps),
            adaptive_second=True,
            consider_second_light=True,
        )

        rnk = _rank_from_packidx_rowwise(pidx, num_packs)
        pack_index[li] = pidx
        rank_in_pack[li] = rnk

    return pack_index.to(weights.device), rank_in_pack.to(weights.device)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, minimizing max per-replica load.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_extra = num_phy - num_log
    assert num_extra >= 0
    device = weight.device
    dtype_i64 = torch.int64
    dtype_f = weight.dtype

    # base one per logical
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    base = torch.arange(num_log, dtype=dtype_i64, device=device).unsqueeze(0).expand(n, -1)
    phy2log[:, :num_log] = base
    rank[:, :num_log] = 0
    logcnt = torch.ones((n, num_log), dtype=dtype_i64, device=device)

    if num_extra == 0:
        return phy2log, rank, logcnt

    ar = torch.arange(n, dtype=dtype_i64, device=device)

    # Tail size based on CV of weights (clamped to [0.05, 0.15] fraction)
    mean = weight.mean(dim=-1).clamp_min(1e-12)
    std = weight.std(dim=-1)
    cv = (std / mean).mean().item()
    tail_frac = min(0.15, max(0.05, cv / 2.0))
    tail = max(1, int(round(num_extra * tail_frac)))
    bulk = num_extra - tail
    col = num_log

    # Bulk: D'Hondt (benefit = w / r)
    for _ in range(max(0, bulk)):
        benefit = weight / logcnt.to(dtype_f)
        best = benefit.argmax(dim=-1)
        phy2log[:, col] = best
        rank[:, col] = logcnt[ar, best]
        logcnt[ar, best] += 1
        col += 1

    # Tail: per-step per-row A/B between Sainte-Laguë and Huntington–Hill
    # Choose the one that yields smaller predicted new peak
    for _ in range(max(0, tail)):
        r_f = logcnt.to(dtype_f)
        # candidates under S and H
        benef_S = weight / (2.0 * r_f - 1.0)
        benef_H = weight / torch.sqrt(r_f * (r_f + 1.0))
        idx_S = benef_S.argmax(dim=-1)
        idx_H = benef_H.argmax(dim=-1)

        # compute current top-2 averages
        avg = weight / r_f
        top2_vals, _ = torch.topk(avg, k=min(2, num_log), dim=-1)
        if top2_vals.shape[-1] == 1:
            second = top2_vals[:, 0]
        else:
            second = top2_vals[:, 1]

        # predicted peaks
        newS = weight[ar, idx_S] / (r_f[ar, idx_S] + 1.0)
        newH = weight[ar, idx_H] / (r_f[ar, idx_H] + 1.0)
        peakS = torch.maximum(second, newS)
        peakH = torch.maximum(second, newH)
        use_S = peakS <= peakH
        best_idx = torch.where(use_S, idx_S, idx_H)

        phy2log[:, col] = best_idx
        rank[:, col] = logcnt[ar, best_idx]
        logcnt[ar, best_idx] += 1
        col += 1

    # Strengthened one-move fix-up per row with donor-specific baseline and second-order tie-break:
    # donors = top-2 by avg; receivers = bottom-2 by avg; evaluate all pairs and choose the single best
    # move that strictly reduces the global max. Tie-break by the predicted new second-highest.
    if num_log > 1 and num_extra > 0:
        r_f = logcnt.to(dtype_f)
        avg = weight / r_f
        kd = min(2, num_log)
        kr = min(2, num_log)
        top_vals, top_idx = torch.topk(avg, k=kd, dim=-1, largest=True)
        cur_max = top_vals[:, 0]
        second = top_vals[:, 1] if kd > 1 else top_vals[:, 0]
        bot_vals, bot_idx = torch.topk(avg, k=kr, dim=-1, largest=False)
        argmax_idx = avg.argmax(dim=-1, keepdim=True)  # [n,1]

        donors = top_idx  # [n, kd]
        receivers = bot_idx  # [n, kr]

        # counts and weights for donors/receivers
        cd = logcnt.gather(1, donors).to(dtype_f)  # [n, kd]
        cr = logcnt.gather(1, receivers).to(dtype_f)  # [n, kr]
        wd = weight.gather(1, donors).to(dtype_f)
        wr = weight.gather(1, receivers).to(dtype_f)

        # candidate new averages
        new_d = wd / (cd - 1.0).clamp_min(1.0)  # [n, kd] (masked later when cd<=1)
        new_r = wr / (cr + 1.0)                 # [n, kr]

        # donor-specific baseline: if donor is current argmax -> second, else cur_max
        is_argmax = (donors == argmax_idx)  # [n, kd]
        base_per_d = torch.where(is_argmax, second.unsqueeze(1), cur_max.unsqueeze(1))  # [n, kd]

        # Broadcast to pairs
        kd_sz = donors.shape[1]
        kr_sz = receivers.shape[1]
        base_b = base_per_d.unsqueeze(2).expand(-1, kd_sz, kr_sz)
        new_d_b = new_d.unsqueeze(2).expand(-1, kd_sz, kr_sz)
        new_r_b = new_r.unsqueeze(1).expand(-1, kd_sz, kr_sz)

        # Validity mask: donor count > 1 and donor != receiver
        valid_cd = (cd > 1.0).unsqueeze(2).expand(-1, kd_sz, kr_sz)
        d_idx_b = donors.unsqueeze(2).expand(-1, kd_sz, kr_sz)
        r_idx_b = receivers.unsqueeze(1).expand(-1, kd_sz, kr_sz)
        valid = valid_cd & (d_idx_b != r_idx_b)

        # Predicted peak and predicted second-highest among {base, new_d, new_r}
        vals3 = torch.stack([base_b, new_d_b, new_r_b], dim=-1)  # [n, kd, kr, 3]
        top2 = torch.topk(vals3, k=2, dim=-1, largest=True).values  # [..., 2]
        pred_peak = top2[..., 0]
        pred_second = top2[..., 1]

        inf = torch.tensor(float('inf'), dtype=pred_peak.dtype, device=pred_peak.device)
        pred_peak = torch.where(valid, pred_peak, inf)
        pred_second = torch.where(valid, pred_second, inf)

        # Choose lexicographically: minimize peak, then second
        flat_peak = pred_peak.view(n, -1)
        min_peak_vals, _ = torch.min(flat_peak, dim=-1, keepdim=True)  # [n,1]
        is_min_peak = (flat_peak <= (min_peak_vals + 1e-12))
        flat_second = pred_second.view(n, -1)
        # Mask others by +inf to find min second among min-peak ties
        masked_second = torch.where(is_min_peak, flat_second, inf)
        best_flat = torch.argmin(masked_second, dim=-1)  # [n]
        best_vals = torch.gather(flat_peak, 1, best_flat.view(-1, 1)).squeeze(1)

        improve = best_vals + 1e-12 < cur_max
        rows = torch.nonzero(improve, as_tuple=False).squeeze(1)

        if rows.numel() > 0:
            for ri in rows.tolist():
                bf = int(best_flat[ri].item())
                di = bf // kr_sz
                rj = bf % kr_sz
                d = int(donors[ri, di].item())
                r = int(receivers[ri, rj].item())
                # select a donor physical column with highest rank
                donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() == 0:
                    continue
                maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
                col_idx = donor_cols[maxr_idx]
                # move it to receiver with new rank
                new_rank = int(logcnt[ri, r].item())
                phy2log[ri, col_idx] = r
                rank[ri, col_idx] = new_rank
                # update counts
                logcnt[ri, d] -= 1
                logcnt[ri, r] += 1

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
    L, num_logical_experts = weight.shape
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

    # Step 1: pack groups to nodes using balanced_packing (k=2 best-swap, 1 step)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes, refine_steps=1)

    # Compute permutation logical -> meta-logical (within-node contiguous layout)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (within nodes) with diversity-aware tie-breaker
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)  # per-replica avg loads
    # Labels for diversity are the meta-logical ids per replica
    labels_per_phy = phy2mlog
    pack_index, rank_in_pack = _balanced_packing_diverse(tokens_per_phy, labels_per_phy, num_gpus // num_nodes, refine_steps=2)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Map back to global logical ids
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [L*num_nodes, num_physical_experts/num_nodes]
    pphy2mlog = (pphy2mlog.view(L, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes, device=weight.device).view(1, -1, 1)
                 ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(L, -1)
    logcnt = mlogcnt.view(L, -1).gather(-1, log2mlog)
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