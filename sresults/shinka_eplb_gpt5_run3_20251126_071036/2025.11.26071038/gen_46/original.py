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


def _kcandidate_refine_row(
    weights: torch.Tensor,
    pack_idx: torch.Tensor,
    num_packs: int,
    k: int = 2,
    max_swaps: int = 1,
    adaptive_second: bool = False,
    consider_second_light: bool = False,
    allow_two_two: bool = False,
) -> torch.Tensor:
    """
    Bounded refinement on a single row.

    - Evaluate up to k x k best 1x1 swaps between the heaviest pack and
      the lightest pack; optionally include the second-lightest pack too.
    - Optionally evaluate a single 2x2 exchange (top-2 from heaviest vs bottom-2 from lightest),
      and apply it only if it strictly reduces the max load more than the best 1x1 swap.
    - Strict improvement guards; stop early if no improvement.

    weights: [N] float (CPU)
    pack_idx: [N] int64 (CPU)
    """
    if max_swaps <= 0 or num_packs <= 1:
        return pack_idx

    device = weights.device
    pack_w = torch.zeros(num_packs, dtype=weights.dtype, device=device)
    pack_w.scatter_add_(0, pack_idx, weights)

    swaps_done = 0
    added_adaptive = False

    while swaps_done < max_swaps or (adaptive_second and not added_adaptive):
        # Identify heaviest; for candidates on light side consider 1 or 2 lightest
        h = int(torch.argmax(pack_w).item())
        # Sort ascending loads to get lightest and second-lightest
        # Keeping it cheap: argsort for num_packs is tiny
        light_order = torch.argsort(pack_w, descending=False)
        l0 = int(light_order[0].item())
        light_candidates = [l0]
        if consider_second_light and num_packs >= 2:
            l1 = int(light_order[1].item())
            if l1 != l0 and l1 != h:
                light_candidates.append(l1)

        if h in light_candidates and len(light_candidates) == 1:
            # all packs equal
            break

        # Base delta against best light candidate at time of evaluation
        best_choice = None  # (new_delta, hi_idx, lj_idx, chosen_light)
        base_delta = None

        # precompute heavy indices and weights once
        heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
        if heavy_idx.numel() == 0:
            break
        hw_all = weights[heavy_idx]
        k_h = min(k, hw_all.numel())
        topk_hw_vals, topk_pos_h = torch.topk(hw_all, k=k_h, largest=True)

        # Evaluate 1x1 swaps for l0 and (optionally) l1
        for l in light_candidates:
            if l == h:
                continue
            light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
            if light_idx.numel() == 0:
                continue
            lw_all = weights[light_idx]
            k_l = min(k, lw_all.numel())
            # bottom-k via -topk
            bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=k_l, largest=True)
            bottomk_lw = -bottomk_lw_vals

            # current delta
            delta = float((pack_w[h] - pack_w[l]).item())
            if delta <= 1e-12:
                continue

            # all pair deltas
            diff = topk_hw_vals.unsqueeze(1) - bottomk_lw.unsqueeze(0)  # [k_h, k_l]
            cand_new_delta = (delta - 2.0 * diff).abs()
            flat_idx = int(torch.argmin(cand_new_delta).item())
            ih = flat_idx // k_l
            jl = flat_idx % k_l
            best_nd = float(cand_new_delta[ih, jl].item())
            if base_delta is None or best_nd < base_delta - 0.0:
                base_delta = best_nd
                best_choice = (
                    best_nd,
                    heavy_idx[topk_pos_h[ih]],
                    light_idx[bottomk_pos_l[jl]],
                    l,
                )

        # Optionally evaluate a single 2x2 exchange against the lightest pack l0 only
        two_two_applied = False
        two_two_candidate = None
        if allow_two_two and hw_all.numel() >= 2:
            l = l0
            if l != h:
                light_idx0 = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
                if light_idx0.numel() >= 2:
                    # choose top-2 heavy and bottom-2 light
                    kh2 = min(2, hw_all.numel())
                    kl2 = min(2, light_idx0.numel())
                    t_h_vals, t_h_pos = torch.topk(hw_all, k=kh2, largest=True)
                    lw0 = weights[light_idx0]
                    b_l_vals, b_l_pos = torch.topk(-lw0, k=kl2, largest=True)
                    b_l_vals = -b_l_vals

                    # total weight difference exchanged
                    delta0 = float((pack_w[h] - pack_w[l]).item())
                    sum_h = float(t_h_vals.sum().item())
                    sum_l = float(b_l_vals.sum().item())
                    new_delta_22 = abs(delta0 - 2.0 * (sum_h - sum_l))
                    two_two_candidate = (new_delta_22, t_h_pos, b_l_pos, l)

        # Decide: apply 2x2 only if strictly better than 1x1 best
        applied = False
        if two_two_candidate is not None:
            nd22, hpos22, lpos22, l22 = two_two_candidate
            if base_delta is None or nd22 + 1e-12 < base_delta:
                # commit 2x2
                # map local positions to global indices
                hi1 = heavy_idx[hpos22[0]]
                lj1 = torch.nonzero(pack_idx == l22, as_tuple=False).squeeze(1)[lpos22[0]]
                if hpos22.numel() >= 2 and lpos22.numel() >= 2:
                    hi2 = heavy_idx[hpos22[1]]
                    lj2 = torch.nonzero(pack_idx == l22, as_tuple=False).squeeze(1)[lpos22[1]]
                else:
                    hi2 = None
                    lj2 = None

                # perform swaps
                wi1 = float(weights[hi1].item())
                wj1 = float(weights[lj1].item())
                pack_idx[hi1] = l22
                pack_idx[lj1] = h
                pack_w[h] = pack_w[h] - wi1 + wj1
                pack_w[l22] = pack_w[l22] - wj1 + wi1

                if hi2 is not None and lj2 is not None:
                    wi2 = float(weights[hi2].item())
                    wj2 = float(weights[lj2].item())
                    pack_idx[hi2] = l22
                    pack_idx[lj2] = h
                    pack_w[h] = pack_w[h] - wi2 + wj2
                    pack_w[l22] = pack_w[l22] - wj2 + wi2

                swaps_done += 1
                applied = True
                two_two_applied = True

        if not applied and best_choice is not None:
            best_nd, hi, lj, lsel = best_choice
            # verify strict improvement w.r.t. the chosen light pack
            delta_sel = float((pack_w[h] - pack_w[lsel]).item())
            if best_nd + 1e-12 < delta_sel:
                wi = float(weights[hi].item())
                wj = float(weights[lj].item())
                pack_idx[hi] = lsel
                pack_idx[lj] = h
                pack_w[h] = pack_w[h] - wi + wj
                pack_w[lsel] = pack_w[lsel] - wj + wi
                swaps_done += 1

                # allow an extra attempt if the improvement is shallow (<20%)
                if adaptive_second and not added_adaptive:
                    improve_ratio = 1.0 - (best_nd / max(delta_sel, 1e-12))
                    if improve_ratio < 0.20:
                        added_adaptive = True
                continue
            else:
                break
        elif applied and two_two_applied:
            # Completed a 2x2 swap; optionally allow another if adaptive is enabled
            if not adaptive_second:
                continue
            else:
                # If improvement shallow, allow one extra attempt
                # Recompute current h/l delta to estimate improvement ratio
                h2 = int(torch.argmax(pack_w).item())
                l2 = int(torch.argmin(pack_w).item())
                delta2 = float((pack_w[h2] - pack_w[l2]).item())
                # We don't have old delta easily; just allow a single extra step when adaptive is on.
                if not added_adaptive:
                    added_adaptive = True
                continue
        else:
            # no improving move
            break

    return pack_idx


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     refine_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Capacity-constrained greedy with bounded refinement.
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

    # Work on CPU for control-flow loops
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

        # Bounded refinement (heaviest vs lightest; no 2x2; second-light disabled)
        pidx = _kcandidate_refine_row(
            w,
            pidx,
            num_packs,
            k=2,
            max_swaps=int(refine_steps),
            adaptive_second=False,
            consider_second_light=False,
            allow_two_two=False,
        )

        # ranks deterministic
        rnk = _rank_from_packidx_rowwise(pidx, num_packs)
        pack_index[i] = pidx
        rank_in_pack[i] = rnk

    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


def _balanced_packing_diverse(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int,
    refine_steps_default: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same as balanced_packing but with diversity-aware tie-break under near-ties:
      - If candidate packs' loads are within eps, prefer the pack with fewer same-label items.
    Additionally applies adaptive refinement depth per row based on imbalance.
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
        # label_counter per pack
        label_cnt = [dict() for _ in range(num_packs)]
        pidx = torch.empty(N, dtype=torch.int64)

        mean_w = float(torch.mean(w).item()) if N > 0 else 0.0
        eps = 1e-6 * max(mean_w, 1e-12)
        lam = 1e-8 * max(mean_w, 1e-12)  # tiny penalty

        for g in order.tolist():
            cand = [p for p in range(num_packs) if counts[p] < capacity]
            min_load = min(load[p] for p in cand)
            eff_load = []
            for p in cand:
                ld = load[p]
                if ld - min_load <= eps:
                    same = label_cnt[p].get(int(labs[g].item()), 0)
                    eff_load.append(ld + lam * same)
                else:
                    eff_load.append(ld)
            best_idx = int(torch.argmin(torch.tensor(eff_load)).item())
            best_pack = cand[best_idx]

            pidx[g] = best_pack
            load[best_pack] += float(w[g].item())
            counts[best_pack] += 1
            lab = int(labs[g].item())
            label_cnt[best_pack][lab] = label_cnt[best_pack].get(lab, 0) + 1

        # Compute imbalance ratio to adapt depth
        pack_w = torch.zeros(num_packs, dtype=w.dtype)
        pack_w.scatter_add_(0, pidx, w)
        delta = float((pack_w.max() - pack_w.min()).item())
        mean_ld = float(pack_w.mean().item())
        ratio = delta / max(mean_ld, 1e-12)
        steps = 3 if ratio > 0.12 else refine_steps_default

        # refinement with expanded light candidates and optional 2x2
        pidx = _kcandidate_refine_row(
            w,
            pidx,
            num_packs,
            k=2,
            max_swaps=int(steps),
            adaptive_second=True,
            consider_second_light=True,
            allow_two_two=True,
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

    # Bulk: D'Hondt (benefit = w / r); choose tail via tiny A/B/C around 10%
    tail0 = max(1, int(round(0.10 * num_extra)))
    tail_candidates = sorted(set([max(1, tail0 - 1), tail0, min(num_extra, tail0 + 1)]))
    best_tail = tail_candidates[0]
    best_pred_peak = None

    # Simulate tail options with bulk as D'Hondt
    for tl in tail_candidates:
        bulk = max(0, num_extra - tl)
        cnt_sim = torch.ones_like(logcnt)
        # Apply bulk D'Hondt on simulated counts
        if bulk > 0:
            for _ in range(bulk):
                benefit = weight / cnt_sim.to(dtype_f)
                best = benefit.argmax(dim=-1)
                cnt_sim[ar, best] += 1
        # Simulate tl steps of Sainte-Laguë
        if tl > 0:
            for _ in range(tl):
                denom = (2.0 * cnt_sim.to(dtype_f) - 1.0)
                benefit = weight / denom
                best = benefit.argmax(dim=-1)
                cnt_sim[ar, best] += 1
        # Predicted peak average per row
        avg_sim = weight / cnt_sim.to(dtype_f)
        pred_peak = avg_sim.max(dim=-1).values.mean().item()  # average across rows
        if best_pred_peak is None or pred_peak < best_pred_peak - 0.0:
            best_pred_peak = pred_peak
            best_tail = tl

    # Now actually allocate extras: first bulk D'Hondt then best-tail Sainte-Laguë
    bulk = max(0, num_extra - best_tail)
    col = num_log
    if bulk > 0:
        for _ in range(bulk):
            benefit = weight / logcnt.to(dtype_f)
            best = benefit.argmax(dim=-1)
            phy2log[:, col] = best
            rank[:, col] = logcnt[ar, best]
            logcnt[ar, best] += 1
            col += 1
    if best_tail > 0:
        for _ in range(best_tail):
            denom = (2.0 * logcnt.to(dtype_f) - 1.0)
            benefit = weight / denom
            best = benefit.argmax(dim=-1)
            phy2log[:, col] = best
            rank[:, col] = logcnt[ar, best]
            logcnt[ar, best] += 1
            col += 1

    # Strengthened replication fix-up per row, capped at 2 moves (conditional second)
    if num_log > 1 and num_extra > 0:
        def single_best_move(ri: int) -> tuple[bool, int, int]:
            """Return (improves, donor_id, receiver_id) for row ri."""
            r_f = logcnt[ri].to(dtype_f)
            avg = weight[ri] / r_f
            kdon = min(2, num_log)
            krec = min(2, num_log)
            top_vals, top_idx = torch.topk(avg, k=kdon, largest=True)
            bot_vals, bot_idx = torch.topk(avg, k=krec, largest=False)
            cur_max = float(top_vals[0].item())
            second = float((top_vals[1].item() if kdon > 1 else top_vals[0].item()))
            best_pair = None
            best_new_peak = None

            # Evaluate all donor/receiver pairs
            for di in range(kdon):
                d = int(top_idx[di].item())
                cd = int(logcnt[ri, d].item())
                if cd <= 1:
                    continue
                for rj in range(krec):
                    r = int(bot_idx[rj].item())
                    if d == r:
                        continue
                    cr = int(logcnt[ri, r].item())
                    new_d = float(weight[ri, d].item()) / float(cd - 1)
                    new_r = float(weight[ri, r].item()) / float(cr + 1)
                    candidate_peak = max(second, new_d, new_r)
                    if candidate_peak + 1e-12 < cur_max:
                        if best_new_peak is None or candidate_peak < best_new_peak:
                            best_new_peak = candidate_peak
                            best_pair = (d, r)
            return (best_pair is not None, *(best_pair if best_pair is not None else (-1, -1)))

        rows = torch.arange(n, dtype=torch.int64, device=device).tolist()
        for ri in rows:
            # First move
            improved, d, r = single_best_move(ri)
            if not improved:
                continue
            # choose donor physical column with highest rank
            donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
            if donor_cols.numel() == 0:
                continue
            maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
            col_idx = donor_cols[maxr_idx]
            old_ranks = (logcnt[ri, d].item(), logcnt[ri, r].item())  # before move

            # Apply first move
            new_rank = int(logcnt[ri, r].item())
            phy2log[ri, col_idx] = r
            rank[ri, col_idx] = new_rank
            logcnt[ri, d] -= 1
            logcnt[ri, r] += 1

            # Check improvement depth; if shallow (<10%), try a second move
            # Compute old peak before (need recompute using old counts); approximate using updated counts and re-evaluate
            # We recompute peaks precisely now
            avg_after = weight[ri] / logcnt[ri].to(dtype_f)
            new_peak = float(avg_after.max().item())

            # Reconstruct old peak approximately:
            # To get old peak, undo the move virtually:
            old_cd, old_cr = old_ranks
            old_avg_d = float(weight[ri, d].item()) / float(old_cd)
            old_avg_r = float(weight[ri, r].item()) / float(old_cr)
            # Max old peak is at least max of these two, but compute full
            # For accuracy, rebuild old counts vector cheaply
            cnt_tmp = logcnt[ri].clone()
            cnt_tmp[d] += 1
            cnt_tmp[r] -= 1
            old_peak = float((weight[ri] / cnt_tmp.to(dtype_f)).max().item())

            if new_peak > 0.9 * old_peak:
                improved2, d2, r2 = single_best_move(ri)
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

    # Step 1: pack groups to nodes using balanced_packing (bounded refine, 1 step)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes, refine_steps=1)

    # Compute permutation logical -> meta-logical (within-node contiguous layout)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (within nodes) with diversity-aware tie-breaker and adaptive refinement
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)  # per-replica avg loads
    labels_per_phy = phy2mlog
    pack_index, rank_in_pack = _balanced_packing_diverse(tokens_per_phy, labels_per_phy, num_gpus // num_nodes, refine_steps_default=2)
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
        # hierarchical policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global policy
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
    # Scatter physical ids into per-logical replica slots
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