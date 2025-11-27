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


def _argmax_tiebreak(primary: torch.Tensor,
                     secondary: torch.Tensor,
                     prefer_secondary_small: bool = True) -> torch.Tensor:
    """
    Row-wise argmax with deterministic tie-breaking:
      1) maximize primary
      2) among ties, minimize secondary if prefer_secondary_small else maximize secondary
      3) among ties, pick the smallest index

    primary: [L, N]
    secondary: [L, N]
    returns best_idx: [L] int64
    """
    L, N = primary.shape
    device = primary.device
    # Step 1: primary max
    max_primary = primary.max(dim=-1, keepdim=True).values
    cand1 = primary == max_primary

    # Step 2: secondary (min or max) among cand1
    if prefer_secondary_small:
        sec_vals = torch.where(cand1, secondary, torch.tensor(float('inf'), device=device, dtype=secondary.dtype))
        best_sec = sec_vals.min(dim=-1, keepdim=True).values
        cand2 = cand1 & (secondary == best_sec)
    else:
        sec_vals = torch.where(cand1, secondary, torch.tensor(float('-inf'), device=device, dtype=secondary.dtype))
        best_sec = sec_vals.max(dim=-1, keepdim=True).values
        cand2 = cand1 & (secondary == best_sec)

    # Step 3: smallest index among final candidates
    idxs = torch.arange(N, device=device, dtype=torch.int64).unsqueeze(0).expand(L, -1)
    idx_masked = torch.where(cand2, idxs, torch.full_like(idxs, fill_value=N + 1))
    best_idx = idx_masked.min(dim=-1).indices
    return best_idx


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

    # Greedy assignment to lightest pack with remaining capacity
    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (p for p in range(num_packs) if pack_items[p] < groups_per_pack),
                key=lambda p: pack_weights[p],
            )
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += float(weight[i, group].item())
            pack_items[pack] += 1

    # Bounded refinement per layer to reduce max imbalance:
    # - Candidates: receivers among the lightest 3 packs if available
    # - Choose the swap minimizing predicted global max after swap
    # - Optional 2x2 swap if strictly better than best 1x1
    # - Optional chained two-swap fallback if imbalance is pronounced
    if groups_per_pack > 1:
        max_swaps = 2  # keep small to preserve speed
        for i in range(num_layers):
            w = weight[i]  # CPU
            for _ in range(max_swaps):
                packs = pack_index[i]
                # Compute pack loads
                pack_w = torch.zeros(num_packs, dtype=w.dtype)
                pack_w.scatter_add_(0, packs, w)

                # Imbalance ratio guard
                delta = float((pack_w.max() - pack_w.min()).item())
                mean_ld = float(pack_w.mean().item())
                ratio = delta / max(mean_ld, 1e-12)
                if delta <= 1e-9:
                    break

                # Heaviest pack
                h = int(torch.argmax(pack_w))
                # Lightest candidates: up to 3 lightest
                light_order = torch.argsort(pack_w, descending=False)
                light_candidates = []
                for k in range(min(3, num_packs)):
                    l = int(light_order[k].item())
                    if l != h:
                        light_candidates.append(l)
                if not light_candidates:
                    break

                # Precompute max among other packs quickly per l
                def others_max(exclude_a: int, exclude_b: int) -> float:
                    tmp = pack_w.clone()
                    tmp[exclude_a] = float('-inf')
                    tmp[exclude_b] = float('-inf')
                    return float(tmp.max().item())

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

                # Evaluate best 1x1 swap across candidate lights by predicted new global peak
                best_choice = None  # (new_peak, hi, lj, chosen_light, new_heavy_ld, new_light_ld)
                for l in light_candidates:
                    light_idx = torch.nonzero(packs == l, as_tuple=False).squeeze(1)
                    if light_idx.numel() == 0:
                        continue
                    lw_all = w[light_idx]
                    kl = min(2, lw_all.numel())
                    if kl == 0:
                        continue
                    bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=kl, largest=True)
                    bottomk_lw = -bottomk_lw_vals

                    # Compare all kh x kl pairs
                    for ih in range(kh):
                        wi = float(topk_hw[ih].item())
                        hi = h_sel[ih]
                        for jl in range(kl):
                            wj = float(bottomk_lw[jl].item())
                            lj = light_idx[bottomk_pos_l[jl]]

                            new_h = float(pack_w[h].item()) - wi + wj
                            new_l = float(pack_w[l].item()) - wj + wi
                            other_max = others_max(h, l)
                            candidate_peak = max(other_max, new_h, new_l)
                            if (best_choice is None) or (candidate_peak < best_choice[0] - 0.0):
                                best_choice = (candidate_peak, hi, lj, l, new_h, new_l)

                # Optionally evaluate a 2x2 exchange for absolute lightest l0
                two_two_candidate = None  # (new_peak, hi1, lj1, hi2, lj2, l0)
                if hw_all.numel() >= 2:
                    l0 = int(light_order[0].item())
                    if l0 != h:
                        light_idx0 = torch.nonzero(packs == l0, as_tuple=False).squeeze(1)
                        if light_idx0.numel() >= 2:
                            kh2 = min(2, hw_all.numel())
                            kl2 = min(2, light_idx0.numel())
                            t_h_vals, t_h_pos = torch.topk(hw_all, k=kh2, largest=True)
                            lw0 = w[light_idx0]
                            b_l_vals, b_l_pos = torch.topk(-lw0, k=kl2, largest=True)
                            b_l_vals = -b_l_vals
                            hi1 = heavy_idx[t_h_pos[0]]
                            lj1 = light_idx0[b_l_pos[0]]
                            hi2 = heavy_idx[t_h_pos[1]] if kh2 >= 2 and kl2 >= 2 else None
                            lj2 = light_idx0[b_l_pos[1]] if kh2 >= 2 and kl2 >= 2 else None

                            wi1 = float(w[hi1].item()); wj1 = float(w[lj1].item())
                            new_h = float(pack_w[h].item()) - wi1 + wj1
                            new_l = float(pack_w[l0].item()) - wj1 + wi1
                            if hi2 is not None and lj2 is not None:
                                wi2 = float(w[hi2].item()); wj2 = float(w[lj2].item())
                                new_h = new_h - wi2 + wj2
                                new_l = new_l - wj2 + wi2
                            other_max = others_max(h, l0)
                            new_peak_22 = max(other_max, new_h, new_l)
                            two_two_candidate = (new_peak_22, hi1, lj1, hi2, lj2, l0)

                applied = False
                # If 2x2 strictly better than 1x1 (lower new global peak), apply it
                if two_two_candidate is not None:
                    peak22, hi1, lj1, hi2, lj2, lsel22 = two_two_candidate
                    better_than_1x1 = (best_choice is None) or (peak22 + 1e-12 < best_choice[0])
                    # Require strict improvement vs current peak
                    cur_peak = float(pack_w.max().item())
                    if better_than_1x1 and peak22 + 1e-12 < cur_peak:
                        wi1 = float(w[hi1].item()); wj1 = float(w[lj1].item())
                        pack_index[i, hi1] = lsel22
                        pack_index[i, lj1] = h
                        pack_w[h] = pack_w[h] - wi1 + wj1
                        pack_w[lsel22] = pack_w[lsel22] - wj1 + wi1
                        if hi2 is not None and lj2 is not None:
                            wi2 = float(w[hi2].item()); wj2 = float(w[lj2].item())
                            pack_index[i, hi2] = lsel22
                            pack_index[i, lj2] = h
                            pack_w[h] = pack_w[h] - wi2 + wj2
                            pack_w[lsel22] = pack_w[lsel22] - wj2 + wi2
                        applied = True

                if not applied and best_choice is not None:
                    candidate_peak, hi, lj, lsel, _, _ = best_choice
                    cur_peak = float(pack_w.max().item())
                    if candidate_peak + 1e-12 < cur_peak:
                        wi = float(w[hi].item())
                        wj = float(w[lj].item())
                        # Commit 1x1 swap
                        pack_index[i, hi] = lsel
                        pack_index[i, lj] = h
                        pack_w[h] = pack_w[h] - wi + wj
                        pack_w[lsel] = pack_w[lsel] - wj + wi
                        # continue searching
                        continue
                    else:
                        # optional chained fallback if imbalance pronounced
                        if ratio > 0.10:
                            # Attempt two sequential heavy-light swaps using nearest target heuristic
                            # First swap
                            heavy_idx = torch.nonzero(pack_index[i] == h, as_tuple=False).squeeze(1)
                            light_idx = torch.nonzero(pack_index[i] == light_candidates[0], as_tuple=False).squeeze(1)
                            if heavy_idx.numel() > 0 and light_idx.numel() > 0:
                                lw_sorted, lw_perm = torch.sort(w[light_idx])
                                hw_vals = w[heavy_idx]
                                delta_hl = float((pack_w[h] - pack_w[light_candidates[0]]).item())
                                target = hw_vals - (delta_hl / 2.0)
                                pos = torch.searchsorted(lw_sorted, target)
                                pos = torch.clamp(pos, 0, lw_sorted.numel() - 1)
                                cand_pos = torch.stack([pos, torch.clamp(pos - 1, 0, lw_sorted.numel() - 1)], dim=1)
                                cand_lw = lw_sorted[cand_pos]
                                resid = (delta_hl - 2.0 * (hw_vals.unsqueeze(1) - cand_lw)).abs()
                                best_flat = int(torch.argmin(resid).item())
                                hi0 = heavy_idx[best_flat // 2]
                                j_sorted_idx = int(cand_pos.view(-1)[best_flat].item())
                                lj0 = light_idx[lw_perm[j_sorted_idx]]
                                wi0 = float(w[hi0].item()); wj0 = float(w[lj0].item())
                                # Commit first swap if helps peak
                                new_h0 = float(pack_w[h].item()) - wi0 + wj0
                                new_l0 = float(pack_w[light_candidates[0]].item()) - wj0 + wi0
                                other_max0 = others_max(h, light_candidates[0])
                                peak0 = max(other_max0, new_h0, new_l0)
                                if peak0 + 1e-12 < cur_peak:
                                    pack_index[i, hi0] = light_candidates[0]
                                    pack_index[i, lj0] = h
                                    pack_w[h] = pack_w[h] - wi0 + wj0
                                    pack_w[light_candidates[0]] = pack_w[light_candidates[0]] - wj0 + wi0
                                    # Second swap attempt (lightest again)
                                    heavy_idx2 = torch.nonzero(pack_index[i] == int(torch.argmax(pack_w)), as_tuple=False).squeeze(1)
                                    l2 = int(torch.argmin(pack_w))
                                    light_idx2 = torch.nonzero(pack_index[i] == l2, as_tuple=False).squeeze(1)
                                    if heavy_idx2.numel() > 0 and light_idx2.numel() > 0:
                                        lw_sorted2, lw_perm2 = torch.sort(w[light_idx2])
                                        hw_vals2 = w[heavy_idx2]
                                        delta_hl2 = float((pack_w[int(torch.argmax(pack_w))] - pack_w[l2]).item())
                                        target2 = hw_vals2 - (delta_hl2 / 2.0)
                                        pos2 = torch.searchsorted(lw_sorted2, target2)
                                        pos2 = torch.clamp(pos2, 0, lw_sorted2.numel() - 1)
                                        cand_pos2 = torch.stack([pos2, torch.clamp(pos2 - 1, 0, lw_sorted2.numel() - 1)], dim=1)
                                        cand_lw2 = lw_sorted2[cand_pos2]
                                        resid2 = (delta_hl2 - 2.0 * (hw_vals2.unsqueeze(1) - cand_lw2)).abs()
                                        best_flat2 = int(torch.argmin(resid2).item())
                                        hi2 = heavy_idx2[best_flat2 // 2]
                                        j_sorted_idx2 = int(cand_pos2.view(-1)[best_flat2].item())
                                        lj2 = light_idx2[lw_perm2[j_sorted_idx2]]
                                        wi2 = float(w[hi2].item()); wj2 = float(w[lj2].item())
                                        new_h2 = float(pack_w[int(torch.argmax(pack_w))].item()) - wi2 + wj2
                                        new_l2 = float(pack_w[l2].item()) - wj2 + wi2
                                        other_max2 = others_max(int(torch.argmax(pack_w)), l2)
                                        peak2 = max(other_max2, new_h2, new_l2)
                                        if peak2 + 1e-12 < peak0:
                                            pack_index[i, hi2] = l2
                                            pack_index[i, lj2] = int(torch.argmax(pack_w))
                                            # pack_w updated approximately; next outer iteration recomputes it anyway
                                    # After chain, continue outer loop
                                    continue
                        # No improving move
                        break
                elif applied:
                    continue
                else:
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
    dtype_f = weight.dtype

    # Initialize base mapping (one replica per logical expert)
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    base = torch.arange(num_log, dtype=dtype_i64, device=device).unsqueeze(0).expand(n, -1)
    phy2log[:, :num_log] = base
    rank[:, :num_log] = 0
    logcnt = torch.ones(n, num_log, dtype=dtype_i64, device=device)

    if num_redundant == 0:
        return phy2log, rank, logcnt

    arange_n = torch.arange(n, dtype=dtype_i64, device=device)

    # Bulk-tail split
    T_tail = max(1, (num_redundant + 9) // 10)  # last ~10% picks
    bulk = num_redundant - T_tail
    col = num_log

    # Helper: pick argmax with deterministic tie-break by smaller current avg (weight/logcnt) then index
    def pick_best_idx(benefit: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        avg = weight / counts.to(dtype_f)
        return _argmax_tiebreak(benefit, avg, prefer_secondary_small=True)

    # Bulk: D'Hondt (benefit = weight / r)
    for _ in range(max(0, bulk)):
        benefit = weight / logcnt.to(dtype_f)
        best = pick_best_idx(benefit, logcnt)
        phy2log[:, col] = best
        rank[:, col] = logcnt[arange_n, best]
        logcnt[arange_n, best] += 1
        col += 1

    # Tail: per-step A/B between Sainte-Laguë and Huntington–Hill with second-order tie-breaker
    for _ in range(max(0, T_tail)):
        r_f = logcnt.to(dtype_f)
        # Compute candidates under S and H with tie-breaking
        benef_S = weight / (2.0 * r_f - 1.0)
        idx_S = pick_best_idx(benef_S, logcnt)
        benef_H = weight / torch.sqrt(r_f * (r_f + 1.0))
        idx_H = pick_best_idx(benef_H, logcnt)

        # For each row, simulate both choices and pick the one with smaller predicted new peak,
        # tie-broken by smaller predicted new second-highest average.
        chosen = torch.empty(n, dtype=dtype_i64, device=device)
        for ri in range(n):
            cd = logcnt[ri].clone()
            # S simulation
            cdS = cd.clone()
            s = int(idx_S[ri].item())
            cdS[s] += 1
            avgS = weight[ri] / cdS.to(dtype_f)
            peakS = float(avgS.max().item())
            if num_log >= 2:
                top2S = torch.topk(avgS, k=2).values
                secS = float(top2S[1].item())
            else:
                secS = peakS

            # H simulation
            cdH = cd.clone()
            h = int(idx_H[ri].item())
            cdH[h] += 1
            avgH = weight[ri] / cdH.to(dtype_f)
            peakH = float(avgH.max().item())
            if num_log >= 2:
                top2H = torch.topk(avgH, k=2).values
                secH = float(top2H[1].item())
            else:
                secH = peakH

            if peakS < peakH - 1e-12:
                chosen[ri] = s
            elif peakH < peakS - 1e-12:
                chosen[ri] = h
            else:
                # tie on peak, break by second-highest
                if secS < secH - 1e-12:
                    chosen[ri] = s
                elif secH < secS - 1e-12:
                    chosen[ri] = h
                else:
                    # final tie-breaker by lower index (deterministic)
                    chosen[ri] = min(s, h)

        # Commit tail pick
        phy2log[:, col] = chosen
        rank[:, col] = logcnt[arange_n, chosen]
        logcnt[arange_n, chosen] += 1
        col += 1

    # Guarded 1-move fix-up per row:
    # donors: top-2 by avg load with count > 1, receivers: bottom-2 by avg
    # choose candidate that minimizes new peak; tie-break by new second-highest avg, then (donor, receiver)
    if num_log > 1 and num_redundant > 0:
        for ri in range(n):
            counts = logcnt[ri].clone()
            avg = weight[ri] / counts.to(dtype_f)
            cur_top2 = torch.topk(avg, k=min(2, num_log))
            cur_max = float(cur_top2.values[0].item())
            # donor and receiver candidate sets
            kdon = min(2, num_log)
            krec = min(2, num_log)
            donors = torch.topk(avg, k=kdon, largest=True).indices.tolist()
            receivers = torch.topk(avg, k=krec, largest=False).indices.tolist()

            best = None  # (new_peak, new_second, d, r)
            for d in donors:
                cd = int(counts[d].item())
                if cd <= 1:
                    continue
                for r in receivers:
                    if d == r:
                        continue
                    cr = int(counts[r].item())
                    # simulate counts
                    cnt_sim = counts.clone()
                    cnt_sim[d] -= 1
                    cnt_sim[r] += 1
                    avg_new = weight[ri] / cnt_sim.to(dtype_f)
                    topk2 = torch.topk(avg_new, k=min(2, num_log)).values
                    new_peak = float(topk2[0].item())
                    new_second = float((topk2[1].item() if num_log >= 2 else topk2[0].item()))
                    if new_peak + 1e-12 < cur_max:
                        if best is None:
                            best = (new_peak, new_second, d, r)
                        else:
                            if new_peak < best[0] - 0.0:
                                best = (new_peak, new_second, d, r)
                            elif abs(new_peak - best[0]) <= 0.0:
                                if new_second < best[1] - 0.0:
                                    best = (new_peak, new_second, d, r)
                                elif abs(new_second - best[1]) <= 0.0:
                                    # final tie: lower donor, then receiver
                                    if (d < best[2]) or (d == best[2] and r < best[3]):
                                        best = (new_peak, new_second, d, r)

            if best is not None:
                _, _, d, r = best
                # move donor's highest-rank replica (deterministic pick)
                donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() > 0:
                    maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
                    col_idx = donor_cols[maxr_idx]
                    new_rank = int(logcnt[ri, r].item())
                    phy2log[ri, col_idx] = r
                    rank[ri, col_idx] = new_rank
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