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
    max_swaps: int = 2,
    adaptive_second: bool = False,
) -> torch.Tensor:
    """
    Bounded k-candidate best-improvement refinement on a single row.
    - Evaluate up to k x k pair swaps between heaviest and lightest packs.
    - Apply the single swap that minimizes |delta - 2*(wi - wj)| if it strictly improves.
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
        l = int(torch.argmin(pack_w).item())
        if h == l:
            break
        delta = float((pack_w[h] - pack_w[l]).item())
        if delta <= 1e-12:
            break

        heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
        light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
        if heavy_idx.numel() == 0 or light_idx.numel() == 0:
            break

        # top-k heaviest in heavy pack
        hw = weights[heavy_idx]
        k_h = min(k, hw.numel())
        topk_hw, topk_pos_h = torch.topk(hw, k=k_h, largest=True)

        # bottom-k lightest in light pack
        lw = weights[light_idx]
        k_l = min(k, lw.numel())
        bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw, k=k_l, largest=True)
        bottomk_lw = -bottomk_lw_vals

        # Evaluate all pairs
        diff = topk_hw.unsqueeze(1) - bottomk_lw.unsqueeze(0)  # [k_h, k_l]
        new_delta = (delta - 2.0 * diff).abs()
        best_flat = int(torch.argmin(new_delta).item())
        bi = best_flat // k_l
        bj = best_flat % k_l
        candidate_new_delta = float(new_delta[bi, bj].item())

        if candidate_new_delta + 1e-12 < delta:
            hi = heavy_idx[topk_pos_h[bi]]
            lj = light_idx[bottomk_pos_l[bj]]
            wi = float(weights[hi].item())
            wj = float(weights[lj].item())

            # commit swap
            pack_idx[hi] = l
            pack_idx[lj] = h
            pack_w[h] = pack_w[h] - wi + wj
            pack_w[l] = pack_w[l] - wj + wi

            swaps_done += 1
            if adaptive_second and not added_adaptive:
                improve_ratio = 1.0 - (candidate_new_delta / max(delta, 1e-12))
                if improve_ratio < 0.20:
                    added_adaptive = True
            continue
        else:
            # no improving swap found
            break

    return pack_idx


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

    # Greedy assignment on CPU for lightweight control-flow
    order = weight.float().argsort(-1, descending=True).cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")

    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        pidx = pack_index[i]
        for g in order[i].tolist():
            # choose lightest pack with remaining capacity
            best_pack = None
            best_load = None
            for p in range(num_packs):
                if pack_items[p] < groups_per_pack:
                    ld = pack_weights[p]
                    if best_load is None or ld < best_load:
                        best_pack = p
                        best_load = ld
            pidx[g] = best_pack
            pack_weights[best_pack] += float(weight[i, g].item())
            pack_items[best_pack] += 1

        # bounded refinement (k=2) with adaptive steps based on imbalance
        w_row = weight[i].cpu()
        pack_w = torch.zeros(num_packs, dtype=w_row.dtype)
        pack_w.scatter_add_(0, pidx, w_row)
        delta = float((pack_w.max() - pack_w.min()).item())
        mean_ld = float(pack_w.mean().item())
        steps = 3 if delta / max(mean_ld, 1e-12) > 0.12 else 2
        pack_index[i] = _kcandidate_refine_row(
            w_row, pidx.clone(), num_packs, k=2, max_swaps=int(steps), adaptive_second=False
        )

    # Compute deterministic ranks once from final pack indices
    rank_in_pack = torch.empty_like(pack_index, dtype=torch.int64)
    for i in range(num_layers):
        rank_in_pack[i] = _rank_from_packidx_rowwise(pack_index[i], num_packs)

    return pack_index, rank_in_pack


def _balanced_packing_diverse(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int,
    refine_steps_default: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diversity-aware variant of balanced packing:
      - Same greedy scheme to assign the next heaviest item to the lightest pack with capacity.
      - If multiple candidate packs are near-tied in load, prefer the pack with fewer items of the same label.
      - Bounded refinement afterwards (best-swap), with adaptive steps based on imbalance.

    weights: [L, N], labels: [L, N] int64
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
        load = [0.0] * num_packs
        counts = [0] * num_packs
        label_cnt = [dict() for _ in range(num_packs)]
        pidx = torch.empty(N, dtype=torch.int64)

        mean_w = float(torch.mean(w).item()) if N > 0 else 0.0
        eps = 1e-6 * max(mean_w, 1e-12)
        lam = 1e-8 * max(mean_w, 1e-12)  # negligible penalty

        for g in order.tolist():
            cand = [p for p in range(num_packs) if counts[p] < capacity]
            min_load = min(load[p] for p in cand)
            eff = []
            lbl = int(labs[g].item())
            for p in cand:
                ld = load[p]
                if ld - min_load <= eps:
                    same = label_cnt[p].get(lbl, 0)
                    eff.append(ld + lam * same)
                else:
                    eff.append(ld)
            best_idx = int(torch.argmin(torch.tensor(eff)).item())
            best_pack = cand[best_idx]
            pidx[g] = best_pack
            load[best_pack] += float(w[g].item())
            counts[best_pack] += 1
            label_cnt[best_pack][lbl] = label_cnt[best_pack].get(lbl, 0) + 1

        # Adaptive refinement depth based on imbalance
        pack_w = torch.zeros(num_packs, dtype=w.dtype)
        pack_w.scatter_add_(0, pidx, w)
        delta = float((pack_w.max() - pack_w.min()).item())
        mean_ld = float(pack_w.mean().item())
        ratio = delta / max(mean_ld, 1e-12)
        steps = 3 if ratio > 0.12 else refine_steps_default

        # bounded refinement (k=2) with one possible adaptive extra
        pidx = _kcandidate_refine_row(w, pidx, num_packs, k=2, max_swaps=int(steps), adaptive_second=True)

        pack_index[li] = pidx
        rank_in_pack[li] = _rank_from_packidx_rowwise(pidx, num_packs)

    return pack_index.to(weights.device), rank_in_pack.to(weights.device)


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
    dtype_f = weight.dtype
    dtype_i64 = torch.int64

    # Initialize base mapping (one replica per logical expert)
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    base = torch.arange(num_log, dtype=dtype_i64, device=device).unsqueeze(0).expand(n, -1)
    phy2log[:, :num_log] = base
    rank[:, :num_log] = 0
    logcnt = torch.ones(n, num_log, dtype=dtype_i64, device=device)

    if num_redundant == 0:
        return phy2log, rank, logcnt

    arangen = torch.arange(n, dtype=dtype_i64, device=device)

    def _argmax_tiebreak(benefit: torch.Tensor, avg: torch.Tensor) -> torch.Tensor:
        """
        Rowwise argmax with deterministic tie-breaking:
          1) Among max benefit ties, choose the smaller current avg (weight/logcnt).
          2) If still tied, choose the lowest index.
        benefit, avg: [n, num_log]
        returns indices: [n]
        """
        maxv = benefit.max(dim=-1, keepdim=True).values
        mask = benefit == maxv
        inf = torch.tensor(float('inf'), dtype=avg.dtype, device=avg.device)
        avg_masked = torch.where(mask, avg, inf)
        return torch.argmin(avg_masked, dim=-1)

    # Hybrid allocation: D'Hondt for bulk, then per-step Sainte-Laguë vs Huntington–Hill tail
    tail = max(1, (num_redundant + 9) // 10)
    bulk = num_redundant - tail

    col = num_log
    # Bulk phase (D'Hondt): benefit = weight / r
    for _ in range(max(0, bulk)):
        r_f = logcnt.to(dtype_f)
        benefit = weight / r_f
        best = _argmax_tiebreak(benefit, weight / r_f)
        phy2log[:, col] = best
        rank[:, col] = logcnt[arangen, best]
        logcnt[arangen, best] += 1
        col += 1

    # Tail phase: per-row peak-aware choice between Sainte-Laguë and D'Hondt
    for _ in range(max(0, tail)):
        r_f = logcnt.to(dtype_f)
        avg_cur = weight / r_f
        # current second-highest average per row
        if num_log > 1:
            top2_vals = torch.topk(avg_cur, k=2, dim=-1, largest=True).values
            second = top2_vals[:, 1]
        else:
            second = avg_cur[:, 0]

        benef_S = weight / (2.0 * r_f - 1.0)
        benef_D = weight / r_f
        idx_S = _argmax_tiebreak(benef_S, avg_cur)
        idx_D = _argmax_tiebreak(benef_D, avg_cur)

        newS = weight[arangen, idx_S] / (r_f[arangen, idx_S] + 1.0)
        newD = weight[arangen, idx_D] / (r_f[arangen, idx_D] + 1.0)
        peakS = torch.maximum(second, newS)
        peakD = torch.maximum(second, newD)

        better_S = peakS + 1e-12 < peakD
        tie_SD = torch.isclose(peakS, peakD, rtol=0.0, atol=1e-12)
        prefer_S_on_tie = newS <= newD
        use_S = better_S | (tie_SD & prefer_S_on_tie)

        best_idx = torch.where(use_S, idx_S, idx_D)
        phy2log[:, col] = best_idx
        rank[:, col] = logcnt[arangen, best_idx]
        logcnt[arangen, best_idx] += 1
        col += 1

    # Vectorized strengthened replication fix-up:
    # consider donors=top-2 of avg, receivers=bottom-2 of avg, choose the single best move per row.
    if num_log > 1 and num_redundant > 0:
        r_f = logcnt.to(dtype_f)
        avg = weight / r_f
        kd = min(2, num_log)
        kr = min(2, num_log)

        top_vals, top_idx = torch.topk(avg, k=kd, dim=-1, largest=True)  # [n, kd]
        cur_max = top_vals[:, 0]
        sec = (top_vals[:, 1] if kd > 1 else top_vals[:, 0])
        bot_vals, bot_idx = torch.topk(avg, k=kr, dim=-1, largest=False)  # [n, kr]

        # Gather counts and weights
        cd = logcnt.gather(1, top_idx)        # [n, kd]
        cr = logcnt.gather(1, bot_idx)        # [n, kr]
        wd = weight.gather(1, top_idx).to(dtype_f)  # [n, kd]
        wr = weight.gather(1, bot_idx).to(dtype_f)  # [n, kr]

        # New averages after moving one replica: donor loses one, receiver gains one
        new_d = wd / (cd.to(dtype_f) - 1.0).clamp_min(1.0)  # avoid div by 0 in invalids
        new_r = wr / (cr.to(dtype_f) + 1.0)

        # Broadcast shapes for pairwise evaluation
        new_d_b = new_d.unsqueeze(2).expand(-1, kd, kr)  # [n, kd, kr]
        new_r_b = new_r.unsqueeze(1).expand(-1, kd, kr)  # [n, kd, kr]
        sec_b = sec.view(-1, 1, 1).expand(-1, kd, kr)    # [n, kd, kr]

        # Valid masks: donors must have count > 1 and donor != receiver
        valid_cd = (cd > 1).unsqueeze(2).expand(-1, kd, kr)
        d_idx_b = top_idx.unsqueeze(2).expand(-1, kd, kr)
        r_idx_b = bot_idx.unsqueeze(1).expand(-1, kd, kr)
        valid_diff = (d_idx_b != r_idx_b)
        valid = valid_cd & valid_diff

        # Predicted new peak per pair
        new_peak = torch.maximum(sec_b, torch.maximum(new_d_b, new_r_b))
        inf = torch.tensor(float('inf'), dtype=new_peak.dtype, device=new_peak.device)
        new_peak = torch.where(valid, new_peak, inf)

        # Pick best pair per row
        best_vals, best_flat = torch.min(new_peak.view(n, -1), dim=-1)
        improve = best_vals + 1e-12 < cur_max
        rows = torch.nonzero(improve, as_tuple=False).squeeze(1)

        if rows.numel() > 0:
            kr_size = kr
            for ri in rows.tolist():
                bf = int(best_flat[ri].item())
                di = bf // kr_size
                rj = bf % kr_size
                d = int(top_idx[ri, di].item())
                r = int(bot_idx[ri, rj].item())

                # move one physical replica from expert d to r: choose the one with highest rank
                donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() == 0:
                    continue
                maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
                col_idx = donor_cols[maxr_idx]

                new_rank = int(logcnt[ri, r].item())
                phy2log[ri, col_idx] = r
                rank[ri, col_idx] = new_rank

                logcnt[ri, d] -= 1
                logcnt[ri, r] += 1

            # Optional second move for shallow improvements (<15%) only
            imp = (cur_max[rows] - best_vals[rows]) / cur_max[rows].clamp_min(1e-12)
            shallow_rows = rows[imp < 0.15]
            if shallow_rows.numel() > 0:
                for ri in shallow_rows.tolist():
                    # recompute best move for this row under updated counts
                    rfi = logcnt[ri].to(dtype_f)
                    avgi = weight[ri] / rfi
                    kdi = min(2, num_log)
                    kri = min(2, num_log)
                    top_vals_i, top_idx_i = torch.topk(avgi, k=kdi, largest=True)
                    bot_vals_i, bot_idx_i = torch.topk(avgi, k=kri, largest=False)
                    cur_max_i = float(top_vals_i[0].item())
                    sec_i = float((top_vals_i[1].item() if kdi > 1 else top_vals_i[0].item()))
                    best_new_peak = None
                    best_pair = None
                    for di2 in range(kdi):
                        d2 = int(top_idx_i[di2].item())
                        cd2 = int(logcnt[ri, d2].item())
                        if cd2 <= 1:
                            continue
                        for rj2 in range(kri):
                            r2 = int(bot_idx_i[rj2].item())
                            if d2 == r2:
                                continue
                            cr2 = int(logcnt[ri, r2].item())
                            new_dv = float(weight[ri, d2].item()) / float(cd2 - 1)
                            new_rv = float(weight[ri, r2].item()) / float(cr2 + 1)
                            cand_peak = max(sec_i, new_dv, new_rv)
                            if cand_peak + 1e-12 < cur_max_i:
                                if best_new_peak is None or cand_peak < best_new_peak:
                                    best_new_peak = cand_peak
                                    best_pair = (d2, r2)
                    if best_pair is not None:
                        d2, r2 = best_pair
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

    # Step 3: pack physical_experts to GPUs with diversity-aware tie-breaker
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    labels_per_phy = phy2mlog  # prefer spreading same meta-logical ids across GPUs under near ties
    pack_index, rank_in_pack = _balanced_packing_diverse(
        tokens_per_phy, labels_per_phy, num_gpus // num_nodes, refine_steps_default=2
    )
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