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
        h = int(torch.argmax(pack_w).item())
        light_order = torch.argsort(pack_w, descending=False)
        l0 = int(light_order[0].item())
        light_candidates = [l0]
        if consider_second_light and num_packs >= 2:
            l1 = int(light_order[1].item())
            if l1 != l0 and l1 != h:
                light_candidates.append(l1)

        if h in light_candidates and len(light_candidates) == 1:
            break

        heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
        if heavy_idx.numel() == 0:
            break
        hw_all = weights[heavy_idx]
        k_h = min(k, hw_all.numel())
        topk_hw_vals, topk_pos_h = torch.topk(hw_all, k=k_h, largest=True)

        best_choice = None  # (new_delta, hi_idx, lj_idx, chosen_light)
        base_delta = None

        # try lightest and optional second-lightest
        for l in light_candidates:
            if l == h:
                continue
            light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
            if light_idx.numel() == 0:
                continue
            lw_all = weights[light_idx]
            k_l = min(k, lw_all.numel())
            bottomk_lw_vals, bottomk_pos_l = torch.topk(-lw_all, k=k_l, largest=True)
            bottomk_lw = -bottomk_lw_vals

            delta = float((pack_w[h] - pack_w[l]).item())
            if delta <= 1e-12:
                continue

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

        # 2x2 candidate (only against the lightest pack)
        two_two_candidate = None
        if allow_two_two and hw_all.numel() >= 2:
            l = l0
            if l != h:
                light_idx0 = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
                if light_idx0.numel() >= 2:
                    kh2 = min(2, hw_all.numel())
                    kl2 = min(2, light_idx0.numel())
                    t_h_vals, t_h_pos = torch.topk(hw_all, k=kh2, largest=True)
                    lw0 = weights[light_idx0]
                    b_l_vals, b_l_pos = torch.topk(-lw0, k=kl2, largest=True)
                    b_l_vals = -b_l_vals
                    delta0 = float((pack_w[h] - pack_w[l]).item())
                    sum_h = float(t_h_vals.sum().item())
                    sum_l = float(b_l_vals.sum().item())
                    new_delta_22 = abs(delta0 - 2.0 * (sum_h - sum_l))
                    two_two_candidate = (new_delta_22, t_h_pos, b_l_pos, l)

        applied = False
        if two_two_candidate is not None:
            nd22, hpos22, lpos22, l22 = two_two_candidate
            if base_delta is None or nd22 + 1e-12 < base_delta:
                hi1 = heavy_idx[hpos22[0]]
                lj1 = torch.nonzero(pack_idx == l22, as_tuple=False).squeeze(1)[lpos22[0]]
                wi1 = float(weights[hi1].item())
                wj1 = float(weights[lj1].item())
                pack_idx[hi1] = l22
                pack_idx[lj1] = h
                pack_w[h] = pack_w[h] - wi1 + wj1
                pack_w[l22] = pack_w[l22] - wj1 + wi1

                if hpos22.numel() >= 2 and lpos22.numel() >= 2:
                    hi2 = heavy_idx[hpos22[1]]
                    lj2 = torch.nonzero(pack_idx == l22, as_tuple=False).squeeze(1)[lpos22[1]]
                    wi2 = float(weights[hi2].item())
                    wj2 = float(weights[lj2].item())
                    pack_idx[hi2] = l22
                    pack_idx[lj2] = h
                    pack_w[h] = pack_w[h] - wi2 + wj2
                    pack_w[l22] = pack_w[l22] - wj2 + wi2

                swaps_done += 1
                applied = True

        if not applied and best_choice is not None:
            best_nd, hi, lj, lsel = best_choice
            delta_sel = float((pack_w[h] - pack_w[lsel]).item())
            if best_nd + 1e-12 < delta_sel:
                wi = float(weights[hi].item())
                wj = float(weights[lj].item())
                pack_idx[hi] = lsel
                pack_idx[lj] = h
                pack_w[h] = pack_w[h] - wi + wj
                pack_w[lsel] = pack_w[lsel] - wj + wi
                swaps_done += 1
                if adaptive_second and not added_adaptive:
                    improve_ratio = 1.0 - (best_nd / max(delta_sel, 1e-12))
                    if improve_ratio < 0.20:
                        added_adaptive = True
                continue
            else:
                break
        elif applied:
            if adaptive_second and not added_adaptive:
                added_adaptive = True
            continue
        else:
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

        # Compute imbalance ratio for adaptive refinement
        pack_w = torch.zeros(num_packs, dtype=w.dtype)
        pack_w.scatter_add_(0, pidx, w)
        delta = float((pack_w.max() - pack_w.min()).item())
        mean_ld = float(pack_w.mean().item())
        ratio = delta / max(mean_ld, 1e-12)

        # Bounded refinement with expanded light candidates; enable 2x2 when steps >= 2
        pidx = _kcandidate_refine_row(
            w,
            pidx,
            num_packs,
            k=2,
            max_swaps=int(refine_steps if ratio <= 0.12 else max(2, refine_steps)),
            adaptive_second=True,
            consider_second_light=True,
            allow_two_two=(refine_steps >= 2 or ratio > 0.12),
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
        load = [0.0] * num_packs
        counts = [0] * num_packs
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


def _waterfill_counts_row(w_row: torch.Tensor, total: int) -> torch.Tensor:
    """
    Minimax water-filling for integer counts:
      - Find minimal T such that sum ceil(w_i / T) (clamped to >=1) <= total
      - Use those counts as base, then allocate remaining seats (if any)
        by incremental D'Hondt starting from the base counts.

    w_row: [num_log] float (CPU)
    total: total replicas P (int)
    returns counts: [num_log] int64 summing to total
    """
    num_log = w_row.numel()
    # at least one per logical expert
    assert total >= num_log

    dtype_f = w_row.dtype
    eps = torch.tensor(1e-12, dtype=dtype_f, device=w_row.device)
    hi_val = float(torch.maximum(w_row.max(), eps).item())
    lo = 0.0
    hi = hi_val

    def sum_counts(T: float) -> tuple[torch.Tensor, int]:
        T_val = max(T, 1e-12)
        c = torch.ceil(w_row / T_val).to(torch.int64)
        c = torch.clamp(c, min=1)
        s = int(c.sum().item())
        return c, s

    # Binary search ~ 32 iterations
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        _, s = sum_counts(mid)
        if s > total:
            lo = mid
        else:
            hi = mid

    base_c, s = sum_counts(hi)
    counts = base_c.clone()
    remain = total - s
    if remain > 0:
        # Incremental D'Hondt continuation from base counts
        w = w_row
        for _ in range(remain):
            benefit = w / counts.to(dtype_f)
            # tie-break deterministic by smallest index
            best = int(torch.argmax(benefit).item())
            counts[best] += 1
    return counts


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, minimizing max per-replica load
    via water-filling integer apportionment with exact-seat continuation.

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

    # Trivial case: one per logical
    if num_extra == 0:
        phy2log = torch.arange(num_phy, dtype=dtype_i64, device=device).repeat(n, 1)
        rank = torch.zeros(n, num_phy, dtype=dtype_i64, device=device)
        logcnt = torch.ones(n, num_log, dtype=dtype_i64, device=device)
        return phy2log, rank, logcnt

    # Work row-wise on CPU for control flow
    w_cpu = weight.float().cpu()
    logcnt_cpu = torch.empty((n, num_log), dtype=dtype_i64)

    total = num_phy
    for ri in range(n):
        counts = _waterfill_counts_row(w_cpu[ri], total)
        logcnt_cpu[ri] = counts

    # Build phy2log and rank deterministically by compacting repeats
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64)
    rank = torch.empty((n, num_phy), dtype=dtype_i64)
    for ri in range(n):
        offset = 0
        for j in range(num_log):
            cnt = int(logcnt_cpu[ri, j].item())
            if cnt <= 0:
                continue
            phy2log[ri, offset:offset + cnt] = j
            rank[ri, offset:offset + cnt] = torch.arange(cnt, dtype=dtype_i64)
            offset += cnt
        # Safety: ensure all columns filled
        if offset < num_phy:
            # pad (shouldn't happen, but keep deterministic fallback)
            pad = num_phy - offset
            phy2log[ri, offset:] = 0
            rank[ri, offset:] = 0

    return phy2log.to(device), rank.to(device), logcnt_cpu.to(device)


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

    # Step 1: pack groups to nodes using balanced_packing (bounded refine, 1 step adaptive)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes, refine_steps=1)

    # Compute permutation logical -> meta-logical (within-node contiguous layout)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = PermOps.inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads using water-filling
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (within nodes) with diversity-aware tie-breaker and adaptive refinement
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)  # per-replica avg loads
    labels_per_phy = phy2mlog
    pack_index, rank_in_pack = _balanced_packing_diverse(tokens_per_phy, labels_per_phy, num_gpus // num_nodes, refine_steps_default=2)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = PermOps.inverse(phy2pphy)

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
