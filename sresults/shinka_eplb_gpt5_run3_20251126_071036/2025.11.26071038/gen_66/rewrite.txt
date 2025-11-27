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


class _GreedyPacker:
    """
    Capacity-constrained greedy packer with bounded, deterministic refinements.

    Stages:
      - Greedy assign sorted items to the currently lightest pack with remaining capacity.
      - Optional bounded refinement with targeted 1x1 swaps chosen to minimize post-swap peak.
    """

    def __init__(self):
        pass

    @staticmethod
    def _rank_from_packidx_row(pack_idx_row: torch.Tensor, num_packs: int) -> torch.Tensor:
        """
        Deterministic ranks within each pack by ascending original item id.
        pack_idx_row: [N]
        returns rank_row: [N]
        """
        N = pack_idx_row.numel()
        device = pack_idx_row.device
        rank_row = torch.empty(N, dtype=torch.int64, device=device)
        for p in range(num_packs):
            ids = torch.nonzero(pack_idx_row == p, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            ids_sorted = torch.sort(ids).values
            rank_row[ids_sorted] = torch.arange(ids_sorted.numel(), dtype=torch.int64, device=device)
        return rank_row

    @staticmethod
    def _refine_once(weights: torch.Tensor,
                     pack_idx: torch.Tensor,
                     num_packs: int) -> tuple[bool, torch.Tensor]:
        """
        Single best 1x1 swap chosen among donor packs {heaviest, second-heaviest}
        and receiver packs {lightest, second-lightest, third-lightest}.

        Returns (applied, updated_pack_idx)
        """
        device = weights.device
        # compute pack loads
        pack_w = torch.zeros(num_packs, dtype=weights.dtype, device=device)
        pack_w.scatter_add_(0, pack_idx, weights)
        cur_peak = float(pack_w.max().item())

        # donors: heaviest, second-heaviest
        donors = torch.topk(pack_w, k=min(2, num_packs), largest=True).indices.tolist()
        # receivers: lightest, second, third
        rec_cnt = min(3, num_packs)
        receivers = torch.topk(pack_w, k=rec_cnt, largest=False).indices.tolist()

        best = None  # (new_peak, new_second, hi, lj, h, l, wi, wj)
        for h in donors:
            heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
            if heavy_idx.numel() == 0:
                continue
            hw = weights[heavy_idx]
            for l in receivers:
                if l == h:
                    continue
                light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
                if light_idx.numel() == 0:
                    continue
                lw = weights[light_idx]
                # imbalance between h and l
                delta = float((pack_w[h] - pack_w[l]).item())
                if delta <= 1e-12:
                    continue
                # choose best pair via searchsorted on light ascending
                lw_sorted, lw_perm = torch.sort(lw)  # ascending
                target = hw - (delta / 2.0)
                pos = torch.searchsorted(lw_sorted, target)
                pos = torch.clamp(pos, 0, lw_sorted.numel() - 1)
                cand_pos = torch.stack([pos, torch.clamp(pos - 1, 0, lw_sorted.numel() - 1)], dim=1)
                cand_lw = lw_sorted[cand_pos]
                resid = (delta - 2.0 * (hw.unsqueeze(1) - cand_lw)).abs()
                best_flat = int(torch.argmin(resid).item())
                ih = best_flat // 2
                jl = best_flat % 2
                j_sorted = int(cand_pos[ih, jl].item())

                wi = float(hw[ih].item())
                wj = float(lw_sorted[j_sorted].item())
                hi = heavy_idx[ih]
                lj = light_idx[lw_perm[j_sorted]]
                # predicted loads after swap
                new_h = float(pack_w[h].item()) - wi + wj
                new_l = float(pack_w[l].item()) - wj + wi
                # compute new peak and second peak
                tmp = pack_w.clone()
                tmp[h] = new_h
                tmp[l] = new_l
                top2_vals = torch.topk(tmp, k=min(2, num_packs), largest=True).values
                new_peak = float(top2_vals[0].item())
                new_second = float((top2_vals[1].item() if top2_vals.numel() > 1 else top2_vals[0].item()))
                if new_peak + 1e-12 < cur_peak:
                    if best is None or (new_peak < best[0] - 1e-12) or (abs(new_peak - best[0]) <= 1e-12 and new_second < best[1]):
                        best = (new_peak, new_second, hi, lj, h, l, wi, wj)

        if best is None:
            return False, pack_idx

        # commit
        _, _, hi, lj, h, l, wi, wj = best
        pack_idx[hi] = l
        pack_idx[lj] = h
        return True, pack_idx

    @staticmethod
    def _refine_with_chain(weights: torch.Tensor,
                           pack_idx: torch.Tensor,
                           num_packs: int,
                           allow_two_swap_chain: bool = False) -> torch.Tensor:
        """
        Apply single best swap; optionally attempt a two-swap chain only if no single strict improvement is possible and
        imbalance is pronounced (>10%). Commit chain only if it strictly reduces the final max.
        """
        device = weights.device
        # compute pack loads baseline
        pack_w0 = torch.zeros(num_packs, dtype=weights.dtype, device=device)
        pack_w0.scatter_add_(0, pack_idx, weights)
        base_peak = float(pack_w0.max().item())

        # try single swap
        applied, pidx = _GreedyPacker._refine_once(weights, pack_idx.clone(), num_packs)
        if applied:
            return pidx

        if not allow_two_swap_chain:
            return pack_idx

        # pronounced imbalance?
        mean_ld = float(pack_w0.mean().item())
        if mean_ld <= 0:
            return pack_idx
        imbalance = (float(pack_w0.max().item()) - float(pack_w0.min().item())) / mean_ld
        if imbalance <= 0.10:
            return pack_idx

        # attempt chain on a copy
        ptmp = pack_idx.clone()
        # first move (may not strictly improve)
        changed, ptmp = _GreedyPacker._refine_once(weights, ptmp, num_packs)
        if not changed:
            return pack_idx
        # second move
        _, ptmp2 = _GreedyPacker._refine_once(weights, ptmp.clone(), num_packs)

        pack_w2 = torch.zeros(num_packs, dtype=weights.dtype, device=device)
        pack_w2.scatter_add_(0, ptmp2, weights)
        if float(pack_w2.max().item()) + 1e-12 < base_peak:
            return ptmp2
        return pack_idx

    def pack(self,
             weight: torch.Tensor,
             num_packs: int,
             refine_steps: int = 1,
             labels: torch.Tensor | None = None,
             allow_two_swap_chain: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy packing with bounded refinement.

        Parameters:
            weight: [L, N]
            num_packs: M
            refine_steps: number of refinement iterations (each attempts at most one swap)
            labels: optional [L, N] for diversity-aware tie-breaks under near ties
            allow_two_swap_chain: enable two-swap chain fallback (GPU stage only)

        Returns:
            pack_index: [L, N], pack id per item
            rank_in_pack: [L, N], 0..N/M-1
        """
        L, N = weight.shape
        assert N % num_packs == 0
        capacity = N // num_packs

        # Fast path for capacity 1
        if capacity == 1:
            pack_index = torch.arange(N, dtype=torch.int64, device=weight.device).expand(L, -1).contiguous()
            rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
            return pack_index, rank_in_pack

        # Work on CPU for control-flow efficiency
        w_cpu = weight.float().cpu()
        pack_index = torch.full((L, N), -1, dtype=torch.int64)
        rank_in_pack = torch.full_like(pack_index, -1)
        if labels is not None:
            lab_cpu = labels.long().cpu()
        eps_scale = 1e-6
        lam_scale = 1e-8

        for li in range(L):
            w = w_cpu[li]
            order = torch.argsort(w, descending=True)
            load = [0.0] * num_packs
            counts = [0] * num_packs
            pidx = torch.empty(N, dtype=torch.int64)
            # diversity counters if labels provided
            label_cnt = [dict() for _ in range(num_packs)] if labels is not None else None

            mean_w = float(w.mean().item()) if N > 0 else 0.0
            eps = eps_scale * max(mean_w, 1e-12)
            lam = lam_scale * max(mean_w, 1e-12)

            for g in order.tolist():
                candidates = [p for p in range(num_packs) if counts[p] < capacity]
                # current min load among candidates
                min_load = min(load[p] for p in candidates)
                # effective load with tiny diversity penalty when near-tied
                eff = []
                if labels is None:
                    for p in candidates:
                        eff.append(load[p])
                else:
                    lbl = int(lab_cpu[li, g].item())
                    for p in candidates:
                        ld = load[p]
                        if ld - min_load <= eps:
                            same = label_cnt[p].get(lbl, 0)
                            eff.append(ld + lam * same)
                        else:
                            eff.append(ld)
                best_idx = int(torch.argmin(torch.tensor(eff)).item())
                best_pack = candidates[best_idx]

                pidx[g] = best_pack
                counts[best_pack] += 1
                load[best_pack] += float(w[g].item())
                if labels is not None:
                    lbl = int(lab_cpu[li, g].item())
                    label_cnt[best_pack][lbl] = label_cnt[best_pack].get(lbl, 0) + 1

            # bounded refinement
            steps = max(int(refine_steps), 0)
            if steps > 0:
                for s in range(steps):
                    if allow_two_swap_chain and s == 0:
                        # allow chain only at first step
                        pidx = self._refine_with_chain(w, pidx, num_packs, allow_two_swap_chain=True)
                    else:
                        applied, pidx = self._refine_once(w, pidx, num_packs)
                        if not applied:
                            break

            rnk = self._rank_from_packidx_row(pidx, num_packs)
            pack_index[li] = pidx
            rank_in_pack[li] = rnk

        return pack_index.to(weight.device), rank_in_pack.to(weight.device)


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
    packer = _GreedyPacker()
    return packer.pack(weight, num_packs, refine_steps=refine_steps)


def _adaptive_tail_length(weight_row: torch.Tensor, num_extra: int) -> int:
    """
    Adaptive Sainte-Laguë tail fraction based on dispersion.
    tail = clamp(1, num_extra, round(alpha * num_extra * s)), where
    s is coefficient of variation clamped to [0.7, 1.3], alpha=0.10.
    """
    if num_extra <= 0:
        return 0
    mean = float(weight_row.mean().item()) if weight_row.numel() > 0 else 0.0
    if mean <= 0:
        return max(1, int(round(0.10 * num_extra)))
    std = float(weight_row.std().item())
    cv = (std / mean) if mean > 0 else 0.0
    s = min(1.3, max(0.7, cv))
    tail = int(round(0.10 * num_extra * s))
    tail = max(1, min(num_extra, tail))
    return tail


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, minimizing the maximum
    average per-replica load.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    L, num_log = weight.shape
    num_extra = num_phy - num_log
    assert num_extra >= 0
    device = weight.device
    dtype_i64 = torch.int64
    dtype_f = weight.dtype

    # initialize base mapping: one per logical expert
    phy2log = torch.empty((L, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((L, num_phy), dtype=dtype_i64, device=device)
    base = torch.arange(num_log, dtype=dtype_i64, device=device).unsqueeze(0).expand(L, -1)
    phy2log[:, :num_log] = base
    rank[:, :num_log] = 0
    logcnt = torch.ones((L, num_log), dtype=dtype_i64, device=device)

    if num_extra == 0:
        return phy2log, rank, logcnt

    rows = torch.arange(L, dtype=dtype_i64, device=device)
    # Precompute per-row tail decision (A/B between SL and HH)
    tail_len = torch.tensor(
        [_adaptive_tail_length(weight[i], num_extra) for i in range(L)],
        dtype=torch.int64,
        device=device,
    )
    bulk_len = (num_extra - tail_len).clamp_min(0)

    # Decide per-row tail policy by simulating on counts after bulk
    use_SL_tail = torch.zeros(L, dtype=torch.bool, device=device)
    for i in range(L):
        bulk = int(bulk_len[i].item())
        cnt_sim = torch.ones(num_log, dtype=dtype_i64, device=device)
        if bulk > 0:
            for _ in range(bulk):
                benefit = weight[i] / cnt_sim.to(dtype_f)
                bi = int(torch.argmax(benefit).item())
                cnt_sim[bi] += 1
        # simulate tail for SL and HH
        T = int(tail_len[i].item())
        if T <= 0:
            use_SL_tail[i] = True  # irrelevant
            continue
        cntS = cnt_sim.clone()
        cntH = cnt_sim.clone()
        for _ in range(T):
            denomS = (2.0 * cntS.to(dtype_f) - 1.0)
            benefS = weight[i] / denomS
            idxS = int(torch.argmax(benefS).item())
            cntS[idxS] += 1

            denomH = torch.sqrt(cntH.to(dtype_f) * (cntH.to(dtype_f) + 1.0))
            benefH = weight[i] / denomH
            idxH = int(torch.argmax(benefH).item())
            cntH[idxH] += 1
        peakS = float((weight[i] / cntS.to(dtype_f)).max().item())
        peakH = float((weight[i] / cntH.to(dtype_f)).max().item())
        use_SL_tail[i] = peakS <= peakH

    # Perform actual allocation with per-row bulk/tail policy
    col = num_log
    for step in range(num_extra):
        in_bulk = step < bulk_len  # [L] bool
        # D'Hondt candidates
        benef_D = weight / logcnt.to(dtype_f)
        idx_D = benef_D.argmax(dim=-1)

        # Tail candidates
        denomS = (2.0 * logcnt.to(dtype_f) - 1.0)
        benef_S = weight / denomS
        idx_S = benef_S.argmax(dim=-1)

        denomH = torch.sqrt(logcnt.to(dtype_f) * (logcnt.to(dtype_f) + 1.0))
        benef_H = weight / denomH
        idx_H = benef_H.argmax(dim=-1)

        # Choose per row
        choose_tail = (~in_bulk)  # bool tensor
        choose_S = choose_tail & use_SL_tail
        choose_H = choose_tail & (~use_SL_tail)

        best_idx = torch.where(in_bulk, idx_D, torch.where(choose_S, idx_S, idx_H))
        phy2log[:, col] = best_idx
        # ranks are current counts before increment
        rank[:, col] = logcnt[rows, best_idx]
        # increment counts
        logcnt[rows, best_idx] += 1
        col += 1

    # One-commit fix-up per row (top-2 donors -> bottom-2 receivers)
    if num_log > 1 and num_extra > 0:
        for i in range(L):
            r_f = logcnt[i].to(dtype_f)
            avg = weight[i] / r_f
            kdon = min(2, num_log)
            krec = min(2, num_log)
            top_vals, top_idx = torch.topk(avg, k=kdon, largest=True)
            bot_vals, bot_idx = torch.topk(avg, k=krec, largest=False)
            cur_max = float(top_vals[0].item())
            second = float((top_vals[1].item() if kdon > 1 else top_vals[0].item()))
            best = None  # (new_peak, new_second, d, r)

            for di in range(kdon):
                d = int(top_idx[di].item())
                cd = int(logcnt[i, d].item())
                if cd <= 1:
                    continue
                for rj in range(krec):
                    r = int(bot_idx[rj].item())
                    if r == d:
                        continue
                    cr = int(logcnt[i, r].item())
                    new_d = float(weight[i, d].item()) / float(cd - 1)
                    new_r = float(weight[i, r].item()) / float(cr + 1)
                    candidate_peak = max(second, new_d, new_r)
                    # second-highest after move (approx by max among second and the non-peak term)
                    candidate_second = sorted([second, new_d, new_r], reverse=True)[1]
                    if candidate_peak + 1e-12 < cur_max:
                        if best is None or (candidate_peak < best[0] - 1e-12) or (abs(candidate_peak - best[0]) <= 1e-12 and candidate_second < best[1]):
                            best = (candidate_peak, candidate_second, d, r)
            if best is not None:
                _, _, d, r = best
                # move one highest-rank replica from d to r
                donor_cols = torch.nonzero(phy2log[i] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() == 0:
                    continue
                maxr_idx = torch.argmax(rank[i, donor_cols]).item()
                col_idx = donor_cols[maxr_idx]
                new_rank = int(logcnt[i, r].item())
                phy2log[i, col_idx] = r
                rank[i, col_idx] = new_rank
                logcnt[i, d] -= 1
                logcnt[i, r] += 1

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

    packer = _GreedyPacker()

    # Step 1: pack groups to nodes (balanced packing on tokens per group)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = packer.pack(tokens_per_group, num_nodes, refine_steps=1)
    # logical -> meta-logical (contiguous within node)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = PermOps.inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads
    # tokens_per_mlog: [L*num_nodes, num_logical_experts/num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (still within nodes)
    # avg per replica load as weights to pack
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    labels_per_phy = phy2mlog  # encourage spreading same meta-logical ids
    pack_index, rank_in_pack = packer.pack(tokens_per_phy,
                                           num_gpus // num_nodes,
                                           refine_steps=2,
                                           labels=labels_per_phy,
                                           allow_two_swap_chain=True)
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
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # global policy (treat as single node)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

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
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1),
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