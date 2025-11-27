# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses:
- Equal-Proportions (Huntington–Hill) replica allocation for discrete rounding,
  with a small peak-first donor->receiver fix-up to further smooth the peak.
- Peak-aware, diversity-tied GPU packing with a round-robin flavor that only
  considers the best two candidate packs by projected load, plus a tiny bounded
  refinement (1x1 with 2x2 fallback) and adaptive refine depth.
"""

import torch
from collections import defaultdict


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64,
                     device=perm.device).expand(perm.shape),
    )
    return inv


def _balanced_packing_lpt_with_fix(weight: torch.Tensor,
                                   num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LPT packing with exact capacity and a single bounded 1x1 swap if needed.

    Parameters:
        weight: [X, n]
        num_packs: m

    Returns:
        pack_index: [X, n] in [0, m)
        rank_in_pack: [X, n] in [0, n/m)
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    cap = num_groups // num_packs

    if cap == 1 or num_packs == 1:
        idx = torch.arange(num_groups, dtype=torch.int64, device=weight.device)
        pack_index = (idx % num_packs).expand(num_layers, num_groups).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    sorted_idx_all = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((num_layers, num_groups), -1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, -1)

    for i in range(num_layers):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        groups_in_pack = [[] for _ in range(num_packs)]
        for g in sorted_idx_all[i].tolist():
            # choose among packs with capacity the one with minimal load; tie: smaller count
            best_p = None
            best_load = None
            best_cnt = None
            for p in range(num_packs):
                if counts[p] >= cap:
                    continue
                ld = loads[p]
                ct = counts[p]
                if (best_p is None or ld < best_load or
                    (abs(ld - best_load) <= 1e-12 and ct < best_cnt)):
                    best_p = p
                    best_load = ld
                    best_cnt = ct
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = counts[best_p]
            counts[best_p] += 1
            loads[best_p] += float(weight[i, g].item())
            groups_in_pack[best_p].append(g)

        # One bounded 1x1 swap if imbalance is notable
        if num_packs >= 2:
            cur_max = max(loads)
            cur_min = min(loads)
            if cur_max > cur_min and (cur_max - cur_min) > 0:
                order = sorted(range(num_packs), key=lambda p: loads[p])
                l = order[0]
                h = order[-1]
                if groups_in_pack[h] and groups_in_pack[l]:
                    # consider top-1 from heavy and bottom-1 from light
                    h_idx_tensor = torch.tensor(groups_in_pack[h], dtype=torch.int64, device=weight.device)
                    l_idx_tensor = torch.tensor(groups_in_pack[l], dtype=torch.int64, device=weight.device)
                    h_w = weight[i, h_idx_tensor]
                    l_w = weight[i, l_idx_tensor]
                    ai = int(torch.topk(h_w, 1).indices[0].item()) if h_w.numel() > 0 else None
                    bi = int(torch.topk(l_w, 1, largest=False).indices[0].item()) if l_w.numel() > 0 else None
                    if ai is not None and bi is not None:
                        a_item = int(h_idx_tensor[ai].item())
                        b_item = int(l_idx_tensor[bi].item())
                        wa = float(h_w[ai].item())
                        wb = float(l_w[bi].item())
                        new_h = loads[h] - wa + wb
                        new_l = loads[l] - wb + wa
                        new_peak = max(new_h, new_l, *[loads[p] for p in range(num_packs) if p not in (h, l)])
                        if new_peak + 1e-12 < cur_max:
                            # apply swap
                            loads[h] = new_h
                            loads[l] = new_l
                            groups_in_pack[h][ai] = b_item
                            groups_in_pack[l][bi] = a_item
                            pack_index[i, a_item] = l
                            pack_index[i, b_item] = h
                            # ranks update for affected packs only
                            for rr, gg in enumerate(groups_in_pack[h]):
                                rank_in_pack[i, gg] = rr
                            for rr, gg in enumerate(groups_in_pack[l]):
                                rank_in_pack[i, gg] = rr

    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


def _equal_proportions_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Equal Proportions (Huntington–Hill) apportionment for integer counts c_i >= 1,
    sum c_i == target_total, maximizing priorities p_i = w_i / sqrt(c_i (c_i+1)).
    Returns counts that tend to minimize the peak average w_i / c_i after rounding.

    Parameters:
        w: [num_log], float CPU tensor
        target_total: total replicas

    Returns:
        counts: [num_log], int64 CPU tensor
    """
    num_log = int(w.numel())
    assert target_total >= num_log

    counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
    if target_total == num_log or num_log == 0:
        return counts

    extras = target_total - num_log
    if float(w.max().item() if num_log > 0 else 0.0) == 0.0:
        # All-zero: even spreading
        base_add = extras // num_log
        rem = extras % num_log
        if base_add > 0:
            counts += base_add
        if rem > 0:
            counts[:rem] += 1
        return counts

    # Batched priority assignment (tiny batches for speed+accuracy)
    while extras > 0:
        k = min(extras, max(1, min(num_log, 32)))
        cf = counts.to(w.dtype)
        denom = torch.sqrt(cf * (cf + 1.0))
        priority = w / torch.clamp(denom, min=1e-12)
        topk_idx = torch.topk(priority, k=k).indices
        counts[topk_idx] += 1
        extras -= k

    return counts


def _replicate_experts_equalprop(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equal-Proportions replication with tiny peak-aware fix-ups.

    Parameters:
        weight: [X, num_log] (CPU float)
        num_phy: total number of replicas

    Returns:
        phy2log: [X, num_phy], logical expert id for each physical expert
        rank:    [X, num_phy], replica rank per logical expert
        logcnt:  [X, num_log], replica counts per logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device
    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)

    phy2log_list = []
    rank_list = []
    logcnt_list = []

    for i in range(n):
        w = weight[i]
        counts = _equal_proportions_counts_row(w, num_phy)

        # Up to 2 guarded donor->receiver moves among top-2 donors and bottom-2 receivers
        if num_log > 1:
            max_iters = 2
            for _ in range(max_iters):
                counts_safe = torch.clamp(counts, min=1)
                avg = w / counts_safe.to(w.dtype)
                cur_peak = float(avg.max().item())
                can_donate = (counts > 1)
                if not bool(can_donate.any()):
                    break
                kd = int(min(2, int(can_donate.sum().item())))
                donors = torch.topk(avg.masked_fill(~can_donate, float("-inf")), k=kd).indices.tolist()
                receivers = torch.topk(-avg, k=min(2, num_log)).indices.tolist()

                best_pair = None
                best_key = None  # (new_peak, new_second, donor_post_avg)
                for d in donors:
                    if counts[d] <= 1:
                        continue
                    for r in receivers:
                        if d == r:
                            continue
                        c_try = counts.clone()
                        c_try[d] -= 1
                        c_try[r] += 1
                        avg_try = w / c_try.to(w.dtype)
                        new_peak = float(avg_try.max().item())
                        if new_peak + 1e-12 >= cur_peak:
                            continue
                        if num_log >= 2:
                            top2_vals = torch.topk(avg_try, k=min(2, num_log)).values
                            new_second = float(top2_vals[-1].item()) if top2_vals.numel() >= 2 else float("-inf")
                        else:
                            new_second = float("-inf")
                        donor_post = float((w[d] / float(counts[d] - 1)).item()) if counts[d] > 1 else float("inf")
                        cand = (new_peak, new_second, donor_post, d, r)
                        if best_pair is None or cand < best_key:
                            best_pair = (d, r)
                            best_key = cand
                if best_pair is None:
                    break
                d, r = best_pair
                counts[d] -= 1
                counts[r] += 1

        logcnt_list.append(counts)

        # Build phy2log and rank: contiguous blocks per logical expert
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    logcnt = torch.stack(logcnt_list, dim=0)
    return phy2log, rank, logcnt


def _pack_diverse_rr_peak(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Peak-aware, diversity-tied packing with exact capacity per pack:
    - Greedy LPT by projected load but only among the best two packs per item
      (small candidate set); ties broken by least label frequency, then count.
    - Fast path when labels are unique.
    - Tiny bounded refinement: 1x1 swap among (heaviest, lightest/second-lightest);
      fallback 2x2 for (heaviest, lightest) if needed.
    - Adaptive refinement depth based on residual imbalance.

    Parameters:
        weights: [X, n] float
        labels:  [X, n] int (logical id for replicas)
        num_packs: m

    Returns:
        pack_index: [X, n]
        rank_in_pack: [X, n]
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    if cap == 1 or num_packs == 1:
        idx = torch.arange(n_items, dtype=torch.int64, device=weights.device)
        pack_index = (idx % num_packs).expand(num_layers, n_items).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=weights.device)
    rank_in_pack = torch.full_like(pack_index, -1)

    sorted_idx_all = weights.sort(dim=-1, descending=True).indices  # [X, n]

    for i in range(num_layers):
        row_w = weights[i]
        row_labels = labels[i]
        sorted_idx = sorted_idx_all[i].tolist()

        # Quick duplicate check
        lbl_sorted, _ = torch.sort(row_labels)
        all_unique = (torch.unique_consecutive(lbl_sorted).numel() == row_labels.numel())

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)]
        groups = [[] for _ in range(num_packs)]

        eps = 1e-6 * float(row_w.mean().item() if row_w.numel() > 0 else 1.0)

        for g in sorted_idx:
            wv = float(row_w[g].item())
            lab = int(row_labels[g].item())
            # Build projected loads and get the best two candidate packs by load
            cand = []
            for p in range(num_packs):
                if counts[p] >= cap:
                    continue
                base = loads[p] + wv
                cand.append((base, p))
            cand.sort(key=lambda x: x[0])
            if not cand:
                # should not happen if capacities are exact
                continue
            # Consider at most two best projected packs
            chosen_p = cand[0][1]
            if len(cand) >= 2:
                p1 = cand[0][1]
                p2 = cand[1][1]
                # tie-break by diversity then count then base
                rep1 = label_counts[p1].get(lab, 0)
                rep2 = label_counts[p2].get(lab, 0)
                cnt1 = counts[p1]
                cnt2 = counts[p2]
                base1 = cand[0][0]
                base2 = cand[1][0]
                # Near-tie by base allows diversity preference
                if (base2 - base1) <= eps:
                    if rep2 < rep1 or (rep1 == rep2 and (cnt2 < cnt1 or (cnt1 == cnt2 and base2 < base1))):
                        chosen_p = p2
                else:
                    # If the heavier projected load still produces fewer repeats and significantly close, pick it
                    if rep2 + 0 < rep1 and (base2 <= base1 + eps * 10):
                        chosen_p = p2

            pack_index[i, g] = chosen_p
            rank_in_pack[i, g] = counts[chosen_p]
            counts[chosen_p] += 1
            loads[chosen_p] += wv
            label_counts[chosen_p][lab] += 1
            groups[chosen_p].append(g)

        # Bounded refinement
        def _refine_once() -> bool:
            if num_packs < 2:
                return False
            order = sorted(range(num_packs), key=lambda p: loads[p])
            l1 = order[0]
            h1 = order[-1]
            r_candidates = [l1]
            for p in order[1:]:
                if p != h1:
                    r_candidates.append(p)
                if len(r_candidates) >= 2:
                    break

            cur_max = max(loads)
            other_max_base = [loads[p] for p in range(num_packs)]
            best1 = None  # (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb)

            # 1x1 among donors {h1} and receivers {l1, maybe l2}
            for r in r_candidates:
                if not groups[h1] or not groups[r]:
                    continue
                if loads[h1] <= loads[r]:
                    continue
                d_idx = torch.tensor(groups[h1], dtype=torch.int64, device=weights.device)
                r_idx = torch.tensor(groups[r], dtype=torch.int64, device=weights.device)
                d_w = row_w[d_idx]
                r_w = row_w[r_idx]
                kd = min(2, d_w.numel())
                kr = min(2, r_w.numel())
                if kd == 0 or kr == 0:
                    continue
                d_top = torch.topk(d_w, kd).indices.tolist()
                r_bot = torch.topk(r_w, kr, largest=False).indices.tolist()

                other_max = max([loads[p] for p in range(num_packs) if p not in (h1, r)], default=float("-inf"))
                other_min = min([loads[p] for p in range(num_packs) if p not in (h1, r)], default=float("inf"))

                for ai in d_top:
                    a_item = int(d_idx[ai].item())
                    wa = float(d_w[ai].item())
                    la = int(row_labels[a_item].item())
                    for bi in r_bot:
                        b_item = int(r_idx[bi].item())
                        wb = float(r_w[bi].item())
                        lb = int(row_labels[b_item].item())
                        new_d = loads[h1] - wa + wb
                        new_r = loads[r] - wb + wa
                        new_peak = max(new_d, new_r, other_max)
                        new_bottom = min(new_d, new_r, other_min)
                        new_imb = new_peak - new_bottom
                        # small diversity penalty: avoid introducing duplicates if possible
                        penalty = 0
                        penalty += 1 if label_counts[h1].get(lb, 0) > 0 else 0
                        penalty += 1 if label_counts[r].get(la, 0) > 0 else 0
                        cand = (new_peak, new_imb, penalty, h1, r, ai, bi, a_item, b_item, wa, wb)
                        if best1 is None or cand < best1:
                            best1 = cand

            applied = False
            if best1 is not None and best1[0] + 1e-12 < cur_max:
                _, _, _, d, r, ai, bi, a_item, b_item, wa, wb = best1
                loads[d] = loads[d] - wa + wb
                loads[r] = loads[r] - wb + wa
                groups[d][ai] = b_item
                groups[r][bi] = a_item
                pack_index[i, a_item] = r
                pack_index[i, b_item] = d
                # update ranks for both packs
                for rr, gg in enumerate(groups[d]):
                    rank_in_pack[i, gg] = rr
                for rr, gg in enumerate(groups[r]):
                    rank_in_pack[i, gg] = rr
                # update label counts quickly
                la = int(row_labels[a_item].item())
                lb = int(row_labels[b_item].item())
                label_counts[d][la] -= 1
                if label_counts[d][la] == 0:
                    del label_counts[d][la]
                label_counts[d][lb] = label_counts[d].get(lb, 0) + 1
                label_counts[r][lb] -= 1
                if label_counts[r][lb] == 0:
                    del label_counts[r][lb]
                label_counts[r][la] = label_counts[r].get(la, 0) + 1
                applied = True
            else:
                # 2x2 fallback for (h1, l1)
                l1 = order[0]
                if groups[h1] and groups[l1]:
                    h_idx = torch.tensor(groups[h1], dtype=torch.int64, device=weights.device)
                    l_idx = torch.tensor(groups[l1], dtype=torch.int64, device=weights.device)
                    h_w = row_w[h_idx]
                    l_w = row_w[l_idx]
                    if h_w.numel() >= 2 and l_w.numel() >= 2:
                        ai, aj = torch.topk(h_w, 2).indices.tolist()
                        bi, bj = torch.topk(l_w, 2, largest=False).indices.tolist()
                        a_items = (int(h_idx[ai].item()), int(h_idx[aj].item()))
                        b_items = (int(l_idx[bi].item()), int(l_idx[bj].item()))
                        wa = float(h_w[ai].item() + h_w[aj].item())
                        wb = float(l_w[bi].item() + l_w[bj].item())
                        other_max = max([loads[p] for p in range(num_packs) if p not in (h1, l1)], default=float("-inf"))
                        other_min = min([loads[p] for p in range(num_packs) if p not in (h1, l1)], default=float("inf"))
                        new_h = loads[h1] - wa + wb
                        new_l = loads[l1] - wb + wa
                        new_peak = max(new_h, new_l, other_max)
                        cur_max2 = max(loads)
                        if new_peak + 1e-12 < cur_max2:
                            loads[h1] = new_h
                            loads[l1] = new_l
                            groups[h1][ai] = b_items[0]
                            groups[h1][aj] = b_items[1]
                            groups[l1][bi] = a_items[0]
                            groups[l1][bj] = a_items[1]
                            pack_index[i, a_items[0]] = l1
                            pack_index[i, a_items[1]] = l1
                            pack_index[i, b_items[0]] = h1
                            pack_index[i, b_items[1]] = h1
                            for rr, gg in enumerate(groups[h1]):
                                rank_in_pack[i, gg] = rr
                            for rr, gg in enumerate(groups[l1]):
                                rank_in_pack[i, gg] = rr
                            applied = True
            return applied

        mean_load = sum(loads) / max(1, num_packs)
        delta = max(loads) - min(loads)
        rel = (delta / max(mean_load, 1e-12)) if mean_load > 0 else 0.0
        refine_steps = 1 if rel <= 0.02 else (2 if rel <= 0.12 else 3)
        for _ in range(refine_steps):
            if not _refine_once():
                break

    return pack_index, rank_in_pack


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

    # Step 1: pack groups to nodes with LPT + one bounded fix
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _balanced_packing_lpt_with_fix(
        tokens_per_group, num_nodes
    )
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = _inverse(log2mlog)

    # Step 2: replicate within nodes using Equal Proportions apportionment
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = _replicate_experts_equalprop(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node with peak-aware, diversity-tied packing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = _pack_diverse_rr_peak(
        tokens_per_phy, phy2mlog, gpus_per_node
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [num_layers * num_nodes, num_phy_per_node]
    # convert back to global logical indexing
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
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
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
    # scatter physical indices into logical->physical map
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