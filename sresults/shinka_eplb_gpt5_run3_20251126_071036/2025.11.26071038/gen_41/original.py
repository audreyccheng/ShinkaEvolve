# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses:
- Water-filling replica allocation to minimize the maximum per-replica load,
  with a conditional second donor->receiver move when the first improvement
  is shallow (guarded and capped at two moves).
- Diversity-aware heap-based packing for GPU placement within nodes,
  with bounded refinement:
    * 1x1 swap between heaviest and {lightest, second-lightest}
    * Optional 2x2 swap between heaviest and lightest (top-2 vs bottom-2),
      applied only if it strictly reduces the global max more than best 1x1.
    * Adaptive second refinement step when imbalance is high (delta/mean > 0.12).
"""

import torch
from collections import defaultdict


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
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Longest-processing-time greedy with capacity constraints
    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((num_layers, num_groups),
                             fill_value=-1,
                             dtype=torch.int64,
                             device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_loads = [0.0] * num_packs
        pack_counts = [0] * num_packs
        for g in indices[i].tolist():
            # choose pack with capacity and minimal load
            best_p = None
            best_load = None
            for p in range(num_packs):
                if pack_counts[p] >= groups_per_pack:
                    continue
                if best_load is None or pack_loads[p] < best_load:
                    best_load = pack_loads[p]
                    best_p = p
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = pack_counts[best_p]
            pack_counts[best_p] += 1
            pack_loads[best_p] += float(weight[i, g].item())
    return pack_index, rank_in_pack


def _waterfill_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Compute integer replica counts c_i >= 1 that approximately minimize max_i w_i / c_i
    subject to sum c_i == target_total using water-filling + greedy fill.

    Parameters:
        w: [num_log], float tensor (on CPU)
        target_total: int, total replicas to allocate

    Returns:
        counts: [num_log], int64
    """
    num_log = w.numel()
    assert target_total >= num_log  # at least one per expert

    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=w.device)

    maxw = float(w.max().item()) if num_log > 0 else 0.0
    # Binary search T such that sum max(1, ceil(w_i / T)) <= target_total
    lo = 0.0
    hi = max(maxw, 1.0)
    # Handle all-zero quickly
    if maxw == 0.0:
        counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
        extras = target_total - num_log
        if extras > 0:
            base_add = extras // num_log
            rem = extras % num_log
            if base_add > 0:
                counts += base_add
            if rem > 0:
                counts[:rem] += 1
        return counts

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        # counts_i = max(1, ceil(w_i / mid))
        c = torch.ceil(w / mid).to(torch.int64)
        c = torch.maximum(c, torch.ones_like(c))
        s = int(c.sum().item())
        if s <= target_total:
            hi = mid
        else:
            lo = mid

    # Base counts from hi guarantee <= target_total
    counts = torch.ceil(w / hi).to(torch.int64)
    counts = torch.maximum(counts, torch.ones_like(counts))
    s = int(counts.sum().item())

    # Greedy water-filling for remaining extras
    extras = target_total - s
    while extras > 0:
        k = min(extras, num_log)
        # Select top-k by current w_i / c_i
        scores = w / counts.to(w.dtype)
        topk_idx = torch.argsort(scores, descending=True)[:k]
        counts[topk_idx] += 1
        extras -= k
    return counts


def replicate_experts_waterfill(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Water-filling replication to minimize the maximum per-replica load,
    plus a bounded donor→receiver fix-up:
      - Perform one best move among top-2 donors and bottom-2 receivers.
      - If improvement is shallow (new_peak > 0.9 * old_peak), attempt one more
        guarded move with the same candidate bounds. Cap at two moves/row.

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
    shallow_improve_ratio = 0.9  # if first improvement is shallow, try a second move
    k_d = 2  # donors candidates
    k_r = 2  # receivers candidates

    phy2log_list = []
    rank_list = []
    logcnt_list = []

    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)
    for i in range(n):
        w = weight[i]  # [num_log], float CPU
        counts = _waterfill_counts_row(w, num_phy)  # int64

        def _one_fixup(counts_in: torch.Tensor) -> tuple[torch.Tensor, float]:
            counts_try = counts_in.clone()
            avg = w / counts_try.to(w.dtype)
            cur_max = float(avg.max().item())
            donors_mask = counts_try > 1
            if not bool(donors_mask.any()):
                return counts_try, cur_max
            kd = int(min(k_d, int(donors_mask.sum().item())))
            kr = int(min(k_r, num_log))
            avg_mask = avg.clone()
            avg_mask[~donors_mask] = float("-inf")
            donors = torch.topk(avg_mask, k=kd).indices.tolist()
            receivers = torch.topk(-avg, k=kr).indices.tolist()
            best_pair = None
            best_peak = cur_max
            for d in donors:
                for r in receivers:
                    if d == r:
                        continue
                    c_try = counts_try.clone()
                    c_try[d] -= 1
                    c_try[r] += 1
                    peak = float((w / c_try.to(w.dtype)).max().item())
                    if peak + 1e-9 < best_peak:
                        best_peak = peak
                        best_pair = (d, r)
            if best_pair is not None:
                d, r = best_pair
                counts_try[d] -= 1
                counts_try[r] += 1
                return counts_try, best_peak
            return counts_in, cur_max

        # First fix-up
        counts_after, peak_after = _one_fixup(counts)
        peak_before = float((w / counts.to(w.dtype)).max().item())
        counts = counts_after

        # Conditional second fix-up if improvement is shallow
        if peak_after > shallow_improve_ratio * peak_before:
            counts_second, peak_second = _one_fixup(counts)
            if peak_second + 1e-12 < peak_after:
                counts = counts_second

        logcnt_list.append(counts)

        # Build phy2log and rank (contiguous blocks per logical expert)
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        # ranks: 0..count-1 for each expert
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    logcnt = torch.stack(logcnt_list, dim=0)
    return phy2log, rank, logcnt


def pack_diverse_heap(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diversity-aware heap-like greedy packing with exact capacity per pack,
    followed by bounded adaptive refinement:
      - 1x1 swap between heaviest and {lightest, second-lightest} (choose best).
      - Optional 2x2 swap between heaviest and lightest (top-2 vs bottom-2)
        only if it strictly reduces global max more than the best 1x1 swap.
      - Up to 2 refinement passes when initial imbalance is high.

    Parameters:
        weights: [X, n], float CPU
        labels:  [X, n], int64 CPU
        num_packs: number of packs

    Returns:
        pack_index: [X, n], assigned pack id per item
        rank_in_pack: [X, n], position within pack
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    if cap == 1:
        # Each pack gets exactly one item
        idx = torch.arange(n_items, dtype=torch.int64, device=weights.device)
        pack_index = (idx % num_packs).expand(num_layers, n_items).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=weights.device)
    rank_in_pack = torch.full_like(pack_index, -1)

    sorted_idx_all = weights.sort(dim=-1, descending=True).indices  # [X, n]

    # Parameters for refinement
    light_candidates = 2           # consider lightest and second-lightest
    enable_pair_pair = True        # allow 2x2 swap (bounded)
    imbalance_threshold = 0.12     # if delta/mean > threshold, allow second pass
    max_refine_steps = 2           # at most two refinement passes

    for i in range(num_layers):
        row_w = weights[i]
        row_labels = labels[i]
        sorted_idx = sorted_idx_all[i].tolist()

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)]
        pack_groups = [[] for _ in range(num_packs)]  # items per pack

        # Greedy placement: prefer lower repeats, then smaller load, then fewer items
        for g in sorted_idx:
            lab = int(row_labels[g].item())
            wv = float(row_w[g].item())

            best_p = None
            best_key = None  # (rep, load, count)
            for p in range(num_packs):
                if counts[p] >= cap:
                    continue
                rep = label_counts[p].get(lab, 0)
                key = (rep, loads[p], counts[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_p = p

            pack_index[i, g] = best_p
            rank_in_pack[i, g] = counts[best_p]
            counts[best_p] += 1
            loads[best_p] += wv
            label_counts[best_p][lab] += 1
            pack_groups[best_p].append(g)

        # Adaptive bounded refinement
        def _refine_once() -> bool:
            # Identify heaviest and candidate light packs
            h = max(range(num_packs), key=lambda p: loads[p])
            order = sorted([p for p in range(num_packs) if p != h], key=lambda p: loads[p])
            if not order:
                return False
            cand_l = order[:min(light_candidates, len(order))]
            cur_max = max(loads)
            cur_min = min(loads)
            cur_imb = cur_max - cur_min
            if cur_imb <= 0:
                return False

            best1 = None
            best1_peak = cur_max
            best1_imb = cur_imb
            # 1x1 swaps between h and cand light packs
            if pack_groups[h]:
                h_idx_tensor = torch.tensor(pack_groups[h], dtype=torch.int64, device=row_w.device)
                h_w = row_w[h_idx_tensor]
                kh = min(2, h_w.numel())
                if kh > 0:
                    h_top = torch.topk(h_w, kh).indices.tolist()
                    for l in cand_l:
                        if not pack_groups[l] or loads[h] <= loads[l]:
                            continue
                        l_idx_tensor = torch.tensor(pack_groups[l], dtype=torch.int64, device=row_w.device)
                        l_w = row_w[l_idx_tensor]
                        kl = min(2, l_w.numel())
                        if kl == 0:
                            continue
                        l_bot = torch.topk(l_w, kl, largest=False).indices.tolist()

                        other_max = max([loads[p] for p in range(num_packs) if p != h and p != l], default=float("-inf"))
                        other_min = min([loads[p] for p in range(num_packs) if p != h and p != l], default=float("inf"))

                        for ai in h_top:
                            a_item = int(h_idx_tensor[ai].item())
                            wa = float(h_w[ai].item())
                            for bi in l_bot:
                                b_item = int(l_idx_tensor[bi].item())
                                wb = float(l_w[bi].item())

                                new_h = loads[h] - wa + wb
                                new_l = loads[l] - wb + wa
                                new_peak = max(new_h, new_l, other_max)
                                new_bottom = min(new_h, new_l, other_min)
                                new_imb = new_peak - new_bottom

                                # Keep the swap that minimizes new_peak first, then new_imb
                                if (best1 is None or
                                    (new_peak + 1e-9 < best1_peak) or
                                    (abs(new_peak - best1_peak) <= 1e-9 and new_imb + 1e-9 < best1_imb)):
                                    best1 = (h, l, ai, bi, a_item, b_item, wa, wb)
                                    best1_peak = new_peak
                                    best1_imb = new_imb

            # Optional 2x2 swap on (h, lightest) only if it beats best 1x1 in reducing peak
            best2 = None
            best2_peak = cur_max
            if enable_pair_pair and pack_groups[h]:
                # choose the single lightest pack
                l_star = order[0] if order else None
                if l_star is not None and pack_groups[l_star]:
                    h_idx_tensor = torch.tensor(pack_groups[h], dtype=torch.int64, device=row_w.device)
                    l_idx_tensor = torch.tensor(pack_groups[l_star], dtype=torch.int64, device=row_w.device)
                    h_w = row_w[h_idx_tensor]
                    l_w = row_w[l_idx_tensor]
                    if h_w.numel() >= 2 and l_w.numel() >= 2:
                        top_h2 = torch.topk(h_w, 2).indices.tolist()
                        bot_l2 = torch.topk(l_w, 2, largest=False).indices.tolist()
                        ai, aj = top_h2[0], top_h2[1]
                        bi, bj = bot_l2[0], bot_l2[1]
                        wa_sum = float(h_w[ai].item() + h_w[aj].item())
                        wb_sum = float(l_w[bi].item() + l_w[bj].item())
                        other_max = max([loads[p] for p in range(num_packs) if p != h and p != l_star], default=float("-inf"))
                        other_min = min([loads[p] for p in range(num_packs) if p != h and p != l_star], default=float("inf"))
                        new_h = loads[h] - wa_sum + wb_sum
                        new_l = loads[l_star] - wb_sum + wa_sum
                        new_peak2 = max(new_h, new_l, other_max)
                        new_bottom2 = min(new_h, new_l, other_min)
                        new_imb2 = new_peak2 - new_bottom2
                        # Keep candidate
                        best2 = (h, l_star, ai, aj, bi, bj, new_peak2, new_imb2,
                                 int(h_idx_tensor[ai].item()), int(h_idx_tensor[aj].item()),
                                 int(l_idx_tensor[bi].item()), int(l_idx_tensor[bj].item()),
                                 wa_sum, wb_sum)
                        best2_peak = new_peak2

            # Decide which swap to apply
            applied = False
            if best1 is not None and best1_peak + 1e-9 < cur_max:
                # If 2x2 available and beats 1x1 by peak, use it
                if best2 is not None and best2_peak + 1e-9 < best1_peak:
                    # Apply 2x2 on (h, l_star)
                    h_sel, l_sel, ai, aj, bi, bj, new_peak2, new_imb2, a_i, a_j, b_i, b_j, wa_sum, wb_sum = best2
                    # Update loads
                    loads[h_sel] = loads[h_sel] - wa_sum + wb_sum
                    loads[l_sel] = loads[l_sel] - wb_sum + wa_sum
                    # Swap membership: replace two items in each pack
                    # Ensure indices ai, aj, bi, bj refer to current list positions
                    # We stored local indices; apply in order
                    pack_groups[h_sel][ai] = b_i
                    pack_groups[h_sel][aj] = b_j
                    pack_groups[l_sel][bi] = a_i
                    pack_groups[l_sel][bj] = a_j
                    # Update indices and ranks for affected packs
                    pack_index[i, a_i] = l_sel
                    pack_index[i, a_j] = l_sel
                    pack_index[i, b_i] = h_sel
                    pack_index[i, b_j] = h_sel
                    for r, g in enumerate(pack_groups[h_sel]):
                        rank_in_pack[i, g] = r
                    for r, g in enumerate(pack_groups[l_sel]):
                        rank_in_pack[i, g] = r
                    applied = True
                else:
                    # Apply best 1x1
                    h_sel, l_sel, ai, bi, a_item, b_item, wa, wb = best1
                    loads[h_sel] = loads[h_sel] - wa + wb
                    loads[l_sel] = loads[l_sel] - wb + wa
                    pack_groups[h_sel][ai] = b_item
                    pack_groups[l_sel][bi] = a_item
                    pack_index[i, a_item] = l_sel
                    pack_index[i, b_item] = h_sel
                    for r, g in enumerate(pack_groups[h_sel]):
                        rank_in_pack[i, g] = r
                    for r, g in enumerate(pack_groups[l_sel]):
                        rank_in_pack[i, g] = r
                    applied = True

            return applied

        # Compute initial imbalance and adapt refinement depth
        mean_load = sum(loads) / max(1, num_packs)
        delta = max(loads) - min(loads)
        refine_steps = 2 if (mean_load > 0 and delta / max(mean_load, 1e-12) > imbalance_threshold) else 1
        refine_steps = min(refine_steps, max_refine_steps)

        for _ in range(refine_steps):
            changed = _refine_once()
            if not changed:
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

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes (capacity-aware LPT)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate within nodes using water-filling + conditional second fix-up
    # Reorder to meta-logical within nodes: [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts_waterfill(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node with diversity-aware heap + bounded refinement
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # each row corresponds to a (layer, node)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = pack_diverse_heap(
        tokens_per_phy, phy2mlog, gpus_per_node
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

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
        # hierarchical policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global policy (treat as single node)
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