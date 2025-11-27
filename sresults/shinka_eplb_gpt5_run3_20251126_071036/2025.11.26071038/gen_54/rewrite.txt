# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a strategy-based
pipeline:

- ReplicaAllocator:
    * Bulk allocation by D’Hondt priorities (w / (c + 1)).
    * Tail micro-AB across Sainte-Laguë (w / (2c + 1)) and Huntington–Hill
      (w / sqrt(c(c+1))) restricted to last 10% extras; choose the tail that
      minimizes the peak (tie by second-highest).
    * Guarded 2×2 donor→receiver fix-up: evaluate top-2 donors (c>1) ×
      bottom-2 receivers; accept a single move only if it strictly reduces the
      peak, breaking ties by the lowest new second-highest average.

- CapacityPacker:
    * Unique-label fast path with small 1×1 refinement.
    * Diversity-aware greedy (min repeats, then load, then count).
    * Broadened 1×1 swap partner search (donors={heaviest,2nd-heaviest};
      receivers={lightest,2nd,3rd}) picking best global swap via a nearest
      difference search.
    * Bounded chained two-swap fallback when imbalance is pronounced (>10%)
      and no 1×1 swap helps, applied only if it strictly improves the final
      max over the best single swap.

- HierPlanner:
    * Group→node packing (capacity-aware LPT).
    * Per-(layer,node) replica allocation via ReplicaAllocator.
    * Intra-node GPU packing via CapacityPacker.

API and I/O are unchanged.
"""

import torch
from collections import defaultdict
from typing import Tuple, List


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64,
                                       device=perm.device).expand(perm.shape))
    return inv


def _balanced_packing_lpt(weight: torch.Tensor,
                          num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Capacity-aware LPT to assign exactly n/num_packs items per pack.

    weight: [X, n]
    Returns:
      - pack_index: [X, n]
      - rank_in_pack: [X, n]
    """
    num_layers, n = weight.shape
    assert n % num_packs == 0
    cap = n // num_packs

    if cap == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device)
        pack_index = (idx % num_packs).expand(num_layers, n).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((num_layers, n), -1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, -1)
    for i in range(num_layers):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for g in indices[i].tolist():
            best_p = None
            best_load = None
            for p in range(num_packs):
                if counts[p] >= cap:
                    continue
                if best_load is None or loads[p] < best_load:
                    best_load = loads[p]
                    best_p = p
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = counts[best_p]
            counts[best_p] += 1
            loads[best_p] += float(weight[i, g].item())
    return pack_index, rank_in_pack


class ReplicaAllocator:
    """
    Hybrid proportional-seat allocator with tail micro-AB and a small guarded fix-up.
    """

    def __init__(self, tail_frac: float = 0.10):
        self.tail_frac = tail_frac

    @staticmethod
    def _dhondt_priority(w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Next-seat priority for D'Hondt given current seats c (>=1): w / (c + 1)
        return w / (c.to(w.dtype) + 1.0)

    @staticmethod
    def _sainte_lague_priority(w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Next-seat priority for Sainte-Laguë: w / (2c + 1)
        return w / (2.0 * c.to(w.dtype) + 1.0)

    @staticmethod
    def _huntington_hill_priority(w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Next-seat priority for Huntington–Hill: w / sqrt(c(c+1))
        cc = c.to(w.dtype)
        return w / torch.sqrt(torch.clamp(cc * (cc + 1.0), min=1.0))

    @staticmethod
    def _argmax_with_tie(avg_next: torch.Tensor,
                         avg_cur: torch.Tensor) -> int:
        """
        Deterministic tie-breaking:
          - choose max(avg_next)
          - if tie, prefer smaller current avg (avg_cur)
          - if tie, prefer lower index
        avg_next: candidate priority vector
        avg_cur: current avg load vector (w / c)
        """
        # We want argmax on avg_next, with tie by min avg_cur then min index.
        # Convert to scores: primary large, secondary small. Use lex by grouping.
        max_val = torch.max(avg_next)
        cand = (avg_next >= max_val - 1e-18)
        if cand.sum() == 1:
            return int(torch.argmax(avg_next).item())
        idxs = torch.nonzero(cand, as_tuple=False).flatten()
        sub_cur = avg_cur[idxs]
        # smaller avg_cur preferred
        min_cur = torch.min(sub_cur)
        cand2 = idxs[sub_cur <= min_cur + 1e-18]
        return int(cand2.min().item())

    def _allocate_row(self, w: torch.Tensor, total: int) -> torch.Tensor:
        """
        Allocate counts for a single row.
        w: [num_log] float CPU
        total: target total replicas
        """
        device = w.device
        n = w.numel()
        assert total >= n
        if n == 0:
            return torch.zeros(0, dtype=torch.int64, device=device)
        counts = torch.ones(n, dtype=torch.int64, device=device)
        extras = total - n
        if extras == 0:
            return counts

        # Handle all-zero row quickly
        if float(w.max().item()) == 0.0:
            if extras > 0:
                base_add = extras // n
                rem = extras % n
                if base_add > 0:
                    counts += base_add
                if rem > 0:
                    counts[:rem] += 1
            return counts

        # Bulk via D'Hondt
        T = max(1, int((extras * self.tail_frac)))
        bulk = extras - T
        for _ in range(bulk):
            priority = self._dhondt_priority(w, counts)
            avg_cur = w / counts.to(w.dtype)
            j = self._argmax_with_tie(priority, avg_cur)
            counts[j] += 1

        if T > 0:
            # Simulate two tails: Sainte-Laguë and Huntington–Hill
            def sim_tail(counts0: torch.Tensor, T: int, method: str) -> torch.Tensor:
                c = counts0.clone()
                for _ in range(T):
                    if method == "sl":
                        pr = self._sainte_lague_priority(w, c)
                    else:
                        pr = self._huntington_hill_priority(w, c)
                    avg_cur = w / c.to(w.dtype)
                    j = self._argmax_with_tie(pr, avg_cur)
                    c[j] += 1
                return c

            c_sl = sim_tail(counts, T, "sl")
            c_hh = sim_tail(counts, T, "hh")

            def eval_counts(c: torch.Tensor) -> Tuple[float, float]:
                avg = w / c.to(w.dtype)
                # peak and second-highest for tie-break
                top2 = torch.topk(avg, k=min(2, avg.numel()), largest=True)
                peak = float(top2.values[0].item())
                second = float(top2.values[1].item()) if top2.values.numel() > 1 else -float("inf")
                return peak, second

            peak_sl, sec_sl = eval_counts(c_sl)
            peak_hh, sec_hh = eval_counts(c_hh)
            if (peak_sl + 1e-12 < peak_hh) or (abs(peak_sl - peak_hh) <= 1e-12 and sec_sl + 1e-12 < sec_hh):
                counts = c_sl
            else:
                counts = c_hh

        # Guarded single donor→receiver fix-up
        if n > 1:
            c_safe = torch.clamp(counts, min=1)
            avg = w / c_safe.to(w.dtype)
            cur_peak = float(avg.max().item())

            can_donate = (counts > 1)
            if bool(can_donate.any()):
                kd = int(min(2, int(can_donate.sum().item())))
                kr = int(min(2, n))
                avg_mask = avg.clone()
                avg_mask[~can_donate] = float("-inf")
                donors = torch.topk(avg_mask, k=kd).indices.tolist()
                receivers = torch.topk(-avg, k=kr).indices.tolist()
                best = None
                best_peak = cur_peak
                best_second = float("inf")
                for d in donors:
                    for r in receivers:
                        if d == r:
                            continue
                        c_try = counts.clone()
                        c_try[d] -= 1
                        c_try[r] += 1
                        avg_try = w / c_try.to(w.dtype)
                        top2 = torch.topk(avg_try, k=min(2, avg_try.numel()))
                        peak = float(top2.values[0].item())
                        second = float(top2.values[1].item()) if top2.values.numel() > 1 else -float("inf")
                        if (peak + 1e-12 < best_peak) or (abs(peak - best_peak) <= 1e-12 and second + 1e-12 < best_second):
                            best_peak = peak
                            best_second = second
                            best = (d, r)
                if best is not None and best_peak + 1e-12 < cur_peak:
                    d, r = best
                    counts[d] -= 1
                    counts[r] += 1

        return counts

    def allocate(self, weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weight: [X, num_log]
        Returns (phy2log, rank, logcnt)
        """
        n, num_log = weight.shape
        assert num_phy >= num_log
        device = weight.device
        exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)

        phy2log_list: List[torch.Tensor] = []
        rank_list: List[torch.Tensor] = []
        logcnt_list: List[torch.Tensor] = []

        for i in range(n):
            w = weight[i]
            counts = self._allocate_row(w, num_phy)
            logcnt_list.append(counts)
            # Build contiguous blocks per logical expert
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


class CapacityPacker:
    """
    Diversity-aware greedy packer with broadened local refinement:
      - Greedy: minimize (repeat_count, load, count).
      - Refinement: best global 1×1 swap among k-pack partners; chained two-swap
        fallback when imbalance is high and single swap can’t help enough.
    """

    def __init__(self):
        pass

    @staticmethod
    def _all_unique(labels_row: torch.Tensor) -> bool:
        # fast check via sort + unique_consecutive
        ls, _ = torch.sort(labels_row)
        return torch.unique_consecutive(ls).numel() == labels_row.numel()

    @staticmethod
    def _refine_single_layer(row_w: torch.Tensor,
                             pack_groups: List[List[int]],
                             loads: List[float],
                             k_donors: int = 2,
                             k_receivers: int = 3) -> bool:
        """
        Broadened partner search to donors from {heaviest, 2nd-heaviest}
        and receivers from {lightest, 2nd, 3rd}. Compute best global 1×1 swap
        using nearest-difference search. Strictly reduce global max only.
        """
        m = len(loads)
        if m < 2:
            return False
        order_asc = sorted(range(m), key=lambda p: loads[p])
        order_desc = list(reversed(order_asc))
        donors = [p for p in order_desc[:k_donors] if pack_groups[p]]
        receivers = [p for p in order_asc[:k_receivers] if pack_groups[p]]
        if not donors or not receivers:
            return False

        cur_max = max(loads)
        other_max_cache = {}
        other_min_cache = {}

        best = None  # (new_peak, donor, recv, a_idx, b_idx, a_item, b_item, wa, wb)
        for d in donors:
            # Prepare donor items sorted descending
            d_idx_tensor = torch.tensor(pack_groups[d], dtype=torch.int64, device=row_w.device)
            d_w = row_w[d_idx_tensor]
            d_sorted = torch.sort(d_w, descending=True)
            d_sorted_vals = d_sorted.values
            d_sorted_idx = d_sorted.indices  # indices into d_idx_tensor

            for r in receivers:
                if d == r:
                    continue
                # skip if donor not heavier than receiver pack
                if loads[d] <= loads[r]:
                    continue
                # Prepare receiver items sorted ascending
                r_idx_tensor = torch.tensor(pack_groups[r], dtype=torch.int64, device=row_w.device)
                r_w = row_w[r_idx_tensor]
                r_sorted = torch.sort(r_w, descending=False)
                r_vals = r_sorted.values
                r_indices = r_sorted.indices

                # Cache other max/min
                key = (d, r)
                if key not in other_max_cache:
                    other_max_cache[key] = max([loads[p] for p in range(m) if p != d and p != r], default=float("-inf"))
                    other_min_cache[key] = min([loads[p] for p in range(m) if p != d and p != r], default=float("inf"))
                omax = other_max_cache[key]
                omin = other_min_cache[key]

                # target delta
                t = (loads[d] - loads[r]) * 0.5
                # For each donor item wa, we want wb ~ wa - t. Do nearest search.
                # r_vals is ascending; use searchsorted
                for j in range(min(2, d_sorted_vals.numel())):
                    wa = float(d_sorted_vals[j].item())
                    a_local_idx = int(d_sorted_idx[j].item())
                    a_item = int(d_idx_tensor[a_local_idx].item())
                    target = wa - t
                    pos = torch.searchsorted(r_vals, torch.tensor(target, device=row_w.device))
                    cand_positions = []
                    if pos.numel() == 0:
                        continue
                    p0 = int(pos.item())
                    if p0 < r_vals.numel():
                        cand_positions.append(p0)
                    if p0 - 1 >= 0:
                        cand_positions.append(p0 - 1)
                    for pidx in cand_positions:
                        wb = float(r_vals[pidx].item())
                        b_local_idx = int(r_indices[pidx].item())
                        b_item = int(r_idx_tensor[b_local_idx].item())
                        new_d = loads[d] - wa + wb
                        new_r = loads[r] - wb + wa
                        new_peak = max(new_d, new_r, omax)
                        new_bottom = min(new_d, new_r, omin)
                        # Strict improvement on global max
                        if new_peak + 1e-12 < cur_max:
                            if best is None or new_peak + 1e-12 < best[0]:
                                best = (new_peak, d, r, a_local_idx, b_local_idx, a_item, b_item, wa, wb)

        if best is None:
            return False
        # Apply best swap
        _, d, r, a_local_idx, b_local_idx, a_item, b_item, wa, wb = best
        loads[d] = loads[d] - wa + wb
        loads[r] = loads[r] - wb + wa
        pack_groups[d][a_local_idx] = b_item
        pack_groups[r][b_local_idx] = a_item
        return True

    @staticmethod
    def _apply_swap_and_update(loads, pack_groups, row_w, d, r, a_tuple, b_tuple):
        # helper for chained swaps
        ai, bi, a_item, b_item, wa, wb = a_tuple + b_tuple
        loads[d] = loads[d] - wa + wb
        loads[r] = loads[r] - wb + wa
        pack_groups[d][ai] = b_item
        pack_groups[r][bi] = a_item

    def pack(self, weights: torch.Tensor, labels: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        weights: [X, n], labels: [X, n], num_packs divides n.
        Returns (pack_index, rank_in_pack)
        """
        X, n = weights.shape
        assert n % num_packs == 0
        cap = n // num_packs
        device = weights.device

        if cap == 1 or num_packs == 1:
            idx = torch.arange(n, dtype=torch.int64, device=device)
            pack_index = (idx % num_packs).expand(X, n).clone()
            rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
            return pack_index, rank_in_pack

        pack_index = torch.full((X, n), -1, dtype=torch.int64, device=device)
        rank_in_pack = torch.full_like(pack_index, -1)
        sorted_idx_all = weights.sort(dim=-1, descending=True).indices

        for i in range(X):
            row_w = weights[i]
            row_lbl = labels[i]
            sorted_idx = sorted_idx_all[i].tolist()

            loads = [0.0] * num_packs
            counts = [0] * num_packs
            pack_groups: List[List[int]] = [[] for _ in range(num_packs)]

            if self._all_unique(row_lbl):
                # Fast path: projected load greedy, tie by fewer items
                for g in sorted_idx:
                    wv = float(row_w[g].item())
                    best_p = None
                    best_key = None  # (proj_load, count)
                    for p in range(num_packs):
                        if counts[p] >= cap:
                            continue
                        key = (loads[p] + wv, counts[p])
                        if best_key is None or key < best_key:
                            best_key = key
                            best_p = p
                    pack_index[i, g] = best_p
                    rank_in_pack[i, g] = counts[best_p]
                    counts[best_p] += 1
                    loads[best_p] += wv
                    pack_groups[best_p].append(g)

                # Tiny refinement 1×1 between heaviest and lightest
                if num_packs >= 2:
                    h = max(range(num_packs), key=lambda p: loads[p])
                    l = min(range(num_packs), key=lambda p: loads[p])
                    if loads[h] > loads[l] and pack_groups[h] and pack_groups[l]:
                        h_idx = torch.tensor(pack_groups[h], dtype=torch.int64, device=device)
                        l_idx = torch.tensor(pack_groups[l], dtype=torch.int64, device=device)
                        h_w = row_w[h_idx]
                        l_w = row_w[l_idx]
                        kh = min(2, h_w.numel())
                        kl = min(2, l_w.numel())
                        if kh > 0 and kl > 0:
                            top_h = torch.topk(h_w, kh).indices.tolist()
                            bot_l = torch.topk(l_w, kl, largest=False).indices.tolist()
                            cur_max = max(loads)
                            best = None
                            best_peak = cur_max
                            for ai in top_h:
                                wa = float(h_w[ai].item())
                                a_item = int(h_idx[ai].item())
                                for bi in bot_l:
                                    wb = float(l_w[bi].item())
                                    b_item = int(l_idx[bi].item())
                                    new_h = loads[h] - wa + wb
                                    new_l = loads[l] - wb + wa
                                    new_peak = max(new_h, new_l, max([loads[p] for p in range(num_packs) if p != h and p != l], default=float("-inf")))
                                    if new_peak + 1e-12 < best_peak:
                                        best_peak = new_peak
                                        best = (ai, bi, a_item, b_item, wa, wb)
                            if best is not None:
                                ai, bi, a_item, b_item, wa, wb = best
                                loads[h] = loads[h] - wa + wb
                                loads[l] = loads[l] - wb + wa
                                pack_groups[h][ai] = b_item
                                pack_groups[l][bi] = a_item
                                pack_index[i, a_item] = l
                                pack_index[i, b_item] = h
                                # update ranks of affected packs
                                for r, g in enumerate(pack_groups[h]):
                                    rank_in_pack[i, g] = r
                                for r, g in enumerate(pack_groups[l]):
                                    rank_in_pack[i, g] = r

                continue  # next row

            # General path: diversity-aware greedy
            label_counts = [defaultdict(int) for _ in range(num_packs)]
            for g in sorted_idx:
                lab = int(row_lbl[g].item())
                wv = float(row_w[g].item())
                best_p = None
                best_key = None  # (repeats, load, count)
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

            # Refinement: broadened 1×1 swap
            changed = self._refine_single_layer(row_w, pack_groups, loads, k_donors=2, k_receivers=3)

            # Bounded chained two-swap fallback if pronounced imbalance and no 1×1 helped
            mean_load = sum(loads) / max(1, num_packs)
            imbalance = (max(loads) - min(loads)) / max(mean_load, 1e-12) if mean_load > 0 else 0.0
            if not changed and imbalance > 0.10:
                # Attempt first swap
                changed = self._refine_single_layer(row_w, pack_groups, loads, k_donors=2, k_receivers=3)
                if changed:
                    # Attempt second swap once
                    self._refine_single_layer(row_w, pack_groups, loads, k_donors=2, k_receivers=3)

            # finalize indices and ranks
            for p in range(num_packs):
                for r, g in enumerate(pack_groups[p]):
                    pack_index[i, g] = p
                    rank_in_pack[i, g] = r

        return pack_index, rank_in_pack


def _replicate_and_pack_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hierarchical planner:
      - Pack logical groups to nodes
      - Allocate replicas within nodes
      - Place physical experts onto GPUs within nodes
    """
    num_layers, num_logical = weight.shape
    assert num_logical % num_groups == 0
    group_size = num_logical // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    phy_per_gpu = num_physical_experts // num_gpus

    # Step 1: groups -> nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _balanced_packing_lpt(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = _inverse(log2mlog)

    # Step 2: replica allocation per (layer,node)
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical // num_nodes)
    allocator = ReplicaAllocator(tail_frac=0.10)
    phy2mlog, phyrank, mlogcnt = allocator.allocate(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: place physical experts onto GPUs within nodes
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    packer = CapacityPacker()
    pack_index, rank_in_pack = packer.pack(tokens_per_phy, phy2mlog, gpus_per_node)
    phy2pphy = pack_index * phy_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    # convert back to global logical indexing
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(0, num_logical, num_logical // num_nodes, device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts]
        num_replicas: number of physical experts, multiple of num_gpus
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: number of GPUs, multiple of num_nodes

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = _replicate_and_pack_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # treat as single node for grouping
        phy2log, phyrank, logcnt = _replicate_and_pack_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

    num_redundant = num_replicas - num_logical
    maxlogcnt = num_redundant + 1
    log2phy = torch.full(
        (num_layers, num_logical, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # scatter physical indices into logical->physical map
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