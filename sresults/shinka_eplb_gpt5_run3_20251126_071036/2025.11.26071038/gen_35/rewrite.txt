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


class CapacityPacker:
    """
    Capacity-constrained packer:
      - Greedy assign sorted items to the currently lightest pack with remaining capacity.
      - Optional small local refinement with targeted swaps between the heaviest
        and lightest packs to reduce imbalance.
    """

    def __init__(self, refine_steps: int = 1):
        self.refine_steps = int(refine_steps)

    @staticmethod
    def _rank_from_packidx(pack_idx: torch.Tensor, num_packs: int) -> torch.Tensor:
        """
        Given pack_idx: [N], compute rank_in_pack: [N] deterministically
        by ascending original item id within each pack.
        """
        N = pack_idx.numel()
        device = pack_idx.device
        dtype_i64 = torch.int64
        ranks = torch.empty(N, dtype=dtype_i64, device=device)
        for p in range(num_packs):
            ids = torch.nonzero(pack_idx == p, as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            ids_sorted = torch.sort(ids).values
            ranks[ids_sorted] = torch.arange(ids_sorted.numel(), dtype=dtype_i64, device=device)
        return ranks

    @staticmethod
    def _pack_loads(weights: torch.Tensor, pack_idx: torch.Tensor, num_packs: int) -> torch.Tensor:
        """
        Compute per-pack loads given weights [N] and pack_idx [N].
        Returns pack_sums [num_packs].
        """
        pack_w = torch.zeros(num_packs, dtype=weights.dtype, device=weights.device)
        pack_w.scatter_add_(0, pack_idx, weights)
        return pack_w

    def _refine_single_layer(
        self,
        weights: torch.Tensor,
        pack_idx: torch.Tensor,
        num_packs: int,
        capacity: int,
    ) -> torch.Tensor:
        """
        Targeted refinement by swapping one item between heaviest and lightest packs
        per iteration (up to self.refine_steps). Uses a searchsorted-based best-swap
        selection for improved reduction of max imbalance.
        """
        if self.refine_steps <= 0:
            return pack_idx

        # Compute pack loads
        pack_w = self._pack_loads(weights, pack_idx, num_packs)

        for _ in range(self.refine_steps):
            h = int(torch.argmax(pack_w).item())
            l = int(torch.argmin(pack_w).item())
            if h == l:
                break
            delta = float(pack_w[h] - pack_w[l])
            if delta <= 1e-9:
                break

            heavy_idx = torch.nonzero(pack_idx == h, as_tuple=False).squeeze(1)
            light_idx = torch.nonzero(pack_idx == l, as_tuple=False).squeeze(1)
            if heavy_idx.numel() == 0 or light_idx.numel() == 0:
                break

            w = weights
            hw = w[heavy_idx]
            lw = w[light_idx]
            lw_sorted, lw_perm = torch.sort(lw)  # ascending

            if lw_sorted.numel() == 0 or hw.numel() == 0:
                break

            # For each heavy item, find light item closest to target = hw - delta/2
            target = hw - (delta / 2.0)
            pos = torch.searchsorted(lw_sorted, target)
            pos = torch.clamp(pos, 0, lw_sorted.numel() - 1)
            cand_pos = torch.stack(
                [pos, torch.clamp(pos - 1, 0, lw_sorted.numel() - 1)], dim=1
            )  # [H, 2]
            cand_lw = lw_sorted[cand_pos]  # [H, 2]
            resid = (delta - 2.0 * (hw.unsqueeze(1) - cand_lw)).abs()
            best_flat = int(torch.argmin(resid).item())
            best_h_index = best_flat // 2
            best_option = best_flat % 2
            j_sorted_idx = int(cand_pos[best_h_index, best_option].item())

            wi = float(hw[best_h_index].item())
            wj = float(lw_sorted[j_sorted_idx].item())
            new_delta = abs(delta - 2.0 * (wi - wj))
            if new_delta < delta - 1e-9:
                hi = heavy_idx[best_h_index]
                lj = light_idx[lw_perm[j_sorted_idx]]
                pack_idx[hi] = l
                pack_idx[lj] = h
                # Update loads incrementally
                pack_w[h] = pack_w[h] - wi + wj
                pack_w[l] = pack_w[l] - wj + wi
            else:
                break

        return pack_idx

    def pack(self, weight: torch.Tensor, num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack n weighted objects to m packs, such that each bin contains exactly
        n/m objects and the weights of all packs are as balanced as possible.

        Parameters:
            weight: [L, N], weights
            num_packs: M, number of packs

        Returns:
            pack_index: [L, N], pack id in [0, M)
            rank_in_pack: [L, N], rank in each pack [0, N/M)
        """
        L, N = weight.shape
        assert N % num_packs == 0
        capacity = N // num_packs
        device = weight.device
        dtype_i64 = torch.int64

        # Fast path if capacity == 1 (each item is its own pack)
        if capacity == 1:
            pack_index = torch.arange(N, dtype=dtype_i64, device=device).expand(L, -1).contiguous()
            rank_in_pack = torch.zeros((L, N), dtype=dtype_i64, device=device)
            return pack_index, rank_in_pack

        pack_index = torch.empty((L, N), dtype=dtype_i64, device=device)
        rank_in_pack = torch.empty((L, N), dtype=dtype_i64, device=device)

        for li in range(L):
            w = weight[li]
            order = torch.argsort(w, descending=True)
            load = [0.0] * num_packs
            counts = [0] * num_packs
            pidx = torch.empty(N, dtype=dtype_i64, device=device)

            # Greedy placement to the lightest pack with remaining capacity
            for g in order.tolist():
                best_pack = None
                best_load = None
                for p in range(num_packs):
                    if counts[p] < capacity:
                        pl = load[p]
                        if best_load is None or pl < best_load:
                            best_pack = p
                            best_load = pl
                pidx[g] = best_pack
                load[best_pack] += float(w[g].item())
                counts[best_pack] += 1

            # Optional small refinement
            pidx = self._refine_single_layer(w, pidx, num_packs, capacity)

            # Deterministic ranks within each pack
            rnk = self._rank_from_packidx(pidx, num_packs)
            pack_index[li] = pidx
            rank_in_pack[li] = rnk

        return pack_index, rank_in_pack


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
    packer = CapacityPacker(refine_steps=refine_steps)
    return packer.pack(weight, num_packs)


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
      - Bounded refinement afterwards (same best-swap routine), with adaptive steps based on imbalance.

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
        lam = 1e-8 * max(mean_w, 1e-12)

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

        # compute imbalance ratio and adapt refinement steps
        pack_w = torch.zeros(num_packs, dtype=w.dtype)
        pack_w.scatter_add_(0, pidx, w)
        delta = float((pack_w.max() - pack_w.min()).item())
        mean_ld = float(pack_w.mean().item())
        ratio = delta / max(mean_ld, 1e-12)
        steps = 3 if ratio > 0.12 else refine_steps_default

        # reuse CapacityPacker refinement
        pidx = CapacityPacker(refine_steps=int(steps))._refine_single_layer(w, pidx, num_packs, capacity)

        # ranks
        rnk = CapacityPacker._rank_from_packidx(pidx, num_packs)
        pack_index[li] = pidx
        rank_in_pack[li] = rnk

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
    for _ in range(bulk):
        benefit = weight / logcnt
        best = benefit.max(dim=-1).indices
        phy2log[:, col] = best
        rank[:, col] = logcnt[arangen, best]
        logcnt[arangen, best] += 1
        col += 1

    # Tail phase (Sainte-Laguë): benefit = weight / (2r - 1)
    if tail > 0:
        for _ in range(tail):
            denom = (2 * logcnt - 1).to(weight.dtype)
            benefit = weight / denom
            best = benefit.max(dim=-1).indices
            phy2log[:, col] = best
            rank[:, col] = logcnt[arangen, best]
            logcnt[arangen, best] += 1
            col += 1

    # Strengthened replication fix-up per row:
    # Evaluate moves from top-2 donors (by avg load) to bottom-2 receivers and
    # apply the single best move if it strictly reduces the peak.
    if num_log > 1 and num_redundant > 0:
        avg = weight / logcnt.to(weight.dtype)  # [n, num_log]
        kdon = min(2, num_log)
        krec = min(2, num_log)
        top2_vals, top2_idx = torch.topk(avg, k=kdon, dim=-1, largest=True)
        bot2_vals, bot2_idx = torch.topk(avg, k=krec, dim=-1, largest=False)
        cur_max = avg.max(dim=-1).values
        argmax_idx = avg.argmax(dim=-1)

        rows = torch.arange(n, dtype=torch.int64, device=device)
        for ri in rows.tolist():
            best_new_peak = None
            best_pair = None

            donors = top2_idx[ri].tolist()
            receivers = bot2_idx[ri].tolist()

            for d in donors:
                cd = int(logcnt[ri, d].item())
                if cd <= 1:
                    continue
                for r in receivers:
                    if d == r:
                        continue
                    cr = int(logcnt[ri, r].item())

                    # Baseline peak ignoring donor if donor is current max
                    baseline_other = float(cur_max[ri].item())
                    if d == int(argmax_idx[ri].item()):
                        # second-best under current configuration
                        baseline_other = float(torch.topk(avg[ri], k=2, largest=True).values[1].item())

                    new_d = float(weight[ri, d].item()) / float(cd - 1)
                    new_r = float(weight[ri, r].item()) / float(cr + 1)
                    candidate_peak = max(baseline_other, new_d, new_r)

                    if candidate_peak + 1e-12 < float(cur_max[ri].item()):
                        if best_new_peak is None or candidate_peak < best_new_peak:
                            best_new_peak = candidate_peak
                            best_pair = (d, r)

            if best_pair is not None:
                d, r = best_pair
                # Choose a physical column corresponding to donor's highest rank (prefer the last replica)
                donor_cols = torch.nonzero(phy2log[ri] == d, as_tuple=False).squeeze(1)
                if donor_cols.numel() == 0:
                    continue
                maxr_idx = torch.argmax(rank[ri, donor_cols]).item()
                col_idx = donor_cols[maxr_idx]

                # Assign this physical replica to receiver with new rank equal to current receiver count
                new_rank = int(logcnt[ri, r].item())
                phy2log[ri, col_idx] = r
                rank[ri, col_idx] = new_rank

                # Update counts
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

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes, refine_steps=1)

    # Compute permutation logical -> meta-logical (within node contiguous layout)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = PermOps.inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (still within nodes), with diversity-aware tie-breaker
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    labels_per_phy = phy2mlog  # prefer spreading same meta-logical ids across GPUs under near ties
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