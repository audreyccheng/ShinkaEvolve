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


def _stripe_pack_single_row(w: torch.Tensor, num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zig-zag stripe packing for a single row:
      - Sort indices by descending weight.
      - Partition into capacity layers of size num_packs.
      - Assign layer k in ascending pack order if k even, descending if k odd.
    Returns pack_index_row, rank_in_pack_row (both shape [N], int64).
    """
    N = w.numel()
    assert N % num_packs == 0
    capacity = N // num_packs
    device = w.device
    dtype_i64 = torch.int64

    if capacity == 1:
        # identity mapping: item i -> pack i
        pack_index = torch.arange(N, dtype=dtype_i64, device=device)
        rank_in_pack = torch.zeros(N, dtype=dtype_i64, device=device)
        return pack_index, rank_in_pack

    order = torch.argsort(w, descending=True)
    pack_index = torch.empty(N, dtype=dtype_i64, device=device)
    rank_in_pack = torch.empty(N, dtype=dtype_i64, device=device)

    for k in range(capacity):
        start = k * num_packs
        layer = order[start:start + num_packs]
        if k % 2 == 1:
            layer = torch.flip(layer, dims=[0])  # reverse for zig-zag
        # assign: j-th element in layer -> pack j, rank k
        # layer[j] is original index of item
        # packs go 0..num_packs-1
        pack_ids = torch.arange(num_packs, dtype=dtype_i64, device=device)
        pack_index[layer] = pack_ids
        rank_in_pack[layer] = k

    return pack_index, rank_in_pack


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     refine_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced packing using zig-zag stripe assignment with exact capacity per pack.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        refine_steps: kept for API compatibility (ignored)

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack (0..n/m-1)
    """
    L, N = weight.shape
    assert N % num_packs == 0, "Number of items must be divisible by num_packs."

    # operate on CPU or current device directly (weight already moved to CPU by caller)
    pack_index = torch.empty((L, N), dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.empty((L, N), dtype=torch.int64, device=weight.device)

    # Stripe pack each row independently
    for li in range(L):
        pidx, rnk = _stripe_pack_single_row(weight[li], num_packs)
        pack_index[li] = pidx
        rank_in_pack[li] = rnk

    return pack_index, rank_in_pack


def _waterfill_counts_single_row(w: torch.Tensor, total: int) -> torch.Tensor:
    """
    Water-filling replica allocation for a single row.
    Given weights w: [M] and total replicas 'total' (>= M), find integer counts c >= 1
    that minimize the maximum average load max_i w_i / c_i via:
      - Solve continuous r_i = max(1, w_i / T) with sum r_i = total by binary search on T.
      - Floor r_i to integers and distribute remaining replicas by largest marginal benefit
        Δ_i = w_i/floor_i − w_i/(floor_i + 1).

    Returns c: [M] int64 with sum(c) == total.
    """
    M = w.numel()
    assert total >= M, "Total replicas must be at least number of logical experts."
    device = w.device
    dtype_f = w.dtype
    dtype_i64 = torch.int64

    if total == M:
        return torch.ones(M, dtype=dtype_i64, device=device)

    max_w = torch.max(w).item() if M > 0 else 0.0
    # Binary search for T in (0, max_w], ensure S(lo) >= total and S(hi) <= total
    lo = 0.0
    hi = max(max_w, 1.0)
    # Iterate fixed steps for robustness
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mid_safe = max(mid, 1e-12)
        r = torch.maximum(torch.ones(M, dtype=dtype_f, device=device), w / mid_safe)
        S = float(r.sum().item())
        if S > total:
            lo = mid
        else:
            hi = mid
    T = hi
    rstar = torch.maximum(torch.ones(M, dtype=dtype_f, device=device), w / max(T, 1e-12))
    floor_c = torch.floor(rstar).to(dtype_i64)
    # Ensure at least 1
    floor_c = torch.clamp(floor_c, min=1)

    used = int(floor_c.sum().item())
    remain = total - used
    if remain > 0:
        # Marginal peak reduction by incrementing count by +1
        floor_f = floor_c.to(dtype_f)
        benefit = w / floor_f - w / (floor_f + 1.0)
        # In rare cases where benefit has ties or zeros, topk is deterministic enough
        k = min(remain, M)
        # Use topk selection and then add 1 to those indices; if remain > M we do multiple passes
        # Handle remain possibly > M by repeating full increments
        full_rounds, last = divmod(remain, M)
        if full_rounds > 0:
            floor_c += full_rounds
        if last > 0:
            top_vals, top_idx = torch.topk(benefit, k=last, largest=True)
            floor_c[top_idx] += 1
    # No case for remain < 0 should occur due to floor <= rstar and sum(rstar)=total
    return floor_c


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized (water-filling + optimal rounding).

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

    # Base arrays
    phy2log = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((n, num_phy), dtype=dtype_i64, device=device)
    logcnt = torch.empty((n, num_log), dtype=dtype_i64, device=device)

    if num_extra == 0:
        # One-to-one
        base = torch.arange(num_log, dtype=dtype_i64, device=device).unsqueeze(0).expand(n, -1)
        phy2log[:, :num_log] = base
        rank[:, :num_log] = 0
        logcnt.fill_(1)
        return phy2log, rank, logcnt

    # Compute counts per row via water-filling
    for ri in range(n):
        wrow = weight[ri]
        cnt = _waterfill_counts_single_row(wrow, num_phy)
        logcnt[ri] = cnt

        # Build mapping columns deterministically: enumerate experts by id and emit cnt copies
        cols = []
        ranks = []
        for eid in range(num_log):
            c = int(cnt[eid].item())
            if c <= 0:
                continue
            cols.extend([eid] * c)
            ranks.extend(list(range(c)))
        cols_t = torch.tensor(cols, dtype=dtype_i64, device=device)
        ranks_t = torch.tensor(ranks, dtype=dtype_i64, device=device)
        # Ensure exact length num_phy
        assert cols_t.numel() == num_phy
        phy2log[ri] = cols_t
        rank[ri] = ranks_t

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

    # Step 1: pack groups to nodes with stripe packing on total tokens per group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes, refine_steps=0)

    # Compute permutation logical -> meta-logical (contiguous within node)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = PermOps.inverse(log2mlog)

    # Step 2: replicate experts within nodes based on local loads (water-filling)
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs (within nodes) using stripe packing
    # Build per-replica average load tensor
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes, refine_steps=0)
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
        # use global load-balance policy (treat as a single group across nodes)
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