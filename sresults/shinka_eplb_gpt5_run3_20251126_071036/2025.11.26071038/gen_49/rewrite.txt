# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

Completely new approach:
  - Minimax-threshold replica allocation via binary search for optimal counts.
  - Heap-based fixed-capacity LPT packer for group->node and replica->GPU.
  - Deterministic, CPU-only and lightweight control flow.

Inputs/outputs remain identical to the original implementation.
"""

import torch
import math
import heapq


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


def _fixed_capacity_heap_pack_row(weights_row: torch.Tensor, num_bins: int, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a single row of weights into 'num_bins' bins, each with exact capacity 'capacity',
    minimizing imbalance greedily:
      - Sort items descending by weight.
      - Assign each item to the currently lightest bin with remaining capacity (min-heap).
    Deterministic tie-breaking by bin id.

    weights_row: [N] on CPU
    returns: (pack_idx_row [N], rank_in_bin_row [N]) int64
    """
    N = weights_row.numel()
    pack_idx_row = torch.empty(N, dtype=torch.int64)
    rank_row = torch.empty(N, dtype=torch.int64)

    # Initialize heap with (load, bin_id, count, next_rank)
    heap = [(0.0, b, 0, 0) for b in range(num_bins)]
    heapq.heapify(heap)

    # Sort items by descending weight
    order = torch.argsort(weights_row, descending=True)

    for g in order.tolist():
        # Pop the currently lightest bin with available capacity
        while True:
            if not heap:
                raise RuntimeError("Heap empty before assigning all items.")
            load, b, cnt, nxt = heapq.heappop(heap)
            if cnt < capacity:
                # Assign here
                pack_idx_row[g] = b
                rank_row[g] = nxt
                load_new = load + float(weights_row[g].item())
                cnt_new = cnt + 1
                nxt_new = nxt + 1
                # If bin still has room, push back; otherwise bin is saturated and removed
                if cnt_new < capacity:
                    heapq.heappush(heap, (load_new, b, cnt_new, nxt_new))
                # If full capacity reached, we don't push it back
                break
            # If bin is saturated, do not push back and keep popping
    return pack_idx_row, rank_row


def _fixed_capacity_heap_pack(weights: torch.Tensor, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-row version of heap-based fixed-capacity packer.

    weights: [L, N] CPU or CUDA
    num_bins: M
    Returns:
        pack_index: [L, N] int64
        rank_in_pack: [L, N] int64
    """
    L, N = weights.shape
    assert N % num_bins == 0, "N must be divisible by num_bins."
    capacity = N // num_bins
    if capacity == 1:
        pack_index = torch.arange(N, dtype=torch.int64, device=weights.device).expand(L, -1).contiguous()
        rank_in_pack = torch.zeros((L, N), dtype=torch.int64, device=weights.device)
        return pack_index, rank_in_pack

    w_cpu = weights.float().cpu()
    pack_index = torch.empty((L, N), dtype=torch.int64)
    rank_in_pack = torch.empty((L, N), dtype=torch.int64)
    for li in range(L):
        pidx_row, rnk_row = _fixed_capacity_heap_pack_row(w_cpu[li], num_bins, capacity)
        pack_index[li] = pidx_row
        rank_in_pack[li] = rnk_row
    return pack_index.to(weights.device), rank_in_pack.to(weights.device)


def _group_to_node_mapping(weight: torch.Tensor, num_groups: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assign expert groups to nodes with fixed capacity (groups_per_node) using heap-based LPT.

    weight: [L, num_logical_experts]
    Returns:
        group_pack_index: [L, num_groups]
        group_rank_in_pack: [L, num_groups]
    """
    L, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, num_groups]
    group_pack_index, group_rank_in_pack = _fixed_capacity_heap_pack(tokens_per_group, num_nodes)
    return group_pack_index, group_rank_in_pack


def _minimax_replica_counts_row(b: torch.Tensor, R: int) -> torch.Tensor:
    """
    Given loads b (size N, float) and total replicas R (>= N), find integer counts r >= 1 minimizing
    max_i b_i / r_i exactly via monotone thresholding:
        r_i(T) = max(1, ceil(b_i / T))
    Choose the smallest T with sum_i r_i(T) <= R by binary search on T.

    If sum < R after thresholding due to discreteness, allocate the remaining M = R - sum
    replicas by repeatedly giving one replica to the expert with the largest current average b_i / r_i.
    Returns r: int64 shape [N].
    """
    N = b.numel()
    assert R >= N, "Total replicas must be at least number of experts (>=1 each)."
    # Edge: if all b == 0, just distribute evenly with at least 1 each
    if torch.all(b <= 0):
        r = torch.ones(N, dtype=torch.int64)
        # Distribute remaining evenly by index to keep deterministic
        M = R - N
        if M > 0:
            base = (M // N)
            rem = (M % N)
            r += base
            if rem > 0:
                # first rem indices get +1
                r[:rem] += 1
        return r

    # Binary search for T in (0, max(b)]
    bmax = float(torch.max(b).item())
    lo = 0.0
    hi = bmax
    # Guard against zeros
    def sum_counts_at(T: float) -> int:
        # r_i = max(1, ceil(b_i/T))
        if T <= 0.0:
            return 10**18  # force high
        tmp = torch.ceil(b / T)
        tmp = torch.clamp(tmp, min=1.0)
        return int(tmp.sum().item())

    # hi is definitely feasible (sum <= R) because ceil(b_i/hi) <= 1 for positive b_i, and we clamp min 1 => sum == N
    # lo is infeasible (sum huge). 40 iters for robust precision.
    for _ in range(40):
        mid = (lo + hi) / 2.0
        s = sum_counts_at(mid)
        if s > R:
            lo = mid
        else:
            hi = mid

    # Build counts at T* = hi
    Tstar = hi if hi > 0.0 else bmax
    counts = torch.ceil(b / Tstar).clamp(min=1.0).to(torch.int64)
    s = int(counts.sum().item())
    M = R - s
    if M <= 0:
        return counts

    # Greedy give remainder to largest current average b_i / r_i
    # To stay deterministic, break ties by smaller index (torch.argmax already deterministic on CPU).
    # We'll update counts in-place and recompute averages each step (N is typically small enough).
    for _ in range(M):
        avg = b / counts.to(b.dtype)
        idx = int(torch.argmax(avg).item())
        counts[idx] += 1

    return counts


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, minimizing the maximum
    per-replica load exactly via minimax-threshold allocation.

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
    if num_extra == 0:
        # Base one-to-one mapping
        device = weight.device
        phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(L, 1)
        rank = torch.zeros(L, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(L, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    device = weight.device
    dtype_i64 = torch.int64
    dtype_f = weight.dtype

    phy2log = torch.empty((L, num_phy), dtype=dtype_i64, device=device)
    rank = torch.empty((L, num_phy), dtype=dtype_i64, device=device)
    logcnt = torch.empty((L, num_log), dtype=dtype_i64, device=device)

    w_cpu = weight.float().cpu()
    for li in range(L):
        b = w_cpu[li]  # [num_log]
        R = num_phy
        counts = _minimax_replica_counts_row(b, R)  # int64 CPU

        # Build mapping deterministically by ascending logical id
        col = 0
        for e in range(num_log):
            c = int(counts[e].item())
            if c <= 0:
                # should not happen due to >=1 clamp
                continue
            phy2log[li, col:col + c] = e
            # ranks 0..c-1
            rank[li, col:col + c] = torch.arange(c, dtype=dtype_i64)
            col += c

        if col != num_phy:
            # This should never happen; safeguard to avoid uninitialized values
            # Fill any remaining with last expert id
            last_e = max(0, num_log - 1)
            while col < num_phy:
                phy2log[li, col] = last_e
                rank[li, col] = int(counts[last_e].item())
                counts[last_e] += 1
                col += 1

        logcnt[li] = counts.to(dtype_i64, device=device)

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical policy:
      1) Assign groups -> nodes by heap-LPT with fixed capacity.
      2) Within each node, compute optimal replica counts via minimax-threshold.
      3) Assign replicas -> GPUs within each node by heap-LPT with fixed capacity.
    """
    L, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    gpus_per_node = num_gpus // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus
    phy_per_node = num_physical_experts // num_nodes

    # Step 1: pack groups to nodes (fixed capacity heap-based)
    group_pack_index, group_rank_in_pack = _group_to_node_mapping(weight, num_groups, num_nodes)

    # Compute permutation logical -> meta-logical (node-contiguous layout)
    device = weight.device
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=device)).flatten(-2)
    mlog2log = PermOps.inverse(log2mlog)

    # Step 2: replicate experts within nodes optimally (minimax) based on local loads
    # tokens_per_mlog: [L * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, phy_per_node)

    # Step 3: assign physical replicas to GPUs within nodes (fixed capacity heap-based)
    # Per replica average load = tokens_per_mlog / mlogcnt, then gather by phy2mlog
    avg_per_mlog = tokens_per_mlog / mlogcnt.clamp_min(1)
    tokens_per_phy = avg_per_mlog.gather(-1, phy2mlog)  # [L*num_nodes, phy_per_node]
    pack_index, rank_in_pack = _fixed_capacity_heap_pack(tokens_per_phy, gpus_per_node)
    # Compute per-node physical index to per-GPU position
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack  # [L*num_nodes, phy_per_node]
    pphy2phy = PermOps.inverse(phy2pphy)

    # Map back to global logical ids and consolidate nodes
    Lnodes = tokens_per_mlog.shape[0] // num_nodes
    # phy2mlog corresponds to per-node logical ids in [0, num_logical_experts//num_nodes)
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [L*num_nodes, phy_per_node]
    # Shift per-node logical ranges to global [0, num_logical_experts)
    pphy2mlog = (pphy2mlog.view(L, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes, device=device).view(1, -1, 1)
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
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # hierarchical policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global policy (single group across all nodes)
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