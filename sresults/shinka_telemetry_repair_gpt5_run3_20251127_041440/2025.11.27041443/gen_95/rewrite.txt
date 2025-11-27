# EVOLVE-BLOCK-START
"""
Global weighted POCS for network telemetry repair.

Core idea:
- Alternate projections onto two convex invariant sets:
  (1) Link symmetry: a.tx = b.rx and a.rx = b.tx.
  (2) Router conservation: Σ_tx(router) = Σ_rx(router).
- Projections are minimum-change under dominance- and confidence-aware weights,
  with per-interface, share/HHI-adaptive move caps to avoid overcorrections.
- Includes soft-zero stabilization, non-negativity, and early stopping.

Confidence is calibrated from measurement/link/router residuals and move scales,
with peer smoothing and "untouched" boosts.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H = 0.02                 # ~2% link hardening tolerance
    ZERO_THRESH = 1.0            # Mbps near-zero threshold
    ZERO_EPS = 1e-9
    MAX_ITERS = 15               # POCS iterations
    EARLY_STOP_LINK = 0.012      # stop when max link residual below ~1.2%
    EARLY_STOP_ROUTER = 0.025    # stop when router residual below ~2.5%
    WEIGHT_FLOOR = 0.40          # minimum direction weight
    WEIGHT_CEIL = 2.50           # maximum direction weight
    SHARE_PROTECT = 0.50         # additional weight factor for dominant share
    CAP_REL_BASE = 0.10          # base per-iteration relative cap 10%
    CAP_REL_MIN = 0.03           # min per-iteration relative cap 3%
    CAP_REL_MAX = 0.15           # max per-iteration relative cap 15%
    SOFT_ZERO_FACTOR = 2.0       # if all four sides < 2*ZERO_THRESH -> zero
    PEER_SMOOTH = 0.10           # confidence peer smoothing
    UNTOUCHED_BOOST = 0.02       # slight boost for untouched, well-synced counters
    CLIP_HIT_PENALTY = 0.95      # confidence penalty if caps hit

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def adaptive_tau(v1: float, v2: float) -> float:
        # Slightly tighter for large rates, looser for low.
        if v1 >= 100.0 and v2 >= 100.0:
            return 0.015
        if v1 < ZERO_THRESH or v2 < ZERO_THRESH:
            return 0.03
        return TAU_H

    def median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2 - 2])

    # Extract base data
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}
    local_router_of: Dict[str, Any] = {}
    remote_router_of: Dict[str, Any] = {}

    for iid, d in telemetry.items():
        orig_rx[iid] = float(d.get('rx_rate', 0.0))
        orig_tx[iid] = float(d.get('tx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        ct = d.get('connected_to')
        peer_of[iid] = ct if ct in telemetry else None
        local_router_of[iid] = d.get('local_router')
        remote_router_of[iid] = d.get('remote_router')

    # Build router_ifaces from topology when possible
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router fields if topology missing
        for iid, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Fill missing remote_router via peer's local_router if available
    for iid in telemetry:
        if not remote_router_of.get(iid):
            p = peer_of.get(iid)
            if p and p in telemetry:
                remote_router_of[iid] = telemetry[p].get('local_router')

    # Build link pairs (unique undirected)
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b not in telemetry or a == b:
            continue
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))

    # Initialize working values
    x_rx: Dict[str, float] = {i: max(0.0, orig_rx[i]) for i in telemetry}
    x_tx: Dict[str, float] = {i: max(0.0, orig_tx[i]) for i in telemetry}

    # Initial weights per direction (confidence-aware, dominance-aware)
    w_rx: Dict[str, float] = {}
    w_tx: Dict[str, float] = {}

    def init_weights() -> None:
        # Link-mismatch confidence for directions
        for i in telemetry:
            if status.get(i) != 'up':
                w_rx[i] = WEIGHT_CEIL
                w_tx[i] = WEIGHT_CEIL
                continue
            p = peer_of.get(i)
            # Use original values for initialization to avoid bias
            if p and p in telemetry and status.get(p) == 'up':
                c_tx = clamp01(1.0 - rel_diff(orig_tx[i], orig_rx[p]))
                c_rx = clamp01(1.0 - rel_diff(orig_rx[i], orig_tx[p]))
            else:
                c_tx = 0.6
                c_rx = 0.6
            wtx = max(WEIGHT_FLOOR, 0.5 + 0.5 * c_tx)
            wrx = max(WEIGHT_FLOOR, 0.5 + 0.5 * c_rx)
            w_tx[i] = min(WEIGHT_CEIL, wtx)
            w_rx[i] = min(WEIGHT_CEIL, wrx)
        # Share protection: amplify weights for dominant shares to move less
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                continue
            sum_tx = sum(max(x_tx[i], 0.0) for i in ups) + ZERO_EPS
            sum_rx = sum(max(x_rx[i], 0.0) for i in ups) + ZERO_EPS
            # HHI per direction (for later caps)
            for i in ups:
                share_t = max(x_tx[i], 0.0) / sum_tx
                share_r = max(x_rx[i], 0.0) / sum_rx
                w_tx[i] = min(WEIGHT_CEIL, w_tx[i] * (1.0 + SHARE_PROTECT * share_t))
                w_rx[i] = min(WEIGHT_CEIL, w_rx[i] * (1.0 + SHARE_PROTECT * share_r))

    init_weights()

    # Track cap hits for confidence penalties
    cap_hit_rx: Dict[str, bool] = {i: False for i in telemetry}
    cap_hit_tx: Dict[str, bool] = {i: False for i in telemetry}

    # Helper: compute router residuals
    def router_residuals(curr_rx: Dict[str, float], curr_tx: Dict[str, float]) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                res[r] = 0.0
                continue
            srx = sum(curr_rx[i] for i in ups)
            stx = sum(curr_tx[i] for i in ups)
            denom = max(1.0, srx, stx)
            res[r] = abs(stx - srx) / denom
        return res

    # Helper: compute link residuals (max over both directions)
    def max_link_residual() -> float:
        m = 0.0
        for a, b in link_pairs:
            if status.get(a) != 'up' or status.get(b) != 'up':
                continue
            r1 = rel_diff(x_tx[a], x_rx[b])
            r2 = rel_diff(x_rx[a], x_tx[b])
            # only consider residuals above tolerance
            tau1 = adaptive_tau(x_tx[a], x_rx[b])
            tau2 = adaptive_tau(x_rx[a], x_tx[b])
            m = max(m, max(max(0.0, r1 - tau1), max(0.0, r2 - tau2)))
        return m

    # Soft-zero helper for a link
    def soft_zero_link(a: str, b: str) -> bool:
        if status.get(a) != 'up' or status.get(b) != 'up':
            return False
        if max(x_rx[a], x_tx[a], x_rx[b], x_tx[b]) < SOFT_ZERO_FACTOR * ZERO_THRESH:
            x_rx[a] = x_tx[a] = 0.0
            x_rx[b] = x_tx[b] = 0.0
            return True
        return False

    # Link projection: weighted averaging on each direction
    def project_links() -> None:
        for a, b in link_pairs:
            a_up = (status.get(a) == 'up')
            b_up = (status.get(b) == 'up')
            if not a_up or not b_up:
                # Down link: force zeros already in status handling later
                continue

            # Near-zero stabilization
            if soft_zero_link(a, b):
                continue

            # a.tx vs b.rx
            v1, v2 = x_tx[a], x_rx[b]
            tau1 = adaptive_tau(v1, v2)
            w1, w2 = max(WEIGHT_FLOOR, w_tx.get(a, 1.0)), max(WEIGHT_FLOOR, w_rx.get(b, 1.0))
            # Weighted projection (exact). Keep even within tau to reduce noise symmetrically.
            v = (w1 * v1 + w2 * v2) / max(ZERO_EPS, w1 + w2)
            x_tx[a] = max(0.0, v)
            x_rx[b] = max(0.0, v)

            # a.rx vs b.tx
            v1b, v2b = x_rx[a], x_tx[b]
            tau2 = adaptive_tau(v1b, v2b)
            w1b, w2b = max(WEIGHT_FLOOR, w_rx.get(a, 1.0)), max(WEIGHT_FLOOR, w_tx.get(b, 1.0))
            vb = (w1b * v1b + w2b * v2b) / max(ZERO_EPS, w1b + w2b)
            x_rx[a] = max(0.0, vb)
            x_tx[b] = max(0.0, vb)

    # Router projection: exact weighted projection with cap saturation (one equality per router)
    def project_routers() -> None:
        nonlocal cap_hit_rx, cap_hit_tx
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if len(ups) == 0:
                continue

            sum_tx_r = sum(x_tx[i] for i in ups)
            sum_rx_r = sum(x_rx[i] for i in ups)
            gap = sum_tx_r - sum_rx_r  # want gap -> 0

            denom = max(1.0, sum_tx_r, sum_rx_r)
            # Adaptive skip if gap is within tolerance
            n_active = len(ups)
            tau_router = min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active)) ** 0.5))
            if abs(gap) / denom <= tau_router:
                continue

            # Compute shares and HHI per direction for caps
            sum_tx_pos = sum(max(x_tx[i], 0.0) for i in ups) + ZERO_EPS
            sum_rx_pos = sum(max(x_rx[i], 0.0) for i in ups) + ZERO_EPS
            shares_tx = {i: max(x_tx[i], 0.0) / sum_tx_pos for i in ups}
            shares_rx = {i: max(x_rx[i], 0.0) / sum_rx_pos for i in ups}
            hhi_tx = sum(shares_tx[i] ** 2 for i in ups)
            hhi_rx = sum(shares_rx[i] ** 2 for i in ups)
            hhi = 0.5 * (hhi_tx + hhi_rx)

            # Cap per-interface relative change (share/HHI aware)
            cap_tx_abs: Dict[str, float] = {}
            cap_rx_abs: Dict[str, float] = {}
            for i in ups:
                share_t = shares_tx[i]
                share_r = shares_rx[i]
                # stiffer when HHI and share are high
                cap_rel_t = CAP_REL_BASE * (1.0 - 0.5 * math.sqrt(max(0.0, hhi))) * (1.0 - 0.4 * share_t)
                cap_rel_r = CAP_REL_BASE * (1.0 - 0.5 * math.sqrt(max(0.0, hhi))) * (1.0 - 0.4 * share_r)
                cap_rel_t = min(CAP_REL_MAX, max(CAP_REL_MIN, cap_rel_t))
                cap_rel_r = min(CAP_REL_MAX, max(CAP_REL_MIN, cap_rel_r))
                cap_tx_abs[i] = cap_rel_t * max(1.0, x_tx[i])
                cap_rx_abs[i] = cap_rel_r * max(1.0, x_rx[i])

            # Active sets for saturation algorithm
            U_tx = set(ups)
            U_rx = set(ups)
            fixed_tx: Dict[str, float] = {i: 0.0 for i in ups}
            fixed_rx: Dict[str, float] = {i: 0.0 for i in ups}

            # Precompute inverse weights to avoid division by zero
            inv_w_tx = {i: 1.0 / max(ZERO_EPS, w_tx.get(i, 1.0)) for i in ups}
            inv_w_rx = {i: 1.0 / max(ZERO_EPS, w_rx.get(i, 1.0)) for i in ups}

            # Saturation loop
            remaining = 10 * len(ups) + 10  # safety bound
            while remaining > 0:
                remaining -= 1
                sum_inv_tx = sum(inv_w_tx[i] for i in U_tx) if U_tx else 0.0
                sum_inv_rx = sum(inv_w_rx[i] for i in U_rx) if U_rx else 0.0
                S = sum_inv_tx + sum_inv_rx
                S_fixed = sum(fixed_tx[i] for i in ups) - sum(fixed_rx[i] for i in ups)
                # If no degrees of freedom remain, break
                if S <= ZERO_EPS:
                    break
                # Compute Lagrange multiplier to close remaining gap (subject to fixed contributions)
                lam = 2.0 * (gap + S_fixed) / S
                # Candidate deltas
                violate = False
                cand_tx: Dict[str, float] = {}
                cand_rx: Dict[str, float] = {}
                for i in list(U_tx):
                    dti = -lam * 0.5 * inv_w_tx[i]
                    # Saturate check
                    if abs(dti) > cap_tx_abs[i] + 1e-12:
                        # Fix at cap
                        dti_clip = cap_tx_abs[i] if dti > 0 else -cap_tx_abs[i]
                        fixed_tx[i] += dti_clip
                        U_tx.remove(i)
                        violate = True
                    else:
                        cand_tx[i] = dti
                for i in list(U_rx):
                    dri = +lam * 0.5 * inv_w_rx[i]
                    if abs(dri) > cap_rx_abs[i] + 1e-12:
                        dri_clip = cap_rx_abs[i] if dri > 0 else -cap_rx_abs[i]
                        fixed_rx[i] += dri_clip
                        U_rx.remove(i)
                        violate = True
                    else:
                        cand_rx[i] = dri
                if not violate:
                    # Apply candidates and fixed deltas
                    for i, dti in cand_tx.items():
                        x_tx[i] = max(0.0, x_tx[i] + dti)
                        if abs(dti) >= cap_tx_abs[i] - 1e-12:
                            cap_hit_tx[i] = True
                    for i, dri in cand_rx.items():
                        x_rx[i] = max(0.0, x_rx[i] + dri)
                        if abs(dri) >= cap_rx_abs[i] - 1e-12:
                            cap_hit_rx[i] = True
                    # Apply fixed contributions to those already clipped (no-op here, as we accumulated in fixed arrays)
                    for i in fixed_tx:
                        if i not in cand_tx and abs(fixed_tx[i]) > 0.0 and i not in U_tx:
                            # Already accounted in fixed set; apply at end
                            pass
                    for i in fixed_rx:
                        if i not in cand_rx and abs(fixed_rx[i]) > 0.0 and i not in U_rx:
                            pass
                    # Now apply fixed deltas
                    for i in ups:
                        if i not in cand_tx and i not in U_tx and abs(fixed_tx[i]) > 0.0:
                            x_tx[i] = max(0.0, x_tx[i] + fixed_tx[i])
                            cap_hit_tx[i] = True
                            fixed_tx[i] = 0.0
                        if i not in cand_rx and i not in U_rx and abs(fixed_rx[i]) > 0.0:
                            x_rx[i] = max(0.0, x_rx[i] + fixed_rx[i])
                            cap_hit_rx[i] = True
                            fixed_rx[i] = 0.0
                    break  # finished projection for this router

            # If we exit due to no degrees left, we've applied partial closure only.
            # Non-negativity will be enforced below.

    # Enforce status down -> zero (hard constraint)
    def enforce_status_zero() -> None:
        for i in telemetry:
            if status.get(i) != 'up':
                x_rx[i] = 0.0
                x_tx[i] = 0.0

    # Non-negativity clamp (safety)
    def clamp_nonneg() -> None:
        for i in telemetry:
            x_rx[i] = max(0.0, x_rx[i])
            x_tx[i] = max(0.0, x_tx[i])

    # Iterative POCS loop
    for it in range(MAX_ITERS):
        # Link projection
        project_links()
        # Router projection
        project_routers()
        # Status and non-negativity
        enforce_status_zero()
        clamp_nonneg()

        # Optional: refresh weights once midway based on current residuals to focus on remaining mismatches
        if it == (MAX_ITERS // 2):
            # Update weights using current link mismatches; keep share protection
            for i in telemetry:
                if status.get(i) != 'up':
                    w_rx[i] = WEIGHT_CEIL
                    w_tx[i] = WEIGHT_CEIL
                    continue
                p = peer_of.get(i)
                if p and p in telemetry and status.get(p) == 'up':
                    c_tx = clamp01(1.0 - rel_diff(x_tx[i], x_rx[p]))
                    c_rx = clamp01(1.0 - rel_diff(x_rx[i], x_tx[p]))
                else:
                    c_tx = 0.6
                    c_rx = 0.6
                w_tx[i] = min(WEIGHT_CEIL, max(WEIGHT_FLOOR, 0.5 + 0.5 * c_tx))
                w_rx[i] = min(WEIGHT_CEIL, max(WEIGHT_FLOOR, 0.5 + 0.5 * c_rx))
            # Re-apply share protection
            for r, ifs in router_ifaces.items():
                ups = [i for i in ifs if status.get(i) == 'up']
                if not ups:
                    continue
                sum_tx = sum(max(x_tx[i], 0.0) for i in ups) + ZERO_EPS
                sum_rx = sum(max(x_rx[i], 0.0) for i in ups) + ZERO_EPS
                for i in ups:
                    share_t = max(x_tx[i], 0.0) / sum_tx
                    share_r = max(x_rx[i], 0.0) / sum_rx
                    w_tx[i] = min(WEIGHT_CEIL, w_tx[i] * (1.0 + SHARE_PROTECT * share_t))
                    w_rx[i] = min(WEIGHT_CEIL, w_rx[i] * (1.0 + SHARE_PROTECT * share_r))

        # Early stopping if residuals small
        rr = router_residuals(x_rx, x_tx)
        max_rtr = max(rr.values()) if rr else 0.0
        max_link = max_link_residual()
        if max_link <= EARLY_STOP_LINK and max_rtr <= EARLY_STOP_ROUTER:
            break

    # Final soft-zero stabilization on near-zero links
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            soft_zero_link(a, b)

    # Final router residuals for confidence
    final_router_res = router_residuals(x_rx, x_tx)

    # Confidence calibration
    conf_rx: Dict[str, float] = {}
    conf_tx: Dict[str, float] = {}

    # Precompute bundle residual medians for optional bundle consistency signal
    bundle_e_map_tx: Dict[Tuple[Any, Any], List[float]] = {}
    bundle_e_map_rx: Dict[Tuple[Any, Any], List[float]] = {}
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        bundle_e_map_tx.setdefault((ra, rb), []).append(x_tx[a] - x_rx[b])
        bundle_e_map_rx.setdefault((rb, ra), []).append(x_tx[b] - x_rx[a])
    bundle_e_med_tx: Dict[Tuple[Any, Any], float] = {k: median(v) for k, v in bundle_e_map_tx.items()}
    bundle_e_med_rx: Dict[Tuple[Any, Any], float] = {k: median(v) for k, v in bundle_e_map_rx.items()}

    # Router totals for share-based stability
    router_sum_rx: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if status.get(i) == 'up']
        router_sum_rx[r] = sum(x_rx.get(i, 0.0) for i in ups)
        router_sum_tx[r] = sum(x_tx.get(i, 0.0) for i in ups)

    for i in telemetry:
        p = peer_of.get(i)
        # Measurement residuals
        r_meas_rx = rel_diff(x_rx[i], orig_rx[i])
        r_meas_tx = rel_diff(x_tx[i], orig_tx[i])

        # Link residuals
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            r_link_rx = rel_diff(x_rx[i], x_tx[p])
            r_link_tx = rel_diff(x_tx[i], x_rx[p])
        else:
            r_link_rx = 0.2
            r_link_tx = 0.2

        # Router residual
        rtr = final_router_res.get(local_router_of.get(i), 0.0)

        # Bundle consistency (distance to median residual within direction)
        bcons_rx = 0.0
        bcons_tx = 0.0
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            lr = local_router_of.get(i)
            rr = local_router_of.get(p)
            e_tx = x_tx[i] - x_rx[p]
            e_rx = x_rx[i] - x_tx[p]
            med_tx = bundle_e_med_tx.get((lr, rr), 0.0)
            med_rx = bundle_e_med_rx.get((lr, rr), 0.0)
            # closeness factor in [0,1]
            bcons_tx = clamp01(1.0 - abs(e_tx - med_tx) / (abs(med_tx) + max(x_tx[i], 1.0)))
            bcons_rx = clamp01(1.0 - abs(e_rx - med_rx) / (abs(med_rx) + max(x_rx[i], 1.0)))

        # Base confidence blend
        base_rx = 1.0 - (0.55 * r_meas_rx + 0.30 * r_link_rx + 0.15 * rtr)
        base_tx = 1.0 - (0.55 * r_meas_tx + 0.30 * r_link_tx + 0.15 * rtr)
        base_rx = clamp01(base_rx)
        base_tx = clamp01(base_tx)

        # Dominance stability: downweight confidence for dominant shares
        r_id = local_router_of.get(i)
        sum_r_rx = max(1.0, router_sum_rx.get(r_id, 1.0))
        sum_r_tx = max(1.0, router_sum_tx.get(r_id, 1.0))
        share_r = x_rx[i] / sum_r_rx
        share_t = x_tx[i] / sum_r_tx
        stab_rx = clamp01(1.0 - 0.5 * share_r)
        stab_tx = clamp01(1.0 - 0.5 * share_t)

        # Clip/move penalty
        pen_rx = CLIP_HIT_PENALTY if cap_hit_rx.get(i, False) else 1.0
        pen_tx = CLIP_HIT_PENALTY if cap_hit_tx.get(i, False) else 1.0

        c_rx = clamp01(0.85 * base_rx + 0.05 * bcons_rx + 0.10 * stab_rx) * pen_rx
        c_tx = clamp01(0.85 * base_tx + 0.05 * bcons_tx + 0.10 * stab_tx) * pen_tx

        # Down interface -> zero is strong invariant; raise floor
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)

        conf_rx[i] = clamp01(c_rx)
        conf_tx[i] = clamp01(c_tx)

    # Peer smoothing
    new_conf_rx = dict(conf_rx)
    new_conf_tx = dict(conf_tx)
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            new_conf_tx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[a] + PEER_SMOOTH * conf_rx[b])
            new_conf_rx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[b] + PEER_SMOOTH * conf_tx[a])
            new_conf_rx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[a] + PEER_SMOOTH * conf_tx[b])
            new_conf_tx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[b] + PEER_SMOOTH * conf_rx[a])
    conf_rx, conf_tx = new_conf_rx, new_conf_tx

    # Untouched boost (minimal change and good link symmetry)
    for i in telemetry:
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            # RX
            if rel_diff(x_rx[i], orig_rx[i]) < 0.01:
                if rel_diff(x_rx[i], x_tx[p]) <= adaptive_tau(x_rx[i], x_tx[p]):
                    conf_rx[i] = min(0.98, conf_rx[i] + UNTOUCHED_BOOST)
            # TX
            if rel_diff(x_tx[i], orig_tx[i]) < 0.01:
                if rel_diff(x_tx[i], x_rx[p]) <= adaptive_tau(x_tx[i], x_rx[p]):
                    conf_tx[i] = min(0.98, conf_tx[i] + UNTOUCHED_BOOST)

    # Assemble final result
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        my_status = status.get(i, 'unknown')
        peer_id = peer_of.get(i)
        # Status confidence: do not flip, but penalize inconsistent peer or down+nonzero readings
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            if telemetry[peer_id].get('interface_status', 'unknown') != my_status:
                status_conf = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_conf = min(status_conf, 0.6)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), max(0.0, x_rx.get(i, 0.0)), clamp01(conf_rx.get(i, 0.6)))
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), max(0.0, x_tx.get(i, 0.0)), clamp01(conf_tx.get(i, 0.6)))
        repaired['interface_status'] = (my_status, my_status, status_conf)

        # Copy metadata unchanged
        repaired['connected_to'] = data.get('connected_to')
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')

        result[i] = repaired

    return result

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Main entry point that will be called by the evaluator.

    Args:
        telemetry: Network interface telemetry data
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary containing repaired results with confidence scores
    """
    return repair_network_telemetry(telemetry, topology)


if __name__ == "__main__":
    # Simple test case
    test_telemetry = {
        'if1_to_if2': {
            'interface_status': 'up',
            'rx_rate': 100.0,
            'tx_rate': 95.0,
            'connected_to': 'if2_to_if1',
            'local_router': 'router1',
            'remote_router': 'router2'
        },
        'if2_to_if1': {
            'interface_status': 'up',
            'rx_rate': 95.0,  # Should match if1's TX
            'tx_rate': 100.0,  # Should match if1's RX
            'connected_to': 'if1_to_if2',
            'local_router': 'router2',
            'remote_router': 'router1'
        }
    }

    test_topology = {
        'router1': ['if1_to_if2'],
        'router2': ['if2_to_if1']
    }

    result = run_repair(test_telemetry, test_topology)

    print("Repair results:")
    for if_id, data in result.items():
        print(f"\n{if_id}:")
        print(f"  RX: {data['rx_rate']}")
        print(f"  TX: {data['tx_rate']}")
        print(f"  Status: {data['interface_status']}")