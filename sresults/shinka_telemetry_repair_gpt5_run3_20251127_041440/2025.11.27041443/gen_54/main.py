# EVOLVE-BLOCK-START
"""
ADMM/POCS consensus repair for network telemetry.

Algorithm overview (fundamentally different approach):
- Global alternating projections across convex sets with relaxed updates:
  1) Link symmetry projection (R3) with adaptive Huber-like relaxation and soft-zero pre-snap.
  2) Router conservation projection (R1) via closed-form weighted projection, per-iteration dominance caps.
  3) Tiny bundle consensus smoothing pass (±5%) to stabilize parallel links.
- Iterate the projections (3–4 rounds) to reach consensus instead of single-pass corrections.
- Confidence calibrated from movement magnitude, link and router residuals,
  plus strong-scale guards and peer smoothing.

Maintains function signature and outputs.
"""
from typing import Dict, Any, Tuple, List, Optional
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H = 0.02                 # base symmetry tolerance (~2%)
    TAU_HIGH = 0.015             # high-rate tighter tolerance
    TAU_LOW = 0.03               # near-zero looser tolerance
    ZERO_THRESH = 1.0            # Mbps near-zero threshold
    ZERO_EPS = 1e-9

    # POCS / ADMM iteration params
    N_ITERS = 4                  # total projection rounds
    LINK_RELAX_MIN = 0.70        # min relaxation toward equality on links
    LINK_RELAX_MAX = 0.90        # max relaxation toward equality on links
    RESYNC_MAX_F = 0.40          # max one-sided nudge toward mean (guarded by confidence gap)
    STRONG_SCALE_GUARD = 0.08    # skip re-sync if target strongly scaled already

    # Router projection params
    ROUTER_RELAX = 0.60          # damping in router hyperplane projection
    PER_VAR_REL_CLIP = 0.10      # ±10% per-var cap per iteration
    DOMINANCE_CAP = 0.50         # ≤50% of per-iteration correction per variable

    # Bundle smoothing
    BUNDLE_FRAC = 0.60           # bundle dominance threshold
    BUNDLE_CLIP = 0.05           # ±5% bundle intra-consensus clip

    # Confidence post-calibration
    PEER_SMOOTH = 0.10
    CLIP_HIT_PENALTY = 0.97      # mild penalty per strong scale/clip
    UNTOUCHED_BOOST = 0.02       # small boost for near-unchanged, well-synced counters

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def adaptive_tau(v1: float, v2: float) -> float:
        if v1 >= 100.0 and v2 >= 100.0:
            return TAU_HIGH
        if v1 < ZERO_THRESH or v2 < ZERO_THRESH:
            return TAU_LOW
        return TAU_H

    def soft_beta(mismatch: float, v1: float, v2: float) -> float:
        # Adaptive snap strength toward equality: larger when mismatch is large.
        # Base from 0.7 to 0.9 as mismatch grows above 10% up to 30%.
        span = 0.20
        t = 0.10
        x = 0.0 if mismatch <= t else min(1.0, (mismatch - t) / span)
        beta = LINK_RELAX_MIN + (LINK_RELAX_MAX - LINK_RELAX_MIN) * x
        # Slightly stronger if one side is near-zero vs the other
        if (v1 < ZERO_THRESH and v2 >= ZERO_THRESH) or (v2 < ZERO_THRESH and v1 >= ZERO_THRESH):
            beta = min(LINK_RELAX_MAX, beta + 0.05)
        return clamp01(beta)

    # Build basic maps
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, Optional[str]] = {}
    local_router_of: Dict[str, Any] = {}
    remote_router_of: Dict[str, Any] = {}

    for iid, d in telemetry.items():
        orig_rx[iid] = float(d.get('rx_rate', 0.0))
        orig_tx[iid] = float(d.get('tx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        p = d.get('connected_to')
        peer_of[iid] = p if p in telemetry else None
        local_router_of[iid] = d.get('local_router')
        remote_router_of[iid] = d.get('remote_router')

    # Derive missing remote router from peer local_router
    for iid in telemetry:
        if not remote_router_of.get(iid):
            p = peer_of.get(iid)
            if p and p in telemetry:
                remote_router_of[iid] = telemetry[p].get('local_router')

    # Router->interfaces mapping (use provided topology if any)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router fields (still valuable to enforce flow conservation)
        for iid, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Unique link pairs
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b not in telemetry or a == b:
            continue
        if (b, a) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))
    in_pairs = {x for ab in link_pairs for x in ab}

    # Initialize working rates with non-negative originals
    rx: Dict[str, float] = {i: max(0.0, orig_rx[i]) for i in telemetry}
    tx: Dict[str, float] = {i: max(0.0, orig_tx[i]) for i in telemetry}

    # Initial directional confidences (refined during/after projections)
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    # Track scale and clip impacts for confidence
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    clip_hit_rx: Dict[str, bool] = {i: False for i in telemetry}
    clip_hit_tx: Dict[str, bool] = {i: False for i in telemetry}

    # Helper: compute per-router residuals
    def router_residuals(vals_rx: Dict[str, float], vals_tx: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                out[r] = 0.0
                continue
            srx = sum(vals_rx[i] for i in ups)
            stx = sum(vals_tx[i] for i in ups)
            denom = max(1.0, srx, stx)
            out[r] = abs(srx - stx) / denom
        return out

    # Pre-pass: enforce status down => zero and soft-zero on tiny links
    for i in telemetry:
        if status.get(i) != 'up':
            rx[i] = 0.0
            tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            # already zeroed above
            continue
        if max(rx[a], tx[a], rx[b], tx[b]) < 1.5 * ZERO_THRESH:
            rx[a] = tx[a] = rx[b] = tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.95)
            conf_tx[a] = max(conf_tx[a], 0.95)
            conf_rx[b] = max(conf_rx[b], 0.95)
            conf_tx[b] = max(conf_tx[b], 0.95)

    # Iterative POCS loop
    for it in range(N_ITERS):
        # 1) Link symmetry projection (R3) with adaptive relaxation
        for a, b in link_pairs:
            if status.get(a) != 'up' or status.get(b) != 'up':
                continue

            # Direction 1: a.tx vs b.rx
            x = tx[a]
            y = rx[b]
            if max(x, y) < 2.0 * ZERO_THRESH:
                # keep small values coherent at zero if routers not imbalanced
                tx[a] = 0.0
                rx[b] = 0.0
            else:
                d = rel_diff(x, y)
                tau = adaptive_tau(x, y)
                mean = 0.5 * (x + y)
                if d <= tau:
                    # small mismatch: gentle equalization
                    alpha = LINK_RELAX_MIN
                else:
                    alpha = soft_beta(d, x, y)
                tx_new = x + alpha * (mean - x)
                rx_new = y + alpha * (mean - y)
                tx_new = max(0.0, tx_new)
                rx_new = max(0.0, rx_new)
                # update scale trackers
                if tx[a] > ZERO_EPS:
                    scl = tx_new / max(ZERO_EPS, tx[a])
                    scaled_tx_factor[a] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit_tx[a] = True
                if rx[b] > ZERO_EPS:
                    scl = rx_new / max(ZERO_EPS, rx[b])
                    scaled_rx_factor[b] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit_rx[b] = True
                tx[a] = tx_new
                rx[b] = rx_new

            # Direction 2: a.rx vs b.tx
            x = rx[a]
            y = tx[b]
            if max(x, y) < 2.0 * ZERO_THRESH:
                rx[a] = 0.0
                tx[b] = 0.0
            else:
                d = rel_diff(x, y)
                tau = adaptive_tau(x, y)
                mean = 0.5 * (x + y)
                if d <= tau:
                    alpha = LINK_RELAX_MIN
                else:
                    alpha = soft_beta(d, x, y)
                rx_new = x + alpha * (mean - x)
                tx_new = y + alpha * (mean - y)
                rx_new = max(0.0, rx_new)
                tx_new = max(0.0, tx_new)
                if rx[a] > ZERO_EPS:
                    scl = rx_new / max(ZERO_EPS, rx[a])
                    scaled_rx_factor[a] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit_rx[a] = True
                if tx[b] > ZERO_EPS:
                    scl = tx_new / max(ZERO_EPS, tx[b])
                    scaled_tx_factor[b] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit_tx[b] = True
                rx[a] = rx_new
                tx[b] = tx_new

        # 2) Router conservation projection (R1) via weighted projection with caps
        for r, ifs in router_ifaces.items():
            up_ifs = [i for i in ifs if status.get(i) == 'up']
            if len(up_ifs) < 2:
                continue
            srx = sum(rx[i] for i in up_ifs)
            stx = sum(tx[i] for i in up_ifs)
            denom_r = max(1.0, srx, stx)
            gap = srx - stx
            rel_gap = abs(gap) / denom_r
            # Skip if already close
            n_active = len(up_ifs)
            tau_router = min(0.07, max(0.03, 0.05 * math.sqrt(2.0 / max(2, n_active))))
            if rel_gap <= tau_router:
                continue

            # Build vectors
            vals: List[float] = []
            signs: List[int] = []
            idx_map: List[Tuple[str, str]] = []
            weights: List[float] = []
            for i in up_ifs:
                # RX variable
                vals.append(rx[i]); signs.append(+1); idx_map.append((i, 'rx'))
                c = clamp01(conf_rx.get(i, 0.7))
                w = max(1e-6, (c**2) / max(rx[i], ZERO_THRESH))
                if rx[i] < ZERO_THRESH: w *= 3.0
                weights.append(w)
            for i in up_ifs:
                # TX variable
                vals.append(tx[i]); signs.append(-1); idx_map.append((i, 'tx'))
                c = clamp01(conf_tx.get(i, 0.7))
                w = max(1e-6, (c**2) / max(tx[i], ZERO_THRESH))
                if tx[i] < ZERO_THRESH: w *= 3.0
                weights.append(w)

            invw = [1.0 / max(1e-12, w) for w in weights]
            ax0 = sum(s * v for s, v in zip(signs, vals))
            denom = sum(invw)
            if denom <= 1e-12:
                continue
            # Damped lambda
            lam = 2.0 * ax0 / denom
            lam *= ROUTER_RELAX

            # Raw deltas and dominance cap
            deltas = [-(iw) * s * lam * 0.5 for s, iw in zip(signs, invw)]
            # Apply dominance cap iteratively
            for _ in range(2):
                absd = [abs(d) for d in deltas]
                total = sum(absd) + 1e-12
                shares = [d / total for d in absd]
                mshare = max(shares)
                if mshare <= DOMINANCE_CAP + 1e-6:
                    break
                k = shares.index(mshare)
                # reduce offender's move by scaling its inv-weight
                invw[k] *= (mshare / DOMINANCE_CAP)
                denom2 = sum(invw)
                if denom2 <= 1e-12:
                    break
                lam2 = 2.0 * ax0 / denom2
                lam2 *= ROUTER_RELAX
                deltas = [-(iw) * s * lam2 * 0.5 for s, iw in zip(signs, invw)]

            # Per-variable relative cap ±10% and apply
            scale = 1.0
            caps = [PER_VAR_REL_CLIP * max(v, ZERO_THRESH) for v in vals]
            for dlt, cap in zip(deltas, caps):
                if abs(dlt) > cap:
                    scale = min(scale, cap / max(abs(dlt), 1e-12))
            if scale < 1.0:
                deltas = [d * scale for d in deltas]

            for (iid, side), dv, v_old in zip(idx_map, deltas, vals):
                v_new = max(0.0, v_old + dv)
                if side == 'rx':
                    if rx[iid] > ZERO_EPS:
                        scl = v_new / max(ZERO_EPS, rx[iid])
                        scaled_rx_factor[iid] *= scl
                        if abs(scl - 1.0) >= 0.10 or abs(dv) >= PER_VAR_REL_CLIP * max(v_old, ZERO_THRESH) - 1e-12:
                            clip_hit_rx[iid] = True
                    rx[iid] = v_new
                else:
                    if tx[iid] > ZERO_EPS:
                        scl = v_new / max(ZERO_EPS, tx[iid])
                        scaled_tx_factor[iid] *= scl
                        if abs(scl - 1.0) >= 0.10 or abs(dv) >= PER_VAR_REL_CLIP * max(v_old, ZERO_THRESH) - 1e-12:
                            clip_hit_tx[iid] = True
                    tx[iid] = v_new

        # 3) Bundle intra-consensus smoothing (tiny ±5%)
        # Group interfaces by (local_router, remote_router)
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                continue
            sum_rx_side = sum(rx[i] for i in ups)
            sum_tx_side = sum(tx[i] for i in ups)
            bundles: Dict[Tuple[Any, Any], List[str]] = {}
            for i in ups:
                key = (local_router_of.get(i), remote_router_of.get(i))
                bundles.setdefault(key, []).append(i)

            # RX side
            for key, members in bundles.items():
                if len(members) < 2:
                    continue
                gsum = sum(rx[i] for i in members)
                if sum_rx_side > ZERO_EPS and gsum / sum_rx_side >= BUNDLE_FRAC:
                    rels = []
                    for i in members:
                        base = max(ZERO_THRESH, orig_rx.get(i, 0.0))
                        rels.append(rx[i] / base)
                    s = sum(rels) / len(rels)
                    s = max(1.0 - BUNDLE_CLIP, min(1.0 + BUNDLE_CLIP, s))
                    for i in members:
                        base = max(ZERO_THRESH, orig_rx.get(i, 0.0))
                        new_v = max(0.0, base * s)
                        if rx[i] > ZERO_EPS:
                            scl = new_v / rx[i]
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_rx[i] = True
                        rx[i] = new_v
            # TX side
            for key, members in bundles.items():
                if len(members) < 2:
                    continue
                gsum = sum(tx[i] for i in members)
                if sum_tx_side > ZERO_EPS and gsum / sum_tx_side >= BUNDLE_FRAC:
                    rels = []
                    for i in members:
                        base = max(ZERO_THRESH, orig_tx.get(i, 0.0))
                        rels.append(tx[i] / base)
                    s = sum(rels) / len(rels)
                    s = max(1.0 - BUNDLE_CLIP, min(1.0 + BUNDLE_CLIP, s))
                    for i in members:
                        base = max(ZERO_THRESH, orig_tx.get(i, 0.0))
                        new_v = max(0.0, base * s)
                        if tx[i] > ZERO_EPS:
                            scl = new_v / tx[i]
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_tx[i] = True
                        tx[i] = new_v

        # 4) Confidence-gap micro re-sync on links (guarded by strong-scale flags)
        # Helps close stubborn asymmetries without fighting router projection
        for a, b in link_pairs:
            if status.get(a) != 'up' or status.get(b) != 'up':
                continue
            # Direction 1: a.tx vs b.rx
            x, y = tx[a], rx[b]
            if max(x, y) > ZERO_EPS:
                d = rel_diff(x, y)
                tau = adaptive_tau(x, y)
                if d > tau:
                    ca, cb = conf_tx.get(a, 0.7), conf_rx.get(b, 0.7)
                    if ca >= cb and abs(scaled_rx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                        f = min(RESYNC_MAX_F, max(0.0, ca - cb))
                        target = 0.5 * (x + y)
                        new_y = max(0.0, y + f * (target - y))
                        if rx[b] > ZERO_EPS:
                            scl = new_y / rx[b]
                            scaled_rx_factor[b] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_rx[b] = True
                        rx[b] = new_y
                    elif cb > ca and abs(scaled_tx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                        f = min(RESYNC_MAX_F, max(0.0, cb - ca))
                        target = 0.5 * (x + y)
                        new_x = max(0.0, x + f * (target - x))
                        if tx[a] > ZERO_EPS:
                            scl = new_x / tx[a]
                            scaled_tx_factor[a] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_tx[a] = True
                        tx[a] = new_x
            # Direction 2: a.rx vs b.tx
            x, y = rx[a], tx[b]
            if max(x, y) > ZERO_EPS:
                d = rel_diff(x, y)
                tau = adaptive_tau(x, y)
                if d > tau:
                    ca, cb = conf_rx.get(a, 0.7), conf_tx.get(b, 0.7)
                    if ca >= cb and abs(scaled_tx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                        f = min(RESYNC_MAX_F, max(0.0, ca - cb))
                        target = 0.5 * (x + y)
                        new_y = max(0.0, y + f * (target - y))
                        if tx[b] > ZERO_EPS:
                            scl = new_y / tx[b]
                            scaled_tx_factor[b] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_tx[b] = True
                        tx[b] = new_y
                    elif cb > ca and abs(scaled_rx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                        f = min(RESYNC_MAX_F, max(0.0, cb - ca))
                        target = 0.5 * (x + y)
                        new_x = max(0.0, x + f * (target - x))
                        if rx[a] > ZERO_EPS:
                            scl = new_x / rx[a]
                            scaled_rx_factor[a] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_rx[a] = True
                        rx[a] = new_x

        # 5) Safety enforce down => zero each round
        for i in telemetry:
            if status.get(i) != 'up':
                rx[i] = 0.0
                tx[i] = 0.0

    # Final router residuals for confidence
    rtr_resid = router_residuals(rx, tx)

    # Confidence calibration
    conf_out_rx: Dict[str, float] = {}
    conf_out_tx: Dict[str, float] = {}
    for i in telemetry:
        # Measurement deltas
        r_meas_rx = rel_diff(rx[i], orig_rx[i])
        r_meas_tx = rel_diff(tx[i], orig_tx[i])

        # Link residuals
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            r_link_rx = rel_diff(rx[i], tx[p])
            r_link_tx = rel_diff(tx[i], rx[p])
        else:
            r_link_rx = 0.2
            r_link_tx = 0.2

        # Router residual
        rtr = rtr_resid.get(local_router_of.get(i), 0.0)

        base_rx = clamp01(1.0 - (0.55 * r_meas_rx + 0.35 * r_link_rx + 0.10 * rtr))
        base_tx = clamp01(1.0 - (0.55 * r_meas_tx + 0.35 * r_link_tx + 0.10 * rtr))

        # Strong scale penalty
        alpha_rx = abs(scaled_rx_factor.get(i, 1.0) - 1.0)
        alpha_tx = abs(scaled_tx_factor.get(i, 1.0) - 1.0)
        c_rx = clamp01(0.92 * base_rx + 0.08 * (1.0 - min(0.5, alpha_rx)))
        c_tx = clamp01(0.92 * base_tx + 0.08 * (1.0 - min(0.5, alpha_tx)))

        # Clip-hit penalty
        if clip_hit_rx.get(i, False) or alpha_rx >= 0.10:
            c_rx *= CLIP_HIT_PENALTY
        if clip_hit_tx.get(i, False) or alpha_tx >= 0.10:
            c_tx *= CLIP_HIT_PENALTY

        # Untouched boost for small change and good symmetry
        if r_meas_rx < 0.01 and r_link_rx <= adaptive_tau(rx[i], tx.get(p, rx[i]) if p else rx[i]):
            c_rx = min(0.98, c_rx + UNTOUCHED_BOOST)
        if r_meas_tx < 0.01 and r_link_tx <= adaptive_tau(tx[i], rx.get(p, tx[i]) if p else tx[i]):
            c_tx = min(0.98, c_tx + UNTOUCHED_BOOST)

        # Floor for down interfaces
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)

        conf_out_rx[i] = c_rx
        conf_out_tx[i] = c_tx

    # Peer smoothing
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            conf_out_tx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_out_tx[a] + PEER_SMOOTH * conf_out_rx[b])
            conf_out_rx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_out_rx[a] + PEER_SMOOTH * conf_out_tx[b])
            conf_out_tx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_out_tx[b] + PEER_SMOOTH * conf_out_rx[a])
            conf_out_rx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_out_rx[b] + PEER_SMOOTH * conf_out_tx[a])

    # Assemble result
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        my_status = status.get(i, 'unknown')
        peer_id = peer_of.get(i)
        # Status confidence: penalize if peer status inconsistent or down-with-traffic
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            if telemetry[peer_id].get('interface_status', 'unknown') != my_status:
                status_conf = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_conf = min(status_conf, 0.6)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), max(0.0, rx.get(i, 0.0)), clamp01(conf_out_rx.get(i, 0.6)))
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), max(0.0, tx.get(i, 0.0)), clamp01(conf_out_tx.get(i, 0.6)))
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