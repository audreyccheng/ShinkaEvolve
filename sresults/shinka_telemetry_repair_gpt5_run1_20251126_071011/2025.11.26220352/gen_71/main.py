# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Implements a staged pipeline:
  1) Signal collection and link hardening
  2) Router-level flow conservation with multiplicative + staged additive passes
  3) Targeted pair reconciliation with remaining per-interface budget
  4) Confidence calibration with scale/cap penalties and change taper

Maintains core invariants:
  - Link Symmetry: my_tx ≈ their_rx, my_rx ≈ their_tx
  - Flow Conservation at routers: sum(in) = sum(out)
  - Status consistency (down -> zero traffic)
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants (Hodor timing tolerance)
    HARDENING_THRESHOLD = 0.02
    EPS = 1e-9

    # -------- Helpers --------
    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"  # treat unknown as up (conservative)

    def nz_float(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, v)

    def rel_diff(a: float, b: float) -> float:
        a = float(a)
        b = float(b)
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def pair_tolerance(a: float, b: float) -> float:
        # Traffic-aware tolerance to account for timing skew on tiny links
        traffic = max(abs(a), abs(b), 1.0)
        return max(HARDENING_THRESHOLD, 2.5 / traffic)

    # -------- Topology/Graph utilities --------
    class TelemetryGraph:
        def __init__(self, tele: Dict[str, Dict[str, Any]], topo: Dict[str, List[str]]):
            self.tele = tele
            self.topo_all = topo
            self.ifaces = list(tele.keys())
            self.peers: Dict[str, str] = {i: tele[i].get("connected_to") for i in self.ifaces}
            # Only include interfaces that exist in telemetry
            self.routers: Dict[str, List[str]] = {}
            for rid, lst in topo.items():
                self.routers[rid] = [i for i in lst if i in tele]

        def get_peer(self, iface: str) -> str:
            p = self.peers.get(iface)
            return p if p in self.tele else None

        def router_ifaces_up(self, pre_state: Dict[str, Dict[str, Any]], router_id: str) -> List[str]:
            return [i for i in self.routers.get(router_id, []) if pre_state[i]['local_status'] == 'up']

    graph = TelemetryGraph(telemetry, topology)

    # -------- Stage 1: Link hardening (status-aware) --------
    pre: Dict[str, Dict[str, Any]] = {}
    for iface, data in telemetry.items():
        local_status = norm_status(data.get('interface_status', 'unknown'))
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        peer_id = graph.get_peer(iface)
        had_peer = bool(peer_id)
        peer_status = 'unknown'
        peer_rx = peer_tx = 0.0
        if had_peer:
            p = telemetry[peer_id]
            peer_status = norm_status(p.get('interface_status', 'unknown'))
            peer_rx = nz_float(p.get('rx_rate', 0.0))
            peer_tx = nz_float(p.get('tx_rate', 0.0))

        pre_rx = rx_orig
        pre_tx = tx_orig
        rx_link_diff = 0.0
        tx_link_diff = 0.0

        if local_status == 'down':
            pre_rx = 0.0
            pre_tx = 0.0
        else:
            if had_peer and peer_status == 'up':
                # Enforce link symmetry with triage substitution within tolerance
                rx_link_diff = rel_diff(rx_orig, peer_tx)  # my RX vs peer TX
                tx_link_diff = rel_diff(tx_orig, peer_rx)  # my TX vs peer RX
                rx_ok = rx_link_diff <= HARDENING_THRESHOLD
                tx_ok = tx_link_diff <= HARDENING_THRESHOLD
                if (not rx_ok) and tx_ok:
                    pre_rx = peer_tx
                elif (not tx_ok) and rx_ok:
                    pre_tx = peer_rx
                elif (not rx_ok) and (not tx_ok):
                    # Both violated -> average signals (reduces single-sided large errors)
                    pre_rx = 0.5 * (rx_orig + peer_tx)
                    pre_tx = 0.5 * (tx_orig + peer_rx)

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'local_status': local_status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer,
            'rx_link_diff': rx_link_diff,
            'tx_link_diff': tx_link_diff,
            # Pre-change flags
            'pre_rx_changed': abs(pre_rx - rx_orig) > 1e-12,
            'pre_tx_changed': abs(pre_tx - tx_orig) > 1e-12,
        }

    # -------- Stage 2: Router flow conservation (multiplicative + staged additive) --------
    # Initialize post-router values from pre
    post_router: Dict[str, Dict[str, float]] = {i: {'rx': pre[i]['pre_rx'], 'tx': pre[i]['pre_tx']} for i in graph.ifaces}
    # Track edits for confidence and remaining pair budgets
    # Absolute deltas (cumulative) and relative usage per direction
    router_delta_abs: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in graph.ifaces}
    router_delta_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in graph.ifaces}
    s_bounded_map: Dict[str, float] = {}
    router_imbalance_pre: Dict[str, float] = {}

    # Multiplicative pre-step (Recommendation #1)
    for rid, if_list in graph.routers.items():
        up_ifaces = graph.router_ifaces_up(pre, rid)
        if len(up_ifaces) < 2:
            router_imbalance_pre[rid] = 0.0
            s_bounded_map[rid] = 1.0
            continue

        sum_rx = sum(pre[i]['pre_rx'] for i in up_ifaces)
        sum_tx = sum(pre[i]['pre_tx'] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        imb = abs(sum_tx - sum_rx) / denom
        router_imbalance_pre[rid] = imb

        if (sum_rx + sum_tx) <= 1e-9 or imb <= HARDENING_THRESHOLD:
            s_bounded_map[rid] = 1.0
            continue

        # Choose less-trusted direction by avg residuals; break ties by larger absolute imbalance
        rx_resids: List[float] = []
        tx_resids: List[float] = []
        for i in up_ifaces:
            peer = graph.get_peer(i)
            if peer and pre[peer]['local_status'] == 'up':
                tx_resids.append(rel_diff(pre[i]['pre_tx'], pre[peer]['pre_rx']))
                rx_resids.append(rel_diff(pre[i]['pre_rx'], pre[peer]['pre_tx']))
        avg_tx_resid = sum(tx_resids) / len(tx_resids) if tx_resids else 0.0
        avg_rx_resid = sum(rx_resids) / len(rx_resids) if rx_resids else 0.0

        # Direction to scale
        dir_to_scale = None
        if abs(avg_tx_resid - avg_rx_resid) > 0.05:
            dir_to_scale = 'tx' if avg_tx_resid > avg_rx_resid else 'rx'
        else:
            dir_to_scale = 'tx' if sum_tx > sum_rx else 'rx'

        # Ratio and moderated factor with wide bounds, then per-interface 15% cap
        if dir_to_scale == 'tx':
            s = sum_rx / max(sum_tx, EPS)
        else:
            s = sum_tx / max(sum_rx, EPS)
        s_bounded = clamp(s, 0.5, 2.0)
        alpha = clamp(imb / 0.15, 0.25, 0.6)
        k = 1.0 + alpha * (s_bounded - 1.0)
        s_bounded_map[rid] = s_bounded

        for i in up_ifaces:
            if dir_to_scale == 'tx':
                old = post_router[i]['tx']
                new = old * k
                # Cap multiplicative impact to ±15% and abs 0.15*max(v,1.0)
                cap_hi = old * 1.15
                cap_lo = old * 0.85
                new = clamp(new, cap_lo, cap_hi)
                # Absolute cap again (matches ±15% floor even for tiny links)
                cap_abs = 0.15 * max(old, 1.0)
                new = clamp(new, old - cap_abs, old + cap_abs)
                post_router[i]['tx'] = max(0.0, new)
                delta_abs = abs(new - old)
                router_delta_abs[i]['tx'] += delta_abs
                router_delta_rel[i]['tx'] += delta_abs / max(old, 1.0)
            else:
                old = post_router[i]['rx']
                new = old * k
                cap_hi = old * 1.15
                cap_lo = old * 0.85
                new = clamp(new, cap_lo, cap_hi)
                cap_abs = 0.15 * max(old, 1.0)
                new = clamp(new, old - cap_abs, old + cap_abs)
                post_router[i]['rx'] = max(0.0, new)
                delta_abs = abs(new - old)
                router_delta_abs[i]['rx'] += delta_abs
                router_delta_rel[i]['rx'] += delta_abs / max(old, 1.0)

    # Staged additive redistribution (Recommendations #2 and #3)
    # Base pre-router values for per-direction budget tracking
    base_pre_router: Dict[str, Dict[str, float]] = {i: {'rx': pre[i]['pre_rx'], 'tx': pre[i]['pre_tx']} for i in graph.ifaces}
    # Track cumulative additive usage for cap/stress decisions
    add_used_abs: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in graph.ifaces}
    cap_stressed: Dict[str, Dict[str, bool]] = {i: {'rx': False, 'tx': False} for i in graph.ifaces}

    stage_caps = [0.25, 0.35]  # pass1, pass2; optional pass3 handles up to 0.45 on selected ifaces

    def redistribute_pass(stage_idx: int, rid: str, allowed_ifaces: List[str]) -> None:
        up_ifaces = [i for i in allowed_ifaces if pre[i]['local_status'] == 'up']
        if len(up_ifaces) < 2:
            return
        sum_rx2 = sum(post_router[i]['rx'] for i in up_ifaces)
        sum_tx2 = sum(post_router[i]['tx'] for i in up_ifaces)
        denom2 = max(sum_rx2, sum_tx2, 1.0)
        imb2 = abs(sum_tx2 - sum_rx2) / denom2
        if (sum_rx2 + sum_tx2) <= 1e-9 or imb2 <= HARDENING_THRESHOLD:
            return

        # Choose direction and needed delta to reduce imbalance
        if sum_tx2 > sum_rx2:
            dir_to_adj = 'tx'
            need = -(sum_tx2 - sum_rx2)  # decrease TX
        else:
            dir_to_adj = 'rx'
            need = +(sum_rx2 - sum_tx2)  # increase RX

        # Elastic router delta guard (fraction of average up traffic)
        avg_tx_conf = 0.0
        avg_rx_conf = 0.0
        if up_ifaces:
            avg_tx_conf = sum(max(0.0, 1.0 - pre[i]['tx_link_diff']) for i in up_ifaces) / len(up_ifaces)
            avg_rx_conf = sum(max(0.0, 1.0 - pre[i]['rx_link_diff']) for i in up_ifaces) / len(up_ifaces)
        guard_frac = clamp(0.15 + 0.5 * imb2 + 0.5 * abs(avg_tx_conf - avg_rx_conf), 0.15, 0.35)
        avg_traffic = (sum_rx2 + sum_tx2) / 2.0
        router_cap_abs = guard_frac * avg_traffic
        need = clamp(need, -router_cap_abs, router_cap_abs)

        # Prepare values and per-interface remaining caps for this stage
        vals = {i: post_router[i][dir_to_adj] for i in up_ifaces}
        sum_v = max(1e-9, sum(vals.values()))
        weights: Dict[str, float] = {}
        remaining_caps: Dict[str, float] = {}

        for i in up_ifaces:
            v_i = vals[i]
            # Directional confidence proxy
            conf_dir = max(0.0, 1.0 - (pre[i]['tx_link_diff'] if dir_to_adj == 'tx' else pre[i]['rx_link_diff']))
            # Pair residual severity after current router edits
            peer = graph.get_peer(i)
            if peer and pre[peer]['local_status'] == 'up':
                if dir_to_adj == 'tx':
                    resid_pair = rel_diff(post_router[i]['tx'], post_router[peer]['rx'])
                    tol_pair = pair_tolerance(post_router[i]['tx'], post_router[peer]['rx'])
                else:
                    resid_pair = rel_diff(post_router[i]['rx'], post_router[peer]['tx'])
                    tol_pair = pair_tolerance(post_router[i]['rx'], post_router[peer]['tx'])
                sev = min(2.0, resid_pair / max(tol_pair, EPS))
            else:
                sev = 0.0
                resid_pair = 0.0
                tol_pair = HARDENING_THRESHOLD

            vol_term = v_i / sum_v
            w = 0.6 * (1.0 - conf_dir) + 0.25 * sev + 0.15 * vol_term
            if v_i < 1.0:
                w *= 0.5
            # Reduce weight if previously consumed >70% of total cap across passes
            total_cap_allowed = 0.45 * max(base_pre_router[i][dir_to_adj], 1.0)
            used = add_used_abs[i][dir_to_adj]
            if total_cap_allowed > 0 and (used / total_cap_allowed) >= 0.70:
                w *= 0.7
            # Boost for very inconsistent pairs
            if resid_pair > 2.0 * tol_pair:
                w += 0.1
            weights[i] = max(0.02, w)

            # Remaining per-interface cap for this stage (cumulative limit up to stage_caps[stage_idx])
            stage_cap_abs = stage_caps[stage_idx] * max(base_pre_router[i][dir_to_adj], 1.0)
            rem_cap = max(0.0, stage_cap_abs - add_used_abs[i][dir_to_adj])
            remaining_caps[i] = rem_cap

        # Iterative distribution honoring per-interface caps
        applied: Dict[str, float] = {i: 0.0 for i in up_ifaces}
        remaining = need
        for _ in range(5):
            active = [i for i in up_ifaces if remaining_caps[i] > 1e-12]
            if not active or abs(remaining) <= 1e-9:
                break
            wsum = sum(weights[i] for i in active)
            if wsum <= 0.0:
                break
            for i in active:
                if abs(remaining) <= 1e-9:
                    break
                share = weights[i] / wsum
                prop = remaining * share
                # Clamp per-interface by remaining cap
                prop = clamp(prop, -remaining_caps[i], remaining_caps[i])
                applied[i] += prop
                remaining -= prop
                remaining_caps[i] -= abs(prop)

        # Apply and update accounting
        for i, d in applied.items():
            if abs(d) <= 0.0:
                continue
            oldv = post_router[i][dir_to_adj]
            post_router[i][dir_to_adj] = max(0.0, oldv + d)
            add_used_abs[i][dir_to_adj] += abs(d)
            # Flag stress if consumed >70% of total cap (0.45 of base)
            total_cap_allowed = 0.45 * max(base_pre_router[i][dir_to_adj], 1.0)
            if total_cap_allowed > 0 and (add_used_abs[i][dir_to_adj] / total_cap_allowed) >= 0.70:
                cap_stressed[i][dir_to_adj] = True

    # Run two staged additive passes
    for rid, if_list in graph.routers.items():
        up_ifaces = graph.router_ifaces_up(pre, rid)
        if len(up_ifaces) < 2:
            continue
        redistribute_pass(0, rid, up_ifaces)
        redistribute_pass(1, rid, up_ifaces)

    # Optional pass 3 (up to 0.45) for low-confidence or tiny links where imbalance remains
    for rid, if_list in graph.routers.items():
        up_ifaces = graph.router_ifaces_up(pre, rid)
        if len(up_ifaces) < 2:
            continue
        sum_rx2 = sum(post_router[i]['rx'] for i in up_ifaces)
        sum_tx2 = sum(post_router[i]['tx'] for i in up_ifaces)
        denom2 = max(sum_rx2, sum_tx2, 1.0)
        imb2 = abs(sum_tx2 - sum_rx2) / denom2
        if imb2 <= HARDENING_THRESHOLD:
            continue
        # Filter: only ifaces with conf_dir < 0.6 or v < 5 Mbps in the scaled direction will participate
        # We'll pass all up_ifaces; redistribute_pass will use remaining caps up to 0.45 via remaining_caps calc.
        redistribute_pass(1, rid, up_ifaces)  # reuse stage_idx=1 caps for gate; remaining cap allows up to 0.45 through repeats

    # Merge router deltas (multiplicative + additive) for calibration
    for i in graph.ifaces:
        for d in ('rx', 'tx'):
            delta_abs = add_used_abs[i][d]
            base = max(base_pre_router[i][d], 1.0)
            if delta_abs > 0:
                router_delta_rel[i][d] += delta_abs / base
                router_delta_abs[i][d] += delta_abs

    # -------- Stage 3: Targeted pair reconciliation (Recommendation #4) --------
    post: Dict[str, Dict[str, float]] = {i: {'rx': post_router[i]['rx'], 'tx': post_router[i]['tx']} for i in graph.ifaces}
    pair_adj_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in graph.ifaces}

    visited_pairs = set()
    for a in graph.ifaces:
        b = graph.get_peer(a)
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)
        if pre[a]['local_status'] != 'up' or pre[b]['local_status'] != 'up':
            continue

        # Only reconcile pairs touched by router edits (preserves calibration on untouched links)
        touched = (router_delta_abs[a]['rx'] > 0 or router_delta_abs[a]['tx'] > 0 or
                   router_delta_abs[b]['rx'] > 0 or router_delta_abs[b]['tx'] > 0)
        if not touched:
            continue

        # A.tx ↔ B.rx
        a_tx, b_rx = post[a]['tx'], post[b]['rx']
        resid_tx = rel_diff(a_tx, b_rx)
        tol_tx = pair_tolerance(a_tx, b_rx)
        if resid_tx > tol_tx:
            conf_a_tx = max(0.0, 1.0 - pre[a]['tx_link_diff'])
            conf_b_rx = max(0.0, 1.0 - pre[b]['rx_link_diff'])
            # Move lower-confidence more
            if conf_a_tx <= conf_b_rx:
                alpha_a, alpha_b = 0.35, 0.20
            else:
                alpha_a, alpha_b = 0.20, 0.35
            move_a = alpha_a * (b_rx - a_tx)
            move_b = alpha_b * (a_tx - b_rx)
            # Remaining cap per-direction = 20% of pre-router minus router edits consumed
            cap_a = max(0.0, 0.20 * max(base_pre_router[a]['tx'], 1.0) - router_delta_abs[a]['tx'])
            cap_b = max(0.0, 0.20 * max(base_pre_router[b]['rx'], 1.0) - router_delta_abs[b]['rx'])
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)
            new_a_tx = max(0.0, a_tx + move_a)
            new_b_rx = max(0.0, b_rx + move_b)
            pair_adj_rel[a]['tx'] = max(pair_adj_rel[a]['tx'], abs(new_a_tx - a_tx) / max(a_tx, 1.0))
            pair_adj_rel[b]['rx'] = max(pair_adj_rel[b]['rx'], abs(new_b_rx - b_rx) / max(b_rx, 1.0))
            post[a]['tx'] = new_a_tx
            post[b]['rx'] = new_b_rx
            # Account pair moves into cumulative router_delta_abs (for calibration util)
            router_delta_abs[a]['tx'] += abs(new_a_tx - a_tx)
            router_delta_abs[b]['rx'] += abs(new_b_rx - b_rx)

        # A.rx ↔ B.tx
        a_rx, b_tx = post[a]['rx'], post[b]['tx']
        resid_rx = rel_diff(a_rx, b_tx)
        tol_rx = pair_tolerance(a_rx, b_tx)
        if resid_rx > tol_rx:
            conf_a_rx = max(0.0, 1.0 - pre[a]['rx_link_diff'])
            conf_b_tx = max(0.0, 1.0 - pre[b]['tx_link_diff'])
            if conf_a_rx <= conf_b_tx:
                alpha_a, alpha_b = 0.35, 0.20
            else:
                alpha_a, alpha_b = 0.20, 0.35
            move_a = alpha_a * (b_tx - a_rx)
            move_b = alpha_b * (a_rx - b_tx)
            cap_a = max(0.0, 0.20 * max(base_pre_router[a]['rx'], 1.0) - router_delta_abs[a]['rx'])
            cap_b = max(0.0, 0.20 * max(base_pre_router[b]['tx'], 1.0) - router_delta_abs[b]['tx'])
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)
            new_a_rx = max(0.0, a_rx + move_a)
            new_b_tx = max(0.0, b_tx + move_b)
            pair_adj_rel[a]['rx'] = max(pair_adj_rel[a]['rx'], abs(new_a_rx - a_rx) / max(a_rx, 1.0))
            pair_adj_rel[b]['tx'] = max(pair_adj_rel[b]['tx'], abs(new_b_tx - b_tx) / max(b_tx, 1.0))
            post[a]['rx'] = new_a_rx
            post[b]['tx'] = new_b_tx
            router_delta_abs[a]['rx'] += abs(new_a_rx - a_rx)
            router_delta_abs[b]['tx'] += abs(new_b_tx - b_tx)

    # -------- Stage 4: Confidence calibration (Recommendation #5) --------
    # Compute post-repair router imbalance
    router_imbalance_post: Dict[str, float] = {}
    for rid, if_list in graph.routers.items():
        up_ifaces = [i for i in if_list if pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_imbalance_post[rid] = 0.0
            continue
        sum_rx = sum(post[i]['rx'] for i in up_ifaces)
        sum_tx = sum(post[i]['tx'] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        router_imbalance_post[rid] = abs(sum_tx - sum_rx) / denom

    # Build final results
    result: Dict[str, Dict[str, Tuple]] = {}
    for iface, data in telemetry.items():
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))
        local_status = pre[iface]['local_status']
        repaired_status = data.get('interface_status', 'unknown')

        # Final repaired values
        rx_repaired = post[iface]['rx']
        tx_repaired = post[iface]['tx']

        # Enforce zero on down interfaces
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Link-based confidence
        peer = graph.get_peer(iface)
        if peer and local_status == 'up' and pre[peer]['local_status'] == 'up':
            rx_resid = rel_diff(rx_repaired, post[peer]['tx'])
            tx_resid = rel_diff(tx_repaired, post[peer]['rx'])
            rx_link_conf = max(0.0, 1.0 - rx_resid)
            tx_link_conf = max(0.0, 1.0 - tx_resid)
        elif norm_status(repaired_status) == 'down':
            rx_link_conf = 0.9 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.9 if tx_repaired == 0.0 else 0.5
        else:
            rx_link_conf = 0.6
            tx_link_conf = 0.6

        # Router factor from post imbalance
        rid = data.get('local_router')
        imb_post = router_imbalance_post.get(rid, 0.0)
        router_factor = max(0.2, 1.0 - imb_post)

        # Change taper penalty (two-slope)
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_weight = 0.4 if rx_change < 0.15 else 0.5
        tx_weight = 0.4 if tx_change < 0.15 else 0.5
        rx_change_factor = max(0.2, 1.0 - rx_weight * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - tx_weight * min(1.0, tx_change))

        rx_conf = rx_link_conf * router_factor * rx_change_factor
        tx_conf = tx_link_conf * router_factor * tx_change_factor

        # Scale intensity penalty using bounded router ratio for this router
        s_b = s_bounded_map.get(rid, 1.0)
        s_dev = abs(1.0 - s_b)
        if s_dev > 0.25:
            rx_conf -= 0.05
            tx_conf -= 0.05
        elif s_dev > 0.15:
            rx_conf -= 0.03
            tx_conf -= 0.03

        # Cap-intensity penalty when interface consumed >70% of cumulative cap
        # Total allowed ≈ router (≤0.45 base) + pair (≤0.20 base)
        total_cap_allowed_rx = 0.65 * max(base_pre_router[iface]['rx'], 1.0)
        total_cap_allowed_tx = 0.65 * max(base_pre_router[iface]['tx'], 1.0)
        used_rx = router_delta_abs[iface]['rx']
        used_tx = router_delta_abs[iface]['tx']
        util_rx = used_rx / total_cap_allowed_rx if total_cap_allowed_rx > 0 else 0.0
        util_tx = used_tx / total_cap_allowed_tx if total_cap_allowed_tx > 0 else 0.0
        if util_rx > 0.70:
            rx_conf -= min(0.08, 0.4 * (util_rx - 0.70) / max(1e-6, 0.30))
        if util_tx > 0.70:
            tx_conf -= min(0.08, 0.4 * (util_tx - 0.70) / max(1e-6, 0.30))

        # Pair reconciliation penalty proportional to relative adjustment
        rx_conf -= min(0.05, 0.25 * pair_adj_rel[iface]['rx'])
        tx_conf -= min(0.05, 0.25 * pair_adj_rel[iface]['tx'])

        # Small bonus for untouched directions across all passes
        rx_untouched = (not pre[iface]['pre_rx_changed']) and (router_delta_rel[iface]['rx'] == 0.0) and (pair_adj_rel[iface]['rx'] == 0.0)
        tx_untouched = (not pre[iface]['pre_tx_changed']) and (router_delta_rel[iface]['tx'] == 0.0) and (pair_adj_rel[iface]['tx'] == 0.0)
        if rx_untouched:
            rx_conf += 0.03
        if tx_untouched:
            tx_conf += 0.03

        rx_confidence = max(0.0, min(1.0, rx_conf))
        tx_confidence = max(0.0, min(1.0, tx_conf))

        # Status confidence (keep status, just calibrate)
        status_confidence = 1.0
        if peer:
            peer_stat = norm_status(telemetry[peer].get('interface_status', 'unknown'))
            if norm_status(repaired_status) != peer_stat:
                status_confidence = min(status_confidence, 0.5)
        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output
        out_entry: Dict[str, Tuple] = {}
        out_entry['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        out_entry['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        out_entry['interface_status'] = (data.get('interface_status', 'unknown'), repaired_status, status_confidence)
        # Copy metadata unchanged
        out_entry['connected_to'] = data.get('connected_to')
        out_entry['local_router'] = data.get('local_router')
        out_entry['remote_router'] = data.get('remote_router')
        result[iface] = out_entry

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