"""Light local obstacle avoidance for follow_path (no costmap, no planner).

Every control tick it slices the lidar scan to a small box ahead of the drone, clusters the
points, and returns ONE number: a sideways offset to add to the next waypoint.

  * two pole-like clusters whose gap ~ gate width  -> GATE: offset = the gate centre
  * a cluster blocking the corridor (cylinder/wall) -> AVOID: offset = go around the free side
  * nothing                                         -> offset fades back to 0 (planned path)

All lateral values are measured against the PLANNED path line (origin + fwd), so the answer
does not shift when the drone itself moves sideways.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label


def quat_to_matrix(q):
    """q = (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def cluster_xy(s, l, cell, min_points, z=None):
    """Group (s, l) points into blobs on a grid. Returns list of dicts (s/l extents, count)."""
    if len(s) == 0:
        return []
    i = np.floor((s - s.min()) / cell).astype(int)
    j = np.floor((l - l.min()) / cell).astype(int)
    grid = np.zeros((i.max() + 3, j.max() + 3), dtype=bool)
    grid[i + 1, j + 1] = True
    # dilate by one cell so a pole seen as a few scattered returns is still one blob
    labels, n = label(binary_dilation(grid, iterations=1), structure=np.ones((3, 3)))
    point_labels = labels[i + 1, j + 1]
    out = []
    for k in range(1, n + 1):
        m = point_labels == k
        if m.sum() < min_points:
            continue
        out.append({"count": int(m.sum()),
                    "s_min": float(s[m].min()), "s_max": float(s[m].max()),
                    "l_min": float(l[m].min()), "l_max": float(l[m].max()),
                    "width": float(l[m].max() - l[m].min())})
        if z is not None:
            out[-1]["z_min"], out[-1]["z_max"] = float(z[m].min()), float(z[m].max())
    return out


def find_gate(clusters, cfg):
    """Return (pair, centre_l) for the best pair of narrow poles spaced like the gate, else None."""
    poles = [c for c in clusters if c["width"] <= cfg["pole_width"]]
    best = None
    for a in range(len(poles)):
        for b in range(a + 1, len(poles)):
            left, right = (poles[a], poles[b]) if poles[a]["l_min"] > poles[b]["l_min"] else (poles[b], poles[a])
            gap = left["l_min"] - right["l_max"]                      # inner edge to inner edge
            if abs(gap - cfg["gate_width"]) > cfg["gate_tolerance"]:
                continue
            if abs(left["s_min"] - right["s_min"]) > cfg["gate_depth_tolerance"]:
                continue                                              # poles not side by side
            err = abs(gap - cfg["gate_width"])
            if best is None or err < best[0]:
                best = (err, (left, right), 0.5 * (left["l_min"] + right["l_max"]))
    return None if best is None else (best[1], best[2])


class LocalAvoider:
    """Stateful: call update() every tick; read .offset (m, +left of the planned line)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.offset = 0.0
        self.speed_scale = 1.0
        self.state = "CLEAR"          # CLEAR | GATE | AVOID | BLOCKED
        self._side_hold_until = 0.0
        self._side = 0                # +1 pass on the left of the obstacle, -1 on the right
        self._last_t = None
        self._last_seen = 0.0
        self._gate = None             # locked gate: {"c": world xy centre, "poles": [world xy, world xy]}
        self._event = None            # one-line debug text, read once via pop_event()
        self._last_event_t = 0.0
        self._prev_state = "CLEAR"

    # ------------------------------------------------------------------ #
    def update(self, scan, pos, quat, origin, fwd, now):
        """scan: (N,3) lidar xyz in sensor frame; pos: drone xyz (map); quat: (w,x,y,z);
        origin/fwd: a point on the planned path line and its unit xy direction."""
        cfg = self.cfg
        dt = 0.1 if self._last_t is None else min(max(now - self._last_t, 1e-3), 0.5)
        self._last_t = now
        fwd = np.asarray(fwd[:2], dtype=float)
        fwd /= (np.linalg.norm(fwd) + 1e-9)
        left = np.array([-fwd[1], fwd[0]])

        target, state, speed = 0.0, "CLEAR", 1.0
        clusters = self._clusters(scan, pos, quat, origin, fwd, left) if scan is not None else []
        pos_l = float((np.asarray(pos[:2]) - np.asarray(origin[:2])) @ left)
        for c in clusters:           # world xy of each blob, so a gate can be remembered
            sm, lm = 0.5 * (c["s_min"] + c["s_max"]), 0.5 * (c["l_min"] + c["l_max"])
            c["w"] = np.asarray(pos[:2]) + fwd * sm + left * (lm - pos_l)
        all_clusters = list(clusters)
        self._ctx = (scan, pos, quat, origin, fwd)

        gate = find_gate(clusters, cfg) if clusters else None
        if gate is not None:
            (lp, rp), centre = gate
            self._gate = {"c": 0.5 * (lp["w"] + rp["w"]), "poles": [lp["w"], rp["w"]]}
            target, state = centre, "GATE"
            clusters = [c for c in clusters if c is not lp and c is not rp]
        elif self._gate is not None:
            # gate seen earlier: stay locked until it is behind the drone, poles are never obstacles
            if float((self._gate["c"] - np.asarray(pos[:2])) @ fwd) < -cfg["gate_release"]:
                self._gate = None
            else:
                target, state = float((self._gate["c"] - np.asarray(origin[:2])) @ left), "GATE"
                clusters = [c for c in clusters
                            if min(np.linalg.norm(c["w"] - w) for w in self._gate["poles"]) > cfg["pole_mask"]]
        if clusters:
            blockers = self._blockers(clusters, target if state == "GATE" else 0.0)
            if blockers:
                target, state, speed = self._avoid(blockers, clusters, now)
        if state != "CLEAR":
            self._last_seen = now
        self._debug(state, all_clusters, now)

        # hold the last decision briefly so one missed frame does not snap the path back
        if state == "CLEAR" and now - self._last_seen < cfg["hold_time"]:
            target, state = self.offset, self.state
        target = float(np.clip(target, -cfg["max_shift"], cfg["max_shift"]))

        step = cfg["max_rate"] * dt                                   # rate-limited, smooth
        self.offset += float(np.clip(target - self.offset, -step, step))
        self.speed_scale, self.state = speed, state
        return self.offset

    # ------------------------------------------------------------------ #
    def pop_event(self):
        """One short debug line when the state changed (or every few s while stuck), else None."""
        e, self._event = self._event, None
        return e

    def _debug(self, state, clusters, now):
        if not self.cfg.get("debug"):
            return
        changed = state != self._prev_state
        stuck = state in ("BLOCKED", "AVOID") and now - self._last_event_t > self.cfg["debug_period"]
        if changed or stuck:
            near = sorted(clusters, key=lambda c: c["s_min"])[:6]
            blobs = " ".join(f"(s={c['s_min']:.1f} l={0.5*(c['l_min']+c['l_max']):+.2f} w={c['width']:.2f} n={c['count']} z={c.get('z_min', 0):+.2f}..{c.get('z_max', 0):+.2f})"
                             for c in near) or "none"
            tilt = self._tilt()
            self._event = (f"avoid {self._prev_state}->{state} offset {self.offset:+.2f} | tilt r={tilt[0]:+.0f} p={tilt[1]:+.0f} "
                           f"fwd={tilt[2]:+.0f}deg | {len(clusters)} blobs: {blobs}")
            if state == "BLOCKED" and self._prev_state != "BLOCKED":
                self._dump()
            self._last_event_t = now
        self._prev_state = state

    def _tilt(self):
        """roll, pitch of the drone and heading of the leg, degrees (for debugging)."""
        _, pos, q, _, fwd = self._ctx
        w, x, y, z = q
        roll = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
        return roll, pitch, float(np.degrees(np.arctan2(fwd[1], fwd[0])))

    def _dump(self):
        """Save the raw scan + pose of the first few BLOCKED events so they can be replayed offline."""
        d = self.cfg.get("dump_dir")
        if not d or getattr(self, "_dumps", 0) >= self.cfg.get("dump_max", 3):
            return
        import os, time as _t
        os.makedirs(d, exist_ok=True)
        scan, pos, q, origin, fwd = self._ctx
        f = os.path.join(d, f"blocked_{int(_t.time())}.npz")
        np.savez(f, scan=np.asarray(scan), pos=np.asarray(pos), quat=np.asarray(q),
                 origin=np.asarray(origin), fwd=np.asarray(fwd))
        self._dumps = getattr(self, "_dumps", 0) + 1
        self._event += f" | scan saved {f}"

    # ------------------------------------------------------------------ #
    def _clusters(self, scan, pos, quat, origin, fwd, left):
        cfg = self.cfg
        raw = np.asarray(scan, dtype=float)[:, :3]
        if cfg.get("z_mode", "ground") == "ground":
            # scan z = height above the ground (the sim lidar does this): rotate x/y by yaw only and
            # slice around the drone's own height above ground
            w, x, y, z = quat
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            c, sn = np.cos(yaw), np.sin(yaw)
            p = np.c_[c*raw[:, 0] - sn*raw[:, 1], sn*raw[:, 0] + c*raw[:, 1], raw[:, 2] - pos[2]]
        else:
            p = raw @ quat_to_matrix(quat).T                        # scan z relative to the sensor
        r = np.linalg.norm(p[:, :2], axis=1)
        keep = (np.abs(p[:, 2] + cfg["sensor_z"]) <= cfg["z_band"]) & (r > cfg["body_radius"])
        p = p[keep]
        if len(p) == 0:
            return []
        xy = p[:, :2] + np.asarray(pos[:2])                           # world xy
        d = xy - np.asarray(origin[:2])
        s = (xy - np.asarray(pos[:2])) @ fwd                          # distance ahead of the drone
        l = d @ left                                                  # lateral vs planned line
        box = (s >= cfg["s_min"]) & (s <= cfg["s_max"]) & (np.abs(l) <= cfg["l_max"])
        return cluster_xy(s[box], l[box], cfg["cell"], cfg["min_points"], z=p[box, 2])

    def _blockers(self, clusters, centre):
        """Clusters that sit in the corridor the drone needs around lateral `centre`."""
        half = self.cfg["drone_radius"] + self.cfg["clearance"]
        return [c for c in clusters if c["l_max"] > centre - half and c["l_min"] < centre + half]

    def _avoid(self, blockers, clusters, now):
        cfg = self.cfg
        near = min(blockers, key=lambda c: c["s_min"])
        # merge everything overlapping the nearest blocker so a wall made of pieces is one wall
        lo, hi = near["l_min"], near["l_max"]
        for c in blockers:
            lo, hi = min(lo, c["l_min"]), max(hi, c["l_max"])
        margin = cfg["drone_radius"] + cfg["clearance"]
        cands = {+1: hi + margin, -1: lo - margin}                    # lane left / right of it

        def lane_free(l):
            return all(not (c["l_max"] > l - margin and c["l_min"] < l + margin)
                       for c in clusters if c not in blockers)

        ok = {k: v for k, v in cands.items() if abs(v) <= cfg["max_shift"] and lane_free(v)}
        if not ok:
            near_s = min(c["s_min"] for c in blockers)
            # no way round: crawl while it is still far, hover (x/y only) when close; re-evaluated every tick
            return self.offset, "BLOCKED", (0.0 if near_s < cfg["stop_distance"] else cfg["min_speed_scale"])
        if now < self._side_hold_until and self._side in ok:
            side = self._side
        else:
            side = min(ok, key=lambda k: abs(ok[k]))                  # cheaper detour
            if side != self._side:
                self._side_hold_until = now + cfg["side_hold"]
            self._side = side
        dist = max(near["s_min"], 0.0)
        speed = float(np.clip((dist - cfg["s_min"]) / max(cfg["slow_distance"], 1e-3), cfg["min_speed_scale"], 1.0))
        return ok[side], "AVOID", speed
