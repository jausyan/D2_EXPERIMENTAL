"""Light local avoidance of CYLINDERS for follow_path (no costmap, no planner).

Every control tick it slices the lidar scan to a small box ahead of the drone, clusters the
points, keeps only the blobs that are cylinder-sized, and returns ONE number: a sideways offset
to add to the next waypoint.

  * cylinder in the corridor ahead -> detour lane around it (big margin), kept until passed
  * gate poles (thin), gate bars / walls (wide), everything else -> ignored
  * nothing                       -> offset fades back to 0 (planned path)

Lateral values are measured against the PLANNED path line (origin + fwd).
"""
import numpy as np

from utils.cylinder import cluster_xy, is_cylinder, quat_to_matrix


class LocalAvoider:
    """Stateful: call update() every tick; read .offset (m, +left of the planned line)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.offset = 0.0
        self.speed_scale = 1.0
        self.state = "CLEAR"          # CLEAR | AVOID | BLOCKED
        self._side_hold_until = 0.0
        self._side = 0                # +1 pass on the left of the obstacle, -1 on the right
        self._last_t = None
        self._last_seen = 0.0
        self._event = None            # one-line debug text, read once via pop_event()
        self._last_event_t = 0.0
        self._prev_state = "CLEAR"
        self._pass = None             # last detour: {"lane": offset, "pts": world xy of what we pass}
        self._lane = None

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
        for c in clusters:           # world xy of each blob (for the pass memory)
            sm, lm = 0.5 * (c["s_min"] + c["s_max"]), 0.5 * (c["l_min"] + c["l_max"])
            c["w"] = np.asarray(pos[:2]) + fwd * sm + left * (lm - pos_l)
        all_clusters = list(clusters)
        self._ctx = (scan, pos, quat, origin, fwd)

        if clusters:
            blockers = self._blockers(clusters, 0.0)
            if blockers:
                target, state, speed = self._avoid(blockers, clusters, now, pos, origin)
        if state == "CLEAR" and self._pass is not None:
            # still beside what we detoured around: keep the lane until it is well behind us
            ahead = max(float((w - np.asarray(pos[:2])) @ fwd) for w in self._pass["pts"])
            if ahead > -cfg["pass_distance"]:
                target, state = self._pass["lane"], "AVOID"
            else:
                self._pass = None
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
        blobs = cluster_xy(s[box], l[box], cfg["cell"], cfg["min_points"], z=p[box, 2])
        return [c for c in blobs if is_cylinder(c, cfg)]      # cylinders only (shared with Nav2)

    def _blockers(self, clusters, centre):
        """Clusters inside the (generous) corridor around lateral `centre`."""
        half = self.cfg["drone_radius"] + self.cfg["obstacle_clearance"]
        return [c for c in clusters if c["l_max"] > centre - half and c["l_min"] < centre + half]

    def _avoid(self, blockers, clusters, now, pos, origin):
        """Pick a lane (lateral offset): around a cylinder with a big margin, or through a wide enough
        gap between two cylinders with a small margin."""
        cfg = self.cfg
        r = cfg["drone_radius"]
        m_out, m_in = r + cfg["obstacle_clearance"], r + cfg["clearance"]
        # merge blobs whose gap is too small to fly through into one obstacle
        ivs = sorted([c["l_min"], c["l_max"]] for c in clusters)
        merged = [ivs[0]]
        for lo, hi in ivs[1:]:
            if lo - merged[-1][1] < 2 * m_in:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        def clear_of(lane, margin_for):
            for k, (lo, hi) in enumerate(merged):
                need = margin_for.get(k, m_in)
                if lane > lo - need and lane < hi + need:
                    return False
            return True

        cands = []                                                    # (cost, lane, unused)
        for k, (lo, hi) in enumerate(merged):
            for lane in (hi + m_out, lo - m_out):                     # around obstacle k, big margin
                if abs(lane) <= cfg["max_shift"] and clear_of(lane, {k: m_out}):
                    cands.append((abs(lane), lane, False))
            if k + 1 < len(merged):                                   # through the gap k -> k+1
                lane = 0.5 * (hi + merged[k + 1][0])
                if abs(lane) <= cfg["max_shift"] and clear_of(lane, {}):
                    cands.append((abs(lane), lane, False))
        near_s = min(c["s_min"] for c in blockers)
        if not cands:
            # no way round: crawl while it is still far, hover (x/y only) when close; re-evaluated every tick
            return self.offset, "BLOCKED", (0.0 if near_s < cfg["stop_distance"] else cfg["min_speed_scale"])
        best = min(cands)
        if self._lane is not None and now < self._side_hold_until:   # do not flip lanes every frame
            same = [c for c in cands if abs(c[1] - self._lane) < 0.4]
            if same:
                best = min(same, key=lambda c: abs(c[1] - self._lane))
        if self._lane is None or abs(best[1] - self._lane) >= 0.4:
            self._side_hold_until = now + cfg["side_hold"]
        self._lane = best[1]
        self._pass = {"lane": best[1], "pts": [c["w"] for c in clusters]}
        speed = float(np.clip((near_s - cfg["s_min"]) / max(cfg["slow_distance"], 1e-3), cfg["min_speed_scale"], 1.0))
        # slow down only while the drone is still moving over to the lane; once it is on it, go on
        speed = 1.0 - (1.0 - speed) * min(1.0, abs(best[1] - self.offset) / 0.5)
        return best[1], "AVOID", speed
