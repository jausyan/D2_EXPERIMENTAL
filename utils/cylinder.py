"""Cylinder detection shared by both mission stacks (own avoider and Nav2).

One place that answers "which lidar blobs are cylinders?", so the size limits are tuned once
(config: avoid.cyl_min / avoid.cyl_max, the same keys for both stacks).
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


def cluster_xy(s, l, cell, min_points, z=None, keep_mask=False):
    """Group 2D points into blobs on a grid. Returns a list of dicts (extents, count[, mask])."""
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
        c = {"count": int(m.sum()),
             "s_min": float(s[m].min()), "s_max": float(s[m].max()),
             "l_min": float(l[m].min()), "l_max": float(l[m].max()),
             "width": float(l[m].max() - l[m].min())}
        if z is not None:
            c["z_min"], c["z_max"] = float(z[m].min()), float(z[m].max())
        if keep_mask:
            c["mask"] = m
        out.append(c)
    return out


def is_cylinder(c, cfg):
    """Cylinder-sized blob? Gate poles are smaller, gate bars / walls are bigger."""
    size = np.hypot(c["s_max"] - c["s_min"], c["l_max"] - c["l_min"])
    return cfg["cyl_min"] <= size <= cfg["cyl_max"]


def level_scan(scan, quat, cfg, ground_z=None):
    """Lidar points rotated to world axes (levelled), relative to the sensor, filtered to the
    height band around the drone and outside the drone's own body. Returns (points, keep_mask)."""
    p = np.asarray(scan, dtype=float)[:, :3] @ quat_to_matrix(quat).T
    r = np.linalg.norm(p[:, :2], axis=1)
    keep = (np.abs(p[:, 2] + cfg["sensor_z"]) <= cfg["z_band"]) & (r > cfg["body_radius"])
    return p, keep


def cylinder_points(scan, quat, cfg):
    """Points belonging to cylinder-sized blobs, levelled, relative to the drone (x fwd/world axes).
    Used by the Nav2 bridge. Returns an (M, 3) array (z relative to the drone origin)."""
    p, keep = level_scan(scan, quat, cfg)
    p = p[keep]
    if len(p) == 0:
        return np.zeros((0, 3))
    r = np.linalg.norm(p[:, :2], axis=1)
    p = p[r <= cfg["s_max"]]
    blobs = cluster_xy(p[:, 0], p[:, 1], cfg["cell"], cfg["min_points"], keep_mask=True)
    sel = np.zeros(len(p), dtype=bool)
    for c in blobs:
        if is_cylinder(c, cfg):
            sel |= c["mask"]
    out = p[sel].copy()
    out[:, 2] += cfg["sensor_z"]
    return out
