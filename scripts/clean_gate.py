"""Clean a gate model: remove the floor and stray points, keep only the gate.

    python3 scripts/clean_gate.py data/gate_sim.npy data/gate_sim_clean.npy

1. Floor: find the lowest dense horizontal layer (RANSAC plane with a near-vertical normal),
   then drop every point within --floor-margin of it, or below it.
2. Stray points: keep only the largest connected cluster (points closer than --eps).
"""
import argparse

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def find_floor_height(pts, thresh=0.03, iters=300, seed=0):
    """RANSAC for a horizontal plane in the lowest part of the cloud; returns its z."""
    rng = np.random.default_rng(seed)
    low = pts[pts[:, 2] < np.percentile(pts[:, 2], 40)]
    best_z, best_n = None, 0
    for _ in range(iters):
        z0 = low[rng.integers(len(low)), 2]
        n = np.count_nonzero(np.abs(pts[:, 2] - z0) < thresh)
        if n > best_n:
            best_z, best_n = z0, n
    # refine with the inliers' mean
    inl = pts[np.abs(pts[:, 2] - best_z) < thresh, 2]
    return float(inl.mean()), len(inl)


def largest_cluster(pts, eps):
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return pts[labels == np.bincount(labels).argmax()]


def opening_centre(pts, cell=0.05):
    """Centre of the gate opening: the biggest empty hole, fully enclosed by the frame, seen from the front (y-z)."""
    y0, z0 = pts[:, 1].min(), pts[:, 2].min()
    iy = ((pts[:, 1] - y0) / cell).astype(int)
    iz = ((pts[:, 2] - z0) / cell).astype(int)
    grid = np.zeros((iy.max() + 1, iz.max() + 1), bool)
    grid[iy, iz] = True
    grid = binary_dilation(grid, iterations=1)           # close small gaps in the frame
    labels, n = label(~grid)
    border = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = [(np.count_nonzero(labels == k), k) for k in range(1, n + 1) if k not in border]
    if not holes:
        return None
    _, k = max(holes)
    gy, gz = np.nonzero(labels == k)
    return np.array([pts[:, 0].mean(), y0 + (gy.min() + gy.max() + 1) / 2 * cell, z0 + (gz.min() + gz.max() + 1) / 2 * cell])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--floor-margin", type=float, default=0.08, help="m above the floor plane to cut")
    ap.add_argument("--eps", type=float, default=0.15, help="m, cluster linking distance")
    args = ap.parse_args()

    pts = np.load(args.src).astype(np.float32)
    print(f"input        : {pts.shape}")

    floor_z, n_floor = find_floor_height(pts)
    print(f"floor plane  : z = {floor_z:.3f} ({n_floor} points)")
    pts = pts[pts[:, 2] > floor_z + args.floor_margin]
    print(f"after floor  : {pts.shape}")

    pts = largest_cluster(pts, args.eps)
    print(f"after cluster: {pts.shape}")

    centre = opening_centre(pts)
    if centre is not None:
        print(f"gate opening centre -> model_center: [{centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}]")
    else:
        print("no closed opening found (frame has a gap?), measure model_center by hand")
    np.save(args.dst, pts.astype(np.float32))
    print(f"min {pts.min(0)}  max {pts.max(0)}  centre {(pts.min(0) + pts.max(0)) / 2}")
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
