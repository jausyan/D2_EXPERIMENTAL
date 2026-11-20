"""Turn the arena mesh (Collada .dae) into a point cloud (.pcd), no flying needed.

    python3 scripts/mesh_to_pcd.py ~/px4_ws/safmc2027_arena/meshes/arena_safmc2027.dae maps/arena_map.pcd

Points are sampled uniformly over the triangle surfaces (the arena's own geometry, so it is exact).
The mesh frame is the Gazebo world frame: for the SAFMC arena the origin is the START zone centre and
the start gate sits at x = 2.5, i.e. the same frame as the waypoints in config/config.yaml.

    --density 600     points per m^2 of surface (600 ~ one point every 4 cm)
    --skip line       drop meshes by material name (default: floor markings)
    --floor           add a flat ground grid at z = 0 (not in the mesh)
    --origin X Y YAW  where the drone starts in the world (default 0 0 0); the cloud is moved into
                      that frame, so it matches a FAST-LIO map (whose origin is the start pose)
Then draw on it with:  python3 scripts/path_draw.py --no-align   (the frame is already exact)
"""
import argparse
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def read_dae(path, skip):
    """Return {material: (N, 3, 3) triangles}."""
    root = ET.parse(path).getroot()
    unit = root.find("c:asset/c:unit", NS)
    scale = float(unit.get("meter")) if unit is not None else 1.0
    out = {}
    for mesh in root.iterfind(".//c:geometry/c:mesh", NS):
        sources = {s.get("id"): np.array(s.find("c:float_array", NS).text.split(), dtype=float).reshape(-1, 3)
                   for s in mesh.iterfind("c:source", NS)}
        vertices = {v.get("id"): v.find("c:input[@semantic='POSITION']", NS).get("source")[1:]
                    for v in mesh.iterfind("c:vertices", NS)}
        for tri in mesh.iterfind("c:triangles", NS):
            material = (tri.get("material") or "none").replace("-mat", "")
            if material in skip:
                continue
            inputs = tri.findall("c:input", NS)
            stride = 1 + max(int(i.get("offset")) for i in inputs)
            vin = next(i for i in inputs if i.get("semantic") == "VERTEX")
            src = vin.get("source")[1:]
            pos = sources[vertices.get(src, src)] * scale
            idx = np.array(tri.find("c:p", NS).text.split(), dtype=int).reshape(-1, stride)[:, int(vin.get("offset"))]
            out.setdefault(material, []).append(pos[idx].reshape(-1, 3, 3))
    return {m: np.vstack(t) for m, t in out.items()}


def sample_surface(tris, density, rng):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    counts = rng.poisson(area * density)
    idx = np.repeat(np.arange(len(tris)), counts)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[idx] + r1[:, None] * (b[idx] - a[idx]) + r2[:, None] * (c[idx] - a[idx])


def write_pcd(path, pts):
    pts = np.asarray(pts, dtype="<f4")
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
           f"COUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(pts.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dae")
    ap.add_argument("pcd")
    ap.add_argument("--density", type=float, default=600.0)
    ap.add_argument("--skip", nargs="*", default=["line"], help="material names to leave out")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), default=[0.0, 0.0, 0.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    parts = read_dae(args.dae, set(args.skip))
    clouds = []
    for name, tris in parts.items():
        pts = sample_surface(tris, args.density, rng)
        print(f"{name:8s} {len(tris):6d} triangles -> {len(pts):7d} points")
        clouds.append(pts)
    pts = np.vstack(clouds)
    if args.floor:
        lo, hi = pts.min(0), pts.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.1), np.arange(lo[1], hi[1], 0.1))
        pts = np.vstack([pts, np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]])

    x0, y0, yaw = args.origin[0], args.origin[1], math.radians(args.origin[2])
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts - np.array([x0, y0, 0.0])
    pts = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]]
    write_pcd(args.pcd, pts)
    print(f"{len(pts)} points, x {pts[:,0].min():.1f}..{pts[:,0].max():.1f}, y {pts[:,1].min():.1f}..{pts[:,1].max():.1f}, "
          f"z {pts[:,2].min():.2f}..{pts[:,2].max():.2f}  ->  {args.pcd}")


if __name__ == "__main__":
    main()
