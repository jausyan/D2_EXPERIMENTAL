"""Draw the flight path on the arena map in RViz2, then publish / save it.

    1. python3 scripts/path_draw.py                       (needs maps/arena_map.pcd from FAST-LIO)
    2. rviz2 -d config/path_draw.rviz
    3. In RViz pick the "Publish Point" tool (top toolbar) and click on the map: every click adds
       a waypoint (x, y from the click, z = flight.cruise_altitude).
    4. In the terminal:   u undo | x clear | l load config waypoints | p publish + save | q quit
       altitude (height above ground, per waypoint):  + / -  raise / lower the last waypoint by 0.1 m
                                                      z      type an altitude for any waypoint
       New clicks use the altitude of the last waypoint (start: flight.cruise_altitude).

The map is shown in the *track frame* (the frame config/config.yaml waypoints are written in):
the gate is found in the map with the same alignment the mission uses and the map is moved so the
gate centre sits at gate.center. So what you draw can be pasted into config/config.yaml as is.
Use --no-align to draw in the raw FAST-LIO frame (origin = drone start) instead.

Topics (all frame `map`, latched):
    /path_draw/arena      solid arena mesh (--mesh)  /path_draw/ground    ground plane
    /path_draw/map        arena point cloud          /path_draw/gate      gate model (red)
    /path_draw/waypoints  numbered markers           /path_draw/path      smoothed spline (nav_msgs/Path)
    /path_draw/polyline   straight waypoint line     /path_draw/clearance spline coloured by wall clearance
`p` also writes config/path_drawn.yaml and prints the `path:` block for config/config.yaml.
"""
import argparse
import re
import select
import sys
import termios
import tty
from pathlib import Path as FsPath

import numpy as np
import pygicp
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROOT = FsPath(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.utils_ import align_gate, load_config, opening_centre, resolve_path, smooth_path  # noqa: E402

FRAME = "map"
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
PCD_TYPES = {"F": "f", "U": "u", "I": "i"}


def read_pcd(path):
    """x, y, z of a PCD file (ascii or binary, any extra fields)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline().decode().strip()
            if not line or line.startswith("#"):
                continue
            key, *val = line.split()
            hdr[key] = val
            if key == "DATA":
                break
        names = [n if n != "_" else f"pad{i}" for i, n in enumerate(hdr["FIELDS"])]
        dtype = np.dtype([(n, PCD_TYPES[t] + str(s)) for n, t, s in zip(names, hdr["TYPE"], hdr["SIZE"])])
        n = int(hdr["POINTS"][0])
        if hdr["DATA"][0] == "binary":
            arr = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
            return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        if hdr["DATA"][0] == "ascii":
            arr = np.loadtxt(f, ndmin=2)
            ix = [hdr["FIELDS"].index(c) for c in "xyz"]
            return arr[:, ix].astype(np.float64)
    raise ValueError(f"unsupported PCD DATA type in {path}")


def make_cloud(points, stamp):
    pts = np.asarray(points, dtype=np.float32)
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = FRAME, stamp
    msg.height, msg.width = 1, len(pts)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    msg.is_bigendian, msg.point_step = False, 12
    msg.row_step, msg.is_dense = 12 * len(pts), True
    msg.data = pts.tobytes()
    return msg


class PathDraw(Node):
    def __init__(self, args):
        super().__init__("path_draw")
        self.cfg = load_config()
        self.out_file = ROOT / "config" / "path_drawn.yaml"
        self.altitude = self.cfg["flight"]["cruise_altitude"]   # altitude for the next clicked waypoint
        self.tty_old = None
        self.keep_z = args.keep_z
        self.waypoints = []                      # track frame (x, y, z)

        self.arena_pub = self.create_publisher(MarkerArray, "/path_draw/arena", LATCHED)
        self.floor_pub = self.create_publisher(PointCloud2, "/path_draw/floor", LATCHED)
        self.map_pub = self.create_publisher(PointCloud2, "/path_draw/map", LATCHED)
        self.gate_pub = self.create_publisher(PointCloud2, "/path_draw/gate", LATCHED)
        self.wp_pub = self.create_publisher(MarkerArray, "/path_draw/waypoints", LATCHED)
        self.path_pub = self.create_publisher(Path, "/path_draw/path", LATCHED)
        self.poly_pub = self.create_publisher(Path, "/path_draw/polyline", LATCHED)
        self.clear_pub = self.create_publisher(Marker, "/path_draw/clearance", LATCHED)
        self.create_subscription(PointStamped, "/clicked_point", self.clicked_cb, 10)

        self.load_map(args)
        self.publish_arena(args)
        if args.load:
            self.load_config_waypoints()
        self.redraw()

    # ---- map + gate ------------------------------------------------------ #
    def load_map(self, args):
        stamp = self.get_clock().now().to_msg()
        raw = read_pcd(args.map)
        self.say(f"map: {len(raw)} points from {args.map}")
        pts = pygicp.downsample(raw, args.voxel)
        gate_cfg = self.cfg["gate"]
        model = np.load(resolve_path(gate_cfg["pcd_path"])).astype(np.float64)
        mc = gate_cfg.get("model_center", "auto")
        model_center = opening_centre(model) if mc in (None, "auto") else np.array(mc, dtype=float)
        track_center = np.array(gate_cfg["center"], dtype=float)
        shift = track_center - model_center

        if args.no_align:
            self.to_track = np.eye(4)
            self.say("--no-align: drawing in the map frame as it is")
        else:
            T, score = align_gate(pts, model, self.cfg)
            self.say(f"gate found in the map, alignment score {score:.2f}")
            if score < self.cfg["align"]["min_score"]:
                self.say("WARNING: low score, the gate may be wrong. Check the red gate cloud, or use --no-align.")
            shift_T = np.eye(4)
            shift_T[:3, 3] = -shift                  # track frame -> model frame
            M = T @ shift_T                          # track frame -> map frame
            self.to_track = np.linalg.inv(M)         # map frame -> track frame
        track_pts = pts @ self.to_track[:3, :3].T + self.to_track[:3, 3]
        # model in the track frame: it sits where gate.center says
        gate_in_track = model + shift
        self.map_pts = track_pts
        self.obstacles = cKDTree(track_pts[(track_pts[:, 2] > 0.2) & (track_pts[:, 2] < 2.5)]) \
            if np.any((track_pts[:, 2] > 0.2) & (track_pts[:, 2] < 2.5)) else None
        self.map_pub.publish(make_cloud(track_pts, stamp))
        self.gate_pub.publish(make_cloud(gate_in_track, stamp))

    def publish_arena(self, args):
        """Solid, coloured arena (the real mesh) plus a ground plane, easy to read from any angle."""
        mesh = args.mesh or self.cfg.get("arena", {}).get("mesh")
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        # ground over the WHOLE arena: configured bounds plus whatever the map covers, plus a margin
        x0, x1, y0, y1 = self.cfg.get("arena", {}).get("bounds", [-2.5, 17.5, -17.5, 2.5])
        lo, hi = self.map_pts.min(0), self.map_pts.max(0)
        margin = 1.5
        x0, x1 = min(x0, lo[0]) - margin, max(x1, hi[0]) + margin
        y0, y1 = min(y0, lo[1]) - margin, max(y1, hi[1]) + margin
        ground = Marker()
        ground.header.frame_id, ground.header.stamp = FRAME, now
        ground.ns, ground.id, ground.type, ground.action = "ground", 0, Marker.CUBE, Marker.ADD
        ground.pose.position = Point(x=float((x0 + x1) / 2), y=float((y0 + y1) / 2), z=-0.02)
        ground.pose.orientation.w = 1.0
        ground.scale.x, ground.scale.y, ground.scale.z = float(x1 - x0), float(y1 - y0), 0.02
        ground.color = ColorRGBA(r=0.55, g=0.55, b=0.58, a=1.0)
        arr.markers.append(ground)
        if mesh and FsPath(mesh).expanduser().exists():
            view = self.lit_copy(FsPath(mesh).expanduser())
            m = Marker()
            m.header.frame_id, m.header.stamp = FRAME, now
            m.ns, m.id, m.type, m.action = "arena", 1, Marker.MESH_RESOURCE, Marker.ADD
            m.mesh_resource = "file://" + str(view)
            m.mesh_use_embedded_materials = True
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)     # a = 0: use the mesh's own colours
            arr.markers.append(m)
            self.say(f"arena mesh: {mesh}")
        else:
            self.say("no arena mesh given (config arena.mesh or --mesh), showing the point cloud only")
        self.arena_pub.publish(arr)
        # RViz's Publish Point tool only hits things drawn as a selectable point cloud, so the floor is
        # also a dense grid of points (every 10 cm): a click anywhere on the ground registers.
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-6, 0.1), np.arange(y0, y1 + 1e-6, 0.1))
        self.floor_pub.publish(make_cloud(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)], now))
        self.say(f"ground covers x {x0:.1f}..{x1:.1f}, y {y0:.1f}..{y1:.1f} ({gx.size} click points)")

    @staticmethod
    def lit_copy(dae):
        """A copy of the mesh with self-lit colours: faces whose normals point inward (or that are seen
        from behind) stay coloured instead of turning black."""
        text = dae.read_text()

        def glow(match):
            block = match.group(0)
            diffuse = re.search(r'<diffuse><color sid="diffuse">([\d.\s-]+)</color>', block)
            if not diffuse:
                return block
            r, g, b = (float(v) * 0.6 for v in diffuse.group(1).split()[:3])
            return re.sub(r'<emission><color sid="emission">[^<]*</color></emission>',
                          f'<emission><color sid="emission">{r:.3f} {g:.3f} {b:.3f} 1</color></emission>', block)

        text = re.sub(r"<effect .*?</effect>", glow, text, flags=re.S)
        out = ROOT / "maps" / ".arena_view.dae"
        out.parent.mkdir(exist_ok=True)
        out.write_text(text)
        return out

    # ---- editing --------------------------------------------------------- #
    def clicked_cb(self, msg):
        z = msg.point.z if self.keep_z else self.altitude
        self.waypoints.append(np.array([msg.point.x, msg.point.y, z]))
        self.say(f"waypoint #{len(self.waypoints) - 1}: [{msg.point.x:.2f}, {msg.point.y:.2f}, {z:.2f}]")
        self.redraw()

    def handle_key(self, key):
        if key == "u":
            if self.waypoints:
                self.waypoints.pop()
                self.say(f"undo, {len(self.waypoints)} waypoints left")
                self.redraw()
        elif key in ("+", "="):
            self.nudge(0.1)
        elif key == "-":
            self.nudge(-0.1)
        elif key == "z":
            self.ask_altitude()
        elif key == "x":
            self.waypoints.clear()
            self.say("cleared")
            self.redraw()
        elif key == "l":
            self.load_config_waypoints()
            self.redraw()
        elif key == "p":
            self.publish_and_save()

    def nudge(self, dz):
        if not self.waypoints:
            self.altitude = max(0.0, round(self.altitude + dz, 2))
            self.say(f"next waypoint altitude {self.altitude:.2f} m")
            return
        self.waypoints[-1][2] = max(0.0, round(self.waypoints[-1][2] + dz, 2))
        self.altitude = self.waypoints[-1][2]
        self.say(f"waypoint #{len(self.waypoints) - 1} altitude {self.altitude:.2f} m (new clicks use it too)")
        self.redraw()

    def ask_altitude(self):
        if not self.waypoints:
            self.say("no waypoints yet")
            return
        if self.tty_old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.tty_old)    # normal line input for the prompt
        try:
            sys.stdout.write(f"\r\033[Kwaypoint number [Enter = last, {len(self.waypoints) - 1}] and altitude in m, "
                             "e.g. '4 0.3' or just '0.3': ")
            sys.stdout.flush()
            parts = sys.stdin.readline().split()
        finally:
            if self.tty_old is not None:
                tty.setcbreak(sys.stdin.fileno())
        try:
            if len(parts) == 1:
                idx, z = len(self.waypoints) - 1, float(parts[0])
            elif len(parts) == 2:
                idx, z = int(parts[0]), float(parts[1])
            else:
                return
            self.waypoints[idx][2] = max(0.0, z)
        except (ValueError, IndexError):
            self.say("not understood, nothing changed")
            return
        self.altitude = self.waypoints[idx][2]
        self.say(f"waypoint #{idx} altitude {self.altitude:.2f} m")
        self.redraw()

    def load_config_waypoints(self):
        self.waypoints = [np.array(p, dtype=float) for p in load_config()["path"]["waypoints"]]
        self.say(f"loaded {len(self.waypoints)} waypoints from config/config.yaml")

    # ---- publishing ------------------------------------------------------ #
    def spline(self):
        if len(self.waypoints) < 2:
            return None
        pc = self.cfg["path"]
        return smooth_path(np.array(self.waypoints), pc["smoothing_factor"], pc["num_samples"])[:, :3]

    def clearance(self, spl):
        if self.obstacles is None:
            return np.full(len(spl), np.inf)
        return self.obstacles.query(spl)[0]

    def redraw(self):
        now = self.get_clock().now().to_msg()
        wp = np.array(self.waypoints).reshape(-1, 3)
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        for i, p in enumerate(wp):
            sphere = Marker()
            sphere.header.frame_id, sphere.header.stamp = FRAME, now
            sphere.ns, sphere.id, sphere.type = "waypoint", i, Marker.SPHERE
            sphere.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
            sphere.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=1.0)
            arr.markers.append(sphere)
            label = Marker()
            label.header = sphere.header
            label.ns, label.id, label.type = "label", i, Marker.TEXT_VIEW_FACING
            label.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.3)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.3
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"{i}  z={p[2]:.2f}"
            arr.markers.append(label)
        self.wp_pub.publish(arr)

        poly = Path()
        poly.header.frame_id, poly.header.stamp = FRAME, now
        for p in wp:
            poly.poses.append(self.pose(p, now))
        self.poly_pub.publish(poly)

        spl = self.spline()
        path = Path()
        path.header.frame_id, path.header.stamp = FRAME, now
        clear = Marker()
        clear.header.frame_id, clear.header.stamp = FRAME, now
        clear.ns, clear.id, clear.type, clear.action = "clearance", 0, Marker.LINE_STRIP, Marker.ADD
        clear.pose.orientation.w = 1.0
        clear.scale.x = 0.08
        if spl is not None:
            dist = self.clearance(spl)
            min_clear = self.cfg.get("path", {}).get("min_clearance", 0.5)
            for p, d in zip(spl, dist):
                path.poses.append(self.pose(p, now))
                clear.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
                clear.colors.append(ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0) if d < min_clear
                                    else ColorRGBA(r=0.1, g=0.8, b=1.0, a=1.0))
            self.last_info = (float(np.linalg.norm(np.diff(spl, axis=0), axis=1).sum()), float(dist.min()), min_clear)
        else:
            self.last_info = None
        self.path_pub.publish(path)
        self.clear_pub.publish(clear)
        if self.last_info:
            length, dmin, lim = self.last_info
            flag = "OK" if dmin >= lim else f"TOO CLOSE (< {lim} m, shown red)"
            self.say(f"path {length:.1f} m, closest wall {dmin:.2f} m {flag}")

    @staticmethod
    def pose(p, stamp):
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = FRAME, stamp
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        ps.pose.orientation.w = 1.0
        return ps

    def publish_and_save(self):
        if not self.waypoints:
            self.say("no waypoints yet")
            return
        wp = [[round(float(v), 3) for v in p] for p in self.waypoints]
        self.redraw()
        with open(self.out_file, "w") as f:
            yaml.safe_dump({"path": {"waypoints": wp}}, f, default_flow_style=None, sort_keys=False)
        lines = ["", f"# saved to {self.out_file}", "# paste into config/config.yaml", "path:", "  waypoints:"]
        lines += [f"    - [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]" for p in wp]
        self.say("\n".join(lines) + "\n  (spline published on /path_draw/path)")

    def say(self, text):
        sys.stdout.write(f"\r\033[K{text}\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "arena_map.pcd"))
    ap.add_argument("--mesh", default=None, help="arena .dae to show as a solid model (default: arena.mesh in config)")
    ap.add_argument("--voxel", type=float, default=0.05, help="map downsampling (m)")
    ap.add_argument("--no-align", action="store_true", help="skip gate alignment, use the raw FAST-LIO frame")
    ap.add_argument("--keep-z", action="store_true", help="use the clicked z instead of flight.cruise_altitude")
    ap.add_argument("--load", action="store_true", help="start from the waypoints in config/config.yaml")
    args = ap.parse_args()
    rclpy.init()
    node = PathDraw(args)
    print("\nIn RViz: Publish Point tool, click on the map.   u undo | x clear | l load config | p publish+save | q quit"
          "\naltitude: + / - raise / lower the last waypoint 0.1 m | z type an altitude\n")
    tty_in = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if tty_in else None
    node.tty_old = old
    try:
        if tty_in:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if select.select([sys.stdin], [], [], 0)[0]:
                chunk = sys.stdin.read(1) if tty_in else sys.stdin.readline()
                if chunk == "":
                    break
                for key in chunk.lower():
                    if key == "q":
                        raise KeyboardInterrupt
                    node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        if tty_in:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
