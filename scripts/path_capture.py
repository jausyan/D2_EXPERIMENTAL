"""Path planning tool: fly/move the drone by hand and record waypoints.

Subscribes to the drone's local pose and shows live position, distance moved and path length.

    c  capture the current position as a waypoint
    u  undo the last waypoint
    h  set HOME (0, 0, 0) to the current position, x axis = the drone's current heading
    p  print the path (numpy array + config.yaml `path.waypoints` format)
    q  quit

Coordinates are relative to home, with x pointing where the drone faced when `h` was pressed.
That is the same "gate frame" the waypoints in config/config.yaml are written in. Z is height
above home, so press `h` while standing on the ground at the start point.
Pressing `h` again re-expresses every captured waypoint relative to the new home.

RViz (config/path.rviz), fixed frame `map`, shows everything relative to home:
    /path_capture/path     captured waypoints line
    /path_capture/markers  numbered waypoint spheres + home axes
    /path_capture/drone    drone pose
    /path_capture/trail    where the drone has been

    ros2 run rviz2 rviz2 -d config/path.rviz &
    python3 scripts/path_capture.py            # topic from config/config.yaml
    python3 scripts/path_capture.py --topic /rian/local_position/pose
"""
import argparse
import math
import select
import sys
import termios
import tty
from pathlib import Path as FsPath

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray

ROOT = FsPath(__file__).resolve().parent.parent
FRAME = "map"


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def default_topic():
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
        return f"{cfg['node']['mavros_namespace']}/local_position/pose"
    except Exception:
        return "/mavros/local_position/pose"


class PathCapture(Node):
    def __init__(self, topic):
        super().__init__("path_capture")
        self.pose = None                       # latest PoseStamped
        self.home = np.zeros(3)                # map frame
        self.home_yaw = 0.0
        self.points = []                       # captured waypoints, absolute (map frame)
        self.trail = Path()
        self.trail.header.frame_id = FRAME
        self.last_trail_xy = None

        self.create_subscription(PoseStamped, topic, self.pose_cb, qos_profile_sensor_data)
        self.path_pub = self.create_publisher(Path, "/path_capture/path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/path_capture/markers", 10)
        self.drone_pub = self.create_publisher(PoseStamped, "/path_capture/drone", 10)
        self.trail_pub = self.create_publisher(Path, "/path_capture/trail", 10)
        self.create_timer(0.1, self.publish_drone)
        self.create_timer(0.5, self.publish_path)

    # ---- frames ---------------------------------------------------------- #
    def to_home(self, p):
        """Map-frame point -> home frame (x along the heading at home)."""
        d = np.asarray(p, dtype=float) - self.home
        c, s = math.cos(self.home_yaw), math.sin(self.home_yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

    def position(self):
        p = self.pose.pose.position
        return np.array([p.x, p.y, p.z])

    def rel_points(self):
        return np.array([self.to_home(p) for p in self.points]).reshape(-1, 3)

    # ---- callbacks ------------------------------------------------------- #
    def pose_cb(self, msg):
        self.pose = msg

    def publish_drone(self):
        if self.pose is None:
            return
        rel = self.to_home(self.position())
        yaw = yaw_of(self.pose.pose.orientation) - self.home_yaw
        msg = PoseStamped()
        msg.header.frame_id = FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, rel)
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)
        self.drone_pub.publish(msg)

        if self.last_trail_xy is None or np.linalg.norm(rel - self.last_trail_xy) > 0.05:
            self.last_trail_xy = rel
            self.trail.header.stamp = msg.header.stamp
            self.trail.poses.append(msg)
            self.trail.poses = self.trail.poses[-5000:]
            self.trail_pub.publish(self.trail)

    def publish_path(self):
        now = self.get_clock().now().to_msg()
        rel = self.rel_points()
        path = Path()
        path.header.frame_id, path.header.stamp = FRAME, now
        for p in rel:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, p in enumerate(rel):
            sphere = Marker()
            sphere.header.frame_id, sphere.header.stamp = FRAME, now
            sphere.ns, sphere.id = "waypoint", i
            sphere.type, sphere.action = Marker.SPHERE, Marker.ADD
            sphere.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (0.1, 0.9, 0.2, 1.0)
            arr.markers.append(sphere)
            label = Marker()
            label.header = sphere.header
            label.ns, label.id = "label", i
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.25)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = str(i)
            arr.markers.append(label)
        # home axes: x red, y green, z blue
        for k, (col, d) in enumerate([((1, 0, 0), (1, 0, 0)), ((0, 1, 0), (0, 1, 0)), ((0, 0, 1), (0, 0, 1))]):
            axis = Marker()
            axis.header.frame_id, axis.header.stamp = FRAME, now
            axis.ns, axis.id = "home", k
            axis.type, axis.action = Marker.ARROW, Marker.ADD
            axis.points = [Point(), Point(x=float(d[0]), y=float(d[1]), z=float(d[2]))]
            axis.scale.x, axis.scale.y, axis.scale.z = 0.05, 0.1, 0.1
            axis.color.r, axis.color.g, axis.color.b, axis.color.a = (*map(float, col), 1.0)
            arr.markers.append(axis)
        self.marker_pub.publish(arr)

    # ---- keys ------------------------------------------------------------ #
    def require_pose(self):
        if self.pose is None:
            self.say("no pose received yet, is the drone / sim running?")
            return False
        return True

    def say(self, text):
        sys.stdout.write(f"\r\033[K{text}\n")
        sys.stdout.flush()

    def handle_key(self, key):
        if key == "c" and self.require_pose():
            self.points.append(self.position())
            r = self.to_home(self.points[-1])
            self.say(f"captured #{len(self.points) - 1}: [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}]")
        elif key == "u":
            if self.points:
                r = self.to_home(self.points.pop())
                self.say(f"undo: removed [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}], {len(self.points)} left")
            else:
                self.say("nothing to undo")
        elif key == "h" and self.require_pose():
            self.home = self.position()
            self.home_yaw = yaw_of(self.pose.pose.orientation)
            self.trail.poses.clear()
            self.last_trail_xy = None
            self.say(f"home set at map [{self.home[0]:.2f}, {self.home[1]:.2f}, {self.home[2]:.2f}] "
                     f"yaw {math.degrees(self.home_yaw):.1f} deg, x axis = current heading")
        elif key == "p":
            self.print_path()

    def print_path(self):
        rel = self.rel_points()
        if len(rel) == 0:
            self.say("no waypoints captured yet")
            return
        lines = ["", "# numpy", "path = np.array(["]
        lines += [f"    [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]," for p in rel]
        lines += ["])", "", "# config/config.yaml", "path:", "  waypoints:"]
        lines += [f"    - [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]" for p in rel]
        lines.append(f"# {len(rel)} points, length {self.length(rel):.2f} m")
        self.say("\n".join(lines))

    @staticmethod
    def length(rel):
        return float(np.linalg.norm(np.diff(rel, axis=0), axis=1).sum()) if len(rel) > 1 else 0.0

    def status(self):
        if self.pose is None:
            return "waiting for pose..."
        rel = self.to_home(self.position())
        pts = self.rel_points()
        since = f"{np.linalg.norm(rel - pts[-1]):.2f} m from last" if len(pts) else "no points"
        yaw = math.degrees(yaw_of(self.pose.pose.orientation) - self.home_yaw)
        return (f"pos [{rel[0]:7.2f} {rel[1]:7.2f} {rel[2]:6.2f}]  yaw {yaw:6.1f}  "
                f"{since}  | {len(pts)} pts, {self.length(pts):.2f} m   [c]apture [u]ndo [h]ome [p]rint [q]uit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=None, help="local pose topic (default: from config/config.yaml)")
    args = ap.parse_args()
    topic = args.topic or default_topic()

    rclpy.init()
    node = PathCapture(topic)
    print(f"Listening to {topic}\n  c capture | u undo | h set home | p print | q quit\n")

    tty_in = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if tty_in else None
    try:
        if tty_in:
            tty.setcbreak(sys.stdin.fileno())      # keys without Enter, output stays normal
        tick = 0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if select.select([sys.stdin], [], [], 0)[0]:
                chunk = sys.stdin.read(1) if tty_in else sys.stdin.readline()
                if chunk == "":
                    break
                for key in chunk.lower():
                    if key == "q":
                        raise KeyboardInterrupt
                    node.handle_key(key)
            tick += 1
            if tick % 4 == 0:
                sys.stdout.write("\r\033[K" + node.status()[:200])
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if tty_in:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
