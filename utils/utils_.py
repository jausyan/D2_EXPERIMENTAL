"""Helper functions and the ROS 2 drone node used by mission.py."""
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pygicp
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

# reliable + transient_local: an order sent before the other drone subscribed is still delivered
COMMAND_QOS = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(path):
    """Resolve a config path relative to the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# Lidar helpers
# --------------------------------------------------------------------------- #
def pointcloud2_to_xyz(msg):
    """sensor_msgs/PointCloud2 -> (N, 3) float32, NaNs dropped."""
    pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float32)


def custom_msg_to_xyz(msg):
    """livox_ros_driver2/CustomMsg -> (N, 3) float32."""
    return np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)


def subscribe_lidar(node, lidar_cfg, callback):
    """Subscribe to the lidar using the message type from config; callback gets (N, 3) xyz."""
    if lidar_cfg["msg_type"] == "pointcloud2":
        return node.create_subscription(
            PointCloud2, lidar_cfg["topic"], lambda m: callback(pointcloud2_to_xyz(m)),
            rclpy.qos.qos_profile_sensor_data)
    if lidar_cfg["msg_type"] == "custom":
        from livox_ros_driver2.msg import CustomMsg  # only needed with the real driver
        return node.create_subscription(
            CustomMsg, lidar_cfg["topic"], lambda m: callback(custom_msg_to_xyz(m)), 10)
    raise ValueError(f"lidar.msg_type must be 'pointcloud2' or 'custom', got {lidar_cfg['msg_type']!r}")


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def quarternion_mult(q, r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]


def point_rotation_by_quarternion(point, q):
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    step1 = quarternion_mult(q, r)
    step2 = quarternion_mult(step1, q_conj)
    return step2[1:]


def factor(p, start=np.array([0.0, 0.0, 0.0]), clip=3.5):
    norm = np.linalg.norm(p - start)
    if norm > clip:
        return 0.0
    return np.exp(-0.5*norm)/10


def yaw_from_orientation(orientation):
    siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.atan2(siny_cosp, cosy_cosp)


def orientation_to_quat(orientation):
    return [orientation.w, orientation.x, orientation.y, orientation.z]


def smooth_path(path, smoothing_factor=0.5, num_samples=200):
    """Fit a spline through `path` and return `num_samples` points as (N, 4) homogeneous."""
    path = np.array(path)
    k = 3 if len(path) >= 4 else min(2, len(path) - 1)
    tck, _ = splprep([path[:, 0], path[:, 1], path[:, 2]], s=smoothing_factor, k=k)
    pts = np.array(splev(np.linspace(0, 1, num_samples), tck)).T
    return np.hstack((pts, np.ones((num_samples, 1))))


def ramer_douglas_peucker(path, epsilon=0.05):
    if len(path) < 3:
        return path
    start, end = path[0], path[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    dmax, index = 0.0, 0

    if line_len > 1e-10:
        line_dir = line_vec / line_len
        for i in range(1, len(path) - 1):
            t = np.clip(np.dot(path[i] - start, line_dir), 0, line_len)
            d = np.linalg.norm(path[i] - (start + t * line_dir))
            if d > dmax:
                dmax, index = d, i
    else:
        for i in range(1, len(path) - 1):
            d = np.linalg.norm(path[i] - start)
            if d > dmax:
                dmax, index = d, i

    if dmax > epsilon:
        left = ramer_douglas_peucker(path[:index + 1], epsilon)
        right = ramer_douglas_peucker(path[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([start, end])


def gate_to_world(point, gate_matrix, pose, altitude=None):
    """Transform a point from the gate frame to the map frame.

    gate_matrix: 4x4 transform from the GICP alignment.
    pose: geometry_msgs/PoseStamped captured when the gate was aligned.
    altitude: if given, overrides the resulting z.
    """
    p = np.append(np.asarray(point, dtype=float)[:3], 1.0)
    p = point_rotation_by_quarternion(gate_matrix @ p, orientation_to_quat(pose.pose.orientation))
    p = np.array(p[:3], dtype=float)
    p[0] += pose.pose.position.x
    p[1] += pose.pose.position.y
    p[2] = altitude if altitude is not None else p[2] + pose.pose.position.z
    return p


# --------------------------------------------------------------------------- #
# Gate alignment
# --------------------------------------------------------------------------- #
def _pose_to_T(t, yaw):
    """Transform that maps model points into the body frame of a drone standing at (t, yaw)
    in the model's capture frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(t, dtype=float)
    return T


def _alignment_score(T, model, tree, radius):
    """Fraction of model points that land within `radius` of a scan point (0..1)."""
    moved = model @ T[:3, :3].T + T[:3, 3]
    dist, _ = tree.query(moved, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(dist)))


def align_gate(scan, model, cfg):
    """Find the transform model -> body frame (4x4) and a quality score in 0..1.

    GICP alone only converges when it starts close to the answer, so many starting poses
    (drone offset forward/back, left/right, yawed) are tried on a coarse cloud, the best ones
    are refined on the full cloud, and the result with the highest score wins.
    """
    a = cfg["align"]
    scan = np.asarray(scan, dtype=np.float64)
    scan = scan[np.linalg.norm(scan, axis=1) < a["max_range"]]
    model = np.asarray(model, dtype=np.float64)
    coarse = a["coarse_voxel"]
    scan_c = pygicp.downsample(scan, coarse)
    model_c = pygicp.downsample(model, coarse)
    tree_c = cKDTree(scan_c)

    candidates = []
    for x, y, yaw in itertools.product(a["search_x"], a["search_y"], a["search_yaw_deg"]):
        T0 = _pose_to_T([x, y, 0.0], math.radians(yaw))
        T = pygicp.align_points(scan_c, model_c, method=cfg["gate"]["align_method"],
                                initial_guess=T0.astype(np.float32),
                                max_correspondence_distance=a["coarse_max_dist"])
        candidates.append((_alignment_score(T, model_c, tree_c, a["coarse_score_radius"]), T))
    candidates.sort(key=lambda c: -c[0])

    tree = cKDTree(scan)
    best_score, best_T = -1.0, None
    for _, T in candidates[:a["refine_top"]]:
        T = pygicp.align_points(scan, model, method=cfg["gate"]["align_method"],
                                initial_guess=T.astype(np.float32),
                                max_correspondence_distance=a["fine_max_dist"])
        score = _alignment_score(T, model, tree, a["fine_score_radius"])
        if score > best_score:
            best_score, best_T = score, T
    return best_T, best_score


# --------------------------------------------------------------------------- #
# Drone node
# --------------------------------------------------------------------------- #
class DroneNode(Node):
    def __init__(self, cfg):
        node_cfg = cfg["node"]
        super().__init__(node_cfg["name"])
        self.cfg = cfg
        self.mavros_namespace = node_cfg["mavros_namespace"]

        self.zero_alt = 0.0
        self.current_pose = None
        self.current_state = None
        self.current_scan = None
        self.gate_pcd = None
        self.gate_center = None
        self.initial_guess = np.eye(4)

        ns = self.mavros_namespace
        self.vel_pub = self.create_publisher(Twist, f'{ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, f'{ns}/setpoint_position/local', 10)
        self.pose_sub = self.create_subscription(PoseStamped, f'{ns}/local_position/pose', self._pose_cb, rclpy.qos.qos_profile_sensor_data)
        self.status_sub = self.create_subscription(State, f'{ns}/state', self._status_cb, 10)
        self.livox_sub = subscribe_lidar(self, cfg["lidar"], self._livox_cb)
        self.setmode_client = self.create_client(SetMode, f'{ns}/set_mode')
        self.arming_client = self.create_client(CommandBool, f'{ns}/cmd/arming')

        self._setup_commands(cfg["command"])

    # ---- callbacks ------------------------------------------------------- #
    def _pose_cb(self, msg):
        self.current_pose = msg

    def _status_cb(self, msg):
        self.current_state = msg

    def _livox_cb(self, xyz):
        self.current_scan = xyz

    # ---- helpers --------------------------------------------------------- #
    def _wait_for_pose(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.current_pose is not None:
                return
            self.get_logger().info("Waiting for FCU connection...")
            time.sleep(1)

    def _relative_position(self):
        p = self.current_pose.pose.position
        return np.array([p.x, p.y, p.z - self.zero_alt])

    def _set_mode(self, mode):
        while rclpy.ok():
            req = SetMode.Request()
            req.custom_mode = mode
            future = self.setmode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().mode_sent:
                self.get_logger().info(f"{mode} enabled")
                return
            self.get_logger().info(f"Failed to set {mode} mode, retrying...")
            time.sleep(1)

    def _arm(self):
        while rclpy.ok():
            req = CommandBool.Request()
            req.value = True
            future = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().success:
                self.get_logger().info("Vehicle armed")
                return
            self.get_logger().info("Failed to arm vehicle, retrying...")
            time.sleep(1)

    def _stream_setpoint(self, pose, count):
        for _ in range(count):
            self.local_pos_pub.publish(pose)
            rclpy.spin_once(self)
            time.sleep(0.1)

    # ---- gate ------------------------------------------------------------ #
    def set_start_gate_pcd(self):
        gate_cfg = self.cfg["gate"]
        path = resolve_path(gate_cfg["pcd_path"])
        self.get_logger().info(f"Loading start gate point cloud from: {path}")
        model = np.load(path).astype(np.float64)
        if gate_cfg.get("finish_offset_x") is not None:
            finish = model.copy()
            finish[:, 0] += gate_cfg["finish_offset_x"]
            model = np.vstack((model, finish))
        self.gate_pcd = model
        # Where the gate centre is in the model file, and where it must sit in the track frame
        # (the frame waypoints / poses are written in). The two differ by this shift.
        self.gate_model_center = np.array(gate_cfg["model_center"], dtype=float)
        self.gate_center = np.array(gate_cfg["center"], dtype=float)
        self.get_logger().info(f"Start gate point cloud loaded with shape: {self.gate_pcd.shape}")

    def centering_gate(self):
        """Locate the gate in the live scan.

        Returns (gate_center_in_body_frame, 4x4 matrix track frame -> body frame, score).
        Retries with a fresh scan until the alignment score reaches align.min_score.
        """
        if self.gate_pcd is None:
            self.get_logger().error("Start gate point cloud not set.")
            return None
        a = self.cfg["align"]

        shift = np.eye(4)
        shift[:3, 3] = -(self.gate_center - self.gate_model_center)   # track frame -> model frame
        best = (None, -1.0)
        for attempt in range(1, a["max_attempts"] + 1):
            self.current_scan = None
            while self.current_scan is None and rclpy.ok():
                self.get_logger().info("Waiting for current scan...")
                rclpy.spin_once(self)
            t0 = time.time()
            T, score = align_gate(self.current_scan, self.gate_pcd, self.cfg)
            self.get_logger().info(
                f"Gate alignment attempt {attempt}/{a['max_attempts']}: score {score:.2f} "
                f"(need {a['min_score']:.2f}), {self.current_scan.shape[0]} points, {time.time() - t0:.1f} s")
            if score > best[1]:
                best = (T, score)
            if score >= a["min_score"]:
                break
        else:
            self.get_logger().warn(f"Gate alignment below min_score, using best score {best[1]:.2f}")
        T, score = best
        matrix = T @ shift
        center = matrix @ np.append(self.gate_center, 1.0)
        self.get_logger().info(f"Gate centre in drone frame: x {center[0]:.2f}, y {center[1]:.2f}, z {center[2]:.2f}")
        return center[:3], matrix, score

    # ---- flight ---------------------------------------------------------- #
    def takeoff(self, target_altitude=None):
        cfg = self.cfg["controller"]["takeoff"]
        if target_altitude is None:
            target_altitude = self.cfg["flight"]["takeoff_altitude"]

        # The lidar is not needed in flight until the mission ends; stop processing it
        self.destroy_subscription(self.livox_sub)
        self.livox_sub = None

        self._wait_for_pose()
        self.zero_alt = self.current_pose.pose.position.z

        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.current_pose.pose.position.x
        takeoff_pose.pose.position.y = self.current_pose.pose.position.y
        takeoff_pose.pose.position.z = self.zero_alt + target_altitude
        takeoff_pose.pose.orientation = self.current_pose.pose.orientation

        self._stream_setpoint(takeoff_pose, cfg["preflight_setpoints"])
        self._set_mode('OFFBOARD')
        self._arm()

        while rclpy.ok() and abs(self.current_pose.pose.position.z - takeoff_pose.pose.position.z) > cfg["height_tolerance"]:
            self.local_pos_pub.publish(takeoff_pose)
            rclpy.spin_once(self)
            time.sleep(0.1)

        self.get_logger().info(f"Takeoff to {target_altitude} meters complete.")

    def land(self):
        self._set_mode('AUTO.LAND')

    def move_to(self, point, tolerance=0.1, hold_duration=0.0, method="position", max_speed=None):
        self.get_logger().info(f"Moving to point: {point}")
        point = np.asarray(point, dtype=float)
        if method == "position":
            target_pose = PoseStamped()
            target_pose.pose.position.x = float(point[0])
            target_pose.pose.position.y = float(point[1])
            target_pose.pose.position.z = float(self.zero_alt + point[2])
            target_pose.pose.orientation = self.current_pose.pose.orientation
            start_time = None
            while rclpy.ok():
                self.local_pos_pub.publish(target_pose)
                rclpy.spin_once(self)
                if np.linalg.norm(self._relative_position() - point) > tolerance:
                    start_time = None
                elif start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= hold_duration:
                    break
        elif method == "velocity":
            vel_cmd = Twist()
            start_time = None
            while rclpy.ok():
                direction = point - self._relative_position()
                error = np.linalg.norm(direction)
                if error < tolerance:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= hold_duration:
                        break
                    vel = np.zeros(3)
                else:
                    start_time = None
                    vel = direction
                    if max_speed is not None and error > max_speed:
                        vel = direction / error * max_speed
                vel_cmd.linear.x, vel_cmd.linear.y, vel_cmd.linear.z = map(float, vel)
                self.vel_pub.publish(vel_cmd)
                rclpy.spin_once(self)

    def follow_path(self, path):
        cfg = self.cfg["controller"]["follow_path"]
        path = np.asarray(path)
        last_log = 0.0
        for i, point in enumerate(path):
            while rclpy.ok():
                pos = self._relative_position()
                direction = point - pos
                if time.time() - last_log >= 0.5:
                    last_log = time.time()
                    self.get_logger().info(
                        f"FOLLOWING PATH {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} (waypoint {i + 1}/{len(path)})")
                if np.linalg.norm(direction) < cfg["step_tolerance"]:
                    break

                vel_cmd = Twist()
                vel_cmd.linear.y = direction[1]
                # Ease off forward speed while moving laterally fast
                if abs(vel_cmd.linear.y) >= cfg["lateral_slowdown_threshold"]:
                    vel_cmd.linear.x = direction[0] / abs(cfg["lateral_slowdown_gain"] * vel_cmd.linear.y)
                else:
                    vel_cmd.linear.x = direction[0]
                vel_cmd.linear.z = direction[2]

                # Cap the commanded speed
                max_speed = cfg.get("max_speed")
                speed = np.linalg.norm([vel_cmd.linear.x, vel_cmd.linear.y, vel_cmd.linear.z])
                if max_speed and speed > max_speed:
                    k = max_speed / speed
                    vel_cmd.linear.x *= k
                    vel_cmd.linear.y *= k
                    vel_cmd.linear.z *= k

                # Slow down on approach to the final point (second half of the path only)
                dist_to_end = np.linalg.norm(path[-1] - point)
                if dist_to_end < cfg["slow_down_distance"] and i > path.shape[0] / 2:
                    scale = dist_to_end / cfg["slow_down_distance"]
                    vel_cmd.linear.x *= scale
                    vel_cmd.linear.y *= scale
                    vel_cmd.linear.z *= scale
                self.vel_pub.publish(vel_cmd)
                rclpy.spin_once(self)
        self.move_to(point, tolerance=cfg["final_tolerance"])
        self.get_logger().info("PATH FOLLOWING DONE")

    def spin_drone(self, duration, angle_step=10.0, step_duration=0.5):
        self._wait_for_pose()
        self.zero_alt = self.current_pose.pose.position.z
        self._stream_setpoint(self.current_pose, self.cfg["controller"]["takeoff"]["preflight_setpoints"])
        self._set_mode('OFFBOARD')
        self._arm()

        current_yaw = yaw_from_orientation(self.current_pose.pose.orientation)
        start_time = last_step_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < duration:
            if (time.time() - last_step_time) >= step_duration:
                last_step_time = time.time()
                current_yaw += math.radians(angle_step)
            target_pose = PoseStamped()
            target_pose.pose.position = self.current_pose.pose.position
            target_pose.pose.orientation.w = math.cos(current_yaw / 2.0)
            target_pose.pose.orientation.z = math.sin(current_yaw / 2.0)
            self.local_pos_pub.publish(target_pose)
            rclpy.spin_once(self)

    # ---- inter-drone commands (/mission/order, std_msgs/String) ----------- #
    def _setup_commands(self, cmd_cfg):
        self.cmd_cfg = cmd_cfg
        self.cmd_inbox = []   # received orders not consumed yet
        self.cmd_sent = []    # our own publications, so the echo is ignored
        self.cmd_pub = self.create_publisher(String, cmd_cfg["order_topic"], COMMAND_QOS)
        self.cmd_sub = self.create_subscription(String, cmd_cfg["order_topic"], self._command_cb, COMMAND_QOS)

    def _command_cb(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if text in self.cmd_sent:
            self.cmd_sent.remove(text)
            return
        self.cmd_inbox.append(text)

    def pub_command(self, command):
        """Publish an order string, e.g. pub_command('4_verny_turn')."""
        command = command.strip()
        self.cmd_sent.append(command)
        self.cmd_pub.publish(String(data=command))
        self.get_logger().info(f"Published: '{command}' to {self.cmd_cfg['order_topic']}")

    def wait_command(self, command, timeout=None, hold_position=True):
        """Block until exactly `command` arrives (case-sensitive string match).
        An order received earlier still counts. Holds position while waiting.
        Returns True, or False on timeout."""
        timeout = timeout or self.cmd_cfg["timeout"]
        command = command.strip()
        self.get_logger().info(f"Waiting for '{command}' on {self.cmd_cfg['order_topic']} (timeout {timeout:.0f} s)")
        hold = None
        if hold_position and self.current_pose is not None:
            hold = PoseStamped()
            hold.pose = self.current_pose.pose
        start = time.time()
        while rclpy.ok():
            if command in self.cmd_inbox:
                self.cmd_inbox.remove(command)
                self.get_logger().info(f"Received: '{command}'")
                return True
            if time.time() - start > timeout:
                self.get_logger().warn(f"Timeout waiting for '{command}' after {timeout:.0f} s")
                return False
            if hold is not None:
                self.local_pos_pub.publish(hold)
            rclpy.spin_once(self, timeout_sec=0.05)
        return False
