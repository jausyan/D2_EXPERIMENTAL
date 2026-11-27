#!/usr/bin/env python3
"""Nav2 bridge: the three small things Nav2 needs from MAVROS + the lidar.

  * TF   odom -> base_link   (position + yaw only, z = 0: Nav2 plans in 2D)
  * /nav2/odom               nav_msgs/Odometry for the controller (velocity in base_link)
  * /nav2/cylinders          PointCloud2 (frame base_link) holding ONLY cylinder-sized blobs,
                             so gate poles, gate bars and walls never enter the costmap

Run via nav2/bringup.launch.py, or directly:  python3 nav2/bridge.py --config config/config.yaml
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cylinder import cylinder_points  # noqa: E402
from utils.utils_ import load_config, orientation_to_quat, subscribe_lidar, yaw_from_orientation  # noqa: E402


class Nav2Bridge(Node):
    def __init__(self, cfg):
        super().__init__("nav2_bridge")
        self.cfg = cfg
        ns = cfg["node"]["mavros_namespace"]
        self.pose = None
        self.vel = (0.0, 0.0, 0.0)
        self.tf_pub = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/nav2/odom", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/nav2/cylinders", 5)
        self.create_subscription(PoseStamped, f"{ns}/local_position/pose", self._pose_cb, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, f"{ns}/local_position/velocity_local", self._vel_cb, qos_profile_sensor_data)
        subscribe_lidar(self, cfg["lidar"], self._lidar_cb)
        self.create_timer(1.0 / 20.0, self._publish_odom)
        self.get_logger().info(f"Nav2 bridge up: {ns}/local_position -> TF/odom, {cfg['lidar']['topic']} -> /nav2/cylinders")

    def _pose_cb(self, msg):
        self.pose = msg

    def _vel_cb(self, msg):
        v = msg.twist.linear
        self.vel = (v.x, v.y, v.z)

    def _publish_odom(self):
        if self.pose is None:
            return
        p = self.pose.pose.position
        yaw = yaw_from_orientation(self.pose.pose.orientation)
        now = self.get_clock().now().to_msg()      # own clock: never depends on FCU/sim time
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = p.x, p.y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = now, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = p.x, p.y
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        c, s = math.cos(yaw), math.sin(yaw)         # world velocity -> base_link
        od.twist.twist.linear.x = c * self.vel[0] + s * self.vel[1]
        od.twist.twist.linear.y = -s * self.vel[0] + c * self.vel[1]
        self.odom_pub.publish(od)

    def _lidar_cb(self, xyz):
        if self.pose is None:
            return
        q = orientation_to_quat(self.pose.pose.orientation)
        pts = cylinder_points(xyz, q, self.cfg["avoid"])      # levelled, world axes, relative to drone
        yaw = yaw_from_orientation(self.pose.pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        if len(pts):                                           # world axes -> base_link (undo yaw)
            pts = np.c_[c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1], pts[:, 2]]
        header = Header(frame_id="base_link")
        header.stamp = self.get_clock().now().to_msg()
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, pts.astype(np.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args, ros_args = parser.parse_known_args()
    cfg = load_config(args.config) if args.config else load_config()
    rclpy.init(args=ros_args)
    node = Nav2Bridge(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
