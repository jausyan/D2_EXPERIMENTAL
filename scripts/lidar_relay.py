"""Make the simulated lidar + IMU readable by FAST-LIO2.

1. Cloud format. The sim publishes x, y, z, intensity, timestamp(float64). FAST-LIO's Velodyne
   reader (lidar_type 2) needs x, y, z, intensity, time (float32, seconds since scan start) and
   ring (uint16); without them it logs "Failed to find match for field 'time'/'ring'" and drops
   every scan.  /livox/lidar -> /livox/lidar_fastlio
2. Clock. The lidar is stamped with Gazebo time (/clock) but MAVROS stamps the IMU with the
   computer's wall clock (1.79e9 s vs ~200 s). FAST-LIO matches the two by stamp, so they never
   overlap and every scan is skipped ("No point, skip this scan!").
   The IMU is restamped with the latest /clock value.  /rian/imu/data -> /rian/imu_fastlio

    python3 scripts/lidar_relay.py
    python3 scripts/lidar_relay.py --imu-in /rian/imu/data_raw --imu-out /rian/imu_fastlio
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

OUT_DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'time', 'ring'],
                      'formats': ['<f4', '<f4', '<f4', '<f4', '<f4', '<u2'],
                      'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})
FIELDS = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
          PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
          PointField(name='time', offset=16, datatype=PointField.FLOAT32, count=1),
          PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1)]


class Relay(Node):
    def __init__(self, topic_in, topic_out, imu_in, imu_out, default_period):
        super().__init__('lidar_relay')
        self.sim_time = None       # latest /clock, seconds
        self.clock_wall = None     # wall time when it arrived
        self.tick = 0.1            # sim seconds between /clock messages (measured)
        self.rtf = 1.0             # sim seconds per wall second (measured)
        self.last_imu_stamp = 0.0
        self.create_subscription(Clock, '/clock', self.clock_cb, qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, imu_out, 100)
        self.create_subscription(Imu, imu_in, self.imu_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, topic_out, 10)   # reliable: FAST-LIO subscribes reliable
        self.sub = self.create_subscription(PointCloud2, topic_in, self.cb, qos_profile_sensor_data)
        self.period = default_period     # s, scan duration used to spread per-point times
        self.last_stamp = None
        self.count = 0

    def clock_cb(self, msg):
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        wall = time.monotonic()
        if self.sim_time is not None and sim > self.sim_time and wall > self.clock_wall:
            self.tick = sim - self.sim_time
            rtf = self.tick / (wall - self.clock_wall)
            self.rtf += 0.2 * (rtf - self.rtf)            # smoothed
        self.sim_time, self.clock_wall = sim, wall

    def imu_cb(self, msg):
        # /clock only ticks every 0.1 s, but the IMU runs several times faster: interpolate
        # between ticks with the wall clock so every IMU sample gets its own, increasing stamp.
        if self.sim_time is None:
            return
        frac = min((time.monotonic() - self.clock_wall) * self.rtf, self.tick * 0.999)
        stamp = max(self.sim_time + frac, self.last_imu_stamp + 1e-4)
        self.last_imu_stamp = stamp
        msg.header.stamp = Time(sec=int(stamp), nanosec=int((stamp % 1.0) * 1e9))
        self.imu_pub.publish(msg)

    def cb(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and 0.02 < stamp - self.last_stamp < 0.5:
            self.period = stamp - self.last_stamp      # follow the real scan rate
        self.last_stamp = stamp

        pts = point_cloud2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True)
        n = len(pts)
        if n == 0:
            return
        out = np.zeros(n, dtype=OUT_DTYPE)
        for k in ('x', 'y', 'z', 'intensity'):
            out[k] = pts[k]
        # points are ordered in time inside a scan: spread them evenly over the scan period
        out['time'] = np.linspace(0.0, self.period * (n - 1) / n, n, dtype=np.float32)
        out['ring'] = 0

        res = PointCloud2()
        res.header = msg.header
        res.height, res.width = 1, n
        res.fields = FIELDS
        res.is_bigendian = False
        res.point_step = OUT_DTYPE.itemsize
        res.row_step = res.point_step * n
        res.is_dense = True
        res.data = out.tobytes()
        self.pub.publish(res)
        self.count += 1
        if self.count % 20 == 1:
            self.get_logger().info(f"relayed {self.count} scans, {n} points, scan period {self.period:.3f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='topic_in', default='/livox/lidar')
    ap.add_argument('--out', dest='topic_out', default='/livox/lidar_fastlio')
    ap.add_argument('--imu-in', default='/rian/imu/data_raw')
    ap.add_argument('--imu-out', default='/rian/imu_fastlio')
    ap.add_argument('--period', type=float, default=0.1, help='scan duration if it cannot be measured (s)')
    args = ap.parse_args()
    rclpy.init()
    rclpy.spin(Relay(args.topic_in, args.topic_out, args.imu_in, args.imu_out, args.period))


if __name__ == '__main__':
    main()
