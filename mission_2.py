import argparse
import time
import numpy as np
import rclpy
from utils.utils_ import (DroneNode, gate_to_world, load_config, ramer_douglas_peucker, smooth_path)

def main():
    parser = argparse.ArgumentParser(description="SAFMC mission")
    parser.add_argument("--config", "-c", default=None,
                        help="config file (default: config/config.yaml), e.g. config/config_sim.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    ctl = cfg["controller"]
    rclpy.init()
    node = DroneNode(cfg)
    # --- Localise the gate and precompute everything in the map frame ------
    node.set_start_gate_pcd()
    rclpy.spin_once(node)
    gate_center, gate_mat, gate_score = node.centering_gate()

    while node.current_pose is None:
        node.get_logger().info("Waiting for current pose...")
        rclpy.spin_once(node)
    pose = node.current_pose

    # keep_z: every waypoint / pose flies at the altitude written in the config
    to_world = lambda p: gate_to_world(p, gate_mat, pose, keep_z=True)
    center_area = to_world(cfg["poses"]["center_area"])
    comm_start = to_world(cfg["poses"]["comm_start"])
    land_area = to_world(cfg["poses"]["land_area"])

    gate_world = to_world(cfg["gate"]["center"])
    node.get_logger().info(
        f"GATE CENTER (map) {gate_world[0]:.2f}, {gate_world[1]:.2f}, {gate_world[2]:.2f}  alignment score {gate_score:.2f}")

    spline = smooth_path(cfg["path"]["waypoints"], cfg["path"]["smoothing_factor"], cfg["path"]["num_samples"])
    world_path = np.array([to_world(p) for p in spline])
    # RDP-simplified variant is computed for inspection/visualisation only
    simple_path = ramer_douglas_peucker(world_path)  # noqa: F841

    # --- Mission sequence ---------------------------------------------------
    node.takeoff()
    time.sleep(0.5)
    node.pub_command("READY-2")
    node.wait_command("GO-2", hold_position=True)   # default timeout from config (command.timeout)
    node.move_to(comm_start, tolerance=ctl["comm_start"]["tolerance"])
    time.sleep(1.0)
    node.follow_path(world_path)
    node.move_to(center_area, tolerance=ctl["move_to"]["tolerance"])
    node.move_to(land_area)
    node.land()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
