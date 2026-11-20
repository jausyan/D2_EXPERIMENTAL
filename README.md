# SAFMC_2026
Safmc 2026 control program

```
mission.py          mission sequence only
utils/utils_.py     helpers + DroneNode (ROS 2 / MAVROS / GICP / /mission/order)
config/config.yaml  mission parameters, gate model, poses, waypoints
config/mavros_param.yaml  MAVROS parameters
config/path.rviz    RViz layout for scripts/path_capture.py
config/path_draw.rviz  RViz layout for scripts/path_draw.py (draw the path on the arena map)
config/fastlio_mid360_sim.yaml  FAST-LIO2 config for the simulation
data/               gate point clouds
scripts/            mesh_to_pcd.py (arena mesh -> maps/arena_map.pcd), path_draw.py (draw the path in RViz), path_capture.py (record waypoints), capture_gate.py, clean_gate.py, view_gate.py, lidar_relay.py
setup/              SSH helper scripts
```

Run: `python3 mission.py`
