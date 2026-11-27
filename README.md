# SAFMC_2026
Safmc 2026 control program

```
mission.py          Stack A: mission sequence, path following + own cylinder avoidance (utils/avoidance.py)
mission_nav2.py     Stack B: same mission, path following through the light Nav2 stack
nav2/               Stack B: bringup.launch.py, nav2_params.yaml, bridge.py (TF/odom + cylinder cloud)
utils/cylinder.py   cylinder detection shared by both stacks (avoid.cyl_min / cyl_max)
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

Run Stack A: `python3 mission.py [--config config/config_sim.yaml]`

Run Stack B (two terminals):
```
ros2 launch nav2/bringup.launch.py config:=config/config.yaml   # controller_server + bridge
python3 mission_nav2.py
```
