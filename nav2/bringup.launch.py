"""Start the light Nav2 stack: controller_server + lifecycle_manager + our bridge.

  ros2 launch nav2/bringup.launch.py                       # config/config.yaml
  ros2 launch nav2/bringup.launch.py config:=config/config_sim.yaml
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parent.parent


def generate_launch_description():
    params = str(ROOT / "nav2" / "nav2_params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(ROOT / "config" / "config.yaml")),
        ExecuteProcess(cmd=["python3", str(ROOT / "nav2" / "bridge.py"), "--config", LaunchConfiguration("config")],
                       output="screen"),
        Node(package="nav2_controller", executable="controller_server", name="controller_server",
             output="screen", parameters=[params], remappings=[("cmd_vel", "/nav2/cmd_vel")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager",
             output="screen", parameters=[params]),
    ])
