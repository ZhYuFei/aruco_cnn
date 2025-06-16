import os
import launch
import yaml
from ament_index_python.packages import get_package_share_directory
import xacro  # 一定要使用 sudo apt install ros-foxy-xacro安装 不要使用pip
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import Shutdown, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    params_file_1 = os.path.join(get_package_share_directory('aruco_launch'), 'config', 'tv.yaml')
    params_file_2 = os.path.join(get_package_share_directory('aruco_launch'), 'config', 'tv.yaml')

    assert os.path.exists(params_file_1), f"Not Found: {params_file_1}"
    assert os.path.exists(params_file_2), f"Not Found: {params_file_2}, 你必须手动创建此文件, 即使文件是空的"

    with open(params_file_1, 'r') as file:
        params_1 = yaml.safe_load(file)
    with open(params_file_2, 'r') as file:
        params_2 = yaml.safe_load(file)

    def get_param(x, **kwargs) -> dict:
        data = {}
        if x in params_1:
            data.update(params_1[x]['ros__parameters'])
        if x in params_2:
            data.update(params_2[x]['ros__parameters'])
        data.update(kwargs)
        return data

    no_get_no_send_arg = DeclareLaunchArgument(
        'no_get_no_send',
        default_value='false',  # 默认为false
        description='If no get serial data, then do not send data'
    )
    no_get_no_send = LaunchConfiguration('no_get_no_send')

    # 预测节点
    detector_node = Node(
        package='aruco_detector',  # Python 包的名称
        executable='aruco_detector',  # 在 setup.py 中定义的入口点名称
        name='aruco_detector',  # 节点名称（可选）
        output='screen',  # 打印输出到终端
    )

   
    robot_xacro_filename = get_param("gimbal_description").get('robot', 'rm_gimbal')
    xacro_file = os.path.join(get_package_share_directory("gimbal_description"),
                              "urdf", f"{robot_xacro_filename}.xacro")
    robot_desc = xacro.process_file(xacro_file).toxml()  # URDF

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_desc}],
        output="screen"
    )

    return LaunchDescription([
        no_get_no_send_arg,
        detector_node,
        robot_state_publisher_node
    ])
