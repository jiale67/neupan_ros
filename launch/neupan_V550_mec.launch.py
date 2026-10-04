"""
NeuPAN 规划器 + wheeltec V550_mec（麦轮全向）实车。

本 launch 只启动 NeuPAN 节点和 rviz2，底盘和雷达需要先在另一个终端启动:
    ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
    ros2 launch neupan_ros neupan_V550_mec.launch.py

节点参数全部在 config/neupan_node_V550_mec.yaml。

启动后节点待命并打印 "waiting /goal_pose ..."，在 rviz2 用 "2D Goal Pose" 点目标点，
校验通过后开始规划；到达后回到待命，等下一个目标点。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('neupan_ros')

    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'neupan_node_V550_mec.yaml'),
        description='neupan_node 参数文件')

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='是否启动 rviz2（无显示器的车载电脑上设 false，在 PC 上开 rviz2）')

    neupan_node = Node(
        package='neupan_ros',
        executable='neupan_node',
        name='neupan_node',  # 与参数文件里的节点名一致
        output='screen',
        parameters=[params_file],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'neupan_V550_mec.rviz')],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        declare_params_file,
        declare_use_rviz,
        neupan_node,
        rviz,
    ])
