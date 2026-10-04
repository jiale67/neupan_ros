#!/usr/bin/env python3

"""
neupan_core is the main class for the neupan_ros package. It is used to run the NeuPAN algorithm in the ROS 2 framework, which subscribes to the laser scan and localization information, and publishes the velocity command to the robot.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
from math import sin, cos, atan2, hypot, copysign, isfinite, sqrt

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import LaserScan

from neupan import neupan
from neupan.util import get_transform, WrapToPi


TF_EXCEPTIONS = (LookupException, ConnectivityException, ExtrapolationException)


class neupan_core(Node):

    """
    工作流程:
      1. 启动后待命，不规划、不发速度，周期打印 "waiting /goal_pose ..."
      2. 收到 goal_topic（默认 /goal_pose，rviz2 的 2D Goal Pose）并校验通过后，
         按 当前位姿 -> 目标点 建直线全局路径，开始 NeuPAN 规划
      3. 到达目标后发零速度，回到 1 等待下一个目标点
    运行中再收到有效目标点会直接替换当前目标、重新规划。
    """

    def __init__(self) -> None:

        super().__init__("neupan_node")

        # ros parameters，取值见 config/neupan_node_V550_mec.yaml
        #
        # 所有参数都按 dynamic_typing 声明，取值后再自己转类型，yaml 里写
        # 3 / 3.0 / "3"、true / "true" 都能用。
        self.planner_config_file = self.resolve_pkg_path(
            self.param("config_file", "config/neupan_planner_V550_mec.yaml")
        )
        self.dune_checkpoint = self.resolve_pkg_path(
            self.param("dune_checkpoint", "model/pretrain_V550_mec/model_5000.pth")
        )

        # 坐标系。默认值对应 wheeltec 实车（turn_on_wheeltec_robot.launch.py）
        self.map_frame = str(self.param("map_frame", "odom_combined"))
        self.base_frame = str(self.param("base_frame", "base_footprint"))
        self.lidar_frame = str(self.param("lidar_frame", "laser"))

        # 话题名
        self.scan_topic = str(self.param("scan_topic", "/scan"))
        self.cmd_vel_topic = str(self.param("cmd_vel_topic", "/cmd_vel"))
        self.goal_topic = str(self.param("goal_topic", "/goal_pose"))

        self.marker_size = float(self.param("marker_size", 0.05))
        self.marker_z = float(self.param("marker_z", 0.3))

        self.scan_angle_range = self.parse_float_array(
            self.param("scan_angle_range", [-3.15, 3.15]), 2, "scan_angle_range"
        )
        self.scan_range = self.parse_float_array(
            self.param("scan_range", [0.0, 8.0]), 2, "scan_range"
        )
        self.scan_downsample = max(1, int(self.param("scan_downsample", 1)))
        self.flip_angle = self.parse_bool(self.param("flip_angle", False))

        # 雷达数据超时（秒）。规划中超过这个时间没收到 scan 就发零速度停车，
        # 防止雷达掉线后规划器在没有障碍物信息的情况下继续开。<=0 关闭此检查。
        # 同时也用于目标点校验: 雷达数据不新鲜时拒绝开始。
        self.scan_timeout = float(self.param("scan_timeout", 0.5))

        # 目标点离当前位置的最大距离（m），超过则拒绝，防止 rviz 误点到远处。<=0 不限制。
        self.max_goal_distance = float(self.param("max_goal_distance", 10.0))

        # omni_yaw_mode: **仅 kinematics: 'omni'（2 自由度）** 下机器人朝向怎么
        # 控制。那个模型不管朝向 (robot.py 的 linear_omni_model: A=I，B 第三行
        # [0,0]，theta 不可控)，所以角速度只能在规划器外面补。
        #
        # kinematics: 'omni3' 不用这一套 —— 它的控制量是 u = (vx, vy, w)，
        # 角速度由规划器直接算出，下面所有 omni_yaw_* 参数对它**全部不生效**
        # (见 generate_twist_msg)。V550 现在用的就是 omni3。
        #   none    -> 不发角速度，保持当前朝向蟹步平移
        #   heading -> 车头对准行进方向
        #   spin    -> 以固定角速度自转
        #
        # 这**不是**纯观感的开环补偿: 转向 -> 观测/几何变化 -> 规划出的方向变化
        # -> 又驱动转向，是一个真实的闭环。所以朝向环必须比规划器慢，下面的默认值
        # 都是按"慢于规划器"选的。
        self.omni_yaw_mode = str(self.param("omni_yaw_mode", "none")).strip().lower()
        # heading 模式的朝向环 P 增益 (rad/s per rad)
        self.omni_yaw_gain = float(self.param("omni_yaw_gain", 1.5))
        # 角速度上限 rad/s。只影响 horizon 的前瞻质量，不影响当帧的碰撞判断：
        # 每个周期只执行 opt_vel 的第一列，而 DUNE 每帧都用**真实** theta 重算距离。
        self.omni_yaw_max = float(self.param("omni_yaw_max", 1.0))
        # spin 模式的固定角速度
        self.omni_spin_speed = float(self.param("omni_spin_speed", 0.8))
        # heading 模式下低于这个速度就不追朝向，免得停下来时因为速度矢量方向
        # 噪声乱转
        self.omni_yaw_min_speed = float(self.param("omni_yaw_min_speed", 0.02))
        # 朝向参考的一阶低通**时间常数**，单位秒。按真实 dt 换算
        # alpha = dt/(tau+dt)，与控制循环频率解耦。
        self.omni_yaw_tau = float(self.param("omni_yaw_tau", 0.3))
        # 朝向误差死区 rad。小于它就不发角速度，避免持续微幅修正造成的极限环。
        self.omni_yaw_deadband = float(self.param("omni_yaw_deadband", 0.12))
        # 角加速度上限 rad/s^2。限制角速度本身的跳变。
        self.omni_yaw_accel = float(self.param("omni_yaw_accel", 2.0))
        # 平滑后的朝向参考(世界系, rad)，None 表示还没初始化
        self.omni_yaw_ref = None
        # 上一拍的时间戳，用来算真实 dt
        self.omni_yaw_last_t = None
        # 上一拍实际发出的角速度，用于 slew 限幅
        self.omni_yaw_last_rate = 0.0

        if self.omni_yaw_mode not in ("none", "heading", "spin"):
            raise ValueError(
                "omni_yaw_mode expects none|heading|spin, got: %r" % self.omni_yaw_mode
            )

        # loop: 覆盖 yaml 的 ipath.loop。留空则沿用规划器 yaml。
        # 注意 loop=true 时车到点后会折返回起点无限往复，永远不会"到达"，
        # 也就不会回到等待目标点状态。
        loop_para = self.param("loop", False)

        if not self.planner_config_file:
            raise ValueError(
                "No planner config file provided! Please set the parameter config_file"
            )

        # checkpoint 不存在时 NeuPAN 会 input() 询问是否现场训练。ros2 launch 下
        # 节点没有 stdin，input() 会直接 EOFError，报错信息看不出原因，所以提前查。
        if self.dune_checkpoint is not None and not os.path.isfile(self.dune_checkpoint):
            raise FileNotFoundError(
                "DUNE checkpoint not found: %s" % self.dune_checkpoint
            )

        pan = {'dune_checkpoint': self.dune_checkpoint}
        self.neupan_planner = neupan.init_from_yaml(
            self.planner_config_file, pan=pan
        )

        # loop 只能在 planner 建好后改：init_from_yaml 会把 ipath 整段替换掉，
        # 从外面传 ipath={'loop':...} 会丢掉 yaml 里其余的 ipath 配置。
        if str(loop_para).strip():
            self.neupan_planner.ipath.loop = self.parse_bool(loop_para)

        # data
        self.obstacle_points = None  # (2, n)  n number of points
        self.robot_state = None  # (3, 1) [x, y, theta]
        self.goal = None  # (3, 1) [x, y, theta]
        self.stop = False
        self.arrive = False
        # 是否有正在执行的目标点。False = 待命（等 goal_topic），不规划、不发速度。
        self.active = False
        # 最近一次收到 scan 的时间（节点时钟），None 表示还没收到过
        self.last_scan_time = None

        # publisher
        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.plan_pub = self.create_publisher(Path, "/neupan_plan", 10)
        self.ref_state_pub = self.create_publisher(
            Path, "/neupan_ref_state", 10
        )  # current reference state
        self.ref_path_pub = self.create_publisher(
            Path, "/neupan_initial_path", 10
        )  # initial path

        ## for rviz visualization
        self.point_markers_pub_dune = self.create_publisher(
            MarkerArray, "/dune_point_markers", 10
        )
        self.robot_marker_pub = self.create_publisher(Marker, "/robot_marker", 10)
        self.point_markers_pub_nrmp = self.create_publisher(
            MarkerArray, "/nrmp_point_markers", 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscriber
        # 雷达驱动一般是 reliable 发布，sensor_data (best effort) 订阅与之兼容，
        # 反之 reliable 订阅 best effort 发布则收不到。
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)

        self.get_logger().info(
            f"neupan_node ready: map_frame={self.map_frame} base_frame={self.base_frame} "
            f"lidar_frame={self.lidar_frame} scan_topic={self.scan_topic} "
            f"cmd_vel_topic={self.cmd_vel_topic} goal_topic={self.goal_topic} "
            f"kinematics={self.neupan_planner.robot.kinematics} "
            f"loop={self.neupan_planner.ipath.loop}"
        )
        self.log_waiting_goal()

        # 控制循环 50 Hz
        self.timer = self.create_timer(1.0 / 50.0, self.control_loop)

    # ------------------------------------------------------------------
    # 参数工具
    # ------------------------------------------------------------------
    def param(self, name, default):

        """声明并读取一个参数，类型不做约束（见 __init__ 开头的说明）。"""

        self.declare_parameter(
            name, default, ParameterDescriptor(dynamic_typing=True)
        )
        value = self.get_parameter(name).value

        return default if value is None else value

    @staticmethod
    def resolve_pkg_path(value):

        """空 -> None；绝对路径原样返回；相对路径按 neupan_ros 的 share 目录解析。"""

        path = str(value).strip()

        if not path or path.lower() == "none":
            return None

        path = os.path.expanduser(path)

        if os.path.isabs(path):
            return path

        return os.path.join(get_package_share_directory("neupan_ros"), path)

    @staticmethod
    def parse_bool(value, default=False):

        """把参数里的字符串/布尔统一成 bool。留空则取 default。"""

        if isinstance(value, bool):
            return value

        s = str(value).strip().lower()

        if not s:
            return default

        return s in ("1", "true", "yes", "on")

    @staticmethod
    def parse_float_array(value, size, name):

        """[a, b] 列表或 "a b" / "a, b" 字符串 -> float32 数组。"""

        if isinstance(value, str):
            arr = np.fromstring(value.replace(",", " "), dtype=np.float32, sep=" ")
        else:
            arr = np.asarray(list(value), dtype=np.float32)

        if arr.size != size:
            raise ValueError("%s expects %d numbers, got: %r" % (name, size, value))

        return arr

    # ------------------------------------------------------------------
    # 状态切换
    # ------------------------------------------------------------------
    def log_waiting_goal(self):

        self.get_logger().info(
            f"waiting {self.goal_topic} ... "
            f"(rviz2 2D Goal Pose, Fixed Frame = {self.map_frame})"
        )

    def enter_idle(self, reason):

        """结束当前目标，回到待命。发一次零速度，之后待命期间不再发 cmd_vel。"""

        self.active = False
        self.publish_zero_vel()
        self.get_logger().info(reason)
        self.log_waiting_goal()

    def lookup_planar_pose(self, target_frame, source_frame):

        """查 target_frame 下 source_frame 的平面位姿 (x, y, yaw)，取最新的 TF。"""

        tf = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time()
        )
        t = tf.transform.translation

        return t.x, t.y, self.quat_to_yaw(tf.transform.rotation)

    def publish_zero_vel(self):

        self.vel_pub.publish(Twist())
        # omni 朝向环的状态一起清掉。留着 last_rate 会让恢复的第一帧从一个已经
        # 不存在的角速度起算，last_t 留着则会算出巨大的 dt 使滤波和限幅同时失效。
        self.omni_yaw_last_rate = 0.0
        self.omni_yaw_last_t = None

    def scan_age(self):

        """距上一帧 scan 的秒数，没收到过返回 None。"""

        if self.last_scan_time is None:
            return None

        return (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9

    # ------------------------------------------------------------------
    # 控制循环
    # ------------------------------------------------------------------
    def control_loop(self):

        try:
            x, y, yaw = self.lookup_planar_pose(self.map_frame, self.base_frame)
            self.robot_state = np.array([x, y, yaw]).reshape(3, 1)

        except TF_EXCEPTIONS:
            self.get_logger().info(
                f"waiting for tf for the transform from {self.base_frame} "
                f"to {self.map_frame}",
                throttle_duration_sec=1.0,
            )
            # 实车底盘会保持最后一条速度指令。规划中丢了定位，必须明确发零速度停住。
            if self.active:
                self.publish_zero_vel()
            return

        self.get_logger().info(
            f"robot state received {self.robot_state.flatten().tolist()}", once=True
        )

        # 待命: 不规划、不发 cmd_vel（不占用底盘，方便手柄/键盘遥控），只发车体框
        if not self.active:
            self.get_logger().info(
                f"waiting {self.goal_topic} ...", throttle_duration_sec=5.0
            )
            self.robot_marker_pub.publish(self.generate_robot_marker_msg())
            return

        self.ref_path_pub.publish(
            self.generate_path_msg(self.neupan_planner.initial_path)
        )

        # 雷达超时保护。没有它的话雷达一掉线 obstacle_points 就停在最后一帧，
        # 规划器会当成障碍物没变继续开。
        if self.scan_timeout > 0.0:
            scan_age = self.scan_age()

            if scan_age is None or scan_age > self.scan_timeout:
                self.get_logger().warn(
                    f"no scan on {self.scan_topic} for "
                    f"{'ever' if scan_age is None else '%.2f s' % scan_age} "
                    f"(scan_timeout={self.scan_timeout}), publishing zero velocity",
                    throttle_duration_sec=1.0,
                )
                self.publish_zero_vel()
                return

        if self.obstacle_points is None:
            self.get_logger().warn(
                "No obstacle points, only path tracking task will be performed",
                throttle_duration_sec=1.0,
            )

        action, info = self.neupan_planner(self.robot_state, self.obstacle_points)

        self.stop = info["stop"]
        self.arrive = info["arrive"]

        # publish the path and velocity
        self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
        self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))
        self.vel_pub.publish(self.generate_twist_msg(action))

        dune_markers = self.generate_dune_points_markers_msg()
        if dune_markers is not None:
            self.point_markers_pub_dune.publish(dune_markers)

        nrmp_markers = self.generate_nrmp_points_markers_msg()
        if nrmp_markers is not None:
            self.point_markers_pub_nrmp.publish(nrmp_markers)

        self.robot_marker_pub.publish(self.generate_robot_marker_msg())

        if info["arrive"]:
            self.enter_idle(
                f"arrive at the target {np.round(self.goal.flatten(), 3).tolist()}"
            )
            return

        if info["stop"]:
            self.get_logger().warn(
                "neupan stop with the min distance "
                + str(self.neupan_planner.min_distance.detach().item())
                + " threshold "
                + str(self.neupan_planner.collision_threshold),
                throttle_duration_sec=0.5,
            )

    # scan callback
    def scan_callback(self, scan_msg):

        self.last_scan_time = self.get_clock().now()

        if self.robot_state is None:
            return None

        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        n = len(ranges)

        if n == 0:
            self.obstacle_points = None
            return None

        # 按 LaserScan 定义 angle_i = angle_min + i * angle_increment。
        # 不用 linspace(angle_min, angle_max, n)：lslidar 等驱动的 angle_max 是
        # -pi..pi 的端点而不是最后一束的角度，linspace 会让整圈累积偏差一个增量。
        if scan_msg.angle_increment != 0.0:
            angles = scan_msg.angle_min + scan_msg.angle_increment * np.arange(n)
        else:
            angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n)

        if self.flip_angle:
            angles = np.flip(angles)

        # 无效束（inf / nan，驱动的裁剪角度和超量程都填 inf）直接丢掉；
        # 量程再与驱动声明的 range_min/range_max 取交集。
        range_min = max(float(self.scan_range[0]), float(scan_msg.range_min))
        range_max = min(float(self.scan_range[1]), float(scan_msg.range_max))

        mask = (
            (np.arange(n) % self.scan_downsample == 0)
            & np.isfinite(ranges)
            & (ranges >= range_min)
            & (ranges <= range_max)
            & (angles > self.scan_angle_range[0])
            & (angles < self.scan_angle_range[1])
        )

        if not np.any(mask):
            self.obstacle_points = None
            self.get_logger().info("No valid scan points", once=True)
            return None

        point_array = np.vstack(
            (ranges[mask] * np.cos(angles[mask]), ranges[mask] * np.sin(angles[mask]))
        )

        try:
            x, y, yaw = self.lookup_planar_pose(self.map_frame, self.lidar_frame)

            trans_matrix, rot_matrix = get_transform(np.c_[x, y, yaw].reshape(3, 1))
            self.obstacle_points = rot_matrix @ point_array + trans_matrix
            self.get_logger().info("Scan obstacle points Received", once=True)

            return self.obstacle_points

        except TF_EXCEPTIONS:
            self.get_logger().info(
                f"waiting for tf for the transform from {self.lidar_frame} "
                f"to {self.map_frame}",
                throttle_duration_sec=1.0,
            )
            return

    # ------------------------------------------------------------------
    # 目标点
    # ------------------------------------------------------------------
    def goal_callback(self, msg):

        '''
        goal_topic（默认 /goal_pose，即 rviz2 的 2D Goal Pose）。
        校验通过后按 当前位姿 -> 目标点 重建直线路径并开始规划；
        规划中收到新的有效目标点则直接替换。校验不通过只打日志，当前状态不变。
        '''

        goal = self.validate_goal(msg)

        if goal is None:
            if not self.active:
                self.log_waiting_goal()
            return

        self.goal = goal

        self.get_logger().info(
            f"goal accepted: start "
            f"{np.round(self.robot_state.flatten(), 3).tolist()} -> goal "
            f"{np.round(self.goal.flatten(), 3).tolist()} in {self.map_frame}, "
            "start neupan planning"
        )

        self.neupan_planner.update_initial_path_from_goal(self.robot_state, self.goal)
        self.neupan_planner.reset()
        self.stop = False
        self.arrive = False
        self.active = True

    def validate_goal(self, msg):

        """
        校验目标点，通过返回 map_frame 下的 (3,1) [x, y, theta]，否则打 warn 返回 None。

        校验项:
          1. 位置/姿态数值有限，四元数非零
          2. 能转换到 map_frame（frame_id 不同时用 TF 转；空 frame_id 视为 map_frame）
          3. 机器人当前位姿已知（map_frame -> base_frame 的 TF 可用）
          4. 雷达数据新鲜（scan_timeout 内收到过）
          5. 距当前位置 > 规划器的 arrive_threshold（否则一开始就判到达）
          6. 距当前位置 <= max_goal_distance（防误点）
        """

        def reject(reason):
            self.get_logger().warn(f"{self.goal_topic} rejected: {reason}")
            return None

        p = msg.pose.position
        q = msg.pose.orientation

        if not all(isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w)):
            return reject("pose contains nan/inf")

        if sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) < 1e-6:
            return reject("orientation quaternion is all zero")

        gx, gy, gtheta = p.x, p.y, self.quat_to_yaw(q)

        frame_id = msg.header.frame_id.strip().lstrip("/")

        if not frame_id:
            self.get_logger().warn(
                f"{self.goal_topic} has empty frame_id, treating it as {self.map_frame}"
            )
        elif frame_id != self.map_frame:
            try:
                tx, ty, tyaw = self.lookup_planar_pose(self.map_frame, frame_id)
            except TF_EXCEPTIONS:
                return reject(
                    f"cannot transform from frame '{frame_id}' to '{self.map_frame}'"
                )

            gx, gy = (
                tx + cos(tyaw) * gx - sin(tyaw) * gy,
                ty + sin(tyaw) * gx + cos(tyaw) * gy,
            )
            gtheta = WrapToPi(gtheta + tyaw)

        try:
            x, y, yaw = self.lookup_planar_pose(self.map_frame, self.base_frame)
            self.robot_state = np.array([x, y, yaw]).reshape(3, 1)
        except TF_EXCEPTIONS:
            return reject(
                f"robot pose unknown (no tf {self.map_frame} -> {self.base_frame})"
            )

        if self.scan_timeout > 0.0:
            scan_age = self.scan_age()
            if scan_age is None or scan_age > self.scan_timeout:
                return reject(
                    f"no fresh scan on {self.scan_topic} "
                    f"({'never received' if scan_age is None else '%.2f s old' % scan_age})"
                )

        dist = hypot(gx - x, gy - y)
        arrive_threshold = float(getattr(self.neupan_planner.ipath, "arrive_threshold", 0.0))

        if dist <= arrive_threshold:
            return reject(
                f"goal is {dist:.3f} m away, within arrive_threshold {arrive_threshold} m"
            )

        if self.max_goal_distance > 0.0 and dist > self.max_goal_distance:
            return reject(
                f"goal is {dist:.3f} m away, beyond max_goal_distance "
                f"{self.max_goal_distance} m"
            )

        return np.array([[gx], [gy], [gtheta]])

    # generate ros message
    def generate_path_msg(self, path_list):

        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for point in path_list:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame

            ps.pose.position.x = float(point[0, 0])
            ps.pose.position.y = float(point[1, 0])
            ps.pose.orientation = self.yaw_to_quat(float(point[2, 0]))

            path.poses.append(ps)

        return path

    def generate_twist_msg(self, vel):

        if vel is None:
            self.omni_yaw_last_rate = 0.0
            self.omni_yaw_last_t = None
            return Twist()

        if self.stop or self.arrive:
            # 这条路径直接发零 Twist，绕过了 omni_yaw_slew，所以要把 slew 的状态
            # 一起清掉，否则恢复行走时车头先按旧值窜一下。
            self.omni_yaw_last_rate = 0.0
            self.omni_yaw_last_t = None
            return Twist()

        action = Twist()

        kinematics = self.neupan_planner.robot.kinematics

        if kinematics in ("omni", "omni3"):
            # omni/omni3 下 neupan.__call__ 返回的是**世界系**笛卡尔速度
            # (vx, vy) / (vx, vy, w)，见 neupan.py 的 cartesian_vel 分支与
            # robot.linear_omni*_model。
            #
            # wheeltec 底盘的 cmd_vel 是**车体系**（linear.x 前、linear.y 左），
            # 所以这里必须先按当前 yaw 转到车体系，否则 yaw != 0 时方向整体偏掉一个 yaw。
            theta = self.robot_state[2, 0]
            vx_w, vy_w = float(vel[0, 0]), float(vel[1, 0])

            action.linear.x = vx_w * cos(theta) + vy_w * sin(theta)
            action.linear.y = -vx_w * sin(theta) + vy_w * cos(theta)

            if kinematics == "omni3":
                # 角速度是**规划器算出来的**: omni3 的 B 第三行是 [0,0,dt]，
                # theta 是可控状态且进代价函数，车头朝向由 MPC 在 horizon 里统一
                # 优化，和平移一起满足避障约束。
                #
                # 这里**不能**再叠加 omni_yaw_rate(): 那个补偿环是给 2 自由度
                # omni 用的外挂 P 控制器，两个环同时作用就是互相打架。
                #
                # 角速度在平面上世界系与车体系同值(都绕 z)，不需要坐标转换。
                action.angular.z = float(vel[2, 0])
                self.omni_yaw_last_rate = 0.0
            else:
                action.angular.z = float(self.omni_yaw_rate(vx_w, vy_w, theta))
        else:
            action.linear.x = float(vel[0, 0])
            action.angular.z = float(vel[1, 0])

        return action

    def omni_yaw_rate(self, vx_w, vy_w, theta):

        """
        omni（2 自由度）下的朝向控制。

        NeuPAN 的 omni 模型不含朝向: linear_omni_model 的 A 是单位阵、B 第三行是
        [0,0]，代价函数里 theta 项也被去掉了。规划器结构上不可能输出角速度，所以
        这里在规划器外面补一个朝向环。三道措施把朝向环压到明显慢于规划器:
          1. 参考低通用真实时间常数 omni_yaw_tau
          2. 死区 omni_yaw_deadband，消掉小幅持续修正造成的极限环
          3. slew 限幅 omni_yaw_accel，角速度本身不许跳变

        返回车体系 z 角速度。
        """

        if self.omni_yaw_mode == "none":
            return 0.0

        if self.omni_yaw_mode == "spin":
            return self.omni_spin_speed

        # 真实 dt，上下都夹住: stop/arrive 那条路径根本不进这个函数，停一会儿再
        # 恢复时 last_t 已经很旧，不夹的话恢复的第一帧会满舵抽一下。
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.omni_yaw_last_t is None or now <= self.omni_yaw_last_t:
            dt = 0.02
        else:
            dt = min(now - self.omni_yaw_last_t, 0.1)
        self.omni_yaw_last_t = now

        # heading: 车头追行进方向
        speed = hypot(vx_w, vy_w)

        if speed < self.omni_yaw_min_speed:
            # 速度太小时 atan2 的结果基本是噪声，别追；参考也不更新。
            # 但角速度要按 slew 收敛到 0，不能直接砍断。
            return self.omni_yaw_slew(0.0, dt)

        target = atan2(vy_w, vx_w)

        # 一阶低通。角度不能直接加权平均(±pi 附近会算出反方向的均值)，
        # 所以对**差值**做插值再叠回去。
        if self.omni_yaw_ref is None:
            self.omni_yaw_ref = target
        else:
            alpha = 1.0 if self.omni_yaw_tau <= 0.0 else dt / (self.omni_yaw_tau + dt)
            d = WrapToPi(target - self.omni_yaw_ref)
            self.omni_yaw_ref = WrapToPi(self.omni_yaw_ref + alpha * d)

        yaw_err = WrapToPi(self.omni_yaw_ref - theta)

        if abs(yaw_err) < self.omni_yaw_deadband:
            return self.omni_yaw_slew(0.0, dt)

        # 死区外做平滑过渡: 从死区边界起算，避免在边界上角速度阶跃。
        err = yaw_err - copysign(self.omni_yaw_deadband, yaw_err)
        rate = self.omni_yaw_gain * err
        rate = max(-self.omni_yaw_max, min(self.omni_yaw_max, rate))

        return self.omni_yaw_slew(rate, dt)

    def omni_yaw_slew(self, target_rate, dt):

        """角加速度限幅。让 angular.z 只能以 omni_yaw_accel 的速率变化。"""

        max_step = self.omni_yaw_accel * dt
        delta = target_rate - self.omni_yaw_last_rate

        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        self.omni_yaw_last_rate += delta

        return self.omni_yaw_last_rate

    def generate_points_markers_msg(self, points, color):

        if points is None:
            return None

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for index, point in enumerate(points.T):

            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = stamp

            marker.scale.x = self.marker_size
            marker.scale.y = self.marker_size
            marker.scale.z = self.marker_size
            marker.color.a = 1.0

            marker.color.r = color[0] / 255
            marker.color.g = color[1] / 255
            marker.color.b = color[2] / 255

            marker.id = index
            marker.type = Marker.CUBE
            # 点数变少时旧 id 的 marker 不会被覆盖，给个寿命让它自己消失
            marker.lifetime = Duration(seconds=0.2).to_msg()
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = 0.3
            marker.pose.orientation = Quaternion()

            marker_array.markers.append(marker)

        return marker_array

    def generate_dune_points_markers_msg(self):

        return self.generate_points_markers_msg(
            self.neupan_planner.dune_points, (160, 32, 240)
        )

    def generate_nrmp_points_markers_msg(self):

        return self.generate_points_markers_msg(
            self.neupan_planner.nrmp_points, (255, 128, 0)
        )

    def generate_robot_marker_msg(self):

        marker = Marker()

        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.color.a = 1.0
        marker.color.r = 0 / 255
        marker.color.g = 255 / 255
        marker.color.b = 0 / 255

        marker.id = 0

        if self.neupan_planner.robot.shape == "rectangle":
            length = self.neupan_planner.robot.length
            width = self.neupan_planner.robot.width
            wheelbase = self.neupan_planner.robot.wheelbase

            marker.scale.x = float(length)
            marker.scale.y = float(width)
            marker.scale.z = float(self.marker_z)

            marker.type = Marker.CUBE

            x = self.robot_state[0, 0]
            y = self.robot_state[1, 0]
            theta = self.robot_state[2, 0]

            if self.neupan_planner.robot.kinematics == "acker":
                diff_len = (length - wheelbase) / 2
                marker_x = x + diff_len * cos(theta)
                marker_y = y + diff_len * sin(theta)
            else:
                marker_x = x
                marker_y = y

            marker.pose.position.x = float(marker_x)
            marker.pose.position.y = float(marker_y)
            marker.pose.position.z = 0.0
            marker.pose.orientation = self.yaw_to_quat(float(theta))

        return marker

    @staticmethod
    def yaw_to_quat(yaw):

        quater = Quaternion()

        quater.x = 0.0
        quater.y = 0.0
        quater.z = sin(yaw / 2)
        quater.w = cos(yaw / 2)

        return quater

    @staticmethod
    def quat_to_yaw(quater):

        x = quater.x
        y = quater.y
        z = quater.z
        w = quater.w

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw
