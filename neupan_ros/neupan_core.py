#!/usr/bin/env python3

"""
neupan_core: NeuPAN 的 ROS 2 节点。订阅激光与定位，发布速度指令。

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
import time
from collections import deque
from math import sin, cos, atan2, hypot, isfinite, sqrt

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

# torch 的线程池在首次使用时按当时的亲和性和环境变量创建，
# 所以 neupan/torch 由 import_neupan() 在绑核之后导入。
neupan = None
get_transform = None
WrapToPi = None
torch = None


TF_EXCEPTIONS = (LookupException, ConnectivityException, ExtrapolationException)


def parse_cpu_list(value):

    """"4,5" / "3-5" / "2,4-5" / [4, 5] / 5 -> 排序后的核编号列表；"" -> []。"""

    if isinstance(value, bool):
        raise ValueError("cpu_affinity expects core ids, got bool: %r" % value)

    if isinstance(value, int):
        return [value]

    if not isinstance(value, str):
        return sorted({int(v) for v in value})

    cores = set()

    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))

    return sorted(cores)


def pin_process(cores, nice):

    """
    把本进程所有已有线程绑到 cores 并设 nice（Linux 上两者都按线程生效）。
    之后新建的线程继承创建者的设置。返回 (绑核线程数, nice 失败原因或 None)。
    """

    pinned = 0
    nice_error = None

    for tid in os.listdir("/proc/self/task"):
        tid = int(tid)

        try:
            if cores:
                os.sched_setaffinity(tid, cores)
            pinned += 1

            if nice is not None:
                os.setpriority(os.PRIO_PROCESS, tid, nice)

        except ProcessLookupError:
            continue

        except PermissionError as e:
            # 降低 nice 需要 CAP_SYS_NICE
            nice_error = str(e)

    return pinned, nice_error


def import_neupan(num_threads):

    """
    设置线程数环境变量后导入 torch / neupan。
    OMP_WAIT_POLICY=PASSIVE: OpenMP 空闲线程休眠，不自旋占满 CPU。
    """

    global neupan, get_transform, WrapToPi, torch

    if num_threads > 0:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(num_threads)

    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    import torch as _torch

    torch = _torch

    if num_threads > 0:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    from neupan import neupan as _neupan
    from neupan.util import get_transform as _get_transform, WrapToPi as _WrapToPi

    neupan, get_transform, WrapToPi = _neupan, _get_transform, _WrapToPi

    limiter = None

    if num_threads > 0:
        # numpy 已提前导入，环境变量对其 OpenBLAS 无效，运行时用 threadpoolctl 限制
        try:
            from threadpoolctl import threadpool_limits
            limiter = threadpool_limits(limits=num_threads)
        except ImportError:
            pass

    return torch.get_num_threads(), limiter


class neupan_core(Node):

    """
    待命 -> 收到有效 goal_topic -> 按 当前位姿->目标点 建直线路径并规划 -> 到达后发零速度、回到待命。
    规划中收到新的有效目标点直接替换。
    """

    def __init__(self) -> None:

        super().__init__("neupan_node")

        # 参数说明见 config/neupan_node_V550_mec.yaml

        # 必须在导入 torch 之前
        self.setup_cpu(
            cpu_affinity=self.param("cpu_affinity", ""),
            num_threads=int(self.param("num_threads", 0)),
            nice=int(self.param("nice", 0)),
        )

        self.control_frequency = float(self.param("control_frequency", 10.0))
        if self.control_frequency <= 0.0:
            raise ValueError(
                "control_frequency must be > 0, got: %r" % self.control_frequency
            )

        self.planner_config_file = self.resolve_pkg_path(
            self.param("config_file", "config/neupan_planner_V550_mec.yaml")
        )
        self.dune_checkpoint = self.resolve_pkg_path(
            self.param("dune_checkpoint", "model/pretrain_V550_mec/model_5000.pth")
        )

        self.map_frame = str(self.param("map_frame", "odom_combined"))
        self.base_frame = str(self.param("base_frame", "base_footprint"))
        self.lidar_frame = str(self.param("lidar_frame", "laser"))

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
        self.scan_timeout = float(self.param("scan_timeout", 0.5))
        self.max_goal_distance = float(self.param("max_goal_distance", 10.0))

        # 底盘执行延迟补偿
        self.cmd_delay_compensation = self.parse_bool(
            self.param("cmd_delay_compensation", False)
        )
        self.cmd_delay = float(self.param("cmd_delay", 1.0))
        if self.cmd_delay_compensation and self.cmd_delay <= 0.0:
            raise ValueError("cmd_delay must be > 0, got: %r" % self.cmd_delay)
        # 已发指令 (monotonic 时刻, vx, vy, w)，车体系
        self.cmd_history = deque()
        # 交给规划器的状态（补偿后），robot_state 保持测量值
        self.plan_state = None

        loop_para = self.param("loop", False)

        if not self.planner_config_file:
            raise ValueError(
                "No planner config file provided! Please set the parameter config_file"
            )

        # checkpoint 缺失时 NeuPAN 会调用 input()，launch 下没有 stdin，提前报错
        if self.dune_checkpoint is not None and not os.path.isfile(self.dune_checkpoint):
            raise FileNotFoundError(
                "DUNE checkpoint not found: %s" % self.dune_checkpoint
            )

        pan = {'dune_checkpoint': self.dune_checkpoint}
        self.neupan_planner = neupan.init_from_yaml(
            self.planner_config_file, pan=pan
        )

        # 只能在建好后改: 通过 init_from_yaml 传 ipath 会整段替换 yaml 的 ipath
        if str(loop_para).strip():
            self.neupan_planner.ipath.loop = self.parse_bool(loop_para)

        self.disable_planner_grad()

        self.obstacle_points = None  # (2, n)
        self.robot_state = None  # (3, 1) [x, y, theta]
        self.goal = None  # (3, 1) [x, y, theta]
        self.stop = False
        self.arrive = False
        self.active = False  # 是否有正在执行的目标点
        self.last_scan_time = None
        self.plan_times = []
        self.plan_stats_t0 = time.monotonic()

        # publisher
        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.plan_pub = self.create_publisher(Path, "/neupan_plan", 10)
        self.ref_state_pub = self.create_publisher(Path, "/neupan_ref_state", 10)
        self.ref_path_pub = self.create_publisher(Path, "/neupan_initial_path", 10)

        self.point_markers_pub_dune = self.create_publisher(
            MarkerArray, "/dune_point_markers", 10
        )
        self.robot_marker_pub = self.create_publisher(Marker, "/robot_marker", 10)
        self.point_markers_pub_nrmp = self.create_publisher(
            MarkerArray, "/nrmp_point_markers", 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscriber。best effort 订阅可兼容 reliable 发布
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)

        self.get_logger().info(
            f"neupan_node ready: map_frame={self.map_frame} base_frame={self.base_frame} "
            f"lidar_frame={self.lidar_frame} scan_topic={self.scan_topic} "
            f"cmd_vel_topic={self.cmd_vel_topic} goal_topic={self.goal_topic} "
            f"kinematics={self.neupan_planner.robot.kinematics} "
            f"loop={self.neupan_planner.ipath.loop} "
            f"cmd_delay_compensation={self.cmd_delay_compensation}"
            + (f" (cmd_delay={self.cmd_delay} s)" if self.cmd_delay_compensation else "")
        )
        self.log_waiting_goal()

        self.timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)

    # ------------------------------------------------------------------
    # CPU 资源隔离
    # ------------------------------------------------------------------
    def setup_cpu(self, cpu_affinity, num_threads, nice):

        """绑核、设 nice 和线程数，然后导入 neupan。"""

        log = self.get_logger()
        cores = parse_cpu_list(cpu_affinity)

        if cores:
            if not hasattr(os, "sched_setaffinity"):
                log.warn("cpu_affinity: os.sched_setaffinity unavailable on this OS, skipped")
                cores = []
            else:
                available = sorted(os.sched_getaffinity(0))
                invalid = [c for c in cores if c not in available]

                if invalid:
                    log.warn(
                        f"cpu_affinity: cores {invalid} not available "
                        f"(this machine: {available}), ignored"
                    )
                    cores = [c for c in cores if c in available]

                if not cores:
                    log.error(
                        f"cpu_affinity: no valid core in {cpu_affinity!r}, NOT pinned. "
                        f"available cores: {available}"
                    )
                elif cores == available:
                    log.warn(
                        f"cpu_affinity {cores} covers all cores, "
                        "localization is not isolated from NeuPAN"
                    )

        if num_threads <= 0 and cores:
            num_threads = len(cores)

        nice = nice if nice > 0 else None

        if (cores or nice is not None) and hasattr(os, "sched_setaffinity"):
            pinned, nice_error = pin_process(cores, nice)

            if nice_error:
                log.warn(f"nice {nice} not applied: {nice_error}")
        else:
            pinned = 0

        torch_threads, self.thread_limiter = import_neupan(num_threads)

        log.info(
            f"CPU: affinity={cores if cores else 'all'} "
            f"(pinned {pinned} threads, now {sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'}), "
            f"torch threads={torch_threads}, "
            f"blas limit={'%d (threadpoolctl)' % num_threads if self.thread_limiter is not None else 'env only'}, "
            f"nice={os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, 'getpriority') else '?'}, "
            f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY')}"
        )

    # ------------------------------------------------------------------
    # 参数工具
    # ------------------------------------------------------------------
    def param(self, name, default):

        """声明并读取参数。dynamic_typing: yaml 里 3 / 3.0 / "3" 均可，取值后再转类型。"""

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

        if isinstance(value, bool):
            return value

        s = str(value).strip().lower()

        if not s:
            return default

        return s in ("1", "true", "yes", "on")

    @staticmethod
    def parse_float_array(value, size, name):

        """[a, b] 或 "a b" / "a, b" -> float32 数组。"""

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

        """结束当前目标，发一次零速度后待命。"""

        self.active = False
        self.publish_zero_vel()
        self.get_logger().info(reason)
        self.log_waiting_goal()

    def lookup_planar_pose(self, target_frame, source_frame):

        """target_frame 下 source_frame 的最新平面位姿 (x, y, yaw)。"""

        tf = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time()
        )
        t = tf.transform.translation

        return t.x, t.y, self.quat_to_yaw(tf.transform.rotation)

    def disable_planner_grad(self):

        """
        关闭 NRMP 可调参数的梯度。任一参数 requires_grad 时 cvxpylayers 会额外
        计算导数（约 10 ms/帧），推理用不到；关闭后解不变。
        """

        for t in self.neupan_planner.pan.nrmp_layer.adjust_parameters:
            t.requires_grad_(False)

    # ------------------------------------------------------------------
    # 速度发布与延迟补偿
    # ------------------------------------------------------------------
    def publish_cmd(self, twist):

        """所有速度指令都经此发布，以便记录补偿所需的指令历史。"""

        self.vel_pub.publish(twist)

        if not self.cmd_delay_compensation:
            return

        now = time.monotonic()
        self.cmd_history.append((now, twist.linear.x, twist.linear.y, twist.angular.z))
        # 保留一条窗口外的记录，作为窗口起点处正在执行的指令
        while (len(self.cmd_history) > 1
               and self.cmd_history[1][0] < now - self.cmd_delay):
            self.cmd_history.popleft()

    def predict_state(self, state):

        """
        把最近 cmd_delay 秒内已发出、底盘尚未执行的指令在车体系下积分，
        得到这些指令执行完后的位姿。未开启补偿时原样返回。
        """

        if not self.cmd_delay_compensation or not self.cmd_history:
            return state

        now = time.monotonic()
        t_begin = now - self.cmd_delay
        x, y, th = float(state[0, 0]), float(state[1, 0]), float(state[2, 0])
        hist = list(self.cmd_history)

        for i, (tc, vx, vy, w) in enumerate(hist):
            t0 = max(tc, t_begin)
            t1 = hist[i + 1][0] if i + 1 < len(hist) else now
            dt = t1 - t0

            if dt <= 0.0:
                continue

            n = max(1, int(dt / 0.02))
            h = dt / n
            for _ in range(n):
                x += (vx * cos(th) - vy * sin(th)) * h
                y += (vx * sin(th) + vy * cos(th)) * h
                th += w * h

        return np.array([x, y, th]).reshape(3, 1)

    def publish_zero_vel(self):

        self.publish_cmd(Twist())

    def scan_age(self):

        """距上一帧 scan 的秒数，未收到过返回 None。"""

        if self.last_scan_time is None:
            return None

        return (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9

    def record_plan_time(self, dt):

        """每 10 s 打印一次规划耗时，平均值超过控制周期时 warn。"""

        self.plan_times.append(dt)
        now = time.monotonic()

        if now - self.plan_stats_t0 < 10.0:
            return

        times = np.array(self.plan_times) * 1000.0
        period_ms = 1000.0 / self.control_frequency
        msg = (
            f"planning time over {len(times)} cycles: avg {times.mean():.1f} ms, "
            f"max {times.max():.1f} ms (control period {period_ms:.1f} ms)"
        )

        if times.mean() > period_ms:
            self.get_logger().warn(msg + ", planner cannot keep up, lower control_frequency")
        else:
            self.get_logger().info(msg)

        self.plan_times = []
        self.plan_stats_t0 = now

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
            # 底盘会保持最后一条指令，丢定位时必须停车
            if self.active:
                self.publish_zero_vel()
            return

        self.get_logger().info(
            f"robot state received {self.robot_state.flatten().tolist()}", once=True
        )

        # 待命时不发 cmd_vel，不干扰遥控
        if not self.active:
            self.get_logger().info(
                f"waiting {self.goal_topic} ...", throttle_duration_sec=5.0
            )
            self.robot_marker_pub.publish(self.generate_robot_marker_msg())
            return

        self.ref_path_pub.publish(
            self.generate_path_msg(self.neupan_planner.initial_path)
        )

        # 雷达掉线时 obstacle_points 会停在最后一帧，必须停车
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

        t_plan = time.monotonic()
        self.plan_state = self.predict_state(self.robot_state)
        with torch.no_grad():
            action, info = self.neupan_planner(self.plan_state, self.obstacle_points)
        self.record_plan_time(time.monotonic() - t_plan)

        self.stop = info["stop"]
        self.arrive = info["arrive"]

        self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
        self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))
        self.publish_cmd(self.generate_twist_msg(action))

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

    def scan_callback(self, scan_msg):

        self.last_scan_time = self.get_clock().now()

        if self.robot_state is None:
            return None

        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        n = len(ranges)

        if n == 0:
            self.obstacle_points = None
            return None

        # 按 angle_min + i*angle_increment 计算；lslidar 的 angle_max 不是最后一束的角度
        if scan_msg.angle_increment != 0.0:
            angles = scan_msg.angle_min + scan_msg.angle_increment * np.arange(n)
        else:
            angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n)

        if self.flip_angle:
            angles = np.flip(angles)

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

        """校验通过后按 当前位姿->目标点 重建路径并开始规划；不通过则状态不变。"""

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
        通过返回 map_frame 下的 (3,1) [x, y, theta]，否则 warn 并返回 None。
        校验: 数值有限、可转换到 map_frame、定位可用、scan 新鲜、
        arrive_threshold < 距离 <= max_goal_distance。
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

    # ------------------------------------------------------------------
    # ROS 消息
    # ------------------------------------------------------------------
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

        if vel is None or self.stop or self.arrive:
            return Twist()

        action = Twist()
        kinematics = self.neupan_planner.robot.kinematics

        if kinematics in ("omni", "omni3"):
            # 规划器输出世界系速度，cmd_vel 为车体系；用补偿后的 yaw 旋转
            state = self.plan_state if self.plan_state is not None else self.robot_state
            theta = state[2, 0]
            vx_w, vy_w = float(vel[0, 0]), float(vel[1, 0])

            action.linear.x = vx_w * cos(theta) + vy_w * sin(theta)
            action.linear.y = -vx_w * sin(theta) + vy_w * cos(theta)

            if kinematics == "omni3":
                action.angular.z = float(vel[2, 0])
        else:
            action.linear.x = float(vel[0, 0])
            action.angular.z = float(vel[1, 0])

        return action

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
            # 点数减少时旧 id 不会被覆盖，靠 lifetime 自动消失
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

        return atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))
