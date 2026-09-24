#!/usr/bin/env python3

"""
neupan_core is the main class for the neupan_ros package. It is used to run the NeuPAN algorithm in the ROS framework, which subscribes to the laser scan and localization information, and publishes the velocity command to the robot.

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

from neupan import neupan
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool
from math import sin, cos, atan2, hypot, copysign
import numpy as np
from neupan.util import get_transform, WrapToPi
import tf
import sensor_msgs.point_cloud2 as pc2


class neupan_core:
    def __init__(self) -> None:

        rospy.init_node("neupan_node", anonymous=True)

        # ros parameters
        self.planner_config_file = rospy.get_param("~config_file", None)
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.lidar_frame = rospy.get_param("~lidar_frame", "laser_link")
        self.marker_size = float(rospy.get_param("~marker_size", "0.05"))
        self.marker_z = float(rospy.get_param("~marker_z", "1.0"))

        scan_angle_range_para = rospy.get_param("~scan_angle_range", "-3.14 3.14")
        self.scan_angle_range = np.fromstring(
            scan_angle_range_para, dtype=np.float32, sep=" "
        )

        self.scan_downsample = int(rospy.get_param("~scan_downsample", "1"))

        scan_range_para = rospy.get_param("~scan_range", "0.0, 5.0")
        self.scan_range = np.fromstring(scan_range_para, dtype=np.float32, sep=" ")

        self.dune_checkpoint = rospy.get_param("~dune_checkpoint", None)
        self.refresh_initial_path = rospy.get_param("~refresh_initial_path", False)
        self.flip_angle = rospy.get_param("~flip_angle", False)
        self.include_initial_path_direction = rospy.get_param("~include_initial_path_direction", False)

        # ~omni_yaw_mode: omni 下机器人朝向怎么控制。NeuPAN 的 omni 模型不管朝向
        # (robot.py 的 linear_omni_model: A=I，B 第三行 [0,0]，theta 不可控)，
        # 所以角速度只能在规划器外面补。
        #   none    -> 不发角速度，保持 spawn 朝向蟹步平移（原行为）
        #   heading -> 车头对准行进方向，看起来像普通车在开（默认）
        #   spin    -> 以固定角速度自转，麦轮小车原地打转那种效果
        #
        # 注意这**不是**纯观感的开环补偿。limo 的雷达 FOV 只有 ±120°
        # (limo_gazebo.gazebo 的 min_angle/max_angle = ±2.0944)，不是 360°，
        # 所以车一转，盲区跟着转，DUNE 看到的点集就变；加上车体是 0.322x0.22 的
        # 矩形，朝向变了迎风面也变。转向 -> 观测/几何变化 -> 规划出的 phi 变化
        # -> 又驱动转向，这是一个真实的闭环。所以朝向环必须比规划器慢，
        # 否则两者互相追着走就是摇头。下面的默认值都是按"慢于规划器"选的。
        self.omni_yaw_mode = str(rospy.get_param("~omni_yaw_mode", "heading")).strip().lower()
        # heading 模式的朝向环 P 增益 (rad/s per rad)
        self.omni_yaw_gain = float(rospy.get_param("~omni_yaw_gain", "1.5"))
        # 角速度上限 rad/s。默认 1.0 而不是更大，因为规划器的预测 horizon 假设
        # theta 恒定(B 第三行 [0,0])，而朝向环在规划器外面转车，转得越快
        # horizon 里的朝向就越失真。1.0 配 step_time 0.3 x receding 5 = 1.5s
        # horizon，最坏情况 theta 偏 1.5 rad。
        # 注意这只影响 horizon 的前瞻质量，不影响当帧的碰撞判断：每个周期只执行
        # opt_vel 的第一列，而 DUNE 每帧都用**真实** theta 重算距离。
        self.omni_yaw_max = float(rospy.get_param("~omni_yaw_max", "1.0"))
        # spin 模式的固定角速度
        self.omni_spin_speed = float(rospy.get_param("~omni_spin_speed", "0.8"))
        # heading 模式下低于这个速度就不追朝向，免得停下来时因为速度矢量方向
        # 噪声乱转
        self.omni_yaw_min_speed = float(rospy.get_param("~omni_yaw_min_speed", "0.02"))
        # 朝向参考的一阶低通**时间常数**，单位秒。
        #
        # 当初需要它是因为参考本身在抖: 那时 omni 控制量是极坐标 (v, phi)，
        # max_acce 第二项是每步方向角变化上限，3.14 意味着一步内方向可以掉头，
        # 规划器输出的 phi 逐帧跳变(实测相邻采样 head 在 44°~135° 之间摆)。
        # 车头忠实追一个跳变的参考，看起来就是摇头。
        #
        # 控制量改成笛卡尔 (vx, vy) 之后方向信号本身已经很干净(空场景方向角
        # std 0.005°)，这一套滤波/死区/slew 参数是为**旧的**抖动信号调的，
        # 还没有按新信号重新标定 —— 现在偏保守，表现为车头略滞后于行进方向。
        #
        # 这里用时间常数而不是"每拍系数"，是上一版的 bug: 之前 0.25 是每拍乘的
        # 系数，而 run() 跑 50 Hz，等效时间常数只有 0.02*(1-0.25)/0.25 = 0.06 s，
        # 在摇头那个时间尺度上等于没滤波 —— 这就是上次实测改善很小的原因。
        # 现在按真实 dt 换算 alpha = dt/(tau+dt)，改 50 Hz 也不用重调。
        #
        # 0.3 是实测选的。0.6 抖动指标一样好，但车头会明显滞后于行进方向
        # (实测某次 body 速度矢量与车头夹角均值 68.7°，看着像在蟹行)，而且
        # 两次重复之间波动很大(RMS 0.302 / 0.621)。0.3 两次重复稳定在
        # RMS 0.43/0.46、夹角 27.6°/30.7°(与原版 23.8°/26.0° 相当)，
        # 同时车头零反转。
        self.omni_yaw_tau = float(rospy.get_param("~omni_yaw_tau", "0.3"))
        # 朝向误差死区 rad。小于它就不发角速度。
        # 没有死区时车会一直做微幅修正，视觉上就是持续小抖；0.12 rad ≈ 7°，
        # 这个量级的朝向偏差看不出来，但足以吃掉极限环。
        self.omni_yaw_deadband = float(rospy.get_param("~omni_yaw_deadband", "0.12"))
        # 平滑后的朝向参考(世界系, rad)，None 表示还没初始化
        self.omni_yaw_ref = None
        # 上一拍的时间戳，用来算真实 dt
        self.omni_yaw_last_t = None
        # 上一拍实际发出的角速度，用于 slew 限幅
        self.omni_yaw_last_rate = 0.0
        # 角加速度上限 rad/s^2。限制角速度本身的跳变，即使参考突变，车头也是
        # 匀加速转过去而不是瞬间打满。2.0 配 omni_yaw_max 1.0 = 至少 0.5 s 才
        # 能从 0 打到满舵。
        self.omni_yaw_accel = float(rospy.get_param("~omni_yaw_accel", "2.0"))

        if self.omni_yaw_mode not in ("none", "heading", "spin"):
            raise ValueError(
                "~omni_yaw_mode expects none|heading|spin, got: %r" % self.omni_yaw_mode
            )

        # ~goal: "x y theta"，启动时直接用它建路径（机器人当前位姿 -> 终点），
        # 绕开 yaml 里的 ipath.waypoints。留空则沿用 yaml 的 waypoints。
        # 之所以要这个参数而不是只靠 /neupan_goal 话题：run() 在拿到第一帧 TF 后
        # 立刻用 yaml waypoints 建路径，话题发得再快也可能已经晚了，小车会先朝
        # yaml 的第一个 waypoint（默认原点）冲出去。
        goal_para = rospy.get_param("~goal", "")
        self.init_goal = None
        if str(goal_para).strip():
            g = np.fromstring(str(goal_para), dtype=np.float64, sep=" ")
            if g.size == 2:
                g = np.append(g, 0.0)
            if g.size != 3:
                raise ValueError(
                    "~goal expects 'x y' or 'x y theta', got: %r" % goal_para
                )
            self.init_goal = g.reshape(3, 1)

        # ~waypoints: 全局路径折线，"X Y [THETA]; X Y [THETA]; ..."。
        # 优先级 waypoints > goal > yaml 的 ipath.waypoints，三者只有最高的那个生效。
        #
        # 与 ~goal 的区别: goal 只能给一个点，路径必然是 起点->终点 一条直线；
        # waypoints 可以给任意多个中间点，按顺序连成折线，这才是"指定全局路径"。
        #
        # 默认 include_start=true，会把机器人**当前实际位姿**前置为第一个点，
        # 所以这里只写要依次经过的点，不用重复写起点。
        waypoints_para = rospy.get_param("~waypoints", "")
        self.init_waypoints = None
        if str(waypoints_para).strip():
            self.init_waypoints = self.parse_waypoints_param(waypoints_para)

        # 是否把当前位姿前置进 waypoints。
        # false 用于外部全局规划器给的路径已经从机器人当前位置起算的场合，
        # 此时再前置一次会多出一段从当前位姿到路径首点的直线。
        self.waypoints_include_start = self.parse_bool(
            rospy.get_param("~waypoints_include_start", "true"), True
        )

        # ~wait_for_start: true 则启动后先原地待命，收到启动信号才开始规划/发速度。
        #
        # 存在的原因: rviz 加载 0.5~15 s（实测本机 ~14.7 s: 节点 16:34:47 起来，
        # rviz 到 16:35:02 才订阅上 /neupan_plan），而 run() 拿到第一帧 TF 就立刻
        # 开跑，等可视化出来车已经走了一段。这个开关把"什么时候开跑"交给人。
        #
        # 待命期间仍然照常发 /neupan_initial_path 和 /robot_marker，所以 rviz
        # 一出来就能看到完整全局路径，确认无误再点开始。
        self.wait_for_start = self.parse_bool(
            rospy.get_param("~wait_for_start", "false"), False
        )

        # ~start_trigger_mode: 收到 2D Nav Goal 时怎么解释它。
        #   trigger -> 只当启动开关，已建好的路径(~waypoints/~goal)保持不变。
        #              这是配 ~waypoints 用的模式：点哪儿都一样，只是"开始"。
        #   goal    -> 点击位置就是终点，按 当前位姿 -> 点击点 重建直线路径，
        #              同时开始跑。没用 ~waypoints 而想用鼠标指目标时用这个。
        #
        # trigger 模式下 goal_callback 会**整个**跳过重建路径，也就是说运行中再点
        # 也不会改路径。这是故意的：既然全局路径是 ~waypoints 给的折线，一次误点
        # 就把它换成一条直线不划算。要运行中换路径请用 /neupan_waypoints
        # 配 ~refresh_initial_path:=true。
        #
        # 默认取 goal 而不是 trigger，是为了不动改动前的行为: 仓库里其他 example
        # 的 launch 不设这个参数，它们靠点击 2D Nav Goal 指定终点。
        # neupan_gazebo_limo.launch 里显式设成 trigger。
        self.start_trigger_mode = (
            str(rospy.get_param("~start_trigger_mode", "goal")).strip().lower()
        )

        if self.start_trigger_mode not in ("trigger", "goal"):
            raise ValueError(
                "~start_trigger_mode expects trigger|goal, got: %r"
                % self.start_trigger_mode
            )

        # ~loop: 覆盖 yaml 的 ipath.loop。指定终点时通常要 false（到点即停），
        # yaml 默认是 True（到点后重置路径无限往复）。留空则沿用 yaml。
        #
        # 注意 loop=true 时 init_path_with_state / update_initial_path_from_goal 会
        # 再把首点追加到末尾。如果给的 waypoints 本身已经首尾闭合（例如绕一圈回到
        # 起点），loop 应当保持 false，否则末尾多出一个零长段，curve_generator 里
        # direction = diff/length 会除零（只是 RuntimeWarning，点数不变，但没意义）。
        loop_para = rospy.get_param("~loop", "")

        if self.planner_config_file is None:
            raise ValueError(
                "No planner config file provided! Please set the parameter ~config_file"
            )

        pan = {'dune_checkpoint': self.dune_checkpoint}
        self.neupan_planner = neupan.init_from_yaml(
            self.planner_config_file, pan=pan
        )
        # print()

        # loop 只能在 planner 建好后改：init_from_yaml 会把 ipath 整段替换掉，
        # 从外面传 ipath={'loop':...} 会丢掉 yaml 里其余的 ipath 配置。
        if str(loop_para).strip():
            self.neupan_planner.ipath.loop = str(loop_para).strip().lower() in (
                "1", "true", "yes", "on"
            )

        # data
        self.obstacle_points = None  # (2, n)  n number of points
        self.robot_state = None  # (3, 1) [x, y, theta]
        self.goal = None  # (3, 1) [x, y, theta]
        self.stop = False
        # arrive 原来只在 run() 里第一次调用 planner 之后才被赋值，而
        # generate_twist_msg 会读它。待命期间要发零速度就得先有这个属性，
        # 否则 AttributeError。
        self.arrive = False
        # 是否已经放行。wait_for_start=false 时等于一开始就放行，行为与改动前一致。
        self.started = not self.wait_for_start

        # publisher
        self.vel_pub = rospy.Publisher("/neupan_cmd_vel", Twist, queue_size=10)
        self.plan_pub = rospy.Publisher("/neupan_plan", Path, queue_size=10)
        self.ref_state_pub = rospy.Publisher(
            "/neupan_ref_state", Path, queue_size=10
        )  # current reference state
        self.ref_path_pub = rospy.Publisher(
            "/neupan_initial_path", Path, queue_size=10
        )  # initial path

        ## for rviz visualization
        self.point_markers_pub_dune = rospy.Publisher(
            "/dune_point_markers", MarkerArray, queue_size=10
        )
        self.robot_marker_pub = rospy.Publisher("/robot_marker", Marker, queue_size=10)
        self.point_markers_pub_nrmp = rospy.Publisher(
            "/nrmp_point_markers", MarkerArray, queue_size=10
        )

        self.listener = tf.TransformListener()

        # subscriber
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # three types of initial path:
        # 1. from given path
        # 2. from waypoints
        # 3. from goal position
        rospy.Subscriber("/initial_path", Path, self.path_callback)
        rospy.Subscriber("/neupan_waypoints", Path, self.waypoints_callback)
        rospy.Subscriber("/neupan_goal", PoseStamped, self.goal_callback)

        # 启动开关的另一条路: 不想用 rviz 点击时可以直接
        #   rostopic pub -1 /neupan_start std_msgs/Bool "data: true"
        # 发 false 可以重新按下暂停（回到待命，发零速度）。
        rospy.Subscriber("/neupan_start", Bool, self.start_callback)

    @staticmethod
    def parse_bool(value, default=False):

        """把 rosparam 里的字符串/布尔统一成 bool。留空则取 default。"""

        if isinstance(value, bool):
            return value

        s = str(value).strip().lower()

        if not s:
            return default

        return s in ("1", "true", "yes", "on")

    @staticmethod
    def parse_waypoints_param(text):

        """
        解析 ~waypoints 字符串 -> list of (3,1) ndarray [x, y, theta]。

        格式: 点之间用 ';' 或换行分隔，点内用空格/逗号分隔 2 或 3 个数。
            "4 0; 0 2; -1 0; -3 -3"
            "4,0,0 ; 0,2,1.57"
            多行写法（launch 里更好读）:
                4  0
                0  2

        theta 省略时填 nan，由 build_waypoints_path 按"指向下一个点"补齐 —— 不能
        直接填 0，因为 curve_style='line' 下每个 waypoint 的 theta 就是该段的行进
        方向，填 0 会让路径在每个拐点都被拧成朝 +x。
        """

        chunks = [c.strip() for c in str(text).replace("\n", ";").split(";")]
        chunks = [c for c in chunks if c]

        if not chunks:
            raise ValueError("~waypoints is empty after parsing: %r" % text)

        waypoints = []

        for c in chunks:
            v = np.fromstring(c.replace(",", " "), dtype=np.float64, sep=" ")

            if v.size == 2:
                v = np.append(v, np.nan)
            elif v.size != 3:
                raise ValueError(
                    "~waypoints: each point expects 'X Y' or 'X Y THETA', got %r "
                    "in %r" % (c, text)
                )

            waypoints.append(v.reshape(3, 1))

        if len(waypoints) < 1:
            raise ValueError("~waypoints needs at least 1 point, got: %r" % text)

        return waypoints

    def build_waypoints_path(self, waypoints, include_start=True, tag="waypoints"):

        """
        把一串 waypoint 灌进 NeuPAN，生成全局初始路径。这是对外暴露的统一入口:
        ~waypoints 参数、/neupan_waypoints 话题都走这里。

        Args:
            waypoints: list of (3,1) ndarray [x, y, theta]，theta 可为 nan 表示
                       "按指向下一个点自动补"
            include_start: 是否把机器人当前位姿前置为第一个点
            tag: 日志里用来区分来源

        注意 waypoints 必须是 (3,1) 而不是 (4,1)。gctl 的 generate_line 在
        include_gear=True 时会无条件给端点再 vstack 一个 gear，传 4x1 进去会得到
        5x1 的畸形点（实测: 4x1 输入 -> 每个 waypoint 处的点变成
        [x, y, theta, 1, 1]）。下游靠 [0:3] 切片和 [-1,0] 取 gear 侥幸没崩，
        但形状不该是脏的。
        """

        pts = [np.array(p, dtype=np.float64).reshape(3, 1)[0:3] for p in waypoints]

        if include_start:
            if self.robot_state is None:
                rospy.logwarn(
                    "build_waypoints_path(%s): robot state not ready, "
                    "skip prepending current pose", tag
                )
            else:
                start = self.robot_state.reshape(3, 1).copy()
                # 首点已经就是当前位置时不要再前置，否则多一个零长段，
                # curve_generator 里 diff/length 会除零。
                if np.linalg.norm(start[0:2] - pts[0][0:2]) > 1e-3:
                    pts = [start] + pts
                else:
                    pts[0] = start

        if len(pts) < 2:
            rospy.logerr(
                "build_waypoints_path(%s): need >=2 points to build a path, got %d",
                tag, len(pts)
            )
            return None

        # 补 theta: 第 i 点朝向指向第 i+1 点，末点沿用前一点。
        for i in range(len(pts)):
            if np.isnan(pts[i][2, 0]):
                if i + 1 < len(pts):
                    dx = pts[i + 1][0, 0] - pts[i][0, 0]
                    dy = pts[i + 1][1, 0] - pts[i][1, 0]
                    pts[i][2, 0] = atan2(dy, dx)
                else:
                    pts[i][2, 0] = pts[i - 1][2, 0]

        rospy.loginfo(
            "initial path from %s: %d waypoints %s (loop=%s)",
            tag,
            len(pts),
            [np.round(p.flatten(), 3).tolist() for p in pts],
            self.neupan_planner.ipath.loop,
        )

        self.neupan_planner.update_initial_path_from_waypoints(pts)
        self.neupan_planner.reset()

        # 返回补齐 theta 后的点列，调用方要用末点当 goal
        return pts

    def run(self):

        r = rospy.Rate(50)

        while not rospy.is_shutdown():

            try:
                (trans, rot) = self.listener.lookupTransform(
                    self.map_frame, self.base_frame, rospy.Time(0)
                )

                yaw = self.quat_to_yaw_list(rot)
                x, y = trans[0], trans[1]
                self.robot_state = np.array([x, y, yaw]).reshape(3, 1)

            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                rospy.loginfo_throttle(
                    1,
                    "waiting for tf for the transform from {} to {}".format(
                        self.base_frame, self.map_frame
                    ),
                )
                continue

            if self.robot_state is None:
                rospy.logwarn_throttle(1, "waiting for robot state")
                continue

            rospy.loginfo_once(
                "robot state received {}".format(self.robot_state.tolist())
            )

            # 初始路径三条来源，优先级 waypoints > goal > yaml。都只在
            # initial_path 还是 None 时生效（即只建一次），之后要改路径走
            # /initial_path 或 /neupan_waypoints 话题。
            if (
                self.init_waypoints is not None
                and self.neupan_planner.initial_path is None
            ):
                pts = self.build_waypoints_path(
                    self.init_waypoints,
                    include_start=self.waypoints_include_start,
                    tag="~waypoints",
                )
                if pts is not None:
                    # 终点就是折线最后一个点（theta 已补齐）。记下来只为对外一致，
                    # RViz 的点目标会覆盖它。
                    self.goal = pts[-1].copy()

            elif self.init_goal is not None and self.neupan_planner.initial_path is None:
                # 由 ~goal 建路径: 当前位姿 -> 终点。不经过 yaml waypoints，
                # 所以机器人不会先跑去 yaml 里写的那个点。
                self.goal = self.init_goal
                rospy.loginfo(
                    "initial path from ~goal: start {} -> goal {} (loop={})".format(
                        np.round(self.robot_state.flatten(), 3).tolist(),
                        np.round(self.goal.flatten(), 3).tolist(),
                        self.neupan_planner.ipath.loop,
                    )
                )
                self.neupan_planner.update_initial_path_from_goal(
                    self.robot_state, self.goal
                )
                self.neupan_planner.reset()

            elif (
                len(self.neupan_planner.waypoints) >= 1
                and self.neupan_planner.initial_path is None
            ):
                # yaml 的 ipath.waypoints 路线。注意 init_path_with_state 会把当前
                # 位姿前置进 waypoints，所以路径第一段是 当前位姿 -> waypoints[0]，
                # 默认配置下 waypoints[0] 是 (0,0)，机器人会先开向原点。
                self.neupan_planner.set_initial_path_from_state(self.robot_state)
                # print('set initial path', self.neupan_planner.initial_path)

            if self.neupan_planner.initial_path is None:
                rospy.logwarn_throttle(1, "waiting for neupan initial path")
                continue

            rospy.loginfo_once("initial Path Received")
            self.ref_path_pub.publish(
                self.generate_path_msg(self.neupan_planner.initial_path)
            )

            # 待命闸门。位置很讲究: 放在建完路径、发完 /neupan_initial_path 之后，
            # 这样 rviz 慢慢加载出来时全局路径已经在话题上了(latch 靠这里每拍重发)，
            # 人能先看清路径再决定开跑。放在调用 planner 之前，所以待命期间不做
            # 任何优化、不发非零速度。
            if not self.started:
                rospy.loginfo_throttle(
                    5,
                    "waiting for start trigger: publish 2D Nav Goal in rviz "
                    "(/move_base_simple/goal) or 'rostopic pub -1 /neupan_start "
                    "std_msgs/Bool \"data: true\"'",
                )
                # 明确发零速度而不是什么都不发。gazebo 的驱动插件会保持上一条
                # cmd_vel，如果别的节点先发过速度，这里不发就停不住。
                self.vel_pub.publish(Twist())
                # omni 朝向环的状态一起清掉。/neupan_start false 中途暂停时，
                # 留着 last_rate 会让恢复的第一帧从一个已经不存在的角速度起算，
                # last_t 留着则会算出巨大的 dt 使滤波和限幅同时失效。
                self.omni_yaw_last_rate = 0.0
                self.omni_yaw_last_t = None
                # marker 照发，否则 rviz 里待命阶段看不到车体框。
                self.robot_marker_pub.publish(self.generate_robot_marker_msg())
                r.sleep()
                continue

            if self.obstacle_points is None:
                rospy.logwarn_throttle(
                    1, "No obstacle points, only path tracking task will be performed"
                )

            action, info = self.neupan_planner(self.robot_state, self.obstacle_points)

            self.stop = info["stop"]
            self.arrive = info["arrive"]

            if info["arrive"]:
                # print(action)
                rospy.loginfo_throttle(0.1, "arrive at the target")

            # publish the path and velocity
            self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
            self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))
            self.vel_pub.publish(self.generate_twist_msg(action))

            self.point_markers_pub_dune.publish(self.generate_dune_points_markers_msg())
            self.point_markers_pub_nrmp.publish(self.generate_nrmp_points_markers_msg())
            self.robot_marker_pub.publish(self.generate_robot_marker_msg())

            if info["stop"]:
                rospy.logwarn_throttle(
                    0.5,
                    "neupan stop with the min distance "
                    + str(self.neupan_planner.min_distance.detach().item())
                    + " threshold "
                    + str(self.neupan_planner.collision_threshold),
                )

            r.sleep()

    # scan callback
    def scan_callback(self, scan_msg):

        if self.robot_state is None:
            return None

        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        points = []
        # x, y, z, yaw, pitch, roll = self.lidar_offset

        if self.flip_angle:
            angles = np.flip(angles)

        for i in range(len(ranges)):
            distance = ranges[i]
            angle = angles[i]

            if (
                i % self.scan_downsample == 0
                and distance >= self.scan_range[0]
                and distance <= self.scan_range[1]
                and angle > self.scan_angle_range[0]
                and angle < self.scan_angle_range[1]
            ):
                point = np.array([[distance * cos(angle)], [distance * sin(angle)]])
                points.append(point)

        if len(points) == 0:
            self.obstacle_points = None
            rospy.loginfo_once("No valid scan points")
            return None

        point_array = np.hstack(points)

        try:
            (trans, rot) = self.listener.lookupTransform(
                self.map_frame, self.lidar_frame, rospy.Time(0)
            )

            yaw = self.quat_to_yaw_list(rot)
            x, y = trans[0], trans[1]

            trans_matrix, rot_matrix = get_transform(np.c_[x, y, yaw].reshape(3, 1))
            self.obstacle_points = rot_matrix @ point_array + trans_matrix
            rospy.loginfo_once("Scan obstacle points Received")

            return self.obstacle_points

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.loginfo_throttle(
                1,
                "waiting for tf for the transform from {} to {}".format(
                    self.lidar_frame, self.map_frame
                ),
            )
            return

    def path_callback(self, path):

        '''
        直接传入**已稠密化**的全局路径。点间距即 NeuPAN 的 interval
        (set_initial_path 会按实际点距重算 self.interval)，所以点不能太稀，
        否则参考点前瞻会跳。只发拐点请用 /neupan_waypoints。
        '''

        if len(path.poses) < 2:
            rospy.logwarn(
                "/initial_path: need >=2 poses, got %d, ignored", len(path.poses)
            )
            return

        initial_point_list = []

        for i in range(len(path.poses)):
            p = path.poses[i]
            x = p.pose.position.x
            y = p.pose.position.y
            
            if self.include_initial_path_direction:
                theta = self.quat_to_yaw(p.pose.orientation)
            else:
                rospy.loginfo_once("Using the points gradient as the initial path direction")

                if i + 1 < len(path.poses):
                    p2 = path.poses[i + 1]
                    x2 = p2.pose.position.x
                    y2 = p2.pose.position.y
                    theta = atan2(y2 - y, x2 - x)
                else:
                    theta = initial_point_list[-1][2, 0]
            
            points = np.array([x, y, theta, 1]).reshape(4, 1)
            initial_point_list.append(points)

        if self.neupan_planner.initial_path is None or self.refresh_initial_path:
            rospy.loginfo_throttle(0.1, "initial path update from given path")
            self.neupan_planner.set_initial_path(initial_point_list)
            self.neupan_planner.reset()
        else:
            # 这里以前是静默丢弃。启动时用 ~waypoints/~goal 建过路径后
            # initial_path 就不是 None 了，外部再发 /initial_path 完全没反应，
            # 也没有任何提示。
            rospy.logwarn_throttle(
                5,
                "/initial_path ignored: initial path already set. "
                "set ~refresh_initial_path:=true to allow runtime replacement",
            )
    
    def waypoints_callback(self, path):

        '''
        Utilize multiple waypoints (goals) to set the initial path.

        这是运行时"直接传全局路径"的接口之一，只需要发拐点，中间由
        curve_generator 按 ipath.interval 采样。另一个接口是 /initial_path，
        那个要发已经稠密化好的路径。

        每个 pose 只取 x/y，theta 按 ~include_initial_path_direction 决定是用
        pose 里的朝向还是按指向下一点算。
        '''

        if len(path.poses) == 0:
            rospy.logwarn("/neupan_waypoints: empty path, ignored")
            return

        waypoints_list = []

        for i in range(len(path.poses)):
            p = path.poses[i]
            x = p.pose.position.x
            y = p.pose.position.y

            if self.include_initial_path_direction:
                theta = self.quat_to_yaw(p.pose.orientation)
            else:
                rospy.loginfo_once("Using the points gradient as the initial path direction")
                # nan 交给 build_waypoints_path 补: 它在**前置当前位姿之后**再算
                # 方向，所以首点朝向也是对的。原来在这里算会漏掉首点(它的方向
                # 得由 robot_state 指向它自己，算不出来)。
                theta = np.nan

            waypoints_list.append(np.array([x, y, theta]).reshape(3, 1))

        if self.neupan_planner.initial_path is None or self.refresh_initial_path:
            self.build_waypoints_path(
                waypoints_list,
                include_start=self.waypoints_include_start,
                tag="/neupan_waypoints",
            )
        else:
            rospy.logwarn_throttle(
                5,
                "/neupan_waypoints ignored: initial path already set. "
                "set ~refresh_initial_path:=true to allow runtime replacement",
            )

    def goal_callback(self, goal):

        '''
        /neupan_goal (remap 到 rviz 的 /move_base_simple/goal，即 2D Nav Goal)。

        两种语义，由 ~start_trigger_mode 决定:
          goal    -> 点击点当终点，重建 当前位姿 -> 点击点 的直线路径（原行为）
          trigger -> 只放行，不动路径。配 ~waypoints 折线用。
        '''

        if self.start_trigger_mode == "trigger":
            x = goal.pose.position.x
            y = goal.pose.position.y

            if not self.started:
                self.started = True
                rospy.loginfo(
                    "start triggered by 2D Nav Goal at (%.3f, %.3f); "
                    "keeping existing global path (~start_trigger_mode=trigger)",
                    x, y,
                )
            else:
                # 已经在跑了还点，说明可能是想改终点。不静默无视，告诉他该怎么做。
                rospy.logwarn_throttle(
                    2,
                    "2D Nav Goal ignored: already started and "
                    "~start_trigger_mode=trigger keeps the global path. "
                    "use ~start_trigger_mode:=goal to click a new goal, or publish "
                    "/neupan_waypoints with ~refresh_initial_path:=true",
                )
            return

        # goal 模式: 点击点即终点
        if self.robot_state is None:
            # 原来这里会把 None 传给 update_initial_path_from_goal。
            rospy.logwarn("2D Nav Goal ignored: robot state not ready yet")
            return

        x = goal.pose.position.x
        y = goal.pose.position.y
        theta = self.quat_to_yaw(goal.pose.orientation)

        self.goal = np.array([[x], [y], [theta]])

        rospy.loginfo("set neupan goal: %s", [x, y, theta])

        rospy.loginfo_throttle(0.1, "initial path update from goal position")
        self.neupan_planner.update_initial_path_from_goal(self.robot_state, self.goal)
        self.neupan_planner.reset()

        # 点了终点就是要它开跑，顺手放行。否则 wait_for_start=true 配 goal 模式时
        # 点完还得再发一次 /neupan_start，没道理。
        if not self.started:
            self.started = True
            rospy.loginfo("start triggered by 2D Nav Goal (~start_trigger_mode=goal)")

    def start_callback(self, msg):

        '''
        /neupan_start (std_msgs/Bool)。true 放行，false 回到待命。

        不走 rviz 时的启动方式:
            rostopic pub -1 /neupan_start std_msgs/Bool "data: true"

        待命时 run() 每拍发零速度，所以发 false 是能真停住的，不是只停规划。
        '''

        if msg.data:
            if not self.started:
                self.started = True
                rospy.loginfo("start triggered by /neupan_start")
        else:
            if self.started:
                self.started = False
                rospy.logwarn("paused by /neupan_start: publishing zero velocity")


    def quat_to_yaw_list(self, quater):

        x = quater[0]
        y = quater[1]
        z = quater[2]
        w = quater[3]

        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return yaw

    # generate ros message
    def generate_path_msg(self, path_list):

        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = rospy.Time.now()
        path.header.seq = 0

        for index, point in enumerate(path_list):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.header.seq = index

            ps.pose.position.x = point[0, 0]
            ps.pose.position.y = point[1, 0]
            ps.pose.orientation = self.yaw_to_quat(point[2, 0])

            path.poses.append(ps)

        return path

    def generate_twist_msg(self, vel):

        if vel is None:
            self.omni_yaw_last_rate = 0.0
            self.omni_yaw_last_t = None
            return Twist()

        if self.stop or self.arrive:
            # print('stop flag true')
            # 这条路径直接发零 Twist，绕过了 omni_yaw_slew，所以要把 slew 的状态
            # 一起清掉。否则恢复行走时会从一个已经不存在的角速度开始限幅，
            # 车头先按旧值窜一下。
            # last_t 也要清: 停了多久不知道，留着会让恢复的第一帧算出巨大的 dt。
            self.omni_yaw_last_rate = 0.0
            self.omni_yaw_last_t = None
            return Twist()

        action = Twist()

        if self.neupan_planner.robot.kinematics == "omni":
            # omni 下 neupan.__call__ 返回的已经是 (vx, vy)，而且是**世界系**的：
            # 它由 opt_vel 的 (v, phi) 算出，phi 是世界系行进方向角
            # (neupan.py 的 omni 分支 / initial_path.omni_model 都按世界系积分)。
            #
            # gazebo_ros_planar_move 把 linear.x/y 当**车体系**用，UpdateChild 里
            # 按当前 yaw 转到世界系再 SetLinearVel。所以这里必须先转到车体系，
            # 否则 yaw != 0 时方向整体偏掉一个 yaw。
            theta = self.robot_state[2, 0]
            vx_w, vy_w = vel[0, 0], vel[1, 0]

            action.linear.x = vx_w * cos(theta) + vy_w * sin(theta)
            action.linear.y = -vx_w * sin(theta) + vy_w * cos(theta)
            action.angular.z = self.omni_yaw_rate(vx_w, vy_w, theta)
        else:
            action.linear.x = vel[0, 0]
            action.angular.z = vel[1, 0]

        return action

    def omni_yaw_rate(self, vx_w, vy_w, theta):

        """
        omni 下的朝向控制。

        NeuPAN 的 omni 模型不含朝向: linear_omni_model 的 A 是单位阵、B 第三行是
        [0,0]，代价函数里 theta 项也被去掉了(robot.py 的 C0_cost omni 分支只算
        diff_s[0:2])。也就是说 theta 不在优化变量的可达空间内，规划器结构上不可能
        输出角速度 —— 这就是原来 angular 恒为 0 的原因，不是漏发。所以这里在
        规划器外面补一个朝向环。

        当帧的避障不受影响: DUNE 每帧拿的是**实际** robot_state 里的 theta
        (scan_to_point 用它把激光点转到车体系)，车转到哪个角度，碰撞距离就按那个
        角度算。

        但这个环**不是开环**，摇头的根因就在这里: limo 的雷达 FOV 只有 ±120°，
        车一转盲区跟着转，DUNE 的输入点集变；车体又是矩形，朝向变则迎风面变。
        于是 转向 -> 观测/几何变 -> 规划的 phi 变 -> 再驱动转向。两个环频率接近
        就会互相追。三道措施把朝向环压到明显慢于规划器:
          1. 参考低通用真实时间常数 omni_yaw_tau (0.6s >> step_time 0.3s)
          2. 死区 omni_yaw_deadband，消掉小幅持续修正造成的极限环
          3. slew 限幅 omni_yaw_accel，角速度本身不许跳变

        返回车体系 z 角速度，直接填 Twist.angular.z (planar_move 按世界 z 施加，
        平面机器人下两者同轴)。
        """

        if self.omni_yaw_mode == "none":
            return 0.0

        if self.omni_yaw_mode == "spin":
            return self.omni_spin_speed

        # 真实 dt。用 sim time(rospy.Time.now() 在 use_sim_time 下即仿真钟)，
        # 这样 gazebo 的实时率不为 1 时滤波的时间常数依然是仿真世界里的 0.6s。
        #
        # dt 必须上下都夹住。往大的方向尤其重要: stop/arrive 那条路径直接发零
        # Twist，根本不进这个函数，所以停一会儿再恢复时 last_t 已经很旧，
        # 算出来的 dt 可能是几十秒 —— 那样 alpha 趋近 1(滤波失效)、
        # max_step 巨大(限幅失效)，恢复的第一帧会满舵抽一下。
        now = rospy.Time.now().to_sec()
        if self.omni_yaw_last_t is None or now <= self.omni_yaw_last_t:
            # 首帧，或时钟回退(重放 /clock、重置仿真)。此时 dt 不可信，
            # 按标称周期走一拍，别让 dt=0 把 alpha 算成 0。
            dt = 0.02
        else:
            dt = min(now - self.omni_yaw_last_t, 0.1)
        self.omni_yaw_last_t = now

        # heading: 车头追行进方向
        speed = hypot(vx_w, vy_w)

        if speed < self.omni_yaw_min_speed:
            # 速度太小时 atan2 的结果基本是噪声，别追，否则停下来会原地抖。
            # 参考也不更新，这样重新起步时是从上次的朝向接着走，不会跳。
            # 但角速度要按 slew 收敛到 0，不能直接砍断。
            return self.omni_yaw_slew(0.0, dt)

        target = atan2(vy_w, vx_w)

        # 一阶低通。角度不能直接加权平均(±pi 附近会算出反方向的均值)，
        # 所以对**差值**做插值再叠回去。
        # alpha 由 dt 和时间常数算，跟 run() 的频率解耦。
        if self.omni_yaw_ref is None:
            self.omni_yaw_ref = target
        else:
            alpha = 1.0 if self.omni_yaw_tau <= 0.0 else dt / (self.omni_yaw_tau + dt)
            d = WrapToPi(target - self.omni_yaw_ref)
            self.omni_yaw_ref = WrapToPi(self.omni_yaw_ref + alpha * d)

        yaw_err = WrapToPi(self.omni_yaw_ref - theta)

        if abs(yaw_err) < self.omni_yaw_deadband:
            # 已经basically对准了。持续发小角速度只会让车头一直微颤，
            # 而 7° 的偏差看不出来。
            return self.omni_yaw_slew(0.0, dt)

        # 死区外做平滑过渡: 从死区边界起算，避免在边界上角速度阶跃。
        err = yaw_err - copysign(self.omni_yaw_deadband, yaw_err)
        rate = self.omni_yaw_gain * err
        rate = max(-self.omni_yaw_max, min(self.omni_yaw_max, rate))

        return self.omni_yaw_slew(rate, dt)

    def omni_yaw_slew(self, target_rate, dt):

        """
        角加速度限幅。让 angular.z 只能以 omni_yaw_accel 的速率变化，
        这样即使朝向参考突变，车头也是匀加速转过去，不会一拍之内从 +1 翻到 -1
        (那正是之前看到的左右晃)。
        """

        max_step = self.omni_yaw_accel * dt
        delta = target_rate - self.omni_yaw_last_rate

        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        self.omni_yaw_last_rate += delta

        return self.omni_yaw_last_rate

    def generate_dune_points_markers_msg(self):

        marker_array = MarkerArray()

        if self.neupan_planner.dune_points is None:
            return
        else:
            points = self.neupan_planner.dune_points

            for index, point in enumerate(points.T):

                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.seq = 0
                marker.header.stamp = rospy.get_rostime()

                marker.scale.x = self.marker_size
                marker.scale.y = self.marker_size
                marker.scale.z = self.marker_size
                marker.color.a = 1.0

                marker.color.r = 160 / 255
                marker.color.g = 32 / 255
                marker.color.b = 240 / 255

                marker.id = index
                marker.type = 1
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.3
                marker.pose.orientation = Quaternion()

                marker_array.markers.append(marker)

            return marker_array

    def generate_nrmp_points_markers_msg(self):

        marker_array = MarkerArray()

        if self.neupan_planner.nrmp_points is None:
            return
        else:
            points = self.neupan_planner.nrmp_points

            for index, point in enumerate(points.T):

                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.seq = 0
                marker.header.stamp = rospy.get_rostime()

                marker.scale.x = self.marker_size
                marker.scale.y = self.marker_size
                marker.scale.z = self.marker_size
                marker.color.a = 1.0

                marker.color.r = 255 / 255
                marker.color.g = 128 / 255
                marker.color.b = 0 / 255

                marker.id = index
                marker.type = 1
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.3
                marker.pose.orientation = Quaternion()

                marker_array.markers.append(marker)

            return marker_array

    def generate_robot_marker_msg(self):

        marker = Marker()

        marker.header.frame_id = self.map_frame
        marker.header.seq = 0
        marker.header.stamp = rospy.get_rostime()

        marker.color.a = 1.0
        marker.color.r = 0 / 255
        marker.color.g = 255 / 255
        marker.color.b = 0 / 255

        marker.id = 0

        if self.neupan_planner.robot.shape == "rectangle":
            length = self.neupan_planner.robot.length
            width = self.neupan_planner.robot.width
            wheelbase = self.neupan_planner.robot.wheelbase

            marker.scale.x = length
            marker.scale.y = width
            marker.scale.z = self.marker_z

            marker.type = 1

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

            marker.pose.position.x = marker_x
            marker.pose.position.y = marker_y
            marker.pose.position.z = 0
            marker.pose.orientation = self.yaw_to_quat(self.robot_state[2, 0])

        return marker

    @staticmethod
    def yaw_to_quat(yaw):

        quater = Quaternion()

        quater.x = 0
        quater.y = 0
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
