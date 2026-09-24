# neupan_ros

[NeuPAN 规划器](https://github.com/hanruihua/neupan) 的 ROS 封装。本仓库是
在上游基础上改的 fork，主要差异见下面「与上游的差异」一节。

## 环境

- Ubuntu 20.04 / ROS Noetic / Gazebo classic
- Python 3.8（**不是**上游 README 写的 3.10；本工作区用的 NeuPAN 是
  [py38 分支](https://github.com/hanruihua/NeuPAN/tree/py38) 的路线）
- catkin 工作区（本仓库在 `nav_ws/src/` 下）

NeuPAN 本体在 `nav_ws/src/NeuPAN/`，是个独立的 git 仓库，用 `pip install -e` 装。

## 编译

```bash
cd ~/nav_ws && catkin_make && source devel/setup.bash
```


## 两套仿真示例

| 目录 | 机器人 | 运动学 | 说明 |
|---|---|---|---|
| [example/gazebo_V550](example/gazebo_V550) | wheeltec V550_mec | 四麦轮全向 | 本 fork 新增，见该目录 README |
| [example/gazebo_limo](example/gazebo_limo) | AgileX limo | 差速 / 全向可切 | 上游示例，已改为自包含 |

两者都是**自包含**的 —— urdf、mesh、世界文件都在仓库内，不依赖 `limo_ros` 或
`turn_on_wheeltec_robot`。世界文件共用 `example/gazebo_V550/world/`（simple_S1 /
dense_S2 / mix_D1 / mix_D1_new / corridor / maze）。

启动都是两个终端，先环境后规划器：

```bash
# V550（麦轮全向）
roslaunch neupan_ros gazebo_V550_simple_s1.launch
roslaunch neupan_ros neupan_gazebo_V550.launch

# limo（默认差速）
roslaunch neupan_ros gazebo_limo_simple_s1.launch
roslaunch neupan_ros neupan_gazebo_limo.launch
```

终点可以用 `goal:="X Y [THETA]"`，也可以用 rviz 的 2D Nav Goal 点。坐标是
`map_frame`（两个示例都设成 `odom`，与 spawn 位姿重合），不是相对车头的偏移。

## 与上游的差异

**全向底盘控制量改为笛卡尔 `u = (vx, vy)`。** 上游 omni 用极坐标 `(v, φ)`，那个
参数化需要线性化且在大角度下病态，导致约 1 Hz 的方向角抖动。改动涉及 NeuPAN 本体
5 个文件，并新增配置项 `robot.lateral_ratio`。原理、实测数据、回退方法见
[example/gazebo_V550/CARTESIAN_OMNI.md](example/gazebo_V550/CARTESIAN_OMNI.md)。

**新增一批 ROS 参数**：全向底盘的车头朝向补偿环（`~omni_yaw_*`）、启动时直接指定
全局路径（`~goal` / `~waypoints`）、等待启动信号（`~wait_for_start`）。见下面的
参数表。

**新增 `/neupan_start` 话题**，用来放行/暂停。

**删掉的东西**：`rvo_ros` 依赖（上游用它生成动态障碍物，那条路径早已失效；现在动态
世界的障碍物由 `Autonomous_navigation_exploration` 的 waypoint 插件驱动）、
`gazebo_example_setup.sh`、`run_neupan_gazebo_exp.sh`，以及上游 README 里指向已不
存在的 `gazebo_limo_env_complex_20.launch` 的运行步骤。

## Node API

### 发布的话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/neupan_cmd_vel` | `geometry_msgs/Twist` | 速度指令。omni 时 `linear.x/y` 是**车体系**（已由世界系旋转过来），`angular.z` 来自朝向补偿环 |
| `/neupan_plan` | `nav_msgs/Path` | NeuPAN 规划出的局部路径 |
| `/neupan_initial_path` | `nav_msgs/Path` | 全局初始路径（可视化用） |
| `/neupan_ref_state` | `nav_msgs/Path` | 当前参考状态（可视化用） |
| `/dune_point_markers` | `visualization_msgs/MarkerArray` | DUNE 点 |
| `/nrmp_point_markers` | `visualization_msgs/MarkerArray` | NRMP 点 |
| `/robot_marker` | `visualization_msgs/Marker` | 机器人轮廓 |

### 订阅的话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | 激光数据 |
| `/initial_path` | `nav_msgs/Path` | 直接给定初始路径 |
| `/neupan_goal` | `geometry_msgs/PoseStamped` | 给定终点，按 当前位姿 → 终点 生成直线路径。rviz 的 2D Nav Goal 发的就是它 |
| `/neupan_waypoints` | `nav_msgs/Path` | 给定折线路径 |
| `/neupan_start` | `std_msgs/Bool` | 放行 / 暂停。`true` 开始，`false` 回到待命发零速度 |

> 刷新初始路径用上面三个路径话题之一。要**持续**更新需设
> `~refresh_initial_path:=true`，否则只有第一次生效。

### 参数

基础参数：

| 参数 | 类型 / 默认 | 说明 |
|---|---|---|
| `~config_file` | `str` / 无 | NeuPAN 配置 yaml 路径。**必填**，缺了直接报错 |
| `~map_frame` | `str` / `map` | 地图坐标系名。两个示例都用 `odom` |
| `~base_frame` | `str` / `base_link` | 车体坐标系名 |
| `~lidar_frame` | `str` / `laser_link` | 雷达坐标系名 |
| `~dune_checkpoint` | `str` / `None` | DUNE 权重路径。**按车体尺寸训的，不能跨车复用** |
| `~scan_range` | `str` / `0.0, 5.0` | 激光距离范围 |
| `~scan_angle_range` | `str` / `-3.14 3.14` | 激光角度范围 |
| `~scan_downsample` | `int` / `1` | 激光降采样率 |
| `~flip_angle` | `bool` / `False` | 是否翻转扫描角 |
| `~marker_size` | `float` / `0.05` | 点 marker 大小 |
| `~marker_z` | `float` / `1.0` | marker 高度 |
| `~refresh_initial_path` | `bool` / `False` | 是否持续刷新初始路径 |
| `~include_initial_path_direction` | `bool` / `False` | 用给定路径自带的朝向，还是用点的梯度 |

全局路径与启动控制（本 fork 新增）：

| 参数 | 类型 / 默认 | 说明 |
|---|---|---|
| `~goal` | `str` / 空 | `"X Y [THETA]"`，启动时直接建 当前位姿 → 终点 的直线路径 |
| `~waypoints` | `str` / 空 | `"X Y [THETA]; X Y [THETA]; ..."`，折线全局路径 |
| `~waypoints_include_start` | `bool` / `True` | 是否把当前位姿前置为第一个点 |
| `~loop` | `str` / 空 | 覆盖 yaml 的 `ipath.loop`。指定终点时通常要 `false` |
| `~wait_for_start` | `bool` / `False` | `true` 则启动后原地待命，等放行信号。用于等 rviz 加载完 |
| `~start_trigger_mode` | `str` / `goal` | 收到 2D Nav Goal 时怎么解释：`goal` 当终点，`trigger` 只当启动开关（配 `~waypoints` 用） |

优先级 `~waypoints` > `~goal` > yaml 的 `ipath.waypoints`，只有最高的那个生效。
之所以需要这两个参数而不是只靠话题：节点拿到第一帧 TF 就立刻用 yaml 的 waypoints
建路径，话题发得再快也可能已经晚了，车会先朝 yaml 的第一个 waypoint 冲出去。

全向底盘的车头朝向（本 fork 新增，仅 `kinematics: 'omni'` 有效）：

| 参数 | 类型 / 默认 | 说明 |
|---|---|---|
| `~omni_yaw_mode` | `str` / `heading` | `none` 不转（保持 spawn 朝向蟹行）、`heading` 车头追行进方向、`spin` 恒定自转 |
| `~omni_yaw_gain` | `float` / `1.5` | 朝向误差的比例增益 |
| `~omni_yaw_max` | `float` / `1.0` | 角速度上限 rad/s |
| `~omni_yaw_min_speed` | `float` / `0.02` | 低于此速度不追朝向（速度矢量方向此时是噪声） |
| `~omni_yaw_tau` | `float` / `0.3` | 朝向参考的一阶低通时间常数 s |
| `~omni_yaw_deadband` | `float` / `0.12` | 朝向误差死区 rad |
| `~omni_yaw_accel` | `float` / `2.0` | 角速度 slew 限幅 rad/s² |
| `~omni_spin_speed` | `float` / `0.8` | `spin` 模式的自转速度 |

NeuPAN 的 omni 模型**结构上不含朝向**（`B` 第三行是 `[0,0]`，状态代价只算 x/y），
所以规划器不可能输出角速度 —— `omni_yaw_mode: none` 时 `angular.z` 恒为 0 不是 bug。
上面这一套是加在规划器**外面**的补偿环。

> 这组增益是为**极坐标时代**的抖动信号调的。控制量改笛卡尔后方向信号干净了很多，
> 这些值现在偏保守（表现为车头略滞后于行进方向），尚未按新信号重新标定。
