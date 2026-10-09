# neupan_ros

[NeuPAN 规划器](https://github.com/hanruihua/neupan) 的 ROS 2 封装，接 wheeltec
V550_mec（四麦轮全向）实车。本仓库是在上游基础上改的 fork。

## 环境

- Ubuntu 22.04 / ROS 2 Humble（Jetson JetPack 6.2）
- NeuPAN 本体是独立的 pip 包：`pip install -e NeuPAN`
- 底盘、雷达驱动来自 wheeltec 的 ROS 2 工作区（`turn_on_wheeltec_robot` 等），
  `turn_on_wheeltec_robot/config/wheeltec_param.yaml` 里 `car_mode: V550_mec`

## 编译

把本仓库放进与 wheeltec 源码相同的 colcon 工作区的 `src/` 下：

```bash
cd ~/wheeltec_ros2
colcon build --packages-select neupan_ros
source install/setup.bash
```

## 运行

底盘、EKF、雷达由 `turn_on_wheeltec_robot.launch.py` 启动（它已包含
`wheeltec_lidar.launch.py`），本包的 launch 只启动 NeuPAN 节点和 rviz2：

```bash
# 终端 1：底盘 + 雷达
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
# 终端 2：NeuPAN
ros2 launch neupan_ros neupan_V550_mec.launch.py
```

节点参数全部在 [config/neupan_node_V550_mec.yaml](config/neupan_node_V550_mec.yaml)，
改完需要重新 `colcon build`（或用 `--symlink-install` 编译）。也可以临时换一份：
`params_file:=/path/to/xxx.yaml`。

车载电脑没接显示器时加 `use_rviz:=false`，在同一 `ROS_DOMAIN_ID` 的 PC 上开 rviz2：

```bash
rviz2 -d $(ros2 pkg prefix neupan_ros)/share/neupan_ros/rviz/neupan_V550_mec.rviz
```

### 流程

1. 启动后节点待命，每 5 s 打印 `waiting /goal_pose ...`。待命期间**不规划、不发
   `cmd_vel`**（不占用底盘，可以照常用手柄/键盘遥控）。
2. 在 rviz2 用 **2D Goal Pose** 点目标点（Fixed Frame 为 `odom_combined`）。节点校验
   目标点，通过后按 当前位姿 → 目标点 建直线全局路径，开始规划。
3. 到达后发零速度，回到 1，打印 `waiting /goal_pose ...` 等下一个目标点。

规划中再点一个有效目标点会直接替换当前目标。Ctrl+C 退出时节点会发一次零速度。

目标点校验（不通过打 warn，状态不变）：

| 校验 | 说明 |
|---|---|
| 数值有效 | 位置/四元数无 nan/inf，四元数非零 |
| 坐标系 | `frame_id` 为 `map_frame` 直接用；是别的坐标系则用 TF 转到 `map_frame`，转不了拒绝；为空视为 `map_frame` |
| 定位 | `map_frame → base_frame` 的 TF 可用 |
| 雷达 | `scan_timeout` 内收到过 scan |
| 距离下限 | 距当前位置 > 规划器 yaml 的 `ipath.arrive_threshold`（否则一开始就判到达） |
| 距离上限 | 距当前位置 ≤ `max_goal_distance`，防止误点到远处 |

也可以不用 rviz，直接发话题：

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped   "{header: {frame_id: odom_combined}, pose: {position: {x: 1.5, y: 0.0}, orientation: {w: 1.0}}}"
```

`odom_combined` 的原点是 EKF 启动时车的位置，x 轴是当时的车头方向。

### CPU 资源隔离（Jetson）

NeuPAN（torch + cvxpy）默认会吃满所有核，饿死同机的 EKF / 底盘串口 / 雷达驱动，导致定位
崩溃。节点启动时（导入 torch 之前）做了这几件事，参数在
[config/neupan_node_V550_mec.yaml](config/neupan_node_V550_mec.yaml) 开头：

| 参数 | 默认 | 作用 |
|---|---|---|
| `cpu_affinity` | `"4,5"` | 本进程所有线程绑到这些核（Orin Nano 6 核 0~5），定位等进程独占 0~3。`""` 不绑 |
| `num_threads` | `0` | torch / OpenMP / OpenBLAS 线程数，0 = 绑定的核数 |
| `nice` | `10` | 降低 NeuPAN 优先级，CPU 紧张时定位先跑 |
| `control_frequency` | `10.0` | 控制循环频率。MPC 步长 0.3 s；绑 2 核单次规划约 65 ms，20 Hz 跟不上 |

另外固定设置 `OMP_WAIT_POLICY=PASSIVE`：libgomp 工作线程空闲时默认自旋忙等，这是"核
打满"的主要来源之一。

启动日志里会打印实际生效情况，并每 10 s 打印一次规划耗时：

```
CPU: affinity=[4, 5] (pinned 23 threads, now [4, 5]), torch threads=2, blas limit=2 (threadpoolctl), nice=10, ...
planning time over 100 cycles: avg 65.1 ms, max 72.9 ms (control period 100.0 ms)
```

平均耗时超过控制周期时会提示调低 `control_frequency`。运行中核对：

```bash
pid=$(pgrep -f neupan_node | head -1)
taskset -acp $pid                       # 每个线程的亲和性，应全是 4,5
ps -L -o tid,psr,ni,pcpu,comm -p $pid   # psr = 当前所在核，ni = nice
htop                                    # 看 0~3 号核是否还被打满
```

### 雷达

不做车体剔除滤波（仿真里的 `laser_filters` 已删除）：lslidar 驱动按
`wheeltec_param.yaml` 的 `angle_disable_min/max` 把车体后方角度裁掉，裁掉的束是 `inf`，
节点直接丢弃非有限值，并与驱动的 `range_min/range_max` 取交集。

## 与实车的接口

| 项 | 默认值 | 来源 |
|---|---|---|
| 雷达话题 `scan_topic` | `/scan` | lslidar `laserscan_topic` |
| 速度话题 `cmd_vel_topic` | `/cmd_vel` | `wheeltec_robot_node` 订阅，`linear.x/y` + `angular.z`，**车体系** |
| 目标点话题 `goal_topic` | `/goal_pose` | rviz2 2D Goal Pose |
| `map_frame` | `odom_combined` | robot_localization EKF 发 `odom_combined → base_footprint` |
| `base_frame` | `base_footprint` | 车体中心，NeuPAN 车体矩形以它为中心 |
| `lidar_frame` | `laser` | lslidar `frame_id`；`base_footprint → laser` 静态 TF 来自 `robot_model.yaml` |

以上都在 `config/neupan_node_V550_mec.yaml` 里改。

## Node API

### 发布的话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `cmd_vel_topic`（`/cmd_vel`） | `geometry_msgs/Twist` | 速度指令，仅规划中发布。omni/omni3 时 `linear.x/y` 是**车体系**（已由世界系旋转过来） |
| `/neupan_plan` | `nav_msgs/Path` | NeuPAN 规划出的局部路径 |
| `/neupan_initial_path` | `nav_msgs/Path` | 全局初始路径 |
| `/neupan_ref_state` | `nav_msgs/Path` | 当前参考状态 |
| `/dune_point_markers` | `visualization_msgs/MarkerArray` | DUNE 点 |
| `/nrmp_point_markers` | `visualization_msgs/MarkerArray` | NRMP 点 |
| `/robot_marker` | `visualization_msgs/Marker` | 机器人轮廓 |

### 订阅的话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `scan_topic`（`/scan`） | `sensor_msgs/LaserScan` | 激光数据（sensor_data QoS） |
| `goal_topic`（`/goal_pose`） | `geometry_msgs/PoseStamped` | 目标点，唯一的规划触发方式 |

### 参数

见 [config/neupan_node_V550_mec.yaml](config/neupan_node_V550_mec.yaml)，每项都有注释。

| 参数 | 默认 | 说明 |
|---|---|---|
| `config_file` | `config/neupan_planner_V550_mec.yaml` | NeuPAN 规划器配置。相对路径按包的 share 目录解析 |
| `dune_checkpoint` | `model/pretrain_V550_mec/model_5000.pth` | DUNE 权重，按车体尺寸训练，**不能跨车复用**。文件不存在直接报错 |
| `loop` | `false` | 覆盖规划器的 `ipath.loop`。必须 false，否则到点后折返往复、不会回到待命 |
| `scan_topic` / `cmd_vel_topic` / `goal_topic` | 见上表 | 话题名 |
| `map_frame` / `base_frame` / `lidar_frame` | 见上表 | 坐标系 |
| `scan_range` | `[0.0, 8.0]` | 使用的量程 m（再与驱动 `range_min/max` 取交集） |
| `scan_angle_range` | `[-3.15, 3.15]` | 使用的角度范围 rad，雷达系 |
| `scan_downsample` | `3` | 每隔几束取一束 |
| `flip_angle` | `false` | 扫描角是否反向 |
| `scan_timeout` | `0.5` | 规划中超过这么久没收到 scan 就发零速度；也用于目标点校验。`<=0` 关闭 |
| `max_goal_distance` | `10.0` | 目标点距离上限 m，`<=0` 不限制 |
| `cpu_affinity` / `num_threads` / `nice` / `control_frequency` | `"4,5"` / `0` / `10` / `10.0` | CPU 资源隔离，见上文 |
| `marker_size` / `marker_z` | `0.05` / `0.3` | 可视化 |
| `omni_yaw_mode` 及 `omni_yaw_*` | `none` | 仅 `kinematics: 'omni'`（2 自由度）有效的外挂朝向环；V550 用 `omni3`，角速度由规划器直接给，这组参数不生效 |

## 目录

```
config/neupan_node_V550_mec.yaml      节点参数（话题、坐标系、scan 处理等）
config/neupan_planner_V550_mec.yaml   NeuPAN 规划器配置（kinematics: omni3）
model/pretrain_V550_mec/              DUNE 权重
launch/neupan_V550_mec.launch.py      启动 NeuPAN 节点 + rviz2（不含底盘/雷达）
rviz/neupan_V550_mec.rviz             rviz2 配置
neupan_ros/neupan_core.py             节点实现
doc/CARTESIAN_OMNI.md                 全向控制量改笛卡尔 u=(vx,vy[,w]) 的原理与实测
doc/V550_sim_notes.md                 ROS1 仿真阶段的技术记录（归档）
```

## 与上游的差异

- **ROS 2 Humble**，`ament_python` 包，直接接 wheeltec 实车；ROS1 Noetic 的 Gazebo 仿真
  （世界、urdf、mesh、laser_filters、move_base/DWA/TEB 参数）已删除，可在 git 历史中找回。
- **全向底盘控制量改为笛卡尔** `u = (vx, vy, w)`（`kinematics: 'omni3'`），见
  [doc/CARTESIAN_OMNI.md](doc/CARTESIAN_OMNI.md)。
- 规划由 `/goal_pose` 触发：待命 → 收到有效目标点开始规划 → 到达后回到待命。
- 话题名、坐标系全部参数化；新增雷达超时停车与退出时零速度。
