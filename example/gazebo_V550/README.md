# wheeltec V550_mec 接入 NeuPAN 技术文档

**环境**：Ubuntu 20.04 / ROS Noetic / Gazebo classic，catkin 工作区 `nav_ws`
**状态**：静态场景已验证可用；动态场景存在结构性限制，见第 9 节

---

## 文档说明

本文记录把 wheeltec V550_mec（四麦轮全向底盘 + 2D 激光雷达）接入 NeuPAN 规划器的
全过程：接口适配、机体尺寸测定、DUNE 网络重训、仿真环境搭建与实测结果。

第 11 节明确列出哪些结论有实测支撑、哪些没有。看结论前请先读那一节。

---

## 1. 背景与范围

NeuPAN 是端到端的 point-flow 规划器，直接消费激光点云，不需要先建栅格地图。仓库里
原本只有 AgileX limo（差速/阿克曼）的示例。本次工作把 V550_mec 接进来，目的是在
同一套 Gazebo 世界里对比两台车。

两台车的主要差异：

| | limo | V550_mec |
|---|---|---|
| 底盘 | 差速 / 阿克曼 | 四麦轮全向 |
| Gazebo 驱动插件 | skid_steer / planar_move | planar_move |
| 雷达 FOV | ±120° | 360° |
| 雷达点数 / 频率 | 720 / 8 Hz | 360 / 10 Hz |
| 雷达量程 | 0.05~8 m | 0.05~12 m |
| NeuPAN 碰撞矩形 | 0.322 × 0.220 | 0.260 × 0.2446 |

**范围内**：接口适配、尺寸测定、DUNE 重训、仿真启动、静态与动态场景实测。

**范围外**：实车部署；定位方案（当前是纯轮式里程计，无 AMCL / SLAM / EKF，无绝对
定位修正）；动态障碍物预测。

---

## 2. 数据流

```
Gazebo ray 插件 ──/scan_raw──> laser_filters ──/scan──> NeuPAN
  (frame: laser_link)          (剔车体自身点)         neupan_node.py
                                                            │
planar_move 插件 ──/odom──> TF: odom -> base_link ──────────┤
                                                            │
                                            /cmd_vel <───────┘
                                     (linear.x, linear.y; angular.z 恒为 0)
```

关键点：

- **`/scan` 是 laser_link 系**，但 NeuPAN 内部 `scan_callback` 会做
  `lookupTransform(map_frame, lidar_frame)`，所以喂给 DUNE 的点已经是
  **odom（世界）系**，不是机体系。DUNE 再按当前 theta 转回机体系算碰撞。
- **odom 原点 = Gazebo 世界原点**，与 spawn 位置无关。`planar_move` 直接用模型的
  世界位姿发 odom，不是从 spawn 处重新起算。实测 spawn 在 (-3,-3) 时
  `/odom/pose/pose` 读到的就是 (-3.000, -3.000)。**所以 `goal` 参数填的是世界
  坐标，不是"相对车头多远"。**
- `scan_callback` 用 `rospy.Time(0)` 取 TF，拿到的是最新可用变换，不是与扫描
  时间戳对齐的。10 Hz 扫描 / 50 Hz 里程计下最坏约 100 ms 错配，快速运动时点云会
  拖影。**本次未处理。**

---

## 3. 接口适配

### 3.1 点云话题统一为 `/scan`

limo 原本发 `/limo/scan`，NeuPAN launch 里再 remap 回 `/scan`。现在两台车都直接发
`/scan`，remap 全部删除。改动点：limo 的 gazebo 插件 `<topicName>`、两个 NeuPAN
launch 的 remap 行、两个 rviz 配置的 Topic 字段。

`/scan` 这个名字同时是 `costmap_common_params.yaml` 观测源、amcl / gmapping 的
`scan_topic` 默认值、NeuPAN 的默认订阅名，所以全链路不需要任何 remap。

### 3.2 雷达 frame 统一为 `laser_link`

V550 原本叫 `laser`，limo 原本叫 `laser_link`。统一成 `laser_link`，共 13 处：

| 文件 | 处数 | 内容 |
|---|---|---|
| `V550_mec_robot.urdf` | 4 | link 名、`laser_joint` 的 child、`<gazebo reference>`、ray 插件 `<frameName>` |
| `limo_xacro.xacro` | 1 | `limo_laser` 宏注释 |
| `limo_gazebo.gazebo` | 1 | `<frameName>` |
| `limo_four_diff.xacro` / `limo_ackerman.xacro` | 2 | `frame_prefix` 实参 |
| `limo_four_diff.gazebo` / `limo_ackerman.gazebo` | 2 | `frame_prefix` 实参 |
| `neupan_gazebo_limo.launch` | 1 | `lidar_frame` 默认值 |
| `limo_gazebo.rviz` | 1 | RobotModel 的 link 条目 |
| `laser_filter_V550_mec.yaml` | 1 | 注释 |

**注意两个不要改的地方**：`costmap_common_params.yaml` 里的 `laser_scan_sensor`
是观测源名字，`build_map_2d.lua` 里的 `num_laser_scans` 是 cartographer 参数，
都与 frame 无关。

`limo_laser` 宏的 `frame_prefix` 是**完整 link 名**，不再自动拼 `_link`。

V550 的 `laser_joint` 带 `rpy="0 0 -3.14159"`（雷达倒装 180°）。这个由
`lookupTransform` 自动处理，**不要**设 `flip_angle:=true`——那个参数是给
`angle_min`/`angle_max` 反序的雷达用的，设了会把点云镜像。

### 3.3 运动学：全向只控 Vx / Vy

NeuPAN 的 omni 模型**结构上不含朝向**，这不是配置问题：

- `linear_omni_model` 的 A 是单位阵，B 第三行是 `[0, 0]`
- `C0_cost` 的 omni 分支只算 `diff_s[0:2]`

所以规划器不可能输出角速度，`angular.z` 恒为 0 **不是 bug**。`omni_yaw_*` 那一套是
加在规划器**外面**的补偿环，`omni_yaw_mode: none` 时 `omni_yaw_rate()` 直接返回 0.0。

现象：小车保持 spawn 朝向做蟹步平移，车头不转向行进方向。麦轮本身支持这种运动，
避障也不受影响——DUNE 每帧按实际 theta 把激光点转到机体系算碰撞。实测全程 yaw
只有 0.1~1.6° 的接触扰动。

### 3.4 车体自遮挡剔除

雷达装在 base_link 前方 x=0.0835，车体后半边会挡光。实测 base_link 系下的自遮挡：

| 角度 | 距离 |
|---|---|
| ±92~96° | 0.05~0.08 m |
| ±142~147° | 0.11~0.15 m |
| 170~180°（正后） | 0.10~0.22 m |

用 `LaserScanBoxFilter` 而非 `LaserScanRangeFilter`：各方向车体边界到雷达的距离
差很大（正前 0.040 m，正后 0.214 m）。靠 `min_range` 滤净正后方要设 0.23，代价是
车头前方 0.19 m 的真实障碍物全丢。

框按**雷达高度处的车体截面**定，不是整车包围盒：雷达在 z=0.0965，只有这个高度的
车体挡光。实测该高度截面 x[-0.1268, 0.0959] y[±0.0755]，加 0.03 m 余量
（3σ，ray 噪声 stddev=0.01）得框 x[-0.157, 0.126] y[±0.1055]。

余量取 3σ 的理由（单条射线理论泄漏率）：

| 余量 | 泄漏率 |
|---|---|
| 0.005 (0.5σ) | 30.85% |
| 0.010 (1.0σ) | 15.87% |
| 0.020 (2.0σ) | 2.28% |
| 0.030 (3.0σ) | 0.13% ← 取这个 |

第一版只给 5 mm，实测 16 个自遮挡角度漏了 3 个，与 31% 的理论值对得上。

**NeuPAN 必须订 `/scan` 而不是 `/scan_raw`**，否则车尾 0.214 m 内打在自己身上的
点会被当成障碍物，车会以为自己被贴住。

---

## 4. 机体尺寸测定

尺寸是**实测**，不是估算：解析 `base_link.STL` 等网格的二进制顶点，叠加 URDF 的
collision origin 和关节变换，算出各 link 在 base_link 系下的包围盒。

车轮**没有** collision 网格（用圆柱基元），所以按几何参数解析计算：

- 轮 collision：`cylinder radius=0.0372 length=0.048`
- 轮关节：`x = +0.0759 / -0.0817`，`y = ±0.0983`，`z = 0.0167`
- 轮外沿 y = 0.0983 + 0.048/2 = **±0.1223**

（`laser_filter_V550_mec.yaml` 的注释里写轮外沿在 ±0.125，与 costmap footprint 一致，
但那是 footprint 的取值；按 URDF 圆柱实测是 ±0.1223。本文用实测值。）

整车包围盒：**x[-0.1300, 0.1232]，y[±0.1223]**

### 换算成 NeuPAN 的碰撞矩形

NeuPAN 的碰撞矩形**以 base_link 为中心且对称**，而 V550 的 x 向包围盒不对称
（中心偏移 -0.0034）。直接填 `length = 0.1232-(-0.1300) = 0.2532` 会让车尾少覆盖
3.4 mm。所以按 `2 × max(|min|, |max|)` 取：

| | 计算 | 取值 |
|---|---|---|
| length | 2 × max(0.1300, 0.1232) = 2 × 0.1300 | **0.26** |
| width | 2 × 0.1223 | **0.2446** |

代价是车头多 6.8 mm 余量，偏保守，可接受。

**这个矩形比真实车体截面大**（真实截面 0.2532 × 0.1556，宽度方向差 0.089 m），因为
宽度要覆盖车轮。第 8 节的碰撞判定区分了这两个矩形。

---

## 5. DUNE 重训

### 5.1 为什么必须重训（重要陷阱）

`dune.py` 只从 `robot.length` / `robot.width` 推导 G、h，然后
`edge_dim = G.shape[0]`、`state_dim = G.shape[1]`。**任何矩形尺寸都得到同样的矩阵
形状**，所以 limo 的 `pretrain_limo/model_5000.pth` 能被 `load_state_dict`
**静默加载成功**，不报任何错，只是算出来的碰撞距离是错的。

这是最容易踩的坑：没有任何报错提示你用错了权重。

运动学不参与 DUNE 训练——`kinematics: 'omni'` 写在训练 yaml 里只是为了和运行时配置
对齐，对训练结果无影响。

### 5.2 训练配置

`src/NeuPAN/example/dune_train/dune_train_V550_mec.py` 是官方 `dune_train_diff.py`
的逐字复制，**只改了 yaml 文件名**。全部适配都在
`dune_train_V550_mec.yaml`，实质只有两项：

```yaml
robot:
  kinematics: 'omni'    # 装饰性，DUNE 只用 length/width
  length: 0.26          # ← 实测，官方 diff 模板是 1.6
  width: 0.2446         # ← 实测，官方 diff 模板是 2.0
train:
  data_range: [-4, -4, 4, 4]   # ← 官方模板 [-25,-25,25,25]（给 1.6×2.0 大车）
  data_size: 100000
  batch_size: 256
  epoch: 5000
  lr: 5e-5
  lr_decay: 0.5
  decay_freq: 1500
```

`data_range` 必须缩：官方范围是给大车的，不缩的话 10 万样本里绝大多数都"离得很远"，
近距离精度上不去。其余超参照抄官方，**未经调优**。

运行（yaml 路径是相对的，必须在该目录下执行）：

```bash
cd src/NeuPAN/example/dune_train && python3 dune_train_V550_mec.py
```

产物落在 `model/omni_robot_default/`（目录名来自 `robot.py` 的默认命名，与 V550
无关），**不是** launch 默认指向的路径，需要手动拷：

```bash
cp src/NeuPAN/example/dune_train/model/omni_robot_default/model_5000.pth \
   src/neupan_ros/example/gazebo_V550/pretrain_V550_mec/
```

### 5.3 训练过程

CPU 训练，数据集生成约 80 s（10 万次 ECOS 求解，无进度条，看着像卡住），5000 epoch
约 44 min。cvxpy 会报 ECOS FutureWarning，无害（`dune_train.py` 写死了
`solver=cp.ECOS`）。

启动时打印的 G、h 可用来确认 yaml 被正确读入：

```
robot_G = [[0, -0.2600], [0.2446, 0], [0, 0.2600], [-0.2446, 0]]
robot_h = [0.0318] × 4
```

未归一化形式，四条边约束除以各自模长后即 `|x| ≤ 0.1223`、`|y| ≤ 0.13`。

**5000 epoch 是浪费的**：

| | epoch 0 | epoch 250 | epoch 5000 |
|---|---|---|---|
| Mu Loss | 3.93e+00 | 2.43e-04 | 7.62e-05 |
| Distance Loss | 9.71e+00 | 3.16e-06 | 6.29e-07 |
| Fa Loss | 5.00e-01 | 3.11e-05 | 9.85e-06 |
| Fb Loss | 5.55e+00 | 3.75e-05 | 2.88e-06 |

前 250 epoch 掉了 4~6 个数量级，剩下 4750 个只把 Distance Loss 再降 5 倍。**同尺寸
的车下次 1500 epoch 足够**（约 13 min）。train / validate 全程贴合，无过拟合。
epoch 1250 和 2250 各有一次 validate 尖峰（高 3~4 倍），下一条记录即恢复，
train loss 同期未动——Adam 偶发大梯度，学习率衰减后未再出现。

### 5.4 精度验证

Loss 是 MSE，无物理意义。加载权重与 cvxpy 精确解逐点对比（同一个 QP，
`solver=cp.ECOS`）。先确认 QP 最优值就是**点到矩形的欧氏距离**（`p=[0.5,0]` → 0.370000
= 0.5−0.13；`p=[1,1]` → 1.235823，与几何公式一致），所以下表可直接当测距误差读：

| 距离带 | MAE | max\|e\| | bias |
|---|---|---|---|
| 贴身 0~0.1 m | 0.794 mm | 15.5 mm | −0.794 mm |
| 近 0.1~0.3 m | 0.363 mm | 2.6 mm | −0.345 mm |
| 中 0.3~1.0 m | 0.303 mm | 0.9 mm | +0.013 mm |
| 远 1.0~4.0 m | 0.638 mm | 3.7 mm | −0.049 mm |
| 外插 4~6 m（训练范围外） | 1.092 mm | 5.1 mm | −0.248 mm |

参照物：Gazebo ray 传感器 `stddev=0.01`（10 mm）。网络误差比传感器噪声小一个数量级，
**不是精度瓶颈**。

**bias 全为负 = 系统性低估距离 = 偏安全方向**（认为障碍物比实际更近，会更早避让）。
贴身带 3000 点中**高估超过 1 mm 的有 0 个**。

按位置分组（贴身带 3000 点）：长边外 MAE 0.528 mm、短边外 0.594 mm、角点外
0.632 mm。角点略大是预期的——那里是距离函数的不可微处。

`max` 一列在不同随机种子下波动较大（一轮 15.5 mm，另一轮 3000 点最大仅 3.97 mm），
说明是少数孤立点，不是区域性偏差。

**碰撞判定完全一致**：1.5 m 范围内 4000 个随机点，按 `collision_threshold: 0.05`
判定，网络与精确解 **0 处不一致**。

---

## 6. 仿真环境

`gazebo_V550_mec.launch` 从 `turn_on_wheeltec_robot/launch/` 迁移到
`neupan_ros/example/gazebo_V550/launch/gazebo_V550_simple_s1.launch`，与 limo 示例
同构，世界换成 `vehicle_simulator` 的世界文件，两台车跑同一套场景可直接对比。
迁移前已 grep 确认无其他引用。

> 文件名里的 `simple_s1` 与默认世界 `mix_D1_new` 不一致——这是沿用
> `gazebo_limo_simple_s1.launch` 的命名（它也这样）。改名会打断与 limo 的对应关系，
> 故保持原样。实际加载哪个世界看 `world` 参数。

### spawn 高度必须给贴地值

`z = 0.02027`（= 轮半径 0.0372 − 轮心高度 0.0167），**不能像差速车那样给个高度让它
自己掉下来**。`planar_move` 每个物理步把 z 线速度钳成 0，重力基本不生效：实测
spawn 在 z=0.5 时小车只掉到 0.312 就悬在空中。贴地 spawn 后稳定在 base_link
z ≈ 0.0183（接触穿透 1.8 mm），实测末态 z = 0.01830，与预测一致。

### 起点选 (-3, -3)

与 limo 一致，便于对比。**不要用 (0,0)**：`mix_D1_new` 原点附近 0.679 m 就有障碍物，
车体半对角 0.1785 m，一起步就贴着障碍物。(-3,-3) 处净空 1.543 m。

两个世界的障碍物高度都是 0~0.45/0.5 m，V550 雷达在 z ≈ 0.117 m，扫得到。

### base_footprint 的父子方向与实车相反

URDF 里没有 `base_footprint`（实车靠 `robot_model_visualization.launch` 的静态 TF 补）。
仿真里是 `base_link -> base_footprint (-0.02027)`，实车是
`base_footprint -> base_link (+0.02027)`。原因：`planar_move` 广播的
`odom -> base_link` 已把 base_link 定在真实位姿上，base_link 不能有第二个父节点。
TF 查询不看父子方向，costmap 一样能用。

**仿真时不要再起实车那条 `base_to_link` 静态 TF 节点。**

---

## 7. 启动

```bash
# 终端 1：仿真环境
roslaunch neupan_ros gazebo_V550_simple_s1.launch

# 终端 2：规划器 + rviz
roslaunch neupan_ros neupan_gazebo_V550.launch
```

默认 `wait_for_start:=true`，节点起来后原地待命。触发方式：rviz 工具栏点
**2D Nav Goal**（`start_trigger_mode:=goal` 时点击位置即终点），或
`rostopic pub -1 /neupan_start std_msgs/Bool "data: true"`。

常用覆盖：

```bash
# 换静态世界（推荐先用这个验证，见第 8 节）
roslaunch neupan_ros gazebo_V550_simple_s1.launch world:=simple_S1

# 指定终点（世界坐标，不是相对车头）
roslaunch neupan_ros neupan_gazebo_V550.launch goal:="3 0"

# 不等触发直接跑
roslaunch neupan_ros neupan_gazebo_V550.launch wait_for_start:=false
```

关键参数（`neupan_gazebo_V550.launch`，18 项已用 `--dump-params` 验证）：

| 参数 | 值 | 说明 |
|---|---|---|
| `map_frame` | `odom` | |
| `base_frame` | `base_link` | planar_move 的 robotBaseFrame，**不是** base_footprint |
| `lidar_frame` | `laser_link` | |
| `scan_range` | `0.0 8.0` | 雷达能到 12，但 `dune_max_num=100` 限制每帧点数，放到 12 会让远处无关点挤掉近处关键点 |
| `scan_angle_range` | `-3.15 3.15` | 全周 |
| `scan_downsample` | `3` | 360/3=120，与 limo 的 720/6=120 量级持平 |
| `omni_yaw_mode` | `none` | 见 3.3 |
| `goal` | `3 0` | 世界坐标 |

---

## 8. 实测结果

### 8.1 测量方法（先读这段，否则下面的数字会被误读）

碰撞判定用 **SAT（分离轴定理）算两个有向矩形的精确重叠**，位姿从
`/gazebo/model_states` 取**实时值**，约 19 Hz 采样。

同时报告两个矩形：

- **NeuPAN 碰撞矩形** 0.26 × 0.2446：规划器自己的安全边界
- **真实车体截面** 0.2532 × 0.1556：物理接触判据

两者差 0.089 m（宽度方向），因为 NeuPAN 的矩形要覆盖车轮。NeuPAN 矩形被侵入但车体
无接触 = 安全余量被吃掉但没真撞上。

> **两个早期方法错误，结论已作废**：
> ① 最初用"车体中心到障碍物的距离"对比半对角（外接圆），过于保守；
> ② 更严重的是，最初按**世界文件里的静态坐标**算净空——而 `mix_D1_new` 的 21 个
> 障碍物里**有 10 个会动**（`libwaypointObstaclePlugin.so`）。按静态坐标算出的
> "轻微擦碰"结论完全不可靠。下面的数字全部来自实时位姿重测。

### 8.2 静态世界 `simple_S1`（20 障碍物，全静态）

起点 (-3,-3) → 终点 (3,0)，三次重复：

| 次 | 结果 | 到达误差 | NeuPAN 矩形最小间隙 | 车体最小间隙 | 碰撞 |
|---|---|---|---|---|---|
| 1 | 到达 | 0.119 m | +0.4125 m | +0.4369 m | 无 |
| 2 | 到达 | 0.115 m | +0.4118 m | +0.4362 m | 无 |
| 3 | 到达 | 0.119 m | +0.4125 m | +0.4369 m | 无 |

三次几乎完全重合（最近点都是 `box_11`，同一位置同一时刻区间），yaw 范围
0.15~0.99°。**零碰撞，高度可重复。**

### 8.3 动态世界 `mix_D1_new`（21 障碍物，其中 10 个动态）

同一份配置，三次重复：

| 次 | 结果 | 到达误差 | 车体最小间隙 | 碰撞 | 对象 |
|---|---|---|---|---|---|
| 1 | 到达 | 0.150 m | +0.0882 m | 无 | `d1_crossing_02` [动态] |
| 2 | 到达 | 0.139 m | **−0.1648 m** | **物理碰撞** | `d1_crossing_01` [动态] |
| 3 | **未到达** | 10.583 m | **−0.1228 m** | **物理碰撞** | `d1_crossing_01` [动态] |

**三次碰撞全部发生在动态障碍物上，静态障碍物一次都没撞到。**

`d1_crossing_01`：0.25 × 0.2 × 0.5 的盒子，`velocity 0.19` m/s（约为小车速度的 2/3），
`loop 1`，`angular_velocity 0.8`。

两个必须强调的点：

1. **"到达终点"不能作为没碰撞的证据。** 第 2 次撞了 0.165 m 仍然报到达——
   `planar_move` 直接置速度，撞上了照样把车推过去。第 3 次被撞飞 10 m。
   验收动态场景必须单独测碰撞。
2. **这不是调参问题。** NeuPAN 把每帧激光点当静止障碍物，没有速度估计、没有预测，
   对运动目标永远是按上一帧位置在规划。

### 8.4 `d_min` 参数扫描

`collision_threshold` 只是急停判据（`check_stop()` 里 `min_distance < threshold`，
实测全程未触发）。真正控制避障余量的是 `adjust.d_min`。

在动态世界里扫：

| `d_min` | 碰撞 | 到达 | 备注 |
|---|---|---|---|
| 0.01（当前值） | 2/3 次 | 2/3 次 | |
| 0.03 | 1/3 次 | 3/3 次 | |
| 0.05 | 0/1 次 | 是 | 但 NeuPAN 矩形只剩 9.1 mm 余量 |
| 0.08 | 0/1 次 | **否** | 卡在离终点 3.08 m 处 |

**没有一个值能同时做到不撞和能到达。** 余量大到能躲开动态障碍物时，车就过不去了。

**`d_min` 保持 0.01 未改**：只有动态场景的数据，不足以定一个可能拖累静态场景的新值；
而静态场景下 0.01 表现很好（间隙 0.41 m，零碰撞）。

---

## 9. 已知限制

**动态障碍物（结构性，不是配置问题）。** 见 8.3。NeuPAN 无速度估计与预测。要做动态
避障需要在 NeuPAN 之外加一层预测，这不是调参能解决的。**建议：验证链路与调参都用
`simple_S1`（全静态、结果可重复）；`mix_D1_new` 只作压力测试，其结果不能用来评判
NeuPAN 静态性能。**

**定位。** 纯轮式里程计，无 AMCL / SLAM / EKF。仿真里 `planar_move` 直接给真值位姿
所以无漂移，**实车会累积漂移**。

**TF 时间戳错配。** `scan_callback` 用 `rospy.Time(0)` 取最新可用 TF，非时间戳对齐，
最坏约 100 ms。快速运动时点云拖影。未处理。

**`omni_yaw_mode` 若改成 `heading`。** limo 那套抑制摇头的参数
（`omni_yaw_tau=0.3`、`omni_yaw_deadband=0.12`）是针对 limo ±120° FOV 的
"转向↔观测"耦合调的。V550 是 360° 雷达，不存在该耦合，这套值可能过于保守导致车头
明显滞后。**具体数值需实测，目前没有 V550 上的测量结果。**

**训练超参未调优。** 除 `length`/`width`/`data_range` 外全部照抄官方 diff 模板。

---

## 10. 文件清单

新增：

```
neupan_ros/example/gazebo_V550/
├── README.md                                  ← 本文
├── config/neupan_planner_V550_mec.yaml
├── launch/gazebo_V550_simple_s1.launch        ← 从 turn_on_wheeltec_robot 迁入
├── launch/neupan_gazebo_V550.launch
├── rviz/V550_gazebo.rviz
└── pretrain_V550_mec/model_5000.pth           ← 训练产物

NeuPAN/example/dune_train/
├── dune_train_V550_mec.py
└── dune_train_V550_mec.yaml
```

删除：`turn_on_wheeltec_robot/launch/gazebo_V550_mec.launch`（已迁移）

修改：见 3.1、3.2 的表格

---

## 11. 验证边界

**有实测支撑：**

- 尺寸：STL 顶点 + URDF 几何解析
- DUNE 精度：与 cvxpy 精确解逐点对比，5 个距离带 + 位置分组 + 碰撞判定一致性
- URDF 结构：8 个 link、无悬空 joint、无 gazebo 引用不存在的 link
- 两台车 xacro 展开：均为 `laser_link` + `/scan`
- 参数加载：`--dump-params` 18 项
- odom 原点 = 世界原点：读 `/odom` 实测
- spawn 高度：末态 z = 0.01830，与预测一致
- 静态世界：3 次重复，零碰撞
- 动态世界：3 次重复，2 次碰撞
- `d_min` 扫描：0.01 / 0.03 / 0.05 / 0.08

**没有实测支撑（勿当结论）：**

- 实车行为——全部结果来自 Gazebo
- `omni_yaw_mode: heading` 的参数取值
- 6~8 m 段的 DUNE 精度（`scan_range` 上限是 8.0，只验到 6 m）
- 训练超参是否最优
- 动态世界 `d_min=0.05` / `0.08` 各只跑 1 次，样本不足

**已作废的错误结论：** 8.1 里那两条早期方法错误。任何按静态坐标算出的 `mix_D1_new`
净空数字都不可信。

---

## 附录：验证脚本

`/tmp/trace_live.py`——实时位姿 + SAT 矩形重叠检测。换车只需改 `NP`（NeuPAN 矩形）
和 `BODY`（真实车体截面）两个元组。

`/tmp/val_dune.py`、`/tmp/val2.py`——DUNE 权重精度验证。换车改 `L`/`W` 和 checkpoint
路径。

（这些在 `/tmp` 下，重启即失。要长期用请拷进仓库。）

