# Example in Gazebo with LIMO

## 依赖

**不需要额外安装任何包。** 模型文件（urdf / meshes / control config）都在本目录下，
世界文件用的是 `example/gazebo_V550/world/` 里那套。

> 本例原先依赖 `limo_ros` 和 `rvo_ros` 两个外部仓库：
> - `limo_ros`：提供 limo 的 xacro 和 mesh。已迁入 `urdf/`、`meshes/`、
>   `config/limo_four_diff_control.yaml`。
> - `rvo_ros`：上游用它生成动态障碍物（配合已不存在的
>   `gazebo_limo_env_complex_20.launch`）。这条路径早已失效，包已删除。
>   现在动态世界 `mix_D1_new.world` 的障碍物由
>   `Autonomous_navigation_exploration` 的 `libwaypointObstaclePlugin.so` 驱动。

## 运行

两个终端。第一个起仿真环境：

```bash
roslaunch neupan_ros gazebo_limo_simple_s1.launch
```

第二个起 NeuPAN 控制器：

```bash
roslaunch neupan_ros neupan_gazebo_limo.launch
```

终点可以用 rviz 的 2D Nav Goal 点，也可以在启动时给 `goal:="X Y THETA"`。

### 世界与起点

世界可选 `simple_S1` / `dense_S2` / `mix_D1` / `mix_D1_new` / `corridor` / `maze`
（默认 `mix_D1_new`）：

```bash
roslaunch neupan_ros gazebo_limo_simple_s1.launch world:=simple_S1 x:=-3 y:=-3
```

起点由 `gazebo_limo_simple_s1.launch` 的 `x/y/z/yaw` 决定，终点由
`neupan_gazebo_limo.launch` 的 `goal` 决定，两者独立。

### 差速 / 全向

`drive_mode` 默认 `diff`（skid_steer 四轮滑移，只吃 `linear.x` / `angular.z`）。
切 `omni` 时**两个 launch 都要给**，否则规划器还按差速解算：

```bash
roslaunch neupan_ros gazebo_limo_simple_s1.launch drive_mode:=omni
roslaunch neupan_ros neupan_gazebo_limo.launch   drive_mode:=omni goal:="3.2 2.2"
```

`drive_mode:=omni` 会自动选用 `config/neupan_planner_limo_omni.yaml`。omni 下
NeuPAN 的控制量是笛卡尔 `u = (vx, vy)`，细节见
`../gazebo_V550/CARTESIAN_OMNI.md`。

> 注意 omni 的 `z` 默认值不同（`-0.005` 而不是 `0.15`）：planar_move 插件每个物理步
> 把 z 线速度钳成 0，重力不生效，spawn 在哪个高度就悬在那个高度。launch 里已按
> `drive_mode` 自动切换，不用手给。

## 目录内容

| 路径 | 说明 |
|---|---|
| `launch/` | 仿真环境 + NeuPAN 控制器两个 launch |
| `urdf/` | limo 的 xacro 链（diff / omni，**不含 ackerman**） |
| `meshes/` | `limo_base.dae`、`limo_wheel.dae` |
| `config/neupan_planner_limo*.yaml` | 规划器配置（普通 / noloop / omni） |
| `config/limo_four_diff_control.yaml` | ros_control 的关节控制器配置，仅 `ros_control:=true` 时加载 |
| `pretrain_limo/` | DUNE 权重（按 limo 的 0.322 × 0.220 训的，**不能给 V550 用**） |
| `rviz/` | rviz 配置 |
| `dune_train_limo.py` / `.yaml` | DUNE 重训脚本 |

## 已知不一致

- `config/neupan_planner_limo_noloop.yaml` 对应的
  `neupan_gazebo_limo_noloop.launch` 已不存在。要用它得手动
  `config_file:=.../neupan_planner_limo_noloop.yaml`。
- `urdf/` 只迁了 diff 链，`limo_steering_hinge.xacro`（ackerman 用）没迁。
  本例的 `drive_mode` 本来也只有 `diff|omni`，无功能损失。
- Gazebo 里的模型名是 `limo/` 而不是 `limo` —— launch 用
  `-model 'limo$(arg robot_namespace)'`，默认 namespace `/` 被拼进了名字。
  调 `/gazebo/get_model_state` 要用 `limo/`。
