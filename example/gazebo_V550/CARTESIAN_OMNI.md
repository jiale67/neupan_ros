# 全向底盘控制量改为笛卡尔 u = (vx, vy)

**改动范围**：upstream NeuPAN 的 `omni` 运动学分支（5 个文件）+ 2 份 yaml 配置
**影响面**：所有 `kinematics: 'omni'` 的配置。`diff` / `acker` 代码路径逐行未动。
**状态**：空场景与静态避障场景已实测；有一项用户上报现象未能复现，见第 8 节。

本文说明为什么改、改了什么、每处改动背后的原理，以及哪些结论有实测支撑。

---

## 0. 一句话总结

原来 `omni` 的控制量是**极坐标** `u = (v, φ)`（速度大小 + 行进方向角）。这个参数化
需要线性化，而线性化在大角度下是病态的，导致约 1 Hz 的方向角极限环（实测方向角
`std 22.9°`、弧长比 1.11）。改成**笛卡尔** `u = (vx, vy)` 后模型变成**精确线性**
（`A = I`、`B = dt·I`、`C = 0`，不依赖展开点），抖动从原理上消失：空场景方向角
`std 0.005°`、弧长比 `1.00000`。

代价的写法不能照抄 `diff`，因此引入了一个新配置项 `lateral_ratio`，见第 4 节。

---

## 1. 原来的极坐标参数化有什么问题

### 1.1 NeuPAN 的求解结构

NeuPAN 的 NRMP 层是一个 `CvxpyLayer`，也就是**一个凸问题**。但机器人模型
`x_{t+1} = f(x_t, u_t)` 一般是非线性的，所以每帧要在标称轨迹 `(nom_s, nom_u)` 处做
一阶泰勒展开，得到 `x_{t+1} = A x_t + B u_t + C`，再把 `A/B/C` 作为 `Parameter`
喂进凸问题。这是标准的 SCP（sequential convex programming）。

关键点：**线性化误差有多大，取决于每步允许偏离展开点多远**。而"允许偏离多远"就是
`max_acce * step_time` —— 加速度界在这里兼任了信赖域半径。

### 1.2 极坐标下的三个缺陷

原 `linear_omni_model` 的状态转移是

```
x_{t+1} = x_t + v_t · cos(φ_t) · dt
y_{t+1} = y_t + v_t · sin(φ_t) · dt
```

`φ` 是**控制量**。对它求偏导得到的 `B` 依赖展开点 `(v_nom, φ_nom)`，必须线性化。
三个问题：

**(a) 信赖域远超泰勒展开有效范围。** V550 配置里 `max_acce[1] = 3.14`，
`step_time = 0.3`，于是一步允许 `3.14 × 0.3 = 0.942 rad = 54°` 的方向角变化。
在 54° 处线性化位移的**幅值**是 `dt·v·√(1 + dφ²)`，比真实值（恒为 `dt·v`）大
**37.4%**；`dφ = 180°` 时大 **229.7%**。

**(b) 模型误以为"转开方向就能走得更远"。** 上面那个幅值随 `|dφ|` 单调增大，而线性化
出来的横向分量 `dt · v · dφ`（沿 `φ_nom` 的法向）既不消耗前向速度，**也不进代价
函数**（代价只惩罚 `v` 偏离 `ref_speed`，`φ` 完全不受惩罚）。于是 QP 会主动利用这个
假象，把 `φ` 顶到信赖域边界；下一帧按真实模型积分后发现没走那么远，于是反向再顶满 ——
过冲、反号、过冲，形成约 1 Hz 的极限环。

**(c) `v = 0` 处 `φ` 不可辨识。** `B` 的 `φ` 列是 `[-v·sin(φ)·dt, v·cos(φ)·dt, 0]ᵀ`，
`v = 0` 时**恒为零向量**，`φ` 对状态没有任何影响，QP 对它的取值完全无差别。起步和停车
时刻的 `φ` 是纯噪声。另外代码里对 `φ` 没有任何 `WrapToPi`，±π 附近会出现 2π 跳变。

### 1.3 为什么 `diff` 没有这个问题

同样是非线性模型，`diff` 免疫，原因是三重的：

| | `diff` | 极坐标 `omni` |
|---|---|---|
| 朝向 `θ` 的身份 | **状态** | **控制量** `φ` |
| 有没有锚 | 有：`indep_s[:,0:1] == para_s[:,0:1]` 把第 0 步钉在测量值上，`dθ[0] = 0`，第一步线性化误差是 **1e-17** | 无。第 0 步就能跳 54° |
| 动力学 | `θ` 的更新是 `θ + ω·dt`，`B` 第三行 `[0, dt]`，**精确线性** | 幅值随 `dφ` 病态增长 |
| 有没有代价 | 有：`sum_squares(diff_s)` 含全部三行，`θ` 偏离参考要付代价 | 无：`φ` 不在代价里 |

也就是说极坐标 `omni` 恰好把 `diff` 的三道保险全部拆掉了。

补充一点：极坐标只是个**中间表示**，两头本来都是笛卡尔。`neupan.py` 拿到 `(v, φ)`
后立刻 `v·cos(φ), v·sin(φ)` 转回去下发，`initial_path.omni_model` 积分时也要转。
也就是说这个非线性参数化没有换来任何好处，纯粹是在中间插了一次可能出错的转换。

---

## 2. 笛卡尔模型：精确线性

`robot.py` 的 `linear_omni_model` 现在是：

```python
A = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
B = torch.Tensor([[dt, 0], [0, dt], [0, 0]])
C = torch.zeros((3, 1))
```

对应

```
x_{t+1} = x_t + vx_t · dt
y_{t+1} = y_t + vy_t · dt
θ_{t+1} = θ_t
```

`A/B/C` 全是常数，**不依赖展开点 `nom_u`**。这意味着：

- 一阶泰勒展开就是模型本身，**线性化误差恒为 0**，与步长、与 `dφ` 无关
- SCP 外层迭代不引入任何误差，1.2(a)(b) 两个问题同时消失
- `max_acce` 不再兼任信赖域半径，回归它本来的物理含义（对地加速度上限）
- 没有 `v = 0` 奇点（`B` 与 `u` 无关），没有角度需要归一化，1.2(c) 也消失

`nom_u` 保留在函数签名里只为跟 `diff`/`acker` 的调用形式一致，函数体不使用它。

`B` 第三行仍是 `[0, 0]`，即 **`θ` 不可控**。这是 NeuPAN omni 模型原本就有的性质
（2 自由度模型描述 3 自由度底盘），本次改动没有触碰。车头朝向由规划器**外面**的
`omni_yaw_*` 补偿环处理，见第 7 节。

---

## 3. 换了参数化，代价函数必须跟着改

这是整个改动里唯一有设计自由度的地方，也是踩坑最多的地方。

### 3.1 问题：`ref_speed` 是标量，`u` 变成了矢量

`diff`/`acker` 的速度代价是

```python
diff_u = para_p_u * self.indep_u[0, :] - self.para_gamma_b
```

`indep_u[0]` 是标量前向速度，`para_gamma_b = p_u · ref_speed` 也是标量，含义清晰：
**惩罚前进速度偏离参考速度**。`u[1]`（角速度 / 前轮转角）不惩罚。

笛卡尔 `omni` 下 `u = (vx, vy)` 是个矢量，而 `ref_speed` 仍是标量。"速度偏离参考"
在矢量情形下不是唯一确定的，必须先决定**沿哪个方向**比较。

### 3.2 解法：把速度分解到路径坐标系

用参考路径的**单位切向** `tangent`，把速度分成两个分量：

```
u_along = tangent · u          沿路径分量  -> 惩罚它偏离 ref_speed
u_perp  = normal  · u          垂直分量    -> 按 lateral_ratio 打折惩罚
normal  = (-tangent_y, tangent_x)          切向逆时针转 90°
```

代价（`robot.py` 的 `C0_cost` omni 分支）：

```python
diff_u = para_p_u * self.indep_u_along - self.para_gamma_b
if self.lateral_ratio > 0.0:
    diff_u = cp.hstack(
        [diff_u, (self.lateral_ratio * para_p_u) * self.indep_u_perp])
```

切向从哪来：`initial_path.generate_nom_ref_state` 对参考点做**有限差分**

```python
ref_xy  = ref_s[0:2, :]                   # (2, T+1)
tangent = ref_xy[:, 1:] - ref_xy[:, :-1]  # (2, T)
```

**不用** `ref_s` 第三行的 `theta`。对全向底盘 `theta` 是车头朝向，跟运动方向无关，
而且它上一行刚被 `WrapToPi` 改写过。路径末端相邻参考点重合时切向退化为零，此时
`gear` 已置 0、`ref_us` 是 0，切向填 `(1, 0)` 只为避免除零，乘 0 后不影响结果。

### 3.3 为什么要辅助变量 `indep_u_along` / `indep_u_perp`（DPP 约束）

自然的写法是直接把投影写进代价：

```python
para_p_u * cp.sum(cp.multiply(para_gamma_tangent, indep_u), axis=0)   # 不行
```

这是 **参数 × 参数 × 变量**，不满足 DPP（Disciplined Parametrized Programming），
`CvxpyLayer` 会直接拒绝构建。实测 `cp.sum_squares` 那一项 `is_dpp()` 返回 `False`。

绕过办法是引入辅助变量，把投影写成**参数仿射的等式约束**（`bound_su_constraints`）：

```python
tang   = self.para_gamma_tangent
normal = cp.vstack([-tang[1, :], tang[0, :]])
constraints += [self.indep_u_along == cp.sum(cp.multiply(tang,   self.indep_u), axis=0)]
constraints += [self.indep_u_perp  == cp.sum(cp.multiply(normal, self.indep_u), axis=0)]
```

这样代价里只剩 `para_p_u * indep_u_along`（参数 × 变量，DPP 成立），而
`para_p_u` 仍然是个真正的可微 `Parameter`，`adjust` 接口和 `p_u` 的可调性都不变。

> 走过的弯路：我一开始把 `p_u` **预乘**进切向参数来规避 DPP。那样做 `para_p_u` 在
> omni 问题里就不出现了，但它还留在参数列表里声明着，于是报
> `ValueError: The layer's parameters must exactly match problem.parameters`。
> 辅助变量的写法同时解决了 DPP 和这个参数一致性问题。

注意切向是按**单位矢量**传入的，`nrmp.py` 里没有预乘 `p_u`：

```python
state_value_list = self.robot.generate_state_parameter_value(
    nom_s, nom_u, self.q_s * ref_s, self.p_u * ref_us, ref_tangent
)
```

### 3.4 速度限幅：方形包络换成二阶锥

```python
# omni
constraints += [cp.norm(self.indep_u[:, 1:] - self.indep_u[:, :-1], axis=0)
                <= float(self.acce_bound[0, 0])]
constraints += [cp.norm(self.indep_u, axis=0) <= float(self.speed_bound[0, 0])]
# diff / acker 保持原样
constraints += [cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1]) <= self.acce_bound]
constraints += [cp.abs(self.indep_u) <= self.speed_bound]
```

`cp.abs(u) <= bound` 对 `(vx, vy)` 是个**方形**包络，对角方向会放行 `√2 × max_speed`
（1.0 的配置实际能跑到 1.41 m/s）。`cp.norm(u, axis=0)` 限的是真实**对地速度大小**，
各向同性，这才是麦轮该有的物理含义。

**配置含义因此变了**：`max_speed` / `max_acce` 的第一项现在是**模长**上限，第二项对
omni **不再使用**，保留只为维持 `[.., ..]` 两元素的接口形状。

---

## 4. 新配置项 `lateral_ratio`：横移该不该收费

这是本次改动唯一新增的可调参数，只对 `omni` 生效。它是横向速度代价相对 `p_u` 的
**比例**，实际权重 `= lateral_ratio × p_u`。

### 4.1 为什么是"比例"而不是绝对值

起作用的只有它和 `p_u` 的**比值**。同一个绝对值在不同配置下含义完全不同：
V550 的 `p_u = 2.5`，NeuPAN 自带 example 的 `p_u = 1.0`，绝对值 `0.5` 在前者是
`p_u/5`、在后者是 `p_u/2`。

> 走过的弯路：我第一版就是写成绝对值 `lateral_weight`，默认 0.5。结果 NeuPAN 自带
> 的 omni example 从基线的 34 步 / 弧长比 1.0000 退化到 62 步 / 1.0316，默认改 0.0
> 更差（103 步 / 1.1312）。改成比例后默认 1.0，example 精确回到 34 步 / 1.0000。

### 4.2 三种取值的性质

**`lateral_ratio = 1.0`（各向同性，默认值）**

此时两项合起来**恰好等于**"整个速度矢量偏离参考速度矢量"：

```
sum_squares(p_u·u_along − p_u·ref_speed) + sum_squares(p_u·u_perp)
  ≡ sum_squares(p_u·u − p_u·tangent·ref_speed)
```

因为旋转到路径坐标系不改变 2-范数。这是个**恒等式**，不是近似 —— 所以这个默认值
对任何已有 omni 配置都是行为不变的，不需要为它调参。

代价：横移要付全价，而横移对"沿路径前进"这一项毫无贡献，于是障碍物正前方时
**减速比绕开便宜**。极端情况下 QP 的最优解是停在安全边界上（此时 `I_cost` 恰好为 0，
是个稳定局部极小）。实测 `simple_S1` 绕行要 57.2 s。

**`lateral_ratio = 0.0`（横向完全免费）**

避障灵敏，但**自由等于不受控**。横向*位置*还有 `diff_s` 的 x,y 项往回拉，横向*速度*
则完全没有约束，只剩 `norm(u) <= max_speed` 拦着。实测 `|v|` 冲到 **0.60**
（`ref_speed` 只有 0.3），障碍物附近速度方向 ±90° 跳变；无障碍场景也会横向漂移
（example 弧长比 1.13）。

**`lateral_ratio = 0.2`（V550 采用值）**

横移比前进便宜 5 倍。避障够灵敏，同时横向速度仍被约束住 —— 实测 `v_perp` 标准差
只有 0.0028，沿路径速度稳在 0.303（`ref_speed` = 0.3）。

### 4.3 和极坐标版本的关系

极坐标之所以"能避障"，靠的正是 `φ` **完全不受惩罚** —— 那既是抖动的根源，也是避障的
唯一机制。所以它等价于 `lateral_ratio = 0`，只是外面还套了一层需要线性化的非线性模型。
笛卡尔版本把两件事拆开了：方向自由度保留，但**按比例定价**而不是二选一，而且不再经过
线性化步骤。

---

## 5. 逐文件改动清单

切向这个新参数要从路径生成一路传到凸问题里，中间经过 4 层，所以改动点比较分散。

### 5.1 `NeuPAN/neupan/robot/robot.py`（核心）

| 函数 | 改动 |
|---|---|
| `__init__` | 读入 `lateral_ratio`（默认 1.0） |
| `define_variable` | omni 下新增 `indep_u_along` / `indep_u_perp` 两个辅助变量 `(T,)` |
| `state_parameter_define` | omni 下新增 `para_gamma_tangent` `(2, T)`，并按条件加进参数列表 |
| `C0_cost` | omni 分支改为路径坐标系分解 + `lateral_ratio` 加权 |
| `bound_su_constraints` | omni 用二阶锥限幅；新增两条投影等式约束 |
| `generate_state_parameter_value` | 签名加 `ref_tangent=None`；omni 缺它直接 `raise` |
| `linear_omni_model` | 改为常数 `A/B/C`（见第 2 节） |

`para_gamma_b` 仍是 `(T,)` 标量，含义不变（沿路径方向的参考速度大小）。参数列表的
**顺序**必须与 `state_parameter_define` 的返回顺序严格一致，否则 `CvxpyLayer` 会
静默错位。

### 5.2 `NeuPAN/neupan/blocks/initial_path.py`

- `generate_nom_ref_state`：omni 时**多返回第 5 个元素** `unit_tangent`（见 3.2）
- `omni_model`：积分时按世界系笛卡尔速度处理，不再做极坐标转换

```python
# vel 是**世界系**笛卡尔速度 (vx, vy), 与 robot.linear_omni_model 一致。
omni_vel = np.array([[vel[0, 0]], [vel[1, 0]], [0]])
```

### 5.3 `NeuPAN/neupan/blocks/pan.py` 与 `nrmp.py`

两处都只是把 `ref_tangent: torch.Tensor = None` 加到 `forward` 签名末尾并往下传。
放在**末尾**且给默认值，`diff`/`acker` 的调用方完全不用改。

### 5.4 `NeuPAN/neupan/neupan.py`

两处改动。一是取切向时**用切片而不是展开**：

```python
ref_tangent_tensor = nom_input_tensor[4] if len(nom_input_tensor) > 4 else None
opt_state_tensor, opt_vel_tensor, opt_distance_tensor = self.pan(
    *nom_input_tensor[:4], obstacle_points_tensor, point_velocities_tensor,
    ref_tangent_tensor
)
```

如果直接 `*nom_input_tensor` 展开，多出来的切向会顶到 `obs_points` 的位置上 ——
这是个不会报错、只会静默算错的坑。

二是**删掉了极坐标→笛卡尔的回转**。原来这里做 `v·cos(φ), v·sin(φ)`，现在 `action`
就是 `(vx, vy)` 直接下发；`omni_linear_speed` / `omni_orientation` 只作为诊断信息
写进 `info`。

### 5.5 yaml 配置

`gazebo_V550/config/neupan_planner_V550_mec.yaml`：

```yaml
  max_speed: [1.0, 1.0]     # [0] = 对地速度模长上限；[1] 对 omni 不再使用
  max_acce: [1.0, 1.0]      # [0] = 加速度模长上限 -> 每步可变 1.0*0.3 = 0.3 m/s
  lateral_ratio: 0.2
```

`gazebo_limo/config/neupan_planner_limo_omni.yaml` 同步更新了 `max_speed` /
`max_acce` 的取值与注释（它走同一条 omni 代码路径）。**注意：limo 的 omni 配置没有
实测过**，见第 9 节。

---

## 6. 实测结果

### 6.1 空场景：抖动消失

沿 26.57° 的直线路径（笛卡尔那一列是离线测的，即直接驱动规划器、不经仿真器）：

| 指标 | 极坐标（改动前） | 笛卡尔（改动后） |
|---|---|---|
| 行进方向角 | std **22.9°** | `+26.57 ~ +26.62`，std **0.005°** |
| 弧长比 | **1.11** | **1.00000** |
| 轨迹最大偏离 | — | **0.00008 m** |
| `\|v\|` | — | 0.2988（`ref_speed` = 0.3） |

正确方向角是 26.57°，笛卡尔版本稳定输出这个值。

Gazebo 空场景巡航段：`|v|` 恒定 0.299，方向角钉在 34.3°，初始整定结束后 `wz` 为 0.000。

### 6.2 静态避障：`simple_S1` 绕 `box_16`

目标 `(6.0, 3.5)`。这个目标是**挑选过**的：起点到目标的直线净空只有 0.001 m
（必须绕），但目标本身离障碍物 0.849 m（可达）。判据由 `tools/traj_shape.py` 给出。

| `lateral_ratio` | 到达 | 用时 | `v_along` 均值 | `\|v\|` 峰值 | 累计 yaw | 原地打转 |
|---|---|---|---|---|---|---|
| 1.0 各向同性 | 是 | 57.2 s | 0.199 | 0.317 | 85.3° | 0% |
| **0.2（采用）** | 是 | **36.5 s** | **0.3035** | 0.382 | 94.5° | **0%** |
| 0.0 横向免费 | 是 | 36.3 s | 0.308 | **0.603** | 121.2° | 0% |

采用配置的完整数据（`omni_yaw_mode:=heading`）：

```
末点 (5.741, 3.502)  距目标 0.259  到达
时长 36.5s  cmd 帧 1819 (49.8 Hz)
沿路径 v_along : +0.2963 ~ +0.3824   均值 +0.3035
垂直路径 v_perp: -0.0327 ~ +0.0626   |均值| 0.0001  std 0.0028
倒车(v_along<0): 0/1819 帧 = 0.0%
angular.z      : -0.6135 ~ +0.6340   |均值| 0.0102   非零帧 28.7%
yaw            : +0.0 ~ +58.9 deg   总转过 94.5 deg
原地打转(|v|<0.02 且 |wz|>0.05): 0.0% 帧
弧长比 1.0180 | 横向偏离 -0.129~+0.350 m | 轨迹最近障碍 0.2751 m
```

**倒车必须在路径坐标系里判定**，不能看车体系 `linear.x`。全向底盘蟹行时车体系 x
为负是完全正常的，不是倒车。这就是 `tools/obs_metric.py` 存在的原因。

### 6.3 回归检查

- `diff` / `acker`：代码路径逐行未动，构建与运行正常
- NeuPAN 自带 omni example：`lateral_ratio` 默认 1.0（≡ 各向同性，见 4.2 的恒等式），
  弧长比 **1.0000**，无横向漂移
- DUNE 权重**不需要重训**：`G`/`h` 只由 `length`/`width` 决定，与控制量参数化无关

---

## 7. 朝向（yaw）不在规划器里

NeuPAN 的 omni 模型结构上不含朝向：`B` 第三行 `[0, 0]`，`C0_cost` 的状态项只算
`diff_s[0:2]`。所以**规划器不可能输出角速度**，这不是 bug。

车头朝向由 `neupan_core.py` 的 `omni_yaw_rate()` 补偿，它在规划器**外面**：

- `omni_yaw_mode: none`（launch 默认）：直接返回 0.0，车保持 spawn 朝向蟹行
- `heading`：车头追 `atan2(vy, vx)`，即行进方向
- `spin`：恒定自转

`heading` 模式下有一条低通 + 死区 + slew 限幅的链路。**这套参数是为极坐标时代的抖动
信号调的**（`tau` 0.3、`deadband` 0.12 rad、`accel` 2.0、`max` 1.0），笛卡尔下方向
信号已经干净得多（空场景 std 0.005°），这些参数现在偏保守 —— 表现为车头略滞后于行进
方向。**尚未按新信号重新标定**。

世界系→车体系的转换在 `neupan_core.py` 里（`gazebo_ros_planar_move` 把 `linear.x/y`
当车体系用）：

```python
action.linear.x =  vx_w * cos(theta) + vy_w * sin(theta)
action.linear.y = -vx_w * sin(theta) + vy_w * cos(theta)
```

---

## 8. 一个未能复现的上报现象

用户反馈在自己的运行里看到两个现象：**避障时偶尔倒车**、**转向时原地打转**。

**结论：在可达目标下，两种参数化我都没能复现。** `lateral_ratio = 1.0`（即用户当时
跑的那一版）在 `(6.0, 3.5)` 上是干净的 —— 倒车 0 帧、累计 yaw 85.3°、无打转，只是
慢（57 s vs 37 s）。

**能稳定复现两个现象的唯一条件是目标不可达。** 我最初用的目标 `(5.0, 2.5)` 离
`box_16` 净空恰好 0 —— 目标本身就贴在障碍物上。那种情况下：

- 车停在安全边界（0.13 车半长 + 0.15 `d_max` = 0.28 m）动不了
- `|v|` 掉到 0.01~0.05，23.9% 的帧低于 `omni_yaw_min_speed` = 0.02，另有 17.3% 落在
  0.02~0.05 这个不稳定带
- `atan2(vy, vx)` 在这个速度下变成纯噪声，方向 ±85° 反复翻
- `heading` 环忠实追过去 → 累计 yaw **679°**，`angular.z` 47% 的帧顶在 ±1.0，
  3.7% 的帧是原地打转
- 外加零星单帧 `v_along` 负值（1.9% 帧，最长连续 0.02 s），看起来就像倒车

也就是说这两个现象是**同一个根因的两个表征**，而那个根因是目标不可达，不是代价函数。

**因此本次的 `lateral_ratio` 改动是照着一个推断出来的机制做的。** 机制本身有实测支撑
（各向同性确实慢一倍、确实偏好减速而不是绕开），但**用户实际看到的现象没有被重现**。
如果你再次看到这两个现象，请记下当时的目标点 —— 如果它也贴着障碍物，那要修的是目标
可达性检查，而不是这里。

---

## 9. 已知限制

**改动引入 / 暴露出来，但没修的：**

1. **`v_along` 会超过 `ref_speed`。** 各向同性下峰值 0.317、`ratio=0.2` 下 0.382
   （`ref_speed` = 0.3）。推断是状态代价在车落后于参考点时驱动它追赶，与参数化无关，
   极坐标版本大概也有 —— 但**我没有测极坐标的这项基线**，所以这是推断不是结论。
2. **`omni_yaw_*` 增益未按新信号重新标定**（见第 7 节）。
3. **limo 的 omni 配置未实测。** 它走同一条代码路径，配置和注释同步改了，但没跑过。

**结构性限制，与本次改动无关：**

4. **一面横跨参考路径的墙会让任何参数化死锁。** `I_cost` 在障碍物法向的**垂直**方向
   梯度恰好为 0（`gamma_c = lam.T` 就是法向，一面沿 y 展开的墙所有法向都指向 −x，
   于是 `∂I_cost/∂y ≡ 0`）。实测三种墙（对称/偏置/短墙）全部死锁在同一个 x = −0.280。
   这不是权重调不调的问题 —— NeuPAN 是**局部**规划器，前提是参考路径大致可行。
   需要全局重规划来解，本次未做。
5. **`θ` 不可控**（2 自由度模型描述 3 自由度底盘），朝向只能靠外部补偿环。
6. `neupan_core.py` 里 `rospy.Rate(50)` 是硬编码的，与 `step_time: 0.3` 不匹配，未动。

---

## 10. 测量工具

都在 `example/gazebo_V550/tools/`：

| 脚本 | 用途 |
|---|---|
| `obs_metric.py` | 把 `cmd_vel` 投影到**路径切向**，判定倒车 / 原地打转。倒车判据必须在路径坐标系做，见 6.2 |
| `traj_shape.py` | 解析 world 文件里的 box，算轨迹实际净空 + **起点到终点的直线净空** —— 后者用来确认一次运行到底算不算避障测试 |
| `yaw_trace.py` | 0.5 s 分箱的 `\|v\|` / 方向 / yaw / `wz` 时序，用来定位失效发生的时刻 |
| `sweep_lw.sh` | `lateral_ratio` 扫描驱动 |
| `sweep_shape.sh` | 轨迹形状扫描驱动 |

> 脚本里 `pkill -f <pattern>` 必须写在**脚本文件**里，不能直接在命令行循环里用：
> 调用它的 shell 自己的 cmdline 含有那个 pattern，`pkill -f` 会自杀。

---

## 11. 结论的证据等级

| 结论 | 支撑 |
|---|---|
| 笛卡尔模型精确线性，空场景抖动消失 | **实测**（离线 + Gazebo） |
| `lateral_ratio` 三档的时间 / 速度 / yaw 差异 | **实测**（`simple_S1`，目标经可达性校验） |
| `ratio = 1.0` ≡ 各向同性全矢量代价 | **数学恒等式** + example 回归实测 |
| `diff` / `acker` 不受影响 | 代码路径未改 + 构建运行验证 |
| 极坐标线性化误差的具体数值（+37.4% @ 54°） | **离线数值验证** |
| 用户上报的倒车 / 打转的根因 | **未复现**，仅有"目标不可达时两症状同时出现"的实测（见第 8 节） |
| `v_along` 超速的原因 | **推断**，未验证 |
| limo omni 配置 | **未测** |

---

## 12. 如何回退

`src/NeuPAN` **本身是个 git 仓库**（外层 `nav_ws` 不是），改动前的状态在提交
`38fbffe 改动前暂存`。所以回退不需要手改代码：

```bash
cd src/NeuPAN
git diff > /tmp/cartesian_omni.patch   # 先留个底，别直接丢掉
git checkout -- neupan/robot/robot.py neupan/blocks/initial_path.py \
                neupan/blocks/nrmp.py neupan/blocks/pan.py neupan/neupan.py
```

yaml 不在这个仓库里，要手工改回去（`neupan_ros` 侧）：

```yaml
  max_speed: [1.0, 3.14]    # 第二项恢复为 phi 的界
  max_acce: [1.0, 3.14]
  # 删掉 lateral_ratio 这一行
```

想对照原始实现看差异：

```bash
git show HEAD:neupan/robot/robot.py | sed -n '/def linear_omni_model/,/return to_device/p'
```

这条命令能直接看到极坐标版本的 `B` 矩阵，`φ` 列是
`[-v·sin(φ)·dt, v·cos(φ)·dt, 0]ᵀ` —— `v = 0` 时恒为零向量，也就是 1.2(c) 那个缺陷。




