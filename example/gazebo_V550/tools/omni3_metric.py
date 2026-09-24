#!/usr/bin/env python3
"""omni3 (u = vx, vy, w) 的实测指标。

与 obs_metric.py 的区别: omni3 的角速度是**规划器算出来**的, 不是外挂补偿环,
所以要看的是"车头朝向是否跟住了参考路径方向", 而不是"摇头幅度有多大"。

新增两个量:
  yaw_err   : 车头 theta 与**参考路径**方向的偏差。omni3 的状态代价在跟踪它,
              所以它才是朝向控制质量的直接判据。
  body_angle: 车体系速度矢量与车头的夹角。omni 下这个值很大(纯蟹行),
              omni3 下应当很小(车头对准行进方向)。

倒车判据仍然必须在**路径坐标系**里做: 全向底盘横移时车体系 x 分量为负是正常的。
路径切向取 /neupan_plan 前两点之差(世界系)。
"""
import rospy, math, time, sys
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

TAG = sys.argv[1] if len(sys.argv) > 1 else 'omni3'
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
GX = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
GY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

st = {'yaw': None, 'x': None, 'y': None, 'tang': None, 'ref_yaw': None}
rows = []          # t, v_along, v_perp, wz, x, y, yaw, yaw_err, body_angle
traj = []
t0 = time.time()


def cb_model(m):
    if 'V550_mec' not in m.name:
        return
    i = m.name.index('V550_mec')
    p, q = m.pose[i].position, m.pose[i].orientation
    st['x'], st['y'] = p.x, p.y
    st['yaw'] = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
    traj.append((time.time() - t0, p.x, p.y, st['yaw']))


def cb_ref(m):
    """/neupan_ref_state: 参考状态序列。第 0 个点的 theta 就是 omni3 状态代价
    正在跟踪的朝向参考。"""
    if not m.poses:
        return
    q = m.poses[0].pose.orientation
    st['ref_yaw'] = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def cb_plan(m):
    if len(m.poses) < 2:
        return
    a, b = m.poses[0].pose.position, m.poses[1].pose.position
    d = np.array([b.x - a.x, b.y - a.y])
    n = np.linalg.norm(d)
    if n > 1e-9:
        st['tang'] = d / n


def cb_cmd(m):
    if st['yaw'] is None or st['tang'] is None:
        return
    th = st['yaw']
    # cmd_vel 是车体系 -> 转世界系
    vx_w = m.linear.x * math.cos(th) - m.linear.y * math.sin(th)
    vy_w = m.linear.x * math.sin(th) + m.linear.y * math.cos(th)
    t = st['tang']
    v_along = vx_w * t[0] + vy_w * t[1]
    v_perp = -vx_w * t[1] + vy_w * t[0]

    yaw_err = float('nan')
    if st['ref_yaw'] is not None:
        yaw_err = math.atan2(math.sin(st['ref_yaw'] - th), math.cos(st['ref_yaw'] - th))

    # 车体系速度矢量与车头(+x)的夹角
    body_angle = float('nan')
    if math.hypot(m.linear.x, m.linear.y) > 0.05:
        body_angle = abs(math.atan2(m.linear.y, m.linear.x))

    rows.append((time.time() - t0, v_along, v_perp, m.angular.z,
                 st['x'], st['y'], th, yaw_err, body_angle))


rospy.init_node('omni3_metric', anonymous=True)
rospy.Subscriber('/gazebo/model_states', ModelStates, cb_model)
rospy.Subscriber('/neupan_plan', Path, cb_plan)
rospy.Subscriber('/neupan_ref_state', Path, cb_ref)
rospy.Subscriber('/cmd_vel', Twist, cb_cmd)

r = rospy.Rate(20)
while not rospy.is_shutdown() and time.time() - t0 < DUR:
    r.sleep()

D = 180.0 / math.pi
if not rows:
    print('[%s] 没有采到数据 (cmd_vel / model_states / neupan_plan 有没有在发?)' % TAG)
    sys.exit(1)

a = np.array(rows)
t, va, vp, wz, x, y, yaw, yerr, bang = a.T
sp = np.hypot(va, vp)
tr = np.array(traj)

print('=== [%s] %d 帧 cmd_vel, %.1fs (%.1f Hz) ===' % (TAG, len(a), t[-1], len(a) / t[-1]))
print('末点 (%.3f, %.3f)  距目标(%.1f,%.1f) %.3f m'
      % (x[-1], y[-1], GX, GY, math.hypot(x[-1] - GX, y[-1] - GY)))
print('--- 平移 (路径坐标系) ---')
print('  v_along : %+.4f ~ %+.4f  mean %+.4f' % (va.min(), va.max(), va.mean()))
print('  v_perp  : %+.4f ~ %+.4f  |mean| %.4f  std %.4f'
      % (vp.min(), vp.max(), abs(vp.mean()), vp.std()))
print('  倒车 (v_along < -0.01): %d/%d = %.1f%%'
      % ((va < -0.01).sum(), len(a), 100.0 * (va < -0.01).mean()))
print('--- 角速度 (规划器输出, 不是补偿环) ---')
print('  wz      : %+.4f ~ %+.4f  |mean| %.4f  非零帧 %.1f%%'
      % (wz.min(), wz.max(), abs(wz.mean()), 100.0 * (np.abs(wz) > 1e-4).mean()))
if len(tr) > 1:
    print('  yaw     : %+.1f ~ %+.1f deg  总转过 %.1f deg'
          % (tr[:, 3].min() * D, tr[:, 3].max() * D,
             np.sum(np.abs(np.diff(tr[:, 3]))) * D))
print('  原地打转 (|v|<0.02 且 |wz|>0.05): %.1f%% 帧'
      % (100.0 * ((sp < 0.02) & (np.abs(wz) > 0.05)).mean()))
print('--- 朝向跟踪质量 (omni3 的新指标) ---')
m = ~np.isnan(yerr)
if m.any():
    print('  yaw_err (车头 vs 参考路径方向): |mean| %.2f deg  max %.2f deg  RMS %.2f deg'
          % (abs(np.mean(yerr[m])) * D, np.max(np.abs(yerr[m])) * D,
             math.sqrt(np.mean(yerr[m] ** 2)) * D))
m = ~np.isnan(bang)
if m.any():
    print('  body_angle (车体速度矢量 vs 车头): mean %.1f deg  max %.1f deg'
          % (np.mean(bang[m]) * D, np.max(bang[m]) * D))
    print('    参考: omni(2D) 纯蟹行时这个值很大; omni3 对准行进方向时应当很小')
if len(tr) > 2:
    arc = np.sum(np.hypot(np.diff(tr[:, 1]), np.diff(tr[:, 2])))
    net = math.hypot(tr[-1, 1] - tr[0, 1], tr[-1, 2] - tr[0, 2])
    print('--- 轨迹 ---')
    print('  弧长比 %.4f   起点 (%.2f, %.2f) -> 末点 (%.2f, %.2f)'
          % (arc / max(net, 1e-9), tr[0, 1], tr[0, 2], tr[-1, 1], tr[-1, 2]))

np.save('/tmp/omni3_%s.npy' % TAG, a)
print('原始数据 -> /tmp/omni3_%s.npy' % TAG)
