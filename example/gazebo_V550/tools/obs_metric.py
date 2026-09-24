#!/usr/bin/env python3
"""避障场景指标: 重点看"倒车"和"朝向"。

倒车的判据不能用车体系 linear.x < 0 —— 全向底盘横移时车体系 x 分量本来就可能
为负。要看速度在**局部路径切向**上的投影: 投影为负才是真的在往回走。

路径切向取 /neupan_plan 的前两个点之差(世界系)。cmd_vel 是车体系, 先用
/gazebo/model_states 的 yaw 转到世界系再投影。
"""
import rospy, math, time, sys
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

TAG = sys.argv[1] if len(sys.argv) > 1 else 'obs'
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
GX = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
GY = float(sys.argv[4]) if len(sys.argv) > 4 else 2.5

state = {'yaw': None, 'x': None, 'y': None, 'tang': None}
rows = []          # (t, v_along, v_perp, wz, x, y, yaw)
traj = []
t0 = time.time()


def cb_model(m):
    if 'V550_mec' not in m.name:
        return
    i = m.name.index('V550_mec')
    p, q = m.pose[i].position, m.pose[i].orientation
    state['x'], state['y'] = p.x, p.y
    state['yaw'] = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
    traj.append((time.time() - t0, p.x, p.y))


def cb_plan(m):
    if len(m.poses) < 2:
        return
    a, b = m.poses[0].pose.position, m.poses[1].pose.position
    d = np.array([b.x - a.x, b.y - a.y])
    n = np.linalg.norm(d)
    if n > 1e-9:
        state['tang'] = d / n


def cb_cmd(m):
    if state['yaw'] is None or state['tang'] is None:
        return
    th = state['yaw']
    # 车体系 -> 世界系
    vx_w = m.linear.x * math.cos(th) - m.linear.y * math.sin(th)
    vy_w = m.linear.x * math.sin(th) + m.linear.y * math.cos(th)
    t = state['tang']
    v_along = vx_w * t[0] + vy_w * t[1]
    v_perp = -vx_w * t[1] + vy_w * t[0]
    rows.append((time.time() - t0, v_along, v_perp, m.angular.z,
                 state['x'], state['y'], th))


rospy.init_node('obs_metric', anonymous=True)
rospy.Subscriber('/gazebo/model_states', ModelStates, cb_model)
rospy.Subscriber('/neupan_plan', Path, cb_plan)
rospy.Subscriber('/cmd_vel', Twist, cb_cmd)

while time.time() - t0 < DUR and not rospy.is_shutdown():
    time.sleep(0.2)
    if state['x'] is not None and math.hypot(state['x'] - GX, state['y'] - GY) < 0.3:
        break

if len(rows) < 20:
    print('%-14s 样本不足 %d' % (TAG, len(rows)))
    sys.exit()

r = np.array(rows)
t, va, vp, wz = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
x, y, yaw = r[:, 4], r[:, 5], r[:, 6]
spd = np.hypot(va, vp)
moving = spd > 1e-3

back = va < -1e-3
# 连续倒车段
seg, cur = [], 0
for b in back:
    if b:
        cur += 1
    elif cur:
        seg.append(cur)
        cur = 0
if cur:
    seg.append(cur)
hz = len(t) / max(t[-1] - t[0], 1e-6)

a = np.array(traj)
err = math.hypot(a[-1, 1] - GX, a[-1, 2] - GY)

print('=' * 78)
print('%s   目标 (%.1f, %.1f)   末点 (%.3f, %.3f)  距目标 %.3f  %s'
      % (TAG, GX, GY, a[-1, 1], a[-1, 2], err, '到达' if err < 0.35 else '未到达'))
print('时长 %.1fs  cmd 帧 %d (%.1f Hz)' % (t[-1] - t[0], len(t), hz))
print('-' * 78)
print('沿路径 v_along : %+.4f ~ %+.4f   均值 %+.4f' % (va.min(), va.max(), va.mean()))
print('垂直路径 v_perp: %+.4f ~ %+.4f   |均值| %.4f  std %.4f'
      % (vp.min(), vp.max(), abs(vp.mean()), vp.std()))
print('倒车(v_along<0): %d/%d 帧 = %.1f%%   最长连续 %.2f s   累计 %.2f s'
      % (back.sum(), len(back), 100.0 * back.sum() / len(back),
         (max(seg) / hz) if seg else 0.0, (sum(seg) / hz) if seg else 0.0))
print('angular.z      : %+.4f ~ %+.4f   |均值| %.4f   非零帧 %.1f%%'
      % (wz.min(), wz.max(), abs(wz.mean()), 100.0 * np.mean(np.abs(wz) > 1e-4)))
print('yaw            : %+.1f ~ %+.1f deg   总转过 %.1f deg'
      % (math.degrees(yaw.min()), math.degrees(yaw.max()),
         math.degrees(np.sum(np.abs(np.diff(np.unwrap(yaw)))))))
print('原地打转(|v|<0.02 且 |wz|>0.05): %.1f%% 帧'
      % (100.0 * np.mean((spd < 0.02) & (np.abs(wz) > 0.05))))
if moving.sum() > 5:
    ang = np.degrees(np.arctan2(vp[moving], va[moving]))
    print('速度相对路径的夹角: %+.1f ~ %+.1f deg  std %.1f' % (ang.min(), ang.max(), ang.std()))

if seg:
    print('-' * 78)
    print('倒车最严重的那一段前后 (v_along 最负处):')
    k = int(np.argmin(va))
    lo, hi = max(0, k - 6), min(len(t), k + 7)
    print('   %6s %9s %9s %8s | %8s %8s' % ('t', 'v_along', 'v_perp', 'wz', 'x', 'y'))
    for i in range(lo, hi):
        print('   %6.2f %+9.4f %+9.4f %+8.4f | %+8.3f %+8.3f%s'
              % (t[i], va[i], vp[i], wz[i], x[i], y[i], '  <--' if i == k else ''))
