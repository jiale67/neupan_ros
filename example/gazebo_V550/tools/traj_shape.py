#!/usr/bin/env python3
"""记录实际轨迹, 算它跟障碍物的最近距离和绕行量。
用来确认一次运行到底有没有真的避障 —— 如果轨迹离障碍物很远,
那这次运行就只是纯路径跟踪, 不能当避障测试。"""
import rospy, math, time, sys, re
import numpy as np
from gazebo_msgs.msg import ModelStates

TAG = sys.argv[1] if len(sys.argv) > 1 else 'run'
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
GX = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
GY = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5

W = '/home/rjl/nav_ws/src/Autonomous_navigation_exploration/vehicle_simulator/world/simple_S1.world'
txt = open(W).read()
boxes = []
for m in re.finditer(r"<model name='([^']+)'>(.*?)</model>", txt, re.S):
    body = m.group(2)
    pm = re.search(r'<pose[^>]*>([-\d\.eE\s]+)</pose>', body)
    sm = re.search(r'<box>\s*<size>([-\d\.eE\s]+)</size>', body)
    if not pm or not sm:
        continue
    p = [float(v) for v in pm.group(1).split()]
    s = [float(v) for v in sm.group(1).split()]
    if len(p) >= 6 and len(s) >= 3:
        boxes.append((m.group(1), p[0], p[1], p[5], s[0], s[1]))


def rect_pts(cx, cy, yaw, sx, sy, n=24):
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for u in np.linspace(-sx / 2, sx / 2, n):
        for v in [-sy / 2, sy / 2]:
            out.append((cx + u * c - v * s, cy + u * s + v * c))
    for v in np.linspace(-sy / 2, sy / 2, n):
        for u in [-sx / 2, sx / 2]:
            out.append((cx + u * c - v * s, cy + u * s + v * c))
    return np.array(out)


ALL = np.vstack([rect_pts(*b[1:]) for b in boxes])
NAMES = []
for b in boxes:
    NAMES += [b[0]] * (24 * 2 + 24 * 2)

traj = []
t0 = time.time()


def cb(m):
    if 'V550_mec' not in m.name:
        return
    i = m.name.index('V550_mec')
    p = m.pose[i].position
    traj.append((p.x, p.y))


rospy.init_node('traj_shape', anonymous=True)
rospy.Subscriber('/gazebo/model_states', ModelStates, cb)
while time.time() - t0 < DUR and not rospy.is_shutdown():
    time.sleep(0.2)
    if traj and math.hypot(traj[-1][0] - GX, traj[-1][1] - GY) < 0.3:
        break

if len(traj) < 20:
    print('%-16s 样本不足' % TAG)
    sys.exit()

tr = np.array(traj)
mov = [tr[0]]
for p in tr[1:]:
    if np.linalg.norm(p - mov[-1]) > 1e-4:
        mov.append(p)
tr = np.array(mov)

a, b = tr[0], tr[-1]
L = np.linalg.norm(b - a)
d = (b - a) / L
n = np.array([-d[1], d[0]])
dev = (tr - a) @ n
arc = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1)))

# 轨迹每点到所有障碍物的最近距离
dmat = np.linalg.norm(tr[:, None, :] - ALL[None, :, :], axis=2)
dmin_per_pt = dmat.min(axis=1)
k = int(np.argmin(dmin_per_pt))
j = int(np.argmin(dmat[k]))

# 起点->终点直线的净空(基线: 不绕的话会离障碍物多近)
dd = b - a
t = np.clip(((ALL - a) @ dd) / (dd @ dd), 0, 1)
proj = a + t[:, None] * dd
straight_clear = float(np.min(np.linalg.norm(ALL - proj, axis=1)))

print('%-16s 弧长比 %.4f | 横向偏离 %+.3f~%+.3f m | 轨迹最近障碍 %.4f m (%s)'
      % (TAG, arc / L, dev.min(), dev.max(), dmin_per_pt[k], NAMES[j]))
print('%-16s 直线净空 %.4f m -> %s'
      % ('', straight_clear,
         '障碍物确实在路上, 这是有效的避障测试' if straight_clear < 0.14
         else '直线就能过, 这次运行不算避障测试'))
