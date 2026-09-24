#!/usr/bin/env python3
"""单次运行的抖动指标，打一行汇总。配合 sweep.sh 用。"""
import rospy, math, time, sys
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates

TAG = sys.argv[1] if len(sys.argv) > 1 else 'run'
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 28.0
GOAL = (3.0, 0.0)

C, T, P = [], [], []
t0 = time.time()


def straight(pts):
    if len(pts) < 3:
        return 0.0, 1.0
    a, b = np.array(pts[0], float), np.array(pts[-1], float)
    L = np.linalg.norm(b - a)
    if L < 1e-6:
        return 0.0, 1.0
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    dev = max(abs(float(np.dot(np.array(p, float) - a, n))) for p in pts)
    arc = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return dev, arc / L


rospy.init_node('metric', anonymous=True)
rospy.Subscriber('/cmd_vel', Twist,
                 lambda m: C.append((time.time() - t0, m.linear.x, m.linear.y, m.angular.z)))
rospy.Subscriber('/neupan_plan', Path,
                 lambda m: P.append([(q.pose.position.x, q.pose.position.y) for q in m.poses]))


def cb(m):
    if 'V550_mec' in m.name:
        i = m.name.index('V550_mec')
        p = m.pose[i].position
        T.append((time.time() - t0, p.x, p.y))


rospy.Subscriber('/gazebo/model_states', ModelStates, cb)

while time.time() - t0 < DUR and not rospy.is_shutdown():
    time.sleep(0.2)
    if T and math.hypot(T[-1][1] - GOAL[0], T[-1][2] - GOAL[1]) < 0.25:
        break

c = np.array([r for r in C if math.hypot(r[1], r[2]) > 1e-6])
if len(c) < 20 or len(T) < 20:
    print("%-22s 样本不足 cmd=%d truth=%d" % (TAG, len(c), len(T)))
    sys.exit()

t, vx, vy = c[:, 0], c[:, 1], c[:, 2]
ang = np.degrees(np.arctan2(vy, vx))
spd = np.hypot(vx, vy)
d = np.diff(ang)
flips = int(np.sum(np.diff(np.sign(d)) != 0)) / max(t[-1] - t[0], 1e-6)

pdev = np.mean([straight(p)[0] for p in P if len(p) >= 3]) if P else float('nan')

a = np.array(T)
pts = [(x, y) for x, y in zip(a[:, 1], a[:, 2])]
mov = [p for i, p in enumerate(pts) if i == 0 or math.dist(p, pts[i - 1]) > 1e-4]
tdev, trat = straight(mov if len(mov) > 2 else pts)
err = math.hypot(a[-1, 1] - GOAL[0], a[-1, 2] - GOAL[1])

print("%-22s dir: %+6.1f~%+6.1f std %4.1f deg | 反号 %5.1f /s | "
      "plan偏离 %.4f | 轨迹偏离 %.4f 弧长比 %.4f | |v| %.3f | 末点误差 %.3f %s"
      % (TAG, ang.min(), ang.max(), ang.std(), flips, pdev,
         tdev, trat, spd.mean(), err, "到达" if err < 0.3 else ""))
