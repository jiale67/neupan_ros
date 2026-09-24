#!/usr/bin/env python3
"""按时间打印 速度/朝向/角速度, 定位 679deg 的 yaw 是在哪儿累积的。"""
import rospy, math, time, sys
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
GX = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
GY = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

st = {'yaw': None, 'x': None, 'y': None}
rows = []
t0 = time.time()


def cb_m(m):
    if 'V550_mec' not in m.name:
        return
    i = m.name.index('V550_mec')
    p, q = m.pose[i].position, m.pose[i].orientation
    st['x'], st['y'] = p.x, p.y
    st['yaw'] = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def cb_c(m):
    if st['yaw'] is None:
        return
    th = st['yaw']
    vx_w = m.linear.x * math.cos(th) - m.linear.y * math.sin(th)
    vy_w = m.linear.x * math.sin(th) + m.linear.y * math.cos(th)
    rows.append((time.time() - t0, vx_w, vy_w, m.angular.z, st['x'], st['y'], th))


rospy.init_node('yaw_trace', anonymous=True)
rospy.Subscriber('/gazebo/model_states', ModelStates, cb_m)
rospy.Subscriber('/cmd_vel', Twist, cb_c)

while time.time() - t0 < DUR and not rospy.is_shutdown():
    time.sleep(0.2)
    if st['x'] is not None and math.hypot(st['x'] - GX, st['y'] - GY) < 0.3:
        break

r = np.array(rows)
if len(r) < 20:
    print('样本不足', len(r))
    sys.exit()
t, vxw, vyw, wz, x, y, yaw = (r[:, i] for i in range(7))
spd = np.hypot(vxw, vyw)
vdir = np.degrees(np.arctan2(vyw, vxw))

print('%6s %7s %7s | %6s %8s %8s %8s | %s' % (
    't', 'x', 'y', '|v|', '速度方向', 'yaw', 'wz', '每0.5s转过'))
# 0.5s 分箱
nb = int(t[-1] / 0.5) + 1
uy = np.unwrap(yaw)
for b in range(nb):
    sel = np.where((t >= b * 0.5) & (t < (b + 1) * 0.5))[0]
    if len(sel) < 2:
        continue
    turned = math.degrees(uy[sel[-1]] - uy[sel[0]])
    print('%6.1f %+7.3f %+7.3f | %6.3f %+8.1f %+8.1f %+8.3f | %+7.1f %s' % (
        t[sel[0]], x[sel[0]], y[sel[0]], spd[sel].mean(),
        vdir[sel[-1]], math.degrees(yaw[sel[-1]]), wz[sel].mean(), turned,
        '<<< 原地转' if spd[sel].mean() < 0.03 and abs(wz[sel].mean()) > 0.1 else ''))

print()
print('速度 < omni_yaw_min_speed(0.02) 的帧: %.1f%%' % (100.0 * np.mean(spd < 0.02)))
print('速度 0.02~0.05 的帧: %.1f%%  (atan2 在这个区间已经很不稳)'
      % (100.0 * np.mean((spd >= 0.02) & (spd < 0.05))))
print('|v| 分布: p10 %.3f  中位 %.3f  p90 %.3f  max %.3f'
      % (np.percentile(spd, 10), np.median(spd), np.percentile(spd, 90), spd.max()))
print('|v| > ref_speed*1.5 (0.45) 的帧: %.1f%%' % (100.0 * np.mean(spd > 0.45)))
