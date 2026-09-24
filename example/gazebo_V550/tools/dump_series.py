#!/usr/bin/env python3
"""把 cmd_vel 的速度方向角时间序列打出来，看抖动是振荡还是漂移。"""
import rospy, math, time, sys
import numpy as np
from geometry_msgs.msg import Twist

R = []
t0 = time.time()
rospy.init_node('dump_series', anonymous=True)
rospy.Subscriber('/cmd_vel', Twist,
                 lambda m: R.append((time.time() - t0, m.linear.x, m.linear.y)))
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
while time.time() - t0 < dur and not rospy.is_shutdown():
    time.sleep(0.2)

a = np.array([r for r in R if math.hypot(r[1], r[2]) > 1e-6])
if len(a) < 10:
    print("样本不足 %d" % len(a)); sys.exit()
t, vx, vy = a[:, 0], a[:, 1], a[:, 2]
ang = np.degrees(np.arctan2(vy, vx))
spd = np.hypot(vx, vy)
print("样本 %d, 时长 %.1fs" % (len(a), t[-1] - t[0]))
print("\n每 0.5s 平均: t  |v|   dir(deg)")
for k in range(0, int(t[-1] - t[0]) * 2):
    lo, hi = t[0] + k * 0.5, t[0] + (k + 1) * 0.5
    m = (t >= lo) & (t < hi)
    if m.sum():
        print("  %5.1f  %.3f  %+7.1f   (min %+6.1f max %+6.1f)" % (
            lo - t[0], spd[m].mean(), ang[m].mean(), ang[m].min(), ang[m].max()))

# 频谱: 找主振荡频率
w = ang - np.mean(ang)
dt = np.median(np.diff(t))
F = np.fft.rfft(w * np.hanning(len(w)))
fr = np.fft.rfftfreq(len(w), dt)
P = np.abs(F)
k = np.argsort(P[1:])[::-1][:5] + 1
print("\n方向角主振荡频率 (采样 %.1f Hz):" % (1 / dt))
for i in k:
    print("    %.2f Hz  周期 %.2f s  幅值 %.1f" % (fr[i], 1 / max(fr[i], 1e-9), P[i]))
