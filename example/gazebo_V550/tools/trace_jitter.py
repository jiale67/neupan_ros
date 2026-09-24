#!/usr/bin/env python3
"""
抖动归因: 同时记录 规划输出(/neupan_plan, /neupan_cmd_vel) 和 实际响应(/odom, gazebo真值)，
判断抖动来自规划层还是麦轮速度跟踪层。

判据:
  - cmd_vel 本身就在抖  -> 规划层问题
  - cmd_vel 平滑但 odom 抖 -> 速度跟踪问题
  - /neupan_plan 不直    -> 规划层问题(优化解本身弯)
"""
import rospy, math, time, sys
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates

D = dict(cmd=[], odom=[], plan=[], ref=[], truth=[])
t0 = None


def now():
    return time.time() - t0


def cb_cmd(m):
    D['cmd'].append((now(), m.linear.x, m.linear.y, m.angular.z))


def cb_odom(m):
    p, o = m.pose.pose.position, m.pose.pose.orientation
    yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y ** 2 + o.z ** 2))
    t = m.twist.twist
    D['odom'].append((now(), p.x, p.y, yaw, t.linear.x, t.linear.y, t.angular.z))


def cb_truth(m):
    if 'V550_mec' not in m.name:
        return
    i = m.name.index('V550_mec')
    p, o = m.pose[i].position, m.pose[i].orientation
    yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y ** 2 + o.z ** 2))
    v = m.twist[i]
    D['truth'].append((now(), p.x, p.y, yaw, v.linear.x, v.linear.y, v.angular.z))


def path_xy(m):
    return [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses]


def cb_plan(m):
    D['plan'].append((now(), path_xy(m)))


def cb_ref(m):
    D['ref'].append((now(), path_xy(m)))


def straightness(pts):
    """路径点对首末连线的最大垂直偏离 + 总弧长/直线长"""
    if len(pts) < 3:
        return 0.0, 1.0
    a, b = np.array(pts[0]), np.array(pts[-1])
    L = np.linalg.norm(b - a)
    if L < 1e-6:
        return 0.0, 1.0
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    dev = max(abs(float(np.dot(np.array(p) - a, n))) for p in pts)
    arc = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return dev, arc / L


def main():
    global t0
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    rospy.init_node('trace_jitter', anonymous=True)
    t0 = time.time()
    rospy.Subscriber('/neupan_cmd_vel', Twist, cb_cmd)
    rospy.Subscriber('/cmd_vel', Twist, cb_cmd)
    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.Subscriber('/gazebo/model_states', ModelStates, cb_truth)
    rospy.Subscriber('/neupan_plan', Path, cb_plan)
    rospy.Subscriber('/neupan_ref_state', Path, cb_ref)

    t = time.time()
    while time.time() - t < dur and not rospy.is_shutdown():
        time.sleep(0.2)
    report()


def report():
    print("=" * 72)
    for k in D:
        print("  %-6s %d samples" % (k, len(D[k])))
    print("=" * 72)

    # ---- 1. 规划路径的直线性 ----
    if D['plan']:
        rows = [(t, *straightness(p)) for t, p in D['plan'] if len(p) >= 3]
        if rows:
            dev = [r[1] for r in rows]
            rat = [r[2] for r in rows]
            print("\n[1] /neupan_plan 直线性 (%d 帧, 每帧 %d 点)" % (
                len(rows), len(D['plan'][0][1])))
            print("    对首末连线最大偏离: 均值 %.4f m, 最大 %.4f m" % (
                np.mean(dev), np.max(dev)))
            print("    弧长/直线长:        均值 %.4f, 最大 %.4f" % (
                np.mean(rat), np.max(rat)))

    # ---- 2. cmd_vel 平滑性 ----
    if len(D['cmd']) > 3:
        c = np.array(D['cmd'])
        t, vx, vy, wz = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        spd = np.hypot(vx, vy)
        ang = np.degrees(np.arctan2(vy, vx))
        dt = np.diff(t)
        dt[dt <= 0] = 1e-6
        print("\n[2] cmd_vel (车体系, 发布频率 %.1f Hz)" % (len(t) / max(t[-1] - t[0], 1e-6)))
        print("    |v|     范围 %.3f~%.3f  均值 %.3f" % (spd.min(), spd.max(), spd.mean()))
        print("    vx      范围 %+.3f~%+.3f" % (vx.min(), vx.max()))
        print("    vy      范围 %+.3f~%+.3f" % (vy.min(), vy.max()))
        print("    angular 范围 %+.3f~%+.3f" % (wz.min(), wz.max()))
        # 逐拍加速度
        ax = np.abs(np.diff(vx) / dt)
        ay = np.abs(np.diff(vy) / dt)
        print("    逐拍加速度 |dvx/dt| 中位 %.2f 最大 %.2f m/s^2" % (
            np.median(ax), ax.max()))
        print("    逐拍加速度 |dvy/dt| 中位 %.2f 最大 %.2f m/s^2" % (
            np.median(ay), ay.max()))
        # 方向角反转次数(抖动的直接指标)
        dang = np.diff(np.unwrap(np.radians(ang)))
        sign_flip = int(np.sum(np.diff(np.sign(dang)) != 0))
        print("    速度方向角 范围 %+.1f~%+.1f deg, 标准差 %.1f deg" % (
            ang.min(), ang.max(), ang.std()))
        print("    方向角变化率反号次数: %d  (越多越抖)" % sign_flip)

    # ---- 3. 跟踪误差: cmd_vel vs 实际 ----
    if len(D['cmd']) > 3 and len(D['truth']) > 3:
        c = np.array(D['cmd'])
        g = np.array(D['truth'])
        # 真值是世界系速度, 转到车体系再跟 cmd 比
        yaw = g[:, 3]
        vxb = g[:, 4] * np.cos(yaw) + g[:, 5] * np.sin(yaw)
        vyb = -g[:, 4] * np.sin(yaw) + g[:, 5] * np.cos(yaw)
        # 按时间插值对齐
        cx = np.interp(g[:, 0], c[:, 0], c[:, 1])
        cy = np.interp(g[:, 0], c[:, 0], c[:, 2])
        ex, ey = vxb - cx, vyb - cy
        print("\n[3] 速度跟踪误差 (gazebo真值(转车体系) - cmd_vel)")
        print("    vx: 平均偏差 %+.4f  RMS %.4f  最大 %.4f m/s" % (
            ex.mean(), np.sqrt((ex ** 2).mean()), np.abs(ex).max()))
        print("    vy: 平均偏差 %+.4f  RMS %.4f  最大 %.4f m/s" % (
            ey.mean(), np.sqrt((ey ** 2).mean()), np.abs(ey).max()))
        rel = np.sqrt((ex ** 2 + ey ** 2).mean()) / max(np.hypot(cx, cy).mean(), 1e-6)
        print("    相对 RMS 误差: %.1f%%" % (100 * rel))

    # ---- 4. 实际轨迹形状 ----
    for key, label in (('truth', 'gazebo真值'), ('odom', 'odom')):
        if len(D[key]) > 5:
            a = np.array(D[key])
            pts = list(zip(a[:, 1], a[:, 2]))
            # 只取真正在动的段
            mov = [p for i, p in enumerate(pts)
                   if i == 0 or math.dist(p, pts[i - 1]) > 1e-4]
            dev, rat = straightness(mov if len(mov) > 2 else pts)
            print("\n[4] 实际轨迹 (%s): 起(%+.3f,%+.3f) 末(%+.3f,%+.3f)" % (
                label, a[0, 1], a[0, 2], a[-1, 1], a[-1, 2]))
            print("    对首末连线最大横向偏离 %.4f m, 弧长/直线长 %.4f" % (dev, rat))
            print("    yaw %.2f ~ %.2f deg" % (
                math.degrees(a[:, 3].min()), math.degrees(a[:, 3].max())))


if __name__ == '__main__':
    main()
