#!/usr/bin/env python3

"""
实时对比 /cmd_vel 下发速度与 /odom 实际轮速度（Vx、Vy、Wz，均为车体系）。

    ros2 run neupan_ros vel_monitor [--window 20] [--cmd-topic /cmd_vel] [--odom-topic /odom]

时间轴为本节点收到消息的时刻（cmd_vel 无时间戳），odom 曲线因此包含 EKF 输出延迟。
"""

import argparse
import signal
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class VelRecorder(Node):

    def __init__(self, cmd_topic, odom_topic, window):

        super().__init__("vel_monitor")

        self.t0 = time.monotonic()
        self.window = window
        self.lock = threading.Lock()
        # (t, vx, vy, wz)
        self.cmd = deque()
        self.odom = deque()

        self.create_subscription(Twist, cmd_topic, self.cmd_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

    def now(self):

        return time.monotonic() - self.t0

    def append(self, buf, sample):

        with self.lock:
            buf.append(sample)
            # 多留一条窗口外的样本，保证窗口左端的 cmd 阶梯线完整
            while len(buf) > 1 and buf[1][0] < sample[0] - self.window:
                buf.popleft()

    def cmd_callback(self, msg):

        self.append(self.cmd, (self.now(), msg.linear.x, msg.linear.y, msg.angular.z))

    def odom_callback(self, msg):

        # EKF 的 twist 在 child_frame（base_footprint）下，与 cmd_vel 同为车体系
        t = msg.twist.twist
        self.append(self.odom, (self.now(), t.linear.x, t.linear.y, t.angular.z))

    def snapshot(self):

        with self.lock:
            return list(self.cmd), list(self.odom)


def main(args=None):

    # Tk 主循环里 KeyboardInterrupt 会被吞掉，改为 Ctrl+C 关闭窗口
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, lambda *_: plt.close("all"))

    parser = argparse.ArgumentParser(description="cmd_vel vs odom velocity monitor")
    parser.add_argument("--window", type=float, default=20.0, help="显示时间窗口 s")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom_combined")
    opts = parser.parse_args(remove_ros_args(args=None)[1:])

    node = VelRecorder(opts.cmd_topic, opts.odom_topic, opts.window)

    def spin():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.canvas.manager.set_window_title("cmd_vel vs odom")
    labels = ("Vx (m/s)", "Vy (m/s)", "Wz (rad/s)")
    cmd_lines, odom_lines = [], []

    for ax, label in zip(axes, labels):
        # 底盘保持最后一条指令，cmd 画成阶梯线
        cl, = ax.plot([], [], color="red", drawstyle="steps-post", label=opts.cmd_topic)
        ol, = ax.plot([], [], color="green", label=opts.odom_topic)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        cmd_lines.append(cl)
        odom_lines.append(ol)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()

    def update(_):

        cmd, odom = node.snapshot()
        now = node.now()

        if cmd:
            # 最后一条指令延伸到当前时刻
            cmd = cmd + [(now,) + cmd[-1][1:]]

        for i in range(3):
            cmd_lines[i].set_data([s[0] for s in cmd], [s[i + 1] for s in cmd])
            odom_lines[i].set_data([s[0] for s in odom], [s[i + 1] for s in odom])

        t_min = max(0.0, now - opts.window)
        axes[-1].set_xlim(t_min, max(now, opts.window))

        for i, ax in enumerate(axes):
            ys = [s[i + 1] for s in cmd + odom if s[0] >= t_min]
            lo, hi = (min(ys), max(ys)) if ys else (0.0, 0.0)
            pad = max(0.05, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)

        return cmd_lines + odom_lines

    # 持有引用，防止动画被回收
    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    plt.show()

    del anim
    # 先 shutdown 唤醒 spin 线程，等它退出后再销毁节点
    rclpy.try_shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()


if __name__ == "__main__":
    main()
