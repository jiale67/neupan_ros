#!/usr/bin/env python3
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist

from neupan_ros.neupan_core import neupan_core


def main(args=None):
    # 不装 rclpy 自己的 SIGINT 处理: 它收到 Ctrl+C 会先把 context 关掉，之后
    # finally 里发零速度会因 "publisher's context is invalid" 失败，底盘就一直
    # 执行最后一条速度指令。这里让 Ctrl+C 走普通的 KeyboardInterrupt，context
    # 保持有效，先发零速度再关。
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    neupan_node = neupan_core()

    try:
        rclpy.spin(neupan_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 实车底盘会保持最后一条速度指令，退出前必须发一次零速度。
        # 连发几次，避免订阅端队列里还压着旧指令。
        try:
            for _ in range(3):
                neupan_node.vel_pub.publish(Twist())
                time.sleep(0.05)
        except Exception as e:
            print(f"[neupan_node] failed to publish zero velocity on exit: {e}")
        neupan_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
