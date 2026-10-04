#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist

from neupan_ros.neupan_core import neupan_core


def main(args=None):
    rclpy.init(args=args)
    neupan_node = neupan_core()

    try:
        rclpy.spin(neupan_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 实车底盘会保持最后一条速度指令，退出前必须发一次零速度。
        try:
            neupan_node.vel_pub.publish(Twist())
            time.sleep(0.1)
        except Exception:
            pass
        neupan_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
