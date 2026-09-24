#!/usr/bin/env bash
# 轨迹形状 + 实际净空。用法: bash sweep_shape.sh <yaw_mode> <w1> <w2> ...
set -u
cd /home/rjl/nav_ws
source devel/setup.bash
TOOLS=src/neupan_ros/example/gazebo_V550/tools
YAW="$1"; shift
GX=6.0; GY=3.5

for w in "$@"; do
  pkill -9 -f neupan_node >/dev/null 2>&1 || true
  pkill -9 -f traj_shape >/dev/null 2>&1 || true
  sleep 3
  rosservice call /gazebo/set_model_state \
    '{model_state: {model_name: V550_mec, pose: {position: {x: -3.0, y: -3.0, z: 0.02027},
      orientation: {w: 1.0}}, reference_frame: world}}' >/dev/null 2>&1
  sleep 1
  setsid nohup roslaunch neupan_ros neupan_gazebo_V550.launch \
    config_file:="/tmp/lw/w$w.yaml" goal:="$GX $GY" omni_yaw_mode:="$YAW" \
    > "/tmp/sh_$w.log" 2>&1 < /dev/null & disown
  sleep 19
  setsid nohup python3 $TOOLS/traj_shape.py "lw=$w,$YAW" 70 $GX $GY \
    > "/tmp/shm_$w.log" 2>&1 < /dev/null & disown
  sleep 3
  rostopic pub -1 /neupan_start std_msgs/Bool "data: true" >/dev/null 2>&1
  sleep 73
  cat "/tmp/shm_$w.log"
done

pkill -9 -f neupan_node >/dev/null 2>&1 || true
exit 0
