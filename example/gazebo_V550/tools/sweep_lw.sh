#!/usr/bin/env bash
# lateral_weight 扫描。用法: bash sweep_lw.sh <yaw_mode> <cfg...>
set -u
cd /home/rjl/nav_ws
source devel/setup.bash
TOOLS=src/neupan_ros/example/gazebo_V550/tools
YAW="$1"; shift
GX=6.0; GY=3.5

for cfg in "$@"; do
  tag="$(basename "$cfg" .yaml)_$YAW"
  pkill -9 -f neupan_node >/dev/null 2>&1 || true
  pkill -9 -f obs_metric.py >/dev/null 2>&1 || true
  sleep 3
  rosservice call /gazebo/set_model_state \
    '{model_state: {model_name: V550_mec, pose: {position: {x: -3.0, y: -3.0, z: 0.02027},
      orientation: {w: 1.0}}, reference_frame: world}}' >/dev/null 2>&1
  sleep 1
  setsid nohup roslaunch neupan_ros neupan_gazebo_V550.launch \
    config_file:="$cfg" goal:="$GX $GY" omni_yaw_mode:="$YAW" \
    > "/tmp/lw_$tag.log" 2>&1 < /dev/null & disown
  sleep 19
  setsid nohup python3 $TOOLS/obs_metric.py "$tag" 75 $GX $GY \
    > "/tmp/m_$tag.log" 2>&1 < /dev/null & disown
  sleep 3
  rostopic pub -1 /neupan_start std_msgs/Bool "data: true" >/dev/null 2>&1
  sleep 78
  sed -n '2,10p' "/tmp/m_$tag.log"
  echo
done

pkill -9 -f neupan_node >/dev/null 2>&1 || true
exit 0
