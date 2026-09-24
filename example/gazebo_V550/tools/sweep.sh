#!/usr/bin/env bash
# 抖动归因扫描: 每个配置跑一次 (-3,-3)->(3,0)，打一行指标。
# 用法: bash sweep.sh cfg1.yaml cfg2.yaml ...
set -u
cd /home/rjl/nav_ws
source devel/setup.bash
TOOLS=src/neupan_ros/example/gazebo_V550/tools

for cfg in "$@"; do
  tag=$(basename "$cfg" .yaml)
  pkill -9 -f neupan_node >/dev/null 2>&1
  pkill -9 -f 'tools/metric.py' >/dev/null 2>&1
  sleep 2
  rosservice call /gazebo/set_model_state \
    '{model_state: {model_name: V550_mec, pose: {position: {x: -3.0, y: -3.0, z: 0.02027},
      orientation: {w: 1.0}}, reference_frame: world}}' >/dev/null 2>&1
  sleep 1
  setsid nohup roslaunch neupan_ros neupan_gazebo_V550.launch \
    config_file:="$cfg" > "/tmp/sw_$tag.log" 2>&1 < /dev/null & disown
  sleep 15
  setsid nohup python3 $TOOLS/metric.py "$tag" 28 > "/tmp/m_$tag.log" 2>&1 < /dev/null & disown
  sleep 2
  rostopic pub -1 /neupan_start std_msgs/Bool "data: true" >/dev/null 2>&1
  sleep 31
  cat "/tmp/m_$tag.log"
done

pkill -9 -f neupan_node >/dev/null 2>&1
exit 0
