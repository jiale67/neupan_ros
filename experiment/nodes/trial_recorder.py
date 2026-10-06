#!/usr/bin/env python3
"""Records one navigation trial to disk (ch5 Gazebo experiment).

Waits for the first /move_base_simple/goal (t0 = sim time then), records the robot pose
stream (~20 Hz, gaps to ground-truth obstacles), every /cmd_vel and every /exp/planner_status,
stops at 'arrived' | 'timeout' | 'stuck', writes <out_dir>/<trial_id>{.csv,_cmd.csv,_status.csv,.json}
and publishes the outcome on /exp/trial_done. All CSV times are sim seconds relative to t0
(sim time does not start at 0). NaN is written as 'nan'.

Private params: world_file, out_dir, trial_id, goal ("x y"), timeout, stuck_window (30),
stuck_dist (0.1, progress-based), arrive_threshold (0.3), meta (JSON string, copied to the json),
robot_name (V550_mec), record_period (0.05).
"""
import json
import math
import os
import sys
import threading

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from neupan_exp.geometry import BODY_RECT, NP_RECT, footprint_polygon  # noqa: E402
from neupan_exp.metrics import compute_trial_metrics, end_reason, window_progress  # noqa: E402
from neupan_exp.truth import ObstacleTruth  # noqa: E402
from neupan_exp.world import parse_world  # noqa: E402


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fmt(v, nd=6):
    v = float(v)
    return 'nan' if math.isnan(v) else '%.*f' % (nd, v)


class TrialRecorder:
    def __init__(self):
        self.out_dir = rospy.get_param('~out_dir')
        self.trial_id = rospy.get_param('~trial_id')
        self.goal = tuple(float(v) for v in str(rospy.get_param('~goal')).split())
        self.timeout = float(rospy.get_param('~timeout'))
        self.stuck_window = float(rospy.get_param('~stuck_window', 30.0))
        self.stuck_dist = float(rospy.get_param('~stuck_dist', 0.1))
        self.arrive_threshold = float(rospy.get_param('~arrive_threshold', 0.3))
        self.robot_name = rospy.get_param('~robot_name', 'V550_mec')
        self.period = float(rospy.get_param('~record_period', 0.05))
        meta = rospy.get_param('~meta', '{}')
        try:
            self.meta = json.loads(meta) if isinstance(meta, str) else meta
        except ValueError:
            self.meta = meta
        specs = parse_world(rospy.get_param('~world_file'))
        self.truth = ObstacleTruth(specs)
        self.dyn_models = {s.name.split('/')[0] for s in specs if s.dynamic}

        self.lock = threading.Lock()
        self.done = threading.Event()
        self.t0 = None
        self.next_t = 0.0
        self.reason = None
        self.pose = []    # rows: t x y yaw gap_np gap_body nearest dist_goal
        self.cmd = []     # t vx vy w
        self.status = []  # t plan_ms min_dist stop arrive

        self.pub_done = rospy.Publisher('/exp/trial_done', String, queue_size=1, latch=True)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.on_cmd, queue_size=100)
        rospy.Subscriber('/exp/planner_status', Float64MultiArray, self.on_status, queue_size=100)
        rospy.Subscriber('/exp/model_states', ModelStates, self.on_models, queue_size=5)
        rospy.loginfo('trial_recorder %s: goal=%s timeout=%.0f out_dir=%s',
                      self.trial_id, self.goal, self.timeout, self.out_dir)

    def _active(self):
        return self.t0 is not None and self.reason is None

    def on_goal(self, msg):
        with self.lock:
            if self.t0 is None:
                self.t0 = rospy.get_time()
                self.next_t = 0.0
                rospy.loginfo('trial %s started, t0=%.3f', self.trial_id, self.t0)

    def on_cmd(self, msg):
        with self.lock:
            if self._active():
                self.cmd.append((rospy.get_time() - self.t0, msg.linear.x, msg.linear.y,
                                 msg.angular.z))

    def on_status(self, msg):
        d = list(msg.data) + [float('nan')] * 4
        with self.lock:
            if self._active():
                self.status.append((rospy.get_time() - self.t0, d[0], d[1], d[2], d[3]))

    def on_models(self, msg):
        t_abs = rospy.get_time()
        names = [n for n in msg.name if n in self.dyn_models]
        poses = []
        for n in names:
            p = msg.pose[msg.name.index(n)]
            poses.append((p.position.x, p.position.y, yaw_of(p.orientation)))
        with self.lock:
            self.truth.update(names, poses, t_abs)
            if not self._active() or self.robot_name not in msg.name:
                return
            t = t_abs - self.t0
            if t < self.next_t - 1e-6:
                return
            self.next_t = max(self.next_t + self.period, t - 0.5 * self.period)
            p = msg.pose[msg.name.index(self.robot_name)]
            x, y, yaw = p.position.x, p.position.y, yaw_of(p.orientation)
            gap_np, near = self.truth.min_gap(footprint_polygon(x, y, yaw, *NP_RECT))
            gap_body, _ = self.truth.min_gap(footprint_polygon(x, y, yaw, *BODY_RECT))
            dist = math.hypot(x - self.goal[0], y - self.goal[1])
            self.pose.append((t, x, y, yaw, gap_np, gap_body, near, dist))
            ts = [r[0] for r in self.pose]
            ds = [r[7] for r in self.pose]
            prog = window_progress(ts, ds, self.stuck_window)
            r = end_reason(t_abs, self.t0, dist, prog, self.timeout, self.stuck_window,
                           self.stuck_dist, self.arrive_threshold)
            if r:
                self.reason = r
                rospy.loginfo('trial %s ended: %s at t=%.1f s (dist_goal %.2f)',
                              self.trial_id, r, t, dist)
                self.done.set()

    def write(self):
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, self.trial_id)
        with open(base + '.csv', 'w') as f:
            f.write('t,x,y,yaw,gap_np,gap_body,nearest,dist_goal\n')
            for t, x, y, yaw, gn, gb, near, d in self.pose:
                f.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (fmt(t, 4), fmt(x), fmt(y), fmt(yaw),
                                                       fmt(gn), fmt(gb), near, fmt(d)))
        with open(base + '_cmd.csv', 'w') as f:
            f.write('t,vx,vy,w\n')
            for r in self.cmd:
                f.write('%s,%s,%s,%s\n' % (fmt(r[0], 4), fmt(r[1]), fmt(r[2]), fmt(r[3])))
        with open(base + '_status.csv', 'w') as f:
            f.write('t,plan_ms,min_dist,stop,arrive\n')
            for r in self.status:
                f.write('%s,%s,%s,%d,%d\n' % (fmt(r[0], 4), fmt(r[1], 3), fmt(r[2]),
                                              int(r[3] > 0.5), int(r[4] > 0.5)))
        col = lambda a, i: [r[i] for r in a]
        log = {'t': col(self.pose, 0), 'x': col(self.pose, 1), 'y': col(self.pose, 2),
               'yaw': col(self.pose, 3), 'gap_np': col(self.pose, 4), 'gap_body': col(self.pose, 5),
               'cmd_t': col(self.cmd, 0), 'cmd_vx': col(self.cmd, 1), 'cmd_vy': col(self.cmd, 2),
               'cmd_w': col(self.cmd, 3), 'st_t': col(self.status, 0),
               'plan_ms': col(self.status, 1), 'stop': [r[3] > 0.5 for r in self.status]}
        m = compute_trial_metrics(log, self.goal, self.reason, self.arrive_threshold)
        m.update(end_reason=self.reason, trial_id=self.trial_id, goal=list(self.goal),
                 meta=self.meta)
        with open(base + '.json', 'w') as f:
            json.dump(m, f, indent=1)
        return m['outcome']

    def spin(self):
        while not self.done.wait(0.2):
            if rospy.is_shutdown():
                return
        with self.lock:  # callbacks now ignore data; take the lock once for a consistent snapshot
            outcome = self.write()
        rospy.loginfo('trial %s outcome: %s', self.trial_id, outcome)
        self.pub_done.publish(String(data=outcome))
        rospy.sleep(1.0)
        rospy.signal_shutdown('trial done')


if __name__ == '__main__':
    rospy.init_node('trial_recorder')
    TrialRecorder().spin()
