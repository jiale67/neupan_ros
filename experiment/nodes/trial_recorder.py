#!/usr/bin/env python3
"""Records one navigation trial to disk (ch5 Gazebo experiment).

Waits for the first /move_base_simple/goal (t0 = sim time then), records the robot pose
stream (every /exp/model_states message, ~50 Hz, with gaps to ground-truth obstacles), every
/cmd_vel and every /exp/planner_status, the dynamic obstacles' poses at ~10 Hz, stops at
'arrived' | 'timeout' | 'stuck', writes
<out_dir>/<trial_id>{.csv,_cmd.csv,_status.csv,_obs.csv,.json} and publishes the outcome on
/exp/trial_done. All CSV times are sim seconds relative to t0 (sim time does not start at 0).
NaN is written as 'nan'.

Arrival: T = t_arrive = first sample with dist_goal <= arrive_threshold. The recording then
goes on until the robot has settled (planar speed < settle_v_eps for settle_hold s, at most
settle_max s after arrival); e_g is the goal distance at the end of that window (e_g_arrive at
t_arrive). Collisions keep being evaluated while settling. JSON also has settle_s, rtf (sim
seconds per wall second between t0 and the end) and wall_s (wall seconds t0 -> end), plus
pose_received / pose_dropped: /exp/model_states messages received while the trial was active
(t0 until the end) and those the worker skipped because it could not keep up (the receive
callback only queues, the worker always takes the newest message; see SkipAheadQueue);
pose_max_gap: largest gap between two recorded poses; pose_errors: messages whose processing
raised (logged, the worker carries on).

Private params: world_file, out_dir, trial_id, goal ("x y"), timeout, stuck_window (30),
stuck_dist (0.1, progress-based), arrive_threshold (0.3), meta (JSON string, copied to the json),
robot_name (V550_mec), obs_period (0.1), settle_hold (1.0), settle_max (5.0),
settle_v_eps (0.02).
"""
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from neupan_exp.geometry import BODY_RECT, NP_RECT, footprint_polygon  # noqa: E402
from neupan_exp.metrics import (POSE_MAX_BACKLOG, SkipAheadQueue, TrialSequencer,  # noqa: E402
                                compute_trial_metrics, guarded_call, max_gap, rtf_of,
                                window_progress)
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
        self.obs_period = float(rospy.get_param('~obs_period', 0.1))
        self.seq = TrialSequencer(
            self.arrive_threshold, self.timeout, self.stuck_window, self.stuck_dist,
            hold=float(rospy.get_param('~settle_hold', 1.0)),
            max_wait=float(rospy.get_param('~settle_max', 5.0)),
            v_eps=float(rospy.get_param('~settle_v_eps', 0.02)))
        meta = rospy.get_param('~meta', '{}')
        try:
            self.meta = json.loads(meta) if isinstance(meta, str) else meta
        except ValueError:
            self.meta = meta
        specs = parse_world(rospy.get_param('~world_file'))
        self.truth = ObstacleTruth(specs)
        self.dyn_models = {s.name.split('/')[0] for s in specs if s.dynamic}

        self.lock = threading.Lock()
        self.cond = threading.Condition()    # guards the receive queue
        self.queue = SkipAheadQueue(max_backlog=POSE_MAX_BACKLOG)
        self.n_in_trial = 0      # messages received / skipped while the trial is active
        self.n_dropped = 0
        self.n_errors = 0
        self.done = threading.Event()
        self.t0 = None
        self.wall0 = None
        self.sim_end = None   # sim time (s since t0) and wall seconds of the last pose sample
        self.wall_end = None
        self.next_obs_t = 0.0
        self.reason = None
        self.pose = []    # rows: t x y yaw gap_np gap_body nearest dist_goal
        self.pose_t = []  # t / dist_goal columns kept separately (stuck-window progress)
        self.pose_d = []
        self.obs = []     # rows: t name x y yaw (dynamic obstacles, ~obs_period)
        self.cmd = []     # t vx vy w
        self.status = []  # t plan_ms min_dist stop arrive

        self.pub_done = rospy.Publisher('/exp/trial_done', String, queue_size=1, latch=True)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.on_cmd, queue_size=100)
        rospy.Subscriber('/exp/planner_status', Float64MultiArray, self.on_status, queue_size=100)
        rospy.Subscriber('/exp/model_states', ModelStates, self.on_models, queue_size=20)
        threading.Thread(target=self._worker, daemon=True).start()
        rospy.loginfo('trial_recorder %s: goal=%s timeout=%.0f out_dir=%s',
                      self.trial_id, self.goal, self.timeout, self.out_dir)

    def _active(self):
        return self.t0 is not None and self.reason is None

    def on_goal(self, msg):
        with self.lock:
            if self.t0 is None:
                self.t0 = rospy.get_time()
                self.wall0 = time.time()
                self.next_obs_t = 0.0
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
        """Cheap receive callback (also keeps the receive timestamps of /cmd_vel and the status
        stream free of GIL stalls): only extracts the poses and queues them with the receive
        time. The gap computation runs in the worker, so a slow worker can no longer back up
        the subscription and silently time-shift the pose stream."""
        t_abs = rospy.get_time()
        dyn = []
        for n in self.dyn_models:
            if n in msg.name:
                p = msg.pose[msg.name.index(n)]
                dyn.append((n, p.position.x, p.position.y, yaw_of(p.orientation)))
        robot = None
        if self.robot_name in msg.name:
            p = msg.pose[msg.name.index(self.robot_name)]
            robot = (p.position.x, p.position.y, yaw_of(p.orientation))
        with self.cond:
            self.queue.push((t_abs, dyn, robot))
            self.cond.notify()

    def _worker(self):
        while not self.done.is_set() and not rospy.is_shutdown():
            with self.cond:
                before = self.queue.dropped
                item = self.queue.pop()
                if item is None:
                    self.cond.wait(0.2)
                    continue
                skipped = self.queue.dropped - before
            guarded_call(self._process, item + (skipped,), self._on_worker_error)

    def _on_worker_error(self, exc, tb):
        with self.lock:
            self.n_errors += 1
        rospy.logerr('trial %s: pose processing failed (%r); continuing with the next message\n%s',
                     self.trial_id, exc, tb)

    def _process(self, t_abs, dyn, robot, skipped=0):
        names = [d[0] for d in dyn]
        poses = [d[1:] for d in dyn]
        with self.lock:
            self.truth.update(names, poses, t_abs)
            if not self._active() or robot is None:
                return
            # in-trial statistics only (the ~30 s before the goal message would dilute them)
            self.n_in_trial += 1 + skipped
            self.n_dropped += skipped
            t = t_abs - self.t0
            if t >= self.next_obs_t - 1e-6:       # dynamic obstacles, time-decimated
                self.next_obs_t = max(self.next_obs_t + self.obs_period,
                                      t - 0.5 * self.obs_period)
                for n, q in zip(names, poses):
                    self.obs.append((t, n) + tuple(q))
            x, y, yaw = robot
            gap_np, near = self.truth.min_gap(footprint_polygon(x, y, yaw, *NP_RECT))
            gap_body, _ = self.truth.min_gap(footprint_polygon(x, y, yaw, *BODY_RECT))
            dist = math.hypot(x - self.goal[0], y - self.goal[1])
            self.pose.append((t, x, y, yaw, gap_np, gap_body, near, dist))
            self.pose_t.append(t)
            self.pose_d.append(dist)
            prog = window_progress(self.pose_t, self.pose_d, self.stuck_window)
            r = self.seq.update(t_abs, self.t0, t, x, y, dist, prog)
            if self.seq.arrived and self.seq.t_arrive == t:
                rospy.loginfo('trial %s: arrived at t=%.2f s (dist_goal %.3f), settling',
                              self.trial_id, t, dist)
            if r:
                self.reason = r
                self.sim_end, self.wall_end = t, time.time()
                rospy.loginfo('trial %s ended: %s at t=%.1f s (dist_goal %.3f)',
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
        with open(base + '_obs.csv', 'w') as f:
            f.write('t,name,x,y,yaw\n')
            for t, n, x, y, yaw in self.obs:
                f.write('%s,%s,%s,%s,%s\n' % (fmt(t, 4), n, fmt(x), fmt(y), fmt(yaw)))
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
        m = compute_trial_metrics(log, self.goal, self.reason, self.arrive_threshold,
                                  t_arrive=self.seq.t_arrive)
        wall_s = (self.wall_end - self.wall0) if self.wall0 is not None else float('nan')
        m.update(end_reason=self.reason, trial_id=self.trial_id, goal=list(self.goal),
                 meta=self.meta, t_arrive=self.seq.t_arrive, settle_s=self.seq.settle_s,
                 e_g_arrive=self.seq.e_g_arrive, wall_s=wall_s,
                 pose_received=self.n_in_trial, pose_dropped=self.n_dropped,
                 pose_max_gap=max_gap(self.pose_t), pose_errors=self.n_errors,
                 rtf=rtf_of(self.sim_end, wall_s))
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
