#!/usr/bin/env python3
"""Resumable batch runner of the ch5 Gazebo experiment.

Runs the trial matrix (scenes x start/goal pairs x repeats x methods) one fresh headless Gazebo
per trial: `roslaunch neupan_ros exp_trial.launch ...` -> wait for /clock and for the planner to
publish /cmd_vel -> wait until the trial's (relative) start_sim_time -> publish the goal on
/move_base_simple/goal -> wait for the recorder's <trial_id>.json -> stop that launch (and only
that launch's process tree).

Layout of the output directory (default /home/rjl/nav_ws/exp_data/<run_name>):
    manifest.json                     reproducibility record, updated after every trial
    trials/<trial_id>{.json,.csv,_cmd.csv,_status.csv}
    logs/<trial_id>.launch.log        roslaunch stdout+stderr (all attempts, appended)
    logs/roscore.log                  only if this runner started the roscore

Resuming: run again with the same --out; trials with a non-crash json are skipped, crash trials
are retried until 1+--retries attempts have been used (the attempt count is stored in the crash
json). A trial is never silently dropped: if neither the recorder nor a clock/planner start
works, a json with outcome 'crash' (+ reason) is written.

The runner itself uses WALL time: it starts its own roscore (unless one is reachable), removes a
stale /use_sim_time before init_node and subscribes /clock manually. It never kills gzserver
globally: every launch is its own process group (+ an environment token to find re-parented
children); if a gzserver it did not start is running, the batch aborts.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, EXP_DIR)

from neupan_exp.metrics import compute_trial_metrics  # noqa: E402
from neupan_exp.scenarios import load_scenarios, trial_matrix  # noqa: E402

WS = '/home/rjl/nav_ws'
REPOS = {
    'neupan_ros': WS + '/src/neupan_ros',
    'Autonomous_navigation_exploration': WS + '/src/Autonomous_navigation_exploration',
    'NeuPAN': WS + '/src/NeuPAN',
}
V550 = WS + '/src/neupan_ros/example/gazebo_V550/'
HASHED_FILES = [
    V550 + 'world/static_scene.world',
    V550 + 'world/dynamic_scene.world',
    V550 + 'launch/gazebo_V550.launch',
    V550 + 'launch/neupan_V550.launch',
    V550 + 'config/neupan_planner_V550_mec.yaml',
    V550 + 'pretrain_V550_mec/model_5000.pth',
    WS + '/src/Autonomous_navigation_exploration/dwa_local_planner/launch/omni_dwa_V550.launch',
    EXP_DIR + '/config/scenarios.yaml',
]
TOKEN_VAR = 'EXP_RUNNER_TOKEN'
STOP_SIGINT_WAIT = 20.0
INTERRUPT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


class AbortBatch(RuntimeError):
    """Fatal condition for the whole batch (e.g. foreign gzserver) - not a trial crash."""


# ----------------------------------------------------------------------------- generic helpers
def wall_timeout_for(trial):
    return trial.timeout * 1.5 + 90.0


def goal_arg(trial):
    return '%r %r' % (float(trial.goal[0]), float(trial.goal[1]))


def trial_meta(trial, attempt):
    return dict(trial_id=trial.trial_id, method=trial.method, scene=trial.scene,
                world=trial.world, pair_id=trial.pair_id, seed=trial.seed,
                start=list(trial.start), goal=list(trial.goal),
                start_sim_time=trial.start_sim_time, timeout=trial.timeout, attempt=attempt)


def _wait_until(pred, timeout, poll=0.1, alive=None):
    """Poll pred() until truthy -> 'ok'; 'timeout' after `timeout` wall seconds; 'exited' when
    alive() turns False."""
    end = time.monotonic() + timeout
    while True:
        if pred():
            return 'ok'
        if alive is not None and not alive():
            return 'exited'
        if time.monotonic() >= end:
            return 'timeout'
        time.sleep(poll)


def wait_for_sim_time(clock_source, target, wall_timeout, poll=0.05, alive=None):
    """True once clock_source() (latest /clock value or None) >= target within wall_timeout s."""
    def reached():
        v = clock_source()
        return v is not None and v >= target
    return _wait_until(reached, wall_timeout, poll, alive) == 'ok'


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _result_paths(out_dir, trial_id):
    b = os.path.join(out_dir, trial_id)
    return b + '.json', [b + s for s in ('.csv', '_cmd.csv', '_status.csv', '.json', '.json.tmp')]


def crash_json(trial, reason, meta, attempt):
    empty = {k: [] for k in ('t', 'x', 'y', 'yaw', 'gap_np', 'gap_body', 'cmd_t', 'cmd_vx',
                             'cmd_vy', 'cmd_w', 'st_t', 'plan_ms', 'stop')}
    m = compute_trial_metrics(empty, trial.goal, 'crash')
    m.update(end_reason='crash', trial_id=trial.trial_id, goal=list(trial.goal), meta=meta,
             reason=reason, attempts=attempt)
    return m


# ------------------------------------------------------------------------------------ run_one
def run_one(trial, out_dir, launcher, clock_source, goal_pub, wall_timeout=None, ready=None,
            ready_timeout=120.0, clock_timeout=120.0, reset=None, poll=0.1, log_path=None,
            attempt=1, goal_gap=0.2, exit_grace=10.0, precheck=None, goal_subscribers=None):
    """Run one trial; return its outcome ('success', 'collision', 'timeout', 'stuck', 'crash').

    launcher(trial, out_dir, meta_json, log_path) -> handle with poll() (None while running)
    and stop(); clock_source() -> latest /clock seconds or None; goal_pub(x, y) publishes one
    goal (frame odom); ready() -> True once the planner published /cmd_vel since the launch;
    reset() clears clock/ready state right before the launch; precheck(trial) runs before any
    old result is deleted (may raise AbortBatch); goal_subscribers() -> number of subscribers of
    the goal topic (stored in meta). wall_timeout is the time allowed after the goal (default
    timeout*1.5+90). Sim times are relative to the first /clock value of this launch (sim time
    does not start at 0). Any exception except AbortBatch/KeyboardInterrupt is turned into a
    crash json (reason=repr(e)) after the launch has been stopped.
    """
    if wall_timeout is None:
        wall_timeout = wall_timeout_for(trial)
    if precheck:
        precheck(trial)                  # before deleting anything: an abort keeps the old json
    json_path, stale = _result_paths(out_dir, trial.trial_id)
    os.makedirs(out_dir, exist_ok=True)
    for p in stale:                      # an old result must never pass for this attempt
        if os.path.exists(p):
            os.remove(p)
    meta = trial_meta(trial, attempt)
    t_wall0 = time.time()
    handle = None
    reason = None
    try:
        if reset:
            reset()
        handle = launcher(trial, out_dir, json.dumps(meta), log_path)
        alive = lambda: handle.poll() is None  # noqa: E731
        # 1. /clock of this launch (first value = sim-time origin of the trial)
        if _wait_until(lambda: clock_source() is not None, clock_timeout, poll, alive) != 'ok':
            reason = ('launch exited (rc=%s) before /clock appeared' % handle.poll()
                      if handle.poll() is not None
                      else 'no /clock within %.0f s of launch' % clock_timeout)
        else:
            first_clock = clock_source()
            meta['first_clock'] = first_clock
            # 2. planner readiness = first /cmd_vel message after launch
            rdy = ready if ready is not None else (lambda: True)
            if _wait_until(rdy, ready_timeout, poll, alive) != 'ok':
                reason = ('launch exited (rc=%s) before planner was ready' % handle.poll()
                          if handle.poll() is not None
                          else 'planner not ready (no /cmd_vel) within %.0f s' % ready_timeout)
            else:
                target = first_clock + trial.start_sim_time
                now = clock_source()
                meta['late_start'] = bool(now is not None and now > target)
                # 3. relative start time (immediately, if readiness came after it)
                if not wait_for_sim_time(clock_source, target, 3.0 * trial.start_sim_time + 60.0,
                                         poll, alive):
                    reason = ('launch exited (rc=%s) while waiting for start_sim_time'
                              % handle.poll() if handle.poll() is not None
                              else 'sim time did not reach start_sim_time %.1f' % target)
        if reason is None:
            # 4. goal derived from trial.goal, 3 x at 0.2 s
            meta['actual_start_offset'] = clock_source() - first_clock
            meta['wall_goal_time'] = time.time()
            for i in range(3):
                goal_pub(float(trial.goal[0]), float(trial.goal[1]))
                if i < 2:
                    time.sleep(goal_gap)
            if goal_subscribers is not None:
                meta['goal_subscribers'] = int(goal_subscribers())
            # 5. wait for the recorder's json (or exit / wall timeout)
            res = []

            def have_result():
                r = load_result(json_path)
                if r is not None:
                    res.append(r)
                return r is not None

            st = _wait_until(have_result, wall_timeout, poll, alive)
            if st == 'exited' and not have_result():
                reason = 'launch exited (rc=%s) without result json' % handle.poll()
            elif st == 'timeout':
                reason = 'wall timeout: no result json %.0f s after goal' % wall_timeout
            else:
                r = res[-1]
                r.setdefault('meta', {})
                if not isinstance(r['meta'], dict):
                    r['meta'] = {'recorder_meta': r['meta']}
                r['meta'].update({k: meta[k] for k in
                                  ('late_start', 'actual_start_offset', 'first_clock',
                                   'wall_goal_time', 'attempt', 'goal_subscribers')
                                  if k in meta})
                r['attempts'] = attempt
                r['wall_s'] = time.time() - t_wall0
                _atomic_json(json_path, r)
                if exit_grace > 0:       # the launch ends by itself once the recorder is done
                    _wait_until(lambda: handle.poll() is not None, exit_grace, poll)
                return r.get('outcome', 'unknown')
    except (AbortBatch, KeyboardInterrupt):
        raise
    except Exception as e:  # noqa: BLE001 - one bad trial must not halt the batch
        sys.stderr.write('trial %s raised:\n%s' % (trial.trial_id, traceback.format_exc()))
        reason = repr(e)
    finally:
        if handle is not None:
            handle.stop()
    # no recorder json: record the trial as a crash with the reason
    if meta.get('goal_subscribers', 2) < 2:
        meta['crash_detail'] = reason
        reason = 'goal subscribers < 2'
    meta['wall_s'] = time.time() - t_wall0
    _atomic_json(json_path, crash_json(trial, reason, meta, attempt))
    return 'crash'


def run_trial(trial, out_dir, launcher, clock_source, goal_pub, retries=1, **kw):
    """Skip trials that already have a complete non-crash json (meta.attempt set by the runner); retry crashes up to 1+retries attempts."""
    json_path = _result_paths(out_dir, trial.trial_id)[0]
    existing = load_result(json_path)
    attempts = 0
    outcome = None
    if existing is not None:
        outcome = existing.get('outcome')
        meta = existing.get('meta')
        if outcome == 'crash':
            attempts = int(existing.get('attempts', 1))
        elif isinstance(meta, dict) and 'attempt' in meta:
            return outcome          # complete: recorder json patched by this runner
        # else: recorder json the runner never finished (killed before patching) -> run again
    while attempts < 1 + retries:
        attempts += 1
        outcome = run_one(trial, out_dir, launcher, clock_source, goal_pub, attempt=attempts, **kw)
        if outcome != 'crash':
            break
    return outcome


# ------------------------------------------------------------------- real processes (no tests)
def _proc_pids():
    for d in os.listdir('/proc'):
        if d.isdigit():
            yield int(d)


def _proc_comm(pid):
    try:
        with open('/proc/%d/comm' % pid) as f:
            return f.read().strip()
    except OSError:
        return None


def _proc_alive(pid):
    try:
        with open('/proc/%d/stat' % pid) as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def pids_with_token(token):
    """All live processes of this runner's launch (environment carries the token)."""
    key = ('%s=%s' % (TOKEN_VAR, token)).encode()
    out = []
    for pid in _proc_pids():
        try:
            with open('/proc/%d/environ' % pid, 'rb') as f:
                if key in f.read().split(b'\0') and _proc_alive(pid):
                    out.append(pid)
        except OSError:
            continue
    return out


def is_gzserver(comm):
    return bool(comm) and comm.startswith('gzserver')


def running_gzservers():
    return [p for p in _proc_pids() if is_gzserver(_proc_comm(p)) and _proc_alive(p)]


class Proc:
    """A subprocess in its own session/process group, tagged with a unique env token."""

    def __init__(self, argv, log_path):
        self.token = uuid.uuid4().hex
        env = dict(os.environ)
        env[TOKEN_VAR] = self.token
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, 'ab')
        self.log.write(('\n=== %s %s\n' % (datetime.datetime.now().isoformat(), ' '.join(argv)))
                       .encode())
        self.log.flush()
        self.popen = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                      start_new_session=True)

    def poll(self):
        return self.popen.poll()

    def stop(self, sigint_wait=STOP_SIGINT_WAIT):
        """SIGINT the process group, wait, SIGKILL; then SIGKILL anything left that carries our
        token (roslaunch nodes run in their own sessions); verify no gzserver of ours remains."""
        pgid = self.popen.pid
        if self.popen.poll() is None:
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.popen.wait(sigint_wait)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.popen.wait()
        for _ in range(3):
            left = pids_with_token(self.token)
            if not left:
                break
            for p in left:
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(1.0)
        left = pids_with_token(self.token)
        if left:
            names = [(p, _proc_comm(p)) for p in left]
            if any(is_gzserver(n) for _, n in names):
                raise AbortBatch('gzserver of this runner survived SIGKILL: %s' % names)
            sys.stderr.write('warning: processes survived stop: %s\n' % names)
        self.log.close()


def check_no_foreign_gzserver(trial):
    """Every launch of this runner is stopped (and verified) before the next one starts, so any
    gzserver running now belongs to someone else: abort instead of touching it."""
    foreign = running_gzservers()
    if foreign:
        raise AbortBatch('gzserver not started by this runner is running (pids %s); refusing '
                         'to start trial %s (not touching other sessions)'
                         % (foreign, trial.trial_id))


def make_launcher():
    def launcher(trial, out_dir, meta_json, log_path):
        check_no_foreign_gzserver(trial)
        argv = ['roslaunch', 'neupan_ros', 'exp_trial.launch',
                'method:=%s' % trial.method, 'world:=%s' % trial.world,
                'x:=%r' % float(trial.start[0]), 'y:=%r' % float(trial.start[1]),
                'yaw:=%r' % float(trial.start[2]), 'goal:=%s' % goal_arg(trial),
                'timeout:=%r' % float(trial.timeout), 'out_dir:=%s' % out_dir,
                'trial_id:=%s' % trial.trial_id, 'meta:=%s' % meta_json]
        return Proc(argv, log_path)
    return launcher


ROSCORE_PARAMS = ('/rosdistro', '/rosversion')


def top_namespaces(names):
    return {n.split('/')[1] for n in names if n.startswith('/') and len(n) > 1}


def reset_params(get_names, delete, baseline_top, owns_master):
    """Drop parameters left by the previous trial's launch.
    Own roscore: everything except what roscore itself sets. Someone else's roscore: only
    /use_sim_time and the namespaces that did not exist when this runner started (the ones its
    own launches create); anything else belongs to other users and is left alone."""
    for name in get_names():
        if name in ROSCORE_PARAMS:
            continue
        if not owns_master and name != '/use_sim_time' and \
                name.split('/')[1] in baseline_top:
            continue
        try:
            delete(name)
        except KeyError:
            pass


class RosIO:
    """Wall-time rospy node: latest /clock, /cmd_vel counter, goal publisher."""

    def __init__(self, owns_master=True):
        import rospy
        from geometry_msgs.msg import PoseStamped, Twist
        from rosgraph_msgs.msg import Clock
        self.rospy, self.PoseStamped = rospy, PoseStamped
        try:                              # a stale value would make init_node use sim time
            rospy.delete_param('/use_sim_time')
        except KeyError:
            pass
        rospy.init_node('exp_batch_runner', anonymous=True, disable_signals=True)
        self.owns_master = owns_master
        self.baseline_top = top_namespaces(rospy.get_param_names())   # before our first launch
        self._clock = None
        self._n_cmd = 0
        rospy.Subscriber('/clock', Clock, self._on_clock, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self._on_cmd, queue_size=1)
        self.pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=5)

    def _on_clock(self, msg):
        self._clock = msg.clock.to_sec()

    def _on_cmd(self, _msg):
        self._n_cmd += 1

    def reset(self):
        """Before each launch: forget clock/cmd_vel, drop params left by the previous trial."""
        reset_params(self.rospy.get_param_names, self.rospy.delete_param, self.baseline_top,
                     self.owns_master)
        self._clock = None
        self._n_cmd = 0

    def clock(self):
        return self._clock

    def ready(self):
        return self._n_cmd > 0

    def publish_goal(self, x, y):
        if not getattr(self, '_connected', False):   # recorder + planner subscribe
            end = time.time() + 5.0
            while self.pub.get_num_connections() < 2 and time.time() < end:
                time.sleep(0.05)
            self._connected = True
        m = self.PoseStamped()
        m.header.frame_id = 'odom'
        m.header.stamp = self.rospy.Time(0)
        m.pose.position.x, m.pose.position.y = x, y
        m.pose.orientation.w = 1.0
        self.pub.publish(m)

    def goal_subscribers(self):
        return self.pub.get_num_connections()

    def new_trial(self):
        self._connected = False


def ensure_roscore(log_path):
    import rosgraph
    if rosgraph.is_master_online():
        return None
    rc = Proc(['roscore'], log_path)
    end = time.time() + 30
    while not rosgraph.is_master_online():
        if time.time() > end or rc.poll() is not None:
            rc.stop()
            raise AbortBatch('could not start roscore (see %s)' % log_path)
        time.sleep(0.3)
    return rc


# -------------------------------------------------------------------------------------- manifest
def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
    except OSError as e:
        return 'unreadable: %s' % e
    return h.hexdigest()


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', repo] + list(args), capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return 'error: %s' % e


def git_info():
    info = {}
    for name, repo in REPOS.items():
        lines = [l for l in _git(repo, 'status', '--porcelain').splitlines() if l.strip()]
        by = {}
        for l in lines:
            by[l[:2]] = by.get(l[:2], 0) + 1
        info[name] = dict(head=_git(repo, 'rev-parse', 'HEAD').strip(),
                          branch=_git(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip(),
                          n_changed=len(lines), by_status=by,
                          # deletions are summarised by count only; the rest is listed
                          changes_except_deleted=[l for l in lines if ' D' not in l[:2]][:200])
    return info


def _read(path):
    with open(path) as f:
        return f.read()


def hashed_files(config_path):
    """HASHED_FILES with the default scenarios.yaml replaced by the one actually used."""
    default = EXP_DIR + '/config/scenarios.yaml'
    return [os.path.abspath(config_path) if p == default else p for p in HASHED_FILES]


class Manifest:
    def __init__(self, path, args):
        self.path = path
        prev = load_result(path) or {}
        self.data = dict(
            run_name=os.path.basename(os.path.normpath(os.path.dirname(path))),
            cli_args=vars(args), argv=sys.argv, start_wall=_now(), end_wall=None,
            status='running', git=git_info(),
            sha256={(os.path.relpath(p, WS) if p.startswith(WS + '/') else p): sha256(p)
                    for p in hashed_files(args.config)},
            scenarios_yaml=_read(args.config),
            trials=prev.get('trials', {}), history=prev.get('history', []))
        if prev:
            self.data['history'].append({k: prev.get(k) for k in
                                         ('cli_args', 'start_wall', 'end_wall', 'status', 'git',
                                          'sha256')})
            if prev.get('scenarios_yaml') != self.data['scenarios_yaml']:
                sys.stderr.write('WARNING: scenarios.yaml differs from the earlier invocation '
                                 'of this run\n')
        self.save()

    def save(self):
        _atomic_json(self.path, self.data)

    def record(self, trial, out_dir):
        r = load_result(_result_paths(out_dir, trial.trial_id)[0]) or {}
        m = r.get('meta', {}) if isinstance(r.get('meta'), dict) else {}
        self.data['trials'][trial.trial_id] = dict(
            outcome=r.get('outcome'), attempts=r.get('attempts'), T=r.get('T'),
            collisions=r.get('collisions'), late_start=m.get('late_start'),
            actual_start_offset=m.get('actual_start_offset'), reason=r.get('reason'))
        self.save()

    def finish(self, status, error=None):
        self.data['end_wall'] = _now()
        self.data['status'] = status
        if error is not None:
            self.data['error'] = error
        self.save()


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


# ------------------------------------------------------------------------------------------ main
def select_trials(cfg, scenes, methods, limit=None):
    trials = trial_matrix(cfg, scenes, methods)
    if limit is None:
        return trials
    seen, out = {}, []
    for t in trials:
        k = (t.scene, t.method)
        if seen.get(k, 0) < limit:
            seen[k] = seen.get(k, 0) + 1
            out.append(t)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=EXP_DIR + '/config/scenarios.yaml')
    ap.add_argument('--scenes', default='S1,S2')
    ap.add_argument('--methods', default='neupan,aabb,dwa')
    ap.add_argument('--run-name', default='run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    ap.add_argument('--out', default=None, help='default: %s/exp_data/<run_name>' % WS)
    ap.add_argument('--limit', type=int, default=None,
                    help='only the first N trials per scene x method')
    ap.add_argument('--retries', type=int, default=1, help='extra attempts for crash trials')
    ap.add_argument('--dry-run', action='store_true', help='list the trials and exit')
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(WS, 'exp_data', args.run_name)
    return args


def run_matrix(trials, trials_dir, launcher, io, retries=1, logs_dir=None, manifest=None,
               say=print, precheck=None, **kw):
    """Run all trials in order; return {outcome: count}. A trial whose run raises (other than
    AbortBatch/KeyboardInterrupt) is recorded as a crash and the batch goes on. `io` provides
    clock, publish_goal, goal_subscribers, ready, reset, new_trial (RosIO or a fake)."""
    counts = {}
    for i, t in enumerate(trials, 1):
        t0 = time.time()
        say('[%s] (%d/%d) %s ...' % (_now(), i, len(trials), t.trial_id))
        io.new_trial()
        try:
            outcome = run_trial(
                t, trials_dir, launcher, io.clock, io.publish_goal, retries=retries,
                ready=io.ready, reset=io.reset, goal_subscribers=io.goal_subscribers,
                precheck=precheck,
                log_path=os.path.join(logs_dir, t.trial_id + '.launch.log') if logs_dir else None,
                **kw)
        except (AbortBatch, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001 - e.g. a failing stop(); keep the batch alive
            sys.stderr.write('trial %s failed outside run_one:\n%s' % (t.trial_id,
                                                                     traceback.format_exc()))
            jp = _result_paths(trials_dir, t.trial_id)[0]
            old = load_result(jp) or {}
            _atomic_json(jp, crash_json(t, repr(e), trial_meta(t, 1 + int(old.get('attempts', 0))),
                                        1 + int(old.get('attempts', 0))))
            outcome = 'crash'
        counts[outcome] = counts.get(outcome, 0) + 1
        if manifest is not None:
            manifest.record(t, trials_dir)
        say('[%s]     -> %s (%.0f s wall)' % (_now(), outcome, time.time() - t0))
    return counts


def main(argv=None):
    args = parse_args(argv)
    cfg = load_scenarios(args.config)
    trials = select_trials(cfg, args.scenes.split(','), args.methods.split(','), args.limit)
    if args.dry_run:
        for t in trials:
            print(t.trial_id, t.world, 'start_sim_time=%.2f' % t.start_sim_time,
                  'timeout=%.0f' % t.timeout)
        print(len(trials), 'trials')
        return 0
    out = os.path.abspath(args.out)
    trials_dir, logs_dir = os.path.join(out, 'trials'), os.path.join(out, 'logs')
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    manifest = Manifest(os.path.join(out, 'manifest.json'), args)

    def on_signal(signum, _frame):
        raise KeyboardInterrupt('signal %d' % signum)

    old_handlers = {sg: signal.signal(sg, on_signal) for sg in INTERRUPT_SIGNALS}
    roscore, status, error = None, 'complete', None
    try:
        roscore = ensure_roscore(os.path.join(logs_dir, 'roscore.log'))
        io = RosIO(owns_master=roscore is not None)
        counts = run_matrix(trials, trials_dir, make_launcher(), io, retries=args.retries,
                            logs_dir=logs_dir, manifest=manifest,
                            say=lambda m: print(m, flush=True),
                            precheck=check_no_foreign_gzserver)
        print('done:', counts)
    except AbortBatch as e:
        status = 'aborted'
        error = str(e)
        sys.stderr.write('ABORT: %s\n' % e)
    except KeyboardInterrupt:
        status = 'interrupted'
        sys.stderr.write('interrupted\n')
    except Exception:  # noqa: BLE001
        status = 'error'
        error = traceback.format_exc()
        sys.stderr.write(error)
    finally:
        if roscore is not None:
            try:
                roscore.stop()
            except AbortBatch as e:
                sys.stderr.write('%s\n' % e)
        manifest.finish(status, error)
        for sg, h in old_handlers.items():
            signal.signal(sg, h)
    return 0 if status == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
