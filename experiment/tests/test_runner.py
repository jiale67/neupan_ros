import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))

import run_batch as rb  # noqa: E402
from neupan_exp.scenarios import Trial  # noqa: E402


def make_trial(start_sim_time=2.0, timeout=60.0):
    return Trial('S1', 'static_scene', 'neupan', 1, (-9.0, -7.0, 0.7), (6.0, 6.5), 1000,
                 start_sim_time, timeout)


class FakeHandle:
    def __init__(self, on_stop=None):
        self.stopped = False
        self.on_stop = on_stop

    def poll(self):
        return 0 if self.stopped else None

    def stop(self):
        self.stopped = True


class FakeClock:
    """/clock stand-in: None until `t_first`, then +step per call."""
    def __init__(self, first=145.0, step=1.0):
        self.v = first - step
        self.step = step

    def __call__(self):
        self.v += self.step
        return self.v


def write_result(out_dir, trial, meta_json):
    with open(os.path.join(out_dir, trial.trial_id + '.json'), 'w') as f:
        json.dump({'outcome': 'success', 'success': True, 'trial_id': trial.trial_id,
                   'meta': json.loads(meta_json)}, f)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_wait_for_sim_time_reads_clock_msgs(self):
        vals = iter([None, 3.0, 9.9, 10.1])
        self.assertTrue(rb.wait_for_sim_time(lambda: next(vals), 10.0, 2.0, poll=0.001))
        self.assertFalse(rb.wait_for_sim_time(lambda: None, 10.0, 0.05, poll=0.001))

    def test_runner_marks_crash_on_wall_timeout(self):
        trial = make_trial()
        handles = []

        def launcher(t, out_dir, meta_json, log_path):
            h = FakeHandle()
            handles.append(h)
            return h

        outcome = rb.run_one(trial, self.out, launcher, FakeClock(), lambda x, y: None,
                             wall_timeout=0.5, ready=lambda: True, poll=0.01, exit_grace=0)
        self.assertEqual(outcome, 'crash')
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            j = json.load(f)
        self.assertEqual(j['outcome'], 'crash')
        self.assertIn('wall timeout', j['reason'])
        self.assertEqual(j['meta']['seed'], 1000)
        self.assertTrue(handles[0].stopped)

    def test_crash_when_planner_never_ready(self):
        trial = make_trial()
        goals = []
        outcome = rb.run_one(trial, self.out, lambda *a: FakeHandle(), FakeClock(),
                             lambda x, y: goals.append((x, y)), wall_timeout=5,
                             ready=lambda: False, ready_timeout=0.2, poll=0.01, exit_grace=0)
        self.assertEqual(outcome, 'crash')
        self.assertEqual(goals, [])
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            self.assertIn('ready', json.load(f)['reason'])

    def test_skip_existing(self):
        trial = make_trial()
        with open(os.path.join(self.out, trial.trial_id + '.json'), 'w') as f:
            json.dump({'outcome': 'timeout', 'trial_id': trial.trial_id,
                       'meta': {'attempt': 1}}, f)
        calls = []
        outcome = rb.run_trial(trial, self.out, lambda *a: calls.append(a), FakeClock(),
                               lambda x, y: None, retries=1)
        self.assertEqual(outcome, 'timeout')
        self.assertEqual(calls, [])

    def test_crash_retried_then_final(self):
        trial = make_trial()
        calls = []

        def launcher(t, out_dir, meta_json, log_path):
            calls.append(1)
            return FakeHandle()

        kw = dict(wall_timeout=0.1, ready=lambda: True, poll=0.01, exit_grace=0)
        o = rb.run_trial(trial, self.out, launcher, FakeClock(), lambda x, y: None, retries=1, **kw)
        self.assertEqual(o, 'crash')
        self.assertEqual(len(calls), 2)
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            self.assertEqual(json.load(f)['attempts'], 2)
        # resume: final crash is not run again with the same retry budget
        rb.run_trial(trial, self.out, launcher, FakeClock(), lambda x, y: None, retries=1, **kw)
        self.assertEqual(len(calls), 2)
        # a larger budget retries it
        rb.run_trial(trial, self.out, launcher, FakeClock(), lambda x, y: None, retries=2, **kw)
        self.assertEqual(len(calls), 3)

    def test_normal_run_publishes_derived_goal_and_marks_meta(self):
        trial = make_trial(start_sim_time=5.0)
        goals, metas = [], []
        state = {}

        def launcher(t, out_dir, meta_json, log_path):
            metas.append(json.loads(meta_json))
            state['meta_json'] = meta_json
            state['dir'] = out_dir
            return FakeHandle()

        def goal_pub(x, y):
            goals.append((x, y))
            write_result(state['dir'], trial, state['meta_json'])

        outcome = rb.run_one(trial, self.out, launcher, FakeClock(), goal_pub, wall_timeout=5,
                             ready=lambda: True, poll=0.001, exit_grace=0, goal_gap=0.0)
        self.assertEqual(outcome, 'success')
        self.assertEqual(goals, [trial.goal] * 3)
        self.assertEqual(metas[0]['goal'], list(trial.goal))
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            meta = json.load(f)['meta']
        self.assertFalse(meta['late_start'])
        self.assertGreaterEqual(meta['actual_start_offset'], 5.0)

    def test_late_start_when_ready_after_target(self):
        trial = make_trial(start_sim_time=2.0)
        state = {}
        ready_calls = {'n': 0}

        clock = FakeClock()

        def ready():
            clock()                        # sim time keeps running while the planner starts up
            ready_calls['n'] += 1
            return ready_calls['n'] > 10   # clock (+1 per poll) is far past the target by then

        def launcher(t, out_dir, meta_json, log_path):
            state['meta_json'], state['dir'] = meta_json, out_dir
            return FakeHandle()

        def goal_pub(x, y):
            write_result(state['dir'], trial, state['meta_json'])

        outcome = rb.run_one(trial, self.out, launcher, clock, goal_pub, wall_timeout=5,
                             ready=ready, poll=0.001, exit_grace=0, goal_gap=0.0)
        self.assertEqual(outcome, 'success')
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            meta = json.load(f)['meta']
        self.assertTrue(meta['late_start'])
        self.assertGreater(meta['actual_start_offset'], 2.0)
        self.assertEqual(meta['start_sim_time'], 2.0)

    def test_wall_timeout_formula(self):
        self.assertEqual(rb.wall_timeout_for(make_trial(timeout=100.0)), 240.0)


class FakeIO:
    """Stands in for RosIO in run_matrix."""
    def __init__(self, subscribers=2):
        self.clock = FakeClock()
        self.subscribers = subscribers
        self.new_trials = 0

    def ready(self):
        return True

    def reset(self):
        pass

    def new_trial(self):
        self.new_trials += 1

    def goal_subscribers(self):
        return self.subscribers

    def publish_goal(self, x, y):
        pass


class FakeParamServer:
    def __init__(self, names):
        self.names = list(names)

    def get_names(self):
        return list(self.names)

    def delete(self, name):
        self.names.remove(name)


class TestFixRound1(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def _json(self, trial):
        with open(os.path.join(self.out, trial.trial_id + '.json')) as f:
            return json.load(f)

    # 1. per-trial exceptions become crash jsons, batch continues
    def test_launcher_exception_becomes_crash_and_batch_continues(self):
        t1 = make_trial()
        t2 = Trial('S1', 'static_scene', 'dwa', 1, (-9.0, -7.0, 0.7), (6.0, 6.5), 1000, 2.0, 60.0)
        state = {}

        def launcher(t, out_dir, meta_json, log_path):
            if t.method == 'neupan':
                raise RuntimeError('roslaunch missing')
            state['meta_json'] = meta_json
            return FakeHandle()

        io = FakeIO()
        io.publish_goal = lambda x, y: write_result(self.out, t2, state['meta_json'])
        res = rb.run_matrix([t1, t2], self.out, launcher, io, retries=0, wall_timeout=3,
                            poll=0.001, exit_grace=0, goal_gap=0.0, say=lambda *a: None)
        self.assertEqual(res, {'crash': 1, 'success': 1})
        j = self._json(t1)
        self.assertEqual(j['outcome'], 'crash')
        self.assertEqual(j['reason'], repr(RuntimeError('roslaunch missing')))
        self.assertEqual(self._json(t2)['outcome'], 'success')
        self.assertEqual(io.new_trials, 2)

    def test_failing_stop_is_recorded_and_batch_continues(self):
        class BadStop(FakeHandle):
            def stop(self):
                raise OSError('cannot stop')

        t1 = make_trial()
        res = rb.run_matrix([t1], self.out, lambda *a: BadStop(), FakeIO(), retries=0,
                            wall_timeout=0.1, poll=0.001, exit_grace=0, say=lambda *a: None)
        self.assertEqual(res, {'crash': 1})
        self.assertIn('cannot stop', self._json(t1)['reason'])

    def test_abort_batch_propagates_from_trial(self):
        def launcher(*a):
            raise rb.AbortBatch('foreign gzserver')
        with self.assertRaises(rb.AbortBatch):
            rb.run_one(make_trial(), self.out, launcher, FakeClock(), lambda x, y: None,
                       ready=lambda: True, exit_grace=0)

    def _main_args(self, extra=()):
        return ['--scenes', 'S1', '--methods', 'neupan', '--limit', '1', '--out', self.out] \
            + list(extra)

    def test_unexpected_exception_sets_manifest_error(self):
        orig = rb.ensure_roscore
        rb.ensure_roscore = lambda log: (_ for _ in ()).throw(RuntimeError('boom'))
        self.addCleanup(setattr, rb, 'ensure_roscore', orig)
        import io as _io, contextlib
        with contextlib.redirect_stderr(_io.StringIO()) as err:
            rc = rb.main(self._main_args())
        self.assertNotEqual(rc, 0)
        with open(os.path.join(self.out, 'manifest.json')) as f:
            m = json.load(f)
        self.assertEqual(m['status'], 'error')
        self.assertIn('boom', m['error'])
        self.assertIn('Traceback', err.getvalue())

    # 2. manifest hashes/reads args.config
    def test_manifest_uses_args_config(self):
        cfg = os.path.join(self.out, 'my_scenarios.yaml')
        text = '# custom\ntimeout_min: 60\n'
        with open(cfg, 'w') as f:
            f.write(text)
        args = rb.parse_args(['--config', cfg, '--out', self.out])
        m = rb.Manifest(os.path.join(self.out, 'manifest.json'), args)
        self.assertEqual(m.data['scenarios_yaml'], text)
        self.assertIn(rb.sha256(cfg), m.data['sha256'].values())
        default = os.path.join(rb.EXP_DIR, 'config', 'scenarios.yaml')
        self.assertNotIn(rb.sha256(default), m.data['sha256'].values())

    # 3. unpatched non-crash json is incomplete
    def test_resume_reruns_json_without_runner_meta(self):
        trial = make_trial()
        with open(os.path.join(self.out, trial.trial_id + '.json'), 'w') as f:
            json.dump({'outcome': 'success', 'trial_id': trial.trial_id, 'meta': {}}, f)
        calls = []

        def launcher(t, out_dir, meta_json, log_path):
            calls.append(1)
            return FakeHandle()

        o = rb.run_trial(trial, self.out, launcher, FakeClock(), lambda x, y: None, retries=0,
                         wall_timeout=0.1, ready=lambda: True, poll=0.01, exit_grace=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(o, 'crash')   # fake never produces a json; the point is that it re-ran

    # 4. param reset on a foreign master
    def test_reset_params_foreign_master_keeps_others(self):
        base = ['/rosdistro', '/rosversion', '/run_id', '/other_user/x', '/other_user/y/z']
        srv = FakeParamServer(base)
        top = rb.top_namespaces(srv.get_names())
        srv.names += ['/use_sim_time', '/robot_description', '/gazebo/gravity_z',
                      '/neupan_node/goal', '/trial_recorder/out_dir']
        rb.reset_params(srv.get_names, srv.delete, top, owns_master=False)
        self.assertEqual(sorted(srv.names), sorted(base))

    def test_reset_params_own_master_clears_all_but_roscore_params(self):
        srv = FakeParamServer(['/rosdistro', '/rosversion', '/run_id', '/use_sim_time', '/a/b'])
        rb.reset_params(srv.get_names, srv.delete, set(), owns_master=True)
        self.assertEqual(sorted(srv.names), ['/rosdistro', '/rosversion'])

    def test_reset_params_foreign_master_stale_use_sim_time_removed(self):
        srv = FakeParamServer(['/use_sim_time', '/other/x'])
        rb.reset_params(srv.get_names, srv.delete, rb.top_namespaces(['/other/x']),
                        owns_master=False)
        self.assertEqual(srv.names, ['/other/x'])

    # 5. goal subscribers
    def test_goal_subscribers_recorded_and_crash_reason(self):
        trial = make_trial()
        state = {}

        def launcher(t, out_dir, meta_json, log_path):
            state['meta_json'], state['dir'] = meta_json, out_dir
            return FakeHandle()

        o = rb.run_one(trial, self.out, launcher, FakeClock(),
                       lambda x, y: write_result(state['dir'], trial, state['meta_json']),
                       wall_timeout=3, ready=lambda: True, poll=0.001, exit_grace=0,
                       goal_gap=0.0, goal_subscribers=lambda: 1)
        self.assertEqual(o, 'success')       # still published; a result is a result
        self.assertEqual(self._json(trial)['meta']['goal_subscribers'], 1)
        o = rb.run_one(trial, self.out, lambda *a: FakeHandle(), FakeClock(), lambda x, y: None,
                       wall_timeout=0.1, ready=lambda: True, poll=0.01, exit_grace=0,
                       goal_gap=0.0, goal_subscribers=lambda: 0)
        self.assertEqual(o, 'crash')
        j = self._json(trial)
        self.assertEqual(j['reason'], 'goal subscribers < 2')
        self.assertEqual(j['meta']['goal_subscribers'], 0)
        self.assertIn('wall timeout', j['meta']['crash_detail'])

    # 6. gzserver comm
    def test_is_gzserver(self):
        self.assertTrue(rb.is_gzserver('gzserver'))
        self.assertTrue(rb.is_gzserver('gzserver11'))
        self.assertFalse(rb.is_gzserver('gzclient'))
        self.assertFalse(rb.is_gzserver(None))

    # 7. SIGHUP like SIGTERM
    def test_interrupt_signals(self):
        import signal
        self.assertIn(signal.SIGTERM, rb.INTERRUPT_SIGNALS)
        self.assertIn(signal.SIGHUP, rb.INTERRUPT_SIGNALS)

    # 8. precheck before deleting the old crash json
    def test_precheck_abort_keeps_old_crash_json(self):
        trial = make_trial()
        path = os.path.join(self.out, trial.trial_id + '.json')
        with open(path, 'w') as f:
            json.dump({'outcome': 'crash', 'attempts': 1, 'meta': {'attempt': 1}}, f)

        def precheck(t):
            raise rb.AbortBatch('foreign gzserver')

        with self.assertRaises(rb.AbortBatch):
            rb.run_trial(trial, self.out, lambda *a: FakeHandle(), FakeClock(),
                         lambda x, y: None, retries=1, precheck=precheck)
        with open(path) as f:
            self.assertEqual(json.load(f)['attempts'], 1)


if __name__ == '__main__':
    unittest.main()
