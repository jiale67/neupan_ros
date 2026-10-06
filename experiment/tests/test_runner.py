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
            json.dump({'outcome': 'timeout', 'trial_id': trial.trial_id}, f)
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


if __name__ == '__main__':
    unittest.main()
