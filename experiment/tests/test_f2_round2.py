"""Fix F2 review round 2: backlog 1, in-trial pose stats, speed over >= 20 ms pairs, guarded worker."""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))

import analyze  # noqa: E402
from neupan_exp import metrics  # noqa: E402
from neupan_exp.metrics import (POSE_MAX_BACKLOG, SkipAheadQueue, TrialSequencer,  # noqa: E402
                                guarded_call, max_gap)


class TestBacklogOne(unittest.TestCase):
    def test_recorder_backlog_is_one(self):
        self.assertEqual(POSE_MAX_BACKLOG, 1)
        src = open(os.path.join(HERE, '..', 'nodes', 'trial_recorder.py'),
                   encoding='utf-8').read()
        self.assertIn('SkipAheadQueue(max_backlog=POSE_MAX_BACKLOG)', src)

    def simulate(self, backlog, rate_hz=66.0, service_s=0.018, seconds=30.0):
        """Producer at rate_hz, worker needs service_s per item; returns processed stamps."""
        q = SkipAheadQueue(max_backlog=backlog)
        dt = 1.0 / rate_hz
        n_msgs = int(seconds * rate_hz)
        arrivals = [i * dt for i in range(n_msgs)]
        stamps, free, i = [], 0.0, 0
        while i < n_msgs:
            # everything that arrived by `free` is queued; if empty, wait for the next one
            if not len(q):
                free = max(free, arrivals[i])
            while i < n_msgs and arrivals[i] <= free + 1e-12:
                q.push(arrivals[i])
                i += 1
            item = q.pop()
            if item is None:
                continue
            stamps.append(item)
            free += service_s
        return stamps

    def test_backlog_one_gives_even_sampling(self):
        st1 = self.simulate(1)
        gaps = [b - a for a, b in zip(st1, st1[1:])]
        self.assertLess(max(gaps), 0.035)
        self.assertGreater(len(st1) / 30.0, 48.0)

    def test_backlog_three_is_burstier(self):
        gaps3 = [b - a for a, b in zip(*(lambda s: (s, s[1:]))(self.simulate(3)))]
        gaps1 = [b - a for a, b in zip(*(lambda s: (s, s[1:]))(self.simulate(1)))]
        self.assertGreater(max(gaps3), max(gaps1))


class TestMaxGap(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(max_gap([0.0, 0.02, 0.15, 0.17]), 0.13)
        self.assertEqual(max_gap([1.0]), 0.0)
        self.assertEqual(max_gap([]), 0.0)


class TestPoseFlags(unittest.TestCase):
    base = dict(estops_src='cmd', outcome='success')

    def f(self, **kw):
        return analyze.sanity_flags(dict(self.base, **kw))

    def test_gap_flag_replaces_drop_ratio(self):
        self.assertIn('pose_drops', self.f(pose_max_gap=0.12))
        self.assertNotIn('pose_drops', self.f(pose_max_gap=0.1))
        # a high drop ratio alone no longer flags
        self.assertNotIn('pose_drops', self.f(pose_received=100, pose_dropped=60,
                                              pose_max_gap=0.04))
        self.assertNotIn('pose_drops', self.f())           # old data without the field
        self.assertNotIn('pose_drops', self.f(pose_max_gap=''))

    def test_errors_flag(self):
        self.assertIn('pose_errors', self.f(pose_errors=1))
        self.assertNotIn('pose_errors', self.f(pose_errors=0))
        self.assertNotIn('pose_errors', self.f())


class TestSettleSpeedMinDt(unittest.TestCase):
    KW = dict(arrive_threshold=0.3, timeout=150.0, stuck_window=30.0, stuck_dist=0.1)

    def feed(self, s, t, x, y=0.0, dist=0.2):
        return s.update(100.0 + t, 100.0, t, x, y, dist, 1.0)

    def test_duplicate_stamps_do_not_reset_the_hold(self):
        s = TrialSequencer(**self.KW)
        self.feed(s, 10.0, 5.0, dist=0.3)
        r, t = None, 10.0
        for i in range(1, 80):
            t = 10.0 + 0.02 * i
            r = self.feed(s, t, 5.0)
            if r:
                break
            # a duplicate receive stamp with a slightly different (noisy) position
            r = self.feed(s, t, 5.0 + 1e-4)
            if r:
                break
        self.assertEqual(r, 'arrived')
        self.assertAlmostEqual(t - 10.0, 1.0, delta=0.06)

    def test_pair_below_min_dt_carries_previous_speed(self):
        s = TrialSequencer(**self.KW)
        self.feed(s, 10.0, 5.0, dist=0.3)
        self.feed(s, 10.1, 5.0)          # slow
        self.feed(s, 10.101, 5.002)      # 1 ms later, 2 mm away: speed estimate 2 m/s if used
        self.assertLess(s._speeds[-1], 0.02)

    def test_fast_motion_still_blocks_settling(self):
        s = TrialSequencer(**self.KW)
        self.feed(s, 10.0, 5.0, dist=0.3)
        r, t, x = None, 10.0, 5.0
        while r is None and t < 14.0:
            t += 0.025
            x += 0.025 * 0.3
            r = self.feed(s, t, x)
        self.assertIsNone(r)             # moving at 0.3 m/s: never settles before max_wait

    def test_min_dt_constant(self):
        self.assertAlmostEqual(metrics.SETTLE_MIN_DT, 0.02)


class TestGuardedCall(unittest.TestCase):
    def test_exception_does_not_propagate_and_is_reported(self):
        seen = []

        def boom(a):
            raise ValueError('bad message %s' % a)

        ok = guarded_call(boom, (7,), lambda exc, tb: seen.append((exc, tb)))
        self.assertFalse(ok)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0][0], ValueError)
        self.assertIn('Traceback', seen[0][1])
        self.assertIn('bad message 7', seen[0][1])

    def test_success_returns_true_and_skips_handler(self):
        calls = []
        self.assertTrue(guarded_call(lambda a, b: calls.append(a + b), (1, 2), lambda *a: 1 / 0))
        self.assertEqual(calls, [3])

    def test_loop_continues_after_a_failure(self):
        q = SkipAheadQueue(max_backlog=10)
        for i in range(5):
            q.push(i)
        done, errs = [], []

        def process(i):
            if i == 2:
                raise RuntimeError('x')
            done.append(i)

        while len(q):
            guarded_call(process, (q.pop(),), lambda e, tb: errs.append(e))
        self.assertEqual(done, [0, 1, 3, 4])
        self.assertEqual(len(errs), 1)


if __name__ == '__main__':
    unittest.main()
