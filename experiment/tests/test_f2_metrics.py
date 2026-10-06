"""Fix F2: pure helpers of the recorder / analysis (settle, e-stops, J split, accel, std, flags)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neupan_exp.metrics import (TrialSequencer, accel_samples, aggregate, cmd_jerk_split,  # noqa: E402
                                compute_trial_metrics, rtf_of, sample_std, settle_done,
                                status_gap_max, uniform_estops)

NAN = float('nan')


class TestSettleDone(unittest.TestCase):
    def test_not_done_while_moving(self):
        ts = [10.0, 10.1, 10.2, 10.3]
        self.assertFalse(settle_done(ts, [NAN, 0.1, 0.1, 0.1], 10.0))

    def test_done_after_one_second_below_eps(self):
        ts = [10.0 + 0.1 * i for i in range(14)]          # 10.0 .. 11.3
        sp = [1.0] + [0.01] * 13
        self.assertTrue(settle_done(ts, sp, 10.0))
        self.assertFalse(settle_done(ts[:10], sp[:10], 10.0))   # 0.8 s of slow samples only

    def test_exactly_hold_boundary(self):
        ts = [10.0 + 0.1 * i for i in range(12)]          # 10.0 .. 11.1; slow from 10.1
        sp = [1.0] + [0.0] * 11
        self.assertTrue(settle_done(ts, sp, 10.0))        # 11.1 - 10.1 = 1.0
        self.assertFalse(settle_done(ts[:-1], sp[:-1], 10.0))

    def test_interrupted_run_resets(self):
        ts = [10.0 + 0.1 * i for i in range(20)]
        sp = [0.0] * 20
        sp[0] = 1.0
        sp[12] = 0.5                                      # a burst at 11.2 restarts the hold
        self.assertFalse(settle_done(ts, sp, 10.0))
        self.assertTrue(settle_done(ts + [12.3], sp + [0.0], 10.0))

    def test_max_wait_caps_even_if_moving(self):
        ts = [10.0, 12.0, 14.9, 15.0]
        self.assertFalse(settle_done(ts[:3], [NAN, 0.5, 0.5], 10.0))
        self.assertTrue(settle_done(ts, [NAN, 0.5, 0.5, 0.5], 10.0))

    def test_samples_before_arrival_do_not_count(self):
        ts = [8.5, 9.0, 9.5, 10.0, 10.1]
        self.assertFalse(settle_done(ts, [0.0] * 5, 10.0))

    def test_custom_params(self):
        ts = [0.0, 0.5, 1.0]
        self.assertTrue(settle_done(ts, [NAN, 0.05, 0.05], 0.0, hold=0.5, v_eps=0.1))


class TestSequencer(unittest.TestCase):
    KW = dict(arrive_threshold=0.3, timeout=150.0, stuck_window=30.0, stuck_dist=0.1)

    def feed(self, seq, t, x, dist, prog=1.0, t0=100.0):
        return seq.update(t0 + t, t0, t, x, 0.0, dist, prog)

    def test_arrival_does_not_end_immediately(self):
        s = TrialSequencer(**self.KW)
        self.assertIsNone(self.feed(s, 10.0, 5.0, 0.5))
        self.assertIsNone(s.t_arrive)
        self.assertIsNone(self.feed(s, 10.02, 5.0, 0.29))
        self.assertAlmostEqual(s.t_arrive, 10.02)
        self.assertAlmostEqual(s.e_g_arrive, 0.29)

    def test_settles_after_hold_and_reports(self):
        s = TrialSequencer(**self.KW)
        self.feed(s, 10.0, 5.0, 0.3)
        r = None
        t = 10.0
        for i in range(1, 200):
            t = 10.0 + 0.02 * i
            r = self.feed(s, t, 5.0, 0.2)       # standing still
            if r:
                break
        self.assertEqual(r, 'arrived')
        self.assertAlmostEqual(t - 10.0, 1.02, places=2)
        self.assertAlmostEqual(s.settle_s, t - 10.0)

    def test_moving_robot_ends_at_max_wait(self):
        s = TrialSequencer(**self.KW)
        self.feed(s, 10.0, 0.0, 0.3)
        r, t, x = None, 10.0, 0.0
        while r is None and t < 30.0:
            t += 0.02
            x += 0.02 * 0.1
            r = self.feed(s, t, x, 0.1)
        self.assertEqual(r, 'arrived')
        self.assertAlmostEqual(t - 10.0, 5.0, places=1)

    def test_timeout_and_stuck_before_arrival_unchanged(self):
        s = TrialSequencer(**self.KW)
        self.assertEqual(self.feed(s, 151.0, 0.0, 3.0), 'timeout')
        s = TrialSequencer(**self.KW)
        self.assertEqual(self.feed(s, 40.0, 0.0, 3.0, prog=0.01), 'stuck')

    def test_timeout_not_applied_after_arrival(self):
        s = TrialSequencer(**dict(self.KW, timeout=10.0))
        self.assertIsNone(self.feed(s, 9.9, 0.0, 0.2))
        # slightly past timeout while settling and moving: still settling
        self.assertIsNone(self.feed(s, 10.5, 0.1, 0.25))


class TestEstops(unittest.TestCase):
    def cnt(self, cmds, t_end=None):
        t = [0.1 * i for i in range(len(cmds))]
        return uniform_estops(t, [c[0] for c in cmds], [c[1] for c in cmds],
                              [c[2] for c in cmds], t_end)

    Z = (0.0, 0.0, 0.0)
    N = (0.3, 0.0, 0.0)

    def test_zero_runs_counted_once_each(self):
        self.assertEqual(self.cnt([self.N, self.Z, self.Z, self.N, self.Z, self.N]), 2)

    def test_leading_zeros_ignored(self):
        self.assertEqual(self.cnt([self.Z, self.Z, self.N, self.N]), 0)
        self.assertEqual(self.cnt([self.Z, self.Z, self.N, self.Z, self.N]), 1)

    def test_post_arrival_zeros_ignored(self):
        cmds = [self.N, self.N, self.Z, self.N, self.Z]       # t = 0 .. 0.4
        self.assertEqual(self.cnt(cmds, t_end=0.3), 1)
        self.assertEqual(self.cnt(cmds, t_end=0.1), 0)
        self.assertEqual(self.cnt(cmds), 2)

    def test_tiny_nonzero_is_not_zero(self):
        self.assertEqual(self.cnt([self.N, (1e-6, 0.0, 0.0), self.N]), 0)
        self.assertEqual(self.cnt([self.N, (1e-10, -1e-10, 0.0), self.N]), 1)

    def test_empty(self):
        self.assertEqual(uniform_estops([], [], [], []), 0)


class TestJerkSplit(unittest.TestCase):
    def test_example(self):
        t = [0.0, 0.1, 0.2]
        jv, jw = cmd_jerk_split(t, [0, 0.3, 0.3], [0, 0, 0.1], [0, 0, 0.2])
        self.assertAlmostEqual(jv, 0.09 + 0.01)
        self.assertAlmostEqual(jw, 0.04)

    def test_t_end_truncates(self):
        t = [0.0, 0.1, 0.2]
        jv, jw = cmd_jerk_split(t, [0, 0.3, 0.3], [0, 0, 0.1], [0, 0, 0.2], t_end=0.1)
        self.assertAlmostEqual(jv, 0.09)
        self.assertAlmostEqual(jw, 0.0)

    def test_short(self):
        self.assertEqual(cmd_jerk_split([0.0], [1], [1], [1]), (0.0, 0.0))


class TestAccel(unittest.TestCase):
    def test_constant_dt_known_jumps(self):
        t = [0.0, 0.1, 0.2, 0.3]
        vx = [0.1, 0.2, 0.2, 0.25]
        vy = [0.1, 0.1, 0.1, 0.1]
        w = [0.1, 0.1, 0.3, 0.3]
        a = accel_samples(t, vx, vy, w)
        np.testing.assert_allclose(a['a_lin'], [1.0, 0.0, 0.5], atol=1e-9)
        np.testing.assert_allclose(a['a_ang'], [0.0, 2.0, 0.0], atol=1e-9)
        self.assertEqual(a['n_zero_trans'], 0)

    def test_norm_of_vector_difference(self):
        a = accel_samples([0.0, 0.1], [0.0, 0.03], [0.1, 0.14], [0.2, 0.2])
        np.testing.assert_allclose(a['a_lin'], [0.5], atol=1e-9)

    def test_zero_transitions_excluded_and_counted(self):
        t = [0.1 * i for i in range(6)]
        vx = [0.3, 0.3, 0.0, 0.0, 0.3, 0.35]
        z = [0.0] * 6
        a = accel_samples(t, vx, z, z)
        # pairs: (N,N) kept, (N,Z) trans, (Z,Z) dropped, (Z,N) trans, (N,N') kept
        np.testing.assert_allclose(a['a_lin'], [0.0, 0.5], atol=1e-9)
        self.assertEqual(len(a['a_ang']), 2)
        self.assertEqual(a['n_zero_trans'], 2)

    def test_uses_real_dt_and_t_end(self):
        t = [0.0, 0.05, 0.25]
        a = accel_samples(t, [0.0, 0.05, 0.05], [0, 0, 0], [0.1, 0.1, 0.1])
        # first pair is from an all-nonzero start? (0,0,0.1) is non-zero by omega
        np.testing.assert_allclose(a['a_lin'], [1.0, 0.0], atol=1e-9)
        b = accel_samples(t, [0.0, 0.05, 0.05], [0, 0, 0], [0.1, 0.1, 0.1], t_end=0.1)
        self.assertEqual(len(b['a_lin']), 1)

    def test_nonpositive_dt_skipped(self):
        a = accel_samples([0.0, 0.0, 0.1], [0.1, 0.2, 0.3], [0, 0, 0], [0.1, 0.1, 0.1])
        np.testing.assert_allclose(a['a_lin'], [1.0], atol=1e-9)


class TestSampleStd(unittest.TestCase):
    def test_ddof1(self):
        self.assertAlmostEqual(sample_std([1.0, 2.0, 3.0, 4.0]), math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(sample_std([10.0, 20.0]), math.sqrt(50.0))

    def test_nan_for_n_lt_2(self):
        self.assertTrue(math.isnan(sample_std([1.0])))
        self.assertTrue(math.isnan(sample_std([])))

    def test_aggregate_uses_sample_std(self):
        def row(T):
            return dict(outcome='success', success=True, T=T, L=1.0, e_g=0.1, J_du=1.0,
                        d_min_np=0.2, d_min_body=T / 100.0, collisions=0, estops=0,
                        t_avg=1, t95=2, t_max=3, r_over=0, n_cycles=10)
        a = aggregate([row(10.0), row(20.0), row(30.0)])
        self.assertAlmostEqual(a['T_std'], 10.0)
        self.assertAlmostEqual(a['d_min_body_std'], 0.1)
        self.assertTrue(math.isnan(aggregate([row(10.0)])['T_std']))


class TestTrialMetricsSettle(unittest.TestCase):
    def log(self):
        t = np.arange(0, 12.0 + 1e-9, 0.02)
        x = np.minimum(0.5 * t, 5.0)                # reaches 5.0 at t=10, goal at 5.2
        return {'t': t, 'x': x, 'y': np.zeros_like(t), 'yaw': np.zeros_like(t),
                'gap_np': np.full_like(t, 0.3), 'gap_body': np.full_like(t, 0.2),
                'cmd_t': np.array([0.0, 5.0, 11.0]), 'cmd_vx': np.array([0.5, 0.5, 0.0]),
                'cmd_vy': np.zeros(3), 'cmd_w': np.zeros(3),
                'st_t': np.array([1.0, 2.0, 11.0]), 'plan_ms': np.array([10.0, 20.0, 1.0]),
                'stop': np.array([False, False, True])}

    def test_t_arrive_defines_T_and_cuts_L_and_cycles(self):
        m = compute_trial_metrics(self.log(), (5.2, 0.0), 'arrived', t_arrive=9.0)
        self.assertAlmostEqual(m['T'], 9.0)
        self.assertAlmostEqual(m['L'], 4.5, places=3)
        self.assertAlmostEqual(m['e_g_arrive'], 0.7, places=2)
        self.assertAlmostEqual(m['e_g'], 0.2, places=6)       # end of settle window
        self.assertEqual(m['n_cycles'], 2)                     # status after arrival excluded
        self.assertEqual(m['estops'], 0)

    def test_collision_during_settle_counts(self):
        log = self.log()
        gb = np.full_like(log['t'], 0.2)
        gb[log['t'] > 10.5] = -0.01
        log['gap_body'] = gb
        m = compute_trial_metrics(log, (5.2, 0.0), 'arrived', t_arrive=9.0)
        self.assertEqual(m['outcome'], 'collision')
        self.assertEqual(m['collisions'], 1)

    def test_without_t_arrive_unchanged(self):
        m = compute_trial_metrics(self.log(), (5.2, 0.0), 'arrived')
        self.assertAlmostEqual(m['T'], 12.0)
        self.assertEqual(m['n_cycles'], 3)
        self.assertEqual(m['estops'], 1)
        self.assertNotIn('e_g_arrive', m)


class TestRtf(unittest.TestCase):
    def test_rtf(self):
        self.assertAlmostEqual(rtf_of(50.0, 100.0), 0.5)
        self.assertTrue(math.isnan(rtf_of(5.0, 0.0)))


class TestStatusGap(unittest.TestCase):
    def test_max_gap_before_t_end(self):
        self.assertAlmostEqual(status_gap_max([0.0, 0.1, 0.2, 1.0, 1.1], 1.1), 0.8)

    def test_gaps_after_t_end_ignored(self):
        self.assertAlmostEqual(status_gap_max([0.0, 0.1, 0.2, 9.0], 0.25), 0.1, places=6)

    def test_tail_gap_counts(self):
        self.assertAlmostEqual(status_gap_max([0.0, 0.1, 0.2], 2.2), 2.0)

    def test_no_status(self):
        self.assertEqual(status_gap_max([], 5.0), 0.0)


if __name__ == '__main__':
    unittest.main()
