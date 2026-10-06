import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import unittest

from neupan_exp.metrics import end_reason, window_progress

KW = dict(timeout=150.0, stuck_window=30.0, stuck_dist=0.1, arrive_threshold=0.3)


def er(t_rel, dist, progress, t0=145.0, **over):
    kw = dict(KW)
    kw.update(over)
    return end_reason(t0 + t_rel, t0, dist, progress, **kw)


class TestEndReason(unittest.TestCase):
    def test_arrived(self):
        self.assertEqual(er(50.0, 0.2, 1.0), 'arrived')

    def test_arrived_at_threshold_boundary(self):
        self.assertEqual(er(50.0, 0.3, 1.0), 'arrived')
        self.assertIsNone(er(50.0, 0.31, 1.0))

    def test_timeout(self):
        self.assertEqual(er(150.5, 3.0, 1.0), 'timeout')
        self.assertIsNone(er(149.5, 3.0, 1.0))

    def test_stuck_full_window(self):
        self.assertEqual(er(30.0, 3.0, 0.01), 'stuck')
        self.assertEqual(er(80.0, 3.0, 0.01), 'stuck')

    def test_not_stuck_when_progressing(self):
        self.assertIsNone(er(60.0, 3.0, 0.5))

    def test_not_stuck_before_window_filled(self):
        self.assertIsNone(er(10.0, 3.0, 0.01))

    def test_waiting_29s_not_stuck(self):
        self.assertIsNone(er(29.0, 3.0, 0.0))

    def test_precedence_arrived_over_timeout_over_stuck(self):
        self.assertEqual(er(200.0, 0.1, 0.0), 'arrived')
        self.assertEqual(er(200.0, 3.0, 0.0), 'timeout')

    def test_sim_time_offset_irrelevant(self):
        self.assertEqual(er(30.0, 3.0, 0.01, t0=1.0e5), 'stuck')


class TestWindowProgress(unittest.TestCase):
    def test_none_until_window_covered(self):
        self.assertIsNone(window_progress([0, 5, 10], [5, 4, 3], 30.0))

    def test_progress_is_start_minus_min(self):
        ts = [0, 10, 20, 30, 40]
        d = [5.0, 4.0, 3.0, 3.5, 3.4]
        # window [10, 40]: start-of-window dist 4.0, min 3.0
        self.assertAlmostEqual(window_progress(ts, d, 30.0), 1.0)

    def test_lateral_jitter_without_progress_is_zero(self):
        ts = [i * 0.5 for i in range(121)]
        d = [3.0 + 0.2 * (i % 2) for i in range(121)]
        # start-of-window sample is 3.0 (even idx); min 3.0 -> no progress
        self.assertLessEqual(window_progress(ts, d, 30.0), 1e-9)

    def test_never_negative(self):
        ts = [0, 15, 30]
        d = [3.0, 3.5, 4.0]
        self.assertEqual(window_progress(ts, d, 30.0), 0.0)

    def test_start_sample_is_last_at_or_before_window_start(self):
        ts = [0, 5, 20, 36]
        d = [9.0, 5.0, 4.5, 4.4]
        # window start t=6 -> last sample at/before is t=5 (5.0); min within window = 4.4
        self.assertAlmostEqual(window_progress(ts, d, 30.0), 0.6)


if __name__ == '__main__':
    unittest.main()
