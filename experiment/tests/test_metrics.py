import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import unittest
import numpy as np

from neupan_exp.metrics import compute_trial_metrics, aggregate


def straight_log(gap_body):
    t = np.arange(0, 10.0 + 1e-9, 0.05)
    return {'t': t, 'x': 0.3 * t, 'y': np.zeros_like(t), 'yaw': np.zeros_like(t),
            'gap_np': np.full_like(t, 0.3), 'gap_body': gap_body(t),
            'cmd_t': np.array([0.0, 0.1]), 'cmd_vx': np.array([0.3, 0.3]),
            'cmd_vy': np.zeros(2), 'cmd_w': np.zeros(2),
            'st_t': np.array([0.0, 0.1]), 'plan_ms': np.array([10.0, 20.0]),
            'stop': np.array([False, False])}


class TestMetrics(unittest.TestCase):
    def test_collision_then_arrive_is_failure(self):
        log = straight_log(lambda t: np.where((t >= 4) & (t <= 5), -0.01, 0.2))
        m = compute_trial_metrics(log, (3.0, 0.0), 'arrived')
        self.assertEqual(m['outcome'], 'collision')
        self.assertEqual(m['collisions'], 1)
        self.assertFalse(m['success'])
        self.assertAlmostEqual(m['T'], 10.0)
        self.assertAlmostEqual(m['L'], 3.0, places=6)
        self.assertAlmostEqual(m['e_g'], 0.0, places=6)
        self.assertAlmostEqual(m['d_min_body'], -0.01)

    def test_success(self):
        log = straight_log(lambda t: np.full_like(t, 0.2))
        m = compute_trial_metrics(log, (3.0, 0.0), 'arrived')
        self.assertEqual(m['outcome'], 'success')
        self.assertTrue(m['success'])
        self.assertEqual(compute_trial_metrics(log, (3.0, 0.0), 'timeout')['outcome'], 'timeout')

    def test_J_du(self):
        log = straight_log(lambda t: np.full_like(t, 0.2))
        log.update(cmd_t=np.arange(3) * 0.1, cmd_vx=np.array([0, 0.3, 0.3]),
                   cmd_vy=np.array([0, 0, 0.1]), cmd_w=np.zeros(3))
        m = compute_trial_metrics(log, (3.0, 0.0), 'arrived')
        self.assertAlmostEqual(m['J_du'], 0.09 + 0.01)

    def test_estops(self):
        log = straight_log(lambda t: np.full_like(t, 0.2))
        log.update(st_t=np.arange(5) * 0.1, plan_ms=np.ones(5),
                   stop=np.array([0, 1, 1, 0, 1], dtype=bool))
        self.assertEqual(compute_trial_metrics(log, (3.0, 0.0), 'arrived')['estops'], 2)

    def test_r_over_and_t95(self):
        log = straight_log(lambda t: np.full_like(t, 0.2))
        p = np.concatenate([np.arange(1, 101, dtype=float), [150.0]])
        log.update(st_t=np.arange(101) * 0.1, plan_ms=p, stop=np.zeros(101, dtype=bool))
        m = compute_trial_metrics(log, (3.0, 0.0), 'arrived')
        self.assertAlmostEqual(m['r_over'], 100.0 / 101, places=2)
        self.assertEqual(m['t_max'], 150.0)
        self.assertEqual(m['n_cycles'], 101)
        self.assertAlmostEqual(m['t95'], np.percentile(p, 95))

    def test_aggregate_excludes_failed_from_T(self):
        def row(o, T):
            return dict(outcome=o, success=o == 'success', T=T, L=1.0, e_g=0.1, J_du=1.0,
                        d_min_np=0.2, d_min_body=0.3, collisions=0, estops=0,
                        t_avg=1, t95=2, t_max=3, r_over=0, n_cycles=10)
        a = aggregate([row('success', 10.0), row('success', 20.0), row('timeout', 60.0)])
        self.assertAlmostEqual(a['T_mean'], 15.0)
        self.assertAlmostEqual(a['SR'], 66.6667, places=3)
        self.assertEqual(a['outcomes']['timeout'], 1)
        self.assertEqual(a['n_trials'], 3)
        self.assertTrue(np.isnan(aggregate([row('timeout', 60.0)])['T_mean']))

    def test_aggregate_pools_plan_ms(self):
        def row(p):
            return dict(outcome='success', success=True, T=1, L=1, e_g=0, J_du=0, d_min_np=1,
                        d_min_body=1, collisions=0, estops=0, t_avg=0, t95=0, t_max=0,
                        r_over=0, n_cycles=len(p), plan_ms_all=p)
        a = aggregate([row([10.0, 20.0]), row([30.0, 200.0])])
        self.assertAlmostEqual(a['t_avg'], 65.0)
        self.assertEqual(a['t_max'], 200.0)
        self.assertAlmostEqual(a['r_over'], 25.0)

    def test_aggregate_crash_row_nan_aware(self):
        nan = float('nan')
        ok = dict(outcome='success', success=True, T=10.0, L=3, e_g=0.1, J_du=1, d_min_np=0.3,
                  d_min_body=0.2, collisions=0, estops=1, t_avg=5.0, t95=6.0, t_max=7.0,
                  r_over=0.0, n_cycles=2, plan_ms_all=[5.0, 7.0])
        crash = dict(outcome='crash', success=False, T=nan, L=nan, e_g=nan, J_du=nan,
                     d_min_np=nan, d_min_body=nan, collisions=0, estops=0, t_avg=nan,
                     t95=nan, t_max=nan, r_over=nan, n_cycles=0, plan_ms_all=[])
        a = aggregate([ok, crash])
        self.assertAlmostEqual(a['d_min_body_min'], 0.2)
        self.assertAlmostEqual(a['d_min_body_mean'], 0.2)
        self.assertAlmostEqual(a['d_min_body_std'], 0.0)
        self.assertAlmostEqual(a['d_min_np_mean'], 0.3)
        self.assertAlmostEqual(a['t_avg'], 6.0)
        only = aggregate([crash])
        self.assertTrue(np.isnan(only['d_min_body_min']))
        self.assertTrue(np.isnan(only['d_min_body_mean']))
        self.assertTrue(np.isnan(only['t_avg']))
        self.assertEqual(only['outcomes'], {'crash': 1})


if __name__ == '__main__':
    unittest.main()
