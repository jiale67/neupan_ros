import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import math
import unittest

from neupan_exp.scenarios import load_scenarios, trial_matrix

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'scenarios.yaml')


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.cfg = load_scenarios(CFG)

    def test_counts_and_ids(self):
        ms = ['neupan', 'dwa']
        self.assertEqual(len(trial_matrix(self.cfg, ['S2'], ms)), 4 * 10 * 2)
        self.assertEqual(len(trial_matrix(self.cfg, ['S1'], ms)), 4 * 5 * 2)
        t = trial_matrix(self.cfg, ['S1'], ['neupan'])[0]
        self.assertEqual(t.trial_id, '%s_%s_p%d_s%d' % (t.scene, t.method, t.pair_id, t.seed))
        self.assertEqual(t.world, 'static_scene')
        self.assertEqual(t.start_sim_time, 10.0)
        self.assertAlmostEqual(t.start[2], math.atan2(t.goal[1] - t.start[1], t.goal[0] - t.start[0]))

    def test_trial_matrix_shared_seeds(self):
        tm = trial_matrix(self.cfg, ['S2'], ['neupan', 'dwa'])
        by = {(t.method, t.pair_id, t.seed): t for t in tm}
        for t in tm:
            if t.method == 'neupan':
                o = by[('dwa', t.pair_id, t.seed)]
                self.assertEqual(o.start_sim_time, t.start_sim_time)
                self.assertTrue(10.0 <= t.start_sim_time <= 40.0)
                self.assertEqual(t.world, 'dynamic_scene')
        self.assertTrue(len({t.start_sim_time for t in tm}) > 10)
        self.assertEqual(sorted({t.seed for t in tm if t.pair_id == 2})[0], 2000)

    def test_timeout(self):
        for t in trial_matrix(self.cfg, ['S1', 'S2'], ['neupan']):
            d = math.hypot(t.goal[0] - t.start[0], t.goal[1] - t.start[1])
            self.assertAlmostEqual(t.timeout, max(60, 3.0 * d / 0.3))


if __name__ == '__main__':
    unittest.main()
