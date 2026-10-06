import csv
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))

import analyze  # noqa: E402

NAN = float('nan')


def fake_trial(scene, method, seed, outcome, T, plan_ms, d_min=0.3, collisions=0, estops=0):
    return {
        'outcome': outcome, 'success': outcome == 'success', 'T': T, 'L': T * 0.3,
        'e_g': 0.1, 'd_min_np': d_min, 'd_min_body': d_min, 'collisions': collisions,
        'estops': estops, 'J_du': 2.0, 't_avg': sum(plan_ms) / len(plan_ms),
        't95': max(plan_ms), 't_max': max(plan_ms), 'r_over': 0.0,
        'n_cycles': len(plan_ms), 'plan_ms_all': plan_ms,
        'end_reason': 'arrived' if outcome in ('success', 'collision') else outcome,
        'trial_id': '%s_%s_p1_s%d' % (scene, method, seed), 'goal': [1.0, 0.0],
        'meta': {'method': method, 'scene': scene, 'pair_id': 1, 'seed': seed,
                 'start': [0.0, 0.0, 0.0], 'goal': [1.0, 0.0], 'world': 'static_scene'}}


def write_run(root, trials):
    os.makedirs(os.path.join(root, 'trials'))
    for tr in trials:
        with open(os.path.join(root, 'trials', tr['trial_id'] + '.json'), 'w') as f:
            json.dump(tr, f)


class AnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        trials = []
        for m, base in (('neupan', 10.0), ('dwa', 2.0)):
            trials.append(fake_trial('S1', m, 1000, 'success', 20.0, [base, base + 2]))
            trials.append(fake_trial('S1', m, 1001, 'success', 30.0, [base, base + 4]))
            trials.append(fake_trial('S1', m, 1002, 'timeout', 99.0, [base]))
        write_run(self.root, trials)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tables_and_csv(self):
        out = os.path.join(self.root, 'report')
        analyze.main(['--results', self.root, '--out', out])
        tex = open(os.path.join(out, 'tables.tex'), encoding='utf-8').read()
        self.assertIn('\\label{tab:静态仿真结果}', tex)
        self.assertIn('\\label{tab:动态仿真结果}', tex)
        self.assertIn('\\label{tab:仿真实时性}', tex)
        self.assertIn('66.7', tex)
        # T: successes only (20, 30) -> 25.00, never the timeout's 99
        self.assertIn('$25.00\\pm 5.00$', tex)
        self.assertNotIn('99', tex)
        # row order 本文方法, AABB-MPC, DWA
        self.assertLess(tex.index('本文方法'), tex.index('AABB-MPC'))
        self.assertLess(tex.index('AABB-MPC'), tex.index('DWA'))
        self.assertIn('超时 1 次', tex)
        with open(os.path.join(out, 'all_trials.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        with open(os.path.join(out, 'summary_S1.csv'), encoding='utf-8') as f:
            s = {r['method']: r for r in csv.DictReader(f)}
        self.assertAlmostEqual(float(s['neupan']['SR']), 66.7, places=1)
        self.assertAlmostEqual(float(s['neupan']['T_mean']), 25.0)
        self.assertTrue(os.path.exists(os.path.join(out, 'timing.csv')))

    def test_crash_row_does_not_poison_group(self):
        crash = fake_trial('S1', 'aabb', 1000, 'crash', NAN, [1.0])
        crash.update(e_g=NAN, d_min_np=NAN, d_min_body=NAN, plan_ms_all=[], t_avg=NAN,
                     t95=NAN, t_max=NAN, r_over=NAN, n_cycles=0)
        ok = fake_trial('S1', 'aabb', 1001, 'success', 20.0, [5.0, 7.0], d_min=0.25)
        write_run_dir = tempfile.TemporaryDirectory()
        try:
            write_run(write_run_dir.name, [crash, ok])
            out = os.path.join(write_run_dir.name, 'report')
            analyze.main(['--results', write_run_dir.name, '--out', out])
            with open(os.path.join(out, 'summary_S1.csv'), encoding='utf-8') as f:
                s = list(csv.DictReader(f))[0]
            self.assertAlmostEqual(float(s['d_min_body_min']), 0.25)
            self.assertAlmostEqual(float(s['SR']), 50.0)
            self.assertAlmostEqual(float(s['t_avg']), 6.0)
        finally:
            write_run_dir.cleanup()

    def test_representative_trial(self):
        rows = [dict(trial_id='a', success=True, T=10.0, seed=1),
                dict(trial_id='b', success=True, T=30.0, seed=2),
                dict(trial_id='c', success=True, T=20.0, seed=3),
                dict(trial_id='d', success=False, T=5.0, seed=4)]
        rep, failed = analyze.representative(rows)
        self.assertEqual((rep['trial_id'], failed), ('c', False))
        rep, failed = analyze.representative([rows[3], dict(rows[3], trial_id='e', seed=9)])
        self.assertEqual((rep['trial_id'], failed), ('e', True))


class FigureTest(unittest.TestCase):
    def test_figures_from_csv(self):
        with tempfile.TemporaryDirectory() as root:
            tr = fake_trial('S1', 'neupan', 1000, 'success', 5.0, [10.0, 12.0])
            tr['meta']['world'] = 'static_scene'
            write_run(root, [tr])
            tid = tr['trial_id']
            with open(os.path.join(root, 'trials', tid + '.csv'), 'w') as f:
                f.write('t,x,y,yaw,gap_np,gap_body,nearest,dist_goal\n')
                for i in range(10):
                    f.write('%.2f,%.2f,0,0,1,1,none,%.2f\n' % (i * .5, i * .1, 1 - i * .1))
            with open(os.path.join(root, 'trials', tid + '_cmd.csv'), 'w') as f:
                f.write('t,vx,vy,w\n0.1,0.3,0,0\n0.6,0.3,0.1,0\n')
            analyze.main(['--results', root])
            rep = os.path.join(root, 'report')
            for name in ('traj_S1_p1.pdf', 'ts_S1_p1.pdf', 'plan_time_box.pdf'):
                self.assertGreater(os.path.getsize(os.path.join(rep, name)), 1000, name)


if __name__ == '__main__':
    unittest.main()
