"""Fix F2: analysis (pair-matched comparison, uniform e-stops, J split, accel table, flags)."""
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


def row(scene, method, pair, seed, success=True, T=10.0, L=3.0, J_v=1.0, J_w=0.5):
    return dict(scene=scene, method=method, pair=pair, seed=seed, success=success, T=T, L=L,
                J_v=J_v, J_w=J_w, outcome='success' if success else 'timeout')


class TestMatched(unittest.TestCase):
    def rows(self):
        out = []
        for m in ('neupan', 'aabb', 'dwa'):
            for seed in (1, 2, 3):
                out.append(row('S1', m, 1, seed, T=10.0 + seed + (5.0 if m == 'dwa' else 0.0)))
        out[4]['success'] = False        # aabb seed 2 fails
        out[4]['outcome'] = 'timeout'
        return out

    def test_failed_seed_excluded_for_all_methods(self):
        ms = analyze.matched_sets(self.rows())
        self.assertEqual(ms['S1'], {(1, 1), (1, 3)})

    def test_matched_summary_means_over_matched_only(self):
        rows = self.rows()
        ms = analyze.matched_sets(rows)
        s = analyze.matched_summary(rows, ms)
        a = s[('S1', 'neupan')]
        self.assertEqual(a['n_matched'], 2)
        self.assertAlmostEqual(a['T_mean'], (11.0 + 13.0) / 2)
        self.assertAlmostEqual(a['T_std'], math.sqrt(2.0))        # sample std of 11, 13
        self.assertAlmostEqual(s[('S1', 'dwa')]['T_mean'], (16.0 + 18.0) / 2)

    def test_scenes_independent_and_empty_when_method_missing(self):
        rows = self.rows() + [row('S2', 'neupan', 1, 1), row('S2', 'aabb', 1, 1)]
        ms = analyze.matched_sets(rows)
        self.assertEqual(ms['S2'], {(1, 1)})     # methods present in the scene (dwa absent)
        rows2 = [r for r in rows if r['scene'] == 'S1']
        for r in rows2:
            if r['method'] == 'dwa':
                r['success'] = False
        self.assertEqual(analyze.matched_sets(rows2)['S1'], set())
        s = analyze.matched_summary(rows2, analyze.matched_sets(rows2))
        self.assertEqual(s[('S1', 'neupan')]['n_matched'], 0)
        self.assertTrue(math.isnan(s[('S1', 'neupan')]['T_mean']))


class TestSanity(unittest.TestCase):
    BASE = dict(status_gap_max=0.1, n_cycles=100, goal_subscribers=2, late_start=False,
                attempts=1, rtf=1.0)

    def flags(self, **kw):
        d = dict(self.BASE)
        d.update(kw)
        return analyze.sanity_flags(d)

    def test_clean(self):
        self.assertEqual(self.flags(), [])

    def test_each_flag(self):
        self.assertIn('status_gap', self.flags(status_gap_max=0.6))
        self.assertNotIn('status_gap', self.flags(status_gap_max=0.5))
        self.assertIn('no_cycles', self.flags(n_cycles=0))
        self.assertIn('goal_subs<2', self.flags(goal_subscribers=1))
        self.assertIn('late_start', self.flags(late_start=True))
        self.assertIn('retried', self.flags(attempts=2))
        self.assertIn('rtf<0.9', self.flags(rtf=0.85))

    def test_missing_fields_do_not_flag(self):
        self.assertEqual(analyze.sanity_flags(dict(n_cycles=5)), [])
        self.assertEqual(self.flags(rtf='', goal_subscribers='', attempts='', late_start=''), [])
        self.assertEqual(self.flags(rtf=NAN), [])


def write_run(root, specs):
    """specs: list of dict(tid, scene, method, pair, seed, outcome, json_extra, cmd=[(t,vx,vy,w)],
    status_t=[...])"""
    os.makedirs(os.path.join(root, 'trials'))
    for s in specs:
        tid = s['tid']
        meta = {'method': s['method'], 'scene': s['scene'], 'pair_id': s['pair'],
                'seed': s['seed'], 'start': [0, 0, 0], 'goal': [1, 0], 'world': 'static_scene'}
        meta.update(s.get('meta', {}))
        d = {'outcome': s['outcome'], 'success': s['outcome'] == 'success',
             'T': s.get('T', 10.0), 'L': 3.0, 'e_g': 0.05, 'd_min_np': 0.3, 'd_min_body': 0.3,
             'collisions': 0, 'estops': s.get('estops_planner', 0), 'J_du': 2.0, 't_avg': 10.0,
             't95': 12.0, 't_max': 13.0, 'r_over': 0.0, 'n_cycles': 3,
             'plan_ms_all': [10.0, 10.0, 10.0], 'end_reason': 'arrived',
             'trial_id': tid, 'goal': [1.0, 0.0], 'meta': meta}
        d.update(s.get('json_extra', {}))
        with open(os.path.join(root, 'trials', tid + '.json'), 'w') as f:
            json.dump(d, f)
        if 'cmd' in s:
            with open(os.path.join(root, 'trials', tid + '_cmd.csv'), 'w') as f:
                f.write('t,vx,vy,w\n')
                for r in s['cmd']:
                    f.write('%.4f,%.6f,%.6f,%.6f\n' % r)
        if 'status_t' in s:
            with open(os.path.join(root, 'trials', tid + '_status.csv'), 'w') as f:
                f.write('t,plan_ms,min_dist,stop,arrive\n')
                for t in s['status_t']:
                    f.write('%.4f,10.0,1.0,0,0\n' % t)


def cmds(n, dt=0.1, v=0.3):
    return [(i * dt, v, 0.0, 0.0) for i in range(n)]


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        # neupan: leading zeros, one e-stop before arrival, zero after arrival ignored
        c = [(0.0, 0, 0, 0), (0.1, 0, 0, 0), (0.2, 0.3, 0, 0), (0.3, 0.3, 0.1, 0.2),
             (0.4, 0, 0, 0), (0.5, 0.3, 0, 0), (0.6, 0.3, 0, 0), (0.7, 0, 0, 0)]
        specs = [
            dict(tid='S1_neupan_p1_s1', scene='S1', method='neupan', pair=1, seed=1,
                 outcome='success', cmd=c, status_t=[0.0, 0.05],
                 json_extra=dict(T=0.6, t_arrive=0.6, settle_s=1.0, e_g=0.04, e_g_arrive=0.29,
                                 rtf=0.8, wall_s=12.0, estops=5),
                 meta=dict(goal_subscribers=2, late_start=False, attempt=1)),
            dict(tid='S1_aabb_p1_s1', scene='S1', method='aabb', pair=1, seed=1,
                 outcome='success', cmd=cmds(8), status_t=[0.1 * i for i in range(8)],
                 json_extra=dict(T=0.7, t_arrive=0.7, settle_s=0.5, e_g=0.1, e_g_arrive=0.28,
                                 rtf=1.0, wall_s=1.0)),
            dict(tid='S1_dwa_p1_s1', scene='S1', method='dwa', pair=1, seed=1,
                 outcome='timeout', cmd=cmds(8), status_t=[0.1 * i for i in range(8)],
                 json_extra=dict(T=99.0, t_arrive=None, end_reason='timeout')),
        ]
        write_run(self.root, specs)
        self.out = os.path.join(self.root, 'report')
        analyze.main(['--results', self.root, '--out', self.out, '--no-figures'])
        with open(os.path.join(self.out, 'tables.tex'), encoding='utf-8') as f:
            self.tex = f.read()

    def rows(self, name):
        with open(os.path.join(self.out, name), encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def test_uniform_estops_and_planner_kept(self):
        r = {x['trial_id']: x for x in self.rows('all_trials.csv')}
        n = r['S1_neupan_p1_s1']
        self.assertEqual(int(n['estops']), 1)             # zero at t=0.4 only
        self.assertEqual(int(n['estops_planner']), 5)

    def test_j_split_up_to_arrival(self):
        r = {x['trial_id']: x for x in self.rows('all_trials.csv')}
        n = r['S1_neupan_p1_s1']
        # cmds up to t=0.6: 0,0,.3,(.3,.1,.2),0,.3,.3 (zero at 0.7 excluded)
        # dv^2: (.3)^2 + (.1)^2 + (.3)^2+(.1)^2 + .3^2 + 0 = 0.09+0.01+0.09+0.01+0.09
        self.assertAlmostEqual(float(n['J_v']), 0.29, places=6)
        # dw^2: .04 (0.2) + .04 (back to 0) = 0.08
        self.assertAlmostEqual(float(n['J_w']), 0.08, places=6)

    def test_settled_eg_and_new_columns(self):
        r = {x['trial_id']: x for x in self.rows('all_trials.csv')}
        n = r['S1_neupan_p1_s1']
        self.assertAlmostEqual(float(n['e_g']), 0.04)
        self.assertAlmostEqual(float(n['e_g_arrive']), 0.29)
        self.assertAlmostEqual(float(n['rtf']), 0.8)
        self.assertIn('rtf<0.9', n['flags'])
        self.assertIn('status_gap', n['flags'])           # tail gap 0.05 -> t_arrive 0.6

    def test_sanity_file(self):
        txt = open(os.path.join(self.out, 'sanity.txt'), encoding='utf-8').read()
        self.assertIn('S1_neupan_p1_s1', txt)
        self.assertNotIn('S1_aabb_p1_s1', txt)
        self.assertNotIn('S1_dwa_p1_s1', txt)

    def test_accel_csv_and_table(self):
        self.assertIn('\\label{tab:执行加速度}', self.tex)
        acc = {(x['scene'], x['method']): x for x in self.rows('accel.csv')}
        a = acc[('S1', 'aabb')]
        self.assertAlmostEqual(float(a['peak_lin']), 0.0, places=6)
        n = acc[('S1', 'neupan')]
        # non-zero-touching pairs up to t=0.6: (.3,0,0)->(.3,.1,.2): lin 1.0, ang 2.0 ; (.3..)->(.3) 0
        self.assertAlmostEqual(float(n['peak_lin']), 1.0, places=6)
        self.assertAlmostEqual(float(n['peak_ang']), 2.0, places=6)
        self.assertEqual(int(n['n_zero_trans']), 3)       # 0->.3, .3->0, 0->.3 before t=0.6

    def test_matched_table_and_csv(self):
        self.assertIn('\\label{tab:配对成功比较}', self.tex)
        m = self.rows('summary_matched_S1.csv')
        self.assertEqual({x['method'] for x in m}, {'neupan', 'aabb', 'dwa'})
        self.assertEqual(int(m[0]['n_matched']), 0)       # dwa timed out -> nothing matched

    def test_notes_and_columns(self):
        self.assertIn('均值$\\pm$样本标准差', self.tex)
        self.assertIn('$J_{v}$', self.tex)
        self.assertIn('$J_{\\omega}$', self.tex)
        self.assertNotIn('$J_{\\Delta u}$', self.tex)
        self.assertIn('急停', self.tex)
        self.assertIn('合并', self.tex)

    def test_timing_has_rtf(self):
        t = {(x['scene'], x['method']): x for x in self.rows('timing.csv')}
        self.assertAlmostEqual(float(t[('S1', 'neupan')]['rtf_mean']), 0.8)
        self.assertEqual(t[('S1', 'dwa')]['rtf_mean'], 'nan')

    def test_single_trial_std_is_dashes(self):
        # one success per cell -> sample std undefined -> printed '--'
        self.assertIn('--', self.tex)
        self.assertNotIn('\\pm 0.00', self.tex)


class TestOldData(unittest.TestCase):
    def test_old_json_without_new_fields(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, [dict(tid='S2_neupan_p1_s1', scene='S2', method='neupan', pair=1,
                                  seed=1, outcome='success', estops_planner=2)])
            out = os.path.join(root, 'report_v2')
            self.assertEqual(analyze.main(['--results', root, '--out', out, '--no-figures']), 0)
            r = list(csv.DictReader(open(os.path.join(out, 'all_trials.csv'), encoding='utf-8')))[0]
            self.assertEqual(int(r['estops']), 2)         # no _cmd.csv -> planner count
            self.assertEqual(r['e_g'], '0.05')
            tex = open(os.path.join(out, 'tables.tex'), encoding='utf-8').read()
            self.assertIn('tab:执行加速度', tex)
            self.assertIn('到达时刻', tex)                  # e_g fallback note


if __name__ == '__main__':
    unittest.main()
