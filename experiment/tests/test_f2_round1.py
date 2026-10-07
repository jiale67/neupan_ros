"""Fix F2 review round 1: per-cycle accel view, legacy e_g note, provenance scope/digest,
estops_fallback flag, arrival-stop exclusion."""
import csv
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))

import analyze  # noqa: E402
import run_batch as rb  # noqa: E402
from neupan_exp.metrics import accel_samples, uniform_estops  # noqa: E402
from test_f2_analyze import write_run  # noqa: E402
from test_f2_runner import baseline  # noqa: E402
from test_runner import FakeClock, FakeHandle, make_trial, write_result  # noqa: E402


class TestStepSamples(unittest.TestCase):
    def test_per_command_steps_ignore_timestamps(self):
        # receive-time jitter: dt 0.02 then 0.18 around one nominal 0.1 s cycle
        t = [0.0, 0.02, 0.20]
        vx = [0.30, 0.34, 0.40]
        vy = [0.0, 0.03, 0.03]
        w = [0.0, 0.1, 0.15]
        a = accel_samples(t, vx, vy, w)
        np.testing.assert_allclose(a['d_lin'], [0.05, 0.06], atol=1e-9)
        np.testing.assert_allclose(a['d_ang'], [0.1, 0.05], atol=1e-9)

    def test_steps_skip_zero_touching_pairs_but_not_short_dt(self):
        a = accel_samples([0.0, 0.1, 0.1, 0.2], [0.0, 0.3, 0.35, 0.35], [0, 0, 0, 0],
                          [0, 0, 0, 0.2])
        # (0->.3) touches zero; (.3->.35) at dt 0 kept; (.35->.35,w .2) kept
        np.testing.assert_allclose(a['d_lin'], [0.05, 0.0], atol=1e-9)
        np.testing.assert_allclose(a['d_ang'], [0.0, 0.2], atol=1e-9)

    def test_t_end_applies(self):
        a = accel_samples([0.0, 0.1, 0.2], [0.1, 0.2, 0.4], [0, 0, 0], [0.1] * 3, t_end=0.1)
        self.assertEqual(len(a['d_lin']), 1)


class TestArrivalStop(unittest.TestCase):
    N, Z = (0.3, 0.0, 0.0), (0.0, 0.0, 0.0)

    def cnt(self, cmds, t_end, dt=0.1):
        t = [dt * i for i in range(len(cmds))]
        return uniform_estops(t, [c[0] for c in cmds], [c[1] for c in cmds],
                              [c[2] for c in cmds], t_end)

    def test_zero_run_ongoing_at_arrival_is_not_an_estop(self):
        # the planner's own arrival stop: zeros from t=0.3 on, arrival at t=0.5
        self.assertEqual(self.cnt([self.N] * 3 + [self.Z] * 3, t_end=0.5), 0)

    def test_earlier_estop_still_counted_with_arrival_stop(self):
        cmds = [self.N, self.Z, self.N, self.N, self.Z, self.Z]
        self.assertEqual(self.cnt(cmds, t_end=0.5), 1)

    def test_long_ongoing_zero_run_is_an_estop(self):
        cmds = [self.N] + [self.Z] * 15              # zeros from 0.1 to 1.5, arrival at 1.5
        self.assertEqual(self.cnt(cmds, t_end=1.5), 1)

    def test_boundary_one_second(self):
        cmds = [self.N] + [self.Z] * 11              # run starts 0.1, arrival 1.1 -> 1.0 s
        self.assertEqual(self.cnt(cmds, t_end=1.1), 0)

    def test_no_exclusion_without_arrival(self):
        self.assertEqual(self.cnt([self.N] * 3 + [self.Z] * 3, t_end=None), 1)


def cmd_rows(n, step_v=0.05, jitter=False):
    """n commands, vx rising by step_v each, nominal 0.1 s, optional receive-time jitter."""
    rows, t = [], 0.0
    for i in range(n):
        t += 0.1 if not jitter else (0.02 if i % 2 else 0.18)
        rows.append((t, 0.3 + step_v * i, 0.0, 0.0))
    return rows


class TestPerCycleTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        specs = []
        for m, jit in (('neupan', False), ('aabb', True)):
            specs.append(dict(tid='S1_%s_p1_s1' % m, scene='S1', method=m, pair=1, seed=1,
                              outcome='success', cmd=cmd_rows(6, jitter=jit),
                              json_extra=dict(t_arrive=99.0, settle_s=1.0, e_g=0.1)))
        write_run(self.tmp.name, specs)
        self.out = os.path.join(self.tmp.name, 'r')
        analyze.main(['--results', self.tmp.name, '--out', self.out, '--no-figures'])
        with open(os.path.join(self.out, 'accel.csv'), encoding='utf-8') as f:
            self.acc = {(r['scene'], r['method']): r for r in csv.DictReader(f)}
        with open(os.path.join(self.out, 'tables.tex'), encoding='utf-8') as f:
            self.tex = f.read()

    def test_nominal_cycle_view_is_independent_of_timestamp_jitter(self):
        for m in ('neupan', 'aabb'):
            a = self.acc[('S1', m)]
            self.assertAlmostEqual(float(a['nom_peak_lin']), 0.5, places=6)   # 0.05 / 0.1
            self.assertAlmostEqual(float(a['dv_max']), 0.05, places=6)
            self.assertAlmostEqual(float(a['nom_peak_ang']), 0.0, places=6)

    def test_receive_time_view_kept_in_csv_and_shows_the_jitter(self):
        raw = float(self.acc[('S1', 'aabb')]['peak_lin'])
        self.assertGreater(raw, 2.0)                  # 0.05 / 0.02 s
        self.assertAlmostEqual(float(self.acc[('S1', 'neupan')]['peak_lin']), 0.5, places=6)

    def test_table_uses_per_cycle_view_and_notes_jitter(self):
        blk = self.tex[self.tex.index('tab:执行加速度'):]
        blk = blk[:blk.index('\\end{table}')]
        self.assertIn('0.50', blk)
        self.assertNotIn('2.50', blk)                 # aabb's receive-time 0.05/0.02 s
        self.assertIn('时间戳', blk)                   # jitter = recorder timestamp effect
        self.assertIn('0.1', blk)                      # nominal control period named

    def test_docstring_not_stale(self):
        self.assertNotIn('(used in the table)', analyze.accel_summary.__doc__)


class TestLegacyNote(unittest.TestCase):
    NOTE = '无停稳记录'

    def run_tables(self, specs):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, specs)
            out = os.path.join(root, 'r')
            analyze.main(['--results', root, '--out', out, '--no-figures'])
            with open(os.path.join(out, 'tables.tex'), encoding='utf-8') as f:
                return f.read()

    def spec(self, method, outcome, extra, seed=1):
        return dict(tid='S1_%s_p1_s%d' % (method, seed), scene='S1', method=method, pair=1,
                    seed=seed, outcome=outcome, json_extra=extra)

    def test_new_format_timeout_does_not_trigger_note(self):
        tex = self.run_tables([
            self.spec('neupan', 'success', dict(t_arrive=5.0, settle_s=1.0, e_g=0.1)),
            self.spec('neupan', 'timeout', dict(t_arrive=None, settle_s=None, e_g=2.0,
                                                end_reason='timeout'), seed=2)])
        self.assertNotIn(self.NOTE, tex)

    def test_old_arrived_trial_without_key_triggers_note(self):
        tex = self.run_tables([self.spec('neupan', 'success', {})])
        self.assertIn(self.NOTE, tex)

    def test_old_timeout_only_does_not_trigger_note(self):
        tex = self.run_tables([self.spec('neupan', 'timeout', dict(end_reason='timeout'))])
        self.assertNotIn(self.NOTE, tex)


class TestFallbackFlag(unittest.TestCase):
    def test_flag(self):
        f = analyze.sanity_flags(dict(estops_src='planner', outcome='success'))
        self.assertIn('estops_fallback', f)
        self.assertNotIn('estops_fallback', analyze.sanity_flags(
            dict(estops_src='cmd', outcome='success')))
        self.assertNotIn('estops_fallback', analyze.sanity_flags(
            dict(estops_src='planner', outcome='crash')))

    def test_end_to_end_missing_cmd_csv_is_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, [dict(tid='S1_dwa_p1_s1', scene='S1', method='dwa', pair=1, seed=1,
                                  outcome='success')])
            out = os.path.join(root, 'r')
            analyze.main(['--results', root, '--out', out, '--no-figures'])
            with open(os.path.join(out, 'sanity.txt'), encoding='utf-8') as f:
                self.assertIn('estops_fallback', f.read())


class TestProvenanceScopeAndDigest(unittest.TestCase):
    def test_analysis_only_files_not_hashed(self):
        names = {os.path.basename(p) for p in rb.CODE_FILES}
        self.assertNotIn('analyze.py', names)
        self.assertNotIn('check_scenarios.py', names)
        for must in ('trial_recorder.py', 'run_batch.py', 'metrics.py', 'neupan_core.py',
                     'exp_trial.launch', 'scenarios.yaml'):
            self.assertIn(must, names)

    def test_digest_short_stable_and_sensitive(self):
        d = rb.code_digest(baseline())
        self.assertEqual(len(d), 12)
        int(d, 16)
        self.assertEqual(d, rb.code_digest(baseline()))
        b = baseline()
        b['sha256']['f1'] = 'other'
        self.assertNotEqual(d, rb.code_digest(b))
        b = baseline()
        b['heads']['NeuPAN'] = 'zzz'
        self.assertNotEqual(d, rb.code_digest(b))

    def test_digest_in_trial_meta(self):
        with tempfile.TemporaryDirectory() as out:
            def launcher(t, o, mj, lp):
                write_result(o, t, mj)
                return FakeHandle()
            chk = rb.make_code_check(baseline(), baseline)
            rb.run_one(make_trial(), out, launcher, FakeClock(), lambda x, y: None,
                       ready=lambda: True, poll=0.01, exit_grace=0, code_check=chk)
            with open(os.path.join(out, make_trial().trial_id + '.json')) as f:
                meta = json.load(f)['meta']
            self.assertEqual(meta['code_check'], 'ok')
            self.assertEqual(meta['code_digest'], rb.code_digest(baseline()))

    def test_code_check_returns_digest_and_plain_callable_still_works(self):
        self.assertEqual(rb.make_code_check(baseline(), baseline)(make_trial()),
                         rb.code_digest(baseline()))
        with tempfile.TemporaryDirectory() as out:
            def launcher(t, o, mj, lp):
                write_result(o, t, mj)
                return FakeHandle()
            rb.run_one(make_trial(), out, launcher, FakeClock(), lambda x, y: None,
                       ready=lambda: True, poll=0.01, exit_grace=0, code_check=lambda t: None)
            with open(os.path.join(out, make_trial().trial_id + '.json')) as f:
                meta = json.load(f)['meta']
            self.assertEqual(meta['code_check'], 'ok')
            self.assertNotIn('code_digest', meta)


class TestLaunchThrottle(unittest.TestCase):
    def test_model_states_throttle_100(self):
        p = os.path.join(HERE, '..', 'launch', 'exp_trial.launch')
        txt = open(p, encoding='utf-8').read()
        self.assertIn('messages /gazebo/model_states 100 /exp/model_states', txt)


if __name__ == '__main__':
    unittest.main()
