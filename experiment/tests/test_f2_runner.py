"""Fix F2: runner provenance check, manifest environment / git scope."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))

import run_batch as rb  # noqa: E402
from test_runner import FakeClock, FakeHandle, make_trial, write_result  # noqa: E402


def baseline():
    return {'heads': {'neupan_ros': 'aaa', 'NeuPAN': 'bbb', 'Autonomous_navigation_exploration': 'ccc'},
            'sha256': {'f1': 'h1', 'f2': 'h2'}}


class TestProvenance(unittest.TestCase):
    def test_identical_is_ok(self):
        self.assertEqual(rb.verify_provenance(baseline(), baseline()), [])

    def test_head_change_detected(self):
        cur = baseline()
        cur['heads']['NeuPAN'] = 'zzz'
        d = rb.verify_provenance(baseline(), cur)
        self.assertEqual(len(d), 1)
        self.assertIn('NeuPAN', d[0])

    def test_file_hash_change_detected(self):
        cur = baseline()
        cur['sha256']['f2'] = 'changed'
        d = rb.verify_provenance(baseline(), cur)
        self.assertEqual(len(d), 1)
        self.assertIn('f2', d[0])

    def test_code_check_aborts_with_clear_message(self):
        cur = baseline()
        cur['sha256']['f1'] = 'x'
        chk = rb.make_code_check(baseline(), lambda: cur)
        with self.assertRaises(rb.AbortBatch) as cm:
            chk(make_trial())
        self.assertIn('code changed', str(cm.exception))
        rb.make_code_check(baseline(), baseline)(make_trial())    # no raise

    def test_run_one_aborts_before_launch_and_records_ok(self):
        with tempfile.TemporaryDirectory() as out:
            launched = []

            def launcher(t, o, mj, lp):
                launched.append(1)
                write_result(o, t, mj)
                return FakeHandle()

            bad = rb.make_code_check(baseline(), lambda: dict(baseline(), heads={}))
            with self.assertRaises(rb.AbortBatch):
                rb.run_one(make_trial(), out, launcher, FakeClock(), lambda x, y: None,
                           ready=lambda: True, poll=0.01, exit_grace=0, code_check=bad)
            self.assertEqual(launched, [])
            ok = rb.make_code_check(baseline(), baseline)
            rb.run_one(make_trial(), out, launcher, FakeClock(), lambda x, y: None,
                       ready=lambda: True, poll=0.01, exit_grace=0, code_check=ok)
            with open(os.path.join(out, make_trial().trial_id + '.json')) as f:
                j = json.load(f)
            self.assertEqual(j['meta']['code_check'], 'ok')


class TestEnvironment(unittest.TestCase):
    def test_keys_with_fakes(self):
        def run(argv, **kw):
            assert kw.get('timeout') == 10
            return subprocess.CompletedProcess(argv, 0, stdout='Gazebo multi-robot simulator, version 11.9.0\nCopyright\n', stderr='')

        files = {'/proc/cpuinfo': 'processor\t: 0\nmodel name\t: Fake CPU @ 3GHz\n',
                 '/proc/meminfo': 'MemTotal:       16000000 kB\n',
                 '/proc/loadavg': '0.50 0.40 0.30 1/200 999\n'}
        env = rb.collect_environment(run=run, read_file=lambda p: files[p],
                                     version_fn=lambda n: {'torch': '1.13'}.get(n),
                                     threads_fn=lambda: 6, environ={'ROS_DISTRO': 'noetic'},
                                     nproc=12)
        self.assertEqual(env['cpu_model'], 'Fake CPU @ 3GHz')
        self.assertEqual(env['nproc'], 12)
        self.assertEqual(env['loadavg'], [0.5, 0.4, 0.3])
        self.assertEqual(env['mem_total_kb'], 16000000)
        self.assertEqual(env['ros_distro'], 'noetic')
        self.assertEqual(env['torch_num_threads'], 6)
        self.assertEqual(env['gzserver_version'], 'Gazebo multi-robot simulator, version 11.9.0')
        self.assertIn('python', env)
        for k in ('torch', 'cvxpy', 'cvxpylayers', 'scikit-learn', 'numpy', 'scipy'):
            self.assertIn(k, env['versions'])
        self.assertEqual(env['versions']['torch'], '1.13')
        self.assertIsNone(env['versions']['cvxpy'])

    def test_failures_tolerated(self):
        def run(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 10)

        def boom(*a):
            raise OSError('nope')

        env = rb.collect_environment(run=run, read_file=boom, version_fn=boom, threads_fn=boom,
                                     environ={}, nproc=4)
        self.assertIsNone(env['gzserver_version'])
        self.assertIsNone(env['cpu_model'])
        self.assertIsNone(env['torch_num_threads'])
        self.assertIsNone(env['ros_distro'])


class TestGitScope(unittest.TestCase):
    def test_scope_changes(self):
        lines = [' M experiment/a.py', '?? src/x.py', ' D src/old.py', ' M docs/readme.md',
                 'R  src/a -> src/b', ' M example/gazebo_V550/w.world', ' M example/other/q']
        got = rb.scope_changes(lines, ('src/', 'experiment/', 'example/gazebo_V550/'))
        self.assertEqual(got, [' M experiment/a.py', '?? src/x.py', 'R  src/a -> src/b',
                               ' M example/gazebo_V550/w.world'])

    def test_nondeleted_count(self):
        self.assertEqual(rb.count_nondeleted([' M a', ' D b', 'D  c', '?? d']), 2)

    def test_sha256_or_none(self):
        self.assertIsNone(rb.sha256_or_none('/nonexistent/file.so'))
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'abc')
            f.flush()
            self.assertEqual(rb.sha256_or_none(f.name), rb.sha256(f.name))


if __name__ == '__main__':
    unittest.main()
