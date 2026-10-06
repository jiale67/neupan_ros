"""Fix F1: NeuPAN's first control must be rate-limited against the last executed command.

Builds the real planner from the V550 yaml + DUNE checkpoint (exactly like neupan_core) and
checks the first-step constraint `|u_0 - u_prev| <= max_acce * ctrl_period` (omni3: translational
norm and |w| separately). No ROS needed.
"""
import copy
import os
import tempfile
import unittest

import numpy as np
import yaml

from neupan import neupan

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CFG = os.path.join(_PKG, 'example', 'gazebo_V550', 'config', 'neupan_planner_V550_mec.yaml')
CKPT = os.path.join(_PKG, 'example', 'gazebo_V550', 'pretrain_V550_mec', 'model_5000.pth')

CTRL_PERIOD = 0.1
DV_MAX = 1.0 * CTRL_PERIOD   # max_acce[0] * ctrl_period
DW_MAX = 2.0 * CTRL_PERIOD   # max_acce[2] * ctrl_period
TOL = 1e-4

# First action from rest (origin, yaw 0, goal (5,0), no obstacles) computed by the planner
# BEFORE fix F1 (upstream behaviour, NeuPAN commit 517949f). Disabled-by-default must keep it.
PRE_F1_FIRST_ACTION = np.array([[2.9997194e-01], [0.0], [-4.0232538e-21]])


def _build(config_file=CFG):
    return neupan.init_from_yaml(config_file, pan={'dune_checkpoint': CKPT})


def _build_with_robot(robot_overrides=None, drop=()):
    """Planner from a copy of the V550 yaml with robot-section keys overridden / removed."""
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for k in drop:
        cfg['robot'].pop(k, None)
    cfg['robot'].update(robot_overrides or {})
    fd, path = tempfile.mkstemp(suffix='.yaml')
    try:
        with os.fdopen(fd, 'w') as f:
            yaml.safe_dump(cfg, f)
        return _build(path)
    finally:
        os.remove(path)


def _start(planner, start, goal):
    planner.update_initial_path_from_goal(start, goal)
    planner.reset()


def _wall(xs, y):
    return np.vstack([xs, np.full_like(xs, y)])


class TestRateLimit(unittest.TestCase):

    def test_v550_yaml_enables_limit(self):
        p = _build()
        self.assertTrue(p.robot.first_step_rate_limit)
        self.assertAlmostEqual(p.robot.ctrl_period, CTRL_PERIOD)

    def test_first_control_respects_rate_limit_from_rest(self):
        p = _build()
        s = np.array([[0.0], [0.0], [0.0]])
        _start(p, s, np.array([[5.0], [0.0], [0.0]]))
        a, info = p(s, None, None, u_prev=np.zeros((3, 1)))
        self.assertEqual(a.shape, (3, 1))
        self.assertLessEqual(np.hypot(a[0, 0], a[1, 0]), DV_MAX + TOL)
        self.assertLessEqual(abs(a[2, 0]), DW_MAX + TOL)
        # it does move: the window is used, not collapsed to zero
        self.assertGreater(a[0, 0], 0.5 * DV_MAX)

    def test_first_control_respects_rate_limit_nonzero_u_prev(self):
        p = _build()
        s = np.array([[0.0], [0.0], [0.0]])
        _start(p, s, np.array([[5.0], [0.0], [0.0]]))
        u_prev = np.array([[0.0], [0.3], [-0.5]])   # moving sideways and turning: far from optimum
        a, _ = p(s, None, None, u_prev=u_prev)
        d = a - u_prev
        self.assertLessEqual(np.hypot(d[0, 0], d[1, 0]), DV_MAX + TOL)
        self.assertLessEqual(abs(d[2, 0]), DW_MAX + TOL)

    def test_sharp_turn_rate_limited(self):
        # Goal at 90 deg to the current heading (yaw reference jumps by pi/2) with a wall
        # across the path: unconstrained this run jumps |dv| 0.30 m/s and |dw| 1.0 rad/s per cycle.
        p = _build()
        s = np.array([[0.0], [0.0], [0.0]])
        _start(p, s, np.array([[0.0], [5.0], [0.0]]))
        pts = _wall(np.linspace(-0.6, 0.15, 16), 0.8)
        u_prev = np.zeros((3, 1))
        dv, dw = [], []
        for _ in range(40):
            a, info = p(s, pts, None, u_prev=u_prev)
            d = a - u_prev
            dv.append(float(np.hypot(d[0, 0], d[1, 0])))
            dw.append(float(abs(d[2, 0])))
            s = s + a * CTRL_PERIOD      # world-frame (vx, vy, w) integrated over one cycle
            u_prev = a.copy()
        self.assertLessEqual(max(dv), DV_MAX + TOL, dv)
        self.assertLessEqual(max(dw), DW_MAX + TOL, dw)
        # the scenario is demanding: the robot actually turned and moved
        self.assertGreater(s[2, 0], 0.5)
        self.assertGreater(s[1, 0], 0.1)

    def test_u_prev_defaults_to_last_returned_action(self):
        p = _build()
        s = np.array([[0.0], [0.0], [0.0]])
        _start(p, s, np.array([[5.0], [0.0], [0.0]]))
        prev = np.zeros((3, 1))
        for _ in range(5):
            a, _ = p(s, None, None)              # no u_prev -> planner's own last action
            d = a - prev
            self.assertLessEqual(np.hypot(d[0, 0], d[1, 0]), DV_MAX + TOL)
            self.assertLessEqual(abs(d[2, 0]), DW_MAX + TOL)
            s = s + a * CTRL_PERIOD
            prev = a
        self.assertGreater(prev[0, 0], 2.5 * DV_MAX)   # it ramped up over the cycles
        p.reset()                                    # reset -> last action back to zeros
        a, _ = p(s, None, None)
        self.assertLessEqual(np.hypot(a[0, 0], a[1, 0]), DV_MAX + TOL)

    def test_arrive_zero_becomes_next_u_prev(self):
        p = _build()
        s = np.array([[0.0], [0.0], [0.0]])
        _start(p, s, np.array([[5.0], [0.0], [0.0]]))
        for _ in range(3):
            a, _ = p(s, None, None)
            s = s + a * CTRL_PERIOD
        p._last_action = np.array([[0.3], [0.0], [0.0]])
        # check_arrive also needs the path index near the end; closest_point only searches
        # ind_range points ahead per call, so place the index as if the path had been driven.
        p.ipath.point_index = len(p.ipath.cur_curve) - 1
        goal_state = np.array([[5.0], [0.0], [0.0]])
        a, info = p(goal_state, None, None)          # at goal -> arrive -> zeros executed
        self.assertTrue(info['arrive'])
        self.assertTrue(np.array_equal(a, np.zeros((3, 1))))
        self.assertTrue(np.array_equal(p._last_action, np.zeros((3, 1))))
        # the planner's own last action (zeros) is what the next call is limited against
        p.ipath.point_index = 0
        p.ipath.arrive_flag = False
        p.info['arrive'] = False
        a, info = p(np.array([[0.0], [0.0], [0.0]]), None, None)
        self.assertFalse(info['arrive'])
        self.assertLessEqual(np.hypot(a[0, 0], a[1, 0]), DV_MAX + TOL)
        self.assertLessEqual(abs(a[2, 0]), DW_MAX + TOL)

    def test_disabled_by_default(self):
        s = np.array([[0.0], [0.0], [0.0]])
        g = np.array([[5.0], [0.0], [0.0]])
        absent = _build_with_robot(drop=('first_step_rate_limit', 'ctrl_period'))
        self.assertFalse(absent.robot.first_step_rate_limit)
        _start(absent, s, g)
        a_absent, _ = absent(s, None, None)
        # identical to the pre-F1 planner output
        np.testing.assert_allclose(a_absent, PRE_F1_FIRST_ACTION, rtol=0, atol=1e-6)
        # unconstrained from rest the first step exceeds the 0.1 m/s window: the switch matters
        self.assertGreater(np.hypot(a_absent[0, 0], a_absent[1, 0]), DV_MAX + 0.1)

        # u_prev is ignored when disabled. Fresh planner: reset() does not clear PAN's
        # iteration state (pan.current_nom_values), so a reused planner differs anyway.
        ignored = _build_with_robot(drop=('first_step_rate_limit', 'ctrl_period'))
        _start(ignored, s, g)
        a_ignored, _ = ignored(s, None, None, u_prev=np.array([[0.0], [-0.5], [0.8]]))
        np.testing.assert_array_equal(a_ignored, a_absent)

        # explicit false == absent
        off = _build_with_robot({'first_step_rate_limit': False})
        _start(off, s, g)
        a_off, _ = off(s, None, None)
        np.testing.assert_array_equal(a_off, a_absent)

    def test_validation(self):
        with self.assertRaises(ValueError):
            _build_with_robot({'first_step_rate_limit': True, 'ctrl_period': 0.0})
        with self.assertRaises(ValueError):
            _build_with_robot({'first_step_rate_limit': True, 'ctrl_period': -0.1})
        with self.assertRaises(ValueError):
            _build_with_robot({'first_step_rate_limit': True}, drop=('ctrl_period',))
        with self.assertRaises(ValueError):   # omni3 still needs 3 bounds
            _build_with_robot({'first_step_rate_limit': True, 'ctrl_period': 0.1,
                               'max_acce': [1.0, 1.0]})


if __name__ == '__main__':
    unittest.main()
