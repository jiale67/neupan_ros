"""Fix F1 review P-b: executed_planner_action (no ROS master needed)."""
import os
import sys
import types
import unittest

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, _SRC)

from neupan_core import neupan_core  # noqa: E402


def stub(stop=False, arrive=False, dim=3):
    return types.SimpleNamespace(stop=stop, arrive=arrive, last_cmd=np.zeros((dim, 1)))


class TestExecutedPlannerAction(unittest.TestCase):
    def test_none_is_zero(self):
        out = neupan_core.executed_planner_action(stub(), None)
        np.testing.assert_array_equal(out, np.zeros((3, 1)))

    def test_stop_and_arrive_are_zero(self):
        vel = np.array([[0.3], [0.1], [0.2]])
        for kw in (dict(stop=True), dict(arrive=True)):
            np.testing.assert_array_equal(
                neupan_core.executed_planner_action(stub(**kw), vel), np.zeros((3, 1)))

    def test_otherwise_copy_of_action(self):
        vel = np.array([[0.3], [0.1], [0.2]])
        out = neupan_core.executed_planner_action(stub(), vel)
        np.testing.assert_array_equal(out, vel)
        out[0, 0] = 9.0
        self.assertEqual(vel[0, 0], 0.3)           # a copy, not a view
        self.assertEqual(out.shape, (3, 1))

    def test_flat_input_reshaped(self):
        out = neupan_core.executed_planner_action(stub(), [0.1, 0.2, 0.3])
        self.assertEqual(out.shape, (3, 1))


if __name__ == '__main__':
    unittest.main()
