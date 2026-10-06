import os
import unittest
import numpy as np
from neupan_exp.world import ObstacleSpec
from neupan_exp.world import parse_world
from neupan_exp.truth import ObstacleTruth
from neupan_exp.geometry import footprint_polygon, BODY_RECT


class TestTruth(unittest.TestCase):
    def test_velocity_and_labels(self):
        spec = ObstacleSpec('d1', 'box', (0.25, 0.2), 0, 0, 0, True)
        tr = ObstacleTruth([spec], window=0.1)
        self.assertEqual(tuple(tr.velocity('d1')), (0.0, 0.0))
        for k in range(11):
            tr.update(['d1', 'unknown'], [(0.19 * k * 0.01, 0, 0), (5, 5, 0)], k * 0.01)
        self.assertTrue(np.allclose(tr.velocity('d1'), (0.19, 0), atol=1e-6))
        pts = np.array([[0.1 + 0.019, 1.0], [0.0, 0.0]])
        self.assertEqual(list(tr.point_labels(pts)), [0, -1])
        v = tr.point_velocities(pts)
        self.assertTrue(np.allclose(v[:, 0], (0.19, 0)) and np.allclose(v[:, 1], 0))

    def test_multi_collision_dynamic_rejected(self):
        specs = [ObstacleSpec('m/0', 'box', (0.2, 0.2), 0, 0, 0, True, ((0, 0), (1, 0))),
                 ObstacleSpec('m/1', 'box', (0.2, 0.2), 1, 0, 0, True, ((0, 0), (1, 0)))]
        with self.assertRaisesRegex(ValueError, 'm'):
            ObstacleTruth(specs)

    def test_offset_dynamic_rejected(self):
        s = ObstacleSpec('o', 'box', (0.2, 0.2), 0.5, 0, 0, True, ((0, 0), (1, 0)))
        with self.assertRaisesRegex(ValueError, 'o'):
            ObstacleTruth([s])

    def test_static_multi_collision_ok(self):
        specs = [ObstacleSpec('w/0', 'box', (0.2, 0.2), 0, 0, 0, False),
                 ObstacleSpec('w/1', 'box', (0.2, 0.2), 1, 0, 0, False)]
        tr = ObstacleTruth(specs)
        tr.update(['w'], [(9, 9, 0)], 0.0)
        self.assertEqual(tr.point_labels(np.array([[1.0], [0.0]])).tolist(), [-1])

    def test_real_dynamic_world(self):
        w = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                         'example', 'gazebo_V550', 'world', 'dynamic_scene.world')
        tr = ObstacleTruth(parse_world(w))
        self.assertTrue(any(s.dynamic for s in tr.specs))

    def test_collision_gap(self):
        st = ObstacleSpec('w', 'box', (0.5, 0.2), 0, 0, 0, False)
        tr = ObstacleTruth([st])
        fp = footprint_polygon(0, -0.4, 0, *BODY_RECT)
        gap, name = tr.min_gap(fp)
        self.assertAlmostEqual(gap, 0.1777, places=4)
        self.assertEqual(name, 'w')


if __name__ == '__main__':
    unittest.main()
