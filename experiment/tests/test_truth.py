import unittest
import numpy as np
from neupan_exp.world import ObstacleSpec
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

    def test_multi_collision_prefix(self):
        specs = [ObstacleSpec('m/0', 'box', (0.2, 0.2), 0, 0, 0, True),
                 ObstacleSpec('m/1', 'box', (0.2, 0.2), 1, 0, 0, True)]
        tr = ObstacleTruth(specs)
        tr.update(['m'], [(2, 0, 0)], 0.0)
        tr.update(['m'], [(2.05, 0, 0)], 0.1)
        self.assertTrue(np.allclose(tr.velocity('m/1'), (0.5, 0)))

    def test_collision_gap(self):
        st = ObstacleSpec('w', 'box', (0.5, 0.2), 0, 0, 0, False)
        tr = ObstacleTruth([st])
        fp = footprint_polygon(0, -0.4, 0, *BODY_RECT)
        gap, name = tr.min_gap(fp)
        self.assertAlmostEqual(gap, 0.1777, places=4)
        self.assertEqual(name, 'w')


if __name__ == '__main__':
    unittest.main()
