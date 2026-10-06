import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from neupan_exp.geometry import (NP_RECT, BODY_RECT, rect_polygon, footprint_polygon,
                                 polygon_gap, circle_polygon_gap, obstacle_gap,
                                 points_in_obstacle)
from neupan_exp.world import ObstacleSpec


class TestGeometry(unittest.TestCase):
    def test_polygon_gap(self):
        A = rect_polygon(0, 0, 0, 1, 1)
        self.assertAlmostEqual(polygon_gap(A, rect_polygon(2, 0, 0, 1, 1)), 1.0, places=9)
        self.assertAlmostEqual(polygon_gap(A, rect_polygon(0.8, 0, 0, 1, 1)), -0.2, places=9)
        self.assertAlmostEqual(polygon_gap(A, rect_polygon(2, 2, 0, 1, 1)), np.hypot(1, 1), places=9)

    def test_circle_gap(self):
        A = rect_polygon(0, 0, 0, 1, 1)
        self.assertAlmostEqual(circle_polygon_gap(2, 0, 0.5, A), 1.0, places=9)
        self.assertAlmostEqual(circle_polygon_gap(0.2, 0, 0.5, A), -0.8, places=9)
        fp = footprint_polygon(0, 0, 0, *NP_RECT)
        self.assertAlmostEqual(circle_polygon_gap(0.5, 0, 1e-9, fp), 0.37, places=6)
        self.assertAlmostEqual(circle_polygon_gap(1, 1, 1e-9, fp), 1.235823, places=5)

    def test_footprint_rotation(self):
        fp = footprint_polygon(1, 2, np.pi / 2, *BODY_RECT)
        self.assertEqual(fp.shape, (4, 2))
        self.assertAlmostEqual(fp[:, 0].min(), 1 - 0.1223, places=9)
        self.assertAlmostEqual(fp[:, 1].max(), 2 + 0.1232, places=9)

    def test_obstacle_gap_current_pose(self):
        fp = footprint_polygon(0, 0, 0, *NP_RECT)
        box = ObstacleSpec('b', 'box', (0.5, 0.5), 9, 9, 0, True)
        self.assertAlmostEqual(obstacle_gap(fp, box, 1.0, 0, 0), 1.0 - 0.25 - 0.13, places=9)
        cyl = ObstacleSpec('c', 'cylinder', (0.5,), 9, 9, 0, False)
        self.assertAlmostEqual(obstacle_gap(fp, cyl, 1.0, 0, 0), 1.0 - 0.5 - 0.13, places=9)

    def test_points_in_obstacle(self):
        pts = np.array([[0.0, 0.6, 0.0, 2.0], [0.0, 0.0, 0.6, 2.0]])
        box = ObstacleSpec('b', 'box', (1.0, 1.0), 0, 0, 0, False)
        self.assertEqual(points_in_obstacle(pts, box, 0, 0, 0, 0.0).tolist(), [True, False, False, False])
        self.assertEqual(points_in_obstacle(pts, box, 0, 0, 0, 0.2).tolist(), [True, True, True, False])
        cyl = ObstacleSpec('c', 'cylinder', (0.5,), 0, 0, 0, False)
        self.assertEqual(points_in_obstacle(pts, cyl, 0, 0, 0, 0.2).tolist(), [True, True, True, False])
        self.assertEqual(points_in_obstacle(pts, cyl, 2, 2, 0, 0.0).tolist(), [False, False, False, True])


if __name__ == '__main__':
    unittest.main()
