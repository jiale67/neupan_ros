import unittest
import numpy as np
from neupan_exp.aabb import aabb_points


class TestAabb(unittest.TestCase):
    def test_empty(self):
        p, v = aabb_points(np.zeros((2, 0)), None, None)
        self.assertEqual(p.shape, (2, 0))
        self.assertEqual(v.shape, (2, 0))

    def test_l_shape_filled(self):
        a = np.arange(20) * 0.05
        L = np.hstack([np.vstack([a, np.zeros(20)]), np.vstack([np.zeros(20), a + 0.05])])
        pts, v = aabb_points(L, None, None)
        self.assertEqual(pts.shape[0], 2)
        self.assertTrue(np.all(v == 0))
        corner = np.array([[a.max()], [a.max() + 0.05]])
        self.assertLess(np.min(np.linalg.norm(pts - corner, axis=0)), 0.1)

    def test_deterministic(self):
        rng = np.random.RandomState(0)
        P = rng.rand(2, 50)
        a = aabb_points(P, None, None, eps=0.3)
        b = aabb_points(P, None, None, eps=0.3)
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))

    def test_bad_lengths(self):
        P = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            aabb_points(P, None, np.array([0, 1]))
        with self.assertRaises(ValueError):
            aabb_points(P, np.zeros((2, 2)), None)

    def test_dynamic_not_merged_with_static(self):
        wall = np.vstack([np.zeros(11), np.linspace(0, 1, 11)])
        dyn = np.array([[0.1, 0.1, 0.1], [0.4, 0.5, 0.6]])
        vel = np.hstack([np.zeros((2, 11)), np.tile([[0.2], [0.0]], (1, 3))])
        labels = np.array([-1] * 11 + [3] * 3)
        pts, v = aabb_points(np.hstack([wall, dyn]), vel, labels)
        moving = np.linalg.norm(v, axis=0) > 0
        self.assertTrue(moving.any())
        self.assertTrue(np.all(pts[0, moving] >= 0.09))
        self.assertTrue(np.all(v[:, pts[0] < 0.09] == 0))

    def test_noise_kept(self):
        P = np.array([[0.0, 5.0], [0.0, 5.0]])
        v = np.array([[1.0, 2.0], [0.0, 0.0]])
        pts, vv = aabb_points(P, v, None)
        self.assertEqual(pts.shape, (2, 2))
        self.assertTrue(np.allclose(sorted(vv[0]), [1.0, 2.0]))


if __name__ == '__main__':
    unittest.main()
