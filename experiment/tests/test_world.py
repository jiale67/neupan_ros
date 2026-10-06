import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import os
import unittest

from neupan_exp.world import parse_world

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORLD = os.path.join(_HERE, '..', '..', 'example', 'gazebo_V550', 'world')
STATIC = os.path.join(_WORLD, 'static_scene.world')
DYNAMIC = os.path.join(_WORLD, 'dynamic_scene.world')


def by_name(specs, name):
    return [s for s in specs if s.name == name][0]


class TestWorld(unittest.TestCase):
    def test_static(self):
        specs = parse_world(STATIC)
        self.assertEqual(len(specs), 40)
        self.assertFalse(any(s.dynamic for s in specs))
        self.assertTrue(all(s.path == () for s in specs))
        c18 = by_name(specs, 'cylinder_18_0.9_0.9_2.2')
        self.assertEqual(c18.kind, 'cylinder')
        self.assertAlmostEqual(c18.size[0], 0.464521, places=6)
        self.assertAlmostEqual(c18.x, 0.059637, places=6)
        self.assertAlmostEqual(c18.y, -0.036896, places=6)

    def test_dynamic(self):
        dyn = parse_world(DYNAMIC)
        self.assertEqual(sum(s.dynamic for s in dyn), 11)
        self.assertEqual(len(dyn), 33)
        b = by_name(dyn, 'd1_static_box_07')
        self.assertEqual(b.kind, 'box')
        self.assertEqual(b.size, (0.325, 0.125))
        self.assertAlmostEqual(b.yaw, 0.5, places=9)
        self.assertEqual(b.path, ())

    def test_dynamic_path(self):
        d = by_name(parse_world(DYNAMIC), 'd1_crossing_01')
        self.assertTrue(d.dynamic)
        self.assertEqual(d.size, (0.25, 0.2))
        self.assertEqual(len(d.path), 2)
        for got, want in zip(d.path, ((0.46, -2.3), (0.46, 2.3))):
            self.assertAlmostEqual(got[0], want[0], places=9)
            self.assertAlmostEqual(got[1], want[1], places=9)


if __name__ == '__main__':
    unittest.main()
