"""Ground-truth obstacle poses/velocities from Gazebo model states."""
from collections import deque

import numpy as np

from .geometry import obstacle_gap, points_in_obstacle


class ObstacleTruth:
    def __init__(self, specs, window=0.1):
        self.specs = list(specs)
        self.window = window
        self._index = {s.name: i for i, s in enumerate(self.specs)}
        self._hist = {s.name: deque() for s in self.specs if s.dynamic}
        self._pose = {s.name: (s.x, s.y, s.yaw) for s in self.specs}
        # model name -> spec names (multi-collision models are 'model/idx')
        self._by_model = {}
        for s in self.specs:
            self._by_model.setdefault(s.name.split('/')[0], []).append(s.name)

    def update(self, names, poses, t):
        for model, pose in zip(names, poses):
            for sname in self._by_model.get(model, ()):
                if not self.specs[self._index[sname]].dynamic:
                    continue
                pose = tuple(float(v) for v in pose)
                self._pose[sname] = pose
                h = self._hist[sname]
                h.append((t, pose[0], pose[1]))
                while h and h[0][0] < t - self.window - 1e-9:
                    h.popleft()

    def velocity(self, name):
        h = self._hist.get(name)
        if not h or len(h) < 2 or h[-1][0] <= h[0][0]:
            return 0.0, 0.0
        dt = h[-1][0] - h[0][0]
        return (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt

    def point_labels(self, points, margin=0.08):
        pts = np.asarray(points, dtype=float).reshape(2, -1)
        labels = -np.ones(pts.shape[1], dtype=int)
        for i, s in enumerate(self.specs):
            if not s.dynamic:
                continue
            hit = points_in_obstacle(pts, s, *self._pose[s.name], margin) & (labels < 0)
            labels[hit] = i
        return labels

    def point_velocities(self, points, margin=0.08):
        labels = self.point_labels(points, margin)
        vel = np.zeros((2, len(labels)))
        for i in np.unique(labels[labels >= 0]):
            vel[:, labels == i] = np.array(self.velocity(self.specs[i].name))[:, None]
        return vel

    def min_gap(self, footprint):
        best, best_name = np.inf, ''
        for s in self.specs:
            g = obstacle_gap(footprint, s, *self._pose[s.name])
            if g < best:
                best, best_name = g, s.name
        return float(best), best_name
