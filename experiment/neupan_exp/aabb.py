"""AABB obstacle input for the AABB-MPC baseline."""
import numpy as np
from sklearn.cluster import DBSCAN


def _box_samples(lo, hi, spacing):
    """Points along the perimeter of the box [lo, hi], deduplicated, in stable order."""
    (x0, y0), (x1, y1) = lo, hi
    nx = max(int(np.ceil((x1 - x0) / spacing)), 1) + 1
    ny = max(int(np.ceil((y1 - y0) / spacing)), 1) + 1
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    edges = [np.column_stack([xs, np.full(nx, y0)]),
             np.column_stack([np.full(ny, x1), ys]),
             np.column_stack([xs[::-1], np.full(nx, y1)]),
             np.column_stack([np.full(ny, x0), ys[::-1]])]
    p = np.vstack(edges)
    _, first = np.unique(np.round(p, 9), axis=0, return_index=True)
    return p[np.sort(first)]


def aabb_points(points, velocities, labels, eps=0.15, min_samples=2, spacing=0.1):
    pts = np.asarray(points, dtype=float).reshape(2, -1)
    n = pts.shape[1]
    if velocities is None:
        vel = np.zeros((2, n))
    else:
        vel = np.asarray(velocities, dtype=float)
        if vel.shape != (2, n):
            raise ValueError('velocities must have shape (2, %d), got %s' % (n, vel.shape))
    lab = -np.ones(n, dtype=int) if labels is None else np.asarray(labels, dtype=int)
    if lab.shape != (n,):
        raise ValueError('labels must have %d entries, got shape %s' % (n, lab.shape))
    if n == 0:
        return np.zeros((2, 0)), np.zeros((2, 0))

    groups, noise = [], []
    for l in np.unique(lab[lab >= 0]):
        groups.append(np.flatnonzero(lab == l))
    free = np.flatnonzero(lab < 0)
    if len(free):
        cl = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[:, free].T)
        for c in np.unique(cl[cl >= 0]):
            groups.append(free[cl == c])
        noise = free[cl < 0]

    out_p, out_v = [], []
    for idx in groups:
        box = _box_samples(pts[:, idx].min(axis=1), pts[:, idx].max(axis=1), spacing)
        out_p.append(box.T)
        out_v.append(np.tile(vel[:, idx].mean(axis=1)[:, None], (1, box.shape[0])))
    if len(noise):
        out_p.append(pts[:, noise])
        out_v.append(vel[:, noise])
    return np.hstack(out_p), np.hstack(out_v)
