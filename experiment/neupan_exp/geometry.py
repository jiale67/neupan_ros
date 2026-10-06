"""2-D geometry helpers: footprints, SAT-based signed gaps, obstacle queries."""
import numpy as np

NP_RECT = (-0.13, 0.13, -0.1223, 0.1223)
BODY_RECT = (-0.1300, 0.1232, -0.1223, 0.1223)


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def rect_polygon(cx, cy, yaw, lx, ly):
    """Corners (4,2), CCW, of a lx x ly rectangle centred at (cx, cy)."""
    local = np.array([[-lx / 2, -ly / 2], [lx / 2, -ly / 2],
                      [lx / 2, ly / 2], [-lx / 2, ly / 2]])
    return local @ _rot(yaw).T + np.array([cx, cy])


def footprint_polygon(x, y, yaw, xmin, xmax, ymin, ymax):
    """Corners (4,2), CCW, of a base_link-frame rectangle placed at (x, y, yaw)."""
    local = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return local @ _rot(yaw).T + np.array([x, y])


def _edge_normals(P):
    e = np.roll(P, -1, axis=0) - P
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _point_segment_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _min_boundary_dist(P, Q):
    """Shortest distance between the boundaries of two polygons."""
    best = np.inf
    for A, B in ((P, Q), (Q, P)):
        nb = len(B)
        for p in A:
            for i in range(nb):
                best = min(best, _point_segment_dist(p, B[i], B[(i + 1) % nb]))
    return best


def polygon_gap(P, Q):
    """Signed gap between convex polygons: shortest distance if separated (>0),
    otherwise minus the minimum penetration depth (SAT)."""
    min_overlap = np.inf
    separated = False
    for n in np.vstack([_edge_normals(P), _edge_normals(Q)]):
        p, q = P @ n, Q @ n
        overlap = min(p.max(), q.max()) - max(p.min(), q.min())
        if overlap <= 0:
            separated = True
            break
        min_overlap = min(min_overlap, overlap)
    if separated:
        return _min_boundary_dist(P, Q)
    return -float(min_overlap)


def circle_polygon_gap(cx, cy, r, P):
    """Signed gap between a circle and a convex CCW polygon (negative if overlapping)."""
    c = np.array([cx, cy], dtype=float)
    n = len(P)
    d = min(_point_segment_dist(c, P[i], P[(i + 1) % n]) for i in range(n))
    e = np.roll(P, -1, axis=0) - P
    rel = c - P
    cross = e[:, 0] * rel[:, 1] - e[:, 1] * rel[:, 0]
    inside = bool(np.all(cross >= 0))
    return (-d if inside else d) - r


def obstacle_gap(footprint, spec, x, y, yaw):
    """Gap between a footprint polygon and the obstacle `spec` at pose (x, y, yaw)."""
    if spec.kind == 'cylinder':
        return circle_polygon_gap(x, y, spec.size[0], footprint)
    return polygon_gap(footprint, rect_polygon(x, y, yaw, spec.size[0], spec.size[1]))


def points_in_obstacle(points, spec, x, y, yaw, margin):
    """Bool (N,) for points (2,N) within `margin` of the obstacle at pose (x, y, yaw)
    (margin 0: strictly the obstacle's own area)."""
    pts = np.asarray(points, dtype=float).reshape(2, -1)
    local = _rot(yaw).T @ (pts - np.array([[x], [y]]))
    if spec.kind == 'cylinder':
        return np.hypot(local[0], local[1]) - spec.size[0] <= margin
    q = np.abs(local) - np.array([[spec.size[0] / 2], [spec.size[1] / 2]])
    outside = np.hypot(np.maximum(q[0], 0), np.maximum(q[1], 0))
    inside = np.minimum(np.maximum(q[0], q[1]), 0)
    return outside + inside <= margin
