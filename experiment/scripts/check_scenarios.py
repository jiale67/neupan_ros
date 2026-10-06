#!/usr/bin/env python3
"""Validate start/goal pairs of config/scenarios.yaml against the Gazebo worlds.

(a) start/goal gap to every static obstacle (BODY_RECT footprint) >= 0.6 m
(b) dynamic worlds: start/goal distance to the swept band of every dynamic obstacle path
    (polyline distance - obstacle bounding radius - robot half-diagonal) >= 0.4 m
(c) the straight start->goal line, widened by the robot half-width 0.1223, touches >= 2
    obstacles (dynamic ones by their path band), i.e. obstacle avoidance is required.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from neupan_exp.geometry import BODY_RECT, footprint_polygon, obstacle_gap, points_in_obstacle  # noqa: E402
from neupan_exp.scenarios import load_scenarios  # noqa: E402
from neupan_exp.world import parse_world  # noqa: E402

WORLD_DIR = os.path.join(HERE, '..', '..', 'example', 'gazebo_V550', 'world')
MIN_STATIC_GAP, MIN_DYN_BAND, HALF_WIDTH, MIN_BLOCKERS, STEP = 0.6, 0.4, 0.1223, 2, 0.05
ROBOT_HALF_DIAG = math.hypot(max(abs(BODY_RECT[0]), abs(BODY_RECT[1])),
                             max(abs(BODY_RECT[2]), abs(BODY_RECT[3])))


def bound_radius(spec):
    return spec.size[0] if spec.kind == 'cylinder' else 0.5 * math.hypot(*spec.size)


def path_dist(p, spec):
    """Distance from point p to the (closed paths: closed) polyline of a dynamic obstacle."""
    pts = [np.array(w, float) for w in spec.path]
    segs = list(zip(pts[:-1], pts[1:]))
    if len(pts) > 2:
        segs.append((pts[-1], pts[0]))   # closed loop (also harmless for pingpong)
    if len(pts) == 1:
        segs = [(pts[0], pts[0])]
    best = np.inf
    for a, b in segs:
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den == 0 else min(1.0, max(0.0, float((p - a) @ ab) / den))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def band_clearance(p, spec):
    return path_dist(np.asarray(p, float), spec) - bound_radius(spec) - ROBOT_HALF_DIAG


def check_pair(specs, start, goal):
    reasons = []
    sx, sy = start
    gx, gy = goal
    yaw = math.atan2(gy - sy, gx - sx)
    statics = [s for s in specs if not s.dynamic]
    dyns = [s for s in specs if s.dynamic]
    # (a)
    for label, (px, py) in (('start', start), ('goal', goal)):
        fp = footprint_polygon(px, py, yaw, *BODY_RECT)
        for s in statics:
            g = obstacle_gap(fp, s, s.x, s.y, s.yaw)
            if g < MIN_STATIC_GAP:
                reasons.append('(a) %s gap %.3f to %s < %.1f' % (label, g, s.name, MIN_STATIC_GAP))
    # (b)
    for label, p in (('start', start), ('goal', goal)):
        for s in dyns:
            c = band_clearance(p, s)
            if c < MIN_DYN_BAND:
                reasons.append('(b) %s band clearance %.3f to %s < %.1f' % (label, c, s.name, MIN_DYN_BAND))
    # (c)
    length = math.hypot(gx - sx, gy - sy)
    n = max(2, int(math.ceil(length / STEP)) + 1)
    samples = [(sx + (gx - sx) * k / (n - 1), sy + (gy - sy) * k / (n - 1)) for k in range(n)]
    pts = np.array(samples).T
    blockers = [s.name for s in statics
                if points_in_obstacle(pts, s, s.x, s.y, s.yaw, HALF_WIDTH).any()]
    for s in dyns:
        if min(path_dist(np.array(p), s) for p in samples) - bound_radius(s) <= HALF_WIDTH:
            blockers.append(s.name)
    if len(blockers) < MIN_BLOCKERS:
        reasons.append('(c) straight line blocked by only %d obstacles (<%d)' % (len(blockers), MIN_BLOCKERS))
    return reasons, blockers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'config', 'scenarios.yaml'))
    args = ap.parse_args()
    cfg = load_scenarios(args.config)
    bad = 0
    for scene, sc in cfg['scenes'].items():
        specs = parse_world(os.path.join(WORLD_DIR, sc['world'] + '.world'))
        for pair in sc['pairs']:
            reasons, blockers = check_pair(specs, tuple(pair['start']), tuple(pair['goal']))
            tag = 'PASS' if not reasons else 'FAIL'
            bad += bool(reasons)
            print('%s P%d %s start=%s goal=%s blockers=%d %s' % (
                scene, pair['id'], tag, tuple(pair['start']), tuple(pair['goal']), len(blockers),
                '; '.join(reasons)))
    print('%d/%d pairs PASS' % (sum(len(s['pairs']) for s in cfg['scenes'].values()) - bad,
                                 sum(len(s['pairs']) for s in cfg['scenes'].values())))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
