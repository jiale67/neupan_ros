"""Scenario configuration and trial matrix."""
import math
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Trial:
    scene: str
    world: str          # 'static_scene' | 'dynamic_scene'
    method: str
    pair_id: int
    start: tuple        # (x, y, yaw), yaw faces the goal
    goal: tuple         # (x, y)
    seed: int
    start_sim_time: float
    timeout: float

    @property
    def trial_id(self):
        return f"{self.scene}_{self.method}_p{self.pair_id}_s{self.seed}"


def load_scenarios(path):
    with open(path) as f:
        return yaml.safe_load(f)


def trial_matrix(cfg, scenes, methods, nominal_speed=0.3):
    """All trials for `scenes` x pairs x runs x `methods`. seed = 1000*pair_id + run
    (run from 0); start_sim_time depends only on the seed, so methods share it."""
    trials = []
    for scene in scenes:
        sc = cfg['scenes'][scene]
        sst = sc['start_sim_time']
        for pair in sc['pairs']:
            pid = int(pair['id'])
            sx, sy = pair['start']
            gx, gy = pair['goal']
            yaw = math.atan2(gy - sy, gx - sx)
            timeout = max(cfg['timeout_min'],
                          cfg['timeout_factor'] * math.hypot(gx - sx, gy - sy) / nominal_speed)
            for run in range(int(sc['runs'])):
                seed = 1000 * pid + run
                if isinstance(sst, dict):
                    t0 = sst['base'] + float(np.random.default_rng(seed).uniform(0, sst['spread']))
                else:
                    t0 = float(sst)
                for m in methods:
                    trials.append(Trial(scene, sc['world'], m, pid, (sx, sy, yaw), (gx, gy),
                                        seed, t0, timeout))
    return trials
