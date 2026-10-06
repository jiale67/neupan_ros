"""Per-trial metrics and cross-trial aggregation."""
import bisect
import math

import numpy as np

NAN = float('nan')


def _rising(flags):
    f = np.asarray(flags, dtype=bool)
    if f.size == 0:
        return 0
    return int(np.sum(f & ~np.concatenate([[False], f[:-1]])))


def window_progress(ts, dists, window):
    """Progress toward the goal over the trailing `window` seconds of a trial.

    ts ascending (seconds), dists = distance to goal at each ts. Returns None until the
    history covers `window`; otherwise (dist at the start of the window, i.e. the last
    sample at or before ts[-1] - window) minus (min dist from there to the end), >= 0.
    Progress-based (not displacement-based) so lateral jitter does not hide a stall.
    """
    if len(ts) < 2 or ts[-1] - ts[0] < window:
        return None
    i0 = max(bisect.bisect_right(ts, ts[-1] - window + 1e-9) - 1, 0)
    return max(0.0, float(dists[i0]) - float(min(dists[i0:])))


def end_reason(t, t0, dist_goal, progress_window, timeout, stuck_window, stuck_dist,
               arrive_threshold):
    """Trial end condition: 'arrived' | 'timeout' | 'stuck' | None (precedence in that order).

    progress_window: see window_progress (None counts as not yet measurable). 'stuck' needs
    the trial to have run >= stuck_window and progress_window < stuck_dist.
    """
    if dist_goal <= arrive_threshold:
        return 'arrived'
    if t - t0 > timeout:
        return 'timeout'
    if (t - t0 >= stuck_window and progress_window is not None
            and progress_window < stuck_dist):
        return 'stuck'
    return None


def compute_trial_metrics(log, goal, end_reason, arrive_threshold=0.3, budget_ms=100.0):
    """Metrics of one trial.

    log keys: t, x, y, yaw, gap_np, gap_body (pose stream); cmd_t, cmd_vx, cmd_vy, cmd_w
    (one per /cmd_vel); st_t, plan_ms, stop (one per status message).
    end_reason: 'arrived' | 'timeout' | 'stuck' | 'crash'.
    outcome = 'success' if arrived with 0 collisions, 'collision' if arrived with
    collisions > 0, else end_reason. `arrive_threshold` is accepted for interface
    compatibility; arrival is decided by the caller (end_reason).
    Also returns `plan_ms_all`, the raw per-cycle planning times (used by aggregate).
    """
    t = np.asarray(log['t'], float)
    x = np.asarray(log['x'], float)
    y = np.asarray(log['y'], float)
    T = float(t[-1] - t[0]) if t.size > 1 else 0.0
    L = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if x.size > 1 else 0.0
    e_g = float(math.hypot(x[-1] - goal[0], y[-1] - goal[1])) if x.size else NAN
    gap_np = np.asarray(log['gap_np'], float)
    gap_body = np.asarray(log['gap_body'], float)
    collisions = _rising(gap_body <= 0)
    estops = _rising(log['stop'])
    u = np.stack([np.asarray(log['cmd_vx'], float), np.asarray(log['cmd_vy'], float),
                  np.asarray(log['cmd_w'], float)], axis=1)
    J_du = float(np.sum(np.diff(u, axis=0) ** 2)) if len(u) > 1 else 0.0
    p = np.asarray(log['plan_ms'], float)
    if p.size:
        t_avg, t95, t_max = float(p.mean()), float(np.percentile(p, 95)), float(p.max())
        r_over = float(np.mean(p > budget_ms) * 100.0)
    else:
        t_avg = t95 = t_max = r_over = NAN
    if end_reason == 'arrived':
        outcome = 'success' if collisions == 0 else 'collision'
    else:
        outcome = end_reason
    return dict(outcome=outcome, success=outcome == 'success', T=T, L=L, e_g=e_g,
                d_min_np=float(gap_np.min()) if gap_np.size else NAN,
                d_min_body=float(gap_body.min()) if gap_body.size else NAN,
                collisions=collisions, estops=estops, J_du=J_du,
                t_avg=t_avg, t95=t95, t_max=t_max, r_over=r_over, n_cycles=int(p.size),
                plan_ms_all=[float(v) for v in p])


def _ms(vals):
    if not vals:
        return NAN, NAN
    return float(np.mean(vals)), float(np.std(vals))


def _finite(vals):
    """Non-NaN entries of vals as a float array (crash rows carry NaN metrics)."""
    v = np.asarray(list(vals), float)
    return v[~np.isnan(v)]


def _nanred(vals, fn):
    """fn over the non-NaN entries; NaN only if there are none."""
    v = _finite(vals)
    return float(fn(v)) if v.size else NAN


def aggregate(rows, budget_ms=100.0):
    """Aggregate per-trial metric dicts into a flat dict.

    SR (%) over all trials; T, L, e_g, J_du -> `<k>_mean`/`<k>_std` over successful trials
    only (NaN if none); d_min_body/d_min_np -> `_min`, `_mean`, `_std` over all trials
    (NaN-aware: crash rows with NaN are skipped, NaN only if every row is NaN);
    collisions/estops -> `_total` and `_mean`; `outcomes` = outcome->count; n_trials,
    n_success. Timing (t_avg, t95, t_max, r_over): if rows carry `plan_ms_all`, the raw
    per-cycle samples are pooled across trials; otherwise the per-trial values are
    averaged (t_max: maximum; t95: mean of per-trial t95).
    """
    n = len(rows)
    succ = [r for r in rows if r['success']]
    out = {'n_trials': n, 'n_success': len(succ), 'SR': 100.0 * len(succ) / n if n else NAN}
    for k in ('T', 'L', 'e_g', 'J_du'):
        out[k + '_mean'], out[k + '_std'] = _ms([r[k] for r in succ])
    for k in ('d_min_body', 'd_min_np'):
        v = [r[k] for r in rows]
        out[k + '_min'] = _nanred(v, np.min)
        out[k + '_mean'] = _nanred(v, np.mean)
        out[k + '_std'] = _nanred(v, np.std)
    for k in ('collisions', 'estops'):
        v = [r[k] for r in rows]
        out[k + '_total'] = int(np.sum(v)) if v else 0
        out[k + '_mean'] = float(np.mean(v)) if v else NAN
    oc = {}
    for r in rows:
        oc[r['outcome']] = oc.get(r['outcome'], 0) + 1
    out['outcomes'] = oc
    if rows and all(r.get('plan_ms_all') is not None for r in rows):
        p = np.concatenate([np.asarray(r['plan_ms_all'], float) for r in rows])
    else:
        p = np.array([])
    if p.size:
        out.update(t_avg=float(p.mean()), t95=float(np.percentile(p, 95)),
                   t_max=float(p.max()), r_over=float(np.mean(p > budget_ms) * 100.0))
    else:
        out.update(t_avg=_nanred([r['t_avg'] for r in rows], np.mean),
                   t95=_nanred([r['t95'] for r in rows], np.mean),
                   t_max=_nanred([r['t_max'] for r in rows], np.max),
                   r_over=_nanred([r['r_over'] for r in rows], np.mean))
    return out
