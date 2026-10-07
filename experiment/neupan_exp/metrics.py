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


def settle_done(ts, speeds, t_arrive, hold=1.0, max_wait=5.0, v_eps=0.02):
    """True once the post-arrival settle window may close.

    ts ascending sample times (s), speeds[i] = planar speed at ts[i] (from consecutive poses;
    NaN/inf for an unknown one). Done when ts[-1] >= t_arrive + max_wait, or when the trailing
    samples with ts >= t_arrive all have speed < v_eps and span >= hold seconds.
    """
    if len(ts) == 0:
        return False
    eps = 1e-9
    if ts[-1] - t_arrive >= max_wait - eps:
        return True
    i = len(ts) - 1
    while i >= 0 and ts[i] >= t_arrive - eps and speeds[i] < v_eps:
        i -= 1
    first = i + 1
    return first < len(ts) and ts[-1] - ts[first] >= hold - eps


class TrialSequencer:
    """Trial end sequencing of the recorder: running -> (arrived) settling -> done.

    update() is fed every pose sample and returns the final end reason ('timeout' | 'stuck'
    before arrival, 'arrived' once the robot has settled after arrival) or None. Arrival
    (dist <= arrive_threshold) fixes `t_arrive` / `e_g_arrive` but does not end the trial; the
    recorder keeps sampling (collisions still count) until settle_done().
    """

    def __init__(self, arrive_threshold, timeout, stuck_window, stuck_dist, hold=1.0,
                 max_wait=5.0, v_eps=0.02):
        self.kw = dict(timeout=timeout, stuck_window=stuck_window, stuck_dist=stuck_dist,
                       arrive_threshold=arrive_threshold)
        self.hold, self.max_wait, self.v_eps = hold, max_wait, v_eps
        self.t_arrive = None
        self.e_g_arrive = None
        self.settle_s = None
        self._ts, self._speeds, self._prev = [], [], None

    @property
    def arrived(self):
        return self.t_arrive is not None

    def update(self, t_abs, t0, t, x, y, dist, progress):
        if self.t_arrive is None:
            r = end_reason(t_abs, t0, dist, progress, **self.kw)
            if r != 'arrived':
                return r
            self.t_arrive, self.e_g_arrive = t, dist
            self._ts, self._speeds = [t], [float('inf')]
            self._prev = (t, x, y)
            return None
        pt, px, py = self._prev
        self._prev = (t, x, y)
        self._ts.append(t)
        self._speeds.append(math.hypot(x - px, y - py) / (t - pt) if t > pt else float('inf'))
        if settle_done(self._ts, self._speeds, self.t_arrive, self.hold, self.max_wait,
                       self.v_eps):
            self.settle_s = t - self.t_arrive
            return 'arrived'
        return None


def rtf_of(sim_s, wall_s):
    """Real-time factor = simulated seconds per wall second (NaN if wall_s <= 0)."""
    return float(sim_s) / float(wall_s) if wall_s and wall_s > 0 else NAN


def sample_std(vals):
    """Sample standard deviation (ddof=1); NaN for fewer than 2 values."""
    v = np.asarray(list(vals), float)
    return float(np.std(v, ddof=1)) if v.size >= 2 else NAN


def _cmd_arrays(cmd_t, vx, vy, w, t_end):
    t = np.asarray(cmd_t, float)
    u = np.stack([np.asarray(vx, float), np.asarray(vy, float), np.asarray(w, float)], axis=1) \
        if t.size else np.zeros((0, 3))
    if t_end is not None and t.size:
        keep = t <= t_end + 1e-9
        t, u = t[keep], u[keep]
    return t, u


def _zero_rows(u):
    return np.all(np.abs(u) < 1e-9, axis=1)


def uniform_estops(cmd_t, vx, vy, w, t_end=None, arrival_grace=1.0):
    """E-stops by one definition for every method: rising edges of 'published command exactly
    zero (|vx|,|vy|,|w| all < 1e-9)' after the first non-zero command, over commands with
    t <= t_end (None = all). With t_end (= arrival) a zero run that is still going on at t_end
    and started at most `arrival_grace` s before it is the planner's own arrival stop, not an
    e-stop, and is not counted."""
    t, u = _cmd_arrays(cmd_t, vx, vy, w, t_end)
    if not t.size:
        return 0
    z = _zero_rows(u)
    nz = np.flatnonzero(~z)
    if not nz.size:
        return 0
    first = int(nz[0])
    n = _rising(z[first:])
    if t_end is not None and z[-1]:
        k = len(z) - 1
        while k > first and z[k - 1]:
            k -= 1
        if k > first and t_end - t[k] <= arrival_grace + 1e-9:
            n -= 1                           # the trailing run is a counted rising edge
    return n


def cmd_jerk_split(cmd_t, vx, vy, w, t_end=None):
    """(J_v, J_w) = (sum |d(vx,vy)|^2, sum dw^2) over consecutive commands with t <= t_end."""
    t, u = _cmd_arrays(cmd_t, vx, vy, w, t_end)
    if len(t) < 2:
        return 0.0, 0.0
    d = np.diff(u, axis=0)
    return float(np.sum(d[:, :2] ** 2)), float(np.sum(d[:, 2] ** 2))


def accel_samples(cmd_t, vx, vy, w, t_end=None, min_dt=0.0):
    """Executed-command accelerations between consecutive commands (t <= t_end):
    a_lin = |d(vx,vy)|/dt, a_ang = |dw|/dt (real timestamps; pairs with dt <= min_dt skipped).
    Pairs touching an all-zero command are left out (e-stops/starts would dominate);
    n_zero_trans counts those where exactly one of the two commands is all-zero.
    min_dt > 0 drops pairs closer together than that (planners may publish twice within one
    control cycle, which would divide a normal step by a near-zero interval);
    n_sub_cycle counts the non-zero-transition pairs dropped for that reason.
    d_lin / d_ang are the per-command steps |d(vx,vy)| and |dw| of every pair not touching a
    zero command, independent of the (recorder receive-time) timestamps: dividing them by the
    nominal control period gives the acceleration the planner actually commanded per cycle."""
    t, u = _cmd_arrays(cmd_t, vx, vy, w, t_end)
    a_lin, a_ang, trans, sub, d_lin, d_ang = [], [], 0, 0, [], []
    if len(t) >= 2:
        z = _zero_rows(u)
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            if z[i] != z[i - 1]:
                trans += 1
                continue
            if z[i] or z[i - 1]:
                continue
            d_lin.append(float(np.hypot(u[i, 0] - u[i - 1, 0], u[i, 1] - u[i - 1, 1])))
            d_ang.append(float(abs(u[i, 2] - u[i - 1, 2])))
            if dt <= 0:
                continue
            if dt < min_dt:
                sub += 1
                continue
            a_lin.append(float(np.hypot(u[i, 0] - u[i - 1, 0], u[i, 1] - u[i - 1, 1]) / dt))
            a_ang.append(float(abs(u[i, 2] - u[i - 1, 2]) / dt))
    return dict(a_lin=np.array(a_lin), a_ang=np.array(a_ang), n_zero_trans=trans,
                n_sub_cycle=sub, d_lin=np.array(d_lin), d_ang=np.array(d_ang))


def status_gap_max(st_t, t_end):
    """Largest gap (s) between consecutive status messages up to t_end, including the tail gap
    from the last message to t_end. 0.0 without status messages."""
    ts = [float(v) for v in st_t if v <= t_end + 1e-9]
    if not ts:
        return 0.0
    gaps = [b - a for a, b in zip(ts[:-1], ts[1:])] + [max(0.0, t_end - ts[-1])]
    return float(max(gaps))


def compute_trial_metrics(log, goal, end_reason, arrive_threshold=0.3, budget_ms=100.0,
                          t_arrive=None):
    """Metrics of one trial.

    log keys: t, x, y, yaw, gap_np, gap_body (pose stream); cmd_t, cmd_vx, cmd_vy, cmd_w
    (one per /cmd_vel); st_t, plan_ms, stop (one per status message).
    end_reason: 'arrived' | 'timeout' | 'stuck' | 'crash'.
    outcome = 'success' if arrived with 0 collisions, 'collision' if arrived with
    collisions > 0, else end_reason. `arrive_threshold` is accepted for interface
    compatibility; arrival is decided by the caller (end_reason).
    t_arrive (seconds from t0, None = not tracked): the first sample inside the arrive
    threshold. Then T = t_arrive, L / J_du / planner e-stops / planning-time statistics only
    cover t <= t_arrive, e_g is the distance at the END of the log (after the settle window),
    e_g_arrive the distance at t_arrive, while collisions and gaps cover the whole log.
    Also returns `plan_ms_all`, the raw per-cycle planning times (used by aggregate).
    """
    t = np.asarray(log['t'], float)
    x = np.asarray(log['x'], float)
    y = np.asarray(log['y'], float)
    gap_np = np.asarray(log['gap_np'], float)
    gap_body = np.asarray(log['gap_body'], float)
    cmd_t = np.asarray(log['cmd_t'], float)
    st_t = np.asarray(log['st_t'], float)
    stop = np.asarray(log['stop'], bool)
    p = np.asarray(log['plan_ms'], float)
    u = np.stack([np.asarray(log['cmd_vx'], float), np.asarray(log['cmd_vy'], float),
                  np.asarray(log['cmd_w'], float)], axis=1)
    extra = {}
    if t_arrive is not None and t.size:
        ia = int(min(np.searchsorted(t, t_arrive - 1e-9), t.size - 1))
        n_pose = ia + 1
        T = float(t_arrive)
        extra['e_g_arrive'] = float(math.hypot(x[ia] - goal[0], y[ia] - goal[1]))
        extra['t_arrive'] = float(t_arrive)
        cmd_keep = cmd_t <= t_arrive + 1e-9
        st_keep = st_t <= t_arrive + 1e-9
        u, stop, p = u[cmd_keep], stop[st_keep], p[st_keep]
    else:
        n_pose = t.size
        T = float(t[-1] - t[0]) if t.size > 1 else 0.0
    L = float(np.sum(np.hypot(np.diff(x[:n_pose]), np.diff(y[:n_pose])))) if n_pose > 1 else 0.0
    e_g = float(math.hypot(x[-1] - goal[0], y[-1] - goal[1])) if x.size else NAN
    collisions = _rising(gap_body <= 0)
    estops = _rising(stop)
    J_du = float(np.sum(np.diff(u, axis=0) ** 2)) if len(u) > 1 else 0.0
    if p.size:
        t_avg, t95, t_max = float(p.mean()), float(np.percentile(p, 95)), float(p.max())
        r_over = float(np.mean(p > budget_ms) * 100.0)
    else:
        t_avg = t95 = t_max = r_over = NAN
    if end_reason == 'arrived':
        outcome = 'success' if collisions == 0 else 'collision'
    else:
        outcome = end_reason
    out = dict(outcome=outcome, success=outcome == 'success', T=T, L=L, e_g=e_g,
               d_min_np=float(gap_np.min()) if gap_np.size else NAN,
               d_min_body=float(gap_body.min()) if gap_body.size else NAN,
               collisions=collisions, estops=estops, J_du=J_du,
               t_avg=t_avg, t95=t95, t_max=t_max, r_over=r_over, n_cycles=int(p.size),
               plan_ms_all=[float(v) for v in p])
    out.update(extra)
    return out


def _ms(vals):
    if not vals:
        return NAN, NAN
    return float(np.mean(vals)), sample_std(vals)


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

    SR (%) over all trials; T, L, e_g, J_du, J_v, J_w -> `<k>_mean`/`<k>_std` (SAMPLE std,
    ddof=1, NaN for n < 2) over successful trials only (NaN if none); d_min_body/d_min_np -> `_min`, `_mean`, `_std` over all trials
    (NaN-aware: crash rows with NaN are skipped, NaN only if every row is NaN);
    collisions/estops -> `_total` and `_mean`; `outcomes` = outcome->count; n_trials,
    n_success. Timing (t_avg, t95, t_max, r_over): if rows carry `plan_ms_all`, the raw
    per-cycle samples are pooled across trials; otherwise the per-trial values are
    averaged (t_max: maximum; t95: mean of per-trial t95).
    """
    n = len(rows)
    succ = [r for r in rows if r['success']]
    out = {'n_trials': n, 'n_success': len(succ), 'SR': 100.0 * len(succ) / n if n else NAN}
    for k in ('T', 'L', 'e_g', 'J_du', 'J_v', 'J_w'):
        out[k + '_mean'], out[k + '_std'] = _ms([r.get(k, NAN) for r in succ])
    for k in ('d_min_body', 'd_min_np'):
        v = [r[k] for r in rows]
        out[k + '_min'] = _nanred(v, np.min)
        out[k + '_mean'] = _nanred(v, np.mean)
        out[k + '_std'] = sample_std(_finite(v))
    for k in ('collisions', 'estops', 'estops_planner'):
        if k not in (rows[0] if rows else {}):
            continue
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
