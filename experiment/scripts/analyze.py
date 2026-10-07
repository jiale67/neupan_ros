#!/usr/bin/env python3
"""Turn a batch run's per-trial logs into the thesis tables (LaTeX + CSV) and figures.

    python3 scripts/analyze.py --results <run_dir> [--out <run_dir>/report]

Reads <run>/trials/<trial_id>.json (+ .csv / _cmd.csv for figures); never writes outside --out.
Outputs: all_trials.csv, summary_<scene>.csv, summary_matched_<scene>.csv, timing.csv,
accel.csv, sanity.txt, tables.tex, traj_<scene>_p<k>.pdf, ts_<scene>_p<k>.pdf, plan_time_box.pdf.

Command-derived metrics (read from <trial>_cmd.csv, commands up to t_arrive, failures: all):
uniform e-stops (rising edges of an all-zero command after the first non-zero one), J_v, J_w,
executed accelerations. Data from before fix F2 (no t_arrive / settle_s / rtf) still loads.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from neupan_exp.geometry import rect_polygon  # noqa: E402
from neupan_exp.metrics import (accel_samples, aggregate, cmd_jerk_split, sample_std,  # noqa: E402
                                status_gap_max, uniform_estops)
from neupan_exp.world import parse_world  # noqa: E402

DEFAULT_WORLD_DIR = os.path.join(HERE, '..', '..', 'example', 'gazebo_V550', 'world')
METHODS = ('neupan', 'aabb', 'dwa')                       # table row order
DISPLAY = {'neupan': '本文方法', 'aabb': 'AABB-MPC', 'dwa': 'DWA'}
OUTCOME_ZH = {'timeout': '超时', 'stuck': '停滞', 'collision': '碰撞', 'crash': '运行异常'}
COLORS = {'neupan': '#0072B2', 'aabb': '#D55E00', 'dwa': '#009E73'}   # Okabe-Ito
STYLES = {'neupan': '-', 'aabb': '--', 'dwa': '-.'}
FONT_CANDIDATES = ('SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
                   'WenQuanYi Zen Hei')
NAN = float('nan')
CTRL_PERIOD = 0.1            # experiment control period (s), see global constraints

ROW_KEYS = ('T', 'L', 'e_g', 'd_min_np', 'd_min_body', 'collisions', 'J_du', 't_avg',
            't95', 't_max', 'r_over', 'n_cycles')


# ----------------------------------------------------------------------------- loading
def _num(v):
    """float(v) if v is a finite real number (not bool/str), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def arrival_time(d):
    """Time of arrival (s from the goal message) = end of the 'path to the goal' window used
    for all command metrics; None when the trial did not arrive. Old json: T of arrived trials."""
    ta = _num(d.get('t_arrive'))
    if ta is not None:
        return ta
    if 't_arrive' not in d and d.get('end_reason') == 'arrived':
        return _num(d.get('T'))
    return None


def command_metrics(base, t_arrive, planner_estops):
    """Metrics derived from <base>_cmd.csv (commands up to t_arrive, or all if it is None)."""
    c = read_csv_cols(base + '_cmd.csv')
    out = dict(estops=planner_estops, estops_src='planner', J_v=NAN, J_w=NAN, a_lin=[], a_ang=[],
               a_lin_cyc=[], a_ang_cyc=[], d_lin=[], d_ang=[], n_zero_trans=0, n_sub_cycle=0)
    if c is None or not len(c.get('t', [])):
        return out
    t_end = t_arrive
    out['estops'] = uniform_estops(c['t'], c['vx'], c['vy'], c['w'], t_end)
    out['estops_src'] = 'cmd'
    out['J_v'], out['J_w'] = cmd_jerk_split(c['t'], c['vx'], c['vy'], c['w'], t_end)
    a = accel_samples(c['t'], c['vx'], c['vy'], c['w'], t_end)
    # same, but ignoring pairs published closer together than half a control cycle (a planner
    # may publish twice inside one 0.1 s cycle; dividing by that dt is meaningless). Its
    # n_sub_cycle counts the pairs dropped for exactly that reason.
    ac = accel_samples(c['t'], c['vx'], c['vy'], c['w'], t_end, min_dt=CTRL_PERIOD / 2.0)
    out.update(a_lin=list(a['a_lin']), a_ang=list(a['a_ang']), n_zero_trans=a['n_zero_trans'],
               n_sub_cycle=ac['n_sub_cycle'], a_lin_cyc=list(ac['a_lin']),
               a_ang_cyc=list(ac['a_ang']), d_lin=list(a['d_lin']), d_ang=list(a['d_ang']))
    return out


def sanity_flags(d):
    """Flags of one trial from a dict with (all optional) status_gap_max, n_cycles,
    goal_subscribers, late_start, attempts, rtf, estops_src + outcome. Missing / non-numeric
    fields never flag. `estops_fallback`: a non-crash trial without a usable _cmd.csv, so the
    planner-reported e-stop count stands in for the uniform one."""
    f = []
    g = _num(d.get('status_gap_max'))
    if g is not None and g > 0.5:
        f.append('status_gap')
    n = _num(d.get('n_cycles'))
    if n is not None and n == 0:
        f.append('no_cycles')
    gs = _num(d.get('goal_subscribers'))
    if gs is not None and gs < 2:
        f.append('goal_subs<2')
    if d.get('late_start') is True:
        f.append('late_start')
    at = _num(d.get('attempts'))
    if at is not None and at > 1:
        f.append('retried')
    r = _num(d.get('rtf'))
    if r is not None and r < 0.9:
        f.append('rtf<0.9')
    if d.get('estops_src') == 'planner' and d.get('outcome') != 'crash':
        f.append('estops_fallback')
    return f


def _status_gap(base, t_arrive):
    st = read_csv_cols(base + '_status.csv')
    if st is None or not len(st.get('t', [])):
        return 0.0
    t_end = t_arrive
    if t_end is None:
        pose = read_csv_cols(base + '.csv')
        t_end = float(pose['t'][-1]) if pose is not None and len(pose['t']) else float(st['t'][-1])
    return status_gap_max(st['t'], t_end)


def load_trials(results):
    """One flat row per trial json (crash rows included). `plan_ms_all` is kept."""
    rows = []
    for p in sorted(glob.glob(os.path.join(results, 'trials', '*.json'))):
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        meta = d.get('meta') or {}
        tid = d.get('trial_id') or meta.get('trial_id') or os.path.basename(p)[:-5]
        parts = tid.split('_')
        r = {k: d.get(k, NAN) for k in ROW_KEYS}
        r['n_cycles'] = d.get('n_cycles', len(d.get('plan_ms_all') or []))
        base = os.path.join(results, 'trials', tid)
        ta = arrival_time(d)
        r['estops_planner'] = d.get('estops', NAN)
        r.update(command_metrics(base, ta, r['estops_planner']))
        r.update(trial_id=tid, outcome=d.get('outcome', 'crash'),
                 success=bool(d.get('success', False)), end_reason=d.get('end_reason', ''),
                 scene=meta.get('scene', parts[0]), method=meta.get('method', parts[1]),
                 pair=int(meta.get('pair_id', parts[2].lstrip('p'))),
                 seed=int(meta.get('seed', parts[3].lstrip('s'))),
                 world=meta.get('world', ''), start=meta.get('start') or d.get('start'),
                 goal=d.get('goal') or meta.get('goal'),
                 plan_ms_all=list(d.get('plan_ms_all') or []),
                 late_start=meta.get('late_start', ''), attempts=d.get('attempts', ''),
                 runner_patched=d.get('runner_patched', ''),
                 t_arrive=ta if ta is not None else NAN,
                 settle_s=_num(d.get('settle_s')) if _num(d.get('settle_s')) is not None else NAN,
                 e_g_arrive=_num(d.get('e_g_arrive')) if _num(d.get('e_g_arrive')) is not None
                 else NAN,
                 has_settle_key='settle_s' in d,
                 rtf=_num(d.get('rtf')) if _num(d.get('rtf')) is not None else NAN,
                 wall_s=_num(d.get('wall_s')) if _num(d.get('wall_s')) is not None else NAN,
                 goal_subscribers=meta.get('goal_subscribers', ''),
                 code_check=meta.get('code_check', ''),
                 status_gap_max=_status_gap(base, ta),
                 json_path=p, base=base)
        r['flags'] = ';'.join(sanity_flags(r))
        rows.append(r)
    rows.sort(key=lambda r: (r['scene'], r['pair'], r['method'], r['seed']))
    return rows


# ----------------------------------------------------------------------------- formatting
def _finite(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def f2(v):
    return '%.2f' % v if _finite(v) else '--'


def pm(mean, std):
    """mean +- SAMPLE std. Without a std (n < 2 values) only the mean is printed, '--' if the
    mean itself is undefined (no value at all)."""
    if not _finite(mean):
        return '--'
    return '$%.2f\\pm %.2f$' % (mean, std) if _finite(std) else '$%.2f$' % mean


def f1(v):
    return '%.1f' % v if _finite(v) else '--'


def outcomes_str(oc):
    return ';'.join('%s:%d' % (k, v) for k, v in sorted(oc.items()))


def _csv_val(v):
    if isinstance(v, float):
        return 'nan' if math.isnan(v) else repr(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def write_csv(path, header, records):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            w.writerow([_csv_val(r.get(k, '')) for k in header])


# ----------------------------------------------------------------------------- statistics
def group(rows):
    """{(scene, method): [rows]} and {scene: [methods present in METHODS order]}."""
    g = {}
    for r in rows:
        g.setdefault((r['scene'], r['method']), []).append(r)
    scenes = {}
    for s, m in g:
        scenes.setdefault(s, []).append(m)
    for s in scenes:
        scenes[s].sort(key=lambda m: METHODS.index(m) if m in METHODS else len(METHODS))
    return g, scenes


def _mean_finite(vals):
    v = [x for x in vals if _finite(x)]
    return float(np.mean(v)) if v else NAN


def summarize(rows):
    """{(scene, method): aggregate dict + pooled cycle count + mean RTF}."""
    g, _ = group(rows)
    out = {}
    for key, rs in g.items():
        a = aggregate(rs)
        a['n_cycles'] = int(sum(len(r['plan_ms_all']) for r in rs))
        a['rtf_mean'] = _mean_finite([r.get('rtf', NAN) for r in rs])
        out[key] = a
    return out


MATCH_KEYS = ('T', 'L', 'J_v', 'J_w')


def matched_sets(rows):
    """{scene: {(pair, seed)}} where every method present in the scene succeeded."""
    by = {}
    for r in rows:
        d = by.setdefault(r['scene'], {}).setdefault((r['pair'], r['seed']), {})
        d[r['method']] = d.get(r['method'], True) and bool(r['success'])
    out = {}
    for scene, cells in by.items():
        methods = {r['method'] for r in rows if r['scene'] == scene}
        out[scene] = {k for k, v in cells.items()
                      if set(v) == methods and all(v.values())}
    return out


def matched_summary(rows, matched):
    """{(scene, method): n_matched, <k>_mean/_std (sample std) of T, L, J_v, J_w over the
    pair-matched set of the scene}."""
    out = {}
    for r in rows:
        out.setdefault((r['scene'], r['method']), [])
        if (r['pair'], r['seed']) in matched.get(r['scene'], set()):
            out[(r['scene'], r['method'])].append(r)
    res = {}
    for key, rs in out.items():
        a = {'n_matched': len(rs)}
        for k in MATCH_KEYS:
            v = [x for x in (r.get(k, NAN) for r in rs) if _finite(x)]
            a[k + '_mean'] = float(np.mean(v)) if v else NAN
            a[k + '_std'] = sample_std(v)
        res[key] = a
    return res


def _accel_stats(rs, lin_key, ang_key):
    lin = np.array([v for r in rs for v in r.get(lin_key, [])], float)
    ang = np.array([v for r in rs for v in r.get(ang_key, [])], float)
    return dict(
        peak_lin=float(lin.max()) if lin.size else NAN,
        p95_lin=float(np.percentile(lin, 95)) if lin.size else NAN,
        peak_ang=float(ang.max()) if ang.size else NAN,
        p95_ang=float(np.percentile(ang, 95)) if ang.size else NAN,
        n_pairs=int(lin.size))


def accel_summary(rows):
    """{(scene, method): executed-command acceleration statistics, pooled over the trials}.

    Primary view (table; `dv_*`, `dw_*`, `nom_*`): per-command steps |d(vx,vy)| (m/s) and
    |dw| (rad/s) between consecutive non-zero commands, max and p95, and the same divided by
    the nominal control period CTRL_PERIOD (`nom_peak_*`, `nom_p95_*`, m/s^2 and rad/s^2).
    The steps do not depend on timestamps. Receive-time view (accel.csv only; `peak_*`,
    `p95_*` use the recorder's /cmd_vel receive times as dt, `*_cyc` additionally drop pairs
    closer than half a cycle, `n_sub_cycle` counts those): the receive times jitter by tens of
    ms, which inflates these numbers. Pairs touching an all-zero command are in neither view
    (`n_zero_trans` counts the zero <-> non-zero transitions)."""
    g, _ = group(rows)
    out = {}
    for key, rs in g.items():
        a = _accel_stats(rs, 'a_lin', 'a_ang')
        a.update({k + '_cyc': v for k, v in _accel_stats(rs, 'a_lin_cyc', 'a_ang_cyc').items()})
        d = _accel_stats(rs, 'd_lin', 'd_ang')
        a.update(dv_max=d['peak_lin'], dv_p95=d['p95_lin'], dw_max=d['peak_ang'],
                 dw_p95=d['p95_ang'], n_steps=d['n_pairs'],
                 nom_peak_lin=d['peak_lin'] / CTRL_PERIOD, nom_p95_lin=d['p95_lin'] / CTRL_PERIOD,
                 nom_peak_ang=d['peak_ang'] / CTRL_PERIOD, nom_p95_ang=d['p95_ang'] / CTRL_PERIOD)
        a['n_zero_trans'] = int(sum(r.get('n_zero_trans', 0) for r in rs))
        a['n_sub_cycle'] = int(sum(r.get('n_sub_cycle', 0) for r in rs))
        a['n_trials'] = len(rs)
        out[key] = a
    return out


def failure_note(summ, scene, methods):
    parts = []
    for m in methods:
        a = summ[(scene, m)]
        fails = {k: v for k, v in a['outcomes'].items() if k != 'success'}
        txt = '、'.join('%s %d 次' % (OUTCOME_ZH.get(k, k), v) for k, v in sorted(fails.items()))
        parts.append('%s：%s' % (DISPLAY.get(m, m), txt if txt else '无失败'))
    return '；'.join(parts) if parts else '本次运行无该场景数据'


def sanity_report(rows):
    flagged = [r for r in rows if r['flags']]
    lines = ['sanity flags (%d of %d trials flagged)' % (len(flagged), len(rows)),
             'flags: status_gap = status stream gap > 0.5 s before arrival; no_cycles = zero '
             'planning cycles; goal_subs<2 = goal message had < 2 subscribers; late_start; '
             'retried = attempts > 1; rtf<0.9']
    lines += ['%s: %s' % (r['trial_id'], r['flags'].replace(';', ', ')) for r in flagged]
    if not flagged:
        lines.append('no flagged trials')
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------- LaTeX tables
STD_NOTE = '$\\pm$ 前为均值，后为样本标准差（$n-1$ 归一化；样本不足 2 个时只列均值）'
ESTOP_NOTE = ('急停为各方法统一口径：自首个非零速度指令起至到达（未到达则至试验结束）止，'
              '下发的 $\\mathrm{cmd\\_vel}$ 恰为全零的上升沿次数')
EG_NOTE = '$e_g$ 为到达后停稳窗口结束时的目标距离'
EG_LEGACY_NOTE = '（无停稳记录的旧数据中 $e_g$ 取到达时刻距离）'


def static_table(summ, scene, legacy_eg=False):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{静态环境测试统计表}', '\t\\label{tab:静态仿真结果}', '\t\\small',
             '\t\\begin{tabular}{lcccccccc}', '\t\t\\toprule',
             '\t\t方法 & $SR$/\\% & $T$/s & $L$/m & $e_g$/m & $d_{\\mathrm{obs}}^{\\min}$/m'
             ' & $J_{v}$ & $J_{\\omega}$ & 平均耗时/ms \\\\', '\t\t\\midrule']
    for m in METHODS:
        a = summ.get((scene, m))
        if a is None:
            lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(['--'] * 8)))
            continue
        cells = [f1(a['SR']), pm(a['T_mean'], a['T_std']), pm(a['L_mean'], a['L_std']),
                 pm(a['e_g_mean'], a['e_g_std']), pm(a['d_min_body_mean'], a['d_min_body_std']),
                 pm(a['J_v_mean'], a['J_v_std']), pm(a['J_w_mean'], a['J_w_std']),
                 f2(a['t_avg'])]
        lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(cells)))
    present = [m for m in METHODS if (scene, m) in summ]
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：表中 ' + STD_NOTE + '。$T$、$L$、$e_g$、$J_{v}$、'
              '$J_{\\omega}$ 仅对成功试验统计，$T$ 为从发出目标到首次进入到达阈值的时间，'
              '$J_{v}=\\sum\\|\\Delta(v_x,v_y)\\|^2$、$J_{\\omega}=\\sum(\\Delta\\omega)^2$ '
              '取至到达时刻的下发指令；' + EG_NOTE + (EG_LEGACY_NOTE if legacy_eg else '') +
              '；$d_{\\mathrm{obs}}^{\\min}$ 为全部试验各自最小物理轮廓间隙的均值$\\pm$样本标准差；'
              '平均耗时由该方法全部试验（含失败试验）的全部规划周期合并计算。'
              '失败类型计数——' + failure_note(summ, scene, present) + '。',
              r'\end{table}']
    return '\n'.join(lines)


def dynamic_table(summ, scene):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{复杂动态环境测试统计表}', '\t\\label{tab:动态仿真结果}', '\t\\small',
             '\t\\begin{tabular}{lcccccccc}', '\t\t\\toprule',
             '\t\t方法 & $SR$/\\% & 碰撞/次 & 急停/次 & $T$/s & $d_{\\mathrm{obs}}^{\\min}$/m'
             ' & $J_{v}$ & $J_{\\omega}$ & 平均耗时/ms \\\\', '\t\t\\midrule']
    for m in METHODS:
        a = summ.get((scene, m))
        if a is None:
            lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(['--'] * 8)))
            continue
        cells = [f1(a['SR']), str(a['collisions_total']), str(a['estops_total']),
                 pm(a['T_mean'], a['T_std']), pm(a['d_min_body_mean'], a['d_min_body_std']),
                 pm(a['J_v_mean'], a['J_v_std']), pm(a['J_w_mean'], a['J_w_std']),
                 f2(a['t_avg'])]
        lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(cells)))
    present = [m for m in METHODS if (scene, m) in summ]
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：表中 ' + STD_NOTE + '。碰撞、急停为全部试验的'
              '累计次数；' + ESTOP_NOTE + '；$T$、$J_{v}$、$J_{\\omega}$（定义同静态表）仅对成功试验统计；'
              '$d_{\\mathrm{obs}}^{\\min}$ 为全部试验各自最小物理轮廓间隙的均值$\\pm$样本标准差；'
              '平均耗时由该方法全部试验（含失败试验）的全部规划周期合并计算。失败类型计数——'
              + failure_note(summ, scene, present) + '。',
              r'\end{table}']
    return '\n'.join(lines)


def matched_table(msumm, scenes):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{配对成功试验的指标比较}', '\t\\label{tab:配对成功比较}', '\t\\small',
             '\t\\begin{tabular}{llccccc}', '\t\t\\toprule',
             '\t\t场景 & 方法 & 配对数 & $T$/s & $L$/m & $J_{v}$ & $J_{\\omega}$ \\\\']
    for scene in scenes:
        lines.append('\t\t\\midrule')
        for mi, m in enumerate(METHODS):
            a = msumm.get((scene, m))
            head = '\\multirow{%d}{*}{%s}' % (len(METHODS), scene) if mi == 0 else ''
            if a is None:
                cells = ['--'] * 5
            else:
                cells = [str(a['n_matched']), pm(a['T_mean'], a['T_std']),
                         pm(a['L_mean'], a['L_std']), pm(a['J_v_mean'], a['J_v_std']),
                         pm(a['J_w_mean'], a['J_w_std'])]
            lines.append('\t\t%s & %s & %s \\\\' % (head, DISPLAY[m], ' & '.join(cells)))
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：每个场景只统计三种方法在同一（起终点对，随机种子）'
              '下全部成功的试验（配对数为该集合的大小），三种方法的均值在同一批试验上计算，'
              '避免因成功试验集合不同造成的偏差；表中 ' + STD_NOTE + '。',
              r'\end{table}']
    return '\n'.join(lines)


def accel_table(acc, scenes):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{下发指令的执行加速度统计}', '\t\\label{tab:执行加速度}', '\t\\small',
             '\t\\begin{tabular}{llccccccc}', '\t\t\\toprule',
             '\t\t场景 & 方法 & 平移峰值/(m/s$^2$) & 平移 $p_{95}$ & 角峰值/(rad/s$^2$) '
             '& 角 $p_{95}$ & 单步 $\\|\\Delta v\\|_{\\max}$/(m/s) '
             '& 单步 $|\\Delta\\omega|_{\\max}$/(rad/s) & 零指令转换/次 \\\\']
    for scene in scenes:
        lines.append('\t\t\\midrule')
        for mi, m in enumerate(METHODS):
            a = acc.get((scene, m))
            head = '\\multirow{%d}{*}{%s}' % (len(METHODS), scene) if mi == 0 else ''
            if a is None or not a['n_steps']:
                cells = ['--'] * 7
            else:
                cells = [f2(a['nom_peak_lin']), f2(a['nom_p95_lin']), f2(a['nom_peak_ang']),
                         f2(a['nom_p95_ang']), '%.3f' % a['dv_max'], '%.3f' % a['dw_max'],
                         str(a['n_zero_trans'])]
            lines.append('\t\t%s & %s & %s \\\\' % (head, DISPLAY[m], ' & '.join(cells)))
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：对各方法按同一口径统计：取 $\\mathrm{cmd\\_vel}$ '
              '记录中相邻两条下发指令的单步变化量 $\\|\\Delta(v_x,v_y)\\|$ 与 $|\\Delta\\omega|$，'
              '除以名义控制周期 0.1 s 得到每个控制周期的执行加速度，取该场景、该方法全部试验'
              '（至到达时刻，失败试验至结束）合并后的峰值与 95 分位数；末两列为单步变化量的峰值。'
              '不用相邻指令的接收时间间隔作分母：记录节点给指令打的接收时间戳存在数十毫秒的抖动'
              '（相邻间隔 0.02～0.18 s），这是记录端时间戳效应而非规划器指令的变化，会把正常步长'
              '放大成假峰值，按接收时间计算的峰值与分位数仅保存在 accel.csv。涉及全零指令的相邻对'
              '（启动、急停、到达）不计入，零指令与非零指令之间的转换次数列于第九列。速度为车体系'
              '指令，车体转向时该坐标系随之旋转，会带来额外的旋转分量。',
              r'\end{table}']
    return '\n'.join(lines)


def timing_table(summ, scenes):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{局部规划实时性统计表}', '\t\\label{tab:仿真实时性}',
             '\t\\begin{tabular}{llccccc}', '\t\t\\toprule',
             '\t\t场景 & 方法 & $t_{\\mathrm{avg}}$/ms & $t_{95}$/ms & $t_{\\max}$/ms'
             ' & $r_{\\mathrm{over}}$/\\% & RTF \\\\']
    for scene in scenes:
        lines.append('\t\t\\midrule')
        for mi, m in enumerate(METHODS):
            a = summ.get((scene, m))
            head = '\\multirow{%d}{*}{%s}' % (len(METHODS), scene) if mi == 0 else ''
            if a is None or not a['n_cycles']:
                cells = ['--'] * 5
            else:
                cells = [f2(a['t_avg']), f2(a['t95']), f2(a['t_max']), f2(a['r_over']),
                         f2(a.get('rtf_mean', NAN))]
            lines.append('\t\t%s & %s & %s \\\\' % (head, DISPLAY[m], ' & '.join(cells)))
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：统计量由各场景、各方法全部试验（含失败试验）的'
              '全部单周期规划耗时样本合并后计算；RTF 为仿真时间与墙钟时间之比的各试验均值'
              '（旧数据无记录时为 --）。',
              r'\end{table}']
    return '\n'.join(lines)


def write_tables(path, summ, scene_methods, msumm=None, acc=None, legacy_eg=False):
    """All tables are always emitted; cells of absent scene/method combos are '--'."""
    scenes = [s for s in ('S1', 'S2')] + sorted(set(scene_methods) - {'S1', 'S2'})
    blocks = ['% generated by experiment/scripts/analyze.py -- do not edit by hand',
              static_table(summ, 'S1', legacy_eg), dynamic_table(summ, 'S2'),
              matched_table(msumm or {}, scenes), accel_table(acc or {}, scenes),
              timing_table(summ, scenes)]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(blocks) + '\n')


# ----------------------------------------------------------------------------- figures
def setup_matplotlib():
    """Returns (plt, use_chinese). Picks a CJK-capable family; warns and falls back if none."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in FONT_CANDIDATES if n in names]
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['axes.unicode_minus'] = False
    if chosen:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = chosen + list(
            matplotlib.rcParams['font.sans-serif'])
        return plt, True
    print('WARNING: no Chinese font found (%s); figure labels fall back to English.'
          % ', '.join(FONT_CANDIDATES), file=sys.stderr)
    return plt, False


def representative(rows):
    """(row, failed): the successful trial with median T (lower median for even counts), or,
    if none succeeded, the last failed trial (highest seed) with failed=True."""
    ok = sorted([r for r in rows if r['success']], key=lambda r: (r['T'], r['seed']))
    if ok:
        return ok[(len(ok) - 1) // 2], False
    return sorted(rows, key=lambda r: r['seed'])[-1], True


def read_csv_cols(path):
    """{column: float array} from a trial csv ('nan' parsed as NaN); None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd, None)
        data = [r for r in rd if r]
    if not head or not data:
        return None
    cols = {}
    for i, k in enumerate(head):
        vals = []
        for r in data:
            try:
                vals.append(float(r[i]))
            except (ValueError, IndexError):
                vals.append(NAN)
        cols[k] = np.array(vals)
    return cols


def load_world(world_dir, world, cache):
    if world not in cache:
        p = os.path.join(world_dir, world + '.world')
        cache[world] = parse_world(p) if world and os.path.exists(p) else None
        if cache[world] is None:
            print('WARNING: world file %s not found; obstacles not drawn.' % p, file=sys.stderr)
    return cache[world]


def draw_world(ax, specs):
    from matplotlib.patches import Circle, Polygon
    for sp in specs or []:
        if sp.dynamic:
            if len(sp.path) > 1:
                pts = np.array(sp.path)
                if len(pts) > 2:
                    pts = np.vstack([pts, pts[:1]])
                ax.plot(pts[:, 0], pts[:, 1], color='#7f7f7f', ls=(0, (4, 3)), lw=1.0, zorder=1)
            continue
        if sp.kind == 'box':
            ax.add_patch(Polygon(rect_polygon(sp.x, sp.y, sp.yaw, *sp.size), closed=True,
                                 fc='#c8c8c8', ec='#555555', lw=0.6, zorder=1))
        else:
            ax.add_patch(Circle((sp.x, sp.y), sp.size[0], fc='#c8c8c8', ec='#555555', lw=0.6,
                                zorder=1))


def save_pdf(fig, path):
    """Save a vector PDF with the CJK font subset. matplotlib 3.1's own PDF backend embeds the
    whole Noto Sans CJK font (~16 MB per file; Type 3 cannot handle its CFF outlines), so use the
    Cairo canvas (pycairo) when available and fall back to the matplotlib backend otherwise."""
    try:
        from matplotlib.backends.backend_cairo import FigureCanvasCairo
        FigureCanvasCairo(fig).print_pdf(path)
        return
    except Exception as e:   # pycairo missing / backend error
        print('WARNING: cairo PDF output unavailable (%s); using matplotlib backend '
              '(large files with CJK fonts).' % e, file=sys.stderr)
    fig.savefig(path)


def plot_traj(plt, zh, path, scene, pair, reps, specs):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_world(ax, specs)
    start = goal = None
    for m, (row, failed) in reps.items():
        c = read_csv_cols(row['base'] + '.csv')
        if c is None:
            continue
        label = DISPLAY[m] if zh else {'neupan': 'Ours', 'aabb': 'AABB-MPC', 'dwa': 'DWA'}[m]
        if failed:
            label += '（失败）' if zh else ' (failed)'
        ax.plot(c['x'], c['y'], color=COLORS[m], ls=STYLES[m], lw=1.6, label=label, zorder=3)
        if failed:
            ax.plot(c['x'][-1], c['y'][-1], 'x', color=COLORS[m], ms=8, mew=2, zorder=4)
        start, goal = row['start'], row['goal']
    if start is not None:
        ax.plot(start[0], start[1], 'o', color='k', ms=7, zorder=5,
                label='起点' if zh else 'start')
        ax.plot(goal[0], goal[1], '*', color='#CC79A7', ms=14, mec='k', mew=0.6, zorder=5,
                label='终点' if zh else 'goal')
    if any(sp.dynamic for sp in specs or []):
        ax.plot([], [], color='#7f7f7f', ls=(0, (4, 3)), lw=1.0,
                label='动态障碍路径' if zh else 'dynamic obstacle path')
    ax.set_aspect('equal')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(('%s 起终点对 %d 代表轨迹' if zh else '%s pair %d representative trajectories')
                 % (scene, pair))
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save_pdf(fig, path)
    plt.close(fig)


def plot_ts(plt, zh, path, scene, pair, reps):
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True)
    ylabels = (('合速度 / (m/s)', '到目标距离 / m', '$d_{body}$ / m') if zh else
               ('speed / (m/s)', 'distance to goal / m', '$d_{body}$ / m'))
    for m, (row, failed) in reps.items():
        c = read_csv_cols(row['base'] + '.csv')
        cmd = read_csv_cols(row['base'] + '_cmd.csv')
        label = DISPLAY[m] if zh else {'neupan': 'Ours', 'aabb': 'AABB-MPC', 'dwa': 'DWA'}[m]
        if failed:
            label += '（失败）' if zh else ' (failed)'
        kw = dict(color=COLORS[m], ls=STYLES[m], lw=1.3, label=label)
        if cmd is not None:
            axs[0].plot(cmd['t'], np.hypot(cmd['vx'], cmd['vy']), **kw)
        if c is not None:
            axs[1].plot(c['t'], c['dist_goal'], **kw)
            axs[2].plot(c['t'], c['gap_body'], **kw)
    axs[2].axhline(0.0, color='k', lw=0.8)
    for ax, yl in zip(axs, ylabels):
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, lw=0.4)
    axs[2].set_xlabel('t / s')
    axs[0].legend(loc='best', fontsize=8)
    axs[0].set_title(('%s 起终点对 %d 时序曲线' if zh else '%s pair %d time series')
                     % (scene, pair))
    fig.tight_layout()
    save_pdf(fig, path)
    plt.close(fig)


def plot_plan_box(plt, zh, path, rows, scene_methods):
    from matplotlib import cbook
    scenes = sorted(scene_methods)
    fig, axs = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes) + 0.8, 4.0),
                            sharey=True, squeeze=False)
    for ax, scene in zip(axs[0], scenes):
        data, labels, cols = [], [], []
        for m in METHODS:
            p = [v for r in rows if r['scene'] == scene and r['method'] == m
                 for v in r['plan_ms_all']]
            if p:
                data.append(p)
                labels.append(DISPLAY[m] if zh else
                              {'neupan': 'Ours', 'aabb': 'AABB-MPC', 'dwa': 'DWA'}[m])
                cols.append(COLORS[m])
        if data:
            # bxp on precomputed stats: boxplot() on ragged lists breaks with newer numpy
            stats = []
            for d, lab in zip(data, labels):
                st = cbook.boxplot_stats(np.asarray(d, float))[0]
                st['label'] = lab
                stats.append(st)
            bp = ax.bxp(stats, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=2, alpha=0.4),
                        medianprops=dict(color='k'))
            for patch, c in zip(bp['boxes'], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
        ax.axhline(100.0, color='#CC0000', ls=':', lw=1.0,
                   label='100 ms 预算' if zh else '100 ms budget')
        ax.set_title(scene)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
    axs[0][0].set_ylabel('单周期规划耗时 / ms' if zh else 'planning time per cycle / ms')
    axs[0][-1].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    save_pdf(fig, path)
    plt.close(fig)


def make_figures(out, rows, scene_methods, world_dir):
    plt, zh = setup_matplotlib()
    worlds = {}
    pairs = sorted({(r['scene'], r['pair']) for r in rows})
    made = []
    for scene, pair in pairs:
        reps = {}
        for m in METHODS:
            rs = [r for r in rows if (r['scene'], r['pair'], r['method']) == (scene, pair, m)
                  and os.path.exists(r['base'] + '.csv')]
            if rs:
                reps[m] = representative(rs)
        if not reps:
            continue
        world = next(iter(reps.values()))[0]['world']
        specs = load_world(world_dir, world, worlds)
        p1 = os.path.join(out, 'traj_%s_p%d.pdf' % (scene, pair))
        p2 = os.path.join(out, 'ts_%s_p%d.pdf' % (scene, pair))
        plot_traj(plt, zh, p1, scene, pair, reps, specs)
        plot_ts(plt, zh, p2, scene, pair, reps)
        made += [p1, p2]
    if any(r['plan_ms_all'] for r in rows):
        p = os.path.join(out, 'plan_time_box.pdf')
        plot_plan_box(plt, zh, p, rows, scene_methods)
        made.append(p)
    return made


# ----------------------------------------------------------------------------- main
TRIAL_HEADER = ['trial_id', 'scene', 'method', 'pair', 'seed', 'outcome', 'success',
                'end_reason', 'T', 'L', 'e_g', 'e_g_arrive', 't_arrive', 'settle_s', 'd_min_np',
                'd_min_body', 'collisions', 'estops', 'estops_planner', 'estops_src', 'J_du',
                'J_v', 'J_w', 't_avg', 't95', 't_max', 'r_over', 'n_cycles', 'rtf', 'wall_s',
                'attempts', 'late_start', 'goal_subscribers', 'code_check', 'status_gap_max',
                'runner_patched', 'flags']
SUMMARY_KEYS = ['n_trials', 'n_success', 'SR', 'T_mean', 'T_std', 'L_mean', 'L_std', 'e_g_mean',
                'e_g_std', 'J_du_mean', 'J_du_std', 'J_v_mean', 'J_v_std', 'J_w_mean', 'J_w_std',
                'd_min_body_min', 'd_min_body_mean',
                'd_min_body_std', 'd_min_np_min', 'd_min_np_mean', 'd_min_np_std',
                'collisions_total', 'collisions_mean', 'estops_total', 'estops_mean',
                'estops_planner_total', 'estops_planner_mean',
                't_avg', 't95', 't_max', 'r_over', 'n_cycles', 'rtf_mean', 'outcomes']
MATCHED_KEYS = ['n_matched'] + [k + sfx for k in MATCH_KEYS for sfx in ('_mean', '_std')]
ACCEL_KEYS = ['nom_peak_lin', 'nom_p95_lin', 'nom_peak_ang', 'nom_p95_ang', 'dv_max', 'dv_p95',
              'dw_max', 'dw_p95', 'n_steps', 'n_zero_trans', 'n_sub_cycle', 'n_trials',
              # receive-time view (inflated by timestamp jitter)
              'peak_lin', 'p95_lin', 'peak_ang', 'p95_ang', 'n_pairs', 'peak_lin_cyc',
              'p95_lin_cyc', 'peak_ang_cyc', 'p95_ang_cyc', 'n_pairs_cyc']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', required=True, help='run directory (contains trials/)')
    ap.add_argument('--out', default=None, help='default: <results>/report')
    ap.add_argument('--world-dir', default=DEFAULT_WORLD_DIR)
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, 'report')
    rows = load_trials(a.results)
    if not rows:
        print('no trial json found under %s/trials' % a.results, file=sys.stderr)
        return 1
    os.makedirs(out, exist_ok=True)
    summ = summarize(rows)
    _, scene_methods = group(rows)

    write_csv(os.path.join(out, 'all_trials.csv'), TRIAL_HEADER, rows)
    for scene, methods in scene_methods.items():
        recs = []
        for m in methods:
            s = dict(summ[(scene, m)])
            s['outcomes'] = outcomes_str(s['outcomes'])
            s.update(scene=scene, method=m)
            recs.append(s)
        write_csv(os.path.join(out, 'summary_%s.csv' % scene), ['scene', 'method'] + SUMMARY_KEYS,
                  recs)
    timing = []
    for scene in sorted(scene_methods):
        for m in scene_methods[scene]:
            s = summ[(scene, m)]
            timing.append(dict(scene=scene, method=m, t_avg=s['t_avg'], t95=s['t95'],
                               t_max=s['t_max'], r_over=s['r_over'], n_cycles=s['n_cycles'],
                               rtf_mean=s['rtf_mean']))
    write_csv(os.path.join(out, 'timing.csv'),
              ['scene', 'method', 't_avg', 't95', 't_max', 'r_over', 'n_cycles', 'rtf_mean'],
              timing)
    matched = matched_sets(rows)
    msumm = matched_summary(rows, matched)
    for scene, methods in scene_methods.items():
        write_csv(os.path.join(out, 'summary_matched_%s.csv' % scene),
                  ['scene', 'method'] + MATCHED_KEYS,
                  [dict(msumm[(scene, m)], scene=scene, method=m) for m in methods])
    acc = accel_summary(rows)
    write_csv(os.path.join(out, 'accel.csv'), ['scene', 'method'] + ACCEL_KEYS,
              [dict(acc[(sc, m)], scene=sc, method=m)
               for sc in sorted(scene_methods) for m in scene_methods[sc]])
    with open(os.path.join(out, 'sanity.txt'), 'w', encoding='utf-8') as f:
        f.write(sanity_report(rows))
    # legacy = an arrived trial whose json predates fix F2 (no `settle_s` key at all); a new-format
    # timeout/stuck trial has settle_s = null and is not legacy
    legacy_eg = any(not r['has_settle_key'] for r in rows if r['end_reason'] == 'arrived')
    write_tables(os.path.join(out, 'tables.tex'), summ, scene_methods, msumm, acc, legacy_eg)
    made = [] if a.no_figures else make_figures(out, rows, scene_methods, a.world_dir)
    print('analyzed %d trials -> %s (%d figures)' % (len(rows), out, len(made)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
