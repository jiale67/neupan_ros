#!/usr/bin/env python3
"""Turn a batch run's per-trial logs into the thesis tables (LaTeX + CSV) and figures.

    python3 scripts/analyze.py --results <run_dir> [--out <run_dir>/report]

Reads <run>/trials/<trial_id>.json (+ .csv / _cmd.csv for figures); never writes outside --out.
Outputs: all_trials.csv, summary_<scene>.csv, timing.csv, tables.tex, traj_<scene>_p<k>.pdf,
ts_<scene>_p<k>.pdf, plan_time_box.pdf.
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
from neupan_exp.metrics import aggregate  # noqa: E402
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

ROW_KEYS = ('T', 'L', 'e_g', 'd_min_np', 'd_min_body', 'collisions', 'estops', 'J_du', 't_avg',
            't95', 't_max', 'r_over', 'n_cycles')


# ----------------------------------------------------------------------------- loading
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
                 json_path=p, base=os.path.join(results, 'trials', tid))
        rows.append(r)
    rows.sort(key=lambda r: (r['scene'], r['pair'], r['method'], r['seed']))
    return rows


# ----------------------------------------------------------------------------- formatting
def _finite(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def f2(v):
    return '%.2f' % v if _finite(v) else '--'


def pm(mean, std):
    return '$%.2f\\pm %.2f$' % (mean, std) if _finite(mean) and _finite(std) else '--'


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


def summarize(rows):
    """{(scene, method): aggregate dict + method/scene/pooled cycle count}."""
    g, _ = group(rows)
    out = {}
    for key, rs in g.items():
        a = aggregate(rs)
        a['n_cycles'] = int(sum(len(r['plan_ms_all']) for r in rs))
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


# ----------------------------------------------------------------------------- LaTeX tables
def static_table(summ, scene):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{静态环境测试统计表}', '\t\\label{tab:静态仿真结果}', '\t\\small',
             '\t\\begin{tabular}{lccccccc}', '\t\t\\toprule',
             '\t\t方法 & $SR$/\\% & $T$/s & $L$/m & $e_g$/m & $d_{\\mathrm{obs}}^{\\min}$/m'
             ' & $J_{\\Delta u}$ & 平均耗时/ms \\\\', '\t\t\\midrule']
    for m in METHODS:
        a = summ.get((scene, m))
        if a is None:
            lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(['--'] * 7)))
            continue
        cells = [f1(a['SR']), pm(a['T_mean'], a['T_std']), pm(a['L_mean'], a['L_std']),
                 pm(a['e_g_mean'], a['e_g_std']), pm(a['d_min_body_mean'], a['d_min_body_std']),
                 pm(a['J_du_mean'], a['J_du_std']), f2(a['t_avg'])]
        lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(cells)))
    present = [m for m in METHODS if (scene, m) in summ]
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：$T$、$L$、$e_g$、$J_{\\Delta u}$ 仅对成功试验'
              '统计；$d_{\\mathrm{obs}}^{\\min}$ 为全部试验各自最小物理轮廓间隙的均值$\\pm$标准差。'
              '失败类型计数——' + failure_note(summ, scene, present) + '。',
              r'\end{table}']
    return '\n'.join(lines)


def dynamic_table(summ, scene):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{复杂动态环境测试统计表}', '\t\\label{tab:动态仿真结果}', '\t\\small',
             '\t\\begin{tabular}{lccccccc}', '\t\t\\toprule',
             '\t\t方法 & $SR$/\\% & 碰撞/次 & 急停/次 & $T$/s & $d_{\\mathrm{obs}}^{\\min}$/m'
             ' & $J_{\\Delta u}$ & 平均耗时/ms \\\\', '\t\t\\midrule']
    for m in METHODS:
        a = summ.get((scene, m))
        if a is None:
            lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(['--'] * 7)))
            continue
        cells = [f1(a['SR']), str(a['collisions_total']), str(a['estops_total']),
                 pm(a['T_mean'], a['T_std']), pm(a['d_min_body_mean'], a['d_min_body_std']),
                 pm(a['J_du_mean'], a['J_du_std']), f2(a['t_avg'])]
        lines.append('\t\t%s & %s \\\\' % (DISPLAY[m], ' & '.join(cells)))
    present = [m for m in METHODS if (scene, m) in summ]
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：碰撞、急停为全部试验的累计次数；$T$、'
              '$J_{\\Delta u}$ 仅对成功试验统计；$d_{\\mathrm{obs}}^{\\min}$ 为全部试验各自最小'
              '物理轮廓间隙的均值$\\pm$标准差。失败类型计数——'
              + failure_note(summ, scene, present) + '。',
              r'\end{table}']
    return '\n'.join(lines)


def timing_table(summ, scenes):
    lines = [r'\begin{table}[!htb]', '\t\\centering',
             '\t\\caption{局部规划实时性统计表}', '\t\\label{tab:仿真实时性}',
             '\t\\begin{tabular}{llcccc}', '\t\t\\toprule',
             '\t\t场景 & 方法 & $t_{\\mathrm{avg}}$/ms & $t_{95}$/ms & $t_{\\max}$/ms'
             ' & $r_{\\mathrm{over}}$/\\% \\\\']
    for scene in scenes:
        lines.append('\t\t\\midrule')
        for mi, m in enumerate(METHODS):
            a = summ.get((scene, m))
            head = '\\multirow{%d}{*}{%s}' % (len(METHODS), scene) if mi == 0 else ''
            if a is None or not a['n_cycles']:
                cells = ['--'] * 4
            else:
                cells = [f2(a['t_avg']), f2(a['t95']), f2(a['t_max']), f2(a['r_over'])]
            lines.append('\t\t%s & %s & %s \\\\' % (head, DISPLAY[m], ' & '.join(cells)))
    lines += ['\t\t\\bottomrule', '\t\\end{tabular}',
              '\t\\par\\vspace{2pt}\\footnotesize 注：统计量由各场景、各方法全部试验的单周期'
              '规划耗时样本合并后计算。',
              r'\end{table}']
    return '\n'.join(lines)


def write_tables(path, summ, scene_methods):
    """All three tables are always emitted; cells of absent scene/method combos are '--'."""
    scenes = [s for s in ('S1', 'S2')] + sorted(set(scene_methods) - {'S1', 'S2'})
    blocks = ['% generated by experiment/scripts/analyze.py -- do not edit by hand',
              static_table(summ, 'S1'), dynamic_table(summ, 'S2'), timing_table(summ, scenes)]
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
                'end_reason', 'T', 'L', 'e_g', 'd_min_np', 'd_min_body', 'collisions', 'estops',
                'J_du', 't_avg', 't95', 't_max', 'r_over', 'n_cycles', 'attempts', 'late_start',
                'runner_patched']
SUMMARY_KEYS = ['n_trials', 'n_success', 'SR', 'T_mean', 'T_std', 'L_mean', 'L_std', 'e_g_mean',
                'e_g_std', 'J_du_mean', 'J_du_std', 'd_min_body_min', 'd_min_body_mean',
                'd_min_body_std', 'd_min_np_min', 'd_min_np_mean', 'd_min_np_std',
                'collisions_total', 'collisions_mean', 'estops_total', 'estops_mean',
                't_avg', 't95', 't_max', 'r_over', 'n_cycles', 'outcomes']


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
                               t_max=s['t_max'], r_over=s['r_over'], n_cycles=s['n_cycles']))
    write_csv(os.path.join(out, 'timing.csv'),
              ['scene', 'method', 't_avg', 't95', 't_max', 'r_over', 'n_cycles'], timing)
    write_tables(os.path.join(out, 'tables.tex'), summ, scene_methods)
    made = [] if a.no_figures else make_figures(out, rows, scene_methods, a.world_dir)
    print('analyzed %d trials -> %s (%d figures)' % (len(rows), out, len(made)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
