#!/usr/bin/env python3
"""Write the LaTeX tables and the in-text numbers of the v12 manuscript from
out_v12/. Nothing in the paper's tables is typed by hand.

    python tables_v12.py --out ../../reports/v12/generated
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_v12')
SC = ['ideal', 'mismatch', 'leakfree', 'severe']
SC_TEX = {'ideal': '\\textsc{aligned}', 'mismatch': '\\textsc{mismatch}',
          'leakfree': '\\textsc{leakfree}', 'severe': '\\textsc{severe}'}
T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201}   # t_{0.975, n-1} for n = 3..12


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def heldout_blocks():
    """H[cfg][sc][seed][policy] = summary dict."""
    H = defaultdict(lambda: defaultdict(dict))
    for d in glob.glob(os.path.join(OUT, 'heldout', '*', '*_s*')):
        cfg = os.path.basename(os.path.dirname(d))
        m = re.match(r'(\w+)_s(\d+)$', os.path.basename(d))
        if not m:
            continue
        sc, seed = m.group(1), int(m.group(2))
        for f in glob.glob(os.path.join(d, '*_summary.json')):
            H[cfg][sc].setdefault(seed, {})[os.path.basename(f)[:-13]] = load(f)
    return H


def short_blocks():
    S = defaultdict(lambda: defaultdict(dict))
    for d in glob.glob(os.path.join(OUT, 'short', '*', 's*')):
        cfg = os.path.basename(os.path.dirname(d))
        seed = int(os.path.basename(d)[1:])
        for sc in SC:
            for f in glob.glob(os.path.join(d, sc, '*_summary.json')):
                S[cfg][sc].setdefault(seed, {})[os.path.basename(f)[:-13]] = load(f)
    return S


# learner key -> (configuration directory, policy file stem)
LEARNERS = {
    'none':      ('anom', 'rl_no_ode_trained'),
    'flat':      ('anom', 'rl_flat_trained'),
    'det':       ('blind_det', 'rl_trained'),
    'logdet':    ('blind_logdet', 'rl_trained'),
    'task':      ('anom', 'rl_trained'),
    'task_init': ('anom', 'rl_initonly_trained'),
    'logdet_init': ('blind_logdet', 'rl_initonly_trained'),
    'ode_det':   ('blind_det', 'ode'),
    'ode_logdet': ('blind_logdet', 'ode'),
    'ode_task':  ('anom', 'ode'),
    'rand':      ('anom', 'rand'),
}


def series(H, key, sc, metric='f1_macro'):
    cfg, pol = LEARNERS[key]
    out = {}
    for seed, pols in H[cfg][sc].items():
        if pol in pols:
            s = pols[pol]
            out[seed] = (s['mission_resolution']['resolution_rate'] if metric == 'miss'
                         else s['clu_at_1'] if metric == 'clu1' else s[metric])
    return out


def paired(H, a, b, sc, metric='f1_macro'):
    A, B = series(H, a, sc, metric), series(H, b, sc, metric)
    seeds = sorted(set(A) & set(B))
    d = np.array([A[s] - B[s] for s in seeds])
    n = len(d)
    if n < 2:
        return None
    m, sd = d.mean(), d.std(ddof=1)
    h = T975.get(n, 1.96) * sd / np.sqrt(n)
    return {'n': n, 'mean': m, 'lo': m - h, 'hi': m + h, 'pos': int((d > 0).sum()), 'neg': int((d < 0).sum())}


def mean_sd(H, key, sc, metric='f1_macro'):
    v = np.array(list(series(H, key, sc, metric).values()))
    if len(v) == 0:
        return None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt_diff(p):
    if p is None:
        return '--'
    star = '$^{\\ast}$' if (p['lo'] > 0 or p['hi'] < 0) else ''
    return f"${p['mean']:+.3f}${star}\\,[${p['lo']:+.3f}$,\\,${p['hi']:+.3f}$]"


def tab_heldout(H, dst):
    rows = [('none', '$Q_0=0$ (no prior)'), ('flat', '$Q_0=\\beta$ (flat optimism)'),
            ('det', '\\textsc{rl}, agnostic $\\det$'), ('logdet', '\\textsc{rl}, agnostic $\\log\\det$'),
            ('task', '\\textsc{rl}, task-aware'), ('task_init', '\\textsc{init}, task-aware'),
            ('ode_det', '\\textsc{ode}, agnostic'), ('ode_task', '\\textsc{ode}, task-aware'),
            ('rand', '\\textsc{rand}')]
    L = []
    best = {sc: max(mean_sd(H, k, sc)[0] for k, _ in rows if mean_sd(H, k, sc)) for sc in SC}
    for k, lbl in rows:
        cells = []
        for sc in SC:
            ms = mean_sd(H, k, sc)
            if ms is None:
                cells.append('--')
                continue
            m, sd, n = ms
            txt = f'{m:.3f}\\,({sd:.3f})'
            cells.append(f'\\textbf{{{m:.3f}}}\\,({sd:.3f})' if abs(m - best[sc]) < 5e-4 else txt)
        L.append(f'{lbl} & ' + ' & '.join(cells) + ' \\\\')
        if k in ('flat', 'task_init'):
            L.append('\\midrule')
    n = mean_sd(H, 'task', 'ideal')[2]
    with open(os.path.join(dst, 'tab_heldout_rows.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    return n


def tab_contrasts(H, dst):
    rows = [('flat', 'none', 'optimism', '$Q_0=\\beta$ vs.\\ $Q_0=0$'),
            ('task', 'flat', 'information, task-aware', 'task-aware vs.\\ flat'),
            ('logdet', 'flat', 'information, agnostic', 'agnostic $\\log\\det$ vs.\\ flat'),
            ('det', 'flat', 'information, degenerate', 'agnostic $\\det$ vs.\\ flat'),
            ('task', 'logdet', 'specification', 'task-aware vs.\\ agnostic $\\log\\det$'),
            ('task', 'task_init', 'biased exploration', '\\textsc{rl} vs.\\ \\textsc{init} (task-aware)')]
    L = []
    for a, b, what, lbl in rows:
        cells = [fmt_diff(paired(H, a, b, sc)) for sc in SC]
        L.append(f'{what} & {lbl} & ' + ' & '.join(cells) + ' \\\\')
    with open(os.path.join(dst, 'tab_contrasts_rows.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def tab_short(S, dst):
    rows = [('rl_no_ode', 'anom', '$Q_0=0$ ($=$ flat, see text)'),
            ('rl', 'blind_det', '\\textsc{rl}, agnostic $\\det$'),
            ('rl', 'blind_logdet', '\\textsc{rl}, agnostic $\\log\\det$'),
            ('rl', 'anom', '\\textsc{rl}, task-aware'),
            ('ode', 'blind_det', '\\textsc{ode}, agnostic'),
            ('ode', 'anom', '\\textsc{ode}, task-aware'),
            ('full', 'anom', '\\textsc{full} payload'),
            ('rand', 'anom', '\\textsc{rand}')]
    L = []
    for pol, cfg, lbl in rows:
        cells = []
        for sc in SC:
            v = [p[pol]['f1_macro'] for p in S[cfg][sc].values() if pol in p]
            cells.append(f'{np.mean(v):.3f}' if v else '--')
        L.append(f'{lbl} & ' + ' & '.join(cells) + ' \\\\')
        if pol == 'rl' and cfg == 'anom':
            L.append('\\midrule')
    with open(os.path.join(dst, 'tab_short_rows.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def tab_bench(dst, tag='J6K6T6', deltas=('0.0', '0.2', '0.5')):
    p = os.path.join(OUT, 'generic', f'summary_{tag}.json')
    S = load(p)
    L = []
    for k, lbl in [('none', '$Q_0=0$'), ('flat', 'flat optimism'), ('agnostic', 'agnostic prior'),
                   ('task', 'task-aware prior')]:
        cells = []
        for h in ('short', 'long'):
            for d in deltas:
                cells.append(f'{S[d][f"{k}_{h}"][0]:.3f}')
        L.append(f'{lbl} & ' + ' & '.join(cells) + ' \\\\')
    L.append('\\midrule')
    for k, lbl in [('agnostic', 'agnostic, no learning'), ('task', 'task-aware, no learning')]:
        cells = ['--'] * len(deltas) + [f'{S[d][f"{k}_prioronly"][0]:.3f}' for d in deltas]
        L.append(f'{lbl} & ' + ' & '.join(cells) + ' \\\\')
    with open(os.path.join(dst, f'tab_bench_rows_{tag}.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def numbers(H, S, dst):
    """Key numbers quoted in the text, as LaTeX macros and a JSON dump."""
    N = {}

    def put(name, val, fmt='{:.3f}'):
        txt = fmt.format(val) if isinstance(val, (float, np.integer, np.floating, int)) else str(val)
        N[name] = f'\\ensuremath{{{txt}}}' if txt[:1] in '+-' else txt

    for sc in SC:
        tag = {'ideal': 'Al', 'mismatch': 'Mi', 'leakfree': 'Lf', 'severe': 'Sv'}[sc]
        for k in LEARNERS:
            ms = mean_sd(H, k, sc)
            if ms:
                put(f'ho{tag}{k.replace("_", "")}', ms[0])
            for met in ('miss', 'clu1'):
                ms = mean_sd(H, k, sc, met)
                if ms:
                    put(f'ho{tag}{k.replace("_", "")}{"clu" if met == "clu1" else met}', ms[0], '{:.2f}')
        for a, b in [('flat', 'none'), ('task', 'flat'), ('logdet', 'flat'), ('det', 'flat'),
                     ('task', 'logdet'), ('task', 'none'), ('task', 'task_init')]:
            p = paired(H, a, b, sc)
            if p:
                put(f'd{tag}{a.replace("_", "")}{b.replace("_", "")}', p['mean'], '{:+.3f}')
                put(f'd{tag}{a.replace("_", "")}{b.replace("_", "")}pos', p['pos'], '{}')
        # none learner spread across blocks
        v = list(series(H, 'none', sc).values())
        if v:
            put(f'none{tag}min', min(v))
            put(f'none{tag}max', max(v))
    N['nblocks'] = str(mean_sd(H, 'task', 'ideal')[2])
    # short horizon (zero-shot) means over the seed blocks
    for sc in SC:
        tag = {'ideal': 'Al', 'mismatch': 'Mi', 'leakfree': 'Lf', 'severe': 'Sv'}[sc]
        for pol, cfg, name in [('ode', 'anom', 'odetask'), ('ode', 'blind_det', 'odedet'),
                               ('rl', 'anom', 'rltask'), ('rl', 'blind_logdet', 'rllogdet'),
                               ('rl', 'blind_det', 'rldet'), ('rl_no_ode', 'anom', 'none'),
                               ('rl_flat', 'anom', 'flat'), ('full', 'anom', 'full'), ('rand', 'anom', 'rand')]:
            v = [p[pol]['f1_macro'] for p in S[cfg][sc].values() if pol in p]
            if v:
                put(f'sh{tag}{name}', float(np.mean(v)))
    # random-instance benchmark
    words = {'0.0': 'Zero', '0.1': 'One', '0.2': 'Two', '0.3': 'Three', '0.4': 'Four', '0.5': 'Five'}
    for tag, pre in (('J6K6T6', 'b'), ('J8K8T8', 'bb')):
        p = os.path.join(OUT, 'generic', f'summary_{tag}.json')
        if not os.path.exists(p):
            continue
        B = load(p)
        R = load(os.path.join(OUT, 'generic', f'instances_{tag}.json'))
        N[f'{pre}ninst'] = str(sum(1 for r in R if r['delta'] == 0.0))
        N[f'{pre}nact'] = f"{np.mean([r['n_actions'] for r in R]):.0f}"
        for d, w in words.items():
            if d not in B:
                continue
            for k, K in (('flat', 'Flat'), ('agnostic', 'Agn'), ('task', 'Task')):
                for h, Hh in (('short', 'Short'), ('long', 'Long')):
                    put(f'{pre}{Hh}{K}{w}', B[d][f'diff_{k}_{h}'][0], '{:+.3f}')
            for k, K in (('none', 'None'), ('flat', 'Flat'), ('agnostic', 'Agn'), ('task', 'Task')):
                put(f'{pre}Reg{K}{w}', B[d][f'{k}_long'][0])
            for k, K in (('agnostic', 'Agn'), ('task', 'Task')):
                put(f'{pre}Only{K}{w}', B[d][f'{k}_prioronly'][0])
            X = [r for r in R if abs(r['delta'] - float(d)) < 1e-9]
            g = lambda key: np.array([r[key] for r in X])
            for k, K in (('agnostic', 'Agn'), ('task', 'Task')):
                put(f'{pre}Pct{K}WorseFlat{w}', 100 * np.mean(g(f'{k}_long_regret') > g('flat_long_regret') + 1e-9), '{:.0f}')
                put(f'{pre}Pct{K}WorseNone{w}', 100 * np.mean(g(f'{k}_long_regret') > g('none_long_regret') + 1e-9), '{:.0f}')
                put(f'{pre}Pct{K}OnlyWorse{w}', 100 * np.mean(g(f'{k}_prioronly_regret') > g('none_long_regret')), '{:.0f}')
                dd = g(f'{k}_long_regret') - g('flat_long_regret')
                bs = np.random.default_rng(0).choice(dd, size=(4000, len(dd))).mean(axis=1)
                put(f'{pre}Info{K}{w}', dd.mean(), '{:+.3f}')
                put(f'{pre}Info{K}{w}Lo', np.percentile(bs, 2.5), '{:+.3f}')
                put(f'{pre}Info{K}{w}Hi', np.percentile(bs, 97.5), '{:+.3f}')
    with open(os.path.join(dst, 'numbers.json'), 'w') as f:
        json.dump(N, f, indent=1)
    with open(os.path.join(dst, 'numbers.tex'), 'w') as f:
        f.write('%% generated by scripts/article_v12/tables_v12.py -- do not edit\n')
        for k, v in sorted(N.items()):
            if re.fullmatch(r'[A-Za-z]+', k):
                f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    return N


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'tex_v12'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    H, S = heldout_blocks(), short_blocks()
    n = tab_heldout(H, a.out)
    tab_contrasts(H, a.out)
    tab_short(S, a.out)
    for tag in ('J6K6T6', 'J8K8T8'):
        if os.path.exists(os.path.join(OUT, 'generic', f'summary_{tag}.json')):
            tab_bench(a.out, tag)
    N = numbers(H, S, a.out)
    print('blocks per held-out cell:', n)
    for sc in SC:
        print(sc, {k: N.get(f'd{ {"ideal":"Al","mismatch":"Mi","leakfree":"Lf","severe":"Sv"}[sc]}{k}')
                   for k in ('flatnone', 'taskflat', 'logdetflat', 'detflat', 'tasklogdet', 'tasktaskinit')})
