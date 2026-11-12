#!/usr/bin/env python3
"""Figures of the v12 manuscript, built only from the outputs under out_v12/.

    python make_figures_v12.py --out ../../reports/v12
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ode_module import S as S_PRIOR
from simulator import S_REAL, S_SEVERE, detection_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_v12')
SENS = ['RGB', 'Multi', 'Hyper', 'LiDAR', 'Thermal', 'Gas']
THETA = [rf'$\theta_{i}$' for i in range(1, 7)]
SC = ['ideal', 'mismatch', 'leakfree', 'severe']
SC_LBL = {'ideal': 'aligned', 'mismatch': 'mismatch', 'leakfree': 'leakfree', 'severe': 'severe'}
# Okabe-Ito, colour-blind safe
COL = {'none': '#000000', 'flat': '#999999', 'agnostic': '#E69F00', 'task': '#0072B2',
       'blind_det': '#D55E00', 'ode': '#009E73'}
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 600,
                     'axes.spines.top': False, 'axes.spines.right': False})


def fig_regimes(dst):
    mats = [(S_PRIOR, r'$\mathbf{S}_{\mathrm{prior}}$ (planner belief)'),
            (detection_matrix('mismatch'), 'mismatch'),
            (S_REAL, 'leakfree'), (S_SEVERE, 'severe')]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9), sharey=True)
    for i, (ax, (m, ttl)) in enumerate(zip(axes, mats)):
        sns.heatmap(m, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax, cbar=False,
                    xticklabels=THETA, yticklabels=SENS, linewidths=0.4, linecolor='white',
                    annot_kws={'fontsize': 7})
        ax.set_title(ttl, fontsize=9)
        if i:
            diff = np.abs(m - S_PRIOR) > 1e-9
            for r, c in zip(*np.nonzero(diff)):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='#D55E00', lw=1.6))
    plt.tight_layout()
    plt.savefig(os.path.join(dst, 'fig_regimes.png'), bbox_inches='tight')
    plt.close()


def fig_bench(dst, tag='J6K6T6'):
    p = os.path.join(OUT, 'generic', f'summary_{tag}.json')
    if not os.path.exists(p):
        print('skip fig_bench: no', p)
        return
    S = json.load(open(p))
    ds = sorted(S, key=float)
    x = np.array([float(d) for d in ds])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, h, ttl in [(axes[0], 'short', '(a) short horizon (first 60 episodes)'),
                       (axes[1], 'long', '(b) long horizon (frozen after 1500)')]:
        ax.axhline(0, color='#555555', lw=0.8)
        for k, lbl in [('flat', r'flat optimism ($Q_0=\beta$)'), ('agnostic', 'task-agnostic prior'),
                       ('task', 'task-aware prior')]:
            m = np.array([S[d][f'diff_{k}_{h}'][0] for d in ds])
            lo = np.array([S[d][f'diff_{k}_{h}'][1] for d in ds])
            hi = np.array([S[d][f'diff_{k}_{h}'][2] for d in ds])
            ax.plot(x, m, marker='o', ms=3.5, lw=1.6, color=COL[k], label=lbl)
            ax.fill_between(x, lo, hi, color=COL[k], alpha=0.18, lw=0)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(r'calibration error $\delta$')
        ax.grid(axis='y', ls=':', alpha=0.6)
    axes[0].set_ylabel('regret difference vs. $Q_0=0$\n(negative = prior helps)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, 1.07))
    plt.tight_layout()
    plt.savefig(os.path.join(dst, 'fig_bench.png'), bbox_inches='tight')
    plt.close()


def fig_heldout(dst):
    agg = json.load(open(os.path.join(OUT, 'aggregate.json')))['heldout_rl_vs_noode']
    cfgs = [('anom', 'task-aware prior', COL['task']),
            ('blind_logdet', 'task-agnostic prior (log det)', COL['agnostic']),
            ('blind_det', 'task-agnostic prior (det)', COL['blind_det'])]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs = np.arange(len(SC))
    w = 0.16
    series = [('none', r'$Q_0=0$ (no prior)', COL['none'])]
    if any('rl_flat_trained' in agg.get(f'anom|{sc}', {}).get('rows', [{}])[0] for sc in SC):
        series.append(('flat', r'flat optimism', COL['flat']))
    series += [(c, l, col) for c, l, col in cfgs]
    for i, (key, lbl, col) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        for j, sc in enumerate(SC):
            cfg = 'anom' if key in ('none', 'flat') else key
            cell = agg.get(f'{cfg}|{sc}')
            if not cell:
                continue
            pol = {'none': 'rl_no_ode_trained', 'flat': 'rl_flat_trained'}.get(key, 'rl_trained')
            vals = [r[pol]['f1'] for r in cell['rows'] if pol in r]
            if not vals:
                continue
            ax.scatter([xs[j] + off] * len(vals), vals, s=10, color=col, alpha=0.55, lw=0)
            ax.plot([xs[j] + off - 0.06, xs[j] + off + 0.06], [np.mean(vals)] * 2, color=col, lw=2.2,
                    label=lbl if j == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([SC_LBL[s] for s in SC])
    ax.set_ylabel(r'held-out F$_1$ macro')
    ax.grid(axis='y', ls=':', alpha=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    plt.tight_layout()
    plt.savefig(os.path.join(dst, 'fig_heldout.png'), bbox_inches='tight')
    plt.close()


def fig_cost(dst, cfg='anom', sc='leakfree'):
    pols = {'ode': ('ODE-only', COL['ode'], 'o'), 'rl': ('RL (task-aware prior)', COL['task'], 's'),
            'rl_no_ode': (r'RL, $Q_0=0$', COL['none'], '^'), 'rand': ('Random', '#777777', 'D'),
            'full': ('Full payload', '#CC79A7', 'P')}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for p, (lbl, col, mk) in pols.items():
        f1s, wts = [], []
        for d in sorted(glob.glob(os.path.join(OUT, 'short', cfg, 's*'))):
            sp = os.path.join(d, sc, f'{p}_summary.json')
            rp = os.path.join(d, sc, f'{p}_records.jsonl')
            if not (os.path.exists(sp) and os.path.exists(rp)):
                continue
            f1s.append(json.load(open(sp))['f1_macro'])
            with open(rp) as fh:
                wts.append(np.mean([json.loads(l)['weight_g'] for l in fh if l.strip()]))
        if not f1s:
            continue
        ax.errorbar(np.mean(wts), np.mean(f1s), yerr=[[np.mean(f1s) - min(f1s)], [max(f1s) - np.mean(f1s)]],
                    fmt=mk, color=col, ms=7, mec='black', mew=0.5, capsize=2, label=lbl)
    ax.set_xlabel('mean sensor payload per dispatch (g)')
    ax.set_ylabel(r'F$_1$ macro')
    ax.grid(ls=':', alpha=0.6)
    ax.legend(fontsize=7, frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(dst, 'fig_cost.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'img_v12'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (fig_regimes, fig_bench, fig_heldout):   # fig_cost is optional, not in the paper
        f(a.out)
        print('done', f.__name__)
