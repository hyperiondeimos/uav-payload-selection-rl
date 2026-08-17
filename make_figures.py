#!/usr/bin/env python3
"""Generate every figure of the paper.

Usage:
    python make_figures.py                 # writes to ./img
    python make_figures.py --out path/to/dir
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from metrics import _per_anomaly, f1_macro
from ode_module import ANOMALIES, S as S_PRIOR
from rl_agents import Policy
from simulator import S_REAL, run_seed

SENSORS_LBL = ['RGB', 'Multi', 'Hyper', 'LiDAR', 'Thermal', 'Gas']
THETA_LBL = [rf'$\theta_{i}$' for i in range(1, 7)]
ANOM_LBL = [rf'$A_{i}$' for i in range(1, 7)]

# Elsevier artwork spec: bitmapped art at >= 500 dpi.
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200, 'savefig.dpi': 600})

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')


def collect(policy_name, seeds, scenario, transfer=False, freeze_after=None):
    """Ordered phase records; optionally train on the first `freeze_after`
    seeds and then freeze the agent."""
    recs = []
    pol = Policy(policy_name) if (transfer or freeze_after is not None) else policy_name
    for i, seed in enumerate(seeds):
        if freeze_after is not None and i == freeze_after and pol.agent:
            pol.agent.epsilon = 0.0
            pol.agent.eps_min = 0.0
            pol.agent.eta = 0.0
        r, _ = run_seed(seed, pol if not isinstance(pol, str) else policy_name, scenario)
        for x in r:
            x['seed'] = seed
        recs.append((i, r))
    return recs


def fig_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, m, ttl in [(axes[0], S_PRIOR, r'$\mathbf{S}_{\mathrm{prior}}$ (ODE belief)'),
                       (axes[1], S_REAL, r'$\mathbf{S}_{\mathrm{real}}$ (leakfree reality)')]:
        sns.heatmap(m, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1,
                    xticklabels=THETA_LBL, yticklabels=SENSORS_LBL, ax=ax,
                    cbar=False, linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 8})
        ax.set_title(ttl, fontsize=10)
    changed = [(0, 4), (1, 0), (2, 0)] + [(4, k) for k in range(6)]
    for (r, c) in changed:
        axes[1].add_patch(plt.Rectangle((c, r), 1, 1, fill=False,
                                        edgecolor='red', lw=2))
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_regimes.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_regimes.png')


def fig_learning():
    seeds = list(range(7000, 7060))
    recs = collect('rl', seeds, 'leakfree', transfer=True)
    flat = [x for _, r in recs for x in r]
    ep = [x['episode'] for x in flat]
    rew = np.array([x['reward'] for x in flat])
    eps = [x['epsilon'] for x in flat]
    qs = [x['q_size'] for x in flat]
    w = 20
    ma = np.convolve(rew, np.ones(w) / w, mode='valid')

    fig, ax1 = plt.subplots(figsize=(7.4, 3.2))
    ax1.plot(range(len(ma)), ma, color='#1565C0', lw=1.8, label=f'reward MA({w})')
    ax1.axhline(0, ls='--', color='gray', lw=0.8, label='_nolegend_')
    ax1.set_xlabel('training episode')
    ax1.set_ylabel(f'reward MA({w})', color='#1565C0')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax2 = ax1.twinx()
    ax2.plot(ep, eps, color='#C62828', lw=1.2, ls='--', label=r'$\varepsilon$')
    ax2.plot(ep, np.array(qs) / max(qs), color='#2E7D32', lw=1.2, ls=':',
             label=r'$|Q|$ (norm.)')
    ax2.set_ylabel(r'$\varepsilon$  /  $|Q|$ (normalized)')
    ax2.set_ylim(0, 1.05)
    lines = [ln for ln in ax1.get_lines() + ax2.get_lines()
             if not ln.get_label().startswith('_')]
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_learning.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_learning.png')


def fig_peranom():
    seeds = list(range(7000, 7040))
    pols = {'ode': 'ODE-only', 'rl': 'RL (ODE-init)', 'rl_no_ode': r'RL$\backslash$ODE'}
    data = {}
    for p in pols:
        flat = [x for _, r in collect(p, seeds, 'leakfree') for x in r]
        pa = _per_anomaly(flat)
        data[p] = [pa.get(a, {'f1': 0})['f1'] for a in range(1, 7)]
    x = np.arange(6)
    wd = 0.26
    colors = ['#2E7D32', '#EF6C00', '#C62828']
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for i, (p, lbl) in enumerate(pols.items()):
        ax.bar(x + (i - 1) * wd, data[p], wd, label=lbl, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(ANOM_LBL)
    ax.set_ylabel(r'$F_1$')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis='y', ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_peranom.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_peranom.png')


ANOMALY_THETA = {a: ANOMALIES[a]['theta'] for a in ANOMALIES}


def specificity(sensors, aid, s_mat):
    return min(max(s_mat[s - 1, k - 1] for s in sensors) for k in ANOMALY_THETA[aid])


def predict(sensors, s_mat):
    """Identify a cluster by the anomaly whose required characteristics the
    embarked payload covers best in the worst case."""
    return max(range(1, 7), key=lambda a: specificity(sensors, a, s_mat))


def confmat(records, s_mat):
    by_cluster = defaultdict(list)
    for x in records:
        by_cluster[(x['seed'], x['cluster_id'])].append(x)
    cm = np.zeros((6, 6), int)
    for rs in by_cluster.values():
        rs = sorted(rs, key=lambda r: (r['mission'], r['phase']))
        a = rs[0]['anomaly_id']
        hit = next((r for r in rs if r['result'] == 'true_positive'), None)
        if hit is not None:
            cm[a - 1, a - 1] += 1
        else:
            last = rs[-1]
            if last['coverage_obs'] >= 0.4:
                cm[a - 1, predict(last['sensors'], s_mat) - 1] += 1
    return cm


def fig_confusion():
    train = list(range(7000, 7150))
    ev = list(range(7150, 7210))
    abl = collect('rl_no_ode', train + ev, 'leakfree', freeze_after=len(train))
    abl_ev = [x for i, r in abl if i >= len(train) for x in r]
    ode_ev = [x for _, r in collect('ode', ev, 'leakfree') for x in r]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for ax, recs, ttl in [(axes[0], ode_ev, 'ODE-only (wrong prior)'),
                          (axes[1], abl_ev, r'RL$\backslash$ODE (learned)')]:
        cm = confmat(recs, S_REAL)
        rown = cm.sum(1, keepdims=True)
        norm = np.divide(cm, np.where(rown == 0, 1, rown))
        sns.heatmap(norm, annot=cm, fmt='d', cmap='Blues', vmin=0, vmax=1,
                    xticklabels=ANOM_LBL, yticklabels=ANOM_LBL, ax=ax,
                    cbar=False, linewidths=0.5, linecolor='white')
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('identified')
        ax.set_ylabel('actual')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_confusion.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_confusion.png')


def fig_cost():
    seeds = list(range(7000, 7040))
    pols = {'ode': ('ODE-only', '#2E7D32', 'o'),
            'rl': ('RL (ODE-init)', '#EF6C00', 's'),
            'rl_no_ode': (r'RL$\backslash$ODE', '#C62828', '^'),
            'rand': ('Random', '#777777', 'D'),
            'full': ('Full-payload', '#6A1B9A', 'P')}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for p, (lbl, col, mk) in pols.items():
        flat = [x for _, r in collect(p, seeds, 'leakfree') for x in r]
        f1 = f1_macro(flat)
        wt = np.mean([x['weight_g'] for x in flat])
        ax.scatter(wt, f1, s=95, c=col, marker=mk, label=lbl,
                   edgecolors='black', linewidths=0.6, zorder=3)
        ax.annotate(lbl, (wt, f1), textcoords='offset points', xytext=(7, 4),
                    fontsize=8)
    ax.set_xlabel('mean sensor payload carried per mission (g)')
    ax.set_ylabel(r'$F_1$ macro')
    ax.grid(ls=':', alpha=0.6)
    ax.set_title('Cost vs. accuracy (leakfree, short horizon)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_cost.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_cost.png')


def fig_battery():
    import energy_shepard as es
    from ode_module import W_UTIL_G

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    ws = np.linspace(0, 1200, 60)
    sh = [es.autonomy_minutes(es.DRONE['base_mass'] + w / 1000.0) for w in ws]
    lin = [max(0, 25.0 * (1 - w / 1000.0)) for w in ws]
    ax1.plot(ws, sh, color='#2E7D32', lw=2.2, label='Shepard energy model')
    ax1.plot(ws, lin, color='#C62828', lw=2.0, ls='--', label='linear weight proxy')
    ax1.axvline(W_UTIL_G, color='gray', ls=':', lw=1.2)
    ax1.text(W_UTIL_G - 20, 22, 'useful\nlimit', ha='right', fontsize=8, color='gray')
    ax1.set_xlabel('sensor payload (g)')
    ax1.set_ylabel('autonomy (min)')
    ax1.set_ylim(0, 26)
    ax1.grid(ls=':', alpha=0.6)
    ax1.legend(fontsize=8)
    ax1.set_title('(a) autonomy vs payload', fontsize=10)

    mass = es.DRONE['base_mass'] + 0.470
    power = es.hover_power_W(mass)
    it, ts, vs = 0.0, [], []
    q_us = es.BATT['Q'] * (1 - es.BATT['Q_reserve'])
    t = 0.0
    while it < q_us:
        i = power / max(es.shepard_voltage(it, 0.0), 1.0)
        i = power / max(es.shepard_voltage(it, i), 1.0)
        v = es.shepard_voltage(it, i)
        if v <= es.BATT['V_cut']:
            break
        ts.append(t / 60.0)
        vs.append(v)
        it += i / 3600.0
        t += 1.0
    ax2.plot(ts, vs, color='#1565C0', lw=2.2)
    ax2.axhline(es.BATT['V_cut'], color='#C62828', ls='--', lw=1.2)
    ax2.text(0.4, es.BATT['V_cut'] + 0.08, 'cutoff', fontsize=8, color='#C62828')
    ax2.set_xlabel('flight time (min)')
    ax2.set_ylabel('pack voltage (V)')
    ax2.grid(ls=':', alpha=0.6)
    ax2.set_title('(b) Shepard discharge (470 g payload)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_battery.png'), bbox_inches='tight')
    plt.close()
    print('wrote fig_battery.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=IMG, help='output directory for the figures')
    args = ap.parse_args()
    IMG = args.out
    os.makedirs(IMG, exist_ok=True)

    fig_regimes()
    fig_learning()
    fig_peranom()
    fig_confusion()
    fig_cost()
    fig_battery()
    print('all figures written to', IMG)
