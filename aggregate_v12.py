#!/usr/bin/env python3
"""Aggregate the v12 runs under out_v12/ into the tables used by the paper.

Held-out: for every prior configuration x regime x seed block, read the
summary JSONs written by run_leakfree_eval.py, then report the per-block F1
[95% CI] and the across-block mean, plus how many blocks give disjoint CIs
for the contrast of interest (prior-seeded learner vs Q0 = 0 learner).
Short horizon: same, from run_campaign.py summaries.

Usage:
    python aggregate_v12.py            # prints tables, writes out_v12/aggregate.json
"""
import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_v12')
SC = ['ideal', 'mismatch', 'leakfree', 'severe']


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def cell(s):
    ci = s['ci_95']['f1_macro']
    return {'f1': s['f1_macro'], 'lo': ci['low'], 'hi': ci['high'], 'mcc': s['mcc'],
            'miss': s['mission_resolution']['resolution_rate'], 'clu1': s['clu_at_1'],
            'reward': s.get('reward_mean')}


def heldout():
    out = defaultdict(dict)          # out[cfg][(sc, seed)][policy] = cell
    for d in glob.glob(os.path.join(ROOT, 'heldout', '*', '*_s*')):
        cfg = os.path.basename(os.path.dirname(d))
        m = re.match(r'(\w+)_s(\d+)$', os.path.basename(d))
        sc, seed = m.group(1), int(m.group(2))
        pols = {}
        for f in glob.glob(os.path.join(d, '*_summary.json')):
            pols[os.path.basename(f)[:-len('_summary.json')]] = cell(load(f))
        if pols:
            out[cfg][(sc, seed)] = pols
    return out


def short():
    out = defaultdict(dict)          # out[cfg][(sc, seed)][policy] = cell
    for d in glob.glob(os.path.join(ROOT, 'short', '*', 's*')):
        cfg = os.path.basename(os.path.dirname(d))
        seed = int(os.path.basename(d)[1:])
        for sc in SC:
            pols = {}
            for f in glob.glob(os.path.join(d, sc, '*_summary.json')):
                pols[os.path.basename(f)[:-len('_summary.json')]] = cell(load(f))
            if pols:
                out[cfg][(sc, seed)] = pols
    return out


def rel(a, b):
    if a['lo'] > b['hi']:
        return '+'
    if b['lo'] > a['hi']:
        return '-'
    return '='


def table(data, title, a_pol, b_pol, extra=()):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)
    agg = {}
    for cfg in sorted(data):
        for sc in SC:
            blocks = sorted(s for (x, s) in data[cfg] if x == sc)
            if not blocks:
                continue
            rows = [data[cfg][(sc, s)] for s in blocks]
            if not all(a_pol in r and b_pol in r for r in rows):
                continue
            ma = sum(r[a_pol]['f1'] for r in rows) / len(rows)
            mb = sum(r[b_pol]['f1'] for r in rows) / len(rows)
            sig = ''.join(rel(r[a_pol], r[b_pol]) for r in rows)
            ex = {p: sum(r[p]['f1'] for r in rows) / len(rows) for p in extra if all(p in r for r in rows)}
            per = '  '.join(f"s{s}:{r[a_pol]['f1']:.3f}[{r[a_pol]['lo']:.3f},{r[a_pol]['hi']:.3f}]"
                            f"/{r[b_pol]['f1']:.3f}[{r[b_pol]['lo']:.3f},{r[b_pol]['hi']:.3f}]"
                            for s, r in zip(blocks, rows))
            exs = ' '.join(f'{p}={v:.3f}' for p, v in ex.items())
            print(f'{cfg:<14}{sc:<9} {a_pol}={ma:.3f} {b_pol}={mb:.3f} d={ma - mb:+.3f} sig={sig:<4} {exs}')
            print(f'{"":<24}{per}')
            agg[f'{cfg}|{sc}'] = {'blocks': blocks, 'mean_a': ma, 'mean_b': mb, 'sig': sig,
                                  'extra': ex, 'rows': rows}
    return agg


if __name__ == '__main__':
    H, S = heldout(), short()
    res = {}
    res['heldout_rl_vs_noode'] = table(H, 'HELD-OUT  rl_trained vs rl_no_ode_trained (F1, mean over seed blocks)',
                                       'rl_trained', 'rl_no_ode_trained',
                                       extra=('rl_initonly_trained', 'rl_decay_trained', 'rl_flat_trained',
                                              'rl_untrained', 'ode', 'rand'))
    res['heldout_rl_vs_flat'] = table(H, 'HELD-OUT  rl_trained vs rl_flat_trained', 'rl_trained', 'rl_flat_trained')
    res['heldout_init_vs_rl'] = table(H, 'HELD-OUT  rl_trained vs rl_initonly_trained', 'rl_trained',
                                      'rl_initonly_trained')
    res['short_rl_vs_noode'] = table(S, 'SHORT  rl vs rl_no_ode (F1, mean over seed blocks)', 'rl', 'rl_no_ode',
                                     extra=('rl_initonly', 'rl_flat', 'ode', 'rand', 'full'))
    res['short_rl_vs_flat'] = table(S, 'SHORT  rl vs rl_flat', 'rl', 'rl_flat')
    with open(os.path.join(ROOT, 'aggregate.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
