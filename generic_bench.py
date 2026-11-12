#!/usr/bin/env python3
"""Random-instance benchmark: task-conditioned subset selection seeded by an
optimal-design prior.

It abstracts the UAV payload problem (J sensors, K characteristics, T tasks,
a weight budget, max-coverage detection) to cross the two ways an analytical
prior can be wrong:

  criterion   'agnostic' = log det(eps I + F_C) over all K characteristics
              (one ranking for every task), 'task' = log det over Theta_t
  control     'flat' = Q0 = beta for every action (optimism, no information),
              'none' = Q0 = 0 with uniform exploration
  calibration S_real = clip(S_prior + delta * N(0, 1), 0, 1)

Each random instance is solved by three tabular learners that differ only in
the prior (Q0 = beta * u_t and softmax(u_t)-biased exploration, as in
rl_agents.py), plus the prior-only argmax. Expected rewards are analytic, so
regret is exact. Learners share the random stream of each instance.

Usage:
    python generic_bench.py --instances 300 --deltas 0 0.1 0.2 0.3 0.4 --out out_v12/generic
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import erf, sqrt

import numpy as np

THR_TP, THR_PD, THR_FP = 0.70, 0.40, 0.20
R_TP, R_PD, R_FP, R_FN = 1.0, 0.5, -0.5, -1.0
NOISE = 0.10
W_DET, W_EFF, ALPHA_S = 0.5, 0.3, 0.35
ETA, EPS0, EPS_MIN, EPS_DECAY, BETA, MIX = 0.20, 1.0, 0.05, 0.97, 0.30, 0.5
EPS_PRIOR = 1e-2
PRIORS = ('none', 'flat', 'agnostic', 'task')


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def make_instance(rng, J=6, K=6, T=6):
    S = np.zeros((J, K))
    for j in range(J):
        strong = rng.choice(K, size=int(rng.integers(1, 4)), replace=False)
        S[j, strong] = rng.uniform(0.6, 0.95, size=len(strong))
        weak = rng.random(K) < 0.4
        weak[strong] = False
        S[j, weak] = rng.uniform(0.0, 0.3, size=int(weak.sum()))
    for k in range(K):                       # every characteristic observable
        if S[:, k].max() < 0.6:
            S[int(rng.integers(J)), k] = rng.uniform(0.6, 0.95)
    sigma = rng.uniform(0.08, 0.20, size=J)
    w = rng.uniform(80, 480, size=J)
    budget = 0.5 * w.sum()
    tasks = [tuple(sorted(rng.choice(K, size=int(rng.integers(1, 4)), replace=False)))
             for _ in range(T)]
    actions = [c for r in range(1, J + 1) for c in combinations(range(J), r)
               if w[list(c)].sum() <= budget]
    return {'S': S, 'sigma': sigma, 'tasks': tasks, 'actions': actions, 'J': J, 'K': K}


def coverage_table(S, tasks, actions):
    cov = np.zeros((len(tasks), len(actions)))
    for a, c in enumerate(actions):
        best = S[list(c)].max(axis=0)
        for t, th in enumerate(tasks):
            cov[t, a] = best[list(th)].mean()
    return cov


def expected_reward(cov, actions, J):
    p_fn = _phi((THR_FP - cov) / NOISE)
    p_fp = _phi((THR_PD - cov) / NOISE) - p_fn
    p_pd = _phi((THR_TP - cov) / NOISE) - p_fn - p_fp
    p_tp = 1.0 - p_fn - p_fp - p_pd
    r_det = R_TP * p_tp + R_PD * p_pd + R_FP * p_fp + R_FN * p_fn
    size = np.array([len(c) for c in actions])
    return W_DET * r_det + W_EFF * (1.0 - ALPHA_S * size / J), p_tp


def prior_utility(inst, kind):
    S, sigma, tasks, actions, K = (inst[k] for k in ('S', 'sigma', 'tasks', 'actions', 'K'))
    info = (S ** 2) / (sigma[:, None] ** 2)            # J x K diagonal FIM entries
    T, A = len(tasks), len(actions)
    if kind == 'none':
        return np.zeros((T, A))
    if kind == 'flat':                                 # optimism without information
        return np.ones((T, A))
    u = np.zeros((T, A))
    for a, c in enumerate(actions):
        d = info[list(c)].sum(axis=0)                  # diagonal of F_C
        if kind == 'agnostic':
            u[:, a] = np.log(EPS_PRIOR + d).sum()
        else:
            for t, th in enumerate(tasks):
                u[t, a] = np.log(EPS_PRIOR + d[list(th)]).sum()
    lo, hi = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    return np.where(hi - lo > 1e-12, (u - lo) / np.where(hi - lo > 1e-12, hi - lo, 1), 1.0)


def run_learner(u, use_prior, bias_explore, cov_real, er, best, size, J, n_train, n_short, seed):
    T, A = u.shape
    rng = np.random.default_rng(seed)
    Q = BETA * u if use_prior else np.zeros_like(u)
    Q = Q.copy()
    eps = EPS0
    online = []
    for ep in range(n_train):
        t = int(rng.integers(T))
        if rng.random() < eps:
            if bias_explore:
                z = np.exp(u[t] - u[t].max())
                p = (1 - MIX) / A + MIX * z / z.sum()
                a = int(rng.choice(A, p=p / p.sum()))
            else:
                a = int(rng.integers(A))
        else:
            a = int(np.argmax(Q[t]))
        obs = cov_real[t, a] + rng.normal(0.0, NOISE)
        r_det = R_TP if obs >= THR_TP else R_PD if obs >= THR_PD else R_FP if obs >= THR_FP else R_FN
        r = W_DET * r_det + W_EFF * (1.0 - ALPHA_S * size[a] / J)
        Q[t, a] += ETA * (r - Q[t, a])
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if ep < n_short:
            online.append(best[t] - er[t, a])
    greedy = Q.argmax(axis=1)
    return float(np.mean(online)), greedy



def solve_instance(args):
    idx, delta, J, K, T, n_train, n_short, seed0 = args
    rng = np.random.default_rng(seed0 + idx)
    inst = make_instance(rng, J, K, T)
    actions = inst['actions']
    A = len(actions)
    size = np.array([len(c) for c in actions])
    S_real = np.clip(inst['S'] + delta * rng.normal(size=inst['S'].shape), 0.0, 1.0)
    cov_real = coverage_table(S_real, inst['tasks'], actions)
    er, ptp = expected_reward(cov_real, actions, J)
    best = er.max(axis=1)
    out = {'idx': idx, 'delta': delta, 'n_actions': A, 'oracle_reward': float(best.mean())}
    learn_seed = seed0 * 7 + idx
    for kind in PRIORS:
        u = prior_utility(inst, kind)
        on, greedy = run_learner(u, kind != 'none', kind in ('agnostic', 'task'), cov_real, er, best, size, J,
                                 n_train, n_short, learn_seed)
        tt = np.arange(len(inst['tasks']))
        out[f'{kind}_short_regret'] = on
        out[f'{kind}_long_regret'] = float((best - er[tt, greedy]).mean())
        out[f'{kind}_long_ptp'] = float(ptp[tt, greedy].mean())
        if kind in ('agnostic', 'task'):
            arg = u.argmax(axis=1)
            out[f'{kind}_prioronly_regret'] = float((best - er[tt, arg]).mean())
            out[f'{kind}_prioronly_ptp'] = float(ptp[tt, arg].mean())
    return out


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, default=300)
    ap.add_argument('--deltas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument('--J', type=int, default=6)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--T', type=int, default=6)
    ap.add_argument('--train', type=int, default=1500)
    ap.add_argument('--short', type=int, default=60)
    ap.add_argument('--seed0', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'out_v12', 'generic'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, d, a.J, a.K, a.T, a.train, a.short, a.seed0)
            for d in a.deltas for i in range(a.instances)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(solve_instance, jobs, chunksize=8))
    tag = f'J{a.J}K{a.K}T{a.T}'
    with open(os.path.join(a.out, f'instances_{tag}.json'), 'w') as f:
        json.dump(rows, f)

    summary = {}
    print(f'{tag}: {a.instances} instances per delta, train={a.train}, short={a.short}')
    print('regret (lower is better); diff = prior - none, 95% bootstrap CI over instances')
    for d in a.deltas:
        R = [r for r in rows if r['delta'] == d]
        s = {}
        for kind in PRIORS:
            for h in ('short', 'long'):
                s[f'{kind}_{h}'] = boot_ci([r[f'{kind}_{h}_regret'] for r in R])
            s[f'{kind}_long_ptp'] = boot_ci([r[f'{kind}_long_ptp'] for r in R])
        for kind in ('flat', 'agnostic', 'task'):
            for h in ('short', 'long'):
                s[f'diff_{kind}_{h}'] = boot_ci([r[f'{kind}_{h}_regret'] - r[f'none_{h}_regret'] for r in R])
            if kind == 'flat':
                continue
            s[f'{kind}_prioronly'] = boot_ci([r[f'{kind}_prioronly_regret'] for r in R])
        summary[str(d)] = s
        f = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]'
        print(f'delta={d:.2f}  long regret none={s["none_long"][0]:.3f} agn={s["agnostic_long"][0]:.3f} '
              f'task={s["task_long"][0]:.3f} flat={s["flat_long"][0]:.3f} | prior-only agn={s["agnostic_prioronly"][0]:.3f} '
              f'task={s["task_prioronly"][0]:.3f}')
        for h in ('short', 'long'):
            print(f'   {h:<5} diff flat {f(s[f"diff_flat_{h}"])}  agn {f(s[f"diff_agnostic_{h}"])}  task {f(s[f"diff_task_{h}"])}')
    with open(os.path.join(a.out, f'summary_{tag}.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)


if __name__ == '__main__':
    main()
