#!/usr/bin/env python3
"""Held-out protocol: train one agent across many worlds with weight transfer,
freeze it (greedy, no further updates), then evaluate on disjoint worlds.

Usage:
    python run_leakfree_eval.py --train 200 --eval 60 --scenario leakfree
    python run_leakfree_eval.py --train 200 --eval 60 --scenario mismatch
"""
import argparse
import json
import os

from metrics import summarize
from rl_agents import Policy
from run_campaign import cluster_at_k
from simulator import run_seed


def eval_records(policy_or_name, eval_seeds, scenario='leakfree', max_attempts=3):
    recs, agg = [], {'n_missions': 0, 'n_resolved': 0, 'phase1': 0,
                     'n_clusters': 0, 'n_clusters_resolved': 0}
    for seed in eval_seeds:
        r, m = run_seed(seed, policy_or_name, scenario, max_attempts=max_attempts)
        for x in r:
            x['seed'] = seed
        recs.extend(r)
        agg['n_missions'] += m['n_missions']
        agg['n_resolved'] += m['n_resolved']
        agg['phase1'] += int(round(m['phase1_resolution_rate'] * m['n_missions']))
        agg['n_clusters'] += m['n_clusters']
        agg['n_clusters_resolved'] += m['n_clusters_resolved']
    mr = {
        'n_missions': agg['n_missions'],
        'n_resolved': agg['n_resolved'],
        'resolution_rate': round(agg['n_resolved'] / agg['n_missions'], 4) if agg['n_missions'] else 0.0,
        'phase1_resolution_rate': round(agg['phase1'] / agg['n_missions'], 4) if agg['n_missions'] else 0.0,
        'n_clusters': agg['n_clusters'],
        'n_clusters_resolved': agg['n_clusters_resolved'],
        'cluster_resolution_rate': round(agg['n_clusters_resolved'] / agg['n_clusters'], 4) if agg['n_clusters'] else 0.0,
    }
    s = summarize(recs, mr)
    s['cluster_resolution_at_k'] = cluster_at_k(recs)
    s['clu_at_1'] = s['cluster_resolution_at_k'][1]['rate']
    return s


def train_policy(name, train_seeds, scenario='leakfree', max_attempts=3):
    """Reuse one policy object across all training seeds (weight transfer)."""
    pol = Policy(name)
    for seed in train_seeds:
        run_seed(seed, pol, scenario, max_attempts=max_attempts)
    return pol


def freeze(pol):
    if pol.agent:
        pol.agent.epsilon = 0.0
        pol.agent.eps_min = 0.0
        pol.agent.eta = 0.0
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=200)
    ap.add_argument('--eval', type=int, default=60)
    ap.add_argument('--seed0', type=int, default=7000)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--scenario', default='leakfree',
                    choices=['ideal', 'mismatch', 'leakfree'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), 'out',
                                f'heldout_{args.scenario}')
    os.makedirs(args.out, exist_ok=True)

    train_seeds = list(range(args.seed0, args.seed0 + args.train))
    eval_seeds = list(range(args.seed0 + args.train,
                            args.seed0 + args.train + args.eval))
    print(f'Scenario={args.scenario}.  train={len(train_seeds)} seeds -> freeze '
          f'-> eval={len(eval_seeds)} held-out seeds\n')

    results = {}
    for name in ('rl', 'rl_decay', 'rl_initonly', 'rl_no_ode'):
        pol = freeze(train_policy(name, train_seeds, scenario=args.scenario,
                                  max_attempts=args.max_attempts))
        results[f'{name}_trained'] = eval_records(
            pol, eval_seeds, scenario=args.scenario, max_attempts=args.max_attempts)
        print(f'  {name}_trained (|Q|={pol.q_size()} entries)')

    # Untrained ODE-biased agent: greedy from the optimistic prior, no learning.
    results['rl_untrained'] = eval_records(
        freeze(Policy('rl')), eval_seeds, scenario=args.scenario,
        max_attempts=args.max_attempts)
    for name in ('ode', 'rand'):
        results[name] = eval_records(name, eval_seeds, scenario=args.scenario,
                                     max_attempts=args.max_attempts)

    for k, s in results.items():
        with open(os.path.join(args.out, f'{k}_summary.json'), 'w') as f:
            json.dump(s, f, indent=2)

    order = ['rl_trained', 'rl_decay_trained', 'rl_initonly_trained',
             'rl_no_ode_trained', 'rl_untrained', 'ode', 'rand']
    lab = {'rl_trained': 'RL trained (fixed beta)',
           'rl_decay_trained': 'RL trained (decaying beta)',
           'rl_initonly_trained': 'RL trained (init-only, unif expl)',
           'rl_no_ode_trained': 'RL\\ODE trained',
           'rl_untrained': 'RL untrained (= ODE prior)',
           'ode': 'ODE-only', 'rand': 'Random'}

    print('\n' + '=' * 74)
    print(f'HELD-OUT EVALUATION ({args.scenario})')
    print('=' * 74)
    print(f'{"policy":<34}{"F1":>7}{"MCC":>7}{"kappa":>7}{"miss":>7}{"clu@1":>7}')
    for k in order:
        s = results[k]
        print(f'{lab[k]:<34}{s["f1_macro"]:>7.3f}{s["mcc"]:>7.3f}'
              f'{s["cohen_kappa"]:>7.3f}'
              f'{s["mission_resolution"]["resolution_rate"]:>7.2f}{s["clu_at_1"]:>7.2f}')

    print('\n' + '-' * 74)
    print('F1 macro with 95% bootstrap CI:')
    for k in order:
        s = results[k]
        ci = s['ci_95']['f1_macro']
        print(f'  {lab[k]:<34} {s["f1_macro"]:.3f}  [{ci["low"]}, {ci["high"]}]')

    def contrast(a_key, b_key):
        a, b = results[a_key], results[b_key]
        ca, cb = a['ci_95']['f1_macro'], b['ci_95']['f1_macro']
        if ca['low'] > cb['high']:
            rel = 'above (CIs disjoint)'
        elif cb['low'] > ca['high']:
            rel = 'below (CIs disjoint)'
        else:
            rel = 'ties (CIs overlap)'
        return (f'  {lab[a_key]} {rel} {lab[b_key]}  '
                f'({a["f1_macro"]:.3f} vs {b["f1_macro"]:.3f})')

    print('\nAgainst the unbiased ablation:')
    for k in ('rl_initonly_trained', 'rl_decay_trained', 'rl_trained'):
        print(contrast(k, 'rl_no_ode_trained'))
    print(contrast('rl_initonly_trained', 'ode'))
    print(f'\nWrote summaries under: {args.out}')


if __name__ == '__main__':
    main()
