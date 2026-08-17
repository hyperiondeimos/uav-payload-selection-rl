#!/usr/bin/env python3
"""Short-horizon campaign: N seeds x policies x scenarios, one fresh agent per
world unless --transfer is given. Produces the short-horizon factorial table.

Usage:
    python run_campaign.py --seeds 40 --scenarios ideal mismatch leakfree
    python run_campaign.py --scenarios leakfree --energy shepard
"""
import argparse
import json
import os
from collections import defaultdict

from metrics import summarize
from simulator import run_seed

POLICIES = ['rl', 'rl_decay', 'ode', 'rand', 'rl_no_ode']
SCENARIOS = ['ideal', 'mismatch']
LABELS = {'rl': 'RL (fixed beta)', 'rl_decay': 'RL (decaying beta)',
          'ode': 'ODE-only (deterministic)', 'rand': 'Random',
          'rl_no_ode': 'RL\\ODE (ablation)', 'full': 'Full-payload'}


def cluster_at_k(records, kmax=3):
    """Fraction of clusters resolved within their first k missions."""
    by_cluster = defaultdict(list)
    for r in records:
        by_cluster[(r['seed'], r['cluster_id'])].append(r)
    resolve_attempt = []
    for rs in by_cluster.values():
        missions = sorted({r['mission'] for r in rs})
        att = None
        for i, m in enumerate(missions, start=1):
            if any(r['mission'] == m and r['result'] == 'true_positive' for r in rs):
                att = i
                break
        resolve_attempt.append(att)
    n_total = len(resolve_attempt)
    out = {}
    for k in range(1, kmax + 1):
        nres = sum(1 for a in resolve_attempt if a is not None and a <= k)
        out[k] = {'n_resolved': nres, 'n_total': n_total,
                  'rate': round(nres / n_total, 4) if n_total else 0.0}
    return out


def run_policy(policy, scenario, seeds, max_attempts, transfer=False):
    from rl_agents import Policy

    all_records = []
    agg = {'n_missions': 0, 'n_resolved': 0, 'phase1_resolved': 0,
           'n_clusters': 0, 'n_clusters_resolved': 0}
    # With transfer a single policy object is reused across seeds, so epsilon
    # decays campaign-wide and the Q-table accumulates.
    shared = Policy(policy) if (transfer and policy in (
        'rl', 'rl_decay', 'rl_initonly', 'rl_no_ode')) else None
    for seed in seeds:
        recs, msum = run_seed(seed, shared if shared is not None else policy,
                              scenario, max_attempts=max_attempts)
        all_records.extend(recs)
        agg['n_missions'] += msum['n_missions']
        agg['n_resolved'] += msum['n_resolved']
        agg['phase1_resolved'] += int(round(msum['phase1_resolution_rate']
                                            * msum['n_missions']))
        agg['n_clusters'] += msum['n_clusters']
        agg['n_clusters_resolved'] += msum['n_clusters_resolved']
    mission_resolution = {
        'n_missions': agg['n_missions'],
        'n_resolved': agg['n_resolved'],
        'resolution_rate': round(agg['n_resolved'] / agg['n_missions'], 4)
                           if agg['n_missions'] else 0.0,
        'phase1_resolution_rate': round(agg['phase1_resolved'] / agg['n_missions'], 4)
                                  if agg['n_missions'] else 0.0,
        'n_clusters': agg['n_clusters'],
        'n_clusters_resolved': agg['n_clusters_resolved'],
        'cluster_resolution_rate': round(agg['n_clusters_resolved'] / agg['n_clusters'], 4)
                                   if agg['n_clusters'] else 0.0,
    }
    summ = summarize(all_records, mission_resolution)
    summ['cluster_resolution_at_k'] = cluster_at_k(all_records)
    summ['clu_at_1'] = summ['cluster_resolution_at_k'][1]['rate']
    summ['n_seeds'] = len(seeds)
    return all_records, summ


def fmt(x, w=7):
    if isinstance(x, float):
        return f'{x:>{w}.3f}'
    return f'{str(x):>{w}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--seed0', type=int, default=7000)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    ap.add_argument('--policies', nargs='+', default=POLICIES)
    ap.add_argument('--transfer', action='store_true',
                    help='reuse one policy object across seeds (weight transfer)')
    ap.add_argument('--energy', choices=['linear', 'shepard'], default='linear',
                    help='autonomy model: linear weight proxy or Shepard battery')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'out'))
    args = ap.parse_args()

    import ode_module
    ode_module.USE_SHEPARD = (args.energy == 'shepard')
    print(f'[info] autonomy model: {args.energy}')

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    results = {}
    print(f'Running {len(seeds)} seeds x {len(args.policies)} policies x '
          f'{len(args.scenarios)} scenarios (max_attempts={args.max_attempts})\n')

    for scenario in args.scenarios:
        outdir = os.path.join(args.out, scenario)
        os.makedirs(outdir, exist_ok=True)
        for policy in args.policies:
            recs, summ = run_policy(policy, scenario, seeds, args.max_attempts,
                                    transfer=args.transfer)
            with open(os.path.join(outdir, f'{policy}_records.jsonl'), 'w') as f:
                for r in recs:
                    f.write(json.dumps(r) + '\n')
            with open(os.path.join(outdir, f'{policy}_summary.json'), 'w') as f:
                json.dump(summ, f, indent=2)
            results[(scenario, policy)] = summ
            print(f'  [{scenario:>8}] {policy:<10} '
                  f'F1={summ["f1_macro"]:.3f}  MCC={summ["mcc"]:.3f}  '
                  f'kappa={summ["cohen_kappa"]:.3f}  '
                  f'miss={summ["mission_resolution"]["resolution_rate"]:.2f}  '
                  f'clu@1={summ["clu_at_1"]:.2f}  n={summ["n_records"]}')

    lines = []

    def emit(s=''):
        lines.append(s)
        print(s)

    emit('\n' + '=' * 78)
    emit('COMPARISON BY SCENARIO')
    emit('=' * 78)
    metric_keys = [('f1_macro', 'F1 macro'), ('mcc', 'MCC'),
                   ('cohen_kappa', 'Cohen kappa'), ('balanced_accuracy', 'Bal. acc.'),
                   ('brier_score', 'Brier (lower)'), ('clu_at_1', 'Cluster@1')]
    for scenario in args.scenarios:
        emit(f'\n--- {scenario.upper()} ---')
        emit(f'{"metric":<20}' + ''.join(fmt(p, 11) for p in args.policies))
        for key, name in metric_keys:
            row = f'{name:<20}'
            for p in args.policies:
                row += fmt(results[(scenario, p)][key], 11)
            emit(row)
        emit(f'{"mission res.":<20}' + ''.join(
            fmt(results[(scenario, p)]['mission_resolution']['resolution_rate'], 11)
            for p in args.policies))
        emit(f'{"F1 95% CI":<20}' + ''.join(
            f'{str(results[(scenario, p)]["ci_95"]["f1_macro"]["low"]) + "-" + str(results[(scenario, p)]["ci_95"]["f1_macro"]["high"]):>11}'
            for p in args.policies))

    if {'ideal', 'mismatch'}.issubset(set(args.scenarios)):
        emit('\n' + '=' * 78)
        emit('RELATIVE DEGRADATION ideal -> mismatch (closer to 0 = more robust)')
        emit('=' * 78)
        emit(f'{"policy":<24}{"dF1%":>9}{"dMCC%":>9}{"dKappa%":>10}{"dBalAcc pp":>12}')
        for p in args.policies:
            i = results[('ideal', p)]
            m = results[('mismatch', p)]

            def rel(a, b):
                return (b - a) / abs(a) * 100 if abs(a) > 1e-9 else float('nan')

            df1 = rel(i['f1_macro'], m['f1_macro'])
            dmcc = rel(i['mcc'], m['mcc'])
            dk = rel(i['cohen_kappa'], m['cohen_kappa'])
            dba = (m['balanced_accuracy'] - i['balanced_accuracy']) * 100
            emit(f'{LABELS.get(p, p):<24}{df1:>8.0f}%{dmcc:>8.0f}%{dk:>9.0f}%{dba:>11.1f}')

    emit('\n' + '=' * 78)
    emit('RANKINGS')
    emit('=' * 78)
    for scenario in args.scenarios:
        rank_f1 = sorted(args.policies, key=lambda p: -results[(scenario, p)]['f1_macro'])
        rank_mcc = sorted(args.policies, key=lambda p: -results[(scenario, p)]['mcc'])
        emit(f'[{scenario}] F1:  ' + ' > '.join(
            f'{p}({results[(scenario, p)]["f1_macro"]:.3f})' for p in rank_f1))
        emit(f'[{scenario}] MCC: ' + ' > '.join(
            f'{p}({results[(scenario, p)]["mcc"]:.3f})' for p in rank_mcc))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'campaign_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({f'{s}/{p}': results[(s, p)] for (s, p) in results}, f, indent=2)
    print(f'\nWrote outputs under: {args.out}')


if __name__ == '__main__':
    main()
