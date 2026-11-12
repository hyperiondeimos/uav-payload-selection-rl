#!/usr/bin/env python3
"""Run every experiment of the v12 manuscript, then build its tables and figures.

    python run_v12.py                 # everything (about 25 min on 16 cores)
    python run_v12.py --skip-runs     # only aggregate, tables and figures

Outputs go to out_v12/; tables and in-text numbers to paper/generated and
figures to paper/ (override with --paper, e.g. the manuscript folder).

Prior configurations (environment variables read by ode_module/simulator):
  orig          PRIOR_MODE=det    FIX_PHASE=0   the v11 code, for reproduction only
  blind_det     PRIOR_MODE=det    FIX_PHASE=1   degenerate task-agnostic prior
  blind_logdet  PRIOR_MODE=logdet FIX_PHASE=1   task-agnostic log det prior
  anom          PRIOR_MODE=anom   FIX_PHASE=1   task-aware prior (the default)
The Q0 = 0 (rl_no_ode) and flat (rl_flat) learners never read the prior, so they
are run once per block and shared across configurations.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_v12')
SC = ['ideal', 'mismatch', 'leakfree', 'severe']
BLOCKS = [7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
SHORT_BLOCKS = [7000, 8000, 9000]
CFG = {
    'orig':         {'PRIOR_MODE': 'det',    'FIX_PHASE': '0'},
    'blind_det':    {'PRIOR_MODE': 'det',    'FIX_PHASE': '1'},
    'blind_logdet': {'PRIOR_MODE': 'logdet', 'FIX_PHASE': '1'},
    'anom':         {'PRIOR_MODE': 'anom',   'FIX_PHASE': '1'},
}


def jobs():
    J = []
    ho = ['run_leakfree_eval.py', '--train', '200', '--eval', '60']
    for s0 in BLOCKS:
        for sc in SC:
            base = [*ho, '--seed0', str(s0), '--scenario', sc]
            d = lambda cfg: os.path.join(OUT, 'heldout', cfg, f'{sc}_s{s0}')
            J.append((d('anom'), CFG['anom'], [*base, '--out', d('anom')]))
            for cfg in ('blind_logdet', 'blind_det'):
                J.append((d(cfg), CFG[cfg], [*base, '--out', d(cfg), '--learners', 'rl', 'rl_initonly']))
            J.append((d('flat'), CFG['anom'], [*base, '--out', d('flat'), '--learners', 'rl_flat',
                                               '--no-baselines']))
            if s0 == 7000:
                J.append((d('orig'), CFG['orig'], [*base, '--out', d('orig')]))
                for eps in ('1e-3', '1e-1'):
                    de = os.path.join(OUT, 'heldout', f'anom_eps{eps}', f'{sc}_s{s0}')
                    J.append((de, {**CFG['anom'], 'EPS_PRIOR': eps}, [*base, '--out', de]))
    pols = ['ode', 'rl', 'rl_initonly', 'rl_no_ode', 'rand', 'full']
    for cfg in ('orig', 'blind_det', 'blind_logdet', 'anom'):
        for s0 in (SHORT_BLOCKS if cfg != 'orig' else [7000]):
            d = os.path.join(OUT, 'short', cfg, f's{s0}')
            J.append((d, CFG[cfg], ['run_campaign.py', '--seeds', '40', '--seed0', str(s0),
                                    '--scenarios', *SC, '--policies', *pols, '--out', d]))
    for s0 in SHORT_BLOCKS:
        d = os.path.join(OUT, 'short', 'flat', f's{s0}')
        J.append((d, CFG['anom'], ['run_campaign.py', '--seeds', '40', '--seed0', str(s0),
                                   '--scenarios', *SC, '--policies', 'rl_flat', '--out', d]))
    return J


def run(job):
    d, env, args = job
    os.makedirs(d, exist_ok=True)
    e = dict(os.environ)
    e.update(env)
    e['PYTHONIOENCODING'] = 'utf-8'
    t = time.time()
    with open(os.path.join(d, 'run.log'), 'w', encoding='utf-8') as f:
        rc = subprocess.call([sys.executable, *args], cwd=HERE, env=e, stdout=f, stderr=subprocess.STDOUT)
    return d, rc, time.time() - t


def share():
    """Copy the prior-free learners into every FIX_PHASE=1 configuration."""
    for s0 in BLOCKS:
        for sc in SC:
            srcs = [os.path.join(OUT, 'heldout', 'anom', f'{sc}_s{s0}', 'rl_no_ode_trained_summary.json'),
                    os.path.join(OUT, 'heldout', 'flat', f'{sc}_s{s0}', 'rl_flat_trained_summary.json')]
            for cfg in ('anom', 'blind_logdet', 'blind_det'):
                dst = os.path.join(OUT, 'heldout', cfg, f'{sc}_s{s0}')
                for src in srcs:
                    if os.path.exists(src) and os.path.isdir(dst) and os.path.dirname(src) != dst:
                        shutil.copy(src, dst)
    for s0 in SHORT_BLOCKS:
        for sc in SC:
            for ext in ('summary.json', 'records.jsonl'):
                src = os.path.join(OUT, 'short', 'flat', f's{s0}', sc, f'rl_flat_{ext}')
                for cfg in ('anom', 'blind_logdet', 'blind_det'):
                    dst = os.path.join(OUT, 'short', cfg, f's{s0}', sc)
                    if os.path.exists(src) and os.path.isdir(dst):
                        shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--skip-runs', action='store_true')
    ap.add_argument('--paper', default=os.path.join(HERE, 'paper'),
                    help='where tables (generated/) and figures go')
    a = ap.parse_args()
    if not a.skip_runs:
        J = jobs()
        print(f'{len(J)} jobs on {a.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for d, rc, dt in ex.map(run, J):
                if rc:
                    print(f'FAILED rc={rc} {os.path.relpath(d, OUT)} (see run.log)', flush=True)
        share()
        for extra in ([], ['--instances', '200', '--J', '8', '--K', '8', '--T', '8']):
            subprocess.check_call([sys.executable, 'generic_bench.py', '--workers', str(a.workers),
                                   '--deltas', '0', '0.1', '0.2', '0.3', '0.4', '0.5', *extra], cwd=HERE)
    py = sys.executable
    subprocess.check_call([py, 'aggregate_v12.py'], cwd=HERE, stdout=subprocess.DEVNULL)
    subprocess.check_call([py, 'tables_v12.py', '--out', os.path.join(a.paper, 'generated')], cwd=HERE)
    subprocess.check_call([py, 'make_figures_v12.py', '--out', a.paper], cwd=HERE)
    stray = os.path.join(a.paper, 'fig_cost.png')     # not used by the manuscript
    if os.path.exists(stray):
        os.remove(stray)


if __name__ == '__main__':
    main()
