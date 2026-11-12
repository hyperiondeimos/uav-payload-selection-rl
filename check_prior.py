#!/usr/bin/env python3
"""Static checks of the priors used in the case study (no learning involved).

Reports, at a representative distance (feasibility is weight-bound in the plot):
  * how many feasible subsets the degenerate det criterion scores at exactly zero;
  * the phase-2 argmax of each prior for each anomaly, with its coverage under
    every regime, and the best achievable coverage (solvability);
  * whether the task-aware argmax changes with the regulariser eps.

    python check_prior.py
"""
import importlib
import os

import numpy as np

DIST = 20.0
REGIMES = ('ideal', 'mismatch', 'leakfree', 'severe')


def load(mode, eps='1e-2'):
    os.environ['PRIOR_MODE'] = mode
    os.environ['EPS_PRIOR'] = eps
    import ode_module
    import simulator
    importlib.reload(ode_module)
    importlib.reload(simulator)
    return ode_module, simulator


def name(om, combo):
    return '+'.join(om.SENSORS[s][0][:5] for s in combo)


def argmax_phase2(om, sim, aid):
    combos, u = om.utility_vector(DIST, anomaly_id=aid)
    pairs = [(uu, c['sensors']) for c, uu in zip(combos, u) if c['sensors'] != sim.PHASE1_SENSORS]
    return max(pairs, key=lambda t: t[0])[1]


def main():
    om, sim = load('det')
    combos, u = om.utility_vector(DIST)
    print(f'feasible subsets at {DIST:.0f} m: {len(combos)}; det criterion: '
          f'{int((u == 0).sum())} score exactly 0, {int((u > 0.1).sum())} score above 0.1')

    feas = [c['sensors'] for c in combos]
    print('\nphase-2 argmax of each prior and its coverage per regime')
    for mode in ('det', 'logdet', 'anom'):
        om, sim = load(mode)
        mats = {r: sim.detection_matrix(r) for r in REGIMES}
        print(f'-- {mode}')
        for aid in sorted(om.ANOMALIES):
            c = argmax_phase2(om, sim, aid)
            cov = ' '.join(f'{r}={sim.coverage_real(c, aid, mats[r]):.2f}' for r in REGIMES)
            print(f'  A{aid} {name(om, c):<30} {cov}')

    om, sim = load('anom')
    print('\nbest achievable coverage per regime (>= 0.70 means a true positive is reachable)')
    for r in REGIMES:
        S = sim.detection_matrix(r)
        best = [max(sim.coverage_real(c, aid, S) for c in feas) for aid in sorted(om.ANOMALIES)]
        print(f'  {r:<9}' + ' '.join(f'A{a}:{b:.2f}' for a, b in zip(sorted(om.ANOMALIES), best)))

    print('\ntask-aware argmax versus eps')
    ref = None
    for eps in ('1e-4', '1e-3', '1e-2', '1e-1', '1'):
        om, sim = load('anom', eps)
        picks = tuple(argmax_phase2(om, sim, a) for a in sorted(om.ANOMALIES))
        ref = ref or picks
        print(f'  eps={eps:<5} ' + ('same as eps=1e-4' if picks == ref else 'DIFFERENT: '
                                     + ', '.join(name(om, p) for p in picks)))


if __name__ == '__main__':
    main()
