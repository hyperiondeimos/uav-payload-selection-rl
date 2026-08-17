#!/usr/bin/env python3
"""Check the two properties the leakfree scenario must satisfy.

For each anomaly it reports (a) the best coverage achievable under S_REAL over
all feasible payloads, which confirms the anomaly stays solvable, and (b) the
coverage of the payload the ODE prior would pick. When (b) < (a) the prior is
no longer coverage-optimal and a learner has room to improve on it.
"""
from itertools import combinations

from ode_module import ANOMALIES, SENSORS, W_UTIL_G, utility_vector
from simulator import PHASE1_SENSORS, S_REAL, coverage_real

DISTANCE_M = 30.0   # representative distance; feasibility is weight-bound here


def feasible_payloads():
    ids = list(SENSORS)
    out = []
    for r in range(1, 7):
        for c in combinations(ids, r):
            if sum(SENSORS[s][1] for s in c) <= W_UTIL_G:
                out.append(c)
    return out


PAYLOADS = feasible_payloads()


def best_coverage(anom):
    return max(coverage_real(c, anom, S_REAL) for c in PAYLOADS)


def best_payload(anom):
    return max(PAYLOADS, key=lambda c: coverage_real(c, anom, S_REAL))


def ode_choice():
    """Phase-2 argmax of the D-optimal utility, excluding the phase-1 payload."""
    combos, u = utility_vector(DISTANCE_M)
    best, best_u = None, -1
    for c, uu in zip(combos, u):
        if c['sensors'] == PHASE1_SENSORS:
            continue
        if uu > best_u:
            best_u, best = uu, c['sensors']
    return best


if __name__ == '__main__':
    ode_payload = ode_choice()
    print(f'{"anomaly":<30}{"best_cov":>9}{"ode_cov":>9}{"gap":>7}  '
          f'best_payload | ode_payload')
    print('-' * 100)
    for aid, a in ANOMALIES.items():
        bc = best_coverage(aid)
        oc = coverage_real(ode_payload, aid, S_REAL)
        bp = '+'.join(SENSORS[s][0][:5] for s in best_payload(aid))
        op = '+'.join(SENSORS[s][0][:5] for s in ode_payload)
        flag = '  <-- ODE suboptimal' if oc + 1e-6 < bc else ''
        solvable = 'SOLVABLE' if bc >= 0.70 else 'hard'
        print(f'A{aid} {a["name"][:26]:<27}{bc:>9.2f}{oc:>9.2f}{bc - oc:>7.2f}  '
              f'{solvable:<9} {bp}  |  {op}{flag}')
