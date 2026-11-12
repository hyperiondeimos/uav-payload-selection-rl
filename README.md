# Optimal-design priors for reinforcement learning: optimism, information and specification

Code for the manuscript *Specification over accuracy: disentangling optimism
and information in optimal-design priors for reinforcement learning*.

The study seeds a tabular Q-learner with a D-optimal (Fisher information)
ranking and separates what the seed contributes: optimism (a flat control,
`Q0 = beta`), ranking information, and the specification of the criterion
(task-aware vs. task-agnostic). It has two parts:

* a random-instance benchmark of task-conditioned subset selection with exact
  regret (`generic_bench.py`);
* a case study in multi-sensor payload selection with four prior-reality
  regimes and a train/freeze/held-out protocol over ten seed blocks.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ and NumPy run the experiments; matplotlib and seaborn are only
needed for the figures. The numbers reported in the manuscript were produced
with Python 3.12.10 and NumPy 2.4.6.

## Reproducing the paper

```bash
python run_v12.py               # all runs, then tables, numbers and figures (~25 min, 16 cores)
python run_v12.py --skip-runs   # rebuild tables, numbers and figures from out_v12/
python check_prior.py           # static checks of the priors (zero scores, argmax, eps)
```

Raw outputs go to `out_v12/`. Every number in the manuscript's tables and
text is written by `tables_v12.py` into `paper/generated/` (`numbers.tex`
holds the in-text values as LaTeX macros) and the figures into `paper/`;
pass `--paper DIR` to write them elsewhere. Nothing is typed by hand.

## Layout

| File | Role |
|---|---|
| `ode_module.py` | sensor catalogue, sensitivity matrix, D-optimal criteria, feasibility |
| `rl_agents.py` | reward, state encoding, action space, learners and baselines |
| `simulator.py` | world generation, two-phase mission logic, detection, regimes |
| `metrics.py` | per-anomaly F1, binary agreement scores, bootstrap intervals |
| `run_campaign.py` | short horizon (fresh agent per world) |
| `run_leakfree_eval.py` | train / freeze / held-out protocol |
| `generic_bench.py` | random-instance benchmark |
| `run_v12.py` | runs every experiment of the paper |
| `aggregate_v12.py`, `tables_v12.py`, `make_figures_v12.py` | tables, numbers and figures |
| `check_prior.py` | static checks of the priors: zero scores, argmax per anomaly, solvability, eps |
| `probe_leakfree.py`, `make_figures.py` | checks and figures of the previous (v11) manuscript |
| `energy_shepard.py` | Shepard battery model (optional autonomy model) |

## Prior configurations

Two environment variables select the prior and the state encoding. The
defaults are the corrected versions.

| Variable | Values | Meaning |
|---|---|---|
| `PRIOR_MODE` | `anom` (default) | `log det(eps I + F[Theta_a, Theta_a])`, task-aware |
| | `logdet` | `log det(eps I + F)` over all characteristics, task-agnostic |
| | `det` | `det F` over all characteristics (degenerate; the v11 criterion) |
| `FIX_PHASE` | `1` (default) | the mission phase is part of the state |
| | `0` | v11 behaviour: phase-1 and phase-2 actions share Q-table entries |
| `EPS_PRIOR` | `1e-2` (default) | weak-prior regulariser `eps` |

`FIX_UTIL=1/0` is accepted as an alias of `PRIOR_MODE=anom/det`.
`PRIOR_MODE=det FIX_PHASE=0` reproduces the numbers of the previous (v11)
manuscript exactly.

## Learners

| Name | Initialisation | Exploration |
|---|---|---|
| `rl_no_ode` | `Q0 = 0` | uniform |
| `rl_flat` | `Q0 = beta` (optimism, no information) | uniform |
| `rl_initonly` | `Q0 = beta * u` | uniform |
| `rl` | `Q0 = beta * u` | `(1 - m) Unif + m softmax(u)` |
| `rl_decay` | as `rl`, bias decaying per dispatch | as `rl` |
| `ode` | deterministic `argmax u`, no learning | |
| `rand`, `full` | uniform choice; heaviest feasible payload | |

## Regimes

| Name | Detection ground truth |
|---|---|
| `ideal` (aligned) | the prior matrix itself |
| `mismatch` | prior with the thermal row halved |
| `leakfree` | structural decoupling (`S_REAL` in `simulator.py`) |
| `severe` | `leakfree` with thermal at 25% and LiDAR on theta4 at 0.5 (`S_SEVERE`) |

## Notes

* In the short horizon a fresh agent faces one world, where no state is ever
  visited twice; it is a zero-shot test of the prior's ranking, and
  `rl_no_ode` and `rl_flat` coincide there by construction.
* A bootstrap interval within one held-out run measures evaluation noise only.
  The paper treats seed blocks as replicates and reports paired differences
  across blocks.
* All randomness derives from the seeds passed to `numpy.random.default_rng`,
  so a clean run reproduces the reported numbers exactly.
