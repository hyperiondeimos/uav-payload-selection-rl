# Multi-sensor UAV payload and fleet selection under an optimal-design prior

Reproducible simulator and experiments for payload-and-fleet selection in
precision silviculture. A D-optimal (Fisher information) prior ranks sensor
subsets, a tabular Q-Learning agent selects the payload and the fleet size, and
autonomy is accounted for by either a linear weight proxy or a Shepard
non-linear battery-discharge model.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ and NumPy are enough to run the campaigns; matplotlib and seaborn
are only needed for the figures.

## Layout

| File | Role |
|---|---|
| `ode_module.py` | sensor catalogue, sensitivity matrix, D-optimal utility, feasibility |
| `energy_shepard.py` | Shepard battery model and momentum-theory hover power |
| `rl_agents.py` | reward, state encoding, action space and the policies |
| `simulator.py` | world generation, two-phase mission logic, detection |
| `metrics.py` | per-anomaly F1, binary agreement scores, bootstrap intervals |
| `run_campaign.py` | short-horizon factorial campaign |
| `run_leakfree_eval.py` | train / freeze / held-out protocol |
| `make_figures.py` | all figures |
| `probe_leakfree.py` | solvability and prior-suboptimality check of the leakfree scenario |

## Scenarios

| Name | Detection ground truth |
|---|---|
| `ideal` | the prior matrix itself |
| `mismatch` | prior with the thermal row halved |
| `leakfree` | structurally decoupled matrix (`S_REAL` in `simulator.py`) |

## Reproducing the results

Short-horizon factorial, 40 seeds starting at 7000, one fresh agent per world:

```bash
python run_campaign.py --seeds 40 --seed0 7000 --scenarios ideal mismatch leakfree \
                       --policies ode rl rl_no_ode rand
```

Held-out protocol, 200 training worlds then 60 disjoint evaluation worlds:

```bash
python run_leakfree_eval.py --train 200 --eval 60 --scenario leakfree
python run_leakfree_eval.py --train 200 --eval 60 --scenario mismatch
python run_leakfree_eval.py --train 200 --eval 60 --scenario ideal
```

Energy-model robustness check, same campaign with the Shepard model in place of
the linear proxy:

```bash
python run_campaign.py --seeds 40 --scenarios leakfree --energy shepard
```

Figures:

```bash
python make_figures.py --out img
```

Leakfree sanity check:

```bash
python probe_leakfree.py
```

## Notes

All randomness derives from the seed passed to `numpy.random.default_rng`, so a
clean run reproduces the reported numbers exactly. Bootstrap intervals use 500
resamples with a fixed seed. Outputs are written under `out/` and figures under
`img/`; neither is tracked.
