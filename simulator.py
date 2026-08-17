"""Mission simulator: world generation, two-phase selection and detection.

Each world holds one cluster per anomaly A1..A6 at a random position in a
36 x 36 m plot; distance_m is the euclidean distance to home at the origin.

A mission runs phase 1 with a fixed RGB+LiDAR payload; on a non-true-positive
outcome it escalates to phase 2, where the policy selects among the remaining
feasible payloads. Each cluster gets at most `max_attempts` missions.

Coverage of an anomaly is the mean, over its required characteristics, of the
best sensor in the fleet for that characteristic. Observed coverage adds
N(0, 0.10) noise and is thresholded into the four outcome classes.

Scenarios:
  ideal     detection uses the prior S (prior equals ground truth)
  mismatch  the thermal row of S is halved at detection time only
  leakfree  detection uses S_REAL, structurally decoupled from the prior
"""
import numpy as np

from ode_module import ANOMALIES, S as S_PRIOR, rgb_confidence
from rl_agents import build_action_space, compute_reward, state_key

PLOT_HALF = 18.0          # 36 x 36 m plot centred at the origin
HOME = (0.0, 0.0)
DET_NOISE_STD = 0.10
SCOUT_CONF_NOISE = 0.05
PHASE1_SENSORS = (1, 4)   # RGB + LiDAR

# Reality under the leakfree scenario. The planner keeps believing S_PRIOR.
# Every anomaly stays solvable, but the prior's argmax-utility payload is no
# longer coverage-optimal.
S_REAL = S_PRIOR.copy()
S_REAL[0, 4] = 0.75       # RGB gains a theta5 proxy
S_REAL[1, 0] = 0.35       # Multispectral over-trusted by the prior on theta1
S_REAL[2, 0] = 0.85       # Hyperspectral under-valued by the prior on theta1
S_REAL[4, :] *= 0.5       # thermal row halved


def detection_matrix(scenario):
    if scenario == 'ideal':
        return S_PRIOR.copy()
    if scenario == 'mismatch':
        s_det = S_PRIOR.copy()
        s_det[4, :] *= 0.5
        return s_det
    if scenario == 'leakfree':
        return S_REAL.copy()
    raise ValueError(f'unknown scenario: {scenario}')


def generate_world(rng):
    world = {}
    for aid in range(1, 7):
        x = float(rng.uniform(-PLOT_HALF, PLOT_HALF))
        y = float(rng.uniform(-PLOT_HALF, PLOT_HALF))
        distance_m = float(np.hypot(x - HOME[0], y - HOME[1]))
        world[aid - 1] = {'cluster_id': aid - 1, 'anomaly_id': aid,
                          'pos': (x, y), 'distance_m': distance_m}
    return world


def scout_confidence(anomaly_id, rng):
    base = rgb_confidence(anomaly_id)
    return float(max(0.1, min(0.95, base + rng.normal(0.0, SCOUT_CONF_NOISE))))


def coverage_real(sensors, anomaly_id, s_det):
    thetas = ANOMALIES[anomaly_id]['theta']
    if not thetas:
        return 0.0
    per_char = [max(s_det[sid - 1, k - 1] for sid in sensors) for k in thetas]
    return float(sum(per_char) / len(per_char))


def classify(avg_obs):
    if avg_obs >= 0.70:
        return 'true_positive'
    if avg_obs >= 0.40:
        return 'partial_detection'
    if avg_obs >= 0.20:
        return 'false_positive'
    return 'false_negative'


def _phase_actions(all_actions, phase_num):
    if phase_num == 1:
        acts = [a for a in all_actions if a['sensors'] == PHASE1_SENSORS]
    else:
        acts = [a for a in all_actions if a['sensors'] != PHASE1_SENSORS]
    for i, a in enumerate(acts):
        a = dict(a)
        acts[i] = a
        a['idx'] = i
    return acts


def run_mission(cluster, policy, scenario, s_det, rng, mission_no, attempts_prev):
    """One mission: phase 1, then phase 2 on a non-true-positive outcome."""
    anom_id = cluster['anomaly_id']
    distance_m = cluster['distance_m']
    conf = scout_confidence(anom_id, rng)
    all_actions = build_action_space(distance_m)
    records = []
    resolved = False

    for phase_num in (1, 2):
        actions = _phase_actions(all_actions, phase_num)
        if not actions:
            continue
        s_key = state_key(anom_id, conf, distance_m, attempts_prev)
        eps_at_select = policy.epsilon
        q_at_select = policy.q_size()
        a = policy.select(s_key, actions, rng)

        avg = coverage_real(a['sensors'], anom_id, s_det)
        avg_obs = float(max(0.0, min(1.0, avg + rng.normal(0.0, DET_NOISE_STD))))
        result = classify(avg_obs)
        reward = compute_reward(
            sensors=a['sensors'], n_drones=a['n_drones'], distance_m=distance_m,
            anomaly_id=anom_id, detection_result=result,
            autonomy_min=a['autonomy_min'], n_attempts=attempts_prev)

        records.append({
            'episode': policy.episode, 'mission': mission_no, 'phase': phase_num,
            'anomaly_id': anom_id, 'anomaly_name': ANOMALIES[anom_id]['name'],
            'cluster_id': cluster['cluster_id'], 'sensors': list(a['sensors']),
            'n_drones': a['n_drones'], 'weight_g': a['weight_g'],
            'autonomy_min': round(a['autonomy_min'], 4),
            'coverage_true': round(avg, 4), 'coverage_obs': round(avg_obs, 4),
            'result': result, 'reward': round(reward, 4),
            'epsilon': round(eps_at_select, 4), 'q_size': q_at_select,
            'policy': policy.name,
        })

        policy.update(s_key, a, reward)
        policy.end_episode()

        if result == 'true_positive':
            resolved = True
            break
    return records, resolved


def run_seed(seed, policy_or_name, scenario, max_attempts=3):
    """Full campaign for one world. Passing a policy NAME builds a fresh policy
    (no transfer); passing a Policy object reuses it across seeds, so epsilon
    keeps decaying and the Q-table keeps filling."""
    from rl_agents import Policy

    rng = np.random.default_rng(seed)
    s_det = detection_matrix(scenario)
    world = generate_world(rng)
    policy = Policy(policy_or_name) if isinstance(policy_or_name, str) else policy_or_name
    state = {cid: {'resolved': False, 'attempts': 0} for cid in world}
    records = []
    mission_no = 0
    missions_resolved = 0
    phase1_resolutions = 0

    while True:
        cands = [cid for cid, s in state.items()
                 if not s['resolved'] and s['attempts'] < max_attempts]
        if not cands:
            break
        min_att = min(state[cid]['attempts'] for cid in cands)
        pool = [cid for cid in cands if state[cid]['attempts'] == min_att]
        cid = int(rng.choice(pool))
        mission_no += 1
        recs, resolved = run_mission(
            world[cid], policy, scenario, s_det, rng,
            mission_no, state[cid]['attempts'])
        for r in recs:
            r['seed'] = seed
        records.extend(recs)
        state[cid]['attempts'] += 1
        if resolved:
            state[cid]['resolved'] = True
            missions_resolved += 1
            if len(recs) == 1 and recs[0]['phase'] == 1:
                phase1_resolutions += 1

    mission_summary = {
        'n_missions': mission_no,
        'n_resolved': missions_resolved,
        'resolution_rate': round(missions_resolved / mission_no, 4) if mission_no else 0.0,
        'phase1_resolution_rate': round(phase1_resolutions / mission_no, 4) if mission_no else 0.0,
        'n_clusters': len(world),
        'n_clusters_resolved': sum(1 for s in state.values() if s['resolved']),
        'epsilon_final': round(policy.epsilon, 4),
        'q_size_final': policy.q_size(),
        'episode_final': policy.episode,
    }
    return records, mission_summary
