"""Policies and reward for the payload-and-fleet selection problem.

Policies:
  rl          ODE-biased Q-Learning: optimistic init Q0 = beta*u plus
              utility-biased exploration
  rl_decay    same, with beta and the exploration mix decaying per episode
  rl_initonly optimistic init with uniform exploration
  rl_no_ode   ablation: Q0 = 0, uniform exploration
  ode         deterministic argmax of the D-optimal utility, no learning
  rand        uniform random over feasible actions
  full        full-payload baseline: most sensors, then heaviest
"""
import math

import numpy as np

from ode_module import (
    ANOMALIES, N_MAX, S, autonomy_minutes, distribute_sensors, utility_vector,
)

ETA = 0.20            # learning rate
EPS_0 = 1.00
EPS_MIN = 0.05
EPS_DECAY = 0.97
BETA = 0.30           # optimistic ODE-init scale
BIAS_DECAY = 0.95     # per-episode decay of the ODE bias (1.0 = fixed)

W_DET = 0.50
W_EFF = 0.30
W_PEN = 0.20
ALPHA_S = 0.35
ALPHA_N = 0.35
ALPHA_D = 0.30
LAMBDA_1 = 0.30
LAMBDA_2 = 0.25
LAMBDA_3 = 0.35
LAMBDA_4 = 0.10
D_REF_MAX = 6000.0
J_TOTAL = 6


def _r_detection(result):
    return {'true_positive': 1.0, 'partial_detection': 0.5,
            'false_positive': -0.5, 'false_negative': -1.0}.get(result, 0.0)


def _r_efficiency(n_sensors, n_drones, distance_m):
    return (1.0 - ALPHA_S * (n_sensors / J_TOTAL)
                - ALPHA_N * (n_drones / N_MAX)
                - ALPHA_D * (distance_m / D_REF_MAX))


def _r_penalty(sensors, anomaly_id, autonomy_min, n_attempts):
    anom_theta = [t - 1 for t in ANOMALIES.get(anomaly_id, {}).get('theta', [])]
    p1 = 1.0 if any(all(S[sid - 1, k] < 0.15 for k in anom_theta) for sid in sensors) else 0.0
    p2 = 1.0 if len(sensors) > 4 else 0.0
    p3 = 1.0 if n_attempts >= 3 else 0.0
    p4 = 1.0 if autonomy_min < 2.5 else 0.0
    return LAMBDA_1 * p1 + LAMBDA_2 * p2 + LAMBDA_3 * p3 + LAMBDA_4 * p4


def compute_reward(sensors, n_drones, distance_m, anomaly_id, detection_result,
                   autonomy_min, n_attempts=0):
    return (W_DET * _r_detection(detection_result)
            + W_EFF * _r_efficiency(len(sensors), n_drones, distance_m)
            - W_PEN * _r_penalty(sensors, anomaly_id, autonomy_min, n_attempts))


def _dist_bin(d):
    if d < 2000:
        return 0
    if d < 4000:
        return 1
    if d < 6000:
        return 2
    return 3


def _conf_bin(c):
    return min(int(c / 0.2), 4)


def _attempt_bin(n):
    return 0 if n == 0 else (1 if n == 1 else 2)


def state_key(anomaly_id, confidence, distance_m, n_attempts):
    return (anomaly_id, _conf_bin(confidence), _dist_bin(distance_m),
            _attempt_bin(n_attempts))


def build_action_space(distance_m):
    """Every feasible (sensor subset, drone count) pair for a given distance."""
    combos, u = utility_vector(distance_m, criterion='D')
    actions = []
    for i, combo in enumerate(combos):
        sensors = combo['sensors']
        for n in range(1, N_MAX + 1):
            assignment = distribute_sensors(sensors, n, distance_m)
            if assignment is not None:
                auto_min = min(autonomy_minutes(d) for d in assignment if d)
                actions.append({
                    'idx': len(actions),
                    'sensors': sensors,
                    'n_drones': n,
                    'assignment': assignment,
                    'utility': float(u[i]),
                    'weight_g': combo['weight_g'],
                    'autonomy_min': auto_min,
                })
    return actions


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class QLearningAgent:
    """Tabular Q-Learning. With ode_bias the unvisited pairs start at beta*u and
    exploration is steered by softmax(u); both knobs are independent so the
    initialization can be biased without biasing the exploration."""

    def __init__(self, ode_bias=True, bias_decay=1.0, bias_explore=None):
        self.ode_bias = ode_bias
        self.bias_explore = ode_bias if bias_explore is None else bias_explore
        self.eta = ETA
        self.beta = BETA
        self.epsilon = EPS_0
        self.eps_min = EPS_MIN
        self.eps_decay = EPS_DECAY
        self.bias_decay = bias_decay
        self.bias = BETA     # current optimistic-init scale
        self.mix = 0.5       # current softmax share of the exploration step
        self.Q = {}
        self.episode = 0

    def q_get(self, s_key, a):
        key = (s_key, a['idx'])
        if key in self.Q:
            return self.Q[key]
        return self.bias * a['utility'] if self.ode_bias else 0.0

    def select_action(self, s_key, actions, rng):
        if not actions:
            raise RuntimeError('no feasible action')
        if rng.random() < self.epsilon:
            if self.bias_explore:
                u = np.array([a['utility'] for a in actions])
                unif = np.ones(len(actions)) / len(actions)
                probs = (1.0 - self.mix) * unif + self.mix * _softmax(u)
                probs /= probs.sum()
                return actions[int(rng.choice(len(actions), p=probs))]
            return actions[int(rng.integers(len(actions)))]
        best_a, best_q = None, -math.inf
        for a in actions:
            q = self.q_get(s_key, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update(self, s_key, a, reward, done=True):
        # Each dispatch ends the episode, so the successor value vanishes and
        # the temporal-difference target reduces to the immediate reward.
        q_sa = self.q_get(s_key, a)
        self.Q[(s_key, a['idx'])] = q_sa + self.eta * (reward - q_sa)

    def end_episode(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        if self.bias_decay < 1.0:
            self.bias *= self.bias_decay
            self.mix *= self.bias_decay
        self.episode += 1

    def q_size(self):
        return len(self.Q)


class Policy:
    """Uniform select/update interface over the learning and baseline policies."""

    def __init__(self, name):
        self.name = name
        if name == 'rl':
            self.agent = QLearningAgent(ode_bias=True, bias_decay=1.0)
        elif name == 'rl_decay':
            self.agent = QLearningAgent(ode_bias=True, bias_decay=BIAS_DECAY)
        elif name == 'rl_initonly':
            self.agent = QLearningAgent(ode_bias=True, bias_decay=1.0,
                                        bias_explore=False)
        elif name == 'rl_no_ode':
            self.agent = QLearningAgent(ode_bias=False)
        else:
            self.agent = None     # 'ode', 'rand' and 'full' are stateless

    @property
    def epsilon(self):
        return self.agent.epsilon if self.agent else 0.0

    @property
    def episode(self):
        return self.agent.episode if self.agent else 0

    def q_size(self):
        return self.agent.q_size() if self.agent else 0

    def select(self, s_key, actions, rng):
        if self.name in ('rl', 'rl_decay', 'rl_initonly', 'rl_no_ode'):
            return self.agent.select_action(s_key, actions, rng)
        if self.name == 'ode':
            return max(actions, key=lambda a: a['utility'])
        if self.name == 'rand':
            return actions[int(rng.integers(len(actions)))]
        if self.name == 'full':
            return max(actions, key=lambda a: (len(a['sensors']), a['weight_g']))
        raise ValueError(self.name)

    def update(self, s_key, a, reward):
        if self.agent:
            self.agent.update(s_key, a, reward, done=True)

    def end_episode(self):
        if self.agent:
            self.agent.end_episode()
