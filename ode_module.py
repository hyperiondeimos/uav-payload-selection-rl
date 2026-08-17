"""Optimal-design (Fisher information) core: sensor catalogue, sensitivity
matrix, D-optimal utility and payload feasibility.

Sensor tuple: (name, weight_g, power_W, noise_sigma).
"""
from itertools import combinations

import numpy as np

SENSORS = {
    1: ('RGB',           120, 2.5, 0.15),
    2: ('Multispectral', 230, 4.0, 0.10),
    3: ('Hyperspectral', 480, 8.5, 0.08),
    4: ('LiDAR',         350, 6.0, 0.12),
    5: ('Thermal',       180, 3.5, 0.10),
    6: ('Gas_aerosol',    80, 1.5, 0.20),
}
SENSOR_IDS = list(SENSORS.keys())

J = 6                                  # selectable sensors
K = 6                                  # measurable characteristics
GPS_WEIGHT_G = 25
W_MAX_G = 1000
W_UTIL_G = W_MAX_G - GPS_WEIGHT_G      # 975 g useful payload
T_BASE = 25.0                          # min, linear autonomy proxy at zero payload
V_MS = 10.0                            # m/s cruise speed
N_MAX = 10                             # max drones per mission

# Prior sensitivity matrix S in [0,1]^{J x K}. In the 'ideal' scenario this is
# also the ground truth of detection; simulator.py decouples the two.
S = np.array([
    [0.10, 0.60, 0.10, 0.70, 0.00, 0.00],   # RGB
    [0.90, 0.70, 0.30, 0.20, 0.10, 0.00],   # Multispectral
    [0.40, 0.30, 0.90, 0.10, 0.10, 0.00],   # Hyperspectral
    [0.00, 0.10, 0.00, 0.90, 0.00, 0.00],   # LiDAR
    [0.70, 0.10, 0.10, 0.00, 0.90, 0.20],   # Thermal
    [0.00, 0.00, 0.00, 0.00, 0.20, 0.90],   # Gas/aerosol
], dtype=np.float64)

# 'theta' lists the characteristics an anomaly requires (1-indexed).
ANOMALIES = {
    1: {'name': 'water_stress',              'min_sensors': (2, 5),       'theta': (1, 5)},
    2: {'name': 'pest_infestation',          'min_sensors': (1, 2),       'theta': (2,)},
    3: {'name': 'fungal_disease',            'min_sensors': (3,),         'theta': (3,)},
    4: {'name': 'planting_failure',          'min_sensors': (1, 4),       'theta': (4,)},
    5: {'name': 'atmospheric_contamination', 'min_sensors': (6, 5),       'theta': (5, 6)},
    6: {'name': 'mixed_A1_A4',               'min_sensors': (1, 2, 5, 4), 'theta': (1, 4, 5)},
}

# When True, autonomy comes from the Shepard model instead of the linear
# weight proxy. Set by run_campaign.py --energy shepard.
USE_SHEPARD = False


def fim_sensor(sensor_id):
    _, _, _, sigma = SENSORS[sensor_id]
    s_j = S[sensor_id - 1]
    return np.diag((s_j ** 2) / (sigma ** 2))


_FIM_INDIVIDUAL = {sid: fim_sensor(sid) for sid in SENSOR_IDS}


def fim_combined(sensor_ids):
    F = np.zeros((K, K), dtype=np.float64)
    for sid in sensor_ids:
        F += _FIM_INDIVIDUAL[sid]
    return F


def criterion_d(F):
    return max(float(np.linalg.det(F)), 0.0)


def payload_weight(sensor_ids):
    return sum(SENSORS[sid][1] for sid in sensor_ids)


def autonomy_minutes(sensor_ids):
    if USE_SHEPARD:
        import energy_shepard as _es
        payload_kg = payload_weight(sensor_ids) / 1000.0
        sensor_w = sum(SENSORS[s][2] for s in sensor_ids)
        return _es.autonomy_minutes(_es.DRONE['base_mass'] + payload_kg, sensor_w)
    return T_BASE * (1.0 - payload_weight(sensor_ids) / W_MAX_G)


def max_range_m(sensor_ids):
    return V_MS * autonomy_minutes(sensor_ids) * 60.0 / 2.0


def is_feasible(sensor_ids, distance_m):
    if payload_weight(sensor_ids) > W_UTIL_G:
        return False
    if distance_m > max_range_m(sensor_ids):
        return False
    return True


def all_combinations(distance_m):
    results = []
    for r in range(1, J + 1):
        for combo in combinations(SENSOR_IDS, r):
            if is_feasible(combo, distance_m):
                F = fim_combined(combo)
                results.append({
                    'sensors': combo,
                    'weight_g': payload_weight(combo),
                    'autonomy_min': autonomy_minutes(combo),
                    'fim': F,
                    'phi_D': criterion_d(F),
                })
    return results


def utility_vector(distance_m, criterion='D'):
    """Min-max normalised D-optimal utility over the feasible subsets."""
    combos = all_combinations(distance_m)
    if not combos:
        return [], np.array([])
    phi_vals = np.array([c['phi_D'] for c in combos], dtype=np.float64)
    phi_min, phi_max = phi_vals.min(), phi_vals.max()
    if phi_max - phi_min < 1e-12:
        u = np.ones(len(combos))
    else:
        u = (phi_vals - phi_min) / (phi_max - phi_min)
    return combos, u


def distribute_sensors(sensor_ids, n_drones, distance_m):
    """Greedy load balance of the payload across n drones, heaviest first."""
    sorted_sensors = sorted(sensor_ids, key=lambda sid: SENSORS[sid][1], reverse=True)
    loads = [0.0] * n_drones
    assign = [[] for _ in range(n_drones)]
    for sid in sorted_sensors:
        i_min = int(np.argmin(loads))
        assign[i_min].append(sid)
        loads[i_min] += SENSORS[sid][1]
    for i in range(n_drones):
        if assign[i] and not is_feasible(tuple(assign[i]), distance_m):
            return None
    return [tuple(a) for a in assign]


def rgb_confidence(anomaly_id):
    """Geometric mean of the RGB row of S over the anomaly's characteristics."""
    anom = ANOMALIES.get(anomaly_id)
    if anom is None:
        return 0.3
    theta_idxs = [t - 1 for t in anom['theta']]
    s_rgb = S[0, theta_idxs]
    if len(s_rgb) == 0:
        return 0.1
    return float(np.exp(np.mean(np.log(np.maximum(s_rgb, 1e-6)))))
