"""Shepard battery-discharge model with momentum-theory hover load.

Terminal voltage during discharge:

    V(it) = E0 - K * Q/(Q - it) * it - R*i + A*exp(-B*it)

with it the extracted charge (Ah), Q the pack capacity, i the current, R the
internal resistance, K the polarization constant and A, B the exponential-zone
parameters. The hover load is power-constant, so current rises as the voltage
sags and the end-of-discharge knee shortens flight time non-linearly.

Parameters correspond to a 4S 5.2 Ah pack on a 1.5 kg quadrotor airframe.
"""
from functools import lru_cache

import numpy as np

BATT = dict(
    Q=5.2,           # Ah, nominal capacity
    E0=16.9,         # V, constant-voltage term
    R=0.050,         # ohm, internal resistance
    K=0.012,         # V/Ah, polarization constant
    A=0.90,          # V, exponential-zone amplitude
    B=3.0,           # 1/Ah, exponential-zone time constant
    V_cut=13.2,      # V, cutoff (~3.30 V/cell)
    Q_reserve=0.15,  # fraction kept as landing reserve
)

DRONE = dict(
    base_mass=1.50,     # kg, frame + battery + motors, no selectable payload
    n_rotors=4,
    rotor_radius=0.12,  # m
    rho=1.225,          # kg/m^3
    fom=0.60,           # rotor figure of merit
    eff=0.75,           # electrical-to-mechanical efficiency (ESC + motor)
    g=9.81,
)
_A_DISK = DRONE['n_rotors'] * np.pi * DRONE['rotor_radius'] ** 2


def hover_power_W(mass_kg):
    """Momentum-theory electrical hover power for a total mass, in watts."""
    thrust = mass_kg * DRONE['g']
    p_mech = thrust ** 1.5 / np.sqrt(2 * DRONE['rho'] * _A_DISK) / DRONE['fom']
    return p_mech / DRONE['eff']


def shepard_voltage(it, i):
    b = BATT
    it = min(it, b['Q'] * 0.999)
    return b['E0'] - b['K'] * b['Q'] / (b['Q'] - it) * it - b['R'] * i \
        + b['A'] * np.exp(-b['B'] * it)


@lru_cache(maxsize=4096)
def _autonomy_cached(mass_mkey, pow_key, dt):
    b = BATT
    power = hover_power_W(mass_mkey / 1000.0) + pow_key / 10.0
    it = 0.0
    t = 0.0
    q_usable = b['Q'] * (1.0 - b['Q_reserve'])
    while it < q_usable:
        # Fixed-point solve of P = V(it, i) * i; V depends weakly on i via R.
        i = power / max(shepard_voltage(it, 0.0), 1.0)
        i = power / max(shepard_voltage(it, i), 1.0)
        v = shepard_voltage(it, i)
        if v <= b['V_cut']:
            break
        it += i * (dt / 3600.0)
        t += dt
    return t / 60.0


def autonomy_minutes(total_mass_kg, sensor_power_W=0.0, dt=5.0):
    """Flight minutes until the cutoff voltage or the usable capacity is spent.
    Memoised on rounded (mass, power) so repeated action-space queries are cheap."""
    return _autonomy_cached(round(total_mass_kg * 1000), round(sensor_power_W * 10), dt)


def autonomy_for_payload(sensor_ids, sensors):
    """Autonomy for base airframe plus payload, including sensor electrical draw."""
    payload_kg = sum(sensors[s][1] for s in sensor_ids) / 1000.0
    sensor_w = sum(sensors[s][2] for s in sensor_ids)
    return autonomy_minutes(DRONE['base_mass'] + payload_kg, sensor_w)


if __name__ == '__main__':
    from ode_module import SENSORS

    print('payload(g)  autonomy_shepard(min)  autonomy_linear(min)')
    for combo in [(), (1,), (1, 4), (2, 5), (2, 3, 5, 6), (1, 2, 3, 4, 5)]:
        w = sum(SENSORS[s][1] for s in combo)
        a_sh = (autonomy_for_payload(combo, SENSORS) if combo
                else autonomy_minutes(DRONE['base_mass']))
        a_lin = 25.0 * (1 - w / 1000.0)
        print(f'  {w:>6}      {a_sh:>10.1f}          {a_lin:>10.1f}')
