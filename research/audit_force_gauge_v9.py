#!/usr/bin/env python3
"""Independent audit via the eliminated velocity recurrence, not hidden-state reuse.

Run with the existing project Python; JSON goes to stdout. This file does not
import or change the canonical pilot. Its arrays include one final unobserved
velocity, allowing reconstruction of the final hidden motor state as well.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def maximum(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run():
    root = Path(__file__).resolve().parent
    saved = json.loads((root / "nonlinear_force_gauge_v9_pilot_results.json").read_text())
    rerun = json.loads(subprocess.check_output(
        [sys.executable, str(root / "nonlinear_force_gauge_v9_pilot.py")], text=True))
    assert saved == rerun
    seed, n = 21661020, 8192
    rho, lam, c, beta = .65, .4, .2, 1.2
    rng = np.random.default_rng(seed)
    z, xi, sensor = rng.normal(0, .7, n), rng.normal(0, .06, n), rng.normal(0, .5, n + 1)

    def f(value):
        return rho * value - c * np.tanh(value)

    # Eliminated recurrence. Initial v[1]=f(0)+m[0]=0. A genuine
    # observation-based controller computes differences of integrated positions.
    v = np.zeros(n + 2)
    x, obs, u = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n)
    obs[0] = sensor[0]
    for t in range(n):
        displacement = obs[t] - (obs[t - 1] if t else 0.)
        u[t] = -.5 * np.tanh(displacement) + z[t]
        v[t + 2] = f(v[t + 1]) + lam * (v[t + 1] - f(v[t])) + beta * u[t] + xi[t]
        x[t + 1] = x[t] + v[t + 1]
        obs[t + 1] = x[t + 1] + sensor[t + 1]
    motor_from_observables = v[1:] - f(v[:-1])

    # Independent hidden-state realization driven by observations in its own
    # coordinate chart. tanh(rho*w+q) is explicitly causal, with no future state.
    w, q, ux = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n)
    xp, op = np.zeros(n + 1), np.zeros(n + 1)
    op[0] = sensor[0]
    for t in range(n):
        ux[t] = -.5 * np.tanh(op[t] - (op[t - 1] if t else 0.)) + z[t]
        w[t + 1] = rho * w[t] + q[t]
        q[t + 1] = (lam * q[t] + lam * c * np.tanh(w[t]) + beta * ux[t]
                    - c * np.tanh(rho * w[t] + q[t]) + xi[t])
        xp[t + 1] = xp[t] + w[t + 1]
        op[t + 1] = xp[t + 1] + sensor[t + 1]
    errors = {
        "velocity": maximum(v[:n + 1], w),
        "position": maximum(x, xp),
        "observation": maximum(obs, op),
        "command": maximum(u, ux),
        "force_gauge": maximum(q, motor_from_observables - c * np.tanh(v[:n + 1])),
        "sensor_difference_initialization": maximum(
            np.diff(np.r_[0., obs]), v[:n + 1] + np.diff(np.r_[0., sensor])),
    }
    assert max(errors.values()) < 1e-10
    assert v[0] == v[1] == q[0] == motor_from_observables[0] == 0.

    # Counterchecks expose which assumptions matter. A transformed nonzero
    # initial state must also transform force; forcing both force variables to
    # zero at v=1 creates distinct physical preparations.
    v0, m0 = 1., 0.
    compatible_q0 = m0 - c * np.tanh(v0)
    initial_compatible_error = abs(f(v0) + m0 - (rho * v0 + compatible_q0))
    zero_force_preparation_gap = abs(f(v0) - rho * v0)
    assert initial_compatible_error < 1e-14 and zero_force_preparation_gap > .15
    # At a nonzero common velocity, removing original physical force m means
    # q=-c*tanh(v), not q=0. Keeping physical intervention semantics agrees.
    physical_disconnect_equivalence = abs(f(v0) - (rho * v0 - c * np.tanh(v0)))
    # An added independent velocity shock eta destroys the convenient placement
    # of noise: qnext acquires phi(vnext+eta), hence its innovation depends on eta.
    eta = .2
    noise_placement_gap = abs(c * (np.tanh(rho * v0 + eta) - np.tanh(rho * v0)))
    assert noise_placement_gap > .02

    result = {
        "status": "INDEPENDENTLY TESTED; eliminated-recurrence audit",
        "seed": seed, "n": n, "numpy": np.__version__,
        "canonical_rerun_json_equal_saved": saved == rerun,
        "eliminated_recurrence": "v[t+2]=f(v[t+1])+lambda*(v[t+1]-f(v[t]))+beta*u[t]+xi[t]",
        "max_absolute_errors": errors,
        "initial_compatible_state_error": float(initial_compatible_error),
        "nonzero_velocity_two_zero_force_preparation_gap": float(zero_force_preparation_gap),
        "same_physical_disconnect_coordinate_error": float(physical_disconnect_equivalence),
        "added_velocity_noise_changes_transformed_innovation": float(noise_placement_gap),
        "scope": [
            "Pilot uses lagged timing, unlike v8 motor-first model.",
            "First measured displacement uses artificial previous observation zero; e[-1]=0 convention.",
            "Alternative is causal, two-state, input-affine, with same additive actuator noise.",
            "Ordinary commands preserve complete observations, including under observation feedback.",
            "A coordinate relabeling alone is not a distinct physical intervention claim.",
            "Zeroing q and zeroing m at nonzero v are different interventions/preparations.",
            "Velocity-independent actuator restriction excludes the alternative.",
            "No novelty, general learner, nonlinear-robot generality, or algorithm superiority follows.",
        ],
    }
    expanded_path = root / 'nonlinear_force_gauge_v9_results.json'
    if expanded_path.exists():
        expanded = json.loads(expanded_path.read_text())
        expanded_rerun = json.loads(subprocess.check_output(
            [sys.executable, str(root / 'nonlinear_force_gauge_v9.py')], text=True))
        assert expanded == expanded_rerun
        force_rms_error = force_max_error = 0.
        bound = 0.
        count = 0
        for policy, rows in expanded['connected'].items():
            for row in rows:
                rng = np.random.default_rng(row['seed'])
                zz = rng.normal(0, .7, n)
                noise = rng.normal(0, .06, n)
                ee = rng.normal(0, .5, n + 1)
                coins = rng.random(n)
                vv = np.zeros(n + 2)
                for t in range(n):
                    dt = vv[t] + ee[t] - (ee[t - 1] if t else 0.)
                    command = zz[t] if policy == 'open_loop' else zz[t] - .5 * np.tanh(dt)
                    if policy == 'zero_atom' and coins[t] < .2:
                        command = 0.
                    vv[t + 2] = f(vv[t + 1]) + lam * (vv[t + 1] - f(vv[t])) + beta * command + noise[t]
                # qnext=mnext-c*tanh(vnext), so no hidden-state reconstruction
                # is needed to independently derive these force statistics.
                difference = c * np.tanh(vv[1:n + 1])
                force_rms_error = max(force_rms_error, abs(float(np.sqrt(np.mean(difference**2))) - row['rms_total_delivered_force_difference']))
                force_max_error = max(force_max_error, abs(float(np.max(np.abs(difference))) - row['max_total_delivered_force_difference']))
                bound = max(bound, *[row[k] for k in ['max_velocity_difference', 'max_position_observation_difference', 'max_command_difference', 'max_force_gauge_error']])
                count += 1
        assert force_rms_error < 1e-12 and force_max_error < 1e-12 and bound < 1e-9
        removal = expanded['component_removal']
        vv = np.zeros(len(removal['preparation_commands']) + 2)
        for t, command in enumerate(removal['preparation_commands']):
            vv[t + 2] = f(vv[t + 1]) + lam * (vv[t + 1] - f(vv[t])) + beta * command
        initial = vv[len(removal['preparation_commands'])]
        va = initial
        response_a = []
        for _ in range(12):
            va = f(va)
            response_a.append(va)
        response_b = initial * rho ** np.arange(1, 13)
        distance = float(np.sqrt(np.mean((np.array(response_a) - response_b)**2)))
        removal_error = max(abs(initial - removal['initial_common_velocity']),
            maximum(response_a, removal['world_A_body_drag_response']),
            maximum(response_b, removal['world_B_motor_drag_response']),
            abs(distance - removal['response_velocity_rmse_distance']),
            abs(distance / 2 - removal['two_world_minimax_expected_velocity_rmse_lower_bound']),
            abs(distance**2 / 4 - removal['two_world_equal_prior_velocity_mse_lower_bound']))
        assert removal_error < 1e-12
        result['expanded_artifact'] = dict(canonical_rerun_json_equal_saved=True,
            independently_reconstructed_trajectories=count,
            maximum_saved_observable_equivalence_error=bound,
            maximum_force_rms_discrepancy=force_rms_error,
            maximum_force_max_discrepancy=force_max_error,
            independent_physical_removal_metric_error=removal_error,
            removal_velocity_response_distance=distance,
            physical_removal_scope='Requires explicitly different physical drag locations/component boundaries; cannot follow from coordinate relabeling alone.')
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
