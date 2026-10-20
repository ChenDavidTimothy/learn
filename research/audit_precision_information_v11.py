#!/usr/bin/env python3
"""Independent batch-Gaussian and nested-data audit of the v11 precision test.

The information calculation conditions the joint finite Gaussian distribution
in one batch rather than reproducing the canonical Riccati recursion. Physical
replays and action-cost derivatives eliminate motor state independently.
"""
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_goal_control_v10 import recurrence, independent_metrics, independent_cost_gradient, independent_cost


ROOT = Path(__file__).resolve().parent


def batch_information(sensor_sd, lam):
    A = np.array([[1., .65, 1.], [0., .65, 1.], [0., 0., lam]])
    motor_direction = np.array([0., 0., 1.])
    powers = [np.linalg.matrix_power(A, k) for k in range(87)]
    # Motor innovations xi0,...,xi68 determine the last-action state s69.
    state_loadings = np.column_stack([powers[68 - k] @ motor_direction for k in range(69)])
    measurement_loadings = np.zeros((70, 69))
    for t in range(70):
        for k in range(t):
            measurement_loadings[t, k] = (powers[t - 1 - k] @ motor_direction)[0]
    covariance_y = .03**2 * measurement_loadings @ measurement_loadings.T + sensor_sd**2 * np.eye(70)
    covariance_sy = .03**2 * state_loadings @ measurement_loadings.T
    if sensor_sd:
        P = .03**2 * state_loadings @ state_loadings.T - covariance_sy @ np.linalg.solve(covariance_y, covariance_sy.T)
    else:
        # Exact-Gaussian conditioning with singular measurement covariance:
        # S*(I-Y^+Y)*S^T. Compute the Moore-Penrose nullspace projection via
        # full SVD to avoid subtracting almost equal covariance matrices.
        _, singular, right = np.linalg.svd(measurement_loadings, full_matrices=True)
        rank = np.count_nonzero(singular > 1e-11 * singular[0])
        nullspace = right[rank:].T
        P = .03**2 * state_loadings @ nullspace @ nullspace.T @ state_loadings.T
    future_row = powers[17][0]
    current = float(future_row @ P @ future_row)
    fresh = float(.03**2 * (powers[16] @ motor_direction)[0]**2)
    variance = current + fresh
    return dict(sensor_sd=sensor_sd, **{'lambda':lam}, posterior_covariance_at_last_action=P.tolist(),
        terminal_variance_current_state=current, terminal_variance_fresh_innovation=fresh,
        terminal_position_conditional_mse_lower_bound=variance,
        terminal_position_conditional_sd=math.sqrt(variance),
        whole_stop_population_probability_upper_bound=math.erf(.3 / math.sqrt(2 * variance)),
        full_current_state_probability_upper_bound=math.erf(.3 / math.sqrt(2 * fresh)))


def independent_filter(model, episode, sensor_sd):
    mean, covariance = np.zeros(3), np.zeros((3, 3))
    rho, lam, c, beta = model
    estimates = []
    for t, u in enumerate(episode['commands']):
        gain = covariance[:, 0] / (covariance[0, 0] + sensor_sd**2)
        mean += gain * (episode['observations'][t] - mean[0])
        covariance -= np.outer(gain, covariance[0])
        estimates.append(mean.copy())
        x, v, m = mean
        slope = rho - c * (1 - np.tanh(v)**2)
        vn = rho * v - c * np.tanh(v) + m
        mean = np.array([x + vn, vn, lam * m + beta * u])
        A = np.array([[1., slope, 1.], [0., slope, 1.], [0., 0., lam]])
        covariance = A @ covariance @ A.T
        if t < 70:
            covariance[2, 2] += .03**2
    return np.array(estimates)


def main():
    prior = json.loads((ROOT / 'goal_control_v10_results.json').read_text())
    old = next(p for p in prior['plants'] if p['true_theta'] == [.65, .7, 0., 1.2])
    data = json.loads((ROOT / 'precision_information_v11_results.json').read_text())
    errors = dict(information_covariance=0., information_scalar=0., long_training=0.,
                  fit_objective=0., replay=0., observations=0., estimates=0., metrics=0., summary=0.,
                  MPC_warm_cost=0., MPC_selected_cost=0., MPC_projected_gradient=0., MPC_action=0., paired_standard_sensor_noise=0.)
    def check(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    for limit in data['information_limits']:
        independent = batch_information(limit['sensor_sd'], limit['lambda'])
        check('information_covariance', independent['posterior_covariance_at_last_action'], limit['posterior_covariance_at_last_action'])
        for key in independent:
            if key != 'posterior_covariance_at_last_action':
                check('information_scalar', independent[key], limit[key])
    episodes = optimizer_records = exact_old_episodes = 0
    for row, previous in zip(data['runs'], old['runs']):
        assert row['seed'] == previous['seed']
        assert row['short_theta'] == previous['linear_theta']
        assert row['long_training_commands'][:512] == previous['training_commands']
        assert row['long_training_positions'][:513] == previous['training_positions']
        # Extend the complete old physical trajectory; preserve its initial
        # sensor stream exactly and append only observations at t>=513.
        rng = np.random.default_rng(row['seed'] + 4000000)
        u = list(previous['training_commands'])
        while len(u) < 8192:
            duration = int(rng.choice((1, 3, 8)))
            u.extend([float(rng.uniform(-1., 1.))] * duration)
        u = np.array(u[:8192])
        states = recurrence(old['true_theta'], u)
        obs = np.r_[previous['training_positions'], states[513:, 0] + rng.normal(0, .2, 8193 - 513)]
        check('long_training', u, row['long_training_commands'])
        check('long_training', obs, row['long_training_positions'])
        # Countercheck: resetting physical velocity/motor at512 would not
        # reproduce the recorded continuation on these actual trajectories.
        reset = recurrence(old['true_theta'], u[512:])
        assert np.max(np.abs(reset[:, 0] + states[512, 0] - states[512:, 0])) > .1
        for record in row['long_fit_records']:
            fitted = recurrence(record['theta'], u)
            check('fit_objective', np.sum((fitted[:, 0] - obs)**2), record['sse'])
        assert row['long_theta'] == min(row['long_fit_records'], key=lambda record: record['sse'])['theta']
        for sd_label, case in row['cases'].items():
            sd = float(sd_label)
            assert sd == case['sensor_sd']
            for j, task in enumerate(case['tasks']):
                assert task['goal_schedule'] == previous['tasks'][j]['goal_schedule']
                for label, episode in task['controllers'].items():
                    model = row['short_theta'] if label == 'short512' else row['long_theta'] if label == 'long8192' else old['true_theta']
                    assert episode['seed'] == row['seed'] + 2000000 + j
                    rngtest = np.random.default_rng(episode['seed'])
                    sensor, motor = rngtest.normal(0, sd, 87), rngtest.normal(0, .03, 86)
                    motor[70:] = 0.
                    true_states = recurrence(old['true_theta'], episode['commands'], noises=motor)
                    check('replay', true_states, episode['states'])
                    check('observations', true_states[:, 0] + sensor, episode['observations'])
                    check('estimates', independent_filter(model, episode, sd), episode['estimates'])
                    values = independent_metrics(true_states, episode['commands'], episode['goals'])
                    for key, value in values.items():
                        if value is None or isinstance(value, (int, bool)):
                            assert value == episode[key]
                        else:
                            check('metrics', value, episode[key])
                    assert np.max(np.abs(episode['commands'])) <= 1.
                    assert np.all(np.array(episode['commands'])[70:] == 0.)
                    warm = np.zeros(8)
                    for t, record in enumerate(episode['optimizers']):
                        if record is None:
                            assert t >= 70
                            continue
                        actions = np.array(record['selected_action_plan'])
                        estimate, goal = episode['estimates'][t], episode['goals'][t]
                        cost, gradient = independent_cost_gradient(actions, estimate, goal, model)
                        projected = np.max(np.abs(actions - np.clip(actions - gradient, -1., 1.)))
                        check('MPC_warm_cost', independent_cost(warm, estimate, goal, model), record['initial_cost'])
                        check('MPC_selected_cost', cost, record['selected_cost'])
                        check('MPC_projected_gradient', projected, record['projected_gradient_max'])
                        check('MPC_action', actions[0], episode['commands'][t])
                        assert record['selected_cost'] <= record['initial_cost'] + 1e-8
                        warm = np.r_[actions[1:], 0.]
                        optimizer_records += 1
                    if sd == .2 and label != 'long8192':
                        old_label = 'linear_MPC' if label == 'short512' else 'oracle_model_EKF'
                        assert episode == previous['tasks'][j]['controllers'][old_label]
                        exact_old_episodes += 1
                    if sd == .02:
                        high = row['cases']['0.2']['tasks'][j]['controllers'][label]
                        high_noise = (np.array(high['observations']) - np.array(high['states'])[:, 0]) / .2
                        low_noise = (np.array(episode['observations']) - np.array(episode['states'])[:, 0]) / .02
                        check('paired_standard_sensor_noise', high_noise, low_noise)
                    episodes += 1
    for sd, summary in data['summary'].items():
        for label, metrics in summary.items():
            for metric in ['normalized_tracking_cost', 'late_position_rmse']:
                clusters = np.array([np.mean([task['controllers'][label][metric] for task in row['cases'][sd]['tasks']]) for row in data['runs']])
                values = dict(mean=np.mean(clusters), mc_se=np.std(clusters, ddof=1) / np.sqrt(12),
                              median=np.median(clusters), q10=np.quantile(clusters, .1), q90=np.quantile(clusters, .9))
                for key, value in values.items():
                    check('summary', value, metrics[metric][key])
            assert metrics['stop_successes'] == sum(task['controllers'][label]['stop_success'] for row in data['runs'] for task in row['cases'][sd]['tasks'])
            assert metrics['task_trials'] == 24
    assert max(errors.values()) < 1e-7
    return dict(status='INDEPENDENTLY TESTED', batch_conditioned_information_cases=len(data['information_limits']),
        nested_training_prefixes_checked=len(data['runs']), physical_episodes_replayed=episodes,
        plans_independently_checked=optimizer_records, v10_episodes_reproduced_exactly=exact_old_episodes,
        maximum_absolute_errors=errors,
        scope=[
            'Batch Gaussian conditioning independently verifies posterior at last action69, terminal propagation17, and final innovation69 propagated16.',
            'Probability bound concerns true terminal position under the forced-coast protocol, conditional on information available before final action innovation; whole success is a subset.',
            'Known parameters, zero initial state, calibrated sensor/noise laws, linearity and observation-adaptive controls are required for the covariance bound.',
            'Future coast observations cannot affect forcedzero physical commands; the bound is not a claim about posterior uncertainty after observing those future measurements.',
            'All short data retained literally; long data continue original velocity/motor history without reset.',
            'Sensor intervention changes both actual measurement precision and filterR while preserving paired standardnormal streams and motor innovations.',
            'Extra training and better sensing costs are not equated; no universal efficiency or architectural conclusion follows.',
            'MC SE uses twelve fitted-model seed clusters, averaging two tasks within seed.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
