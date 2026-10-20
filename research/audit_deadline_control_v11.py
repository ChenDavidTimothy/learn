#!/usr/bin/env python3
"""Independent variable-horizon deadline, endpoint and paired-policy audit."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_goal_control_v10 import recurrence, independent_metrics
from audit_precision_information_v11 import independent_filter, batch_information


def plan(actions, state, goal, theta):
    rho, lam, c, beta = theta
    n, total = len(actions), len(actions) + 16
    v = np.zeros(total + 2)
    dv = np.zeros((total + 2, n))
    def f(value):
        return rho * value - c * np.tanh(value)
    def fp(value):
        return rho - c * (1 - np.tanh(value)**2)
    v[0], v[1] = state[1], f(state[1]) + state[2]
    x, dx, cost, gradient = state[0], np.zeros(n), 0., np.zeros(n)
    for t in range(total):
        u = actions[t] if t < n else 0.
        v[t + 2] = f(v[t + 1]) + lam * (v[t + 1] - f(v[t])) + beta * u
        dv[t + 2] = (fp(v[t + 1]) + lam) * dv[t + 1] - lam * fp(v[t]) * dv[t]
        if t < n:
            dv[t + 2, t] += beta
        x += v[t + 1]
        dx += dv[t + 1]
        cost += (x - goal)**2 + .3 * v[t + 1]**2 + .03 * u**2
        gradient += 2 * (x - goal) * dx + .6 * v[t + 1] * dv[t + 1]
        if t < n:
            gradient[t] += .06 * u
    return float(cost), gradient


def endpoint(episode, sd):
    theta = [.65, .7, 0., 1.2]
    state = episode['estimates'][69]
    last = episode['commands'][69]
    future = recurrence(theta, np.r_[last, np.zeros(16)], state)[-1, 0]
    # Independent rollout of a unit change in the last action, holding its
    # actual conditional state/history fixed.
    other = recurrence(theta, np.r_[last + 1., np.zeros(16)], state)[-1, 0]
    sensitivity = other - future
    bias = future - episode['goals'][-1]
    variance = batch_information(sd, .7)['terminal_position_conditional_mse_lower_bound']
    action = last - bias / sensitivity
    return dict(conditional_mean_endpoint_error=float(bias), conditional_variance=variance,
        conditional_endpoint_mse=float(bias**2 + variance),
        realized_endpoint_error=float(episode['states'][-1][0] - episode['goals'][-1]),
        mean_centering_last_action=float(action), mean_centering_action_is_feasible=bool(abs(action) <= 1.))


def main():
    root = Path(__file__).resolve().parent
    old = json.loads((root / 'precision_information_v11_results.json').read_text())
    data = json.loads((root / 'deadline_control_v11_results.json').read_text())
    theta = old['meta']['plant']
    errors = dict(replay=0., observations=0., estimates=0., metrics=0., endpoint=0.,
                  warm_cost=0., selected_cost=0., projected_gradient=0., action=0.,
                  paired_endpoint_innovation=0., summary=0.)
    def check(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    count = records = preserved_prefixes = 0
    for row, previous in zip(data['runs'], old['runs']):
        assert row['seed'] == previous['seed']
        for sd_label, case in row['cases'].items():
            sd = float(sd_label)
            for j, task in enumerate(case['tasks']):
                original = previous['cases'][sd_label]['tasks'][j]
                assert task['goal_schedule'] == original['goal_schedule']
                old_endpoint = endpoint(original['controllers']['known_parameters'], sd)
                for key, value in old_endpoint.items():
                    if isinstance(value, bool):
                        assert value == task['unaware12_known_endpoint_diagnostic'][key]
                    else:
                        check('endpoint', value, task['unaware12_known_endpoint_diagnostic'][key])
                for model_label in ['short512', 'known_parameters']:
                    unaware = task['variants'][model_label + '_unaware16']
                    aware = task['variants'][model_label + '_aware16']
                    assert unaware['commands'][:63] == aware['commands'][:63]
                    assert unaware['states'][:64] == aware['states'][:64]
                    assert unaware['observations'][:64] == aware['observations'][:64]
                    assert unaware['estimates'][:64] == aware['estimates'][:64]
                    preserved_prefixes += 1
                for name, episode in task['variants'].items():
                    known = name.startswith('known_')
                    aware = name.endswith('_aware16')
                    model = theta if known else previous['short_theta']
                    assert episode['seed'] == row['seed'] + 2000000 + j
                    rng = np.random.default_rng(episode['seed'])
                    sensor, process = rng.normal(0, sd, 87), rng.normal(0, .03, 86)
                    process[70:] = 0.
                    states = recurrence(theta, episode['commands'], noises=process)
                    check('replay', states, episode['states'])
                    check('observations', states[:, 0] + sensor, episode['observations'])
                    check('estimates', independent_filter(model, episode, sd), episode['estimates'])
                    values = independent_metrics(states, episode['commands'], episode['goals'])
                    for key, value in values.items():
                        if value is None or isinstance(value, (bool, int)):
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
                        expected_n = min(8, 70 - t) if aware else 8
                        assert record['planned_control_steps'] == expected_n
                        assert record['announced_remaining_steps'] == (70 - t if aware else None)
                        actions = np.array(record['selected_action_plan'])
                        assert len(actions) == expected_n
                        estimate, goal = episode['estimates'][t], episode['goals'][t]
                        cost, gradient = plan(actions, estimate, goal, model)
                        initial, _ = plan(warm[:expected_n], estimate, goal, model)
                        pg = np.max(np.abs(actions - np.clip(actions - gradient, -1., 1.)))
                        check('warm_cost', initial, record['initial_cost'])
                        check('selected_cost', cost, record['selected_cost'])
                        check('projected_gradient', pg, record['projected_gradient_max'])
                        check('action', actions[0], episode['commands'][t])
                        assert record['selected_cost'] <= record['initial_cost'] + 1e-8
                        warm = np.r_[actions[1:], 0.]
                        records += 1
                    if known:
                        values = endpoint(episode, sd)
                        for key, value in values.items():
                            if isinstance(value, bool):
                                assert value == episode['endpoint_diagnostic'][key]
                            else:
                                check('endpoint', value, episode['endpoint_diagnostic'][key])
                        # Exact known-parameter filtering error and fresh noise
                        # are independent of the issued, observation-adaptive
                        # input. Their realized endpoint residual must therefore
                        # be common across the three paired policy variants.
                        current_residual = values['realized_endpoint_error'] - values['conditional_mean_endpoint_error']
                        old_residual = old_endpoint['realized_endpoint_error'] - old_endpoint['conditional_mean_endpoint_error']
                        check('paired_endpoint_innovation', current_residual, old_residual)
                    count += 1
    for sd_label, summary in data['summary'].items():
        for name, values in summary.items():
            for metric in ['normalized_tracking_cost', 'late_position_rmse']:
                clusters = np.array([np.mean([task['variants'][name][metric] for task in row['cases'][sd_label]['tasks']]) for row in data['runs']])
                independent = dict(mean=np.mean(clusters), mc_se=np.std(clusters, ddof=1) / np.sqrt(12),
                    median=np.median(clusters), q10=np.quantile(clusters, .1), q90=np.quantile(clusters, .9))
                for key, value in independent.items():
                    check('summary', value, values[metric][key])
            episodes = [task['variants'][name] for row in data['runs'] for task in row['cases'][sd_label]['tasks']]
            assert values['stop_successes'] == sum(episode['stop_success'] for episode in episodes)
            assert values['task_trials'] == 24
            assert values['optimizer_nonconvergences'] == sum(not record['success'] for episode in episodes for record in episode['optimizers'] if record)
            if name.startswith('known_'):
                b2 = np.mean([endpoint(episode, float(sd_label))['conditional_mean_endpoint_error']**2 for episode in episodes])
                check('summary', b2, values['mean_squared_conditional_endpoint_bias'])
    assert max(errors.values()) < 1e-7
    return dict(status='INDEPENDENTLY TESTED', episodes_replayed=count, variable_horizon_plans_checked=records,
        identical_policy_prefixes_checked=preserved_prefixes, maximum_absolute_errors=errors,
        scope=[
            'Both added variants have16 predicted coast transitions; only the announced remaining-action horizon differs.',
            'Commands, physical/filter states and observations match exactly until deadline-aware horizon first shrinks at action63.',
            'Original12 versus fixed16 is a separate coast-length comparison; fixed16 versus aware16 isolates deadline metadata.',
            'Endpoint conditional means/variances are applied only to the exact known-parameter Kalman filter.',
            'Known-parameter residual endpoint innovations are identical across paired policies; control changes conditional means, not the information variance.',
            'Mean-centering last action is a counterfactual with earlier history fixed; feasibility is not whole-stop success or global policy optimality.',
            'Announced cutoff is supplied task metadata, not improved physical sensing, additional training or autonomous discovery.',
            'Twelve fit/noise seed clusters determine MC SE after averaging the two tasks.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
