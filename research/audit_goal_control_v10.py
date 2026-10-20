#!/usr/bin/env python3
"""Independent recurrence, goal-control timing, filtering and metric audit.

Audits saved trajectories by eliminating motor state. Structural tests run even
before the canonical JSON is complete; final-artifact checks activate afterward.
Canonical controller calls in the symmetry check are classified as paired
implementation tests, separate from the independent filter/recurrence proof.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
PILOT = '--pilot' in sys.argv
CANONICAL_STEM = 'goal_control_v10_lbfgs_pilot' if PILOT else 'goal_control_v10'
spec = importlib.util.spec_from_file_location('canonical_v10', ROOT / (CANONICAL_STEM + '.py'))
canonical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical)


def recurrence(theta, commands, state=(0., 0., 0.), noises=None):
    rho, lam, c, beta = theta
    n = len(commands)
    def f(value):
        return rho * value - c * np.tanh(value)
    v = np.zeros(n + 2)
    v[0], v[1] = state[1], f(state[1]) + state[2]
    if noises is None:
        noises = np.zeros(n)
    for t, u in enumerate(commands):
        v[t + 2] = f(v[t + 1]) + lam * (v[t + 1] - f(v[t])) + beta * u + noises[t]
    x = state[0] + np.cumsum(np.r_[0., v[1:n + 1]])
    motor = v[1:] - f(v[:-1])
    return np.c_[x, v[:n + 1], motor]


def independent_cost(actions, state, goal, theta):
    commands = np.r_[actions, np.zeros(canonical.PLAN_COAST)]
    states = recurrence(theta, commands, state)[1:]
    return float(np.sum((states[:, 0] - goal)**2 + .3 * states[:, 1]**2 + .03 * commands**2))


def independent_cost_gradient(actions, state, goal, theta):
    rho, lam, c, beta = theta
    n = len(actions)
    total = n + 12
    v, derivative = np.zeros(total + 2), np.zeros((total + 2, n))
    def passive(value):
        return rho * value - c * np.tanh(value)
    def slope(value):
        return rho - c * (1 - np.tanh(value)**2)
    v[0], v[1] = state[1], passive(state[1]) + state[2]
    x, dx, cost, gradient = state[0], np.zeros(n), 0., np.zeros(n)
    for t in range(total):
        u = actions[t] if t < n else 0.
        v[t + 2] = passive(v[t + 1]) + lam * (v[t + 1] - passive(v[t])) + beta * u
        derivative[t + 2] = ((slope(v[t + 1]) + lam) * derivative[t + 1]
                            - lam * slope(v[t]) * derivative[t])
        if t < n:
            derivative[t + 2, t] += beta
        x += v[t + 1]
        dx += derivative[t + 1]
        cost += (x - goal)**2 + .3 * v[t + 1]**2 + .03 * u**2
        gradient += 2 * (x - goal) * dx + .6 * v[t + 1] * derivative[t + 1]
        if t < n:
            gradient[t] += .06 * u
    return float(cost), gradient


def independent_linear_optimum(state, goal, theta):
    # Strictly convex bounded quadratic MPC, solved as bounded linear LS.
    def output(actions):
        states = recurrence(theta, np.r_[actions, np.zeros(12)], state)[1:]
        return np.r_[states[:, 0] - goal, np.sqrt(.3) * states[:, 1], np.sqrt(.03) * actions]
    zero = np.zeros(8)
    baseline = output(zero)
    matrix = np.column_stack([output(np.eye(8)[k]) - baseline for k in range(8)])
    fit = lsq_linear(matrix, -baseline, bounds=(-1., 1.), method='bvls', tol=1e-12, max_iter=1000)
    assert fit.success
    return fit.x


def independent_metrics(states, commands, goals):
    states = np.asarray(states)[1:]
    commands, goals = np.asarray(commands), np.asarray(goals)
    errors = states[:, 0] - goals
    stopped = (np.abs(errors) <= .3) & (np.abs(states[:, 1]) <= .15)
    settle = [t for t in range(28, 70) if all(stopped[t:])]
    return dict(normalized_tracking_cost=float(np.sum(errors[:70]**2 + .3 * states[:70, 1]**2 + .03 * commands[:70]**2) / (70 + np.sum(goals[:70]**2))),
        stop_success=bool(all(stopped[65:])),
        late_position_rmse=float(np.sqrt(np.mean(errors[60:]**2))),
        late_velocity_rms=float(np.sqrt(np.mean(states[60:, 1]**2))),
        coast_position_max_error=float(np.max(np.abs(errors[70:]))),
        coast_velocity_max=float(np.max(np.abs(states[70:, 1]))),
        mean_action_energy=float(np.mean(commands[:70]**2)),
        settled_after_final_goal_steps=settle[0] - 28 if settle else None)


def independent_pd(theta, gains, goals, seed):
    rng = np.random.default_rng(seed)
    sensor, noise = rng.normal(0, .2, 87), rng.normal(0, .03, 86)
    noise[70:] = 0.
    targets = np.array([goals[0] if t < 6 else goals[1] if t < 28 else goals[2] for t in range(86)])
    rho, lam, c, beta = theta
    def f(value):
        return rho * value - c * np.tanh(value)
    v = np.zeros(88)
    commands = []
    states = [[0., 0., 0.]]
    x = sx = sv = 0.
    kp, kd, alpha = gains
    for t in range(86):
        old_sx = sx
        sx += alpha * (x + sensor[t] - sx)
        sv += alpha * ((sx - old_sx) - sv)
        u = float(np.clip(kp * (targets[t] - sx) - kd * sv, -1., 1.)) if t < 70 else 0.
        commands.append(u)
        v[t + 2] = f(v[t + 1]) + lam * (v[t + 1] - f(v[t])) + beta * u + noise[t]
        x += v[t + 1]
        states.append([x, v[t + 1], v[t + 2] - f(v[t + 1])])
    return independent_metrics(states, commands, targets)


def filter_replay(model, episode, method):
    estimates = []
    mean, cov = np.zeros(3), np.zeros((3, 3))
    rho, lam, c, beta = model
    for t, u in enumerate(episode['commands']):
        k = cov[:, 0] / (cov[0, 0] + .2**2)
        mean += k * (episode['observations'][t] - mean[0])
        cov -= np.outer(k, cov[0])
        estimates.append(mean.copy())
        x, v, m = mean
        a = rho - c * (1 - np.tanh(v)**2)
        next_v = rho * v - c * np.tanh(v) + m
        mean = np.array([x + next_v, next_v, lam * m + beta * u])
        matrix = np.array([[1., a, 1.], [0., a, 1.], [0., 0., lam]])
        cov = matrix @ cov @ matrix.T
        if t < 70:
            cov[2, 2] += .03**2
    return np.array(estimates)


def structural_checks():
    rng = np.random.default_rng(99710)
    max_cost_error = max_grad_error = max_scaled_grad_error = max_immediate_action_effect = 0.
    for theta in [np.array([.65, .2, 0., 1.2]), np.array([.65, .7, .2, 1.2]), np.array([.94, .94, .39, 1.9])]:
        state = rng.normal(0, .5, 3)
        actions = rng.uniform(-.8, .8, 8)
        goal = .8
        cost, gradient = canonical.plan_cost(actions, state, goal, theta)
        max_cost_error = max(max_cost_error, abs(cost - independent_cost(actions, state, goal, theta)))
        numeric = np.empty(8)
        for k in range(8):
            step = np.zeros(8)
            step[k] = 1e-6
            numeric[k] = (independent_cost(actions + step, state, goal, theta)
                          - independent_cost(actions - step, state, goal, theta)) / 2e-6
        max_grad_error = max(max_grad_error, float(np.max(np.abs(numeric - gradient))))
        max_scaled_grad_error = max(max_scaled_grad_error, float(np.max(np.abs(numeric - gradient) / np.maximum(1., np.abs(gradient)))))
        first = recurrence(theta, [-1.], state)[1]
        second = recurrence(theta, [1.], state)[1]
        max_immediate_action_effect = max(max_immediate_action_effect, float(np.max(np.abs(first[:2] - second[:2]))))
    assert max_cost_error < 1e-8 and max_scaled_grad_error < 1e-7 and max_immediate_action_effect == 0.

    # Exact similarity transform for exchanging linear mechanical/motor poles.
    rho, lam, beta = .65, .7, 1.2
    T = np.eye(3)
    T[2, 1] = rho - lam
    A = np.array([[1., rho, 1.], [0., rho, 1.], [0., 0., lam]])
    B = np.array([[1., lam, 1.], [0., lam, 1.], [0., 0., rho]])
    b = np.array([0., 0., beta])
    H = np.array([1., 0., 0.])
    Q = np.diag([0., 0., .03**2])
    similarity_error = max(np.max(np.abs(T @ A - B @ T)), np.max(np.abs(T @ b - b)),
                           np.max(np.abs(H @ T - H)), np.max(np.abs(T @ Q @ T.T - Q)))
    assert similarity_error < 1e-14
    means = [np.zeros(3), np.zeros(3)]
    covs = [np.zeros((3, 3)), np.zeros((3, 3))]
    filter_error = 0.
    for t in range(86):
        observation, command = rng.normal(), rng.uniform(-1., 1.)
        for i, matrix in enumerate([A, B]):
            gain = covs[i] @ H / (H @ covs[i] @ H + .2**2)
            means[i] += gain * (observation - H @ means[i])
            # Independent covariance formula, not the canonical Joseph update.
            covs[i] -= np.outer(gain, H @ covs[i])
            means[i] = matrix @ means[i] + b * command
            covs[i] = matrix @ covs[i] @ matrix.T + (Q if t < 70 else 0.)
        filter_error = max(filter_error, float(np.max(np.abs(means[1] - T @ means[0]))),
                           float(np.max(np.abs(covs[1] - T @ covs[0] @ T.T))))
    assert filter_error < 1e-11
    state = np.array([.5, -.4, .2])
    other_state = T @ state
    actions = np.linspace(-.7, .7, 8)
    cost_a, grad_a = canonical.plan_cost(actions, state, 2., (rho, lam, 0., beta))
    cost_b, grad_b = canonical.plan_cost(actions, other_state, 2., (lam, rho, 0., beta))
    objective_symmetry_error = max(abs(cost_a - cost_b), float(np.max(np.abs(grad_a - grad_b))))
    assert objective_symmetry_error < 1e-10
    optimum_a = independent_linear_optimum(state, 2., (rho, lam, 0., beta))
    optimum_b = independent_linear_optimum(other_state, 2., (lam, rho, 0., beta))
    exact_solver_action_error = float(np.max(np.abs(optimum_a - optimum_b)))
    assert exact_solver_action_error < 1e-10
    # Paired canonical closed-loop implementation check, following the proof
    # above. This does not substitute for the independent similarity argument.
    first = canonical.run_episode((rho, lam, 0., beta), (rho, lam, 0., beta),
        'linear_MPC', (2., -2., 1.), 99510)
    second = canonical.run_episode((rho, lam, 0., beta), (lam, rho, 0., beta),
        'linear_MPC', (2., -2., 1.), 99510)
    symmetry_action_error = float(np.max(np.abs(np.array(first['commands']) - second['commands'])))
    symmetry_state_error = float(np.max(np.abs(np.array(first['states']) - second['states'])))
    # Finite-iteration L-BFGS-B can take different paths under roundoff, so do
    # not confuse exact objective/optimal-action invariance with literal
    # implemented-action identity. Its measured discrepancy is retained below.
    if not PILOT:
        assert symmetry_action_error < 1e-10 and symmetry_state_error < 1e-10
    return dict(independent_plan_cost_max_error=max_cost_error,
        independent_action_gradient_max_error=max_grad_error,
        independent_action_gradient_max_scaled_error=max_scaled_grad_error,
        immediate_x_v_action_effect=max_immediate_action_effect,
        linear_similarity_identity_error=float(similarity_error),
        independent_swapped_filter_max_error=filter_error,
        linear_plan_cost_gradient_swap_error=objective_symmetry_error,
        independent_bounded_linear_LS_optimum_swap_error=exact_solver_action_error,
        paired_canonical_swapped_MPC_action_error=symmetry_action_error,
        paired_canonical_swapped_MPC_state_error=symmetry_state_error,
        paired_canonical_primary_cost_difference=abs(first['normalized_tracking_cost'] - second['normalized_tracking_cost']),
        finite_solver_scope='Exact costs/gradients/unique convex optima are invariant. Pilot truncated L-BFGS-B actions differ under numerical optimization; final TRF paired actions agree at numerical precision.')


def final_checks(data):
    errors = dict(plant_replay=0., observations=0., estimates=0., metrics=0., summary=0., training=0., prediction=0., pd_tuning=0.,
                  fit_objective=0., MPC_initial_cost=0., MPC_selected_cost=0., MPC_projected_gradient=0., MPC_action=0.)
    def check(key, a, b):
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        errors[key] = max(errors[key], err)
    episodes = 0
    optimizer_records = 0
    optimizer_max_pg = 0.
    optimizer_fallbacks = 0
    for plant in data['plants']:
        theta = plant['true_theta']
        for run in plant['runs']:
            # Identical training data logged for all learned methods. Rebuild
            # command durations/amplitudes and positions without motor state.
            rng = np.random.default_rng(run['seed'])
            commands = []
            while len(commands) < 512:
                length = int(rng.choice((1, 3, 8)))
                commands.extend([float(rng.uniform(-1., 1.))] * length)
            commands = np.array(commands[:512])
            train_states = recurrence(theta, commands)
            observations = train_states[:, 0] + rng.normal(0, .2, 513)
            check('training', commands, run['training_commands'])
            check('training', observations, run['training_positions'])
            for model_name, records_name, selected_name in [('nonlinear', 'nonlinear_optimization', 'nonlinear_theta'), ('linear', 'linear_optimization', 'linear_theta')]:
                records = run[records_name]
                for fit in records:
                    fit_states = recurrence(fit['theta'], commands)
                    check('fit_objective', np.sum((fit_states[:, 0] - observations)**2), fit['sse'])
                assert min(records, key=lambda fit: fit['sse'])['theta'] == run[selected_name]
            for name, model in [('linear', run['linear_theta']), ('nonlinear', run['nonlinear_theta'])]:
                rngp = np.random.default_rng(run['seed'] + 3000000)
                probe = rngp.uniform(-1., 1., 256)
                true_states, model_states = recurrence(theta, probe), recurrence(model, probe)
                for metric, column in [('position_rmse', 0), ('velocity_rmse', 1)]:
                    score = np.sqrt(np.mean((true_states[:, column] - model_states[:, column])**2))
                    check('prediction', score, run['clean_prediction'][name][metric])
            pd_scores = []
            for gains in canonical.PD_GRID:
                scores = [independent_pd(run['nonlinear_theta'], gains, goals, run['seed'] + 1000000 + j)['normalized_tracking_cost']
                    for j, goals in enumerate([(1., -1., .5), (-1., 1., -.5)])]
                pd_scores.append(float(np.mean(scores)))
            check('pd_tuning', pd_scores, run['PD_grid_scores'])
            assert list(canonical.PD_GRID[int(np.argmin(pd_scores))]) == run['selected_PD']
            for task in run['tasks']:
                for name, episode in task['controllers'].items():
                    rngtest = np.random.default_rng(episode['seed'])
                    sensor = rngtest.normal(0, .2, 87)
                    noise = rngtest.normal(0, .03, 86)
                    noise[70:] = 0.
                    states = recurrence(theta, episode['commands'], noises=noise)
                    check('plant_replay', states, episode['states'])
                    check('observations', states[:, 0] + sensor, episode['observations'])
                    if name in ['nominal_MPC', 'linear_MPC', 'nonlinear_MPC', 'oracle_model_EKF']:
                        model = data['meta']['nominal_theta'] if name == 'nominal_MPC' else run['linear_theta'] if name == 'linear_MPC' else run['nonlinear_theta'] if name == 'nonlinear_MPC' else theta
                        check('estimates', filter_replay(model, episode, name), episode['estimates'])
                    elif name == 'oracle_state_MPC':
                        check('estimates', states[:-1], episode['estimates'])
                    else:
                        sx = sv = 0.
                        alpha = .25 if name == 'fixed_PD' else run['selected_PD'][2]
                        independent_estimates = []
                        for observation in episode['observations'][:-1]:
                            previous = sx
                            sx += alpha * (observation - sx)
                            sv += alpha * ((sx - previous) - sv)
                            independent_estimates.append([sx, sv, 0.])
                        check('estimates', independent_estimates, episode['estimates'])
                    calculated = independent_metrics(states, episode['commands'], episode['goals'])
                    assert np.max(np.abs(episode['commands'])) <= 1.
                    assert np.all(np.array(episode['commands'])[70:] == 0.)
                    for key, value in calculated.items():
                        if isinstance(value, bool) or value is None or isinstance(value, int):
                            assert value == episode[key]
                        else:
                            check('metrics', value, episode[key])
                    warm = np.zeros(8)
                    model = data['meta']['nominal_theta'] if name == 'nominal_MPC' else run['linear_theta'] if name == 'linear_MPC' else run['nonlinear_theta'] if name in ['nonlinear_MPC', 'model_tuned_PD', 'fixed_PD'] else theta
                    for t, record in enumerate(episode['optimizers']):
                        if record is None:
                            continue
                        assert record['selected_cost'] <= record['initial_cost'] + 1e-8
                        assert record['projected_gradient_max'] >= 0.
                        optimizer_records += 1
                        optimizer_max_pg = max(optimizer_max_pg, record['projected_gradient_max'])
                        optimizer_fallbacks += not record['used_optimizer_solution']
                        if 'selected_action_plan' in record:
                            actions = np.array(record['selected_action_plan'])
                            state, goal = episode['estimates'][t], episode['goals'][t]
                            selected_cost, gradient = independent_cost_gradient(actions, state, goal, model)
                            initial_cost = independent_cost(warm, state, goal, model)
                            pg = np.max(np.abs(actions - np.clip(actions - gradient, -1., 1.)))
                            check('MPC_initial_cost', initial_cost, record['initial_cost'])
                            check('MPC_selected_cost', selected_cost, record['selected_cost'])
                            check('MPC_projected_gradient', pg, record['projected_gradient_max'])
                            check('MPC_action', actions[0], episode['commands'][t])
                            warm = np.r_[actions[1:], 0.]
                    episodes += 1
        for name, summary in plant['summary'].items():
            for metric in ['normalized_tracking_cost', 'late_position_rmse', 'late_velocity_rms', 'mean_action_energy']:
                # Two goal schedules form a cluster within one training seed.
                clusters = np.array([np.mean([task['controllers'][name][metric] for task in run['tasks']]) for run in plant['runs']])
                values = dict(mean=np.mean(clusters), mc_se=np.std(clusters, ddof=1) / np.sqrt(12),
                              median=np.median(clusters), q10=np.quantile(clusters, .1), q90=np.quantile(clusters, .9))
                for key, value in values.items():
                    check('summary', value, summary[metric][key])
            success = sum(task['controllers'][name]['stop_success'] for run in plant['runs'] for task in run['tasks'])
            nonconverged = sum(not record['success'] for run in plant['runs'] for task in run['tasks']
                for record in task['controllers'][name]['optimizers'] if record is not None)
            assert summary['stop_successes'] == success and summary['task_trials'] == 24
            assert summary['optimizer_nonconvergences'] == nonconverged
    assert max(errors.values()) < 1e-8
    return dict(status='INDEPENDENTLY TESTED', episodes_replayed=episodes,
        optimizer_records_checked=optimizer_records, optimizer_max_projected_gradient=optimizer_max_pg,
        optimizer_fallback_count=optimizer_fallbacks, maximum_absolute_errors=errors,
        optimizer_scope=('Final saved plans allow independent selected-cost, projected-gradient, initial-warm-cost and first-action reconstruction at every MPC step.'
                         if not PILOT else 'Pilot records verify improvement/failure accounting; full pilot plans are absent, so every selected cost/projected gradient is not independently reconstructed.'))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        out = dict(structural=structural_checks())
        path = ROOT / (CANONICAL_STEM + '_results.json')
        if path.exists() and path.stat().st_size:
            out['final_artifact'] = final_checks(json.loads(path.read_text()))
        else:
            out['final_artifact'] = dict(status='PENDING canonical result completion')
    print(json.dumps(out, indent=2, allow_nan=False))
