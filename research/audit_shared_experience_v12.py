#!/usr/bin/env python3
"""Independent corrected shared-data and same-family model-reuse audit.

No canonical experiment/controller module is imported. Batch physical simulation
and parameter/action derivatives eliminate the hidden motor. This audits saved
artifacts and all RNG streams, not global optimality or universal method quality.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
BASE = 22161020
N = 86
TRAIN_GOALS = [(1., -1., .5), (-1., 1., -.5)]
TEST_GOALS = [(2., -2., 1.), (6., -4., 3.)]
LOWER = np.array([.05, .05, 0., .3])
UPPER = np.array([.95, .95, .4, 2.])


def targets(values):
    return np.array([values[0] if t < 6 else values[1] if t < 28 else values[2] for t in range(N)])


def simulate(theta, inputs, noise=None, derivatives=False):
    """Eliminated recurrence, independently reset for each row of inputs."""
    rho, lam, c, beta = theta
    u = np.atleast_2d(inputs)
    batch, n = u.shape
    v = np.zeros((batch, n + 2))
    x = np.zeros((batch, n + 1))
    dv = np.zeros((batch, n + 2, 4)) if derivatives else None
    J = np.zeros((batch, n + 1, 4)) if derivatives else None
    def passive(z):
        return rho * z - c * np.tanh(z)
    for t in range(n):
        a, b = v[:, t + 1], v[:, t]
        v[:, t + 2] = passive(a) + lam * (a - passive(b)) + beta * u[:, t]
        if noise is not None:
            v[:, t + 2] += np.atleast_2d(noise)[:, t]
        x[:, t + 1] = x[:, t] + a
        if derivatives:
            slope_a = rho - c * (1 - np.tanh(a)**2)
            slope_b = rho - c * (1 - np.tanh(b)**2)
            direct = np.c_[a - lam * b, a - passive(b), -np.tanh(a) + lam * np.tanh(b), u[:, t]]
            dv[:, t + 2] = (slope_a + lam)[:, None] * dv[:, t + 1] - (lam * slope_b)[:, None] * dv[:, t] + direct
            J[:, t + 1] = J[:, t] + dv[:, t + 1]
    motor = v[:, 1:] - passive(v[:, :-1])
    states = np.stack([x, v[:, :n + 1], motor], axis=-1)
    return states, J


def decode(p):
    p = np.asarray(p)
    return np.c_[np.exp(p[:, 0]), np.exp(p[:, 1]), 1 / (1 + np.exp(-p[:, 2])), p[:, 3:]]


def direct_actions(coordinates, observations, goals):
    gains = decode(np.atleast_2d(coordinates))
    kp, kd, alpha, bv, bu1, bu2 = gains.T
    o = np.atleast_2d(observations)
    g = np.atleast_2d(goals)
    batch = len(o)
    sx = np.zeros(batch)
    sv = np.zeros(batch)
    previous = np.zeros(batch)
    older = np.zeros(batch)
    commands = np.zeros((batch, N))
    for t in range(N):
        old_x, old_v = sx.copy(), sv.copy()
        sx = sx + alpha * (o[:, t] - sx)
        sv = sv + alpha * ((sx - old_x) - sv)
        commands[:, t] = np.clip(kp * (g[:, t] - sx) - kd * sv + bv * old_v - bu1 * previous - bu2 * older, -1., 1.)
        older, previous = previous, commands[:, t]
    return commands


def metrics(states, observations, actions, goals):
    state = np.asarray(states)[1:]
    o, u, g = np.asarray(observations), np.asarray(actions), np.asarray(goals)
    scale = 1 + np.mean(g**2)
    actual = np.mean((state[:, 0] - g)**2 + .3 * state[:, 1]**2 + .03 * u**2) / scale
    observed = np.mean((o[1:] - g)**2 + .3 * (o[1:] - o[:-1])**2 + .03 * u**2) / scale
    return dict(actual_cost=float(actual), observed_cost=float(observed), gap_minus_offset=float(observed - actual - 1.6 * .2**2 / scale),
        late_position_rmse=float(np.sqrt(np.mean((state[-10:, 0] - g[-10:])**2))),
        late_velocity_rms=float(np.sqrt(np.mean(state[-10:, 1]**2))),
        stationary_success=bool(np.all(np.abs(state[-10:, 0] - g[-10:]) <= .3) and np.all(np.abs(state[-10:, 1]) <= .15)))


def filter_history(theta, observations, inputs):
    rho, lam, c, beta = theta
    mean, P = np.zeros(3), np.zeros((3, 3))
    estimates = []
    for t, u in enumerate(inputs):
        K = P[:, 0] / (P[0, 0] + .2**2)
        mean += K * (observations[t] - mean[0])
        P -= np.outer(K, P[0])
        estimates.append(mean.copy())
        x, v, m = mean
        a = rho - c * (1 - np.tanh(v)**2)
        vn = rho * v - c * np.tanh(v) + m
        mean = np.array([x + vn, vn, lam * m + beta * u])
        A = np.array([[1., a, 1.], [0., a, 1.], [0., 0., lam]])
        P = A @ P @ A.T
        P[2, 2] += .03**2
    return np.array(estimates)


def action_objective(actions, state, goal, theta):
    rho, lam, c, beta = theta
    n = len(actions)
    v, dv = np.zeros(n + 2), np.zeros((n + 2, n))
    def f(z):
        return rho * z - c * np.tanh(z)
    def slope(z):
        return rho - c * (1 - np.tanh(z)**2)
    v[0], v[1] = state[1], f(state[1]) + state[2]
    x, dx, cost, gradient = state[0], np.zeros(n), 0., np.zeros(n)
    for t, u in enumerate(actions):
        v[t + 2] = f(v[t + 1]) + lam * (v[t + 1] - f(v[t])) + beta * u
        dv[t + 2] = (slope(v[t + 1]) + lam) * dv[t + 1] - lam * slope(v[t]) * dv[t]
        dv[t + 2, t] += beta
        x += v[t + 1]
        dx += dv[t + 1]
        cost += (x - goal)**2 + .3 * v[t + 1]**2 + .03 * u**2
        gradient += 2 * (x - goal) * dx + .6 * v[t + 1] * dv[t + 1]
        gradient[t] += .06 * u
    return float(cost), gradient


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), mc_se=float(a.std(ddof=1) / np.sqrt(len(a))), median=float(np.median(a)),
                q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9)))


def main():
    shared_file = ROOT / 'shared_experience_v12_results.json'
    data_file = ROOT / 'shared_experience_v12_data.npz'
    shared = json.loads(shared_file.read_text())
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == shared['meta']['data_sha256']
    archive = np.load(data_file)
    maximum = dict(training_states=0., training_sensor=0., training_actions=0., CEM_coordinates=0., CEM_fitness=0.,
        parameter_Jacobian_finite_difference=0., fit_SSE=0., fit_singular_values=0., fit_optimality=0.,
        test_states=0., test_sensor=0., test_actions=0., filter_estimates=0., metrics=0.,
        MPC_warm_cost=0., MPC_selected_cost=0., MPC_projected_gradient=0., summary=0., reuse_prefix=0.)
    scaled = dict.fromkeys(maximum, 0.)
    def check(key, a, b):
        a, b = np.asarray(a), np.asarray(b)
        diff = np.abs(a - b)
        maximum[key] = max(maximum[key], float(np.max(diff)))
        scaled[key] = max(scaled[key], float(np.max(diff / np.maximum(1., np.maximum(np.abs(a), np.abs(b))))))
    roles = {}
    def register(seed, role):
        assert seed not in roles, (seed, role, roles.get(seed))
        roles[seed] = role

    # This test is independent of canonical sensitivities: differentiate the
    # eliminated recurrence and compare its analytic result to finite differences.
    fixture = np.random.default_rng(8317).uniform(-1, 1, (3, 12))
    for theta in [[.65, .2, 0., 1.2], [.65, .7, .2, 1.2], [.94, .94, .39, 1.9]]:
        p = np.array(theta)
        _, J = simulate(p, fixture, derivatives=True)
        for k in range(4):
            delta = np.zeros(4)
            delta[k] = 1e-6
            numerical = (simulate(p + delta, fixture)[0][:, :, 0] - simulate(p - delta, fixture)[0][:, :, 0]) / 2e-6
            check('parameter_Jacobian_finite_difference', numerical, J[:, :, k])

    counts = dict(physical_training_episodes=0, virtual_training_episodes=0, shared_test_episodes=0,
                  reused_test_episodes=0, fit_records=0, MPC_plans=0, CEM_generations=0)
    training_targets = np.tile(np.array([targets(g) for g in TRAIN_GOALS]), (192, 1))

    def audit_training(theta, seed, records, arrays, label):
        register(seed, label + ':CEM')
        u, observations, stored = arrays['commands'], arrays['observations'], arrays['states']
        assert u.shape == (384, 86) and observations.shape == (384, 87) and stored.shape == (384, 87, 3)
        assert np.all(stored[:, 0] == 0.)
        states, _ = simulate(theta, u)
        check('training_states', states, stored)
        rng = np.random.default_rng(seed)
        mean = np.array([math.log(.08), math.log(.6), math.log(.6 / .4), 0., 0., 0.])
        sd = np.array([1., 1., 1., .3, .3, .3])
        lower = np.array([math.log(.005), math.log(.02), math.log(.1 / .9), -2., -2., -2.])
        upper = np.array([math.log(.5), math.log(4.), math.log(.98 / .02), 2., 2., 2.])
        floors = np.array([.15, .15, .15, .03, .03, .03])
        previous = None
        all_coordinates = []
        checkpoints = {}
        for generation in range(8):
            coordinates = np.clip(rng.normal(mean, sd, (24, 6)), lower, upper)
            coordinates[0] = mean if previous is None else previous
            if generation == 0:
                coordinates[1] = [math.log(.02), 0., math.log(.8 / .2), 0., 0., 0.]
            record = records[generation]
            assert record['generation'] == generation
            check('CEM_coordinates', coordinates, record['coordinates'])
            observed_fitness = []
            for j in range(24):
                fitness = []
                for k, goals in enumerate(TRAIN_GOALS):
                    index = 2 * (generation * 24 + j) + k
                    stream = seed + 100000 + 100 * (24 * generation + j) + k
                    register(stream, label + ':training')
                    sensor = np.random.default_rng(stream).normal(0, .2, 87)
                    check('training_sensor', states[index, :, 0] + sensor, observations[index])
                    goal = targets(goals)
                    fitness.append(np.mean((observations[index, 1:] - goal)**2 + .3 * np.diff(observations[index])**2 + .03 * u[index]**2) / (1 + np.mean(goal**2)))
                observed_fitness.append(float(np.mean(fitness)))
            check('CEM_fitness', observed_fitness, record['observed_fitness'])
            elites = np.argsort(observed_fitness, kind='stable')[:6]
            winner = int(np.argmin(observed_fitness))
            assert record['elite'] == elites.tolist() and record['winner_index'] == winner
            previous = coordinates[winner].copy()
            mean = np.mean(coordinates[elites], axis=0)
            sd = np.maximum(np.std(coordinates[elites], axis=0), floors)
            all_coordinates.extend(np.repeat(coordinates, 2, axis=0))
            if generation + 1 in [2, 8]:
                checkpoints[generation + 1] = previous.copy()
            counts['CEM_generations'] += 1
        commands = direct_actions(np.array(all_coordinates), observations, training_targets)
        check('training_actions', commands, u)
        counts['virtual_training_episodes' if label.startswith('virtual') else 'physical_training_episodes'] += 384
        return checkpoints

    def audit_episode(theta, episode, goal_values, coordinates=None, model=None):
        rng = np.random.default_rng(episode['seed'])
        sensor, noise = rng.normal(0, .2, 87), rng.normal(0, .03, 86)
        commands = np.array(episode['commands'])
        assert commands.shape == (86,) and np.max(np.abs(commands)) <= 1.
        states = simulate(theta, commands, noise)[0][0]
        check('test_states', states, episode['states'])
        check('test_sensor', states[:, 0] + sensor, episode['observations'])
        goals = targets(goal_values)
        if coordinates is not None:
            predicted = direct_actions(np.atleast_2d(coordinates), np.atleast_2d(episode['observations']), goals[None])[0]
            check('test_actions', predicted, commands)
        if model is not None:
            estimates = filter_history(model, episode['observations'], commands)
            check('filter_estimates', estimates, episode['estimates'])
            warm = np.zeros(16)
            assert len(episode['plans']) == 86
            for t, record in enumerate(episode['plans']):
                actions = np.array(record['actions'])
                n = min(16, 86 - t)
                assert len(actions) == n and np.max(np.abs(actions)) <= 1.
                initial, _ = action_objective(warm[:n], estimates[t], goals[t], model)
                selected, gradient = action_objective(actions, estimates[t], goals[t], model)
                pg = np.max(np.abs(actions - np.clip(actions - gradient, -1., 1.)))
                check('MPC_warm_cost', initial, record['initial_cost'])
                check('MPC_selected_cost', selected, record['selected_cost'])
                check('MPC_projected_gradient', pg, record['projected_gradient_max'])
                check('test_actions', actions[0], commands[t])
                assert record['selected_cost'] <= record['initial_cost'] + 1e-8
                warm = np.r_[actions[1:], 0.]
                counts['MPC_plans'] += 1
        for key, value in metrics(states, episode['observations'], commands, goals).items():
            if isinstance(value, bool):
                assert episode[key] == value
            else:
                check('metrics', value, episode[key])

    for run in shared['runs']:
        pi, replicate = run['plant_index'], run['replicate']
        seed = BASE + 1000000000 * pi + 1000000 * replicate
        assert seed == run['seed']
        theta = [.65, [.2, .7][pi // 2], [0., .2][pi % 2], 1.2]
        assert theta == run['true_theta_evaluator_only']
        key = f'p{pi}_s{replicate}'
        arrays = {name: archive[key + '_' + name] for name in ['states', 'observations', 'commands']}
        checkpoints = audit_training(theta, seed, run['cem'], arrays, 'physical:' + key)
        for j in range(2):
            register(seed + 500000 + j, 'physical:' + key + ':heldout')
        for stage in run['checkpoints']:
            generations = stage['generations']
            segments = generations * 24 * 2
            assert stage['training_resets'] == segments and stage['training_steps'] == segments * 86
            check('CEM_coordinates', checkpoints[generations], stage['selected_coordinates'])
            for label, records in stage['fits'].items():
                columns = [0, 1, 3] if label == 'linear' else [0, 1, 2, 3]
                for record in records:
                    p = np.array(record['theta'])
                    assert np.all(p >= LOWER - 1e-12) and np.all(p <= UPPER + 1e-12)
                    fitted, jac = simulate(p, arrays['commands'][:segments], derivatives=True)
                    residual = (fitted[:, :, 0] - arrays['observations'][:segments]).reshape(-1)
                    matrix = jac[:, :, columns].reshape(-1, len(columns))
                    check('fit_SSE', np.sum(residual**2), record['sse'])
                    check('fit_singular_values', np.linalg.svd(matrix, compute_uv=False), record['singular_values'])
                    gradient = matrix.T @ residual
                    distances = np.where(gradient < 0., UPPER[columns] - p[columns], np.where(gradient > 0., p[columns] - LOWER[columns], 1.))
                    check('fit_optimality', np.max(np.abs(gradient * distances)), record['optimality'])
                    assert record['nfev'] <= 160
                    counts['fit_records'] += 1
                assert stage['models'][label] == min(records, key=lambda value: value['sse'])['theta']
            for j, task in enumerate(stage['tests']):
                assert task['goal'] == list(TEST_GOALS[j])
                for label, episode in task['controllers'].items():
                    assert episode['seed'] == seed + 500000 + j
                    audit_episode(theta, episode, TEST_GOALS[j],
                        coordinates=stage['selected_coordinates'] if label == 'direct' else None,
                        model=stage['models'][label] if label != 'direct' else None)
                    counts['shared_test_episodes'] += 1
    assert len(shared['runs']) == 32 and len(archive.files) == 96
    assert shared['meta']['train_steps_per_seed'] == 33024 and shared['meta']['train_resets_per_seed'] == 384
    assert shared['meta']['test_steps_per_seed'] == 1032
    for cell in shared['summary']:
        stages = [next(cp for cp in run['checkpoints'] if cp['generations'] == cell['generations'])
                  for run in shared['runs'] if run['plant_index'] == cell['plant']]
        for label, summary in cell['controllers'].items():
            means = [np.mean([task['controllers'][label]['actual_cost'] for task in stage['tests']]) for stage in stages]
            for key, value in stats(means).items():
                check('summary', value, summary['cost'][key])
            assert summary['stationary_successes'] == sum(task['controllers'][label]['stationary_success'] for stage in stages for task in stage['tests'])
            assert summary['trials'] == 16
        for label in ['linear', 'nonlinear']:
            paired = [np.mean([task['controllers']['direct']['actual_cost'] - task['controllers'][label]['actual_cost'] for task in stage['tests']]) for stage in stages]
            for key, value in stats(paired).items():
                check('summary', value, cell['paired_differences']['direct_minus_' + label][key])

    reuse_path = ROOT / 'model_reuse_v12_results.json'
    reuse_report = None
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text())
        virtual_path = ROOT / 'model_reuse_v12_data.npz'
        assert reuse['meta']['data_sha256'] == hashlib.sha256(virtual_path.read_bytes()).hexdigest()
        assert reuse['meta']['shared_result_sha256'] == hashlib.sha256(shared_file.read_bytes()).hexdigest()
        virtual = np.load(virtual_path)
        def prefix(episode, g):
            states = np.array(episode['states'])[1:71]
            u, goal = np.array(episode['commands'])[:70], targets(g)[:70]
            return float(np.mean((states[:, 0] - goal)**2 + .3 * states[:, 1]**2 + .03 * u**2) / (1 + np.mean(goal**2)))
        for row, previous in zip(reuse['runs'], shared['runs']):
            assert row['plant'] == previous['plant_index'] and row['replicate'] == previous['replicate']
            assert row['physical_seed'] == previous['seed'] and row['virtual_seed'] == previous['seed'] + 10000000000
            final = previous['checkpoints'][-1]
            assert row['learned_model'] == final['models']['nonlinear']
            key = f"p{row['plant']}_s{row['replicate']}"
            arrays = {name: virtual[key + '_' + name] for name in ['states', 'observations', 'commands']}
            checkpoints = audit_training(row['learned_model'], row['virtual_seed'], row['cem'], arrays, 'virtual:' + key)
            check('CEM_coordinates', checkpoints[8], row['selected_coordinates'])
            for j, task in enumerate(row['tests']):
                assert task['goal'] == list(TEST_GOALS[j])
                episode = task['reused_direct']
                assert episode['seed'] == previous['seed'] + 500000 + j
                audit_episode(previous['true_theta_evaluator_only'], episode, TEST_GOALS[j], coordinates=row['selected_coordinates'])
                for label, original in final['tests'][j]['controllers'].items():
                    check('reuse_prefix', prefix(original, TEST_GOALS[j]), task['prefix70_costs'][label])
                check('reuse_prefix', prefix(episode, TEST_GOALS[j]), task['prefix70_costs']['reused_direct'])
                counts['reused_test_episodes'] += 1
        for cell in reuse['summary']:
            rows = [row for row in reuse['runs'] if row['plant'] == cell['plant']]
            originals = [row['checkpoints'][-1] for row in shared['runs'] if row['plant_index'] == cell['plant']]
            means = [np.mean([task['reused_direct']['actual_cost'] for task in row['tests']]) for row in rows]
            paired = [np.mean([old['controllers']['direct']['actual_cost'] - new['reused_direct']['actual_cost'] for old, new in zip(original['tests'], row['tests'])]) for original, row in zip(originals, rows)]
            for key, value in stats(means).items():
                check('summary', value, cell['reused_cost'][key])
            for key, value in stats(paired).items():
                check('summary', value, cell['paired_original_direct_minus_reused'][key])
            for label, values in cell['prefix70_cost'].items():
                clusters = [np.mean([task['prefix70_costs'][label] for task in row['tests']]) for row in rows]
                for key, value in stats(clusters).items():
                    check('summary', value, values[key])
            assert cell['stationary_successes'] == sum(task['reused_direct']['stationary_success'] for row in rows for task in row['tests'])
            assert cell['trials'] == 16
        assert reuse['meta']['extra_physical_training_steps'] == 0 and reuse['meta']['virtual_steps_per_seed'] == 33024
        assert reuse['meta']['physical_test_steps'] == 5504 and reuse['meta']['physical_test_resets'] == 64
        reuse_report = dict(status='INDEPENDENTLY TESTED', virtual_streams_disjoint_from_physical=True,
            same_family_and_stationary_metadata=True, extra_physical_training=0, extra_virtual_steps_per_seed=33024,
            prefix70_scope='MPC first shortens its16-action horizon at t71, after the audited70-step prefix; prefix scores are secondary diagnostics.')
    assert maximum['parameter_Jacobian_finite_difference'] < 1e-5
    for key in maximum:
        if key != 'parameter_Jacobian_finite_difference':
            assert scaled[key] < 1e-7, (key, maximum[key], scaled[key])
    return dict(status='CORRECTED SHARED-EXPERIENCE ARTIFACTS INDEPENDENTLY VERIFIED',
        counts=counts, globally_distinct_CEM_training_test_role_seeds=len(roles),
        maximum_absolute_errors=maximum, maximum_errors_scaled_by_max1_magnitude=scaled,
        same_family_model_reuse=reuse_report,
        scope=[
            'No canonical experiment implementation imported; independent eliminated physics, parameter/action derivatives, filtering and RNG reconstruction.',
            'Original seed-overlap pilot excluded and retained separately; only corrected globallydisjoint streams audited for inference.',
            'All training episodes have independent supplied physical zero resets; checkpoint datasets are nested prefixes of the same complete physical experience.',
            'CEM currentgeneration winner and elites use only observed fitness, with fresh incumbent reevaluation; no stale global minimum.',
            'All linear/nonlinear fits use same raw commands/positions with correct segmented resets; model family/noise metadata remain privileged structural priors.',
            'MPC has clock/horizon and richer policy class; direct controller is restricted stationary6parameter family containingPD, not a universal model-free method.',
            'Samefamily model reuse controls policy expressivity and timefeatures but adds virtual computation and supplied predictive vocabulary.',
            'First70 costs remove horizon-shortening influence from that prefix without becoming a new retrospectively chosen primary score.',
            'All observations/state traces and numerical failures retained; no global nonlinear optimum or universal learning-method superiority follows.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
