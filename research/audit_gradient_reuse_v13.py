#!/usr/bin/env python3
"""Independent clipped-policy derivatives, validation and fresh-test v13 audit.

No canonical experiment module is imported. Physical dynamics and policy
sensitivities use an eliminated motor recurrence rather than the main code's
three-state sensitivity propagation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_shared_experience_v12 import simulate, targets, metrics, filter_history, action_objective, stats, direct_actions


ROOT = Path(__file__).resolve().parent
LOW = np.array([math.log(.005), math.log(.02), math.log(.1 / .9), -2., -2., -2.])
HIGH = np.array([math.log(.5), math.log(4.), math.log(.98 / .02), 2., 2., 2.])


def independent_roll(theta, p, sensor):
    rho, lam, c, beta = theta
    kp, kd, alpha = np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2]))
    bv, bu1, bu2 = p[3:]
    batch, n = len(sensor), 86
    goal = np.array([targets([(1., -1., .5), (-1., 1., -.5)][k % 2]) for k in range(batch)])
    scale = np.sqrt(batch * n * (1 + np.mean(goal**2, axis=1)))
    velocity, dvelocity = np.zeros((batch, n + 2)), np.zeros((batch, n + 2, 6))
    position, dposition = np.zeros((batch, n + 1)), np.zeros((batch, n + 1, 6))
    u, du, raw = np.zeros((batch, n)), np.zeros((batch, n, 6)), np.zeros((batch, n))
    sx, sv = np.zeros(batch), np.zeros(batch)
    dsx, dsv = np.zeros((batch, 6)), np.zeros((batch, 6))
    e = np.eye(6)
    da = alpha * (1 - alpha) * e[2]
    def f(v):
        return rho * v - c * np.tanh(v)
    def fp(v):
        return rho - c * (1 - np.tanh(v)**2)
    for t in range(n):
        old_sx, old_sv = sx.copy(), sv.copy()
        old_dsx, old_dsv = dsx.copy(), dsv.copy()
        sx = alpha * (position[:, t] + sensor[:, t]) + (1 - alpha) * old_sx
        dsx = alpha * dposition[:, t] + (1 - alpha) * old_dsx + (position[:, t] + sensor[:, t] - old_sx)[:, None] * da
        sv = alpha * (sx - old_sx) + (1 - alpha) * old_sv
        dsv = alpha * (dsx - old_dsx) + (1 - alpha) * old_dsv + (sx - old_sx - old_sv)[:, None] * da
        previous = u[:, t - 1] if t else np.zeros(batch)
        older = u[:, t - 2] if t >= 2 else np.zeros(batch)
        dprevious = du[:, t - 1] if t else np.zeros((batch, 6))
        dolder = du[:, t - 2] if t >= 2 else np.zeros((batch, 6))
        raw[:, t] = kp * (goal[:, t] - sx) - kd * sv + bv * old_sv - bu1 * previous - bu2 * older
        draw = kp * (goal[:, t] - sx)[:, None] * e[0] - kp * dsx
        draw -= kd * sv[:, None] * e[1] + kd * dsv
        draw += old_sv[:, None] * e[3] + bv * old_dsv - previous[:, None] * e[4] - bu1 * dprevious - older[:, None] * e[5] - bu2 * dolder
        u[:, t] = np.clip(raw[:, t], -1., 1.)
        du[:, t] = draw * (np.abs(raw[:, t]) < 1.)[:, None]
        current, prior = velocity[:, t + 1], velocity[:, t]
        velocity[:, t + 2] = f(current) + lam * (current - f(prior)) + beta * u[:, t]
        dvelocity[:, t + 2] = (fp(current) + lam)[:, None] * dvelocity[:, t + 1] - (lam * fp(prior))[:, None] * dvelocity[:, t] + beta * du[:, t]
        position[:, t + 1] = position[:, t] + current
        dposition[:, t + 1] = dposition[:, t] + dvelocity[:, t + 1]
    motor = velocity[:, 1:] - f(velocity[:, :-1])
    states = np.stack([position, velocity[:, :n + 1], motor], axis=-1)
    observations = position + sensor
    residual = np.stack([(position[:, 1:] - goal) / scale[:, None], math.sqrt(.3) * velocity[:, 1:n + 1] / scale[:, None], math.sqrt(.03) * u / scale[:, None]], axis=2)
    jacobian = np.stack([dposition[:, 1:] / scale[:, None, None], math.sqrt(.3) * dvelocity[:, 1:n + 1] / scale[:, None, None], math.sqrt(.03) * du / scale[:, None, None]], axis=2)
    return residual.transpose(1, 2, 0).reshape(-1), jacobian.transpose(1, 2, 0, 3).reshape(-1, 6), states, observations, u, raw


def derivative_fixture():
    sensor = np.random.default_rng(75071013).normal(0, .2, (16, 87))
    p = np.array([math.log(.08), math.log(.6), math.log(.6 / .4), .1, .2, -.05])
    r, J, _, _, actions, raw = independent_roll([.65, .7, .2, 1.2], p, sensor)
    rows = []
    for epsilon in [1e-6, 1e-7, 1e-8]:
        errors, changes, versus_base = [], [], []
        for k in range(6):
            step = np.eye(6)[k] * epsilon
            plus = independent_roll([.65, .7, .2, 1.2], p + step, sensor)
            minus = independent_roll([.65, .7, .2, 1.2], p - step, sensor)
            errors.append(float(np.max(np.abs((plus[0] - minus[0]) / (2 * epsilon) - J[:, k]))))
            changes.append(int(np.count_nonzero((np.abs(plus[4]) < 1) != (np.abs(minus[4]) < 1))))
            versus_base.append(int(np.count_nonzero((np.abs(plus[4]) < 1) != (np.abs(actions) < 1)) + np.count_nonzero((np.abs(minus[4]) < 1) != (np.abs(actions) < 1))))
        rows.append(dict(epsilon=epsilon, coordinate_errors=errors, clip_active_set_changes=changes, changes_relative_to_base=versus_base))
    for row in rows[1:]:
        assert sum(row['clip_active_set_changes']) == sum(row['changes_relative_to_base']) == 0
        assert max(row['coordinate_errors']) < 1e-5
    assert sum(rows[0]['clip_active_set_changes']) == 1
    return dict(finite_difference_steps=rows, minimum_distance_of_raw_action_to_clip_boundary=float(np.min(np.abs(np.abs(raw) - 1))),
        scope='The coarse step crosses a genuine clip kink; two fixed smaller branch-preserving steps verify the unchanged exact a.e. derivative.')


def main():
    data_path = ROOT / 'gradient_reuse_v13_results.json'
    data = json.loads(data_path.read_text())
    shared_path, reuse_path = ROOT / 'shared_experience_v12_results.json', ROOT / 'model_reuse_v12_results.json'
    assert data['meta']['shared_result_sha256'] == hashlib.sha256(shared_path.read_bytes()).hexdigest()
    assert data['meta']['reuse_result_sha256'] == hashlib.sha256(reuse_path.read_bytes()).hexdigest()
    npz_path = ROOT / 'gradient_reuse_v13_data.npz'
    assert data['meta']['data_sha256'] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
    shared, reuse = json.loads(shared_path.read_text()), json.loads(reuse_path.read_text())
    arrays = np.load(npz_path)
    maximum = dict(virtual_sensor_bank=0., objective=0., policy_projected_gradient=0., validation_score=0.,
        virtual_states=0., virtual_observations=0., virtual_actions=0., test_states=0., test_observations=0.,
        test_actions=0., estimates=0., metrics=0., MPC_warm=0., MPC_selected=0., MPC_projected_gradient=0., summary=0.)
    scaled = dict.fromkeys(maximum, 0.)
    def check(key, a, b):
        a, b = np.asarray(a), np.asarray(b)
        difference = np.abs(a - b)
        maximum[key] = max(maximum[key], float(np.max(difference)))
        scaled[key] = max(scaled[key], float(np.max(difference / np.maximum(1., np.maximum(np.abs(a), np.abs(b))))))
    seeds = set()
    for row in shared['runs']:
        for offset in [0, 500000, 500001]:
            seeds.add(row['seed'] + offset)
        for index in range(192):
            for k in range(2):
                seeds.add(row['seed'] + 100000 + 100 * index + k)
                seeds.add(row['seed'] + 10000000000 + 100000 + 100 * index + k)
        seeds.add(row['seed'] + 10000000000)
    def register(seed):
        assert seed not in seeds
        seeds.add(seed)
    counts = dict(optimizer_attempts=0, validation_candidates=0, retained_virtual_episodes=0, physical_test_episodes=0, MPC_plans=0)
    failures = []
    selected = [0] * 5
    tracking_components = {key: [] for key in ['physical_CEM', 'virtual_CEM', 'gradient_selected', 'nonlinear_MPC']}
    failure_causes = {key: dict(position_only=0, velocity_only=0, both=0, neither=0) for key in tracking_components}
    for row, original, virtual in zip(data['runs'], shared['runs'], reuse['runs']):
        assert row['seed'] == original['seed'] == virtual['physical_seed']
        assert row['plant'] == original['plant_index'] and row['replicate'] == original['replicate']
        final = original['checkpoints'][-1]
        model, truth = np.array(row['learned_model']), original['true_theta_evaluator_only']
        assert row['learned_model'] == final['models']['nonlinear']
        key = f"p{row['plant']}_s{row['replicate']}"
        banks = {}
        for label, offset in [('fit', 20000000000), ('validation', 30000000000)]:
            register(row['seed'] + offset)
            bank = np.random.default_rng(row['seed'] + offset).normal(0, .2, (16, 87))
            check('virtual_sensor_bank', bank, arrays[key + '_' + label + '_sensor_errors'])
            banks[label] = bank
        starts = [final['selected_coordinates'], virtual['selected_coordinates'], [math.log(.08), math.log(.6), math.log(.6 / .4), 0., 0., 0.]]
        candidates = []
        for index, (fit, start) in enumerate(zip(row['fits'], starts)):
            assert fit['start'] == start
            p = np.array(fit['coordinates'])
            assert np.all(p >= LOW - 1e-12) and np.all(p <= HIGH + 1e-12)
            residual, jacobian, *_ = independent_roll(model, p, banks['fit'])
            gradient = 2 * jacobian.T @ residual
            pg = np.max(np.abs(p - np.clip(p - gradient, LOW, HIGH)))
            check('objective', residual @ residual, fit['cost'])
            check('policy_projected_gradient', pg, fit['projected_gradient_max'])
            assert fit['nfev'] <= 160 and 0 < fit['njev'] <= fit['nfev']
            assert fit['virtual_steps'] == fit['unique_rollout_calls'] * 16 * 86
            assert fit['nfev'] <= fit['unique_rollout_calls'] <= fit['nfev'] + fit['njev'] + 1
            if not fit['success']:
                failures.append(dict(plant=row['plant'], replicate=row['replicate'], start_index=index,
                    selected=index == row['selected_index'], status=fit['status'], nfev=fit['nfev'], projected_gradient_max=fit['projected_gradient_max']))
            candidates.append(p)
            counts['optimizer_attempts'] += 1
        candidates.extend([np.array(starts[0]), np.array(starts[1])])
        validation = []
        for candidate in candidates:
            residual, *_ = independent_roll(model, candidate, banks['validation'])
            validation.append(float(residual @ residual))
            counts['validation_candidates'] += 1
        check('validation_score', validation, row['virtual_validation_scores'])
        winner = int(np.argmin(validation))
        assert winner == row['selected_index'] and row['selected_coordinates'] == candidates[winner].tolist()
        assert validation[winner] <= min(validation[3:]) + 1e-12
        selected[winner] += 1
        for label in ['fit', 'validation']:
            _, _, states, observations, commands, _ = independent_roll(model, candidates[winner], banks[label])
            check('virtual_states', states, arrays[key + '_' + label + '_states'])
            check('virtual_observations', observations, arrays[key + '_' + label + '_observations'])
            check('virtual_actions', commands, arrays[key + '_' + label + '_commands'])
            counts['retained_virtual_episodes'] += 16
        for j, task in enumerate(row['tests']):
            goals = [(2., -2., 1.), (6., -4., 3.)][j]
            assert task['goal'] == list(goals)
            stream = row['seed'] + 40000000000 + j
            register(stream)
            sensor_rng = np.random.default_rng(stream)
            sensor, motor = sensor_rng.normal(0, .2, 87), sensor_rng.normal(0, .03, 86)
            g = targets(goals)
            for label, episode in task['controllers'].items():
                assert episode['seed'] == stream
                inputs = np.array(episode['commands'])
                states = simulate(truth, inputs, motor)[0][0]
                check('test_states', states, episode['states'])
                check('test_observations', states[:, 0] + sensor, episode['observations'])
                if label != 'nonlinear_MPC':
                    p = starts[0] if label == 'physical_CEM' else starts[1] if label == 'virtual_CEM' else candidates[winner]
                    issued = direct_actions(np.atleast_2d(p), np.atleast_2d(episode['observations']), g[None])[0]
                    check('test_actions', issued, inputs)
                else:
                    estimate = filter_history(model, episode['observations'], inputs)
                    check('estimates', estimate, episode['estimates'])
                    warm = np.zeros(16)
                    for t, plan in enumerate(episode['plans']):
                        actions = np.array(plan['actions'])
                        assert len(actions) == min(16, 86 - t)
                        initial, _ = action_objective(warm[:len(actions)], estimate[t], g[t], model)
                        cost, gradient = action_objective(actions, estimate[t], g[t], model)
                        pg = np.max(np.abs(actions - np.clip(actions - gradient, -1., 1.)))
                        check('MPC_warm', initial, plan['initial_cost'])
                        check('MPC_selected', cost, plan['selected_cost'])
                        check('MPC_projected_gradient', pg, plan['projected_gradient_max'])
                        check('test_actions', actions[0], inputs[t])
                        assert plan['selected_cost'] <= plan['initial_cost'] + 1e-8
                        warm = np.r_[actions[1:], 0.]
                        counts['MPC_plans'] += 1
                for metric, value in metrics(states, episode['observations'], inputs, g).items():
                    if isinstance(value, bool):
                        assert episode[metric] == value
                    else:
                        check('metrics', value, episode[metric])
                assert np.max(np.abs(inputs)) <= 1.
                position_error = states[1:, 0] - g
                components = [np.mean(position_error**2) / (1 + np.mean(g**2)), .3 * np.mean(states[1:, 1]**2) / (1 + np.mean(g**2)), .03 * np.mean(inputs**2) / (1 + np.mean(g**2))]
                tracking_components[label].append(components + [float(np.sqrt(np.mean(position_error[-10:]**2))), float(np.sqrt(np.mean(states[-10:, 1]**2)))])
                position_bad, velocity_bad = np.any(np.abs(position_error[-10:]) > .3), np.any(np.abs(states[-10:, 1]) > .15)
                cause = 'both' if position_bad and velocity_bad else 'position_only' if position_bad else 'velocity_only' if velocity_bad else 'neither'
                failure_causes[label][cause] += 1
                counts['physical_test_episodes'] += 1
    for cell in data['summary']:
        rows = [row for row in data['runs'] if row['plant'] == cell['plant']]
        for label, summary in cell['controllers'].items():
            values = [np.mean([task['controllers'][label]['actual_cost'] for task in row['tests']]) for row in rows]
            for key, value in stats(values).items():
                check('summary', value, summary['cost'][key])
            assert summary['stationary_successes'] == sum(task['controllers'][label]['stationary_success'] for row in rows for task in row['tests'])
            assert summary['trials'] == 16
        for label in ['physical_CEM', 'virtual_CEM', 'nonlinear_MPC']:
            differences = [np.mean([task['controllers'][label]['actual_cost'] - task['controllers']['gradient_selected']['actual_cost'] for task in row['tests']]) for row in rows]
            for key, value in stats(differences).items():
                check('summary', value, cell['paired_differences'][label + '_minus_gradient'][key])
    assert len(arrays.files) == 256 and len(data['runs']) == 32
    assert data['meta']['additional_plant_training_steps'] == 0
    assert data['meta']['plant_test_steps'] == 32 * 2 * 4 * 86 and data['meta']['plant_test_resets'] == 256
    assert data['meta']['training_virtual_steps'] == sum(fit['virtual_steps'] for row in data['runs'] for fit in row['fits'])
    assert data['meta']['validation_virtual_steps'] == 32 * 5 * 16 * 86
    assert data['meta']['retained_trajectory_replay_virtual_steps'] == 32 * 2 * 16 * 86
    assert data['meta']['derivative_verification_virtual_steps'] == 37 * 16 * 86 == 50912
    for key in maximum:
        assert scaled[key] < 1e-6, (key, maximum[key], scaled[key])
    return dict(status='INDEPENDENTLY VERIFIED BOUNDED DIAGNOSTIC', counts=counts,
        selected_candidate_indices=selected, failed_optimizer_records=failures,
        maximum_absolute_errors=maximum, maximum_errors_scaled_by_max1_magnitude=scaled,
        derivative_fixture=derivative_fixture(),
        late_window_failure_counts=failure_causes,
        pooled_equal_task_mean_components={key:dict(zip(['normalized_position_cost', 'normalized_velocity_cost', 'normalized_effort_cost', 'late_position_rmse', 'late_velocity_rms'], np.mean(value, axis=0).tolist())) for key, value in tracking_components.items()},
        scope=[
            'Independent eliminated-state derivatives verify the actual issued clip derivative, including all smoothing/action-memory paths; no straight-through approximation.',
            'Optimization/validation use predicted learned-model states, never hidden true plant states; starts and candidate list fixed before new tests.',
            'All5 validation candidates checked, including unmodified originalwinners; selection is virtualonly and remains finite-bank optimistic.',
            'Fresh physical tests use disjoint +40b streams paired acrossfourmethods; previous physical/model data are reused, not new independent learnedmodels.',
            'Samefamily can demonstrate constructive attainable mean behavior; failedlocalsearch cannot prove incapacity or globaloptimality.',
            'Large extra virtualcompute and supplied model vocabulary remain; no universal necessity or method supremacy follows.',
            'Late multi-step stationarity and mean quadraticcost are different capabilities; all failuretypes and noise are retained.',
            'Virtual call-count arithmetic verified from saved counters; complete optimizer paths were not rerun.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
