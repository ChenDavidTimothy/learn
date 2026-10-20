#!/usr/bin/env python3
"""Independent artifact audit of the unreleased direct-experience pilot.

Not a completed competitive learning comparison. Physical replay eliminates the
motor; CEM reconstruction uses only observed arrays/fitness. The motor-first
contrast uses separate adaptive quadrature and an expanded MC loss identity.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

from audit_goal_control_v10 import recurrence


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'direct_experience_v12_pilot_results.json').read_text())
    theta, sigma, n = [.65, .7, .2, 1.2], .2, 86
    errors = dict(physical_states=0., observations=0., commands=0., goals=0., observed_fitness=0.,
        evaluator_metrics=0., CEM_coordinates=0., CEM_gains=0., summary=0., quadrature=0., counterexample_MC=0.)
    def check(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    training_episodes = test_episodes = 0
    train_noise_seeds = set()
    test_noise_seeds = set()

    def episode_audit(episode, gains, goals, motor_sd):
        rng = np.random.default_rng(episode['seed'])
        sensor = rng.normal(0, sigma, n + 1)
        noise = rng.normal(0, motor_sd, n)
        # Unlike v10/v11, the motor process and control continue through all86
        # steps. No forced zero-command/noise-free coast is introduced.
        states = recurrence(theta, episode['commands'], noises=noise)
        check('physical_states', states, episode['states'])
        check('observations', states[:, 0] + sensor, episode['observations'])
        targets = np.array([goals[0] if t < 6 else goals[1] if t < 28 else goals[2] for t in range(n)])
        check('goals', targets, episode['goals'])
        kp, kd, alpha = gains
        sx = sv = 0.
        commands = []
        for t, observation in enumerate(episode['observations'][:-1]):
            old_sx = sx
            sx = (1 - alpha) * sx + alpha * observation
            sv = (1 - alpha) * sv + alpha * (sx - old_sx)
            commands.append(float(np.clip(kp * (targets[t] - sx) - kd * sv, -1., 1.)))
        check('commands', commands, episode['commands'])
        assert max(np.abs(episode['commands'])) <= 1.
        observations = np.array(episode['observations'])
        commands = np.array(episode['commands'])
        scale = 1 + np.sum(targets**2) / n
        observed = np.sum((observations[1:] - targets)**2 + .3 * (observations[1:] - observations[:-1])**2 + .03 * commands**2) / (n * scale)
        actual = np.sum((states[1:, 0] - targets)**2 + .3 * states[1:, 1]**2 + .03 * commands**2) / (n * scale)
        check('observed_fitness', observed, episode['observed_fitness'])
        offset = 1.6 * sigma**2 / scale
        check('evaluator_metrics', actual, episode['actual_evaluator_cost'])
        check('evaluator_metrics', offset, episode['expected_observation_loss_offset'])
        check('evaluator_metrics', observed - actual - offset, episode['fresh_gap_minus_expected_offset'])
        check('evaluator_metrics', np.sqrt(np.mean((states[-10:, 0] - targets[-10:])**2)), episode['late_position_rmse'])
        check('evaluator_metrics', np.sqrt(np.mean(states[-10:, 1]**2)), episode['late_velocity_rms'])
        success = np.all(np.abs(states[-10:, 0] - targets[-10:]) <= .3) and np.all(np.abs(states[-10:, 1]) <= .15)
        assert bool(success) == episode['stationary_success']
        return float(observed)

    for run in data['runs']:
        rng = np.random.default_rng(run['seed'])
        mean = np.array([math.log(.08), math.log(.6), math.log(.6 / .4)])
        sd = np.ones(3)
        lower = np.array([math.log(.005), math.log(.02), math.log(.1 / .9)])
        upper = np.array([math.log(.5), math.log(4.), math.log(.98 / .02)])
        best = None
        assert len(run['training']) == 48
        for generation in range(4):
            candidates = np.clip(rng.normal(mean, sd, (12, 3)), lower, upper)
            if best is None:
                candidates[0] = mean
                candidates[1] = [math.log(.02), 0., math.log(.8 / .2)]
            else:
                candidates[0] = best['coordinates']
            current = []
            for j, coordinates in enumerate(candidates):
                index = 12 * generation + j
                trial = run['training'][index]
                assert trial['generation'] == generation and trial['candidate'] == j
                check('CEM_coordinates', coordinates, trial['coordinates'])
                gains = [math.exp(coordinates[0]), math.exp(coordinates[1]), 1 / (1 + math.exp(-coordinates[2]))]
                check('CEM_gains', gains, trial['gains'])
                observed_values = []
                for k, goals in enumerate([(1., -1., .5), (-1., 1., -.5)]):
                    episode = trial['episodes'][k]
                    expected_seed = run['seed'] + 100000 + 100 * index + k
                    assert episode['seed'] == expected_seed and expected_seed not in train_noise_seeds
                    train_noise_seeds.add(expected_seed)
                    observed_values.append(episode_audit(episode, gains, goals, 0.))
                    training_episodes += 1
                fitness = float(np.mean(observed_values))
                check('observed_fitness', fitness, trial['observed_fitness'])
                # This reconstruction never consults physical/evaluator cost
                # when selecting candidates, elites, or the final policy.
                current.append(dict(coordinates=coordinates.copy(), gains=gains, fitness=fitness))
                if best is None or fitness < best['fitness']:
                    best = current[-1]
            elite = sorted(current, key=lambda row: row['fitness'])[:4]
            elite_coordinates = np.array([row['coordinates'] for row in elite])
            mean = np.mean(elite_coordinates, axis=0)
            sd = np.maximum(np.std(elite_coordinates, axis=0), .15)
        check('CEM_gains', best['gains'], run['selected_gains'])
        for j, test in enumerate(run['tests']):
            goals = [(2., -2., 1.), (6., -4., 3.)][j]
            assert list(goals) == test['goal_schedule']
            fresh = run['seed'] + 2000000 + j
            assert fresh not in train_noise_seeds and fresh not in test_noise_seeds
            test_noise_seeds.add(fresh)
            assert test['learned']['seed'] == test['fixed_PD']['seed'] == fresh
            episode_audit(test['learned'], run['selected_gains'], goals, .03)
            episode_audit(test['fixed_PD'], [.2, .6, .25], goals, .03)
            test_episodes += 2
    assert not (train_noise_seeds & test_noise_seeds)
    for label, summary in data['summary'].items():
        actual = np.array([np.mean([test[label]['actual_evaluator_cost'] for test in run['tests']]) for run in data['runs']])
        gap = np.array([np.mean([test[label]['fresh_gap_minus_expected_offset'] for test in run['tests']]) for run in data['runs']])
        check('summary', np.mean(actual), summary['mean_actual_cost'])
        check('summary', np.std(actual, ddof=1) / np.sqrt(8), summary['mc_se_actual_cost'])
        check('summary', np.mean(gap), summary['mean_fresh_gap_minus_expected_offset'])
        check('summary', np.std(gap, ddof=1) / np.sqrt(8), summary['mc_se_gap'])
        assert summary['stationary_successes'] == sum(test[label]['stationary_success'] for run in data['runs'] for test in run['tests'])
        assert summary['task_trials'] == 16

    contrast = data['motor_first_counterexample']
    cross, quadrature_error = quad(lambda z: 2 * sigma * z * math.tanh(z) * math.exp(-z*z/2) / math.sqrt(2 * math.pi),
                                  0., np.inf, epsabs=1e-13, epsrel=1e-13)
    check('quadrature', cross, contrast['sensor_cross_moment'])
    rng = np.random.default_rng(data['meta']['seed'] - 1)
    e0 = rng.normal(0, sigma, contrast['n'])
    e1 = rng.normal(0, sigma, contrast['n'])
    zscores = []
    for case in contrast['cases']:
        velocity = -theta[3] * case['k'] * np.tanh(e0 / sigma)
        # Expand the difference algebraically rather than subtracting the
        # canonical squared losses. Both u-energy terms cancel exactly.
        sample_gap = 1.3 * e1**2 + .3 * e0**2 - .6 * e0 * e1 + 2.6 * velocity * e1 - .6 * velocity * e0
        target = 1.6 * sigma**2 + .6 * theta[3] * case['k'] * cross
        check('quadrature', target, case['expected_stage_loss_gap'])
        check('counterexample_MC', np.mean(sample_gap), case['measured_gap'])
        check('counterexample_MC', np.std(sample_gap, ddof=1) / np.sqrt(len(e0)), case['mc_se'])
        zscores.append((case['measured_gap'] - target) / case['mc_se'])
    assert max(errors.values()) < 1e-8
    assert training_episodes * n // 8 == data['meta']['physical_training_steps_per_seed'] == 8256
    return dict(status='UNRELEASED PILOT ARTIFACT INDEPENDENTLY VERIFIED; fair learning comparison pending',
        training_episodes_independently_replayed=training_episodes, fresh_test_episodes_independently_replayed=test_episodes,
        independent_CEM_seeds=8, candidate_policies_checked=384,
        physical_training_steps_per_seed=8256, maximum_absolute_errors=errors,
        independent_adaptive_quadrature_cross_moment=cross, adaptive_quadrature_reported_error=quadrature_error,
        million_sample_contrast_standard_errors_from_expectation=zscores,
        scope=[
            'CEM gain generation, elite updates and globalbest selection reconstructed using observed arrays/fitness only.',
            'True theta/state/evaluator loss are simulator/scorer data and never used for candidate ranking.',
            'Every training candidate uses fresh unique sensor streams; incumbent coordinates are reevaluated with new noise.',
            'Fresh test policy is fixed before its independent test noise; finite training winner fitness remains optimistic.',
            'Expected1.6R offset requires lagged action timing, additive iid sensing, and goals/normalization exogenous to current sensor errors.',
            'Motor-first bounded contrast has policy-dependent bias; this observation-loss bridge is not universal.',
            'Control and motor disturbance continue all86 test steps; no v10/v11 forced coast is used.',
            'Fixed PD is descriptive and weak; same-data model-based competitors and richer controller families remain required.',
            'Eight independent training-seed clusters determine MC SE after averaging two test tasks.',
            'No complete learning-method comparison, competitive robotics claim or general mechanism established.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
