#!/usr/bin/env python3
"""Independent audit of the exploratory model-only continuous-PD repair."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_goal_control_v10 import independent_pd, recurrence, independent_metrics


def main():
    root = Path(__file__).resolve().parent
    final = json.loads((root / 'goal_control_v10_results.json').read_text())
    pilot = json.loads((root / 'goal_control_v10_lbfgs_pilot_results.json').read_text())
    repaired = json.loads((root / 'goal_control_PD_v10_results.json').read_text())
    maximum = dict(virtual_objective=0., replay=0., observation=0., metrics=0., estimates=0., commands=0., summaries=0.)
    def check(key, a, b):
        maximum[key] = max(maximum[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    episodes = objectives = 0
    nfev_cap_records = 0
    for plant, old_plant, repair in zip(final['plants'], pilot['plants'], repaired['plants']):
        assert plant['true_theta'] == old_plant['true_theta'] == repair['true_theta']
        for run, old, improved in zip(plant['runs'], old_plant['runs'], repair['runs']):
            assert run['seed'] == old['seed'] == improved['seed']
            for field in ['training_commands', 'training_positions', 'nonlinear_theta', 'linear_theta',
                          'nonlinear_optimization', 'linear_optimization', 'selected_PD', 'PD_grid_scores', 'clean_prediction']:
                assert run[field] == old[field]
            for label, records in improved['optimization'].items():
                weight = 0. if label == 'continuous_PD_active' else 5.
                for record in records:
                    scores = []
                    for j, goals in enumerate([(1., -1., .5), (-1., 1., -.5)]):
                        result = independent_pd(run['nonlinear_theta'], record['gains'], goals, run['seed'] + 1000000 + j)
                        goal_energy = (6 * goals[0]**2 + 22 * goals[1]**2 + 42 * goals[2]**2) / 70
                        scores.append(result['normalized_tracking_cost'] + weight * (result['late_position_rmse']**2 + .3 * result['late_velocity_rms']**2) / (1 + goal_energy))
                    check('virtual_objective', np.mean(scores), record['score'])
                    objectives += 1
                    nfev_cap_records += record['status'] != -1 and not record['success']
                best = min(records, key=lambda record: record['score'])
                assert improved['gains'][label] == best['gains']
                assert records[-1]['gains'] == run['selected_PD'] and records[-1]['status'] == -1
            for task, original_task in zip(improved['tasks'], run['tasks']):
                assert task['goal_schedule'] == original_task['goal_schedule']
                for label, episode in task['controllers'].items():
                    rng = np.random.default_rng(episode['seed'])
                    sensor, process = rng.normal(0, .2, 87), rng.normal(0, .03, 86)
                    process[70:] = 0.
                    states = recurrence(plant['true_theta'], episode['commands'], noises=process)
                    check('replay', states, episode['states'])
                    check('observation', states[:, 0] + sensor, episode['observations'])
                    metrics = independent_metrics(states, episode['commands'], episode['goals'])
                    for key, value in metrics.items():
                        if value is None or isinstance(value, (bool, int)):
                            assert value == episode[key]
                        else:
                            check('metrics', value, episode[key])
                    kp, kd, alpha = improved['gains'][label]
                    sx = sv = 0.
                    for t, observation in enumerate(episode['observations'][:-1]):
                        previous = sx
                        sx += alpha * (observation - sx)
                        sv += alpha * ((sx - previous) - sv)
                        check('estimates', [sx, sv, 0.], episode['estimates'][t])
                        action = np.clip(kp * (episode['goals'][t] - sx) - kd * sv, -1., 1.) if t < 70 else 0.
                        check('commands', action, episode['commands'][t])
                    episodes += 1
        for label, summary in repair['summary'].items():
            for metric in ['normalized_tracking_cost', 'late_position_rmse', 'late_velocity_rms', 'mean_action_energy']:
                clusters = np.array([np.mean([task['controllers'][label][metric] for task in run['tasks']]) for run in repair['runs']])
                values = dict(mean=np.mean(clusters), mc_se=np.std(clusters, ddof=1) / np.sqrt(12),
                              median=np.median(clusters), q10=np.quantile(clusters, .1), q90=np.quantile(clusters, .9))
                for key, value in values.items():
                    check('summaries', value, summary[metric][key])
            assert summary['stop_successes'] == sum(task['controllers'][label]['stop_success'] for run in repair['runs'] for task in run['tasks'])
            assert summary['task_trials'] == 24
    assert max(maximum.values()) < 1e-7
    return dict(status='INDEPENDENTLY TESTED', pilot_final_training_and_fits_identical=True,
        recorded_virtual_objectives_checked=objectives, physical_test_episodes_replayed=episodes,
        unsuccessful_Powell_records=nfev_cap_records, maximum_absolute_errors=maximum,
        scope=[
            'Both gain objectives use only fitted nonlinear model and supplied virtual sensor/process-noise model; no extra physical interaction.',
            'Only after gain selection does canonical source evaluate true plant on fixed test tasks.',
            'Exploratory repair after weak coarse-grid baseline; not preregistered superiority evidence.',
            'Coarse candidate retained, all optimization failures saved; no global gain-optimality guarantee.',
            'More offline computation, supplied stopping objective, supplied timing and gain family must be disclosed.',
            'Mean/MC SE use twelve fitted-model seed clusters, each averaging two tasks.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
