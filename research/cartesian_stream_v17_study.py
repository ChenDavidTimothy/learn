#!/usr/bin/env python3
"""Fresh fixed-scope local retention study. Source-only until reviewed.

Acquisition is produced separately by immutable acquire.py. Fitting opens only
ordinary records; physical states/IK are confined to the external control driver.
No neural fits, hyperparameter expansion, test selection, or control tuning.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

from cartesian_stream_v17_learn import (
    DT, H, LOCAL_CANDIDATES, digest, rows, local_predictor, forecast_score,
)
from cartesian_stream_v17_control import (
    BOUND, BLOCKS, WARMUP_INTERVALS, PHASE_INTERVALS, CONTROL_INTERVALS,
    TOTAL_INTERVALS, OPTIMIZER_OPTIONS, PlanningFailure, json_ready, run_episode,
)
from cartesian_stream_v17_retention import FrozenLibraryPredictor
from cartesian_stream_v17_acquire import SD, jacobian

SOURCE_DIR = Path(__file__).resolve().parent
REPLICATES = tuple(range(200, 208))
DEPENDENCIES = (
    'cartesian_stream_v17_acquire.py', 'cartesian_stream_v17_learn.py',
    'cartesian_stream_v17_local.py', 'cartesian_stream_v17_control.py',
    'cartesian_stream_v17_retention.py', 'coupled_context_v14_pilot.py',
    'cartesian_stream_v17_acquisition_design.md', 'cartesian_stream_v17_learning_design.md',
    'cartesian_stream_v17_planner_contract.md', 'cartesian_stream_v17_study_design.md',
)


def paths(prefix):
    prefix = str(prefix)
    return {name: Path(prefix + suffix) for name, suffix in (
        ('ordinary', '_ordinary.npz'), ('evaluator', '_evaluator.npz'),
        ('acquisition', '_acquisition.json'), ('fit', '_local_fit.json'),
        ('test_forecasts', '_local_test_forecasts.npz'),
        ('fit_failure', '_local_fit_failure.json'),
        ('results', '_study_results.json'), ('arrays', '_study_arrays.npz'),
        ('failure', '_study_failure.json'),
    )}


def dependencies():
    return {name: digest(SOURCE_DIR/name) for name in DEPENDENCIES}


def verify_acquisition(prefix, replicate):
    """Check provenance without opening evaluator arrays as learner features."""
    if replicate not in REPLICATES:
        raise ValueError('fresh study has the fixed replicate scope200..207')
    p = paths(prefix)
    meta = json.loads(p['acquisition'].read_text())
    assert meta['replicate'] == replicate
    assert meta['status'] == 'EXPLORATORY MOVING ACQUISITION; NO LEARNER OR RELEASE'
    assert meta['source_sha256'] == digest(SOURCE_DIR/'cartesian_stream_v17_acquire.py')
    assert meta['plant_source_sha256'] == digest(SOURCE_DIR/'coupled_context_v14_pilot.py')
    assert meta['design_sha256'] == digest(SOURCE_DIR/'cartesian_stream_v17_acquisition_design.md')
    assert meta['counts'] == [24, 8, 8] and meta['steps'] == 256 and meta['dt'] == DT
    assert set(meta['artifacts']) == {p['ordinary'].name, p['evaluator'].name}
    for name in ('ordinary', 'evaluator'):
        assert digest(p[name]) == meta['artifacts'][p[name].name]
    return meta


def ordinary_arrays(ordinary):
    with np.load(ordinary, allow_pickle=False) as data:
        assert set(data.files) == {'observed', 'commands', 'split', 'trajectory_index', 'dt', 'sensor_sd'}
        assert float(data['dt']) == DT and float(data['sensor_sd']) == SD
        o, a, split = (data[name].copy() for name in ('observed', 'commands', 'split'))
    assert o.shape == (40, 257, 2) and a.shape == (40, 256, 2)
    assert np.all(np.isfinite(o)) and np.all(np.isfinite(a))
    ids = [np.flatnonzero(split == i) for i in range(3)]
    assert [len(x) for x in ids] == [24, 8, 8]
    return o, a, ids


def capture_test_forecasts(predictor, o, a, ids, blocks, stride, progress):
    """Same observed mean-path score as forecast_score, with raw forecasts saved."""
    records, origins, predictions, targets = [], [], [], []
    progress['active_test_records'] = records
    with torch.no_grad():
        for i in ids:
            for t in range(H, a.shape[1]-2*blocks+1, stride):
                progress['active_test_origin'] = dict(trajectory=int(i), time_index=t)
                progress.pop('active_test_prediction', None)
                plan = torch.tensor(a[i, t:t+2*blocks:2], dtype=torch.float64)
                prediction = predictor.paths(o[i, :t+1], a[i, :t], plan).numpy()
                target = o[i, t+1:t+2*blocks+1]
                progress['active_test_prediction'] = prediction
                squared = (prediction-target)**2
                if not np.all(np.isfinite(prediction)) or not np.isfinite(np.mean(squared)):
                    raise FloatingPointError('nonfinite selected test forecast; not used for selection')
                records.append(dict(
                    trajectory=int(i), time_index=t,
                    member_coordinate_mse=float(np.mean(squared)),
                    ensemble_mean_coordinate_mse=float(np.mean((prediction.mean(0)-target)**2)),
                    member_mse_by_future_sample=np.mean(squared, axis=(0, 2)),
                ))
                origins.append([int(i), t]); predictions.append(prediction); targets.append(target)
    score = dict(
        member_coordinate_mse=float(np.mean([r['member_coordinate_mse'] for r in records])),
        ensemble_mean_coordinate_mse=float(np.mean([r['ensemble_mean_coordinate_mse'] for r in records])),
        origins=len(records), records=records,
    )
    raw = dict(origins=np.asarray(origins), member_predictions=np.asarray(predictions),
               observed_targets=np.asarray(targets))
    for key in ('active_test_records', 'active_test_origin', 'active_test_prediction'):
        progress.pop(key, None)
    return score, raw


class ValidationProgress:
    """Transparent paths adapter retaining partial validation records on failure.

    The shared forecast_score still computes the actual selection score. Target
    access here is only for recording that validation computation, never an
    ordinary feature or an input to the underlying predictor.
    """
    def __init__(self, predictor, observations, commands, validation_ids, progress):
        self.predictor = predictor
        self.observations = observations
        self.origins = [(int(i), t) for i in validation_ids
                        for t in range(H, commands.shape[1]-12+1, 24)]
        self.progress = progress
        self.records = []
        self.progress['active_validation_records'] = self.records
        self.cursor = 0

    def paths(self, observations, commands, plan):
        i, t = self.origins[self.cursor]
        self.progress['active_validation_origin'] = dict(trajectory=i, time_index=t)
        self.progress.pop('active_validation_prediction', None)
        assert len(observations) == t+1
        prediction = self.predictor.paths(observations, commands, plan)
        raw = prediction.detach().numpy()
        target = self.observations[i, t+1:t+13]
        squared = (raw-target)**2
        self.progress['active_validation_prediction'] = raw
        self.records.append(dict(trajectory=i, time_index=t,
                                 finite=bool(np.all(np.isfinite(raw)) and np.isfinite(np.mean(squared))),
                                 member_coordinate_mse=float(np.mean(squared)),
                                 ensemble_mean_coordinate_mse=float(np.mean((raw.mean(0)-target)**2)),
                                 member_mse_by_future_sample=np.mean(squared, axis=(0, 2))))
        self.cursor += 1
        return prediction


def fit_local(prefix, replicate):
    """Fit/select the frozen four local candidates; do not run controls."""
    p = paths(prefix)
    if any(p[name].exists() for name in ('fit', 'test_forecasts', 'fit_failure')):
        raise FileExistsError('refusing to overwrite local fitting artifacts')
    acquisition = verify_acquisition(prefix, replicate)
    started = time.perf_counter()
    state = dict(status='IN PROGRESS LOCAL-ONLY FIT', replicate=replicate,
                 source_sha256=digest(__file__), dependency_sha256=dependencies(),
                 ordinary_sha256=digest(p['ordinary']), acquisition_sha256=digest(p['acquisition']),
                 candidates=[])
    raw_forecasts = {}
    try:
        o, a, (train, validation, test) = ordinary_arrays(p['ordinary'])
        X, U, Y, origins = rows(o, a, train)
        assert len(X) == 2880
        scale = dict(context_mean=X.mean(0), context_std=np.maximum(X.std(0), .01),
                     target_mean=Y.mean(0), target_std=np.maximum(Y.std(0), .001))
        X = (X-scale['context_mean'])/scale['context_std']
        Y = (Y-scale['target_mean'])/scale['target_std']
        bootstrap_rows, bootstrap_ids, bootstrap_paths = [], [], []
        for member in range(3):
            seed_path = [20261020, 17, 660, replicate, 0, member]
            rng = np.random.default_rng(np.random.SeedSequence(seed_path))
            chosen = rng.choice(train, len(train), replace=True)
            bootstrap_rows.append(np.concatenate([np.flatnonzero(origins[:, 0] == i) for i in chosen]))
            bootstrap_ids.append(chosen); bootstrap_paths.append(seed_path)
        state.update(scale=scale, training_origins=origins, bootstrap_rows=bootstrap_rows,
                     bootstrap_episode_ids=bootstrap_ids, bootstrap_seed_paths=bootstrap_paths)
        p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
        for ci, config in enumerate(LOCAL_CANDIDATES):
            state['active'] = dict(stage='validation', candidate=ci, neighbors=config[0], ridge=config[1])
            p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
            predictor = local_predictor(X, U, Y, bootstrap_rows, scale, config)
            recorder = ValidationProgress(predictor, o, a, validation, state)
            score = forecast_score(recorder, o, a, validation)
            assert score['origins'] == 80 and recorder.cursor == 80
            state['candidates'].append(dict(neighbors=config[0], ridge=config[1], validation=score))
            for key in ('active_validation_records', 'active_validation_origin', 'active_validation_prediction'):
                state.pop(key, None)
            p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
            print('replicate', replicate, 'local candidate', ci, 'validation complete', flush=True)
        selected = int(np.argmin([r['validation']['member_coordinate_mse'] for r in state['candidates']]))
        state['selected_local_candidate'] = selected
        p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
        # Selection is complete before any test outcome is evaluated.
        predictor = local_predictor(X, U, Y, bootstrap_rows, scale, LOCAL_CANDIDATES[selected])
        test_scores = {}
        for name, blocks, stride, expected in (('one_block', 1, 2, 960), ('six_block', 6, 24, 80)):
            state['active'] = dict(stage='selected_test', name=name)
            score, raw = capture_test_forecasts(predictor, o, a, test, blocks, stride, state)
            assert score['origins'] == expected
            test_scores[name] = score
            for key, value in raw.items():
                raw_forecasts[f'{name}_{key}'] = value
        np.savez_compressed(p['test_forecasts'], **raw_forecasts)
        state.update(
            status='COMPLETE FRESH LOCAL-ONLY FIT/SELECTION; INDEPENDENT AUDIT PENDING',
            test_observed_scores=test_scores, test_forecasts_sha256=digest(p['test_forecasts']),
            acquisition_resources=acquisition['resources'],
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            resources=dict(training_rows=2880, validation_rows=960, test_rows=960,
                           local_candidate_families=4, bootstrap_libraries_per_candidate=3,
                           validation_origins_per_candidate=80, neural_fits=0,
                           new_physical_experience_from_fitting=0),
            wall_seconds=time.perf_counter()-started,
            limitations='Whole-trajectory bootstrap fit variation is not full context uncertainty. '
                        'Test mean paths are evaluated only after validation selection. '
                        'No neural training or comparison is performed in this study.',
        )
        state.pop('active', None)
        p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
        return state
    except BaseException as error:
        state.update(status='FAILED/INTERRUPTED LOCAL FIT; NOT SELECTED FOR CONTROL',
                     error_type=type(error).__name__, error=str(error),
                     wall_seconds=time.perf_counter()-started)
        p['fit_failure'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
        p['fit'].write_text(json.dumps(json_ready(state), indent=2, allow_nan=False)+'\n')
        if raw_forecasts:
            np.savez_compressed(p['test_forecasts'], **raw_forecasts)
        raise


def load_local(prefix, ordinary=None):
    """Reconstruct a fresh selected library, never opening evaluator arrays."""
    p = paths(prefix)
    ordinary = p['ordinary'] if ordinary is None else Path(ordinary)
    meta = json.loads(p['fit'].read_text())
    assert meta['status'] == 'COMPLETE FRESH LOCAL-ONLY FIT/SELECTION; INDEPENDENT AUDIT PENDING'
    assert meta['source_sha256'] == digest(__file__)
    assert meta['dependency_sha256'] == dependencies()
    assert digest(ordinary) == meta['ordinary_sha256']
    o, a, (train, _, _) = ordinary_arrays(ordinary)
    X, U, Y, origins = rows(o, a, train)
    assert np.array_equal(origins, np.asarray(meta['training_origins']))
    scale = meta['scale']
    X = (X-np.asarray(scale['context_mean']))/np.asarray(scale['context_std'])
    Y = (Y-np.asarray(scale['target_mean']))/np.asarray(scale['target_std'])
    bs = [np.asarray(b, dtype=int) for b in meta['bootstrap_rows']]
    ci = int(meta['selected_local_candidate'])
    config = LOCAL_CANDIDATES[ci]
    assert meta['candidates'][ci]['neighbors'] == config[0] and meta['candidates'][ci]['ridge'] == config[1]
    return local_predictor(X, U, Y, bs, scale, config), meta


def make_cases(replicate):
    """Parent preparation/warmup/noise layout with only replicate slot changed."""
    if replicate not in REPLICATES:
        raise ValueError('fresh study has the fixed replicate scope200..207')
    cases = []
    for context_id in range(4):
        initial_path = [20261020, 17, 670, replicate, context_id, 0]
        rng = np.random.default_rng(np.random.SeedSequence(initial_path))
        theta = rng.uniform(-np.pi, np.pi); phi = rng.uniform(.4, np.pi-.4)
        sign = rng.choice([-1., 1.]); speed = rng.uniform(.02, .12)
        direction = rng.uniform(-np.pi, np.pi)
        q = np.asarray([theta-sign*phi/2, sign*phi])
        V = speed*np.asarray([np.cos(direction), np.sin(direction)])
        initial = np.r_[q, np.linalg.solve(jacobian(q), V)]
        warmup_path = [20261020, 17, 670, replicate, context_id, 2]
        rng = np.random.default_rng(np.random.SeedSequence(warmup_path))
        warmup = np.vstack((np.zeros((8, 2)), np.repeat(rng.uniform(-BOUND, BOUND, (4, 2)), 2, axis=0)))
        for noise_repeat in range(2):
            noise_path = [20261020, 17, 670, replicate, context_id, 1, noise_repeat]
            rng = np.random.default_rng(np.random.SeedSequence(noise_path))
            noise = rng.normal(0, SD, (TOTAL_INTERVALS+1, 2))
            cases.append(dict(context_id=context_id, noise_repeat=noise_repeat,
                              initial_seed_path=initial_path, warmup_seed_path=warmup_path,
                              noise_seed_path=noise_path, initial_state=initial.copy(),
                              warmup_commands=warmup.copy(), sensor_noise=noise))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', type=int, required=True, choices=REPLICATES)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    p = paths(args.prefix)
    outputs = ('fit', 'test_forecasts', 'fit_failure', 'results', 'arrays', 'failure')
    if any(p[name].exists() for name in outputs):
        raise FileExistsError('refusing to overwrite any existing study artifact')
    p['results'].parent.mkdir(parents=True, exist_ok=True)
    acquisition = verify_acquisition(args.prefix, args.replicate)
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = dict(
        status='IN PROGRESS FRESH LOCAL RETENTION STUDY', replicate=args.replicate,
        source_sha256=digest(__file__), dependency_sha256=dependencies(),
        acquisition_sha256=digest(p['acquisition']), ordinary_sha256=digest(p['ordinary']),
        evaluator_sha256=digest(p['evaluator']), acquisition_resources=acquisition['resources'],
        acquisition_invocation_wall_seconds=acquisition['wall_seconds'],
        optimizer_options=OPTIMIZER_OPTIONS.copy(), witnesses=[], ordinary_episodes=[],
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
        limitations='One corpus invocation within fixed eight-corpus study. Same fixed body; '
                    'reference is privileged, not an equally informed competitor. '
                    'Live/frozen comparison is total closed-loop append effect; later contexts differ. '
                    'No neural fits, modern benchmark superiority or real-time claim.',
    )
    arrays, progress = {}, {}
    model = None

    def checkpoint():
        result['study_invocation_wall_seconds_excluding_separate_acquisition'] = time.perf_counter()-started
        result['sum_of_separate_invocation_wall_seconds'] = (acquisition['wall_seconds']
            + result['study_invocation_wall_seconds_excluding_separate_acquisition'])
        np.savez_compressed(p['arrays'], **arrays)
        result['arrays_sha256'] = digest(p['arrays'])
        p['results'].write_text(json.dumps(json_ready(result), indent=2, allow_nan=False)+'\n')

    def store(record, raw, category):
        index = len(result[category]); record['array_keys'] = {}
        for name, value in raw.items():
            key = f'{category}{index}_{name}'
            arrays[key] = value; record['array_keys'][name] = key
        result[category].append(record)
        checkpoint()

    try:
        with threadpool_limits(limits=1):
            fit = fit_local(args.prefix, args.replicate)
            result.update(fit_sha256=digest(p['fit']),
                          selected_local_candidate=int(fit['selected_local_candidate']),
                          test_forecasts_sha256=digest(p['test_forecasts']),
                          fit_wall_seconds=fit['wall_seconds'])
            cases = make_cases(args.replicate)
            for i, case in enumerate(cases):
                for key in ('initial_state', 'warmup_commands', 'sensor_noise'):
                    arrays[f'case{i}_{key}'] = case[key]
            checkpoint()
            for case_index, case in enumerate(cases):
                progress.clear(); model = None
                record, raw = run_episode(case, None, 'privileged_nominal_witness', progress)
                record['case_index'] = case_index
                store(record, raw, 'witnesses')
                print('replicate', args.replicate, 'witness', case_index, 'complete', flush=True)
            for case_index, case in enumerate(cases):
                paired_retained = None
                for variant in ('retained', 'frozen'):
                    progress.clear(); model = None
                    load_started = time.perf_counter()
                    base, fit_meta = load_local(args.prefix)
                    assert digest(p['fit']) == result['fit_sha256']
                    model = base if variant == 'retained' else FrozenLibraryPredictor(base)
                    initial_rows = [m.n_rows for m in model.members]
                    predictor_load_seconds = time.perf_counter()-load_started
                    record, raw = run_episode(case, model, model.kind, progress)
                    final_rows = [m.n_rows for m in model.members]
                    if variant == 'retained':
                        assert final_rows == [n+CONTROL_INTERVALS//2 for n in initial_rows]
                    else:
                        assert model.notifications == CONTROL_INTERVALS//2
                        assert len(model.coefficients) == CONTROL_INTERVALS//2
                        assert final_rows == initial_rows
                        raw.update(local_fit_coefficients=np.asarray(model.coefficients),
                                   local_fit_queries=np.asarray(model.queries))
                        assert paired_retained is not None
                        assert record['noise_seed_path'] == paired_retained['noise_seed_path']
                        assert np.array_equal(record['goals'], paired_retained['goals'])
                        record['paired_retained_task_cost'] = paired_retained['score']['task_cost']
                        record['frozen_minus_retained_task_cost'] = (record['score']['task_cost']
                                                                    - paired_retained['score']['task_cost'])
                    record.update(case_index=case_index, variant=variant,
                                  selected_candidate_index=int(fit_meta['selected_local_candidate']),
                                  selected_configuration=model.config,
                                  predictor_load_seconds=predictor_load_seconds,
                                  local_library_initial_rows=initial_rows, local_library_final_rows=final_rows,
                                  real_pair_notifications=CONTROL_INTERVALS//2,
                                  actual_local_row_events=(CONTROL_INTERVALS//2 if variant == 'retained' else 0))
                    store(record, raw, 'ordinary_episodes')
                    if variant == 'retained': paired_retained = record
                    print('replicate', args.replicate, variant, 'case', case_index, 'complete', flush=True)
        ordinary, witnesses = result['ordinary_episodes'], result['witnesses']
        assert len(ordinary) == 16 and len(witnesses) == 8
        result['resources'] = dict(
            acquisition_trajectories=40, acquisition_intervals=10240, acquisition_simulated_seconds=1024.,
            acquisition_independent_scalar_camera_draws=20560,
            local_candidate_families=4, bootstrap_libraries_per_candidate=3, neural_fits=0,
            ordinary_episodes=16, sustained_goal_windows=48, ordinary_decisions=1440,
            ordinary_intervals=3136, ordinary_simulated_seconds=313.6,
            ordinary_camera_read_opportunities=3152, ordinary_scalar_noise_opportunities=6304,
            fresh_initial_preparations=4, fresh_paired_noise_paths=8, distinct_scalar_control_noise_draws=3152,
            retained_actual_row_events=720, retained_member_append_copies=2160,
            frozen_real_pair_notifications=720, frozen_actual_append_copies=0,
            reference_episodes=8, privileged_reference_intervals=1568, privileged_reference_seconds=156.8,
            new_training_experience_from_reference=0,
            ordinary_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in ordinary),
            reference_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in witnesses),
            modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in ordinary for d in e['decisions']),
            unsuccessful_optimizer_terminations=sum(not d['optimizer']['success'] for e in ordinary for d in e['decisions']),
            fallback_decisions=sum(d['fallback_to_initial'] for e in ordinary for d in e['decisions']),
        )
        assert acquisition['resources']['physical_intervals'] == result['resources']['acquisition_intervals']
        assert acquisition['resources']['scalar_noise_draws'] == result['resources']['acquisition_independent_scalar_camera_draws']
        result['status'] = 'COMPLETE FRESH LOCAL RETENTION CORPUS; INDEPENDENT AUDIT PENDING'
        checkpoint()
    except BaseException as error:
        for name in ('states', 'observed', 'commands', 'nfev', 'local_coefficients', 'local_queries'):
            if name in progress: arrays['failed_partial_'+name] = np.asarray(progress[name])
        if isinstance(model, FrozenLibraryPredictor):
            arrays['failed_partial_local_coefficients'] = np.asarray(model.coefficients)
            arrays['failed_partial_local_queries'] = np.asarray(model.queries)
        failure = dict(status='FAILED/INTERRUPTED FRESH STUDY; PARTIAL RECORDS PRESERVED',
                       replicate=args.replicate, source_sha256=digest(__file__),
                       error_type=type(error).__name__, error=str(error), active_episode=progress.get('record'))
        if isinstance(error, PlanningFailure): failure['planning_diagnostics'] = error.diagnostics
        p['failure'].write_text(json.dumps(json_ready(failure), indent=2, allow_nan=False)+'\n')
        result['status'] = failure['status']; checkpoint()
        raise


if __name__ == '__main__':
    main()
