#!/usr/bin/env python3
"""Aggregate the fixed eight-corpus local-retention study, with no filtering.

Source only until all eight artifacts complete. Publication must invoke
--require-audits; an unaudited diagnostic output is explicitly ineligible.
No fit, simulation, model selection, p-value or protocol modification occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPLICATES = tuple(range(200, 208))
MENU = ((128, .001), (128, .1), (512, .001), (512, .1))
DT = .1
VARIANTS = ('retained', 'frozen')
METHODS = VARIANTS + ('privileged_reference',)
QUANTILES = (0., .5, .9, .99, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant {value} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f'{label}: shape/nonfinite inconsistency')
    if not np.allclose(a, b, atol=1e-12, rtol=1e-10):
        raise AssertionError(f'{label}: numerical inconsistency')


def bind(path, artifacts, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    if expected is not None and value != expected:
        raise AssertionError(f'stale/corrupt artifact: {path}')
    key = str(path)
    if key in artifacts and artifacts[key] != value:
        raise AssertionError(f'artifact changed during aggregation: {path}')
    artifacts[key] = value
    return value


def cluster(values):
    """The eight entries are corpus means/differences, never episode rows."""
    x = np.asarray(values, dtype=float)
    assert x.shape[0] == 8 and np.all(np.isfinite(x))
    return dict(mean=np.mean(x, axis=0).tolist(),
                se_across_eight_corpora=(np.std(x, axis=0, ddof=1)/np.sqrt(8)).tolist(),
                min=np.min(x, axis=0).tolist(), max=np.max(x, axis=0).tolist(),
                corpus_values=x.tolist())


def quantiles(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.all(np.isfinite(values))
    return {str(q): float(np.quantile(values, q)) for q in QUANTILES}


def format_path(template, rep):
    return Path(str(template).format(rep=rep, replicate=rep))


def positions_and_velocity(states):
    q, v = states[:, :2], states[:, 2:]
    angle = q[:, 0]+q[:, 1]
    position = np.column_stack((np.cos(q[:, 0])+np.cos(angle),
                                np.sin(q[:, 0])+np.sin(angle)))
    velocity = np.column_stack((-np.sin(q[:, 0])*v[:, 0]-np.sin(angle)*(v[:, 0]+v[:, 1]),
                                np.cos(q[:, 0])*v[:, 0]+np.cos(angle)*(v[:, 0]+v[:, 1])))
    return position, velocity


def episode_summary(episode, arrays, method, rep, result_path):
    keys = episode['array_keys']
    states, observed, commands, nfev = (arrays[keys[name]] for name in
        ('states_evaluator_only', 'observed', 'commands', 'plant_nfev'))
    assert states.shape == (197, 4) and observed.shape == (197, 2)
    assert commands.shape == (196, 2) and nfev.shape == (196,)
    assert all(np.all(np.isfinite(x)) for x in (states, observed, commands, nfev))
    assert np.max(np.abs(commands)) <= .5 and np.all(nfev > 0)
    assert episode['plant_rhs_evaluations'] == int(nfev.sum())
    goal = np.asarray(episode['goals'])
    assert goal.shape == (3, 2)
    anchor = .95*observed[16]
    expected_goals = [anchor]
    for angle in (.04, -.04):
        c, s = np.cos(angle), np.sin(angle)
        expected_goals.append(np.array([[c, -s], [s, c]])@anchor)
    close(goal, expected_goals, 'ordinary-reading target construction')
    position, velocity = positions_and_velocity(states)
    effort = np.sum(commands**2, axis=1)
    actual_phase_costs, phase_effort, precise, phase_entries = [], [], [], []
    assert len(episode['score']['phases']) == 3
    case = int(episode['case_index'])
    for phase, score in enumerate(episode['score']['phases']):
        start, stop = 16+60*phase, 16+60*(phase+1)
        error, speed = position[start+1:stop+1]-goal[phase], velocity[start+1:stop+1]
        squared_effort = effort[start:stop]
        position_cost = DT*np.sum(error**2)
        motion_cost = DT*.25*np.sum(speed**2)
        effort_penalty = DT*.01*np.sum(squared_effort)
        cost = float(position_cost+motion_cost+effort_penalty)
        precision = bool(np.all(np.abs(error[-10:]) <= .03)
                         and np.all(np.abs(speed[-10:]) <= .03))
        close(score['cost'], cost, 'phase cost')
        close(score['integrated_position_error'], position_cost, 'position cost')
        close(score['integrated_motion_penalty'], motion_cost, 'motion cost')
        close(score['integrated_effort_penalty'], effort_penalty, 'effort penalty')
        close(score['position_error_by_sample'], error, 'position profile')
        close(score['cartesian_velocity_by_sample'], speed, 'velocity profile')
        close(score['squared_effort_by_interval'], squared_effort, 'effort profile')
        assert score['precision_success'] == precision
        actual_phase_costs.append(cost); phase_effort.append(float(DT*np.sum(squared_effort)))
        precise.append(precision)
        phase_entries.append(dict(replicate=rep, variant=method, case_index=case,
                                  context_id=int(episode['context_id']), noise_repeat=int(episode['noise_repeat']),
                                  phase=phase, cost=cost, precision_success=precision,
                                  late_max_coordinate_position_error=float(np.max(np.abs(error[-10:]))),
                                  late_max_coordinate_velocity=float(np.max(np.abs(speed[-10:]))),
                                  result_artifact=str(result_path.resolve()), array_keys=keys))
    task_cost = float(sum(actual_phase_costs))
    warmup_effort = float(DT*np.sum(effort[:16]))
    warmup_motion = float(DT*.25*np.sum(velocity[1:17]**2))
    warmup_cost = warmup_motion+.01*warmup_effort
    close(episode['score']['task_cost'], task_cost, 'task cost')
    close(episode['score']['warmup']['integrated_squared_effort'], warmup_effort, 'warmup effort')
    close(episode['score']['warmup']['cost_without_unannounced_goal'], warmup_cost, 'warmup cost')
    close(episode['score']['combined_charged_cost'], task_cost+warmup_cost, 'charged cost')
    assert episode['score']['precise_goals'] == sum(precise)
    entry = dict(replicate=rep, variant=method, case_index=case,
                 context_id=int(episode['context_id']), noise_repeat=int(episode['noise_repeat']),
                 task_cost_18sec=task_cost, phase_costs=actual_phase_costs,
                 first_goal_cost=actual_phase_costs[0], later_goals_total_cost=sum(actual_phase_costs[1:]),
                 later_goal_mean_cost=float(np.mean(actual_phase_costs[1:])),
                 combined_charged_cost=task_cost+warmup_cost, warmup_motion_effort_cost=warmup_cost,
                 task_integrated_squared_effort=sum(phase_effort),
                 warmup_integrated_squared_effort=warmup_effort,
                 combined_integrated_squared_effort=sum(phase_effort)+warmup_effort,
                 phase_precision=precise, precise_goals=int(sum(precise)),
                 result_artifact=str(result_path.resolve()), array_keys=keys)
    return entry, phase_entries


def mean_corpus(episodes):
    assert len(episodes) == 8 and sorted(e['case_index'] for e in episodes) == list(range(8))
    scalar_keys = ('task_cost_18sec', 'first_goal_cost', 'later_goals_total_cost',
                   'later_goal_mean_cost', 'combined_charged_cost', 'warmup_motion_effort_cost',
                   'task_integrated_squared_effort', 'warmup_integrated_squared_effort',
                   'combined_integrated_squared_effort')
    out = {key: float(np.mean([e[key] for e in episodes])) for key in scalar_keys}
    precision = np.asarray([e['phase_precision'] for e in episodes], dtype=bool)
    out.update(phase_costs=np.mean([e['phase_costs'] for e in episodes], axis=0).tolist(),
               phase_precision_fractions=np.mean(precision, axis=0).tolist(),
               precision_fraction=float(np.mean(precision)),
               first_precision_fraction=float(np.mean(precision[:, 0])),
               later_precision_fraction=float(np.mean(precision[:, 1:])),
               precision_count=int(precision.sum()), first_precision_count=int(precision[:, 0].sum()),
               later_precision_count=int(precision[:, 1:].sum()))
    return out


def verify_audit(path, rep, artifacts, required_hashes):
    audit = finite_json(path)
    assert audit['replicate'] == rep and audit['status'].startswith('PASS'), path
    bind(path, artifacts)
    recorded = audit['artifact_sha256']
    assert all(value in recorded.values() for value in required_hashes), ('audit missing current artifacts', path)
    explicit = audit.get('artifact_paths', {})
    for name, value in recorded.items():
        if name in explicit:
            candidate = Path(explicit[name])
        elif Path(name).is_absolute():
            candidate = Path(name)
        else:
            candidates = (Path(path).parent/name, ROOT/name, Path(name))
            candidate = next((p for p in candidates if p.exists()), candidates[0])
        bind(candidate, artifacts, value)
    return dict(path=str(Path(path).resolve()), status=audit['status'], sha256=sha(path))


def verify_forecasts(fit, raw):
    metrics = {}
    for name, length, count in (('one_block', 2, 960), ('six_block', 12, 80)):
        origins = raw[name+'_origins']; prediction = raw[name+'_member_predictions']
        target = raw[name+'_observed_targets']
        assert origins.shape == (count, 2) and prediction.shape == (count, 3, length, 2)
        assert target.shape == (count, length, 2)
        assert np.all(np.isfinite(prediction)) and np.all(np.isfinite(target))
        squared = (prediction-target[:, None])**2
        member_mse = float(np.mean(squared))
        ensemble_mse = float(np.mean((prediction.mean(1)-target)**2))
        score = fit['test_observed_scores'][name]
        assert score['origins'] == count and len(score['records']) == count
        close(score['member_coordinate_mse'], member_mse, name+' test score')
        close(score['ensemble_mean_coordinate_mse'], ensemble_mse, name+' ensemble score')
        for row, origin, errors in zip(score['records'], origins, squared):
            assert (row['trajectory'], row['time_index']) == tuple(origin)
            close(row['member_coordinate_mse'], np.mean(errors), 'per-origin test score')
            close(row['member_mse_by_future_sample'], np.mean(errors, axis=(0, 2)), 'test sample profile')
        metrics[name] = dict(member_coordinate_mse=member_mse, ensemble_mean_coordinate_mse=ensemble_mse,
                             member_mse_by_future_sample=np.mean(squared, axis=(0, 1, 3)).tolist(), origins=count)
    return metrics


def load_corpus(prefix, rep, artifacts, common_dependencies, require_audits, ordinary_audit, physics_audit):
    p = {key: Path(str(prefix)+suffix) for key, suffix in (
        ('acquisition', '_acquisition.json'), ('ordinary', '_ordinary.npz'), ('evaluator', '_evaluator.npz'),
        ('fit', '_local_fit.json'), ('forecasts', '_local_test_forecasts.npz'),
        ('results', '_study_results.json'), ('arrays', '_study_arrays.npz'))}
    for suffix in ('_study_failure.json', '_local_fit_failure.json', '_failure.npz'):
        assert not Path(str(prefix)+suffix).exists(), ('failure artifact exists', prefix)
    result, fit, acquisition = (finite_json(p[k]) for k in ('results', 'fit', 'acquisition'))
    assert result['status'] == 'COMPLETE FRESH LOCAL RETENTION CORPUS; INDEPENDENT AUDIT PENDING'
    assert fit['status'] == 'COMPLETE FRESH LOCAL-ONLY FIT/SELECTION; INDEPENDENT AUDIT PENDING'
    assert acquisition['status'] == 'EXPLORATORY MOVING ACQUISITION; NO LEARNER OR RELEASE'
    assert result['replicate'] == fit['replicate'] == acquisition['replicate'] == rep
    assert result['optimizer_options'] == dict(maxiter=40, maxfun=80, maxls=20, ftol=1e-9, gtol=1e-6)
    dependencies = result['dependency_sha256']
    assert fit['dependency_sha256'] == dependencies
    if common_dependencies is not None: assert dependencies == common_dependencies
    assert result['source_sha256'] == fit['source_sha256']
    bind(ROOT/'cartesian_stream_v17_study.py', artifacts, result['source_sha256'])
    for name, value in dependencies.items(): bind(ROOT/name, artifacts, value)
    assert acquisition['source_sha256'] == dependencies['cartesian_stream_v17_acquire.py']
    assert acquisition['plant_source_sha256'] == dependencies['coupled_context_v14_pilot.py']
    assert acquisition['design_sha256'] == dependencies['cartesian_stream_v17_acquisition_design.md']
    hashes = {key: bind(path, artifacts) for key, path in p.items()}
    assert hashes['acquisition'] == result['acquisition_sha256'] == fit['acquisition_sha256']
    assert hashes['ordinary'] == result['ordinary_sha256'] == fit['ordinary_sha256']
    assert hashes['evaluator'] == result['evaluator_sha256']
    assert hashes['fit'] == result['fit_sha256']
    assert hashes['forecasts'] == result['test_forecasts_sha256'] == fit['test_forecasts_sha256']
    assert hashes['arrays'] == result['arrays_sha256']
    for key in ('ordinary', 'evaluator'): assert acquisition['artifacts'][p[key].name] == hashes[key]
    assert result['acquisition_resources'] == fit['acquisition_resources'] == acquisition['resources']
    ordinary, evaluator, forecasts, arrays = (npz(p[k]) for k in ('ordinary', 'evaluator', 'forecasts', 'arrays'))
    assert ordinary['observed'].shape == (40, 257, 2) and ordinary['commands'].shape == (40, 256, 2)
    assert evaluator['states'].shape == (40, 257, 4) and evaluator['sensor_noise'].shape == (40, 257, 2)
    assert evaluator['plant_nfev'].shape == (40, 256)
    assert float(ordinary['dt']) == DT and float(ordinary['sensor_sd']) == .0002
    for a in (ordinary, evaluator):
        assert all(np.all(np.isfinite(x)) for x in a.values())
    train, val, test = (np.flatnonzero(ordinary['split'] == i) for i in range(3))
    assert [len(train), len(val), len(test)] == [24, 8, 8]
    assert len(fit['candidates']) == 4
    validation_scores = []
    expected_origins = [(int(i), t) for i in val for t in range(16, 245, 24)]
    assert len(expected_origins) == 80
    for ci, candidate in enumerate(fit['candidates']):
        assert (candidate['neighbors'], candidate['ridge']) == MENU[ci]
        score = candidate['validation']; assert score['origins'] == 80 and len(score['records']) == 80
        assert [(r['trajectory'], r['time_index']) for r in score['records']] == expected_origins
        mean = np.mean([r['member_coordinate_mse'] for r in score['records']])
        close(score['member_coordinate_mse'], mean, 'validation mean')
        validation_scores.append(float(mean))
    selected = int(fit['selected_local_candidate'])
    assert selected == result['selected_local_candidate'] == int(np.argmin(validation_scores))
    assert fit['resources']['neural_fits'] == 0
    assert fit['bootstrap_seed_paths'] == [[20261020, 17, 660, rep, 0, m] for m in range(3)]
    forecast_metrics = verify_forecasts(fit, forecasts)
    assert len(result['ordinary_episodes']) == 16 and len(result['witnesses']) == 8
    catalog = {name: [] for name in METHODS}; phases = []; decision_rows = []; failures = []
    pair_records = {}
    for method, records in ((None, result['ordinary_episodes']), ('privileged_reference', result['witnesses'])):
        for episode in records:
            name = method if method is not None else episode['variant']
            assert name in METHODS
            case = int(episode['case_index']); context = int(episode['context_id']); repeat = int(episode['noise_repeat'])
            assert case == 2*context+repeat and 0 <= case < 8
            assert episode['initial_seed_path'] == [20261020, 17, 670, rep, context, 0]
            assert episode['warmup_seed_path'] == [20261020, 17, 670, rep, context, 2]
            assert episode['noise_seed_path'] == [20261020, 17, 670, rep, context, 1, repeat]
            assert len(episode['decisions']) == 90
            entry, phase_entries = episode_summary(episode, arrays, name, rep, p['results'])
            catalog[name].append(entry); phases.extend(phase_entries)
            for phase in phase_entries:
                if not phase['precision_success']: failures.append(phase)
            keys = episode['array_keys']
            noise = arrays[f'case{case}_sensor_noise']
            assert noise.shape == (197, 2)
            pos, _ = positions_and_velocity(arrays[keys['states_evaluator_only']])
            close(arrays[keys['observed']], pos+noise, 'paired sensor path')
            close(arrays[keys['states_evaluator_only']][0], arrays[f'case{case}_initial_state'], 'initial preparation')
            close(arrays[keys['commands']][:16], arrays[f'case{case}_warmup_commands'], 'warmup commands')
            if name == 'privileged_reference': continue
            assert episode['selected_candidate_index'] == selected
            assert (episode['selected_configuration']['neighbors'], episode['selected_configuration']['ridge']) == MENU[selected]
            pair_records[(name, case)] = episode
            before, after = episode['local_library_initial_rows'], episode['local_library_final_rows']
            assert len(before) == len(after) == 3 and before == [2880]*3
            assert after == ([2970]*3 if name == 'retained' else before)
            assert episode['real_pair_notifications'] == 90
            assert episode['actual_local_row_events'] == (90 if name == 'retained' else 0)
            assert arrays[keys['local_fit_coefficients']].shape == (90, 3, 69, 4)
            assert arrays[keys['local_fit_queries']].shape == (90, 66)
            assert np.all(np.isfinite(arrays[keys['local_fit_coefficients']]))
            assert np.all(np.isfinite(arrays[keys['local_fit_queries']]))
            for renewal, decision in enumerate(episode['decisions']):
                assert decision['renewal'] == renewal and decision['time_index'] == 16+2*renewal
                plan = np.asarray(decision['selected_plan']); gradient = np.asarray(decision['selected_gradient'])
                assert plan.shape == gradient.shape == (6, 2) and np.max(np.abs(plan)) <= .5
                projected = plan-np.clip(plan-gradient, -.5, .5)
                close(decision['selected_projected_gradient'], projected, 'projected gradient')
                close(decision['selected_projected_gradient_max'], np.max(np.abs(projected)), 'PG maximum')
                assert decision['selected_cost'] <= decision['initial_cost']+2e-12
                decision_rows.append(dict(replicate=rep, variant=name, case_index=case, renewal=renewal,
                                          phase=int(decision['phase']), time_index=int(decision['time_index']),
                                          optimizer_success=bool(decision['optimizer']['success']),
                                          optimizer_status=decision['optimizer']['status'],
                                          optimizer_message=decision['optimizer']['message'],
                                          nfev=decision['optimizer']['nfev'], njev=decision['optimizer']['njev'],
                                          nit=decision['optimizer']['nit'], fallback=bool(decision['fallback_to_initial']),
                                          projected_gradient_max=float(decision['selected_projected_gradient_max']),
                                          modeled_value_gradient_calls=int(decision['modeled_value_gradient_calls']),
                                          choose_wall_seconds=float(decision['total_choose_wall_seconds']),
                                          result_artifact=str(p['results'].resolve())))
    for case in range(8):
        retained, frozen = pair_records[('retained', case)], pair_records[('frozen', case)]
        close(retained['goals'], frozen['goals'], 'paired targets')
        close(frozen['paired_retained_task_cost'], retained['score']['task_cost'], 'paired retained cost')
        close(frozen['frozen_minus_retained_task_cost'], frozen['score']['task_cost']-retained['score']['task_cost'], 'paired difference')
    means = {name: mean_corpus(entries) for name, entries in catalog.items()}
    resources = dict(
        acquired_trajectories=40, acquisition_intervals=int(ordinary['commands'].shape[0]*ordinary['commands'].shape[1]),
        acquisition_scalar_camera_draws=int(evaluator['sensor_noise'].size),
        acquisition_plant_rhs_evaluations=int(evaluator['plant_nfev'].sum()),
        ordinary_episodes=16, ordinary_goal_windows=48, ordinary_decisions=len(decision_rows),
        ordinary_intervals=sum(len(arrays[e['array_keys']['commands']]) for e in result['ordinary_episodes']),
        ordinary_camera_read_opportunities=sum(len(arrays[e['array_keys']['observed']]) for e in result['ordinary_episodes']),
        ordinary_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in result['ordinary_episodes']),
        initial_preparations=4, paired_noise_paths=8,
        distinct_scalar_control_noise_draws=sum(arrays[f'case{i}_sensor_noise'].size for i in range(8)),
        retained_actual_row_events=sum(e['actual_local_row_events'] for e in result['ordinary_episodes'] if e['variant']=='retained'),
        retained_member_append_copies=int(sum(sum(np.asarray(e['local_library_final_rows'])-e['local_library_initial_rows'])
                                              for e in result['ordinary_episodes'] if e['variant']=='retained')),
        reference_episodes=8, privileged_reference_intervals=sum(len(arrays[e['array_keys']['commands']]) for e in result['witnesses']),
        reference_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in result['witnesses']),
        modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for d in decision_rows),
        unsuccessful_optimizer_terminations=sum(not d['optimizer_success'] for d in decision_rows),
        fallback_decisions=sum(d['fallback'] for d in decision_rows), neural_fits=0,
    )
    assert resources['acquisition_intervals'] == acquisition['resources']['physical_intervals'] == 10240
    assert resources['acquisition_scalar_camera_draws'] == acquisition['resources']['scalar_noise_draws'] == 20560
    assert resources['acquisition_plant_rhs_evaluations'] == acquisition['resources']['plant_rhs_evaluations']
    assert resources['ordinary_intervals'] == result['resources']['ordinary_intervals'] == 3136
    assert resources['ordinary_camera_read_opportunities'] == result['resources']['ordinary_camera_read_opportunities'] == 3152
    assert resources['distinct_scalar_control_noise_draws'] == result['resources']['distinct_scalar_control_noise_draws'] == 3152
    assert resources['retained_actual_row_events'] == result['resources']['retained_actual_row_events'] == 720
    assert resources['retained_member_append_copies'] == result['resources']['retained_member_append_copies'] == 2160
    assert resources['privileged_reference_intervals'] == result['resources']['privileged_reference_intervals'] == 1568
    for key in ('ordinary_plant_rhs_evaluations', 'modeled_value_gradient_calls', 'unsuccessful_optimizer_terminations', 'fallback_decisions'):
        assert resources[key] == result['resources'][key]
    resources.update(acquisition_simulated_seconds=DT*resources['acquisition_intervals'],
                     ordinary_simulated_seconds=DT*resources['ordinary_intervals'],
                     ordinary_scalar_noise_opportunities=2*resources['ordinary_camera_read_opportunities'],
                     privileged_reference_seconds=DT*resources['privileged_reference_intervals'])
    audits = []
    if require_audits:
        audits.append(verify_audit(ordinary_audit, rep, artifacts, [hashes[k] for k in ('results', 'fit', 'ordinary')]))
        audits.append(verify_audit(physics_audit, rep, artifacts, [hashes[k] for k in ('results', 'arrays', 'acquisition')]))
    record = dict(replicate=rep, artifact_paths={key: str(path.resolve()) for key, path in p.items()},
                  artifact_sha256=hashes, selected_local_candidate=selected,
                  selected_configuration=dict(neighbors=MENU[selected][0], ridge=MENU[selected][1]),
                  validation_member_mse=validation_scores, means=means, heldout_forecasts=forecast_metrics,
                  resources=resources, audit_gates=audits,
                  acquisition_wall_seconds=float(acquisition['wall_seconds']), fit_wall_seconds=float(fit['wall_seconds']),
                  study_invocation_wall_seconds=float(result['study_invocation_wall_seconds_excluding_separate_acquisition']))
    return record, dependencies, catalog, phases, decision_rows, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix-template', default=str(ROOT/'cartesian_stream_v17_rep{rep}'))
    parser.add_argument('--output', default=str(ROOT/'cartesian_stream_v17_study_aggregate_results.json'))
    parser.add_argument('--require-audits', action='store_true')
    parser.add_argument('--ordinary-audit-template', default=str(ROOT/'audit_cartesian_stream_v17_study_rep{rep}_results.json'))
    parser.add_argument('--physics-audit-template', default=str(ROOT/'audit_cartesian_stream_v17_rep{rep}_study_physics_results.json'))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError('refusing to overwrite aggregate')
    artifacts = {}; bind(__file__, artifacts)
    corpora, all_phases, decisions, all_failures = [], [], [], []
    catalog = {name: [] for name in METHODS}; common_dependencies = None
    for rep in REPLICATES:
        corpus, common_dependencies, entries, phases, rows_, failures = load_corpus(
            format_path(args.prefix_template, rep), rep, artifacts, common_dependencies, args.require_audits,
            format_path(args.ordinary_audit_template, rep), format_path(args.physics_audit_template, rep))
        corpora.append(corpus); all_phases.extend(phases); decisions.extend(rows_); all_failures.extend(failures)
        for name in METHODS: catalog[name].extend(entries[name])
    scalar_keys = ('task_cost_18sec', 'first_goal_cost', 'later_goals_total_cost', 'later_goal_mean_cost',
                   'combined_charged_cost', 'warmup_motion_effort_cost', 'task_integrated_squared_effort',
                   'warmup_integrated_squared_effort', 'combined_integrated_squared_effort',
                   'precision_fraction', 'first_precision_fraction', 'later_precision_fraction',
                   'phase_costs', 'phase_precision_fractions')
    summary = {}
    for name in METHODS:
        summary[name] = {key: cluster([c['means'][name][key] for c in corpora]) for key in scalar_keys}
        summary[name].update(episodes=64, precision_denominator=192, first_precision_denominator=64,
                             later_precision_denominator=128,
                             precision_count=sum(c['means'][name]['precision_count'] for c in corpora),
                             first_precision_count=sum(c['means'][name]['first_precision_count'] for c in corpora),
                             later_precision_count=sum(c['means'][name]['later_precision_count'] for c in corpora))
    paired = {key: cluster([np.asarray(c['means']['frozen'][key])-np.asarray(c['means']['retained'][key])
                            for c in corpora]) for key in scalar_keys}
    forecasts = {name: {key: cluster([c['heldout_forecasts'][name][key] for c in corpora])
                        for key in ('member_coordinate_mse', 'ensemble_mean_coordinate_mse', 'member_mse_by_future_sample')}
                 for name in ('one_block', 'six_block')}
    optimizer = {}
    for variant in VARIANTS:
        rows_ = [d for d in decisions if d['variant'] == variant]
        optimizer[variant] = dict(
            decisions=len(rows_), unsuccessful_terminations=sum(not d['optimizer_success'] for d in rows_),
            fallback_decisions=sum(d['fallback'] for d in rows_),
            status_counts=dict(Counter(str(d['optimizer_status']) for d in rows_)),
            message_counts=dict(Counter(d['optimizer_message'] for d in rows_)),
            projected_gradient_quantiles=quantiles([d['projected_gradient_max'] for d in rows_]),
            projected_gradient_corpus_maxima=[max(d['projected_gradient_max'] for d in rows_ if d['replicate']==r) for r in REPLICATES],
            worst_projected_gradients=sorted(rows_, key=lambda d:d['projected_gradient_max'], reverse=True)[:20],
            total_modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for d in rows_),
            summed_choose_wall_seconds=sum(d['choose_wall_seconds'] for d in rows_),
        )
    totals = {key: int(sum(c['resources'][key] for c in corpora)) if all(isinstance(c['resources'][key], (int, np.integer)) for c in corpora)
              else float(sum(c['resources'][key] for c in corpora)) for key in corpora[0]['resources']}
    totals.update(acquired_corpora=8, fixed_physical_body_families=1,
                  total_physical_intervals_acquisition_ordinary_reference=(totals['acquisition_intervals']+totals['ordinary_intervals']+totals['privileged_reference_intervals']),
                  distinct_scalar_rng_draws_acquisition_and_control=(totals['acquisition_scalar_camera_draws']+totals['distinct_scalar_control_noise_draws']))
    result = dict(
        status=('COMPLETE EIGHT-CORPUS AGGREGATE WITH BOTH INDEPENDENT AUDIT GATES VERIFIED'
                if args.require_audits else 'COMPLETE DIAGNOSTIC AGGREGATE; NOT PUBLICATION ELIGIBLE WITHOUT AUDIT GATES'),
        publication_eligible=bool(args.require_audits), publication_requires_completed_audits=True,
        analysis_source_sha256=sha(__file__), artifact_sha256=artifacts,
        frozen_common_dependency_sha256=common_dependencies, replicates=list(REPLICATES),
        interpretation='Every primary mean/difference first averages within corpus. SE is descriptive across eight corpus means; '
                       'paired frozen-minus-retained SE is computed from eight paired differences, never marginal SEs. '
                       'One fixed body,32 fresh control preparations,64 paired noise paths,128 ordinary method episodes and8 acquired corpora; acquisition has320 separate trajectories. '
                       'Whole closed-loop append effect, not a per-row gain or universal learning claim.',
        corpora=corpora, summary=summary, paired_frozen_minus_retained=paired,
        ordinary_heldout_forecasts=forecasts,
        selected_local_menu_indices=[c['selected_local_candidate'] for c in corpora],
        selected_local_menu_counts=dict(Counter(str(c['selected_local_candidate']) for c in corpora)),
        menu=[dict(neighbors=n, ridge=r) for n, r in MENU], optimizer=optimizer, resources=totals,
        failures=dict(precision_failures=all_failures,
                      unsuccessful_optimizer_terminations=[d for d in decisions if not d['optimizer_success']],
                      fallback_decisions=[d for d in decisions if d['fallback']]),
        worst_episodes={name: sorted(catalog[name], key=lambda e:e['task_cost_18sec'], reverse=True)[:20] for name in METHODS},
        worst_goal_phases={name: sorted((p for p in all_phases if p['variant']==name), key=lambda p:p['cost'], reverse=True)[:20] for name in METHODS},
        episode_catalog=catalog,
        timing=dict(per_corpus_acquisition_seconds=[c['acquisition_wall_seconds'] for c in corpora],
                    per_corpus_fit_seconds=[c['fit_wall_seconds'] for c in corpora],
                    per_corpus_study_invocation_seconds=[c['study_invocation_wall_seconds'] for c in corpora],
                    interpretation='Process invocation times; sums are not parallel elapsed time or physical interaction delay.'),
        limitations=['Sampled sustained last10endpoint windows, not continuous-time guarantees.',
                     'Reference uses privileged exact state/IK and is not an equally informed learned baseline.',
                     'Ordinary heldout observed forecasts do not establish accurate action contrasts; unchanged weak pilot fork evidence remains.',
                     'No new neural fits or comparison with full modern methods occurs in this retention study.',
                     'Planning latency is reported separately and not charged as physical delay; no real-time claim.',
                     'All precision/optimizer failures retained; no p-values, outcome filtering or added runs.'],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('saved', output, 'publication_eligible', result['publication_eligible'])


if __name__ == '__main__':
    main()
