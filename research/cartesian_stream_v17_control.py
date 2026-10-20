#!/usr/bin/env python3
"""Frozen moving-history witness/control pilot; external physical driver.

Ordinary modeled_loss/choose_action use only a selected ordinary predictor,
noisy observations, issued commands, current goal and previous selected plan.
Physical state, IK, mechanics and actual-score functions are evaluator-only.
Source must receive root review and independent preflight checks before running.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_limits

from cartesian_stream_v17_learn import DT, H, Predictor, digest, load_selected
from cartesian_stream_v17_acquire import SD, endpoint, jacobian
from coupled_context_v14_pilot import flow, mass

BLOCKS = 6
BOUND = .5
WARMUP_INTERVALS = 16
PHASE_INTERVALS = 60
PHASES = 3
CONTROL_INTERVALS = PHASES * PHASE_INTERVALS
TOTAL_INTERVALS = WARMUP_INTERVALS + CONTROL_INTERVALS
OPTIMIZER_OPTIONS = dict(maxiter=40, maxfun=80, maxls=20, ftol=1e-9, gtol=1e-6)


def json_ready(value):
    """Preserve nonfinite failure markers without invalid JSON or lost records."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class PlanningFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def modeled_loss(predictor: Predictor, observations, commands, goal, plan, prepared):
    """Pure ordinary mean-path loss; plan is a differentiable6x2 tensor."""
    paths = predictor.paths(observations, commands, plan, prepared=prepared)
    target = torch.tensor(np.asarray(goal), dtype=torch.float64)
    current = torch.tensor(np.asarray(observations)[-1], dtype=torch.float64)
    previous = torch.cat((current[None, None].expand(paths.shape[0], 1, 2),
                          paths[:, :-1]), dim=1)
    velocity = (paths - previous) / DT
    per_member = torch.sum((paths - target) ** 2 + .25 * velocity ** 2, dim=(1, 2))
    return torch.mean(per_member) + .01 * 2 * torch.sum(plan ** 2)


def choose_action(predictor: Predictor, observations, commands, goal, warmplan):
    """Choose using ordinary information only and the frozen common optimizer.

    Return first command, selected full plan, JSON diagnostics, and optional
    local fit arrays. No goal beyond the current one is an argument.
    """
    observations = np.asarray(observations, dtype=np.float64)
    commands = np.asarray(commands, dtype=np.float64).reshape(-1, 2)
    goal = np.asarray(goal, dtype=np.float64)
    warmplan = np.asarray(warmplan, dtype=np.float64)
    if warmplan.shape != (BLOCKS, 2) or goal.shape != (2,):
        raise ValueError('plan/goal must have shapes (6,2)/(2,)')
    if len(observations) != len(commands) + 1 or len(commands) < H:
        raise ValueError('inconsistent or insufficient ordinary causal history')
    if not all(np.all(np.isfinite(x)) for x in (observations, commands, goal, warmplan)):
        raise ValueError('ordinary inputs must be finite')
    if np.max(np.abs(warmplan)) > BOUND:
        raise ValueError('previous selected plan violates command bounds')

    started = time.perf_counter()
    prepared = predictor.prepare(observations, commands)
    local_arrays = None
    if predictor.kind == 'local':
        local_arrays = dict(
            coefficients=np.stack([m.detach().numpy().copy() for m in prepared[0]]),
            query=prepared[1].detach().numpy().copy(),
        )
    local_diagnostics = [
        {k: v for k, v in d.items() if k not in ('neighbor_indices', 'normalized_weights')}
        for d in prepared[2]
    ]
    evaluations = 0

    def value_gradient(flat):
        nonlocal evaluations
        evaluations += 1
        plan = torch.tensor(np.asarray(flat).reshape(BLOCKS, 2),
                            dtype=torch.float64, requires_grad=True)
        loss = modeled_loss(predictor, observations, commands, goal, plan, prepared)
        gradient, = torch.autograd.grad(loss, plan)
        return float(loss.detach()), gradient.detach().numpy().ravel().copy()

    shifted = np.vstack((warmplan[1:], np.zeros((1, 2))))
    zero = np.zeros((BLOCKS, 2))
    shifted_cost, shifted_gradient = value_gradient(shifted.ravel())
    zero_cost, zero_gradient = value_gradient(zero.ravel())
    shifted_finite = bool(np.isfinite(shifted_cost) and np.all(np.isfinite(shifted_gradient)))
    zero_finite = bool(np.isfinite(zero_cost) and np.all(np.isfinite(zero_gradient)))
    if shifted_finite and (not zero_finite or shifted_cost <= zero_cost):
        initial_name, initial, initial_cost = 'shifted_previous_plan', shifted, shifted_cost
    elif zero_finite:
        initial_name, initial, initial_cost = 'zero_plan', zero, zero_cost
    else:
        raise PlanningFailure('both initial ordinary plans are nonfinite', dict(
            shifted_cost=shifted_cost if np.isfinite(shifted_cost) else None,
            zero_cost=zero_cost if np.isfinite(zero_cost) else None,
            local_fit_diagnostics=local_diagnostics))

    record = dict(
        shifted_plan=shifted, shifted_cost=shifted_cost if np.isfinite(shifted_cost) else None,
        shifted_finite=shifted_finite, zero_plan=zero,
        zero_cost=zero_cost if np.isfinite(zero_cost) else None, zero_finite=zero_finite,
        initial_competitor=initial_name, initial_plan=initial.copy(), initial_cost=initial_cost,
        local_fit_diagnostics=local_diagnostics,
    )
    optimizer_started = time.perf_counter()
    result = None
    try:
        result = minimize(value_gradient, initial.ravel(), jac=True, method='L-BFGS-B',
                          bounds=[(-BOUND, BOUND)] * (2 * BLOCKS),
                          options=OPTIMIZER_OPTIONS.copy())
        candidate = np.asarray(result.x).reshape(BLOCKS, 2)
        feasible = bool(np.all(np.isfinite(candidate)) and np.all(candidate >= -BOUND)
                        and np.all(candidate <= BOUND))
        candidate_cost = float(result.fun)
        candidate_gradient = np.asarray(result.jac)
        candidate_finite = bool(np.isfinite(candidate_cost)
                                and np.all(np.isfinite(candidate_gradient)))
        accept = feasible and candidate_finite and candidate_cost <= initial_cost + 1e-12
        record['optimizer'] = dict(
            success=bool(result.success), status=int(result.status), message=str(result.message),
            nfev=int(result.nfev), njev=int(result.njev), nit=int(result.nit),
            final_plan=candidate.copy(),
            final_cost=candidate_cost if np.isfinite(candidate_cost) else None,
            final_feasible=feasible, final_finite=candidate_finite,
        )
    except Exception as error:
        # The frozen contract retains the finite initial competitor after an
        # unsuccessful optimization. Preserve the exception as a failure.
        candidate = initial
        accept = False
        record['optimizer'] = dict(success=False, status=None, message=str(error),
                                   error_type=type(error).__name__, nfev=None, njev=None, nit=None,
                                   final_plan=None, final_cost=None,
                                   final_feasible=False, final_finite=False)
    record['optimizer_wall_seconds'] = time.perf_counter() - optimizer_started
    selected = candidate.copy() if accept else initial.copy()
    selected_cost, selected_gradient = value_gradient(selected.ravel())
    if not np.isfinite(selected_cost) or not np.all(np.isfinite(selected_gradient)):
        raise PlanningFailure('selected ordinary plan is nonfinite', record)
    projected = selected.ravel() - np.clip(selected.ravel() - selected_gradient, -BOUND, BOUND)
    record.update(
        fallback_to_initial=not accept, selected_plan=selected.copy(), selected_cost=selected_cost,
        selected_gradient=selected_gradient.reshape(BLOCKS, 2),
        selected_projected_gradient=projected.reshape(BLOCKS, 2),
        selected_projected_gradient_max=float(np.max(np.abs(projected))),
        modeled_value_gradient_calls=evaluations,
        total_choose_wall_seconds=time.perf_counter() - started,
    )
    return selected[0].copy(), selected, record, local_arrays


# Everything below is an EXTERNAL physical/evaluation driver. It cannot be
# accessed through modeled_loss or choose_action and supplies them no true state.
def seeded_rng(path):
    return np.random.default_rng(np.random.SeedSequence(path))


def make_cases():
    cases = []
    for context_id in range(4):
        initial_path = [20261020, 17, 670, 0, context_id, 0]
        rng = seeded_rng(initial_path)
        theta = rng.uniform(-np.pi, np.pi)
        phi = rng.uniform(.4, np.pi - .4)
        sign = rng.choice([-1., 1.])
        speed = rng.uniform(.02, .12)
        direction = rng.uniform(-np.pi, np.pi)
        q = np.asarray([theta - sign * phi / 2, sign * phi])
        velocity = speed * np.asarray([np.cos(direction), np.sin(direction)])
        initial_state = np.r_[q, np.linalg.solve(jacobian(q), velocity)]
        warmup_path = [20261020, 17, 670, 0, context_id, 2]
        random_commands = seeded_rng(warmup_path).uniform(-BOUND, BOUND, (4, 2))
        warmup = np.vstack((np.zeros((8, 2)), np.repeat(random_commands, 2, axis=0)))
        for noise_repeat in range(2):
            noise_path = [20261020, 17, 670, 0, context_id, 1, noise_repeat]
            noise = seeded_rng(noise_path).normal(0, SD, (TOTAL_INTERVALS + 1, 2))
            cases.append(dict(context_id=context_id, noise_repeat=noise_repeat,
                              initial_seed_path=initial_path, warmup_seed_path=warmup_path,
                              noise_seed_path=noise_path, initial_state=initial_state.copy(),
                              warmup_commands=warmup.copy(), sensor_noise=noise))
    return cases


def goal_sequence(observed_warmup_endpoint):
    g0 = .95 * np.asarray(observed_warmup_endpoint)
    def rotate(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]]) @ g0
    return np.array([g0, rotate(.04), rotate(-.04)])


def closest_joint_target(q, original_goal):
    """External unit-link IK, both branches and nearest2pi lattice points."""
    original_goal = np.asarray(original_goal, dtype=np.float64)
    radius = float(np.linalg.norm(original_goal))
    clipped_radius = float(np.clip(radius, 0, 2))
    clipped_goal = original_goal.copy()
    if radius > 0:
        clipped_goal *= clipped_radius / radius
    angle = float(np.arctan2(clipped_goal[1], clipped_goal[0]))
    phi = float(np.arccos(np.clip((clipped_radius**2 - 2) / 2, -1, 1)))
    candidates = []
    for branch in (1., -1.):
        q2 = branch * phi
        # At radius0 the shoulder can be arbitrary; choose its nearest value.
        q1 = float(q[0]) if clipped_radius == 0 else angle - np.arctan2(np.sin(q2), 1 + np.cos(q2))
        base = np.asarray([q1, q2])
        winding = np.rint((q - base) / (2 * np.pi)).astype(np.int64)
        target = base + 2 * np.pi * winding
        candidates.append((float(np.sum((q-target)**2)), target, branch, winding))
    distance, target, branch, winding = min(candidates, key=lambda entry: entry[0])
    return target, dict(original_goal=original_goal, clipped_goal=clipped_goal,
                        clipped=bool(radius != clipped_radius), original_radius=radius,
                        clipped_radius=clipped_radius, branch=branch, winding=winding,
                        joint_distance_squared=distance, folded_goal=bool(clipped_radius == 0))


def witness_command(state, current_goal):
    q, velocity = state[:2], state[2:]
    target, diagnostics = closest_joint_target(q, current_goal)
    s = np.sin(q[1])
    h = np.asarray([-s*(2*velocity[0]*velocity[1] + velocity[1]**2), s*velocity[0]**2])
    unbounded = mass(q) @ (-(q-target) - 2*velocity) + h
    command = np.clip(unbounded, -BOUND, BOUND)
    diagnostics.update(joint_target=target, unbounded_command=unbounded,
                       saturated=bool(np.any(command != unbounded)))
    return command, diagnostics


def score_episode(states, commands, goals):
    """Evaluator-only actual integrated score and sustained precision windows."""
    positions = endpoint(states[:, :2])
    velocities = np.asarray([jacobian(z[:2]) @ z[2:] for z in states])
    squared_effort = np.sum(commands**2, axis=1)
    warmup_speed = np.sum(velocities[1:WARMUP_INTERVALS+1]**2, axis=1)
    warmup_effort = squared_effort[:WARMUP_INTERVALS]
    warmup = dict(seconds=WARMUP_INTERVALS*DT,
                  integrated_motion_penalty=float(DT*.25*np.sum(warmup_speed)),
                  integrated_effort_penalty=float(DT*.01*np.sum(warmup_effort)),
                  integrated_squared_effort=float(DT*np.sum(warmup_effort)))
    warmup['cost_without_unannounced_goal'] = (warmup['integrated_motion_penalty']
                                                + warmup['integrated_effort_penalty'])
    phases = []
    for phase, goal in enumerate(goals):
        first = WARMUP_INTERVALS + PHASE_INTERVALS*phase
        last = first + PHASE_INTERVALS
        errors = positions[first+1:last+1] - goal
        actual_velocity = velocities[first+1:last+1]
        position_terms = np.sum(errors**2, axis=1)
        speed_terms = np.sum(actual_velocity**2, axis=1)
        effort_terms = squared_effort[first:last]
        precision = bool(np.all(np.abs(errors[-10:]) <= .03)
                         and np.all(np.abs(actual_velocity[-10:]) <= .03))
        phases.append(dict(phase=phase, goal=goal, endpoint_start_index=first,
                           endpoint_end_index=last, seconds=PHASE_INTERVALS*DT,
                           integrated_position_error=float(DT*np.sum(position_terms)),
                           integrated_motion_penalty=float(DT*.25*np.sum(speed_terms)),
                           integrated_effort_penalty=float(DT*.01*np.sum(effort_terms)),
                           integrated_squared_effort=float(DT*np.sum(effort_terms)),
                           cost=float(DT*np.sum(position_terms+.25*speed_terms+.01*effort_terms)),
                           position_error_by_sample=errors, cartesian_velocity_by_sample=actual_velocity,
                           squared_effort_by_interval=effort_terms,
                           late_max_coordinate_position_error=float(np.max(np.abs(errors[-10:]))),
                           late_max_coordinate_velocity=float(np.max(np.abs(actual_velocity[-10:]))),
                           precision_success=precision))
    total_task = float(sum(p['cost'] for p in phases))
    return dict(warmup=warmup, phases=phases, task_cost=total_task,
                combined_charged_cost=total_task+warmup['cost_without_unannounced_goal'],
                physical_seconds=TOTAL_INTERVALS*DT,
                precise_goals=int(sum(p['precision_success'] for p in phases)))


def run_episode(case, predictor, label, progress):
    """External simulator. predictor=None means privileged witness only."""
    states = [case['initial_state'].copy()]
    observed = [endpoint(states[0][:2]) + case['sensor_noise'][0]]
    commands = []
    nfev = []
    decisions = []
    local_coefficients, local_queries = [], []
    record = dict(kind=label, context_id=case['context_id'], noise_repeat=case['noise_repeat'],
                  initial_seed_path=case['initial_seed_path'], warmup_seed_path=case['warmup_seed_path'],
                  noise_seed_path=case['noise_seed_path'], decisions=decisions,
                  local_library_initial_rows=([m.n_rows for m in predictor.members]
                                              if predictor is not None and predictor.kind == 'local' else None))
    progress.update(record=record, states=states, observed=observed, commands=commands,
                    nfev=nfev, local_coefficients=local_coefficients, local_queries=local_queries)

    def execute(command):
        final, calls = flow(states[-1][:2], states[-1][2:], command, DT)
        states.append(final)
        commands.append(np.asarray(command).copy())
        nfev.append(calls)
        observed.append(endpoint(final[:2]) + case['sensor_noise'][len(commands)])

    for command in case['warmup_commands']:
        execute(command)
    goals = goal_sequence(observed[-1])
    record['goals'] = goals
    previous_plan = np.zeros((BLOCKS, 2))
    for renewal in range(CONTROL_INTERVALS//2):
        phase = renewal // (PHASE_INTERVALS//2)
        current_goal = goals[phase]
        start_index = len(commands)
        if predictor is None:
            command, detail = witness_command(states[-1], current_goal)
            detail.update(renewal=renewal, phase=phase, time_index=start_index, command=command)
        else:
            command, previous_plan, detail, local_arrays = choose_action(
                predictor, observed, commands, current_goal, previous_plan)
            detail.update(renewal=renewal, phase=phase, time_index=start_index, command=command,
                          current_goal=current_goal.copy())
            if local_arrays is not None:
                detail['local_fit_array_index'] = len(local_coefficients)
                local_coefficients.append(local_arrays['coefficients'])
                local_queries.append(local_arrays['query'])
        decisions.append(detail)
        execute(command)
        execute(command)
        if predictor is not None:
            # Exactly one actual held-pair row, never hypothetical observations.
            before_rows = ([m.n_rows for m in predictor.members] if predictor.kind == 'local' else None)
            predictor.append_completed(observed, commands)
            if before_rows is not None:
                after_rows = [m.n_rows for m in predictor.members]
                assert after_rows == [n+1 for n in before_rows], 'one completed actual row must be appended'
                detail['local_library_rows_after_append'] = after_rows
    states, observed, commands, nfev = map(np.asarray, (states, observed, commands, nfev))
    record.update(score=score_episode(states, commands, goals),
                  plant_rhs_evaluations=int(np.sum(nfev)),
                  local_library_final_rows=([m.n_rows for m in predictor.members]
                                            if predictor is not None and predictor.kind == 'local' else None))
    arrays = dict(states_evaluator_only=states, observed=observed, commands=commands, plant_nfev=nfev)
    if local_coefficients:
        arrays['local_fit_coefficients'] = np.asarray(local_coefficients)
        arrays['local_fit_queries'] = np.asarray(local_queries)
    return record, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-prefix', default='research/cartesian_stream_v17_rep0')
    parser.add_argument('--ordinary', default='research/cartesian_stream_v17_rep0_ordinary.npz')
    parser.add_argument('--output-prefix', default='research/cartesian_stream_v17_rep0_control')
    args = parser.parse_args()
    prefix = Path(args.output_prefix)
    result_path = Path(str(prefix)+'_results.json')
    arrays_path = Path(str(prefix)+'_arrays.npz')
    failure_path = Path(str(prefix)+'_failure.json')
    if any(path.exists() for path in (result_path, arrays_path, failure_path)):
        raise FileExistsError('refusing to overwrite control artifacts')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    source_dir = Path(__file__).resolve().parent
    result = dict(
        status='IN PROGRESS: EXPLORATORY MOVING CONTROL; NO RELEASE',
        source_sha256=digest(__file__), ordinary_sha256=digest(args.ordinary),
        fit_sha256=digest(str(args.fit_prefix)+'_fit.json'),
        weights_sha256=digest(str(args.fit_prefix)+'_weights.npz'),
        dependency_sha256={name: digest(source_dir/name) for name in (
            'cartesian_stream_v17_learn.py', 'cartesian_stream_v17_local.py',
            'cartesian_stream_v17_acquire.py', 'coupled_context_v14_pilot.py',
            'cartesian_stream_v17_learning_design.md', 'cartesian_stream_v17_planner_contract.md')},
        scipy=scipy.__version__, torch=torch.__version__, numpy=np.__version__,
        optimizer_options=OPTIMIZER_OPTIONS.copy(), witnesses=[], ordinary_episodes=[],
        limitations='One acquired corpus/four initial preparations/eight paired noise paths. '
                    'Witnesses are privileged diagnostics, not equally informed learned competitors. '
                    'Sampled planner paths use noisy finite differences, not true velocity. '
                    'Physical execution does not charge planning latency; no realtime claim.',
    )
    saved_arrays = {}
    cases = make_cases()
    for i, case in enumerate(cases):
        for key in ('initial_state', 'warmup_commands', 'sensor_noise'):
            saved_arrays[f'case{i}_{key}'] = case[key]
    progress = {}

    def checkpoint():
        result['wall_seconds'] = time.perf_counter()-started
        np.savez_compressed(arrays_path, **saved_arrays)
        result['arrays_sha256'] = digest(arrays_path)
        result_path.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False)+'\n')

    def store_episode(record, arrays, category, index):
        keys = {}
        for name, value in arrays.items():
            key = f'{category}{index}_{name}'
            saved_arrays[key] = value
            keys[name] = key
        record['array_keys'] = keys
        result[category].append(record)
        checkpoint()

    try:
        with threadpool_limits(limits=1):
            # Complete all eight nominal authority witnesses before ordinary use.
            for i, case in enumerate(cases):
                progress.clear()
                record, arrays = run_episode(case, None, 'privileged_nominal_witness', progress)
                store_episode(record, arrays, 'witnesses', i)
                print('witness', i, 'completed', flush=True)
            for case_index, case in enumerate(cases):
                for kind in ('nonlinear', 'local'):
                    progress.clear()
                    load_started = time.perf_counter()
                    selected, fit_meta = load_selected(args.fit_prefix, args.ordinary)
                    assert fit_meta['source_sha256'] == result['dependency_sha256']['cartesian_stream_v17_learn.py']
                    assert fit_meta['local_source_sha256'] == result['dependency_sha256']['cartesian_stream_v17_local.py']
                    assert fit_meta['weights_sha256'] == result['weights_sha256']
                    assert fit_meta['design_sha256'] == result['dependency_sha256']['cartesian_stream_v17_learning_design.md']
                    predictor = selected[kind]
                    record, arrays = run_episode(case, predictor, kind, progress)
                    record.update(case_index=case_index, selected_configuration=predictor.config,
                                  selected_candidate_index=int(fit_meta['selected'][kind]),
                                  load_and_episode_wall_seconds=time.perf_counter()-load_started)
                    index = len(result['ordinary_episodes'])
                    store_episode(record, arrays, 'ordinary_episodes', index)
                    print(kind, 'case', case_index, 'completed', flush=True)
        ordinary = result['ordinary_episodes']
        witnesses = result['witnesses']
        assert len(ordinary) == 16 and len(witnesses) == 8
        result['resources'] = dict(
            ordinary_episodes=16, ordinary_phases=48, ordinary_decisions=16*CONTROL_INTERVALS//2,
            ordinary_intervals=16*TOTAL_INTERVALS, ordinary_simulated_seconds=16*TOTAL_INTERVALS*DT,
            ordinary_camera_read_opportunities=16*(TOTAL_INTERVALS+1),
            ordinary_scalar_noise_opportunities=16*(TOTAL_INTERVALS+1)*2,
            distinct_initial_preparations=4, independent_paired_noise_paths=8,
            distinct_scalar_noise_rng_draws=8*(TOTAL_INTERVALS+1)*2,
            warmup_intervals_ordinary=16*WARMUP_INTERVALS,
            privileged_witness_episodes=8, privileged_witness_intervals=8*TOTAL_INTERVALS,
            privileged_witness_seconds=8*TOTAL_INTERVALS*DT,
            new_training_experience_from_witness=0,
            ordinary_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in ordinary),
            witness_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in witnesses),
            modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in ordinary for d in e['decisions']),
            unsuccessful_optimizer_terminations=sum(not d['optimizer']['success'] for e in ordinary for d in e['decisions']),
            fallback_decisions=sum(d['fallback_to_initial'] for e in ordinary for d in e['decisions']),
            actual_local_rows_appended_per_member_per_local_episode=CONTROL_INTERVALS//2,
        )
        result['status'] = 'COMPLETE EXPLORATORY MOVING WITNESS/CONTROL PILOT; INDEPENDENT AUDIT PENDING'
        checkpoint()
    except BaseException as error:
        # Preserve the exact partial actual stream; never drop a failed episode.
        for name in ('states', 'observed', 'commands', 'nfev', 'local_coefficients', 'local_queries'):
            if name in progress:
                saved_arrays['failed_partial_'+name] = np.asarray(progress[name])
        failure = dict(status='FAILED/INTERRUPTED CONTROL; PARTIAL EXPERIENCE PRESERVED',
                       error_type=type(error).__name__, error=str(error),
                       active_episode=progress.get('record'), source_sha256=digest(__file__))
        if isinstance(error, PlanningFailure):
            failure['planning_diagnostics'] = error.diagnostics
        failure_path.write_text(json.dumps(json_ready(failure), indent=2, allow_nan=False)+'\n')
        result['status'] = failure['status']
        checkpoint()
        raise


if __name__ == '__main__':
    main()
