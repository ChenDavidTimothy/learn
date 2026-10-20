#!/usr/bin/env python3
"""Independent audit: no canonical modules imported; Hamiltonian RK4 physics."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PAST = np.array([-.2, -.15, -.1, -.05])
T, SD, SEED = .2, 1e-4, 50020261020
POSES = [(0., math.pi/2), (1., .3)]
VELOCITIES = [(.02, -.01), (.01, .02), (-.015, .005)]
LABELS = ['pose_only_zero', 'learned_relation_history', 'oracle_state']


def mass(q):
    q = np.atleast_2d(q)
    c = np.cos(q[:, 1])
    return np.array([[3+2*c, 1+c], [1+c, np.ones(len(q))]]).transpose(2, 0, 1)


def integrate(initial, u, time, steps=64):
    initial, u = np.atleast_2d(initial), np.atleast_2d(u)
    p = np.einsum('nij,nj->ni', mass(initial[:, :2]), initial[:, 2:])
    y = np.c_[initial[:, :2], p]
    def rhs(z):
        c, s = np.cos(z[:, 1]), np.sin(z[:, 1])
        det = 2-c*c
        v1 = (z[:, 2]-(1+c)*z[:, 3])/det
        v2 = (-(1+c)*z[:, 2]+(3+2*c)*z[:, 3])/det
        return np.c_[v1, v2, np.broadcast_to(u[:, 0], len(z)), u[:, 1]-s*(v1*v1+v1*v2)]
    dt = time/steps
    for _ in range(steps):
        a = rhs(y)
        b = rhs(y+.5*dt*a)
        c = rhs(y+.5*dt*b)
        d = rhs(y+dt*c)
        y += dt*(a+2*b+2*c+d)/6
    c = np.cos(y[:, 1]); det = 2-c*c
    v = np.c_[(y[:, 2]-(1+c)*y[:, 3])/det,
              (-(1+c)*y[:, 2]+(3+2*c)*y[:, 3])/det]
    return np.c_[y[:, :2], v]


def relation(model, q):
    angles = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    centers = np.array([(a, b) for a in angles for b in angles])
    distance = np.sum(2-2*np.cos(q-centers), axis=1)
    phi = np.r_[1., np.exp(-distance/(2*model['width']**2))]
    K = np.einsum('f,fio->oi', phi, np.array(model['coefficients']).reshape(65, 2, 2))
    return np.linalg.pinv(K, rcond=1e-8), K


def box_optimum(B, z):
    candidates = [np.linalg.lstsq(B, z, rcond=None)[0]]
    for i in range(2):
        j = 1-i
        for fixed in [-1., 1.]:
            u = np.zeros(2); u[i] = fixed
            norm = B[:, j] @ B[:, j]
            u[j] = np.clip(B[:, j] @ (z-fixed*B[:, i])/norm, -1., 1.) if norm else 0.
            candidates.append(u)
    candidates += [np.array([a, b]) for a in [-1., 1.] for b in [-1., 1.]]
    feasible = [u for u in candidates if np.max(np.abs(u)) <= 1.+1e-12]
    return min(feasible, key=lambda u: np.sum((B@u-z)**2))


def stats(a):
    return dict(mean=float(np.mean(a)), mc_se=float(np.std(a, ddof=1)/np.sqrt(len(a))),
                median=float(np.median(a)), q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9)))


def main():
    paths = ['history_context_v15_pilot.py', 'history_context_v15_pilot_results.json',
             'history_context_v15_question.md', 'coupled_learning_v14_results.json']
    result = json.loads((ROOT/paths[1]).read_text())
    parent = json.loads((ROOT/paths[3]).read_text())
    parent_audit = json.loads((ROOT/'audit_coupled_learning_v14_results.json').read_text())
    assert hashlib.sha256((ROOT/paths[3]).read_bytes()).hexdigest() == parent_audit['artifact_sha256'][paths[3]]
    models = [r['models']['corrected_inverse'] for r in parent['runs'] if r['condition'] == 0]
    assert len(models) == 8 and [r['replicate'] for r in parent['runs'] if r['condition'] == 0] == list(range(8))
    errors = dict(physics32vs64=0., past_states=0., past_forward=0., back_forward_error=0.,
                  oracle_endpoint=0., zero_endpoint=0., RNG=0., slope=0., command=0.,
                  bounded_objective=0., final_state=0., endpoint_error=0., endpoint_velocity=0., summary=0.)
    def check(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    assert len(result['worlds']) == 12 and len(result['runs']) == 1536
    expected_worlds = [(pi, vi, sign) for pi in range(2) for vi in range(3) for sign in [-1, 1]]
    oracle_replay_calls, oracle_replay_success = 0, []
    for w, (pi, vi, sign) in zip(result['worlds'], expected_worlds):
        q, v = np.array(POSES[pi]), sign*np.array(VELOCITIES[vi]); initial = np.r_[q, v]
        assert (w['pose_index'], w['velocity_index'], w['sign']) == (pi, vi, sign)
        assert np.array_equal(w['current_pose'], q) and np.array_equal(w['initial_velocity_evaluator_only'], v)
        past = []
        for time in PAST:
            a, b = integrate(initial, np.zeros(2), time, 32), integrate(initial, np.zeros(2), time, 64)
            check('physics32vs64', a, b); past.append(b[0])
        past = np.array(past)
        check('past_states', past, w['exact_past_states_evaluator_only'])
        # Forward from earliest past state also reconstructs all later samples.
        for i, time in enumerate(np.r_[PAST[1:], 0.]):
            state = integrate(past[0], np.zeros(2), time-PAST[0])[0]
            check('past_forward', state, past[i+1] if i < 3 else initial)
        check('back_forward_error', w['back_forward_error'], 0.)
        u = np.array(w['oracle_command']); assert np.max(np.abs(u)) <= 1.
        end = integrate(initial, u, T)[0]
        error = np.linalg.norm(end[:2]-q)
        check('oracle_endpoint', error, w['oracle_endpoint_error']); assert error < 1e-8
        check('zero_endpoint', integrate(initial, np.zeros(2), T)[0], w['zero_endpoint'])
        assert np.isclose(w['ballistic_surrogate_risk_reference_not_nonlinear_certificate'], T*T*np.sum(v*v), rtol=1e-15)
        # Independent optimizer endpoint/call diagnostic, not a proof of optimality.
        def residual(command):
            nonlocal oracle_replay_calls
            oracle_replay_calls += 1
            return integrate(initial, command, T)[0, :2]-q
        start = np.clip(mass(q)[0]@(-2*v/T), -.99, .99)
        fit = least_squares(residual, start, bounds=(-1., 1.), max_nfev=60, ftol=1e-13, xtol=1e-13, gtol=1e-12)
        assert fit.success and np.linalg.norm(residual(fit.x)) < 1e-8
        residual(fit.x) # canonical records a second final residual call
        oracle_replay_success.append(bool(fit.success))

    seeds, commands, initial, stored = set(), [], [], []
    sample_counts = np.zeros((8, 12), dtype=int)
    for row in result['runs']:
        mi, wi = row['model'], row['world']; ni = int(sample_counts[mi, wi]); sample_counts[mi, wi] += 1
        seed = SEED+1000000*mi+10000*wi+ni
        assert row['sensor_seed'] == seed and seed not in seeds; seeds.add(seed)
        w = result['worlds'][wi]; q = np.array(w['current_pose']); v = np.array(w['initial_velocity_evaluator_only'])
        assert np.array_equal(row['current_pose_and_goal_supplied'], q)
        observed = np.array(w['exact_past_states_evaluator_only'])[:, :2]+np.random.default_rng(seed).normal(0, SD, (4, 2))
        check('RNG', observed, row['observed_past_positions'])
        slope = PAST@(observed-q)/(PAST@PAST)
        check('slope', slope, row['estimated_slope'])
        B, K = relation(models[mi], q); target = -2*slope/T
        optimum = box_optimum(B, target)
        expected = [np.zeros(2), optimum, np.array(w['oracle_command'])]
        for label, wanted in zip(LABELS, expected):
            actual = np.array(row['outcomes'][label]['command']); assert np.max(np.abs(actual)) <= 1.+1e-12
            check('command', wanted, actual)
            if label == 'learned_relation_history':
                check('bounded_objective', np.sum((B@actual-target)**2), np.sum((B@optimum-target)**2))
            initial.append(np.r_[q, v]); commands.append(actual); stored.append(row['outcomes'][label])
    assert np.array_equal(sample_counts, np.full((8, 12), 16))
    a, b = integrate(initial, commands, T, 32), integrate(initial, commands, T, 64)
    check('physics32vs64', a, b)
    for state, start, out in zip(b, initial, stored):
        check('final_state', state, out['final_state_evaluator_only'])
        check('endpoint_error', np.sum((state[:2]-start[:2])**2), out['endpoint_squared_error'])
        check('endpoint_velocity', np.linalg.norm(state[2:]), out['endpoint_velocity_norm'])
    for label in LABELS:
        a = [np.mean([r['outcomes'][label]['endpoint_squared_error'] for r in result['runs'] if r['model'] == mi]) for mi in range(8)]
        for key, value in stats(a).items():check('summary', value, result['summary'][label]['endpoint_mse'][key])
        assert result['summary'][label]['trials'] == 1536

    # Enumerate actual v12/v13/v14 stream roles including virtual CEM and checks.
    legacy = {75071013}
    shared = json.loads((ROOT/'shared_experience_v12_results.json').read_text())
    for r in shared['runs']:
        s = r['seed']; assert s == 22161020+1000000000*r['plant_index']+1000000*r['replicate']
        for offset in [0, 10000000000]:
            legacy.add(s+offset)
            legacy.update(s+offset+100000+100*i+k for i in range(192) for k in range(2))
        legacy.update(s+500000+j for j in range(2))
        legacy.update([s+20000000000, s+30000000000])
        legacy.update(s+40000000000+j for j in range(2))
    for r in parent['runs']:
        legacy.update(r['seed']+offset for offset in [100000, 500000, 600000])
    assert not (seeds & legacy) and min(seeds) > max(legacy)
    excluded = json.loads((ROOT/'history_context_v15_seed_collision_pilot_results.json').read_text())
    excluded_seeds = {r['sensor_seed'] for r in excluded['runs']}
    overlap = sorted(excluded_seeds & legacy)
    v14_acquisition = {r['seed']+100000 for r in parent['runs']}
    v12_test = {r['seed']+500000+j for r in shared['runs'] for j in range(2)}
    assert len(overlap) == 24 and len(excluded_seeds & v14_acquisition) == 8 and len(excluded_seeds & v12_test) == 16
    assert result['meta']['seed'] == SEED and result['meta']['true_state_oracle_rollouts'] == 132
    assert result['meta']['actual_test_pulse_integrations'] == 4608
    assert max(errors.values()) < 1e-8, errors
    outcome_velocity = {label:dict(mean=float(np.mean([r['outcomes'][label]['endpoint_velocity_norm'] for r in result['runs']])),
                                    maximum=float(np.max([r['outcomes'][label]['endpoint_velocity_norm'] for r in result['runs']]))) for label in LABELS}
    return dict(status='INDEPENDENTLY VERIFIED PRELIMINARY HISTORY PILOT; NOT LEARNED SUMMARY SELECTION',
        artifact_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        parent_models='Eight canonical information-corrected broad v14 calibrated inverse fits, in replicate order; certified parent hash checked.',
        maximum_absolute_errors=errors,
        independent_oracle_replay=dict(successes=sum(oracle_replay_success), residual_calls=oracle_replay_calls,
                                       canonical_recorded_calls=132, exact_call_match=oracle_replay_calls == 132),
        stream_audit=dict(fresh_sensor_seeds=len(seeds), earlier_v12_v13_v14_role_seeds=len(legacy),
            earlier_maximum_seed=max(legacy), fresh_minimum_seed=min(seeds), fresh_maximum_seed=max(seeds),
            excluded_pilot_collisions=overlap, excluded_v14_acquisition_collisions=8, excluded_v12_heldout_collisions=16),
        canonical_computation=dict(worlds=12, backward_integrations=48, return_checks=12, zero_reference_integrations=12,
            privileged_oracle_residual_calls=132, evaluation_history_samples=1536, future_pulse_integrations=4608,
            total_ODE_calls=4812, total_absolute_simulated_duration=958.8, scalar_past_coordinate_sensor_reads=12288,
            new_learned_model_training_samples=0),
        summary=result['summary'], final_velocity_diagnostic=outcome_velocity,
        limitations=[
            'Exact current pose and identical return goal are supplied. No history selection/compression is learned; anchored slope and cancellation arithmetic are supplied.',
            'Past trajectories are genuine zero-torque continuous histories. Backward integration is an environment construction, not controller access to future noise or true velocity.',
            'No true velocity or mechanics enters the ordinary history action. Privileged commands merely witness individual bounded endpoint feasibility.',
            'Zero torque is a diagnostic, not a globally optimal nonlinear pose-only policy or finite-T error lower bound.',
            'Summary MCSE uses eight acquired-parent-model clusters, each averaging twelve fixed worlds and sixteen fresh measurement draws; it is not population uncertainty over arbitrary worlds.',
            'Position return ends moving. No continuous feedback, stationkeeping, process noise, model retraining or autonomous summary revision is demonstrated.',
            'Source does not retain per-oracle nfev or bounded-solver status records. Independent endpoint and box-optimality checks validate saved commands, not every internal iteration.',
            'Excluded seed-collision run remains preserved; its earlier random streams are not counted as fresh independent evidence.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report = main()
    print(json.dumps(report, indent=2, allow_nan=False))
