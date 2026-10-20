#!/usr/bin/env python3
"""Independent physics/statistics/decision audit of compact v14 relation learning.

No canonical module imported. Hamiltonian vectorized RK4 at32/64 substeps checks
all rest pulses; eigenspectrum stationarity and active-set2D box solutions check
fits/actions independently. Near-zero spectral signs are treated numerically.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
FACTOR = .5 * .1**2
VARIANCE = 2 * 1e-4**2 / FACTOR**2
LABELS = ['constant_forward', 'context_forward', 'naive_inverse', 'corrected_inverse']
WIDTHS, RIDGES = [.5, 1., 2., 4.], [1e-6, 1e-4, .01, .1]
ANGLES = np.linspace(-math.pi, math.pi, 8, endpoint=False)
CENTERS = np.array([(a, b) for a in ANGLES for b in ANGLES])


def hamiltonian(q, u, steps):
    q, u = np.atleast_2d(q), np.atleast_2d(u)
    y = np.c_[q, np.zeros_like(q)]
    def rhs(z):
        c, s = np.cos(z[:, 1]), np.sin(z[:, 1])
        determinant = 2 - c*c
        v1 = (z[:, 2] - (1 + c) * z[:, 3]) / determinant
        v2 = (-(1 + c) * z[:, 2] + (3 + 2*c) * z[:, 3]) / determinant
        return np.c_[v1, v2, u[:, 0], u[:, 1] - s * (v1*v1 + v1*v2)]
    dt = .1 / steps
    for _ in range(steps):
        a = rhs(y)
        b = rhs(y + .5 * dt * a)
        c = rhs(y + .5 * dt * b)
        d = rhs(y + dt * c)
        y += dt * (a + 2*b + 2*c + d) / 6
    c = np.cos(y[:, 1])
    determinant = 2 - c*c
    v = np.c_[(y[:, 2] - (1 + c) * y[:, 3]) / determinant,
              (-(1 + c) * y[:, 2] + (3 + 2*c) * y[:, 3]) / determinant]
    return np.c_[y[:, :2], v]


def features(q, width):
    q = np.atleast_2d(q)
    if width is None:
        return np.ones((len(q), 1))
    # Chordal geometry independently expressed via the cosine identity.
    chord_squared = np.sum(2 - 2*np.cos(q[:, None] - CENTERS[None]), axis=-1)
    return np.c_[np.ones(len(q)), np.exp(-chord_squared / (2*width**2))]


def learned_map(model, q):
    phi = features(q, model['width'])
    W = np.array(model['coefficients']).reshape(phi.shape[1], 2, 2)
    maps = np.einsum('nf,fio->noi', phi, W)
    if 'inverse' in model['label']:
        inverse = maps.copy()
        maps = np.array([np.linalg.pinv(K, rcond=1e-8) for K in inverse])
        return maps, inverse
    return maps, None


def box_optimum(B, target):
    candidates = [np.linalg.lstsq(B, target, rcond=None)[0]]
    for index in [0, 1]:
        other = 1 - index
        for fixed in [-1., 1.]:
            u = np.zeros(2)
            u[index] = fixed
            norm = B[:, other] @ B[:, other]
            u[other] = np.clip(B[:, other] @ (target - fixed*B[:, index]) / norm, -1., 1.) if norm else 0.
            candidates.append(u)
    candidates += [np.array([a, b]) for a in [-1., 1.] for b in [-1., 1.]]
    candidates = [u for u in candidates if np.max(np.abs(u)) <= 1. + 1e-12]
    return min(float(np.sum((B @ u - target)**2)) for u in candidates)


def stats(values):
    a = np.array(values)
    return dict(mean=float(np.mean(a)), mc_se=float(np.std(a, ddof=1)/np.sqrt(len(a))),
        median=float(np.median(a)), q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9)))


def main():
    result = json.loads((ROOT / 'coupled_learning_v14_two_reading_pilot_results.json').read_text())
    archive_path = ROOT / 'coupled_learning_v14_two_reading_pilot_data.npz'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == result['meta']['data_sha256']
    data = np.load(archive_path)
    errors = dict(physics32vs64=0., physics64vsDOP=0., measurement_RNG=0., response=0., spectrum=0.,
        scale=0., regularized_stationarity=0., validation=0., unregularized_residual=0.,
        map_condition=0., inverse_singular_values=0., bounded_cost=0., bounded_optimum_gap=0.,
        task_score=0., predictive_score=0., predictions=0., summary=0., oracle_witness=0.)
    relative = dict.fromkeys(errors, 0.)
    def check(key, a, b):
        a, b = np.asarray(a), np.asarray(b)
        delta = np.abs(a - b)
        errors[key] = max(errors[key], float(np.max(delta)))
        relative[key] = max(relative[key], float(np.max(delta / np.maximum(1., np.maximum(np.abs(a), np.abs(b))))))
    seeds = set()
    def register(seed):
        assert seed not in seeds
        seeds.add(seed)
    counts = dict(acquisition_pulses=0, predictive_pulses=0, task_pulses=0, oracle_witnesses=0, candidates=0, bounded_decisions=0)
    candidate_diagnostics = {label:dict(negative_validation_scores=0, candidates_with_negative_Gram_eigenvalues=0, maximum_floored_eigenvalues=0) for label in LABELS}
    solver_failures = 0
    oracle_q = np.array([row['context'] for row in result['oracle_feasibility']])
    oracle_u = np.array([row['command'] for row in result['oracle_feasibility']])
    oracle_z = np.array([row['goal'] for row in result['oracle_feasibility']])
    oracle32, oracle64 = hamiltonian(oracle_q, oracle_u, 32), hamiltonian(oracle_q, oracle_u, 64)
    check('physics32vs64', oracle32, oracle64)
    expected_context = np.array([(a, b) for a in [-2.7, -1.1, .4, 2.2] for b in [-math.pi, -2.1, -1.2, -.4, 0., .4, 1.2, 2.1]])
    desired = np.array([[.1, 0.], [0., .1], [.1, -.1], [-.1, .1], [.05, .1]])
    assert np.array_equal(oracle_q, np.repeat(expected_context, 5, axis=0))
    assert np.array_equal(oracle_z, np.tile(desired, (32, 1)))
    assert len(set(map(tuple, expected_context))) == 32
    for i, row in enumerate(result['oracle_feasibility']):
        assert row['success'] and np.max(np.abs(oracle_u[i])) <= 1.
        residual = np.linalg.norm((oracle64[i, :2] - oracle_q[i]) / FACTOR - oracle_z[i])
        check('oracle_witness', residual, row['feasibility_error_norm'])
        assert residual < 1e-7
        counts['oracle_witnesses'] += 1

    def dataset(row, key, role, count, offset, narrow):
        rng = np.random.default_rng(row['seed'] + offset)
        register(row['seed'] + offset)
        q = np.c_[rng.uniform(-math.pi, math.pi, count), rng.uniform(-.4 if narrow else -math.pi, .4 if narrow else math.pi, count)]
        u = rng.uniform(-1., 1., (count, 2))
        source = {name:data[key + '_' + role + '_' + name] for name in ['known_preparation', 'commands', 'states_evaluator_only', 'observed_initial', 'observed_final', 'response']}
        assert np.array_equal(q, source['known_preparation']) and np.array_equal(u, source['commands'])
        states32, states64 = hamiltonian(q, u, 32), hamiltonian(q, u, 64)
        check('physics32vs64', states32, states64)
        check('physics64vsDOP', states64, source['states_evaluator_only'])
        e0, e1 = rng.normal(0, 1e-4, (count, 2)), rng.normal(0, 1e-4, (count, 2))
        check('measurement_RNG', q + e0, source['observed_initial'])
        check('measurement_RNG', source['states_evaluator_only'][:, :2] + e1, source['observed_final'])
        check('response', (source['observed_final'] - source['observed_initial'])/FACTOR, source['response'])
        counts['acquisition_pulses' if role == 'train' else 'predictive_pulses'] += count
        return source

    diagnostics = dict(partition={}, q1_command_RMS={})
    for row in result['runs']:
        condition, replicate = row['condition'], row['replicate']
        assert row['seed'] == 22461020 + 1000000000 * condition + 1000000 * replicate
        key = f'c{condition}_s{replicate}'
        acquisition = dataset(row, key, 'train', 640, 100000, bool(condition))
        predictive = dataset(row, key, 'predictive', 128, 600000, False)
        q, u, response = acquisition['known_preparation'], acquisition['commands'], acquisition['response']
        for label in LABELS:
            candidates = row['fit_records'][label]['candidates']
            assert len(candidates) == (4 if label == 'constant_forward' else 16)
            index = 0
            for width in [None] if label == 'constant_forward' else WIDTHS:
                phi, validation_phi = features(q[:512], width), features(q[512:], width)
                inverse = 'inverse' in label
                source, target = (response[:512], u[:512]) if inverse else (u[:512], response[:512])
                X = (phi[:, :, None] * source[:, None, :]).reshape(512, -1)
                raw = X.T @ X
                correction = VARIANCE * np.kron(phi.T @ phi, np.eye(2)) if label == 'corrected_inverse' else np.zeros_like(raw)
                gram = raw - correction
                eigenvalues, vectors = np.linalg.eigh(gram)
                raw_min = np.linalg.eigvalsh(raw)[0]
                scale = max(np.trace(raw)/len(raw), 1e-12)
                rhs = X.T @ target
                psd = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
                for ridge in RIDGES:
                    candidate = candidates[index]
                    assert candidate['label'] == label and candidate['width'] == width and candidate['ridge'] == ridge
                    W = np.array(candidate['coefficients'])
                    assert np.all(np.isfinite(W)) and candidate['numerical_finite']
                    check('spectrum', eigenvalues, candidate['corrected_gram_eigenvalues'])
                    check('spectrum', raw_min, candidate['raw_gram_min_eigenvalue'])
                    check('scale', scale, candidate['regularization_scale'])
                    stored = np.array(candidate['corrected_gram_eigenvalues'])
                    assert candidate['floored_eigenvalues'] == int(np.sum(stored < 0.))
                    tolerance = 1e-9 * max(1., float(np.max(np.abs(eigenvalues))))
                    assert np.sum(eigenvalues < -tolerance) <= candidate['floored_eigenvalues'] <= np.sum(eigenvalues < tolerance)
                    # Correct stationarity is for PSD-projected Gram plus ridge,
                    # not the source's intentionally unregularized residual.
                    residual = (psd + ridge*scale*np.eye(len(gram))) @ W - rhs
                    check('regularized_stationarity', np.linalg.norm(residual)/(1 + np.linalg.norm(rhs)), 0.)
                    check('unregularized_residual', np.linalg.norm(gram @ W - rhs), candidate['training_normal_equation_residual'])
                    matrix = np.einsum('nf,fio->noi', validation_phi, W.reshape(phi.shape[1], 2, 2))
                    val_input, val_target = (response[512:], u[512:]) if inverse else (u[512:], response[512:])
                    predicted = np.einsum('noi,ni->no', matrix, val_input)
                    loss = np.sum((predicted - val_target)**2, axis=1)
                    if inverse:
                        loss -= VARIANCE * np.sum(matrix**2, axis=(1, 2))
                    check('validation', loss.mean(), candidate['validation_score'])
                    diagnostic = candidate_diagnostics[label]
                    diagnostic['negative_validation_scores'] += candidate['validation_score'] < 0.
                    diagnostic['candidates_with_negative_Gram_eigenvalues'] += candidate['floored_eigenvalues'] > 0
                    diagnostic['maximum_floored_eigenvalues'] = max(diagnostic['maximum_floored_eigenvalues'], candidate['floored_eigenvalues'])
                    counts['candidates'] += 1
                    index += 1
            selected = int(np.argmin([candidate['validation_score'] for candidate in candidates]))
            assert selected == row['fit_records'][label]['selected_index']
            assert row['models'][label] == candidates[selected]
        rng = np.random.default_rng(row['seed'] + 500000)
        register(row['seed'] + 500000)
        task_q = np.array([task['known_preparation'] for task in row['tasks']])
        task_z = np.array([task['desired_normalized_displacement'] for task in row['tasks']])
        assert np.array_equal(task_q, oracle_q) and np.array_equal(task_z, oracle_z)
        # Source measurement order is task-major, then all five labels.
        noises = {}
        for i in range(160):
            for label in LABELS + ['oracle_endpoint']:
                noises[i, label] = (rng.normal(0, 1e-4, 2), rng.normal(0, 1e-4, 2))
        for label in LABELS + ['oracle_endpoint']:
            episodes = [task['controllers'][label] for task in row['tasks']]
            commands = np.array([episode['command'] for episode in episodes])
            assert np.max(np.abs(commands)) <= 1. + 1e-12
            states32, states64 = hamiltonian(task_q, commands, 32), hamiltonian(task_q, commands, 64)
            check('physics32vs64', states32, states64)
            check('physics64vsDOP', states64, [episode['state_evaluator_only'] for episode in episodes])
            if label != 'oracle_endpoint':
                B, K = learned_map(row['models'][label], task_q)
            else:
                assert np.array_equal(commands, oracle_u)
            for i, episode in enumerate(episodes):
                stored = np.array(episode['state_evaluator_only'])
                e0, e1 = noises[i, label]
                check('measurement_RNG', task_q[i] + e0, episode['observed_initial'])
                check('measurement_RNG', stored[:2] + e1, episode['observed_final'])
                true_response = (states64[i, :2] - task_q[i])/FACTOR
                check('response', true_response, episode['response_evaluator_only'])
                check('task_score', np.sum((true_response - task_z[i])**2), episode['squared_error'])
                assert episode['saturated'] == bool(np.any(np.abs(commands[i]) > .999999))
                if label != 'oracle_endpoint':
                    check('map_condition', np.linalg.cond(B[i]), episode['response_condition'])
                    if K is None:
                        assert episode['inverse_relation_singular_values'] is None
                    else:
                        check('inverse_singular_values', np.linalg.svd(K[i], compute_uv=False), episode['inverse_relation_singular_values'])
                    cost = np.sum((B[i] @ commands[i] - task_z[i])**2)
                    check('bounded_cost', cost, episode['bounded_solver']['cost'])
                    check('bounded_optimum_gap', cost, box_optimum(B[i], task_z[i]))
                    solver_failures += not episode['bounded_solver']['success']
                    counts['bounded_decisions'] += 1
                else:
                    assert episode['bounded_solver'] is None and episode['response_condition'] == 0.
                counts['task_pulses'] += 1
            inside = np.abs(task_q[:, 1]) <= .4 + 1e-12
            assert inside.sum() == 60 and (~inside).sum() == 100
            key2 = f'condition{condition}:{label}'
            diagnostics['partition'].setdefault(key2, []).append(dict(inside_mse=float(np.mean([episodes[i]['squared_error'] for i in np.where(inside)[0]])),
                outside_mse=float(np.mean([episodes[i]['squared_error'] for i in np.where(~inside)[0]]))))
            grouped = commands.reshape(4, 8, 5, 2)
            # Euclidean RMS deviation from four-q1 command mean, averaged over
            # q2/goal/seed. This is distinct from pairwise or component RMS.
            deviation = grouped - np.mean(grouped, axis=0, keepdims=True)
            diagnostics['q1_command_RMS'].setdefault(key2, []).append(float(np.sqrt(np.mean(np.sum(deviation**2, axis=-1)))))
        true_response = (predictive['states_evaluator_only'][:, :2] - predictive['known_preparation']) / FACTOR
        for label in LABELS:
            B, _ = learned_map(row['models'][label], predictive['known_preparation'])
            prediction = np.einsum('nij,nj->ni', B, predictive['commands'])
            error = np.sum((prediction - true_response)**2, axis=1)
            record = row['predictive_transfer'][label]
            check('predictions', prediction, record['predictions'])
            check('predictive_score', error, record['errors'])
            check('predictive_score', np.mean(error), record['mean_squared_error'])
            check('map_condition', [np.linalg.cond(matrix) for matrix in B], record['response_conditions'])
    for cell in result['summary']:
        rows = [row for row in result['runs'] if row['condition'] == cell['condition']]
        for label, summary in cell['models'].items():
            losses = [np.mean([task['controllers'][label]['squared_error'] for task in row['tasks']]) for row in rows]
            for key, value in stats(losses).items():
                check('summary', value, summary['task_mse'][key])
            assert summary['saturations'] == sum(task['controllers'][label]['saturated'] for row in rows for task in row['tasks'])
            assert summary['trials'] == 1280
            if label in LABELS:
                for key, value in stats([row['predictive_transfer'][label]['mean_squared_error'] for row in rows]).items():
                    check('summary', value, summary['predictive_mse'][key])
        for label in ['constant_forward', 'naive_inverse', 'corrected_inverse']:
            difference = [np.mean([task['controllers'][label]['squared_error'] - task['controllers']['context_forward']['squared_error'] for task in row['tasks']]) for row in rows]
            for key, value in stats(difference).items():
                check('summary', value, cell['paired'][label + '_minus_context_forward'][key])
    assert counts == dict(acquisition_pulses=10240, predictive_pulses=2048, task_pulses=12800, oracle_witnesses=160, candidates=832, bounded_decisions=10240)
    assert result['meta']['response_noise_variance'] == VARIANCE
    for key in ['acquisition_pulse_episodes', 'predictive_test_episodes', 'task_test_episodes']:
        assert result['meta'][key] == {'acquisition_pulse_episodes':10240, 'predictive_test_episodes':2048, 'task_test_episodes':12800}[key]
    assert result['meta']['oracle_targets'] == 160
    assert result['meta']['oracle_feasibility_solver_rollouts'] == 3 * sum(row['nfev'] for row in result['oracle_feasibility']) + 160
    for key in errors:
        assert relative[key] < 1e-6, (key, errors[key], relative[key])
    assert errors['physics32vs64'] < 1e-8 and errors['physics64vsDOP'] < 1e-8
    diagnostics['partition'] = {key:{'inside_mse':stats([row['inside_mse'] for row in rows]), 'outside_mse':stats([row['outside_mse'] for row in rows])} for key, rows in diagnostics['partition'].items()}
    diagnostics['q1_command_RMS'] = {key:stats(rows) for key, rows in diagnostics['q1_command_RMS'].items()}
    return dict(status='INDEPENDENTLY VERIFIED COMPACT REST-CONTEXT LEARNING TEST', counts=counts,
        artifact_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['coupled_learning_v14_two_reading_pilot.py', 'coupled_learning_v14_two_reading_pilot_design.md', 'coupled_learning_v14_two_reading_pilot_results.json', 'coupled_learning_v14_two_reading_pilot_data.npz']},
        globally_disjoint_acquisition_measurement_prediction_role_seeds=len(seeds),
        independent_physics='Hamiltonian(q,p), vectorized RK4 at32/64 steps over.1',
        maximum_absolute_errors=errors, maximum_errors_scaled_by_max1_magnitude=relative,
        bounded_solver_failure_count=solver_failures, candidate_regularization_diagnostics=candidate_diagnostics,
        secondary_diagnostics=diagnostics,
        q1_diagnostic_definition='For each seed, root mean squared Euclidean command deviation from its mean over four q1 values, averaging 4*8*5 task entries; then mean/MCSE across eight seeds. Not root after pooling seeds.',
        scope=[
            'No canonical implementation imported; all rest mechanics, measurement RNG, data, fits/validation, boxoptimality and heldout scores independently checked.',
            'Exact BOTHencoder preparation context and zero velocity are supplied facts; this is not inference from noisy current context or continuousfeedback.',
            'Exact q also reveals initial measurement noise o0-q; using (o1-q)/FACTOR could halve response covariance. Specified methods all use differenced response, not all available information efficiently.',
            'Forward/inverse feature layouts and knownnoiseGram correction validated; indefiniteGrams are projectedPSD thenridged.',
            'Unregularized G*W-rhs field is not a solverfailure or stationarity test for the projected/ridged estimator.',
            'Inverse corrected validation is unbiased only for fixed candidate on independent noise; selected minimum may be negative/optimistic.',
            'All methods share complete physical fit/validation data; common bounded displacementmetric removes commandclipping versus outputmetric confound.',
            'Naiveinverse is a descriptive bias baseline; calibratedinverse remains a serious same-data competing parameterization.',
            'Leading affinity and periodic topology supplied; finitepulse response neednot be exactlyaffine; no arbitraryfunction discovery or modelnecessity claim.',
            'Inside/outside andq1invariance summaries are posthoc diagnostics, not new primarymetrics or new trainingdata.',
            'Oracle is privileged continuousendpoint feasibility witness; its1456rollouts are separate compute, not fair learnerdata.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report = main()
    print(json.dumps(report, indent=2, allow_nan=False))
