#!/usr/bin/env python3
"""Numerical fixtures only; no acquired/validation/test data are loaded."""
import hashlib
import json
from pathlib import Path
import numpy as np
from cartesian_stream_v17_local import LocalPredictor


def main():
    rng = np.random.default_rng(20261020661)
    contexts = rng.normal(size=(180, 66))
    commands = rng.uniform(-1, 1, size=(180, 2))
    targets = rng.normal(size=(180, 4))
    query = rng.normal(size=66)
    model = LocalPredictor(contexts, commands, targets, 128, .1)
    coefficient, diag = model.fit(query)

    # Independent residual-augmentation solve rather than normal equations.
    distances = np.asarray([
        sum(float(np.dot((row-query)[a:b], (row-query)[a:b])) / (b-a)
            for a, b in ((0, 2), (2, 34), (34, 66)))
        for row in contexts
    ])
    indices = sorted(range(len(contexts)), key=lambda i: (distances[i], i))[:128]
    weights = np.exp(-distances[indices] / max(distances[indices[-1]], 1e-12))
    weights /= sum(weights)
    x = np.asarray([np.r_[1., contexts[i]-query, commands[i]] for i in indices])
    regularizer = np.diag(np.r_[0., np.repeat(np.sqrt(.1), 68)])
    augmented_x = np.vstack((np.sqrt(weights)[:, None] * x, regularizer))
    augmented_y = np.vstack((np.sqrt(weights)[:, None] * targets[indices], np.zeros((69, 4))))
    reference = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    coefficient_error = float(np.max(np.abs(reference-coefficient)))
    weight_error = float(np.max(np.abs(weights-diag['normalized_weights'])))
    assert diag['neighbor_indices'] == indices
    assert coefficient_error < 1e-12
    assert weight_error < 1e-14

    # Same observations at exact equal contexts: ridge cannot shrink intercept.
    allzero = np.zeros((2, 66))
    constant = np.asarray([1., -2., 3., -4.])
    flat = LocalPredictor(allzero, np.zeros((2, 2)), np.tile(constant, (2, 1)), 512, .1)
    before, before_diag = flat.fit(np.zeros(66))
    assert np.max(np.abs(before[0]-constant)) < 1e-14
    assert np.max(np.abs(before[1:])) < 1e-14
    assert before_diag['bandwidth_d2'] == 1e-12
    flat.append(np.zeros(66), np.zeros(2), 4*constant)
    after, after_diag = flat.fit(np.zeros(66))
    append_error = float(np.max(np.abs(after[0]-2*constant)))
    assert append_error < 1e-14
    assert after_diag['neighbors_used'] == 3 and flat.n_rows == 3
    assert after_diag['neighbor_indices'] == [0, 1, 2]
    assert np.array_equal(allzero, np.zeros((2, 66)))
    owns_copies = all(not np.shares_memory(a, b) for a, b in (
        (model.context_train, contexts), (model.command_train, commands),
        (model.target_train, targets)))
    assert owns_copies
    result = {
        'status': 'NUMERICAL FIXTURES PASSED; NO ACQUIRED DATA OR FITTED RESEARCH RESULTS',
        'local_source_sha256': hashlib.sha256(Path(__file__).with_name('cartesian_stream_v17_local.py').read_bytes()).hexdigest(),
        'independent_augmented_lstsq_coefficient_max_difference': coefficient_error,
        'independent_weight_max_difference': weight_error,
        'normal_equation_max_residual': diag['normal_equation_max_residual'],
        'constant_intercept_and_append_max_difference': append_error,
        'training_inputs_copied': owns_copies,
    }
    output = Path(__file__).with_name('check_cartesian_stream_v17_local_results.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
