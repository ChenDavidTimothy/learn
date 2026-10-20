#!/usr/bin/env python3
"""Audit v3's explicit drag map and test an exact dissipative replacement.

Run with .venv/bin/python research/dissipative_flow_v4.py. No new dependencies.
Finite stress tests illustrate the proof; they do not estimate rare-event rates.
"""
from __future__ import annotations
import json
import platform
import numpy as np

SEED = 20261020
K, DRAG, BETA = .25, .12, 1.2
RHO = float(np.exp(-K))
ALPHA = DRAG * (1 - RHO) / K


def flow(v, dt=1.):
    rho = np.exp(-K * dt)
    alpha = DRAG * (1 - rho) / K
    return rho * v / (1 + alpha * abs(v))


def derivative(v):
    return RHO / (1 + ALPHA * np.abs(v)) ** 2


def explicit(v):
    return (1 - K) * v - DRAG * v * abs(v)


def stress_test():
    rows = {}
    for name, fn in [('v3_explicit', explicit), ('exact_flow', flow)]:
        v = 20.
        path = [v]
        for _ in range(20):
            v = float(fn(v))
            path.append(v)
            if abs(v) > 1e100:
                break
        rows[name] = {'initial_velocity': 20., 'no_further_forcing': True,
                      'path': path, 'exceeded_1e100': abs(v) > 1e100}
    assert rows['v3_explicit']['exceeded_1e100']
    assert not rows['exact_flow']['exceeded_1e100']
    # Exact flow is a semigroup; explicit Euler is not.
    grid = np.linspace(-30, 30, 12001)
    semigroup_error = float(np.max(np.abs(flow(flow(grid, .3), .7) - flow(grid))))
    fd = (flow(grid + 1e-5) - flow(grid - 1e-5)) / 2e-5
    fd_error = float(np.max(np.abs(fd - derivative(grid))))
    assert semigroup_error < 1e-12
    assert fd_error < 1e-5
    assert np.max(np.abs(flow(grid))) <= RHO / ALPHA
    return {'trajectories': rows, 'semigroup_max_error': semigroup_error,
            'finite_difference_derivative_max_error': fd_error}


def run(seed, amplitude, n=6000, burn=600):
    rng = np.random.default_rng(seed)
    steps = n + burn
    a = rng.normal(0, amplitude, steps)
    w = rng.normal(0, .06, steps)
    e = rng.normal(0, 1., steps + 1)
    v = np.zeros(steps + 1)
    for t in range(steps):
        v[t + 1] = flow(v[t]) + BETA * a[t] + w[t]
    # Measured displacements; latent velocities used only for theorem audit.
    y = v[1:] + np.diff(e)
    t = np.arange(burn + 3, steps)
    X = np.column_stack([np.ones(len(t))] + [a[t - j] for j in range(4)])
    h = np.linalg.lstsq(X, y[t], rcond=None)[0][1:]
    prediction = float(np.mean(derivative(v[t])))
    return {'seed': seed, 'h0': float(h[0]), 'h1': float(h[1]),
            'echo_ratio': float(h[1] / h[0]),
            'stationary_derivative_plugin': prediction,
            'ratio_minus_derivative': float(h[1] / h[0] - prediction),
            'max_abs_velocity': float(np.max(np.abs(v)))}


def main():
    conditions = {}
    for j, amplitude in enumerate((.35, 1.25)):
        runs = [run(SEED + 400000 + 1000 * j + i, amplitude) for i in range(64)]
        errors = np.array([r['ratio_minus_derivative'] for r in runs])
        conditions[str(amplitude)] = {
            'amplitude': amplitude, 'n': 6000, 'burn': 600, 'runs': runs,
            'mean_h0': float(np.mean([r['h0'] for r in runs])),
            'mean_echo_ratio': float(np.mean([r['echo_ratio'] for r in runs])),
            'mean_derivative': float(np.mean([r['stationary_derivative_plugin'] for r in runs])),
            'mean_identity_error': float(errors.mean()),
            'identity_error_mc_se': float(errors.std(ddof=1) / np.sqrt(len(errors)))}
        assert abs(errors.mean()) < 4 * errors.std(ddof=1) / np.sqrt(len(errors))
    print(json.dumps({'meta': {'version': 4, 'seed': SEED, 'python': platform.python_version(),
        'numpy': np.__version__, 'k': K, 'drag': DRAG, 'beta': BETA, 'rho': RHO,
        'alpha': ALPHA, 'sensor_sd': 1., 'process_sd': .06,
        'observations': 'noisy position differences and iid Gaussian actions only',
        'claim': 'correction and identity replication; no learning-method superiority'},
        'stress': stress_test(), 'stable_identity': conditions}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
