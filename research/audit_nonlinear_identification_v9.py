#!/usr/bin/env python3
"""Independent recurrence/objective/metric audit of the v9 identification study.

Does not rerun the canonical experiment. Physical simulation below eliminates
the motor entirely. One fit is independently optimized using numerical finite
differences rather than the canonical analytic sensitivity implementation.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits


def simulate(theta, commands):
    rho, lam, c, beta = theta
    v = np.zeros(len(commands) + 2)
    for t, command in enumerate(commands):
        f1 = rho * v[t + 1] - c * np.tanh(v[t + 1])
        f0 = rho * v[t] - c * np.tanh(v[t])
        v[t + 2] = f1 + lam * (v[t + 1] - f0) + beta * command
    return np.cumsum(v[:len(commands) + 1]), v[:len(commands) + 1]


def stat(values):
    a = np.asarray(values)
    return dict(mean=float(np.mean(a)), mc_se=float(np.std(a, ddof=1) / np.sqrt(len(a))),
                median=float(np.median(a)), q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9)))


def structured_residual(theta, observations, commands):
    rho, lam, c, beta = theta
    d = np.diff(observations)
    def passive(v):
        return rho * v - c * np.tanh(v)
    return passive(d[1:-1]) + lam * (d[1:-1] - passive(d[:-2])) + beta * commands[1:-1] - d[2:]


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "nonlinear_identification_v9_results.json").read_text())
    spec = importlib.util.spec_from_file_location("canonical_v9", root / "nonlinear_identification_v9.py")
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    maxima = dict(trajectory=0., OLS_coefficient=0., singular_values=0., recorded_objective=0.,
                  heldout_metric=0., summary=0., analytic_jacobian=0., structured_jacobian=0., derived_parameter_metric=0.)
    scaled_maxima = dict.fromkeys(maxima, 0.)

    def record(key, a, b):
        error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        maxima[key] = max(maxima[key], error)
        scale = np.maximum(1., np.maximum(np.abs(a), np.abs(b)))
        scaled_maxima[key] = max(scaled_maxima[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / scale)))

    for row in data['clean']:
        commands = np.random.default_rng(row['seed']).normal(0, row['command_sd'], 4096)
        theta = [.65, .4, row['c'], 1.2]
        x, v = simulate(theta, commands)
        xc, vc, _ = canonical.trajectory(theta, commands)
        record('trajectory', x, xc)
        record('trajectory', v, vc)
        d = np.diff(x)
        # Physical index t=1,...,n-2, independent list-based construction.
        regressors = np.array([[d[t], d[t - 1], np.tanh(d[t]), np.tanh(d[t - 1]), commands[t]]
                              for t in range(1, len(commands) - 1)])
        targets = np.array([d[t + 1] for t in range(1, len(commands) - 1)])
        coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]
        record('OLS_coefficient', coef, row['coefficients'])
        record('singular_values', np.linalg.svd(regressors, compute_uv=False), row['singular_values'])

    checked_fits = 0
    refit_case = None
    for condition in data['noisy']:
        evaluation_values = {test: {} for test in ['matched', 'amplitude_shift']}
        for row in condition['runs']:
            rng = np.random.default_rng(row['seed'])
            commands = rng.normal(0, condition['command_sd'], 512)
            true_theta = [.65, .4, condition['c'], 1.2]
            clean_x, _ = simulate(true_theta, commands)
            observations = clean_x + rng.normal(0, .2, len(clean_x))
            d = np.diff(observations)
            X = np.array([[d[t], d[t - 1], np.tanh(d[t]), np.tanh(d[t - 1]), commands[t]]
                          for t in range(1, len(commands) - 1)])
            coef = np.linalg.lstsq(X, d[2:], rcond=None)[0]
            record('OLS_coefficient', coef, row['OLS_coefficients'])
            record('derived_parameter_metric', -coef[3] / coef[2], row['OLS_lambda_ratio'])
            record('derived_parameter_metric', abs(row['OE_theta'][1] - .4), row['OE_abs_lambda_error'])
            record('derived_parameter_metric', np.linalg.norm(np.array(row['OE_theta']) - true_theta), row['OE_parameter_L2_error'])
            for fit in row['all_optimization_starts']:
                fit_x, _ = simulate(fit['theta'], commands)
                record('recorded_objective', np.sum((fit_x - observations)**2), fit['sse'])
                checked_fits += 1
            best = min(row['all_optimization_starts'], key=lambda fit: fit['sse'])
            assert row['OE_theta'] == best['theta']
            for fit in row['structured_optimization_starts']:
                residual = structured_residual(fit['theta'], observations, commands)
                record('recorded_objective', np.sum(residual**2), fit['sse'])
                checked_fits += 1
            sr_best = min(row['structured_optimization_starts'], key=lambda fit: fit['sse'])
            assert row['structured_theta'] == sr_best['theta']
            for test, sd in [('matched', condition['command_sd']), ('amplitude_shift', .9)]:
                test_commands = rng.normal(0, sd, 256)
                x, v = simulate(true_theta, test_commands)
                xp, vp = simulate(row['OE_theta'], test_commands)
                xs, vs = simulate(row['structured_theta'], test_commands)
                raw_v = np.zeros(257)
                for t in range(255):
                    features = [raw_v[t + 1], raw_v[t], np.tanh(raw_v[t + 1]), np.tanh(raw_v[t]), test_commands[t]]
                    raw_v[t + 2] = np.dot(coef, features)
                raw_x = np.cumsum(raw_v)
                metrics = dict(OE_velocity_rmse=float(np.sqrt(np.mean((vp[1:] - v[1:])**2))),
                               OLS_velocity_rmse=float(np.sqrt(np.mean((raw_v[1:] - v[1:])**2))),
                               OE_position_rmse=float(np.sqrt(np.mean((xp[1:] - x[1:])**2))),
                               OLS_position_rmse=float(np.sqrt(np.mean((raw_x[1:] - x[1:])**2))),
                               structured_velocity_rmse=float(np.sqrt(np.mean((vs[1:] - v[1:])**2))),
                               structured_position_rmse=float(np.sqrt(np.mean((xs[1:] - x[1:])**2))))
                for metric, value in metrics.items():
                    record('heldout_metric', value, row['evaluations'][test][metric])
                    evaluation_values[test].setdefault(metric, []).append(value)
            if condition['c'] == .2 and condition['command_sd'] == .7 and refit_case is None:
                refit_case = (commands, observations, best, sr_best)
        for test, metrics in evaluation_values.items():
            for metric, values in metrics.items():
                for name, value in stat(values).items():
                    record('summary', value, condition['evaluation_summary'][test][metric][name])
        for source, target in [('OE_abs_lambda_error', 'OE_abs_lambda_error'), ('OLS_lambda_ratio', 'OLS_lambda_ratio')]:
            for name, value in stat([row[source] for row in condition['runs']]).items():
                record('summary', value, condition[target][name])
        assert condition['all_starts_converged'] == sum(fit['success'] for row in condition['runs'] for fit in row['all_optimization_starts'])

    # Finite differences of a separately implemented eliminated recurrence.
    commands = np.random.default_rng(3109).normal(0, .9, 96)
    for theta in [np.array([.65, .4, .2, 1.2]), np.array([.4, .65, 0., 1.2]), np.array([.94, .94, .39, 1.9])]:
        _, _, jac = canonical.trajectory(theta, commands, True)
        numerical = np.zeros_like(jac)
        for k in range(4):
            step = np.zeros(4)
            step[k] = 1e-6
            numerical[:, k] = (simulate(theta + step, commands)[0] - simulate(theta - step, commands)[0]) / 2e-6
        record('analytic_jacobian', numerical, jac)

    commands, observations, saved_best, saved_sr_best = refit_case
    independent = least_squares(lambda p: simulate(p, commands)[0] - observations,
        [.6, .5, .15, 1.1], bounds=([.05, .05, 0., .3], [.95, .95, .4, 2.]),
        max_nfev=160, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    assert 2 * independent.cost <= saved_best['sse'] + 1e-4
    independent_sr = least_squares(lambda p: structured_residual(p, observations, commands),
        [.6, .5, .15, 1.1], bounds=([.05, .05, 0., .3], [.95, .95, .4, 2.]),
        max_nfev=160, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    assert 2 * independent_sr.cost <= saved_sr_best['sse'] + 1e-4
    theta = np.array([.65, .4, .2, 1.2])
    d = np.diff(observations)
    rho, lam, c, _ = theta
    analytic_sr_jac = np.c_[d[1:-1] - lam * d[:-2],
        d[1:-1] - rho * d[:-2] + c * np.tanh(d[:-2]),
        -np.tanh(d[1:-1]) + lam * np.tanh(d[:-2]), commands[1:-1]]
    numeric_sr_jac = np.zeros_like(analytic_sr_jac)
    for k in range(4):
        step = np.zeros(4)
        step[k] = 1e-6
        numeric_sr_jac[:, k] = (structured_residual(theta + step, observations, commands)
                             - structured_residual(theta - step, observations, commands)) / 2e-6
    record('structured_jacobian', numeric_sr_jac, analytic_sr_jac)
    assert maxima['trajectory'] < 1e-9
    assert maxima['OLS_coefficient'] < 1e-5  # weak clean designs amplify roundoff.
    assert maxima['singular_values'] < 1e-9
    assert maxima['recorded_objective'] < 1e-8
    # Some unconstrained OLS forecasts explode beyond 1e70. Absolute error
    # cannot compare independently ordered floating-point operations there.
    assert scaled_maxima['heldout_metric'] < 1e-7
    assert scaled_maxima['summary'] < 1e-7
    assert maxima['analytic_jacobian'] < 1e-4
    assert maxima['structured_jacobian'] < 1e-7
    assert scaled_maxima['derived_parameter_metric'] < 1e-7
    return dict(status='INDEPENDENTLY TESTED', clean_rows=len(data['clean']), noisy_rows=72,
        recorded_optimizer_objectives_checked=checked_fits, maximum_absolute_errors=maxima,
        maximum_errors_scaled_by_max_1_magnitude=scaled_maxima,
        independent_numerical_jacobian_fit=dict(theta=independent.x.tolist(), sse=float(2 * independent.cost),
            success=bool(independent.success), saved_sse=saved_best['sse']),
        independent_structured_numerical_jacobian_fit=dict(theta=independent_sr.x.tolist(), sse=float(2 * independent_sr.cost),
            success=bool(independent_sr.success), saved_sse=saved_sr_best['sse']),
        audit_conclusions=[
            'Lagged timing, regression indexing, sensitivity derivatives and heldout metrics verified.',
            'OE least squares is Gaussian output-error likelihood under stated process0, exogenous open-loop commands, iid sensor error and known zero state.',
            'Multistarts selected by training SSE; heldout data not used for fit selection.',
            'Numerical fit convergence and three starts do not establish a global optimum.',
            'All estimators get the same commands and noisy positions; OE and structured regression use identical four-parameter supplied family, starts, bounds and max_nfev.',
            'OE versus structured regression controls supplied family/stability regularization but compares different observation-error objectives; no optimized modern-method or general-learning superiority follows.',
            'Clean full regressor rank at c0 does not identify physical pole allocation; coefficient-to-parameter inverse remains nonunique.',
            'Small velocity support gives near-collinear v/tanh(v) features; c near0 gives fragile nonlinear ratio.',
            'Heldout clean replay uses known zero initialization; it is not unknown-state filtering or control.',
        ])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        result = main()
    print(json.dumps(result, indent=2, allow_nan=False))
