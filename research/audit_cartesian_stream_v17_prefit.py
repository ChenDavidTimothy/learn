#!/usr/bin/env python3
"""Pre-fit source-unit audit against separate NumPy equations, not artifact replay.

Imports the units under review only as tested adapters. The reference feature,
residual-target, autonomous-rollout and derivative calculations below do not call
their implementation. No evaluator file or plant is opened.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def context_reference(o, a):
    now = o[-1]
    slopes = (now[None] - o[-1 - np.arange(1, 17)]) / (.1 * np.arange(1, 17)[:, None])
    return np.concatenate([now, slopes.flatten(), a[-16:].flatten()])


def numpy_network(x, state):
    x = np.tanh(state['layers.0.weight'] @ x + state['layers.0.bias'])
    x = np.tanh(state['layers.2.weight'] @ x + state['layers.2.bias'])
    return state['layers.4.weight'] @ x + state['layers.4.bias']


def rollout_reference(o, a, plan, scale, states, local=False):
    query = (context_reference(o, a) - scale['context_mean']) / scale['context_std']
    paths = []
    for state in states:
        history = list(np.array(o[-17:], copy=True))
        commands = list(np.array(a[-16:], copy=True))
        future = []
        for u in plan:
            c = (context_reference(np.array(history), np.array(commands)) - scale['context_mean']) / scale['context_std']
            y = (np.r_[1., c-query, u/.5] @ state if local else numpy_network(np.r_[c, u/.5], state))
            residual = (scale['target_mean'] + scale['target_std'] * y).reshape(2, 2)
            drift = history[-1] - history[-2]
            pair = np.array([history[-1] + drift, history[-1] + 2*drift]) + residual
            future.extend(pair)
            history = (history + list(pair))[-17:]
            commands = (commands + [u.copy(), u.copy()])[-16:]
        paths.append(future)
    return np.array(paths)


def main():
    source = ROOT / 'cartesian_stream_v17_learn.py'
    spec = importlib.util.spec_from_file_location('unit_under_audit', source)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    rng = np.random.default_rng(816274)
    o = rng.normal(0, .1, (3, 65, 2)).cumsum(axis=1)
    a = rng.uniform(-.5, .5, (3, 32, 2)).repeat(2, axis=1)
    X, U, Y, origins = implementation.rows(o, a, [2, 0])
    xr = []; ur = []; yr = []; ixr = []
    for i in [2, 0]:
        for t in range(16, 64, 2):
            xr.append(context_reference(o[i, :t+1], a[i, :t]))
            ur.append(a[i, t] * 2)
            yr.append(np.concatenate([o[i, t+k]-o[i, t]-k*(o[i, t]-o[i, t-1]) for k in [1, 2]]))
            ixr.append([i, t])
    feature_error = float(np.max(abs(X-xr)))
    action_error = float(np.max(abs(U-ur)))
    target_error = float(np.max(abs(Y-yr)))
    assert np.array_equal(origins, ixr)
    assert max(feature_error, action_error, target_error) < 1e-14
    # Future corruption cannot change a forecast-origin context/candidate input.
    altered_o = o.copy(); altered_a = a.copy()
    altered_o[2, 17:] += 1e6; altered_a[2, 17:] -= 1e6
    xa, ua, _, _ = implementation.rows(altered_o, altered_a, [2, 0])
    assert np.array_equal(xa[0], X[0]) and np.array_equal(ua[0], U[0])
    tensor_error = float(np.max(abs(implementation.tensor_context(
        torch.tensor(o[0, :17]), torch.tensor(a[0, :16])).numpy()-context_reference(o[0, :17], a[0, :16]))))
    assert tensor_error < 1e-14
    scale = dict(context_mean=rng.normal(0, .03, 66), context_std=np.linspace(.4, 1.2, 66),
                 target_mean=rng.normal(0, .001, 4), target_std=np.linspace(.01, .03, 4))
    obs = o[0, :21]; act = a[0, :20]; plan = rng.uniform(-.4, .4, (3, 2))
    coefficients = [rng.normal(0, .008, (69, 4)) for _ in range(3)]

    class FixedLocal:
        def __init__(self, coefficient): self.coefficient = coefficient
        def fit(self, query): return self.coefficient.copy(), {'fixture': True}

    torch.manual_seed(9234)
    networks = [implementation.Network(9) for _ in range(3)]
    with torch.no_grad():
        for network in networks:
            for parameter in network.parameters(): parameter.mul_(.07)
    states = [{k: v.detach().numpy().copy() for k, v in network.state_dict().items()} for network in networks]
    errors = {}
    weight = rng.normal(0, .1, (3, 6, 2))
    for kind, members, reference_states in [('nonlinear', networks, states), ('local', [FixedLocal(c) for c in coefficients], coefficients)]:
        predictor = implementation.Predictor(kind, members, scale, {})
        prepared = predictor.prepare(obs, act)
        variable = torch.tensor(plan, dtype=torch.float64, requires_grad=True)
        paths = predictor.paths(obs, act, variable, prepared)
        ref = rollout_reference(obs, act, plan, scale, reference_states, local=kind=='local')
        discrepancy = float(np.max(abs(paths.detach().numpy()-ref)))
        objective = (paths*torch.tensor(weight)).sum() + .1*(paths**2).sum()
        derivative = torch.autograd.grad(objective, variable)[0].numpy()
        fd = np.zeros_like(plan)
        for index in np.ndindex(plan.shape):
            plus = plan.copy(); minus = plan.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
            def loss(p):
                y = rollout_reference(obs, act, p, scale, reference_states, local=kind=='local')
                return np.sum(y*weight)+.1*np.sum(y*y)
            fd[index] = (loss(plus)-loss(minus))/(2e-6)
        derivative_error = float(np.max(abs(derivative-fd)))
        assert discrepancy < 2e-13 and derivative_error < 1e-8
        errors[kind] = dict(three_member_three_block_path_max_error=discrepancy,
                            plan_autograd_vs_reference_central_difference_max_error=derivative_error,
                            max_absolute_plan_derivative=float(np.max(abs(derivative))))

    # Check append integration into the exact completed row (constructor/fit are
    # separately audited); repeated append calls are deliberately not a supported
    # usage assumption here.
    class Recorder:
        def __init__(self): self.records = []
        def append(self, x, u, y): self.records.append((x.copy(), u.copy(), y.copy()))
    recorders = [Recorder() for _ in range(3)]
    predictor = implementation.Predictor('local', recorders, scale, {})
    predictor.append_completed(obs, act)
    t = len(act)-2
    expected_x = (context_reference(obs[:t+1], act[:t])-scale['context_mean'])/scale['context_std']
    expected_y = (np.concatenate([obs[t+k]-obs[t]-k*(obs[t]-obs[t-1]) for k in [1, 2]])-scale['target_mean'])/scale['target_std']
    append_error = max(float(np.max(abs(recorder.records[0][0]-expected_x))) for recorder in recorders)
    append_error = max(append_error, *[float(np.max(abs(r.records[0][1]-act[t]/.5))) for r in recorders],
                       *[float(np.max(abs(r.records[0][2]-expected_y))) for r in recorders])
    assert append_error < 1e-13 and all(len(r.records)==1 for r in recorders)
    imports = []
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import): imports.extend(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append(node.module)
    assert not any(('evaluator' in str(n) or 'coupled' in str(n) or 'plant' in str(n)) for n in imports)
    result = dict(status='PASS PRE-FIT SOURCE-UNIT CHECKS; NO FIT/CONTROL CLAIM',
                  artifact_sha256={p.name: sha(p) for p in [source, ROOT/'cartesian_stream_v17_local.py', ROOT/'cartesian_stream_v17_learning_design.md', ROOT/'cartesian_stream_v17_forks.py']},
                  fixture_seed=816274, source_imports=imports,
                  feature_max_error=feature_error, candidate_input_max_error=action_error,
                  target_correction_max_error=target_error, tensor_context_max_error=tensor_error,
                  completed_append_max_error=append_error, row_count=48,
                  rows_origins_match=True, future_prefix_invariance=True,
                  production_forecast_origins_per_trajectory=list(range(16, 256-12+1, 24)),
                  rollout_checks=errors,
                  limitations='Adapter unit tests import audited source; independent NumPy references do not. No actual fits, bootstrap artifacts, heldout forks or controls reproduced. Static selection/dataflow review complements these checks. Future runtime failure preservation requires separate review.')
    (ROOT/'audit_cartesian_stream_v17_prefit_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
