#!/usr/bin/env python3
"""Frozen executable-structure screen: ordinary data, validation-only choices.

Fit phases are fixed at 2,000 old-only or 8,000 continued updates. No evaluator
archive is imported. Scoring freezes every choice before indexing heldout rows.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from error_transport_baselines import ALPaCA, CoDA
from error_transport_screen import digest, fixed_samples, save_json, synchronize

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'error_transport_screen_data.npz'
NEW = ROOT / 'structure_acquisition_data.npz'
PLAN = ROOT / 'structure_acquisition_plan.md'
BUDGETS = (0, 4, 16, 64)
METHODS = ('meta', 'fit', 'frozen', 'neural', 'coda')
OLD_METHODS = ('programold', 'neural')
GRAPH_NAMES = {'programold': 'old', 'frozen': 'old', 'meta': 'meta', 'fit': 'fit'}


def graph_for(method):
    if method not in GRAPH_NAMES:
        return None, None
    path = ROOT / f'structure_acquisition_graph_{GRAPH_NAMES[method]}.json'
    record = json.loads(path.read_text())
    graph = record.get('graph', record)
    assert isinstance(graph['nodes'], list) and isinstance(graph['features'], list)
    assert len(graph['features']) > 0
    return graph, path


def make(method):
    if method == 'coda':
        return CoDA(num_tasks=40)
    if method == 'neural':
        return ALPaCA(feature_dim=64)
    from structure_acquisition_features import ProgramFeatures
    graph, _ = graph_for(method)
    model = ALPaCA(feature_dim=len(graph['features']))
    model.features = ProgramFeatures(graph)
    return model


def provenance(method, include_new=True):
    files = [Path(__file__), PLAN, OLD, ROOT/'error_transport_baselines.py',
             ROOT/'error_transport_screen.py']
    if include_new:
        files.append(NEW)
    if method in GRAPH_NAMES:
        _, path = graph_for(method)
        files += [path, ROOT/'structure_acquisition_features.py']
    return {p.name: digest(p) for p in files}


def load(device, include_new=True):
    with np.load(OLD, allow_pickle=False) as d:
        ox, oy, osplit = d['x'], d['y'], d['split']
    assert ox.shape == (56, 8, 96, 8) and oy.shape == (56, 8, 96, 2)
    assert np.array_equal(np.flatnonzero(osplit == 0), np.arange(32))
    assert np.array_equal(np.flatnonzero(osplit == 1), np.arange(32, 40))
    tr = osplit == 0
    scale = dict(x_mean=ox[tr].mean((0, 1, 2)), x_std=np.maximum(ox[tr].std((0, 1, 2)), .01),
                 y_mean=oy[tr].mean((0, 1, 2)), y_std=np.maximum(oy[tr].std((0, 1, 2)), .01))
    if include_new:
        with np.load(NEW, allow_pickle=False) as d:
            nx, ny, ns = d['x'], d['y'], d['split']
        assert nx.shape == (36, 8, 96, 8) and ny.shape == (36, 8, 96, 2)
        assert np.array_equal(ns, np.repeat(np.arange(5), [8, 4, 8, 8, 8]))
        ox, oy = np.concatenate((ox, nx)), np.concatenate((oy, ny))
    assert np.isfinite(ox).all() and np.isfinite(oy).all()
    x = torch.as_tensor((ox-scale['x_mean'])/scale['x_std'], dtype=torch.float32, device=device)
    y = torch.as_tensor((oy-scale['y_mean'])/scale['y_std'], dtype=torch.float32, device=device)
    return x, y, scale


def prefix(phase, method, lr):
    return ROOT / f'structure_acquisition_{phase}_{method}_lr{lr:g}_seed0'


def selected_fit(phase, method):
    options = []
    for lr in (.001, .003):
        p = prefix(phase, method, lr)
        report_path, weights = Path(str(p)+'.json'), Path(str(p)+'.pt')
        r = json.loads(report_path.read_text())
        assert r['status'] == 'COMPLETE TRAIN/VALIDATION; TEST NOT SCORED'
        assert (r['phase'], r['method'], r['lr']) == (phase, method, lr)
        assert r['weights_sha256'] == digest(weights)
        expected = provenance(method, include_new=phase != 'old')
        assert r['artifact_sha256'] == expected, ('changed fit dependency', phase, method, lr)
        assert len(r['all_training_losses']) == (2000 if phase == 'old' else 8000)
        assert np.isfinite(r['all_training_losses']).all()
        options.append((r, report_path, weights))
    r, rp, wp = min(options, key=lambda item: item[0]['best_validation']['criterion'])
    return dict(report=str(rp), report_sha256=digest(rp), weights=str(wp), weights_sha256=digest(wp),
                validation=r['best_validation'], parameter_count=r['parameter_count'])


def selected_original_coda():
    """Re-use the earlier validation choice, not any earlier test ranking."""
    options = []
    for lr in (.001, .003):
        rp = ROOT / f'error_transport_screen_coda_lr{lr:g}_seed0.json'
        r = json.loads(rp.read_text()); wp = rp.with_suffix('.pt')
        assert r['status'].startswith('COMPLETE') and r['data_sha256'] == digest(OLD)
        assert r['driver_sha256'] == digest(ROOT/'error_transport_screen.py')
        assert r['baseline_source_sha256'] == digest(ROOT/'error_transport_baselines.py')
        assert r['weights_sha256'] == digest(wp)
        options.append((r, rp, wp))
    r, rp, wp = min(options, key=lambda item: item[0]['best_validation']['criterion'])
    return dict(report=str(rp), report_sha256=digest(rp), weights=str(wp), weights_sha256=digest(wp),
                validation=r['best_validation'], parameter_count=r['parameter_count'])


def initialize(method, phase, model, device):
    if phase == 'old':
        return None
    parent = selected_original_coda() if method == 'coda' else selected_fit('old', 'neural' if method == 'neural' else 'programold')
    saved = torch.load(parent['weights'], map_location=device, weights_only=False)
    state = saved['state_dict']
    if method == 'coda':
        assert state['contexts.weight'].shape == (32, 8)
        expanded = model.contexts.weight.detach().clone()
        expanded[:32] = state['contexts.weight']
        state = dict(state, **{'contexts.weight': expanded})
        model.load_state_dict(state)
    elif method == 'neural':
        model.load_state_dict(state)
    else:
        old_graph, _ = graph_for('programold'); new_graph, _ = graph_for(method)
        n = len(old_graph['features']); f = len(new_graph['features'])
        assert f >= n and new_graph['features'][:n] == old_graph['features'], 'expanded graph must preserve feature prefix'
        assert new_graph['nodes'][:len(old_graph['nodes'])] == old_graph['nodes'], 'expanded graph must preserve old computations and normalizations'
        # Features are executable graph constants. Transfer the learned coefficient
        # distribution only; new dimensions have independent unit precision.
        with torch.no_grad():
            model.K0.zero_(); model.K0[:n].copy_(state['K0'])
            model.raw_precision_cholesky.zero_()
            diagonal = math.log(math.expm1(1.-1e-4))
            model.raw_precision_cholesky.diagonal().fill_(diagonal)
            model.raw_precision_cholesky[:n, :n].copy_(state['raw_precision_cholesky'])
            model.noise_variance.copy_(state['noise_variance'])
        assert torch.equal(model.K0[:n], state['K0']) and torch.count_nonzero(model.K0[n:]) == 0
        assert torch.equal(model.raw_precision_cholesky[:n, :n], state['raw_precision_cholesky'])
        if f > n:
            assert torch.allclose(model.prior_precision()[n:, n:], torch.eye(f-n, device=device), atol=2e-6, rtol=0)
            assert torch.count_nonzero(model.prior_precision()[:n, n:]) == 0
    return parent


def prediction(model, method, xs, ys, xq, adaptation_lr=.03):
    if method == 'coda':
        context = model.adapt_context(xs, ys, steps=100, lr=adaptation_lr)
        with torch.no_grad():
            return model.functional(xq, model.theta0.detach()+context@model.W.detach().T)
    with torch.no_grad():
        coefficients, _ = model.posterior(xs, ys)
        return torch.cat([model.features(q)@coefficients for q in xq.split(128, dim=1)], dim=1)


def evaluate(model, method, x, y, ids, budgets=(4, 16, 64), wrong=False, adaptation_lr=.03):
    arrays, scores = {}, {}
    for n in budgets:
        sx, sy, qx, qy = fixed_samples(x, y, ids, n, shuffle=wrong)
        p = prediction(model, method, sx, sy, qx, adaptation_lr)
        assert p.shape == qy.shape and torch.isfinite(p).all(), ('nonfinite forecast', method, n)
        arrays[str(n)] = p.detach().cpu().numpy()
        scores[str(n)] = (p-qy).square().mean((1, 2)).detach().cpu().tolist()
    return arrays, scores


def validation(model, method, phase, x, y):
    groups = {'old': np.arange(32, 40)}
    if phase != 'old':
        groups['new'] = np.arange(64, 68)
    records = []
    for lr in ((.01, .03, .1) if method == 'coda' else (.03,)):
        start = time.time(); scores = {}
        for label, ids in groups.items():
            _, scores[label] = evaluate(model, method, x, y, ids, adaptation_lr=lr)
        # Each group's body/support mean has equal weight despite 8 vs 4 bodies.
        group_means = {key: float(np.mean(list(value.values()))) for key, value in scores.items()}
        synchronize()
        records.append(dict(adaptation_lr=lr, criterion=float(np.mean(list(group_means.values()))),
                            group_means=group_means, body_scores=scores, wall_seconds=time.time()-start))
    return min(records, key=lambda rec: rec['criterion']), records


def episode(rng, phase, x, y):
    ids = rng.choice(np.arange(32), 8) if phase == 'old' else np.concatenate((rng.choice(np.arange(32), 4), rng.choice(np.arange(56, 64), 4)))
    st = rng.integers(0, 8, 8); n = BUDGETS[int(rng.integers(0, 4))]
    starts = rng.integers(0, 97-n, 8)
    body = torch.as_tensor(ids, device=x.device); support_tr = torch.as_tensor(st, device=x.device)
    times = torch.as_tensor(starts[:, None]+np.arange(n)[None], device=x.device)
    sx, sy = x[body[:, None], support_tr[:, None], times], y[body[:, None], support_tr[:, None], times]
    qt = (st[:, None]+rng.integers(1, 8, (8, 32))) % 8
    qi = rng.integers(0, 96, (8, 32)); qt, qi = torch.as_tensor(qt, device=x.device), torch.as_tensor(qi, device=x.device)
    qx, qy = x[body[:, None], qt, qi], y[body[:, None], qt, qi]
    task_ids = torch.where(body >= 56, body-24, body)
    assert bool(torch.all((task_ids >= 0) & (task_ids < 40)))
    return sx, sy, qx, qy, task_ids, dict(body_ids=ids.tolist(), support_trajectories=st.tolist(), support_starts=starts.tolist(), support=n)


def fit(args):
    phase, method = args.phase, args.method
    assert method in (OLD_METHODS if phase == 'old' else METHODS)
    p = prefix(phase, method, args.lr)
    jp, wp, failure = Path(str(p)+'.json'), Path(str(p)+'.pt'), Path(str(p)+'_failure.pt')
    assert not any(path.exists() for path in (jp, wp, failure)), 'refuse overwrite'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    phase_id = int(phase != 'old'); order = OLD_METHODS if phase == 'old' else METHODS
    seed = int(np.random.SeedSequence([20261020, 18, 42, phase_id, order.index(method)]).generate_state(1)[0])
    torch.manual_seed(seed)
    model = make(method).to(device); parent = initialize(method, phase, model, device)
    x, y, scale = load(device, include_new=phase != 'old')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(np.random.SeedSequence([20261020, 18, 41, phase_id]))
    steps = 2000 if phase == 'old' else 8000
    losses, records, support_counts = [], [], {str(n): 0 for n in BUDGETS}
    choice = None; best = float('inf'); start = time.time(); step = -1; batch = None
    sources = provenance(method, include_new=phase != 'old')
    try:
        for step in range(steps):
            sx, sy, qx, qy, task_ids, metadata = episode(rng, phase, x, y)
            batch = dict(xs=sx, ys=sy, xq=qx, yq=qy, task_ids=task_ids, metadata=metadata)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(sx, sy, qx, qy, task_ids)
            assert torch.isfinite(loss), ('nonfinite training loss', step)
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            assert torch.isfinite(norm), ('nonfinite gradient', step)
            optimizer.step(); losses.append(float(loss.detach()))
            assert all(torch.isfinite(v).all() for v in model.parameters()), ('nonfinite weights', step)
            support_counts[str(metadata['support'])] += 1
            if (step+1) % (500 if phase == 'old' else 2000) == 0:
                model.eval(); current, alternatives = validation(model, method, phase, x, y); model.train()
                assert np.isfinite(current['criterion'])
                records.append(dict(step=step+1, validation=alternatives, mean_recent_training_loss=float(np.mean(losses[-100:]))))
                if current['criterion'] < best:
                    best = current['criterion']; choice = dict(current, step=step+1)
                    torch.save(dict(method=method, phase=phase, state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, scale=scale, choice=choice,
                                    artifact_sha256=sources, parent=parent), wp)
                save_json(jp, dict(status='RUNNING TRAIN/VALIDATION; TEST NOT SCORED', phase=phase, method=method, lr=args.lr,
                                  steps_completed=step+1, best_validation=choice, validation_records=records, all_training_losses=losses,
                                  artifact_sha256=sources, parent=parent))
                print(phase, method, args.lr, step+1, current['criterion'], flush=True)
        assert choice is not None
    except BaseException as exc:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, rng_state=rng.bit_generator.state,
                        torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        batch=batch, parent=parent, artifact_sha256=sources, losses=losses, records=records), failure)
        save_json(jp, dict(status='FAILED; NO RETRY', exception=repr(exc), phase=phase, method=method, lr=args.lr, step=step,
                          all_training_losses=losses, validation_records=records, artifact_sha256=sources, failure_sha256=digest(failure)))
        raise
    synchronize()
    assert provenance(method, include_new=phase != 'old') == sources, 'dependency changed during fit'
    save_json(jp, dict(status='COMPLETE TRAIN/VALIDATION; TEST NOT SCORED', phase=phase, method=method, lr=args.lr, seed=0,
                      initialization_seed=seed, episode_seed_roles=[20261020, 18, 41, phase_id], steps=steps,
                      parameter_count=sum(v.numel() for v in model.parameters()), best_validation=choice,
                      validation_records=records, all_training_losses=losses, support_budget_update_counts=support_counts,
                      training_body_draws=steps*8, training_query_transitions=steps*8*32,
                      training_support_transitions=sum(int(n)*count*8 for n,count in support_counts.items()),
                      artifact_sha256=sources, parent=parent, weights_sha256=digest(wp), wall_seconds=time.time()-start, device=device))


def score():
    output = ROOT/'structure_acquisition_results.json'; ap = ROOT/'structure_acquisition_predictions.npz'
    sp = ROOT/'structure_acquisition_selection.json'; failure = ROOT/'structure_acquisition_score_failure.json'
    assert not any(p.exists() for p in (output, ap, sp, failure)), 'refuse overwrite'
    choices = {method: selected_fit('continue', method) for method in METHODS}
    save_json(sp, dict(status='FROZEN VALIDATION-ONLY SELECTION; TEST NOT SCORED', choices=choices))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y, scale = load(device); arrays, results = {}, {}; start = time.time()
    try:
        for method, ch in choices.items():
            state = torch.load(ch['weights'], map_location=device, weights_only=False)
            model = make(method).to(device); model.load_state_dict(state['state_dict']); model.eval()
            assert state['choice'] == ch['validation']
            assert all(np.array_equal(state['scale'][k], v) for k,v in scale.items())
            results[method] = {}
            for ids, label in ((np.arange(68, 76), 'new_id'), (np.arange(76, 84), 'new_ood'), (np.arange(84, 92), 'old_id')):
                for wrong in (False, True):
                    group = label+('_wrong_support' if wrong else '')
                    predictions, scores = evaluate(model, method, x, y, ids, budgets=BUDGETS, wrong=wrong,
                                                   adaptation_lr=ch['validation']['adaptation_lr'])
                    qy = y[torch.as_tensor(ids, device=device), 1:].flatten(1, 2).cpu().numpy()
                    raw = {n:np.mean(((p-qy)*scale['y_std'])**2, axis=(1, 2)).tolist() for n,p in predictions.items()}
                    results[method][group] = dict(body_ids=ids.tolist(), new_archive_body_ids=(ids-56).tolist(), standardized_body_mse=scores, raw_body_mse=raw)
                    for n,p in predictions.items():
                        arrays[f'{method}_{group}_support{n}'] = p
            print('scored', method, flush=True)
    except BaseException as exc:
        partial = ROOT/'structure_acquisition_score_failure_predictions.npz'
        np.savez_compressed(partial, **arrays)
        save_json(failure, dict(status='FAILED; NO RETRY', exception=repr(exc), choices=choices, completed_results=results,
                               partial_predictions_sha256=digest(partial), selection_sha256=digest(sp)))
        raise
    synchronize(); np.savez_compressed(ap, **arrays)
    save_json(output, dict(status='COMPLETE EXPLORATORY STRUCTURE SCREEN; INDEPENDENT CHECK PENDING', choices=choices, results=results,
                           selection_sha256=digest(sp), predictions_sha256=digest(ap), artifact_sha256={method:provenance(method) for method in METHODS},
                           wall_seconds=time.time()-start,
                           scope='One training seed; new-law ID/OOD and old-law retention bodies; exact supplied q/v, ordinary measured finite transitions. No evaluator features, control or framework claim.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fit', 'score'))
    parser.add_argument('--phase', choices=('old', 'continue'), default='continue')
    parser.add_argument('--method', choices=OLD_METHODS+METHODS)
    parser.add_argument('--lr', type=float, choices=(.001, .003), default=.001)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.mode == 'fit':
        assert args.method is not None; fit(args)
    else:
        score()


if __name__ == '__main__':
    main()
