#!/usr/bin/env python3
"""Bounded independent saved-result check. No learner/environment imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

R = Path(__file__).resolve().parent
COMMIT = '14b87ac5fd855ac21c734c07020cc3b3cfc8a6a2'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text())

def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

def score(path, reference):
    with np.load(path, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    q, v, u, g = (z[k] for k in ('q', 'v', 'u', 'goals'))
    assert q.shape == v.shape == (64, 241, 2)
    assert u.shape == (64, 240, 2) and g.shape == (64, 3, 2)
    assert all(np.isfinite(a).all() for a in z.values() if a.dtype.kind in 'fci')
    assert np.max(np.abs(u)) <= 1 + 1e-7
    goal = np.repeat(g, 80, axis=1)
    delta = q[:, 1:] - goal
    error = np.arctan2(np.sin(delta), np.cos(delta))
    cost = (error * error).sum(axis=2)/2 + .1*(v[:, 1:]**2).sum(axis=2)/2 + .01*(u*u).sum(axis=2)/2
    precise = np.stack([((np.abs(error[:, 80*j+60:80*(j+1)]) <= .15).all(axis=(1, 2)) &
                         (np.abs(v[:, 80*j+61:80*(j+1)+1]) <= .15).all(axis=(1, 2)))
                        for j in range(3)], axis=1)
    close(cost, z['cost'])
    np.testing.assert_array_equal(precise, z['precision'])
    close(cost.mean(axis=1), reference['body_cost'])
    close(cost.mean(), reference['mean_cost'])
    np.testing.assert_array_equal(precise, reference['precise_goals'])
    close(precise.mean(), reference.get('precision_fraction', reference.get('precision')))
    np.testing.assert_array_equal(z['evaluator_tail_observations'], [20, 20, 20])
    np.testing.assert_array_equal(g, z['evaluator_goals'])
    assert z['seed'] == 2026102181000 and str(z['split']) == 'validation'
    pairing = {k: z[k] for k in ('goals', 'evaluator_parameters', 'evaluator_input_matrix')}
    pairing.update(initial_q=q[:, 0], initial_v=v[:, 0])
    return pairing, dict(path=str(path.relative_to(R)), sha256=sha(path),
                        mean_cost=float(cost.mean()), precise_goals=int(precise.sum()),
                        failed_goal_indices=np.argwhere(~precise).tolist(),
                        max_saved_cost_error=float(np.max(np.abs(cost-z['cost']))))

def run():
    s = load(R/'amago_control_selection.json')
    artifacts = {'amago_control_selection.json': sha(R/'amago_control_selection.json')}
    for name, expected in s['sources'].items():
        assert sha(R/name) == expected, name
        frozen = subprocess.check_output(['git', 'show', f'{COMMIT}:research/{name}'], cwd=R.parent)
        assert hashlib.sha256(frozen).hexdigest() == expected, ('prefit source', name)
    assert sha(R/'amago_control_decision.py') == s['decision_source_sha256']
    manifest = load(R/'third_party/amago/UPSTREAM.json')
    assert manifest['commit'] == '54c25ab6da9371614c47352f569a56c0fe938d3b'
    for e in manifest['files']:
        raw = (R/'third_party/amago'/e['path']).read_bytes()
        assert len(raw) == e['bytes']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == e['git_blob']
    reports, rows, baseline = [], [], None
    for directory in ('amago_control_0001', 'amago_control_0003'):
        rp = R/directory/'report.json'
        r = load(rp)
        artifacts[str(rp.relative_to(R))] = sha(rp)
        assert r['sources'] == s['sources'] and r['official_commit'] == manifest['commit']
        assert r['status'].startswith('COMPLETE MAINTAINED BASELINE') and not r['smoke']
        assert r['seed'] == 20261021 and r['batches_per_epoch'] == 50
        assert [h['epoch'] for h in r['history']] == list(range(1, 129))
        assert [c['epoch'] for c in r['candidates']] == [32, 64, 96, 128]
        previous = 0
        for h in r['history']:
            logs = r['local_logs'][previous:h['log_end']]
            updates = [x for x in logs if x['key'] == 'train-update']
            assert len(updates) == 50
            assert sum(x['key'] == 'dataset' for x in logs) == 1
            for x in updates:
                assert all(np.isfinite(value) for value in x['metrics'].values())
                assert x['metrics']['Sequence Length'] == 241
                assert x['metrics']['Batch Size (in Timesteps)'] == 3840
                assert x['metrics']['Unmasked Batch Size (in Timesteps)'] == 3840
            close(np.mean(h['training']['body_cost']), h['training']['mean_cost'])
            close(np.mean(h['training']['precise_goals']), h['training']['precision_fraction'])
            previous = h['log_end']
        assert sum(x['key'] == 'train-update' for x in r['local_logs']) == 6400
        assert sum(x['key'] == 'val' for x in r['local_logs']) == 4
        assert r['training_interactions'] == 128*64*240
        assert r['validation_interactions'] == 4*64*240
        assert not (R/directory/'failure.json').exists()
        for c in r['candidates']:
            selected_row = next(a for a in s['candidates'] if a['training_lr'] == r['lr'] and a['epoch'] == c['epoch'])
            wp, vp = R/directory/c['weights'], R/directory/c['validation']
            assert sha(wp) == c['weights_sha256'] == selected_row['weights_sha256']
            assert sha(vp) == selected_row['validation_sha256']
            assert sha(rp) == selected_row['fit_report_sha256']
            pair, row = score(vp, c['metrics'])
            if baseline is None:
                baseline = pair
            for k in baseline:
                np.testing.assert_array_equal(pair[k], baseline[k])
            close(row['mean_cost'], selected_row['mean_validation_cost'])
            close(row['precise_goals']/192, selected_row['validation_precision'])
            row.update(lr=r['lr'], epoch=c['epoch'], weights_sha256=sha(wp))
            rows.append(row)
        reports.append(dict(lr=r['lr'], epochs=128, gradient_batches=6400,
                            wall_seconds=r['wall_seconds'], report_sha256=sha(rp)))
    winner = min(rows, key=lambda x: x['mean_cost'])
    assert winner['lr'] == s['selected']['training_lr'] and winner['epoch'] == s['selected']['epoch']
    close(winner['mean_cost'], s['selected']['mean_validation_cost'])
    assert s['validation_precision_threshold'] == .75
    assert s['validation_gate'] == (winner['precise_goals']/192 >= .75) is False
    assert s['training_interactions'] == 3932160 and s['validation_interactions'] == 122880
    assert s['official_gradient_minibatches'] == 12800
    d = load(R/'amago_control_rls_validation_diagnostic.json')
    assert d['rho'] == .98 and d['status'].startswith('SECONDARY PAIRED VALIDATION')
    pair, rls = score(R/'amago_control_rls_validation_diagnostic.npz', d)
    for k in baseline:
        np.testing.assert_array_equal(pair[k], baseline[k])
    assert d['interactions'] == 15360 and d['simulated_seconds'] == 768
    artifacts['amago_control_rls_validation_diagnostic.json'] = sha(R/'amago_control_rls_validation_diagnostic.json')
    engineering = []
    for name, counts in [('amago_control_smoke_report.json', (1920, 960)), ('amago_control_fullbatch_profile.json', (15360, 15360))]:
        e = load(R/name)
        assert (e['training_interactions'], e['validation_interactions']) == counts
        artifacts[name] = sha(R/name)
        engineering.append(dict(path=name, training=counts[0], validation=counts[1]))
    # Filename scan supplements the false gate/source review; it is not an OS execution audit.
    fresh_files = [str(p.relative_to(R)) for p in R.glob('amago_control*') if 'confirmation' in p.name or 'fresh' in p.name]
    assert not fresh_files, fresh_files
    return dict(status='PASS', checker_sha256=sha(__file__), frozen_git_commit=COMMIT,
                source_hashes=s['sources'], official_manifest_files_checked=len(manifest['files']),
                artifact_sha256=artifacts, learned_validation=rows, selected=winner,
                fit_logs=reports, secondary_fixed_rho_rls=rls, validation_gate=False,
                interaction_counts=dict(training=3932160, validation=122880,
                    secondary_rls_diagnostic=15360, main_plus_secondary=4070400,
                    discarded_engineering=33600, engineering_reports=engineering,
                    fresh_confirmation=0, tiny_mocked_adapter_checks='separate; no physical transitions'),
                saved_transitions_recomputed=9*64*240,
                scope='Raw trajectories, arithmetic, paired draws, literal selection, hashes and recorded official logs. No independent neural forwards, optimizer replay or physical integration. Fresh absence is saved-artifact/source evidence, not exhaustive process tracing.')

if __name__ == '__main__':
    target = R/'check_amago_control_results.json'
    if target.exists():
        raise FileExistsError(target)
    try:
        result = run()
    except BaseException as exc:
        target.write_text(json.dumps(dict(status='FAIL', error=repr(exc), checker_sha256=sha(__file__)), indent=2)+'\n')
        raise
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], saved_transitions_recomputed=result['saved_transitions_recomputed'],
                         selected_cost=result['selected']['mean_cost'], selected_precision=result['selected']['precise_goals'],
                         rls_cost=result['secondary_fixed_rho_rls']['mean_cost'], rls_precision=result['secondary_fixed_rho_rls']['precise_goals'])))
