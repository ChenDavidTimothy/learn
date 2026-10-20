#!/usr/bin/env python3
"""Build the verified v17 handoff, preserving every preceding v16 byte.

All scientific inputs are immutable snapshots. --check verifies both the release
manifest and exact ledger reconstruction; it writes nothing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TARGET = ROOT/'first_principles_robot_learning_ledger_v17.md'
MANIFEST = ROOT/'ledger_v17_release_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT/name).read_text())


def verify_hash_map(mapping):
    for name, expected in mapping.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        assert path.is_file() and sha(path) == expected, ('stale input', str(path))


def verified(name):
    record = read(name)
    status = record['status']
    assert status.startswith('PASS') or 'PASSED' in status, (name, status)
    for key in ('artifact_sha256', 'final_artifact_sha256', 'audit_cache_and_source_sha256'):
        if key in record:
            verify_hash_map(record[key])
    return record


def gates():
    # Earlier pilots remain evidence, with their stated finite scopes.
    for name in ('audit_cartesian_moving_v17_instrument_results.json',
                 'audit_cartesian_moving_v17_theory_results.json',
                 'audit_cartesian_stream_v17_acquisition_results.json',
                 'audit_cartesian_stream_v17_fit_results.json',
                 'audit_cartesian_stream_v17_forks_results.json',
                 'audit_cartesian_stream_v17_control_results.json',
                 'audit_cartesian_stream_v17_control_physics_results.json',
                 'audit_cartesian_stream_v17_retention_results.json',
                 'audit_cartesian_stream_v17_retention_physics_results.json',
                 'audit_cartesian_stream_v17_action_risk_results.json',
                 'audit_cartesian_stream_v17_study_preflight_results.json'):
        verified(name)
    for rep in range(200, 208):
        ordinary = verified(f'audit_cartesian_stream_v17_study_rep{rep}_results.json')
        physical = verified(f'audit_cartesian_stream_v17_rep{rep}_study_physics_results.json')
        assert ordinary['replicate'] == physical['replicate'] == rep
        assert ordinary['artifact_sha256']['audit_cartesian_stream_v17_study.py'] == sha(ROOT/'audit_cartesian_stream_v17_study.py')
        assert physical['audit_source_sha256'] == sha(ROOT/'audit_cartesian_stream_v17_study_physics.py')
    verified('audit_cartesian_stream_v17_study_results.json')
    physics_final = verified('audit_cartesian_stream_v17_study_physics_results.json')
    assert physics_final['aggregator_source_sha256'] == sha(ROOT/'aggregate_cartesian_stream_v17_study_physics_audit.py')
    verified('audit_cartesian_stream_v17_study_aggregate_results.json')
    optimization = verified('cartesian_stream_v17_optimizer_certificate_results.json')
    verify_hash_map(optimization['input_and_final_audit_sha256'])
    assert optimization['source_sha256'] == sha(ROOT/'cartesian_stream_v17_optimizer_certificate.py')
    assert optimization['design_sha256'] == sha(ROOT/'cartesian_stream_v17_optimizer_certificate_posthoc_design.md')
    assert optimization['arrays_sha256'] == sha(ROOT/'cartesian_stream_v17_optimizer_certificate_arrays.npz')
    assert optimization['scope']['ordinary_decisions'] == 11520
    assert optimization['scope']['replicates'] == list(range(200,208))
    assert not optimization['reconstruction_failures']
    aggregate = read('cartesian_stream_v17_study_aggregate_results.json')
    assert aggregate['publication_eligible'] and aggregate['replicates'] == list(range(200, 208))
    assert aggregate['analysis_source_sha256'] == sha(ROOT/'analyze_cartesian_stream_v17_study.py')
    verify_hash_map(aggregate['artifact_sha256'])
    assert len(aggregate['failures']['precision_failures']) == 310
    assert aggregate['resources']['ordinary_decisions'] == 11520
    assert aggregate['resources']['total_physical_intervals_acquisition_ordinary_reference'] == 119552
    assert [aggregate['summary'][m]['precision_count'] for m in ('retained','frozen','privileged_reference')] == [55,19,192]
    return aggregate


def assemble(a):
    prefix = (ROOT/'ledger_v17_prefix.md').read_text()
    names = {'retained':'Acquired local relation, retain new rows',
             'frozen':'Acquired local relation, frozen library',
             'privileged_reference':'Privileged exact-state nominal reference'}
    table = ['| Mechanism | Mean 18 sec task cost +/- corpus SE | Sustained precise windows /192 |',
             '| --- | ---: | ---: |']
    for name, label in names.items():
        row = a['summary'][name]; cost = row['task_cost_18sec']
        table.append(f"| {label} | {cost['mean']:.10g} +/- {cost['se_across_eight_corpora']:.6g} | {row['precision_count']} |")
    corpus_table = ['| Fresh corpus | Retained task cost | Frozen task cost | Retained / frozen precise windows (each /24) |',
                    '| --- | ---: | ---: | ---: |']
    for corpus in a['corpora']:
        r, f = (corpus['means'][name] for name in ('retained','frozen'))
        corpus_table.append(f"| {corpus['replicate']} | {r['task_cost_18sec']:.9g} | {f['task_cost_18sec']:.9g} | {r['precision_count']} / {f['precision_count']} |")
    r, f = (a['summary'][name] for name in ('retained','frozen'))
    profiles = (f"Retained passes {r['first_precision_count']}/64 first and {r['later_precision_count']}/128 later windows; "
                f"frozen passes {f['first_precision_count']}/64 first and {f['later_precision_count']}/128 later windows.")
    replacements = {'FRESH_TABLE':'\n'.join(table), 'CORPUS_TABLE':'\n'.join(corpus_table),
                    'PRECISION_PROFILE':profiles}
    for key, value in replacements.items():
        prefix = prefix.replace('{{'+key+'}}', value)
    assert not re.search(r'\{\{[^}]+\}\}', prefix), 'unresolved ledger placeholders'
    prior = (ROOT/'first_principles_robot_learning_ledger_v16.md').read_bytes()
    output = prefix.encode()+prior
    assert output.endswith(prior)
    return output


def inputs():
    # Include the complete v17 evidence lineage, not only the final good result.
    paths = set()
    for pattern in ('cartesian*v17*', 'audit_cartesian*v17*',
                    'aggregate_cartesian*v17*', 'analyze_cartesian*v17*'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and p.suffix in ('.py','.md','.json','.npz'))
    paths.update((Path(__file__).resolve(), ROOT/'ledger_v17_prefix.md',
                  ROOT/'first_principles_robot_learning_ledger_v16.md'))
    return {p.name:sha(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        manifest = read(MANIFEST.name)
        verify_hash_map(manifest['artifact_sha256'])
    else:
        assert not TARGET.exists() and not MANIFEST.exists(), 'released ledgers/manifests are immutable'
    aggregate = gates()
    content = assemble(aggregate)
    if args.check:
        assert TARGET.read_bytes() == content
        assert sha(TARGET) == manifest['ledger_sha256']
        print('PASS: all publication gates, release hashes and exact v17/v16 reconstruction')
        return
    manifest = dict(status='VERIFIED V17 RELEASE INPUT MANIFEST; GENERAL LEARNING GOAL REMAINS ACTIVE',
                    ledger_sha256=hashlib.sha256(content).hexdigest(),
                    complete_preceding_v16_sha256=sha(ROOT/'first_principles_robot_learning_ledger_v16.md'),
                    artifact_sha256=inputs())
    TARGET.write_bytes(content)
    MANIFEST.write_text(json.dumps(manifest, indent=2)+'\n')
    print('saved', TARGET, 'with byte-identical complete v16 suffix')


if __name__ == '__main__':
    main()
