#!/usr/bin/env python3
"""Frozen exploratory gate for later learning after feature acquisition."""
import json
from pathlib import Path
import numpy as np
from error_transport_screen import save_json,digest

ROOT=Path(__file__).resolve().parent
def decide():
    path=ROOT/'structure_acquisition_results.json';r=json.loads(path.read_text())
    means={m:{s:{n:float(np.mean(v)) for n,v in d['standardized_body_mse'].items()} for s,d in groups.items()} for m,groups in r['results'].items()}
    strongest=lambda s,n:min(means[m][s][str(n)] for m in ('neural','coda'))
    gates={}
    for n in (16,64):
        other=64 if n==16 else 16
        gates[str(n)]=dict(
            new_id_improves_10_percent_over_neural_and_frozen=means['meta']['new_id'][str(n)]<=.9*min(strongest('new_id',n),means['frozen']['new_id'][str(n)]),
            other_new_id_budget_within_5_percent=means['meta']['new_id'][str(other)]<=1.05*strongest('new_id',other),
            new_ood_within_5_percent_at_both=all(means['meta']['new_ood'][str(k)]<=1.05*strongest('new_ood',k) for k in (16,64)),
            retention_within_5_percent_at_both=all(means['meta']['old_id'][str(k)]<=1.05*strongest('old_id',k) for k in (16,64)),
            correct_support_beats_wrong=means['meta']['new_id'][str(n)]<means['meta']['new_id_wrong_support'][str(n)])
    survive=any(all(g.values()) for g in gates.values())
    criterion_benefit={str(n):means['meta']['new_id'][str(n)]<=.9*means['fit']['new_id'][str(n)] for n in (16,64)}
    return dict(status='EXPLORATORY SCREEN SURVIVES; STRONGER CONFIRMATION OWED' if survive else 'REJECT THIS FEATURE-ACQUISITION FORMULATION; NO TEST-INFORMED GRAMMAR REPAIR',
        means=means,continuation_gates=gates,continuation_pass=survive,post_support_criterion_beats_fit_by_10_percent=criterion_benefit,
        source_sha256=digest(__file__),results_sha256=digest(path),
        scope='Single trained seed and acquired corpus. Guarded at both budgets for OOD/retention. No speed/significance/control/framework claim.')

if __name__=='__main__':
    r=decide();save_json(ROOT/'structure_acquisition_decision.json',r)
    print(json.dumps({k:r[k] for k in ('status','continuation_gates','post_support_criterion_beats_fit_by_10_percent')},indent=2))
