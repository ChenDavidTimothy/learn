#!/usr/bin/env python3
"""Apply the prospectively frozen continuation rules; no fitting or tuning."""
import json
from pathlib import Path
import numpy as np
from error_transport_screen import digest,save_json

ROOT=Path(__file__).resolve().parent
def decision():
    path=ROOT/'implicit_relation_results.json';r=json.loads(path.read_text());results=r['results']
    means={m:{s:{n:float(np.mean(v)) for n,v in d['standardized_body_mse'].items()} for s,d in sets.items()} for m,sets in results.items()}
    failures=[];minsv=1e300;maxcond=0.
    for split,d in results['implicit'].items():
        for n,g in d['diagnostics'].items():
            minsv=min(minsv,min(g['min_singular']));maxcond=max(maxcond,max(g['max_condition']))
            failures += [dict(split=split,budget=n,body=i) for i,f in enumerate(g['conditioning_failure']) if f]
    strongest=lambda s,n:min(means[m][s][str(n)] for m in ('alpaca','coda'))
    accuracy={}
    for n in (16,64):
        other=64 if n==16 else 16
        accuracy[str(n)]=dict(
            advantage_over_both_strong_and_nested=means['implicit']['test_ood'][str(n)]<=.9*min(strongest('test_ood',n),means['explicit']['test_ood'][str(n)]),
            id_within_5_percent_at_both=all(means['implicit']['test_id'][str(k)]<=1.05*strongest('test_id',k) for k in (16,64)),
            opposite_ood_within_5_percent=means['implicit']['test_ood'][str(other)]<=1.05*strongest('test_ood',other),
            correct_support_better_than_wrong=means['implicit']['test_ood'][str(n)]<means['implicit']['test_ood_wrong_support'][str(n)],
            no_conditioning_failure=not failures)
    passes_accuracy=any(all(c.values()) for c in accuracy.values())
    near_parity=all(means['implicit'][s][str(n)]<=1.05*strongest(s,n) for n in (16,64) for s in ('test_id','test_ood')) and not failures
    # Timing is owed only when accuracy can meet the prospective efficiency route.
    outcome='ACCURACY SCREEN SURVIVES; CONFIRMATION OWED' if passes_accuracy else ('ACCURACY PARITY ONLY; REPEATED FAIR TIMING OWED' if near_parity else 'REJECT SECOND CANDIDATE; NO RESCUE TUNING')
    return dict(status=outcome,means=means,accuracy_gate=accuracy,accuracy_gate_pass=passes_accuracy,efficiency_accuracy_prerequisite=near_parity,
        conditioning_failures=failures,minimum_query_singular_value=minsv,maximum_query_condition=maxcond,
        results_sha256=digest(path),source_sha256=digest(__file__),scope='Exploratory gate, no significance or framework claim. ID guard evaluated conservatively at both support budgets. No timing claim from fit logs.')

if __name__=='__main__':
    out=decision();save_json(ROOT/'implicit_relation_decision.json',out);print(json.dumps(out,indent=2))
