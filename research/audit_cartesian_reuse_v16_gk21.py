#!/usr/bin/env python3
"""Guarded comparison to independent complete-menu integration and replay chain."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=ROOT/'cartesian_reuse_v16_gk21.py';design=ROOT/'cartesian_reuse_v16_gk21_design.md';rp=ROOT/'cartesian_reuse_v16_gk21_results.json';dp=ROOT/'cartesian_reuse_v16_gk21_data.npz'
    parentpath=ROOT/'cartesian_reuse_v16_adaptive_results.json';ap=ROOT/'audit_cartesian_reuse_v16_adaptive_results.json'
    r=json.loads(rp.read_text());old=json.loads(parentpath.read_text());a=json.loads(ap.read_text())
    assert sha(source)==r['meta']['source_sha256'];assert sha(dp)==r['meta']['data_sha256']
    assert sha(ROOT/'audit_cartesian_reuse_v16_adaptive.py')==a['audit_source_sha256']
    for name,h in a['artifact_sha256'].items():assert sha(ROOT/name)==h,name
    unchanged=['mixture_candidates','selected_mixture','gaussian_candidates','selected_gaussian','gaussian_gradient_checks','priors','forecasts','feasibility_witnesses','candidate_leading_feasibility_evaluator_only','controls','resources']
    for name in unchanged:assert r[name]==old[name],name
    assert sha(dp)==sha(ROOT/'cartesian_reuse_v16_adaptive_data.npz')
    rows=[];differences=[];statuses=[];tails=[];errors=[]
    for row in a['complete_candidate_integrals']:
        name=row['prior'];gi=row['goal_index'];d=r['first_designs'][name][gi];values=np.array(d['expected_costs']);ref=np.zeros(50)
        assert len(row['checks'])==len(values)==50
        for c in row['checks']:ref[c['candidate']]=c['independent_value']
        diff=abs(values-ref);assert diff.max()<1e-8;differences.extend(diff.tolist())
        assert d['selected']==int(np.argmin(values))==int(np.argmin(ref))
        rv=[ref[0]]+[float(ref[1+16*i:1+16*(i+1)].mean()) for i in range(3)];radius=[0.,.025,.05,.1][int(np.argmin(rv))]
        assert radius==d['random_radius']
        for info in d['integration']:
            assert info['rule']=='gk21' and info['atol']==1e-8 and info['rtol']==1e-7
            assert all(s=='converged' for s in info['statuses']);statuses.extend(info['statuses']);tails.append(info['analytic_truncation_upper_bound']);errors.append(info['estimated_error'])
        rows.append({'prior':name,'goal_index':gi,'maximum_score_difference':float(diff.max()),'selected':d['selected'],'random_radius':radius})
    summary={k:{'episodes':len(v),'mean_cost':float(np.mean([q['cost'] for q in v])),'precision_successes':sum(q['precision_success'] for q in v)} for k,v in r['controls'].items()}
    out={'status':'INDEPENDENT COMPLETE-MENU NUMERICAL COMPARISON PASSED; CONSTRUCTION PILOT ONLY',
        'artifact_sha256':{p.name:sha(p) for p in [source,design,rp,dp,parentpath,ap,ROOT/'audit_cartesian_reuse_v16_adaptive.py']},'audit_source_sha256':sha(Path(__file__)),
        'candidate_integrals_checked':len(differences),'maximum_score_difference_from_independent_implementation':max(differences),'menus':rows,'exactly_unchanged_sections':unchanged,
        'maximum_saved_estimated_integration_error':max(errors),'maximum_analytic_truncation_upper_bound':max(tails),'outcomes':summary,
        'limitations':['Independent code implements likelihood/posterior/response/QP separately; both numerical integrations use SciPy adaptive gk21, with independent tighter tolerance.',
            'Four earlier Genz-Malik failures were additionally checked with independent scrambled Sobol; no rigorous cubature error theorem is claimed.',
            'Same physical outcomes, acquired fit and noise cohort as the exploratory adaptive batch; not a fresh confirmatory test or sustained feedback.',
            'All900 finite-menu integrals checked; no globally optimal continuous first-action claim.']}
    (ROOT/'audit_cartesian_reuse_v16_gk21_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'checked':len(differences),'max_difference':max(differences),'control_equal':True}))

if __name__=='__main__':main()
