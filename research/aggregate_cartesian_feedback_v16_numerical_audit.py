#!/usr/bin/env python3
"""Bind completed independent numerical checks to FINAL fresh-study artifacts."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture_of(result,c):
 p=result['episodes'][c['episode']]['phases'][c['phase']];d=p['design'];i=c['index']
 refined=next((q for q in d['tighter_top_three'] if q['index']==i),None);record=refined['integration'] if refined else d['integration'][i]
 return dict(belief=p['belief_before'],goal=p['goal'],action=d['candidates'][i],value=d['values'][i],integration=record)

def main():
 artifacts={};reports=[];allchecks=[];fullmenus=[];groups=[];maximum=0.;failures=[];ranking=[];radii=[];priors=('mixture','gaussian','matched','isotropic')
 for rep in range(100,108):
  rp=ROOT/f'cartesian_feedback_v16_rep{rep}_results.json';result=json.loads(rp.read_text());assert len(result['episodes'])==80
  source=[p for p in ROOT.glob('cartesian_feedback_v16_study*.py') if sha(p)==result['meta']['source_sha256']];assert source
  artifacts[rp.name]=sha(rp);artifacts[source[0].name]=sha(source[0]);records=[]
  filenames=[f'audit_cartesian_feedback_v16_fresh_numerics_rep{rep}_initial_results.json',f'audit_cartesian_feedback_v16_fresh_numerics_rep{rep}_later_results.json'] if rep<104 else [f'audit_cartesian_feedback_v16_fresh_priority_rep{rep}_results.json']
  for name in filenames:
   path=ROOT/name;r=json.loads(path.read_text());assert r['input_sha256']==sha(rp),(rep,name,'not bound to final input')
   assert r['replicate']==rep and r['rule']=='gk15' and r['partition_edges']==[-7,-2,2,7] and r['atol']==1e-10 and r['rtol']==1e-9
   src=ROOT/('audit_cartesian_feedback_v16_fresh_numerics.py' if rep<104 else 'audit_cartesian_feedback_v16_fresh_priority.py');assert r['audit_source_sha256']==sha(src)
   if rep>=104:
    assert r['input_complete'] and r['input_episodes']==80 and not r['missing_jobs']
    assert r['integral_helper_sha256']==sha(ROOT/'audit_cartesian_feedback_v16_fresh_numerics.py')
    assert r['independent_rawmoment_helper_sha256']==sha(ROOT/'audit_cartesian_feedback_v16_pilot.py')
   else:
    for name,h in r['independent_helper_sha256'].items():assert sha(ROOT/name)==h
   artifacts[name]=sha(path);artifacts[src.name]=sha(src)
   records.extend(r['checks'])
  assert len(records)==(304 if rep<104 else 56),(rep,len(records))
  keys=[(c['episode'],c['phase'],c['index']) for c in records];assert len(set(keys))==len(keys)
  initial_groups={};late_groups={}
  for c in records:
   e=result['episodes'][c['episode']];p=e['phases'][c['phase']];d=p['design'];i=c['index'];assert e['mode']=='dual' and c['prior']==e['prior'] and c['sequence']==e['sequence']
   assert c['original_selected']==d['selected'] and c['original_value']==d['values'][i]
   expected=fixture_of(result,c);assert c['original_estimated_error']==expected['integration']['estimated_error']
   if rep>=104:
    assert c['fixture']==expected and c['fixture_sha256']==hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
   assert abs(c['difference']-(c['value']-c['original_value']))<1e-20
   assert all(q['status']=='converged' for q in c['parts'])
   delta=abs(c['difference']);maximum=max(maximum,delta);limit=max(2e-7,5*(c['estimated_error']+c['original_estimated_error']))
   if delta>limit:failures.append(dict(rep=rep,episode=c['episode'],phase=c['phase'],index=i,difference=c['difference'],threshold=limit))
   if c['phase']==0:initial_groups.setdefault((e['prior'],e['sequence']),[]).append(c)
   else:
    assert c['phase'] in(2,5) and e['branch_evaluator_only']==(rep+e['sequence'])%2 and e['repeat']==(rep//2)%2
    late_groups.setdefault((e['prior'],e['sequence'],c['phase']),[]).append(c)
   allchecks.append((rep,c))
  assert set(initial_groups)=={(p,s) for p in priors for s in (0,1)}
  assert set(late_groups)=={(p,s,t) for p in priors for s in(0,1) for t in(2,5)}
  for group in list(initial_groups.values())+list(late_groups.values()):
   e=result['episodes'][group[0]['episode']];ph=group[0]['phase'];d=e['phases'][ph]['design'];lookup={c['index']:c for c in group};indices=set(lookup)
   expected=set(range(34)) if rep<104 and ph==0 else set(map(int,np.argsort(d['values'])[:3 if ph==0 else 2]))
   assert indices==expected
   best=min(group,key=lambda c:c['value']);selected=d['selected'];losschange=lookup[selected]['value']-best['value']
   row=dict(replicate=rep,prior=e['prior'],sequence=e['sequence'],phase=ph,episode=group[0]['episode'],indices=sorted(indices),original_selected=selected,audited_best=best['index'],audited_selected_excess_over_checked_min=losschange)
   if best['index']!=selected:ranking.append(row)
   groups.append(row)
   if len(group)==34:
    values={i:c['value'] for i,c in lookup.items()};scores=[values[0],float(np.mean([values[i] for i in range(1,17)])),float(np.mean([values[i] for i in range(17,33)]))]
    radius=[0.,.05,.1][int(np.argmin(scores))];rr=dict(replicate=rep,prior=e['prior'],sequence=e['sequence'],scores=scores,original_scores=d['random_radius_values'],radius=radius,original_radius=d['random_radius'],maximum_score_difference=float(np.max(abs(np.array(scores)-d['random_radius_values']))));radii.append(rr);fullmenus.append(row)
  reports.append(dict(replicate=rep,initial_integrals=sum(map(len,initial_groups.values())),later_integrals=sum(map(len,late_groups.values())),full_initial_menus=8 if rep<104 else 0,maximum_absolute_difference=max(abs(c['difference']) for c in records)))
 assert len(allchecks)==1440 and len(fullmenus)==32 and len(groups)==192
 assert not failures,failures
 artifacts['audit_cartesian_feedback_v16_fresh_numerics.py']=sha(ROOT/'audit_cartesian_feedback_v16_fresh_numerics.py')
 artifacts['audit_cartesian_feedback_v16_pilot.py']=sha(ROOT/'audit_cartesian_feedback_v16_pilot.py')
 artifacts['audit_cartesian_reuse_v16_pilot.py']=sha(ROOT/'audit_cartesian_reuse_v16_pilot.py')
 out=dict(status='PASS — independent actual-integral numerical checks under declared completed scope; no analytic/global certificate',artifact_sha256=artifacts,aggregate_audit_source_sha256=sha(Path(__file__)),scope=dict(corpora=8,priors=4,sequences=2,initial_histories=64,initial_top_three_checked_everywhere=True,later_histories=128,later_phases_zero_based=[2,5],later_candidates_per_history=2,later_branch_rule='(rep+sequence)%2',later_repeat_rule='(rep//2)%2',full_initial_menus=32,full_initial_menu_corpora=[100,101,102,103],initial_candidates_full_menu=34,integrals=1440,initial_integrals=1184,later_integrals=256,balanced_policy_relevant_integrals=448,initial_random_radius_averages_checked=32,later_random_radius_averages_checked=0),rule=dict(name='gk15',partition_edges=[-7,-2,2,7],atol=1e-10,rtol=1e-9,independent_core='Raw8Dsecondmoment terminal-map QP, separately reconstructed Gaussian conditioning; no canonical controller imports'),maximum_absolute_difference=maximum,per_corpus=reports,threshold_failures=failures,checked_ranking_changes=ranking,random_radius_changes=[r for r in radii if r['radius']!=r['original_radius']],full_initial_radius_checks=radii,checked_histories=groups,total_independent_integrand_points=sum(c['points'] for _,c in allchecks),summed_per_integral_wall_seconds=sum(c['wall_seconds'] for _,c in allchecks),limitations=['Original full2176initial-integral target deliberately curtailed after sufficient policy-relevant stratified coverage; all completed full menus preserved.','All initial top3 and fixed laterselected/runner strata span all8corpora/4priors/2sequences; unchecked later histories/candidates remain unverified.','Complete initial direction-averaged radius scores checked for100..103 only. Rep107tinyK3mode selected/runner forecasts checked, but its complete hypothetical menu and randomradius averages NOT independently integrated.','Numerical convergence/error estimates are not rigorous enclosures or a global stochastic-control guarantee; analytic Gaussian truncation bound is separate.','No new physical interactions; repeated use of identical frozen corpus/control artifacts adds numerical confidence, not independent physical samples.','The study is fixed prepared localconstantB engineering evidence, not general learning necessity or modernmethod superiority.'])
 path=ROOT/'audit_cartesian_feedback_v16_study_numerical_results.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','scope','maximum_absolute_difference','checked_ranking_changes','random_radius_changes','total_independent_integrand_points')}))
if __name__=='__main__':main()
