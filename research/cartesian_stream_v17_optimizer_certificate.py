#!/usr/bin/env python3
"""Posthoc all-decision convex box bound, with independent saved-model replay.
No canonical imports, model fits, optimization solves or physical rollouts.
Exact-arithmetic theorem; floating-point values are numerically checked only.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
from audit_cartesian_stream_v17_fit import paths,context
from audit_cartesian_stream_v17_precontrol import cost_gradient
ROOT=Path(__file__).resolve().parent
REPS=tuple(range(200,208));BOUND=.5
NUMERICAL=ROOT/'audit_cartesian_stream_v17_study_results.json'
PHYSICAL=ROOT/'audit_cartesian_stream_v17_study_physics_results.json'
DESIGN=ROOT/'cartesian_stream_v17_optimizer_certificate_posthoc_design.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(x,g):return math.fsum(float(abs(v)*(.5+np.sign(v)*a)) for a,v in zip(x.ravel(),g.ravel()))
def checkclose(x,y,atol=2e-9,rtol=5e-8,label='numeric'):
 err=float(abs(np.asarray(x)-np.asarray(y)).max());assert np.allclose(x,y,atol=atol,rtol=rtol),(label,err);return err
def verify_audits():
 numerical=json.loads(NUMERICAL.read_text());physical=json.loads(PHYSICAL.read_text());assert numerical['status'].startswith('PASS ALL EIGHT') and numerical['total_decisions']==11520 and physical['status'].startswith('PASS') and physical['replicates']==list(REPS)
 hashes={NUMERICAL.name:sha(NUMERICAL),PHYSICAL.name:sha(PHYSICAL),DESIGN.name:sha(DESIGN)}
 for name,h in numerical['artifact_sha256'].items():assert sha(ROOT/name)==h;hashes[name]=h
 for name,h in physical['audit_cache_and_source_sha256'].items():assert sha(ROOT/name)==h;hashes[name]=h
 for name,h in physical['final_artifact_sha256'].items():assert sha(ROOT/name)==h;hashes[name]=h
 for rep in REPS:
  p=ROOT/f'audit_cartesian_stream_v17_study_rep{rep}_results.json';a=json.loads(p.read_text());assert a['status'].startswith('PASS') and a['replicate']==rep
  for name,h in a['artifact_sha256'].items():assert sha(Path(a['artifact_paths'][name]))==h;hashes[name]=h
  assert sha(p)==numerical['artifact_sha256'][p.name]
 return hashes

def fixtures():
 # Analytic values only; no optimization solve.
 a=bound(np.array([.25]),np.array([.5]));assert a==.375 and .25**2<=a
 box=bound(np.array([.5]),np.array([-1.]));assert box==0 # f=(x-1)^2 at its box optimum
 nonconvex=bound(np.array([0.]),np.array([0.]));assert nonconvex==0 and .25>nonconvex
 stale=bound(np.array([-.5]),np.array([0.]));assert stale==0 and 1>stale
 return dict(convex_quadratic_x_point25=dict(bound=a,actual_gap=.0625),outward_normal_box_optimum=dict(bound=box,raw_gradient_norm=1.),nonconvex_failure=dict(false_bound=nonconvex,actual_gap=.25),stale_gradient_failure=dict(false_bound=stale,actual_gap=1.))
def summary(records):
 def describe(key):
  values=[r[key] for r in records if r[key] is not None]
  if not values:return dict(defined=0)
  x=np.array(values);return dict(defined=len(values),mean=float(x.mean()),quantiles={str(q):float(np.quantile(x,q)) for q in(0,.5,.9,.99,1)})
 absolute=max(records,key=lambda r:r['absolute_bound']);defined=[r for r in records if r['relative_to_selected_objective'] is not None];relative=max(defined,key=lambda r:r['relative_to_selected_objective']) if defined else None
 return dict(decisions=len(records),selected_objective=describe('selected_objective'),absolute_bound=describe('absolute_bound'),relative_to_selected_objective=describe('relative_to_selected_objective'),worst_absolute=absolute,worst_relative=relative)
def main():
 start=time.time();output=ROOT/'cartesian_stream_v17_optimizer_certificate_results.json';arraypath=ROOT/'cartesian_stream_v17_optimizer_certificate_arrays.npz';assert not output.exists() and not arraypath.exists()
 hashes=verify_audits();tests=fixtures();records=[];plans=[];gradients=[];replay_gradients=[];locations=[];replay_errors=dict(cost=0.,gradient=0.,query=0.,projected_gradient=0.,stable_vs_raw_bound=0.,stable_vs_replayed_bound=0.);allgroups={};percorpus=[];reconstruction_failures=[]
 for rep in REPS:
  prefix=ROOT/f'cartesian_stream_v17_rep{rep}';rp=Path(str(prefix)+'_study_results.json');ap=Path(str(prefix)+'_study_arrays.npz');fp=Path(str(prefix)+'_local_fit.json');result=json.loads(rp.read_text());fit=json.loads(fp.read_text());scale={k:np.array(v,float) for k,v in fit['scale'].items()}
  for p in(rp,ap,fp):assert hashes[p.name]==sha(p)
  with np.load(ap,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
  before=len(records)
  for episode_index,e in enumerate(result['ordinary_episodes']):
   assert e['variant'] in('retained','frozen') and len(e['decisions'])==90;k=e['array_keys'];observations=arrays[k['observed']];commands=arrays[k['commands']];coefficients=arrays[k['local_fit_coefficients']];queries=arrays[k['local_fit_queries']]
   assert coefficients.shape==(90,3,69,4) and queries.shape==(90,66)
   for index,d in enumerate(e['decisions']):
    t=16+2*index;phase=index//30;assert d['time_index']==t and d['renewal']==index and d['phase']==phase
    plan=np.array(d['selected_plan'],float);g=np.array(d['selected_gradient'],float);goal=np.array(d['current_goal'],float);f=float(d['selected_cost'])
    assert plan.shape==g.shape==(6,2) and np.isfinite(plan).all() and np.isfinite(g).all() and np.isfinite(f) and f>=0 and np.max(abs(plan))<=.5
    checkclose(plan[0],commands[t],atol=0,rtol=0,label='actual first action');checkclose(plan[0],d['command'],atol=0,rtol=0,label='stored command');checkclose(goal,e['goals'][phase],atol=0,rtol=0,label='current goal')
    obs=observations[:t+1];act=commands[:t];query=(context(obs,act)-scale['context_mean'])/scale['context_std'];qe=checkclose(query,queries[index],atol=1e-12,rtol=1e-12,label='actual query');replay_errors['query']=max(replay_errors['query'],qe)
    prediction,jac=paths(obs,act,plan,scale,list(coefficients[index]),'local',True);cost,rg=cost_gradient(prediction,jac,obs[-1],goal,plan)
    ce=checkclose(cost,f,label='selected cost');ge=checkclose(rg,g,atol=2e-8,rtol=5e-7,label='selected gradient');replay_errors['cost']=max(replay_errors['cost'],ce);replay_errors['gradient']=max(replay_errors['gradient'],ge)
    pg=plan-np.clip(plan-g,-.5,.5);pe=checkclose(pg,d['selected_projected_gradient'],atol=2e-8,rtol=5e-7,label='projected gradient');checkclose(abs(pg).max(),d['selected_projected_gradient_max'],atol=2e-8,rtol=5e-7,label='PG maximum');replay_errors['projected_gradient']=max(replay_errors['projected_gradient'],pe)
    certificate=bound(plan,g);raw=float(np.sum(plan*g)+.5*np.sum(abs(g)));rb=bound(plan,rg);assert certificate>=0 and np.isfinite(certificate)
    replay_allowance=math.fsum(float(abs(a-b)*(.5+abs(x))) for a,b,x in zip(rg.ravel(),g.ravel(),plan.ravel()))
    rounding=64*np.finfo(float).eps*(1+abs(raw)+certificate+np.sum(abs(g)))
    assert abs(certificate-rb)<=replay_allowance+rounding
    replay_errors['stable_vs_raw_bound']=max(replay_errors['stable_vs_raw_bound'],abs(raw-certificate));replay_errors['stable_vs_replayed_bound']=max(replay_errors['stable_vs_replayed_bound'],abs(rb-certificate))
    record=dict(replicate=rep,variant=e['variant'],episode_index=episode_index,case_index=e['case_index'],context_id=e['context_id'],noise_repeat=e['noise_repeat'],phase=phase,renewal=index,time_index=t,selected_objective=f,absolute_bound=certificate,raw_dot_norm_bound=raw,replayed_gradient_bound=rb,relative_to_selected_objective=(certificate/f if f>0 else None),modeled_optimum_lower_bound=max(0.,f-certificate),selected_cost_replay_error=ce,selected_gradient_max_replay_error=ge,observed_gradient_difference_bound_allowance=replay_allowance)
    records.append(record);plans.append(plan);gradients.append(g);replay_gradients.append(rg);locations.append([rep,0 if e['variant']=='retained' else 1,episode_index,e['case_index'],phase,index,t])
    group=(rep,e['variant'],phase);allgroups.setdefault(group,[]).append(record)
  assert len(records)-before==1440;percorpus.append(dict(replicate=rep,summary=summary(records[before:])))
  print('certificate replay complete',rep,flush=True)
 assert len(records)==11520 and len(allgroups)==48 and all(len(r)==240 for r in allgroups.values())
 # Current final artifacts must still match after computation.
 assert hashes==verify_audits()
 np.savez_compressed(arraypath,locations=np.array(locations,int),selected_plans=np.array(plans),selected_gradients=np.array(gradients),independent_replayed_gradients=np.array(replay_gradients),selected_objectives=np.array([r['selected_objective'] for r in records]),absolute_bounds=np.array([r['absolute_bound'] for r in records]),relative_to_selected_objective=np.array([np.nan if r['relative_to_selected_objective'] is None else r['relative_to_selected_objective'] for r in records]))
 out=dict(status='PASS — all 11520 posthoc fixed local convex objective bounds and selected-gradient provenance replayed; modeled loss only',source_sha256=sha(Path(__file__)),design_sha256=sha(DESIGN),input_and_final_audit_sha256=hashes,arrays_sha256=sha(arraypath),scope=dict(replicates=list(REPS),ordinary_decisions=11520,retained_decisions=5760,frozen_decisions=5760,corpus_variant_phase_groups=48,decisions_per_group=240,physical_interactions_added=0,model_fits_added=0,optimization_solves_added=0),exact_arithmetic_formula='f(x)-min_box f <= sum_i abs(g_i)*(0.5+sign(g_i)*x_i) for convex differentiable f, x in box and g=grad f(x)',fixtures=tests,maximum_numerical_replay_errors=replay_errors,global_summary=summary(records),per_corpus=percorpus,per_variant={v:summary([r for r in records if r['variant']==v]) for v in('retained','frozen')},per_phase={str(p):summary([r for r in records if r['phase']==p]) for p in range(3)},per_corpus_variant_phase=[dict(replicate=rep,variant=v,phase=p,summary=summary(allgroups[(rep,v,p)])) for rep in REPS for v in('retained','frozen') for p in range(3)],all_decisions=records,reconstruction_failures=reconstruction_failures,wall_seconds=time.time()-start,limitations=['Exact convex bound is an exact-arithmetic theorem; floating-point gradients are independently replayed but are not interval-certified real gradients.','Replayed gradient discrepancy allowance compares two floating-point evaluations, not a proven bound on either exact-real error.','Both variant rollouts are affine only because actual fit coefficients and query stay fixed throughout hypothetical planning.','No modeled gap bound implies physical accuracy, closed-loop optimality, task success or complete knowledge.','Relative bound divides by selected objective, not unknown optimum; zero objective ratio is undefined.','All decisions included posthoc, with no new fits, solver calls, physics or model selection.','Prior independent analytic paths and cost-gradient helpers reused, not a fresh independent derivation claim.'])
 output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','scope','maximum_numerical_replay_errors','global_summary','wall_seconds')},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
