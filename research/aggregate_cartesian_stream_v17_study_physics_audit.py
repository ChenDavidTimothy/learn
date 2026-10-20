#!/usr/bin/env python3
"""Hash-bind the eight immutable independent physical audit caches and arithmetic.
No new fit, physics rollout or model selection; outcomes are descriptive only.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
REPS=tuple(range(200,208))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
 x=np.array(x,float);assert x.shape==(8,) and np.isfinite(x).all()
 return dict(corpus_values=x.tolist(),mean=float(x.mean()),descriptive_cluster_SE=float(x.std(ddof=1)/np.sqrt(8)))
def main():
 source=ROOT/'audit_cartesian_stream_v17_study_physics.py';audits=[];hashes={source.name:sha(source)};artifacthashes={};failures=[];percorpus=[]
 for rep in REPS:
  p=ROOT/f'audit_cartesian_stream_v17_rep{rep}_study_physics_results.json';d=json.loads(p.read_text());assert d['status'].startswith('PASS') and d['replicate']==rep and d['audit_source_sha256']==sha(source);hashes[p.name]=sha(p)
  for name,h in d['artifact_sha256'].items():
   assert sha(ROOT/name)==h,(rep,name)
   if name in artifacthashes:assert artifacthashes[name]==h
   artifacthashes[name]=h
  ac=d['acquisition']['physics']['scope'];ct=d['control']['physics']['scope'];assert ac['local_intervals']==10240 and ac['complete_trajectories']==40 and ct['local_intervals']==4704 and ct['complete_trajectories']==24 and ac['local_RK4_substeps']==ct['local_RK4_substeps']==256 and ac['rolling_RK4_substeps']==ct['rolling_RK4_substeps']==512
  c=d['control'];pairs=c['paired_cases'];assert len(pairs)==8
  assert abs(math.fsum(p['frozen_minus_retained_task_cost'] for p in pairs)/8-c['mean_frozen_minus_retained_taskcost'])<1e-10
  for kind in('retained','frozen'):
   assert abs(math.fsum(p[kind+'_task_cost'] for p in pairs)/8-c['summaries'][kind]['mean_task_cost'])<1e-10
   assert sum(p[kind+'_precise_goals'] for p in pairs)==c['summaries'][kind]['total_precise_goals']
  raw=json.loads((ROOT/f'cartesian_stream_v17_rep{rep}_study_results.json').read_text())
  for cat in('witnesses','ordinary_episodes'):
   for e in raw[cat]:
    kind='reference' if cat=='witnesses' else e['variant']
    for phase in e['score']['phases']:
     if not phase['precision_success']:
      failures.append(dict(replicate=rep,kind=kind,case_index=e['case_index'],context_id=e['context_id'],noise_repeat=e['noise_repeat'],phase=phase['phase'],phase_cost=phase['cost'],late_max_coordinate_position_error=phase['late_max_coordinate_position_error'],late_max_coordinate_velocity=phase['late_max_coordinate_velocity'],raw_result=f'cartesian_stream_v17_rep{rep}_study_results.json',raw_array_keys=e['array_keys']))
  percorpus.append(dict(replicate=rep,summary=c['summaries'],paired_mean_frozen_minus_retained_taskcost=c['mean_frozen_minus_retained_taskcost'],reference_saturated_decisions=c['reference_saturated_decisions'],reference_clipped_goals=c['reference_clipped_goals']))
  audits.append(d)
 def sfield(kind,key):return stats([d['control']['summaries'][kind][key] for d in audits])
 summaries={}
 for kind in('reference','retained','frozen'):
  total=sum(d['control']['summaries'][kind]['total_precise_goals'] for d in audits)
  assert sum(f['kind']==kind for f in failures)==192-total
  summaries[kind]=dict(task_cost=sfield(kind,'mean_task_cost'),combined_charged_cost=sfield(kind,'mean_combined_charged_cost'),precision_rate=stats([d['control']['summaries'][kind]['total_precise_goals']/24 for d in audits]),total_precise_goal_windows=total,total_goal_windows=192,phase_precision_counts=np.sum([d['control']['summaries'][kind]['phase_precision_counts'] for d in audits],axis=0).tolist(),phase_mean_cost=[stats([d['control']['summaries'][kind]['phase_mean_costs'][p] for d in audits]) for p in range(3)])
 paired=stats([d['control']['mean_frozen_minus_retained_taskcost'] for d in audits]);pairedprecision=stats([(d['control']['summaries']['retained']['total_precise_goals']-d['control']['summaries']['frozen']['total_precise_goals'])/24 for d in audits]);assert abs(paired['mean']-(summaries['frozen']['task_cost']['mean']-summaries['retained']['task_cost']['mean']))<1e-10
 rolepaths=set()
 for rep in REPS:
  for split,count in enumerate((24,8,8)):
   for i in range(count):
    for role in range(3):rolepaths.add((20261020,17,650,rep,split,i,role))
  for member in range(3):rolepaths.add((20261020,17,660,rep,0,member))
  for context in range(4):
   for role in(0,2):rolepaths.add((20261020,17,670,rep,context,role))
   for repeat in range(2):rolepaths.add((20261020,17,670,rep,context,1,repeat))
 assert len(rolepaths)==1112
 errors={}
 for d in audits:
  for name,x in d['maximum_reconstruction_errors'].items():errors[name]=max(errors.get(name,0.),x)
 scope=dict(corpora=8,acquisition_trajectories=320,acquisition_intervals=81920,control_reference_trajectories=192,control_reference_intervals=37632,all_local_intervals=119552,all_complete_trajectories=512,local_RK4_substeps=256,rolling_RK4_substeps=512,acquisition_Radau_intervals=sum(d['acquisition']['physics']['scope']['Radau_intervals'] for d in audits),control_Radau_intervals=sum(d['control']['physics']['scope']['Radau_intervals'] for d in audits),distinct_RNG_role_paths=1112,reference_commands=5760,ordinary_command_timing_and_scores=11520)
 resources=dict(acquisition_simulated_seconds=8192.,acquisition_independent_scalar_draws=164480,ordinary_episodes=128,ordinary_intervals=25088,ordinary_seconds=2508.8,ordinary_sustained_goal_windows=384,ordinary_decisions=11520,ordinary_camera_read_opportunities=25216,ordinary_scalar_read_opportunities=50432,distinct_control_preparations=32,independent_paired_control_noise_paths=64,distinct_control_scalar_draws=25216,retained_actual_row_events=5760,retained_member_append_copies=17280,frozen_append_copies=0,reference_episodes=64,reference_intervals=12544,reference_seconds=1254.4,reference_training_experiences=0,acquisition_plant_rhs_evaluations=sum(d['acquisition']['resources']['plant_rhs_evaluations'] for d in audits),ordinary_plant_rhs_evaluations=sum(d['control']['resources']['ordinary_plant_rhs_evaluations'] for d in audits),reference_plant_rhs_evaluations=sum(d['control']['resources']['reference_plant_rhs_evaluations'] for d in audits),modeled_value_gradient_calls=sum(d['control']['resources']['modeled_value_gradient_calls'] for d in audits),unsuccessful_optimizer_terminations=sum(d['control']['resources']['unsuccessful_optimizer_terminations'] for d in audits),fallback_decisions=sum(d['control']['resources']['fallback_decisions'] for d in audits))
 out=dict(status='PASS — final eight-corpus hash-bound independent physics/RNG/witness/sustained-score audit; ordinary fitting/plans separate',replicates=list(REPS),aggregator_source_sha256=sha(Path(__file__)),audit_cache_and_source_sha256=hashes,final_artifact_sha256=artifacthashes,scope=scope,resources=resources,maximum_reconstruction_errors=errors,maximum_Radau_error=max(d[k]['physics']['maximum_Radau_difference'] for d in audits for k in('acquisition','control')),per_corpus=percorpus,summaries=summaries,paired_frozen_minus_retained_taskcost=paired,paired_retained_minus_frozen_precision_rate=pairedprecision,all_failed_goal_windows=failures,reference_saturated_decisions=sum(d['control']['reference_saturated_decisions'] for d in audits),reference_clipped_goals=sum(d['control']['reference_clipped_goals'] for d in audits),limitations=['Descriptive SE uses eight corpus means, not overlapping frames, goals or paired noise repeats as independent bodies.','All corpora come from the same fixed body and fixed acquisition/preparation/task laws.','Fitting, selection, ordinary regression/append content and optimizer plans belong to a separate independent artifact audit.','Previously independent Hamiltonian/IK/Christoffel/score helpers reused, not new derivations.','Privileged witness is nominal authority diagnostic, not an equally informed ordinary method or global recovery certificate.','Precision at ten saved endpoints does not certify continuous-time behavior.','Append comparison is a total closed-loop policy effect; shared feedback is not isolated and later histories differ.','Failures and all raw artifacts retained; outcome statistics do not select models, seeds or task changes.','Historical counters reconstructed; no independent walltime or real-time claim.'])
 p=ROOT/'audit_cartesian_stream_v17_study_physics_results.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','scope','resources','maximum_Radau_error','summaries','paired_frozen_minus_retained_taskcost','paired_retained_minus_frozen_precision_rate','reference_saturated_decisions','reference_clipped_goals')},indent=2))
if __name__=='__main__':main()
