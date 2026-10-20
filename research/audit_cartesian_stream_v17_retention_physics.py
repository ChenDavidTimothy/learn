#!/usr/bin/env python3
"""Independent frozen-library actual physics/RNG/score/paired artifact audit.
Scope fixed from protocol before outcome inspection. Ordinary local fits/plans/
constant library content are independently audited by another checker.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_stream_v17_control_physics as prior
mechanics=prior.mechanics
ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-9):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def rng(path):return np.random.default_rng(np.random.SeedSequence(path))

def main():
 start=time.time();rp=ROOT/'cartesian_stream_v17_rep0_retention_results.json';ap=ROOT/'cartesian_stream_v17_rep0_retention_arrays.npz';pp=ROOT/'cartesian_stream_v17_rep0_control_results.json';pa=ROOT/'cartesian_stream_v17_rep0_control_arrays.npz';src=ROOT/'cartesian_stream_v17_retention.py';design=ROOT/'cartesian_stream_v17_retention_design.md'
 r=json.loads(rp.read_text());parent=json.loads(pp.read_text());assert r['status'].startswith('COMPLETE') and len(r['episodes'])==8
 assert sha(src)==r['source_sha256'] and sha(design)==r['design_sha256'] and sha(ap)==r['arrays_sha256'];assert sha(pp)==r['parent_control_results_sha256'] and sha(pa)==parent['arrays_sha256'];assert sha(ROOT/'cartesian_stream_v17_control.py')==r['parent_control_source_sha256']==parent['source_sha256']
 for name,h in parent['dependency_sha256'].items():assert sha(ROOT/name)==h
 fitpath=ROOT/'cartesian_stream_v17_rep0_fit.json';fit=json.loads(fitpath.read_text());assert sha(fitpath)==parent['fit_sha256'] and sha(ROOT/'cartesian_stream_v17_rep0_weights.npz')==parent['weights_sha256'] and sha(ROOT/'cartesian_stream_v17_rep0_ordinary.npz')==parent['ordinary_sha256']
 with np.load(ap) as d:arrays={k:d[k] for k in d.files}
 with np.load(pa) as d:oldarrays={k:d[k] for k in d.files}
 physical=[];commands=[];pairs=[];metadata_rows=[len(b) for b in fit['bootstrap_rows']];selected=fit['local_candidates'][fit['selected']['local']]
 for case_index,e in enumerate(r['episodes']):
  assert e['case_index']==case_index and e['kind']=='local_frozen_library'
  context=case_index//2;repeat=case_index%2;ip=[20261020,17,670,0,context,0];up=[20261020,17,670,0,context,2];npth=[20261020,17,670,0,context,1,repeat]
  init=rng(ip);theta=init.uniform(-np.pi,np.pi);phi=init.uniform(.4,np.pi-.4);sign=init.choice([-1.,1.]);speed=init.uniform(.02,.12);angle=init.uniform(-np.pi,np.pi);q=np.array([theta-sign*phi/2,sign*phi]);velocity=speed*np.array([np.cos(angle),np.sin(angle)]);initial=np.r_[q,np.linalg.solve(mechanics.jac(q),velocity)]
  warm=np.zeros((16,2));warm[8:]=np.repeat(rng(up).uniform(-.5,.5,(4,2)),2,axis=0);noise=rng(npth).normal(0,.0002,(197,2));assert e['initial_seed_path']==ip and e['warmup_seed_path']==up and e['noise_seed_path']==npth and e['context_id']==context and e['noise_repeat']==repeat
  k=e['array_keys'];z=arrays[k['states_evaluator_only']];u=arrays[k['commands']];o=arrays[k['observed']];calls=arrays[k['plant_nfev']]
  assert z.shape==(197,4) and u.shape==(196,2) and o.shape==(197,2) and calls.shape==(196,) and abs(u).max()<=.5 and np.isfinite(z).all() and np.isfinite(o).all()
  ck('fresh_initial_rng',z[0],initial,1e-15);ck('warmup_command_rng',u[:16],warm,1e-15);ck('all_camera_noise_rng',mechanics.endpoint(z[:,:2])+noise,o,1e-15);ck('all_held_pairs',u[16::2],u[17::2],1e-15)
  ck('parent_case_initial',initial,oldarrays[f'case{case_index}_initial_state'],1e-15);ck('parent_case_warmup',warm,oldarrays[f'case{case_index}_warmup_commands'],1e-15);ck('parent_case_noise',noise,oldarrays[f'case{case_index}_sensor_noise'],1e-15)
  online=next(x for x in parent['ordinary_episodes'] if x['kind']=='local' and x['case_index']==case_index);ok=online['array_keys'];oz=oldarrays[ok['states_evaluator_only']];ou=oldarrays[ok['commands']];oo=oldarrays[ok['observed']]
  ck('paired_initial_warmup_states',z[:17],oz[:17],1e-15);ck('paired_initial_warmup_observations',o[:17],oo[:17],1e-15);ck('paired_initial_warmup_commands',u[:16],ou[:16],1e-15)
  # Before the first append notification the two policy inputs/libraries are equal.
  ck('paired_FIRST_command_before_any_append',u[16:18],ou[16:18],1e-12);ck('paired_FIRST_actual_pair_states',z[:19],oz[:19],1e-12)
  goals=prior.phase_goals(o[16]);ck('same_sensor_derived_goals',e['goals'],goals,2e-15);ck('paired_original_goals',goals,online['goals'],2e-15)
  score=prior.reconstruct_score(z,u,goals);prior.check_score(score,e['score']);online_score=prior.reconstruct_score(oz,ou,goals);prior.check_score(online_score,online['score'])
  ck('stored_paired_online_taskcost',e['paired_online_task_cost'],online_score['task_cost'],1e-8);difference=score['task_cost']-online_score['task_cost'];ck('stored_frozen_minus_online_taskcost',difference,e['frozen_minus_online_task_cost'],1e-8)
  assert e['local_library_initial_rows']==e['local_library_final_rows']==metadata_rows and e['real_pair_notifications']==90 and e['actual_local_rows_appended']==0
  assert e['selected_configuration']=={'neighbors':selected['neighbors'],'ridge':selected['ridge']} and len(e['decisions'])==90 and e['plant_rhs_evaluations']==int(calls.sum())
  assert arrays[k['local_fit_coefficients']].shape==(90,3,69,4) and arrays[k['local_fit_queries']].shape==(90,66)
  for j,d in enumerate(e['decisions']):
   t=16+2*j;phase=j//30;assert d['renewal']==j and d['phase']==phase and d['time_index']==t;ck('decision_command_alignment',d['command'],u[t],1e-15);ck('decision_only_current_goal',d['current_goal'],goals[phase],2e-15)
   assert [q['n_rows'] for q in d['local_fit_diagnostics']]==metadata_rows
  physical.append(z);commands.append(u);pairs.append(dict(case_index=case_index,context_id=context,noise_repeat=repeat,frozen_task_cost=score['task_cost'],online_task_cost=online_score['task_cost'],frozen_minus_online_task_cost=difference,frozen_combined_cost=score['combined_charged_cost'],online_combined_cost=online_score['combined_charged_cost'],frozen_precise_goals=score['precise_goals'],online_precise_goals=online_score['precise_goals'],frozen_precision_by_phase=[p['precision_success'] for p in score['phases']],online_precision_by_phase=[p['precision_success'] for p in online_score['phases']],frozen_phase_costs=[p['cost'] for p in score['phases']],online_phase_costs=[p['cost'] for p in online_score['phases']],frozen_integrated_effort=[p['integrated_squared_effort'] for p in score['phases']],online_integrated_effort=[p['integrated_squared_effort'] for p in online_score['phases']]))
 physical=np.array(physical);command=np.array(commands);local=mechanics.advance(physical[:,:-1].reshape(-1,4),command.reshape(-1,2),steps=256);ck('ALL1568local_Hamiltonian_intervals',local,physical[:,1:].reshape(-1,4),1e-8)
 current=physical[:,0].copy();state_max=position_max=0.
 for t in range(196):
  current=mechanics.advance(current,command[:,t],steps=256);state_max=max(state_max,float(abs(current-physical[:,t+1]).max()));position_max=max(position_max,float(abs(mechanics.endpoint(current[:,:2])-mechanics.endpoint(physical[:,t+1,:2])).max()))
 assert state_max<1e-6 and position_max<1e-6;ERROR['all8_complete_Hamiltonian_states']=state_max;ERROR['all8_complete_Hamiltonian_positions']=position_max
 velocity=(mechanics.jac(physical[:,:,:2])@physical[:,:,2:,None])[...,0];speed=np.linalg.norm(velocity,axis=2);joint=np.linalg.norm(physical[:,:,2:],axis=2);jac=abs(np.sin(physical[:,:,1]));qstep=abs(np.diff(physical[:,:,:2],axis=1)).max(axis=2);spots={(i,t) for i in range(8) for t in(0,16,76,136,194)};extremes={}
 for name,x,minimum in[('Cartesian_speed',speed,False),('joint_speed',joint,False),('Jacobian_singularity',jac,True),('joint_step',qstep,False)]:
  flat=int(np.argmin(x) if minimum else np.argmax(x));i,t=np.unravel_index(flat,x.shape);t=min(int(t),195);spots.add((int(i),t));extremes[name]=[int(i),t]
 radau=[]
 for i,t in sorted(spots):
  final=mechanics.radau(physical[i,t],command[i,t]);error=float(abs(final-physical[i,t+1]).max());assert error<1e-8;radau.append(dict(episode=i,interval=t,error=error))
 resources=dict(episodes=8,plant_intervals=1568,physical_seconds=156.8,decisions=720,camera_read_opportunities=1576,scalar_noise_opportunities=3152,new_independent_noise_paths_relative_to_parent=0,actual_local_rows_appended=0,plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in r['episodes']),modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in r['episodes'] for d in e['decisions']));assert resources==r['resources']
 summary=dict(frozen_precise_goals=sum(p['frozen_precise_goals'] for p in pairs),online_precise_goals=sum(p['online_precise_goals'] for p in pairs),frozen_phase_precision=[sum(p['frozen_precision_by_phase'][j] for p in pairs) for j in range(3)],online_phase_precision=[sum(p['online_precision_by_phase'][j] for p in pairs) for j in range(3)],mean_frozen_minus_online_taskcost=math.fsum(p['frozen_minus_online_task_cost'] for p in pairs)/8,context_mean_frozen_minus_online_taskcost=[math.fsum(p['frozen_minus_online_task_cost'] for p in pairs if p['context_id']==i)/2 for i in range(4)])
 artifacts=[rp,ap,pp,pa,src,design,fitpath,ROOT/'cartesian_stream_v17_rep0_weights.npz',ROOT/'cartesian_stream_v17_rep0_ordinary.npz',Path(prior.__file__),Path(mechanics.__file__),*[ROOT/name for name in parent['dependency_sha256']]]
 out=dict(status='PASS — independent frozenlibrary physics/RNG/sustainedscore/pairedoutcome audit; ordinary fits/plans separately audited',artifact_sha256={p.name:sha(p) for p in artifacts},audit_source_sha256=sha(Path(__file__)),maximum_reconstruction_errors={**ERROR,**{'score_'+k:v for k,v in prior.ERROR.items()}},scope=dict(actual_physical_intervals=1568,complete_Hamiltonian_trajectories=8,RK4_substeps_per_interval=256,Radau_intervals=len(radau),paired_cases=8,warmup_and_first_actualpair_equality_checked=True),resources=resources,paired_cases=pairs,summary=summary,independent_Radau_checks=radau,maximum_Radau_error=max(q['error'] for q in radau),extreme_interval_indices=extremes,late_precision_endpoint_indices=[[67,76],[127,136],[187,196]],limitations=['Library-content/ordinaryfit/gradient/optimizer audit is separate; metadata rowcount checks alone are not that proof.','Same8knownpairedcases after originalpilot; exploratorycausalmechanism ablation, not freshpopulationconfirmation.','Freezinglibrary preservescurrenthistory/refitting/feedback, not no learning or no adaptation.','Afterfirstappend, policies mayvisitdifferenthistories; paired differences are totalclosedloopeffect, not identicalpostwarmexperience.','Zero new independentnoise paths relativetoparent; RNGoperations replayknownarrays, not zero physicalcameraopportunities.','Four preparations/two pairednoiserepeats each, not8independentbodies; no null/necessityclaimfrompointcounts alone.','Sampledtenendpointprecision, not continuous-time guarantee.'],wall_seconds=time.time()-start)
 (ROOT/'audit_cartesian_stream_v17_retention_physics_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','maximum_reconstruction_errors','scope','resources','summary','maximum_Radau_error','wall_seconds')}))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
