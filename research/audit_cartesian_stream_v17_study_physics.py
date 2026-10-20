#!/usr/bin/env python3
"""Pre-outcome fixed-scope fresh-study acquisition/physics/RNG/witness audit.
Reuses previously independent Hamiltonian and score/IK helpers, not canonical
plant/controller/learner modules. Fits/ordinary optimization have another audit.
Only run on notified COMPLETE artifacts; no polling or partial outcome inspection.
"""
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_stream_v17_control_physics as prior
mechanics=prior.mechanics
ROOT=Path(__file__).resolve().parent
EXPECTED_SOURCE='a213d5eb'  # full source and design digests checked against artifacts
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-9):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
 ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def rng(path):return np.random.default_rng(np.random.SeedSequence(path))
def readnp(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def quant(x):return {str(q):float(np.quantile(x,q)) for q in (0,.01,.1,.5,.9,.99,1)}
def physics(states,actions,tag,acquisition):
 n,length=actions.shape[:2];positions=mechanics.endpoint(states[:,:,:2])
 local=mechanics.advance(states[:,:-1].reshape(-1,4),actions.reshape(-1,2),steps=256)
 ck(tag+'_ALLlocal_Hamiltonian',local,states[:,1:].reshape(-1,4),1e-8)
 current=states[:,0].copy();emax=pmax=0.
 for t in range(length):
  current=mechanics.advance(current,actions[:,t],steps=512)
  emax=max(emax,float(abs(current-states[:,t+1]).max()))
  pmax=max(pmax,float(abs(mechanics.endpoint(current[:,:2])-positions[:,t+1]).max()))
 assert emax<1e-6 and pmax<1e-6,(tag,emax,pmax)
 ERROR[tag+'_ALLcomplete_Hamiltonian_states']=emax;ERROR[tag+'_ALLcomplete_Hamiltonian_positions']=pmax
 M=mechanics.mass(states[:,:,:2]);energy=.5*np.einsum('nti,ntij,ntj->nt',states[:,:,2:],M,states[:,:,2:]);residual=np.diff(energy,axis=1)-np.sum(actions*np.diff(states[:,:,:2],axis=1),axis=2)
 velocity=(mechanics.jac(states[:,:,:2])@states[:,:,2:,None])[...,0];speed=np.linalg.norm(velocity,axis=2);joint=np.linalg.norm(states[:,:,2:],axis=2);det=abs(np.sin(states[:,:,1]));qstep=abs(np.diff(states[:,:,:2],axis=1));phase=abs(np.diff(states[:,:,:2].sum(axis=2),axis=1));pstep=np.linalg.norm(np.diff(positions,axis=1),axis=2)
 coverage=dict(maximum_energy_work_residual=float(abs(residual).max()),minimum_mass_eigenvalue=float(np.linalg.eigvalsh(M).min()),joint_coordinate_step_abs=quant(qstep.ravel()),distal_phase_step_abs=quant(phase.ravel()),camera_true_step_norm=quant(pstep),cartesian_speed=quant(speed),joint_speed_norm=quant(joint),absolute_jacobian_determinant=quant(det),sampled_near_singular_abs_det_below_point01=int(np.sum(det<.01)))
 extremes={}
 for name,a,minimum in [('Cartesian_speed',speed,False),('joint_speed',joint,False),('Jacobian_singularity',det,True),('joint_step',qstep.max(axis=2),False),('distal_phase',phase,False)]:
  flat=int(np.argmin(a) if minimum else np.argmax(a));i,t=np.unravel_index(flat,a.shape);extremes[name]=[int(i),min(int(t),length-1)]
 spots=({(i,0) for i in range(n)}|{(i,127) for i in range(0,n,4)}) if acquisition else {(i,t) for i in range(n) for t in(0,16,76,136,194)}
 spots|=set(map(tuple,extremes.values()));radau=[]
 for i,t in sorted(spots):
  z=mechanics.radau(states[i,t],actions[i,t]);err=float(abs(z-states[i,t+1]).max());assert err<1e-8,(tag,i,t,err)
  radau.append(dict(trajectory=i,interval=t,maximum_state_difference=err))
 signs=np.sin(states[:,:,1]);cross=signs[:,:-1]*signs[:,1:]<0
 return dict(coverage=coverage,additional_support=dict(sample_bracketed_jacobian_sign_changes=int(cross.sum()),trajectories_with_bracketed_sign_change=int(cross.any(1).sum())),Radau_checks=radau,maximum_Radau_difference=max(x['maximum_state_difference'] for x in radau),extreme_interval_indices=extremes,scope=dict(local_intervals=n*length,complete_trajectories=n,local_RK4_substeps=256,rolling_RK4_substeps=512,Radau_intervals=len(radau)))
def acquire(prefix,rep,seen):
 rp=Path(str(prefix)+'_acquisition.json');op=Path(str(prefix)+'_ordinary.npz');ep=Path(str(prefix)+'_evaluator.npz');meta=json.loads(rp.read_text())
 assert meta['replicate']==rep and meta['counts']==[24,8,8] and meta['steps']==256 and meta['dt']==.1
 for p,key in [(ROOT/'cartesian_stream_v17_acquire.py','source_sha256'),(ROOT/'cartesian_stream_v17_acquisition_design.md','design_sha256'),(ROOT/'coupled_context_v14_pilot.py','plant_source_sha256')]:assert sha(p)==meta[key]
 assert {p.name:sha(p) for p in(op,ep)}==meta['artifacts']
 o=readnp(op);e=readnp(ep);assert set(o)=={'observed','commands','split','trajectory_index','dt','sensor_sd'} and set(e)=={'states','sensor_noise','plant_nfev','seed_paths'}
 z=e['states'];u=o['commands'];obs=o['observed'];noise=e['sensor_noise'];calls=e['plant_nfev'];paths=e['seed_paths'];assert z.shape==(40,257,4) and u.shape==(40,256,2) and obs.shape==noise.shape==(40,257,2) and calls.shape==(40,256) and paths.shape==(40,3,7)
 assert np.isfinite(z).all() and np.isfinite(obs).all() and np.isfinite(u).all() and abs(u).max()<=.5 and np.all(calls>0) and np.issubdtype(calls.dtype,np.integer) and float(o['dt'])==.1 and float(o['sensor_sd'])==.0002
 preparations=[];k=0
 for split,count in enumerate((24,8,8)):
  for index in range(count):
   path=[[20261020,17,650,rep,split,index,role] for role in range(3)]
   for p in path:assert tuple(p) not in seen;seen.add(tuple(p))
   init,act,sens=map(rng,path);theta=init.uniform(-np.pi,np.pi);phi=init.uniform(.4,np.pi-.4);sign=init.choice([-1.,1.]);speed=init.uniform(.02,.12);angle=init.uniform(-np.pi,np.pi);q=np.array([theta-sign*phi/2,sign*phi]);V=speed*np.array([np.cos(angle),np.sin(angle)]);initial=np.r_[q,np.linalg.solve(mechanics.jac(q),V)]
   ck('acq_initial_RNG',initial,z[k,0],2e-15);ck('acq_initial_CartesianV',mechanics.jac(q)@z[k,0,2:],V,2e-15)
   command=np.zeros((256,2));command[8:]=np.repeat(act.uniform(-.5,.5,(124,2)),2,axis=0)
   ck('acq_ALLcommand_RNG',command,u[k],1e-15);ck('acq_ALLnoise_RNG',sens.normal(0,.0002,(257,2)),noise[k],1e-15);ck('acq_ALLseedpaths',path,paths[k],1e-15)
   assert o['split'][k]==split and o['trajectory_index'][k]==index
   preparations.append(dict(split=split,index=index,theta=theta,phi=phi,sign=float(sign),speed=speed,direction=angle));k+=1
 ck('acq_ALLcamera',mechanics.endpoint(z[:,:,:2])+noise,obs,1e-15);ck('acq_heldpairs',u[:,8::2],u[:,9::2],1e-15);assert not np.any(u[:,:8])
 phys=physics(z,u,'acquisition',True)
 for key,val in phys['coverage'].items():
  if isinstance(val,dict):
   for q,v in val.items():ck('acq_coverage_'+key+'_'+q,v,meta['coverage_evaluator_only'][key][q],1e-10)
  else:ck('acq_coverage_'+key,val,meta['coverage_evaluator_only'][key],1e-10)
 resources=dict(trajectories=40,physical_intervals=10240,physical_seconds=1024.,passive_prefix_intervals=320,passive_prefix_seconds=32.,intervention_intervals=9920,intervention_seconds=992.,scalar_noise_draws=20560,cartesian_camera_reads=10280,plant_rhs_evaluations=int(calls.sum()));assert resources==meta['resources']
 return dict(resources=resources,initial_preparations_evaluator_only=preparations,physics=phys),[rp,op,ep]
def control(prefix,rep,seen):
 rp=Path(str(prefix)+'_study_results.json');ap=Path(str(prefix)+'_study_arrays.npz');fp=Path(str(prefix)+'_local_fit.json');op=Path(str(prefix)+'_ordinary.npz');ep=Path(str(prefix)+'_evaluator.npz');acp=Path(str(prefix)+'_acquisition.json');tp=Path(str(prefix)+'_local_test_forecasts.npz');src=ROOT/'cartesian_stream_v17_study.py';result=json.loads(rp.read_text());fit=json.loads(fp.read_text());a=readnp(ap)
 assert result['status'].startswith('COMPLETE FRESH LOCAL RETENTION CORPUS') and result['replicate']==rep and fit['replicate']==rep and len(result['witnesses'])==8 and len(result['ordinary_episodes'])==16
 assert sha(src)==result['source_sha256']==fit['source_sha256'] and sha(src).startswith(EXPECTED_SOURCE)
 for name,h in result['dependency_sha256'].items():assert sha(ROOT/name)==h
 assert fit['dependency_sha256']==result['dependency_sha256']
 for p,key in[(ap,'arrays_sha256'),(fp,'fit_sha256'),(op,'ordinary_sha256'),(ep,'evaluator_sha256'),(acp,'acquisition_sha256'),(tp,'test_forecasts_sha256')]:assert sha(p)==result[key]
 assert fit['ordinary_sha256']==result['ordinary_sha256'] and fit['acquisition_sha256']==result['acquisition_sha256'] and fit['test_forecasts_sha256']==result['test_forecasts_sha256']
 assert result['acquisition_resources']==json.loads(acp.read_text())['resources']
 for member,p in enumerate(fit['bootstrap_seed_paths']):
  assert p==[20261020,17,660,rep,0,member] and tuple(p) not in seen;seen.add(tuple(p))
 cases=[]
 for context in range(4):
  ip=[20261020,17,670,rep,context,0];up=[20261020,17,670,rep,context,2]
  for p in(ip,up):assert tuple(p) not in seen;seen.add(tuple(p))
  init=rng(ip);theta=init.uniform(-np.pi,np.pi);phi=init.uniform(.4,np.pi-.4);sign=init.choice([-1.,1.]);speed=init.uniform(.02,.12);angle=init.uniform(-np.pi,np.pi);q=np.array([theta-sign*phi/2,sign*phi]);V=speed*np.array([np.cos(angle),np.sin(angle)]);initial=np.r_[q,np.linalg.solve(mechanics.jac(q),V)];warm=np.zeros((16,2));warm[8:]=np.repeat(rng(up).uniform(-.5,.5,(4,2)),2,axis=0)
  for repeat in range(2):
   npth=[20261020,17,670,rep,context,1,repeat];assert tuple(npth) not in seen;seen.add(tuple(npth));noise=rng(npth).normal(0,.0002,(197,2));case_index=len(cases)
   ck('control_initial_RNG',initial,a[f'case{case_index}_initial_state'],1e-15);ck('control_warmup_RNG',warm,a[f'case{case_index}_warmup_commands'],1e-15);ck('control_noise_RNG',noise,a[f'case{case_index}_sensor_noise'],1e-15)
   cases.append(dict(context_id=context,noise_repeat=repeat,initial_seed_path=ip,warmup_seed_path=up,noise_seed_path=npth,initial=initial,warm=warm,noise=noise))
 entries=[];summaries={};saturated=clipped=0;selected=int(fit['selected_local_candidate']);config=fit['candidates'][selected];rowcounts=list(map(len,fit['bootstrap_rows']))
 for category in('witnesses','ordinary_episodes'):
  for index,e in enumerate(result[category]):
   case_index=int(e['case_index']);assert 0<=case_index<8;case=cases[case_index]
   for key in('context_id','noise_repeat','initial_seed_path','warmup_seed_path','noise_seed_path'):assert e[key]==case[key]
   k=e['array_keys'];z=a[k['states_evaluator_only']];u=a[k['commands']];o=a[k['observed']];calls=a[k['plant_nfev']]
   assert z.shape==(197,4) and u.shape==(196,2) and o.shape==(197,2) and calls.shape==(196,) and np.isfinite(z).all() and np.isfinite(o).all() and np.isfinite(u).all() and abs(u).max()<=.5 and np.all(calls>0)
   ck('control_actual_initial',case['initial'],z[0],1e-15);ck('control_actual_warm',case['warm'],u[:16],1e-15);ck('control_ALLcamera',mechanics.endpoint(z[:,:2])+case['noise'],o,1e-15);ck('control_ALLheldpairs',u[16::2],u[17::2],1e-15)
   goals=prior.phase_goals(o[16]);ck('control_sensor_goals',goals,e['goals'],2e-15);assert len(e['decisions'])==90 and e['plant_rhs_evaluations']==int(calls.sum())
   for j,d in enumerate(e['decisions']):
    t=16+2*j;phase=j//30;assert d['renewal']==j and d['time_index']==t and d['phase']==phase;ck('control_decision_command',u[t],d['command'],1e-15)
    if category=='ordinary_episodes':ck('ordinary_only_current_goal',d['current_goal'],goals[phase],2e-15)
    else:
     q,v=z[t,:2],z[t,2:];(distance,target,branch,winding),point,radius,r=prior.independent_target(q,goals[phase]);ck('reference_IKtarget',target,d['joint_target'],1e-8);ck('reference_target_endpoint',mechanics.endpoint(np.asarray(d['joint_target'])),point,1e-9);ck('reference_nearest_distance',distance,d['joint_distance_squared'],1e-8)
     for key,x in [('original_goal',goals[phase]),('clipped_goal',point),('original_radius',radius),('clipped_radius',r),('winding',winding)]:ck('reference_'+key,x,d[key],1e-9)
     assert d['branch']==float(branch) and d['clipped']==(radius!=r) and d['folded_goal']==(r==0)
     torque=mechanics.mass(q)@(-q+np.asarray(d['joint_target'])-2*v)+prior.coriolis_from_metric(q,v);ck('reference_computedtorque',torque,d['unbounded_command'],1e-8);ck('reference_boundedtorque',np.clip(torque,-.5,.5),u[t],1e-8)
     sat=bool(np.any(np.clip(torque,-.5,.5)!=torque));assert sat==d['saturated'];saturated+=sat;clipped+=bool(d['clipped'])
   score=prior.reconstruct_score(z,u,goals);prior.check_score(score,e['score']);kind=e['variant'] if category=='ordinary_episodes' else 'reference';summaries.setdefault(kind,[]).append(score)
   if category=='ordinary_episodes':
    assert e['variant'] in('retained','frozen') and e['selected_candidate_index']==selected and e['selected_configuration']==dict(neighbors=config['neighbors'],ridge=config['ridge']) and e['local_library_initial_rows']==rowcounts and e['real_pair_notifications']==90
    assert e['local_library_final_rows']==[n+(90 if kind=='retained' else 0) for n in rowcounts] and e['actual_local_row_events']==(90 if kind=='retained' else 0)
    assert a[k['local_fit_coefficients']].shape==(90,3,69,4) and a[k['local_fit_queries']].shape==(90,66)
   entries.append(dict(category=category,case_index=case_index,kind=kind,states=z,commands=u,observed=o,score=score,record=e))
 pairrows=[]
 for case in range(8):
  group=[x for x in entries if x['case_index']==case];assert len(group)==3 and {x['kind'] for x in group}=={'reference','retained','frozen'}
  for x in group[1:]:ck('paired_ALLwarmstates',x['states'][:17],group[0]['states'][:17],1e-14);ck('paired_ALLwarmobservations',x['observed'][:17],group[0]['observed'][:17],1e-14);ck('paired_ALLgoals',x['record']['goals'],group[0]['record']['goals'],1e-14)
  live=next(x for x in group if x['kind']=='retained');frozen=next(x for x in group if x['kind']=='frozen');ck('paired_FIRSTaction_beforeappend',live['commands'][16:18],frozen['commands'][16:18],1e-12);ck('paired_FIRSTstates_beforeappend',live['states'][:19],frozen['states'][:19],1e-12)
  diff=frozen['score']['task_cost']-live['score']['task_cost'];ck('paired_saved_retained_cost',live['score']['task_cost'],frozen['record']['paired_retained_task_cost'],1e-8);ck('paired_saved_difference',diff,frozen['record']['frozen_minus_retained_task_cost'],1e-8)
  pairrows.append(dict(case_index=case,context_id=case//2,noise_repeat=case%2,retained_task_cost=live['score']['task_cost'],frozen_task_cost=frozen['score']['task_cost'],frozen_minus_retained_task_cost=diff,retained_precise_goals=live['score']['precise_goals'],frozen_precise_goals=frozen['score']['precise_goals']))
 states=np.array([x['states'] for x in entries]);commands=np.array([x['commands'] for x in entries]);phys=physics(states,commands,'control',False)
 ordinary=result['ordinary_episodes'];witness=result['witnesses'];resources=dict(acquisition_trajectories=40,acquisition_intervals=10240,acquisition_simulated_seconds=1024.,acquisition_independent_scalar_camera_draws=20560,local_candidate_families=4,bootstrap_libraries_per_candidate=3,neural_fits=0,ordinary_episodes=16,sustained_goal_windows=48,ordinary_decisions=1440,ordinary_intervals=3136,ordinary_simulated_seconds=313.6,ordinary_camera_read_opportunities=3152,ordinary_scalar_noise_opportunities=6304,fresh_initial_preparations=4,fresh_paired_noise_paths=8,distinct_scalar_control_noise_draws=3152,retained_actual_row_events=720,retained_member_append_copies=2160,frozen_real_pair_notifications=720,frozen_actual_append_copies=0,reference_episodes=8,privileged_reference_intervals=1568,privileged_reference_seconds=156.8,new_training_experience_from_reference=0,ordinary_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in ordinary),reference_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in witness),modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in ordinary for d in e['decisions']),unsuccessful_optimizer_terminations=sum(not d['optimizer']['success'] for e in ordinary for d in e['decisions']),fallback_decisions=sum(d['fallback_to_initial'] for e in ordinary for d in e['decisions']));assert resources==result['resources']
 compact={kind:dict(episodes=len(scores),mean_task_cost=math.fsum(s['task_cost'] for s in scores)/8,mean_combined_charged_cost=math.fsum(s['combined_charged_cost'] for s in scores)/8,task_costs=[s['task_cost'] for s in scores],total_precise_goals=sum(s['precise_goals'] for s in scores),phase_precision_counts=[sum(s['phases'][p]['precision_success'] for s in scores) for p in range(3)],phase_mean_costs=[math.fsum(s['phases'][p]['cost'] for s in scores)/8 for p in range(3)]) for kind,scores in summaries.items()}
 return dict(resources=resources,summaries=compact,paired_cases=pairrows,mean_frozen_minus_retained_taskcost=math.fsum(p['frozen_minus_retained_task_cost'] for p in pairrows)/8,reference_saturated_decisions=saturated,reference_clipped_goals=clipped,physics=phys),[rp,ap,fp,op,ep,acp,tp,src,*[ROOT/name for name in result['dependency_sha256']]]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--replicates',type=int,nargs='+',default=list(range(200,208)));parser.add_argument('--prefix-template',default=str(ROOT/'cartesian_stream_v17_rep{rep}'));args=parser.parse_args();assert len(set(args.replicates))==len(args.replicates) and all(200<=x<=207 for x in args.replicates)
 for rep in args.replicates:
  output=ROOT/f'audit_cartesian_stream_v17_rep{rep}_study_physics_results.json'
  if output.exists():raise FileExistsError(output)
 started=time.time();seen=set();outputs=[]
 for rep in args.replicates:
  ERROR.clear();prior.ERROR.clear();begin=time.time();prefix=Path(args.prefix_template.format(rep=rep));acq,acqpaths=acquire(prefix,rep,seen);ctl,ctlpaths=control(prefix,rep,seen)
  artifacts={p.name:sha(p) for p in [*acqpaths,*ctlpaths,Path(prior.__file__),Path(mechanics.__file__)]}
  out=dict(status='PASS — independent fresh acquired/control complete physics/RNG/witness/score audit; fitting/ordinary decisions separate',replicate=rep,audit_source_sha256=sha(Path(__file__)),artifact_sha256=artifacts,maximum_reconstruction_errors={**ERROR,**{'score_'+k:v for k,v in prior.ERROR.items()}},acquisition=acq,control=ctl,late_precision_endpoint_indices=[[67,76],[127,136],[187,196]],limitations=['Previously independently derived Hamiltonian, Christoffel/IK and scoring helpers reused; no canonical plant/learner/controller imported.','Local fits/selection/gradients/optimizer plans and append content are separately audited; recorded ordinary actions are physical inputs here.','Fixed one-body acquisition/preparation/task law; neither all-purpose learning nor optimality certificate.','Sampled final10endpoint precision and support diagnostics do not certify continuous-time conditions.','Paired controls share noise and warmup; later histories differ, so append comparison is total closed-loop effect.','Historical RHS/modeled counters reconstructed from arrays/records, not independently observed timing.'],wall_seconds=time.time()-begin)
  path=ROOT/f'audit_cartesian_stream_v17_rep{rep}_study_physics_results.json';path.write_text(json.dumps(out,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None,indent=2)+'\n');outputs.append(path.name);print(json.dumps(dict(replicate=rep,status=out['status'],scope_acquisition=acq['physics']['scope'],scope_control=ctl['physics']['scope'],wall_seconds=out['wall_seconds'])),flush=True)
 print(json.dumps(dict(completed=outputs,unique_rng_roles_checked=len(seen),wall_seconds=time.time()-started)),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
