#!/usr/bin/env python3
"""Preplanned independent actual-control physical/RNG/witness/scoring audit.
No canonical control/learner/acquisition/mechanics module imported. Ordinary
optimizer/gradient/library replay is a separate independent artifact audit.
"""
from pathlib import Path
import hashlib,itertools,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as mechanics
ROOT=Path(__file__).resolve().parent
DT=.1;SD=.0002;BOUND=.5;WARM=16;LENGTH=196;PHASE=60
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-9):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def rng(path):return np.random.default_rng(np.random.SeedSequence(path))
def phase_goals(read):
 g=.95*read;answer=[]
 for angle in(0.,.04,-.04):
  c,s=np.cos(angle),np.sin(angle);answer.append(np.array([[c,-s],[s,c]])@g)
 return np.array(answer)

def independent_target(q,goal):
 radius=float(np.linalg.norm(goal));r=min(max(radius,0.),2.);point=np.asarray(goal).copy()
 if radius>0:point*=r/radius
 theta=float(np.arctan2(point[1],point[0]));phi=2*np.arccos(r/2);candidates=[]
 # Enumerate nearby lattice representatives, rather than reuse source inverse.
 for sign in(1.,-1.):
  base=np.array([q[0] if r==0 else theta-sign*phi/2,sign*phi]);center=np.floor((q-base)/(2*np.pi)).astype(int)
  for shift in itertools.product((-1,0,1,2),repeat=2):
   winding=center+shift;target=base+2*np.pi*winding;distance=float((q-target)@(q-target));candidates.append((distance,target,sign,winding))
 winner=min(candidates,key=lambda x:x[0]);return winner,point,radius,r

def coriolis_from_metric(q,v):
 s=np.sin(q[1]);derivatives=np.zeros((2,2,2));derivatives[1]=[[-2*s,-s],[-s,0.]];h=np.zeros(2)
 for i,j,k in itertools.product(range(2),repeat=3):
  christoffel=.5*(derivatives[k,i,j]+derivatives[j,i,k]-derivatives[i,j,k]);h[i]+=christoffel*v[j]*v[k]
 return h

def reconstruct_score(states,commands,goals):
 positions=mechanics.endpoint(states[:,:2]);velocity=(mechanics.jac(states[:,:2])@states[:,2:,None])[...,0];effort=np.sum(commands*commands,axis=1)
 warm_motion=DT*.25*math.fsum(float(x@x) for x in velocity[1:17]);warm_effort=DT*.01*math.fsum(map(float,effort[:16]));warm_sq=DT*math.fsum(map(float,effort[:16]))
 warm=dict(seconds=1.6,integrated_motion_penalty=warm_motion,integrated_effort_penalty=warm_effort,integrated_squared_effort=warm_sq,cost_without_unannounced_goal=warm_motion+warm_effort);phases=[]
 for phase in range(3):
  first=16+60*phase;last=first+60;errors=positions[first+1:last+1]-goals[phase];v=velocity[first+1:last+1];u=effort[first:last]
  pterm=np.sum(errors*errors,axis=1);vterm=np.sum(v*v,axis=1);p=DT*math.fsum(map(float,pterm));motion=DT*.25*math.fsum(map(float,vterm));force=DT*.01*math.fsum(map(float,u));squared=DT*math.fsum(map(float,u));cost=DT*math.fsum(float(x+.25*y+.01*z) for x,y,z in zip(pterm,vterm,u))
  precision=all(abs(float(e))<=.03 for e in errors[-10:].ravel()) and all(abs(float(vv))<=.03 for vv in v[-10:].ravel())
  phases.append(dict(phase=phase,goal=goals[phase],endpoint_start_index=first,endpoint_end_index=last,seconds=6.,integrated_position_error=p,integrated_motion_penalty=motion,integrated_effort_penalty=force,integrated_squared_effort=squared,cost=cost,position_error_by_sample=errors,cartesian_velocity_by_sample=v,squared_effort_by_interval=u,late_max_coordinate_position_error=float(abs(errors[-10:]).max()),late_max_coordinate_velocity=float(abs(v[-10:]).max()),precision_success=precision))
 task=math.fsum(p['cost'] for p in phases)
 return dict(warmup=warm,phases=phases,task_cost=task,combined_charged_cost=task+warm['cost_without_unannounced_goal'],physical_seconds=19.6,precise_goals=sum(p['precision_success'] for p in phases))

def check_score(actual,stored):
 for key in('task_cost','combined_charged_cost','physical_seconds','precise_goals'):ck('score_'+key,actual[key],stored[key],1e-8)
 for key,v in actual['warmup'].items():ck('warmup_'+key,v,stored['warmup'][key],1e-10)
 for p,s in zip(actual['phases'],stored['phases']):
  for key,v in p.items():
   if key=='precision_success':assert v==s[key]
   else:ck('phase_'+key,v,s[key],1e-8)

def main():
 start=time.time();rp=ROOT/'cartesian_stream_v17_rep0_control_results.json';ap=ROOT/'cartesian_stream_v17_rep0_control_arrays.npz';result=json.loads(rp.read_text());src=ROOT/'cartesian_stream_v17_control.py'
 assert result['status'].startswith('COMPLETE') and len(result['ordinary_episodes'])==16 and len(result['witnesses'])==8
 assert sha(src)==result['source_sha256'] and sha(ap)==result['arrays_sha256']
 for name,h in result['dependency_sha256'].items():assert sha(ROOT/name)==h
 fp=ROOT/'cartesian_stream_v17_rep0_fit.json';wp=ROOT/'cartesian_stream_v17_rep0_weights.npz';op=ROOT/'cartesian_stream_v17_rep0_ordinary.npz';fit=json.loads(fp.read_text())
 assert sha(fp)==result['fit_sha256'] and sha(wp)==result['weights_sha256']==fit['weights_sha256'] and sha(op)==result['ordinary_sha256']==fit['ordinary_sha256']
 with np.load(ap) as f:arrays={k:f[k] for k in f.files}
 cases=[];paths=set()
 for context_id in range(4):
  ip=[20261020,17,670,0,context_id,0];prep=rng(ip);theta=prep.uniform(-np.pi,np.pi);phi=prep.uniform(.4,np.pi-.4);sign=prep.choice([-1.,1.]);speed=prep.uniform(.02,.12);angle=prep.uniform(-np.pi,np.pi);q=np.array([theta-sign*phi/2,sign*phi]);V=speed*np.array([np.cos(angle),np.sin(angle)]);initial=np.r_[q,np.linalg.solve(mechanics.jac(q),V)];paths.add(tuple(ip))
  up=[20261020,17,670,0,context_id,2];uniform=rng(up).uniform(-.5,.5,(4,2));warm=np.zeros((16,2));warm[8:]=np.repeat(uniform,2,axis=0);paths.add(tuple(up))
  for repeat in range(2):
   npth=[20261020,17,670,0,context_id,1,repeat];noise=rng(npth).normal(0,SD,(197,2));paths.add(tuple(npth));index=len(cases)
   ck('case_initial_rng',initial,arrays[f'case{index}_initial_state'],1e-15);ck('case_warmup_rng',warm,arrays[f'case{index}_warmup_commands'],1e-15);ck('case_sensor_rng',noise,arrays[f'case{index}_sensor_noise'],1e-15)
   cases.append(dict(context_id=context_id,noise_repeat=repeat,initial_seed_path=ip,warmup_seed_path=up,noise_seed_path=npth,initial_state=initial,warmup=warm,noise=noise))
 assert len(paths)==16
 physical=[];commands_all=[];episode_rows=[];summaries={};witness_saturations=0;witness_clippings=0;originalkeys=set()
 for category in('witnesses','ordinary_episodes'):
  for index,e in enumerate(result[category]):
   case_index=index if category=='witnesses' else e['case_index'];case=cases[case_index]
   assert e['context_id']==case['context_id'] and e['noise_repeat']==case['noise_repeat']
   for name in('initial_seed_path','warmup_seed_path','noise_seed_path'):assert e[name]==case[name]
   keys=e['array_keys'];z=arrays[keys['states_evaluator_only']];u=arrays[keys['commands']];obs=arrays[keys['observed']];calls=arrays[keys['plant_nfev']]
   assert z.shape==(197,4) and u.shape==(196,2) and obs.shape==(197,2) and calls.shape==(196,)
   assert np.isfinite(z).all() and np.isfinite(obs).all() and np.isfinite(u).all() and abs(u).max()<=.5 and np.all(calls>0)
   ck('actual_initial_preparation',z[0],case['initial_state'],1e-15);ck('actual_warmup_commands',u[:16],case['warmup'],1e-15);ck('actual_camera_noise',mechanics.endpoint(z[:,:2])+case['noise'],obs,1e-15)
   goals=phase_goals(obs[16]);ck('goals_from_ordinary_read',goals,e['goals'],2e-15);ck('all_held_pairs',u[16::2],u[17::2],1e-15)
   assert len(e['decisions'])==90 and e['plant_rhs_evaluations']==int(calls.sum())
   for renewal,d in enumerate(e['decisions']):
    t=16+2*renewal;phase=renewal//30;assert d['renewal']==renewal and d['time_index']==t and d['phase']==phase;ck('decision_issued_command',u[t],d['command'],1e-15)
    if category=='ordinary_episodes':ck('decision_only_current_goal',d['current_goal'],goals[phase],2e-15)
    else:
     q,v=z[t,:2],z[t,2:];(distance,target,branch,winding),clipped,radius,r=independent_target(q,goals[phase]);recorded=np.array(d['joint_target']);ck('witness_joint_target',target,recorded,1e-9);ck('witness_target_endpoint',mechanics.endpoint(recorded),clipped,1e-9);ck('witness_minimum_target_distance',distance,d['joint_distance_squared'],1e-9)
     ck('witness_original_goal',goals[phase],d['original_goal'],2e-15);ck('witness_clipped_goal',clipped,d['clipped_goal'],1e-14);ck('witness_original_radius',radius,d['original_radius'],1e-14);ck('witness_clipped_radius',r,d['clipped_radius'],1e-14);assert d['clipped']==(radius!=r) and d['folded_goal']==(r==0)
     # Both branch/lattice diagnostics must describe the saved target.
     ck('witness_winding',winding,d['winding'],1e-15);assert float(branch)==d['branch']
     torque=mechanics.mass(q)@(-q+recorded-2*v)+coriolis_from_metric(q,v);ck('witness_computed_torque',torque,d['unbounded_command'],1e-9);ck('witness_bounded_command',np.clip(torque,-.5,.5),u[t],1e-9)
     saturated=bool(np.any(np.clip(torque,-.5,.5)!=torque));assert saturated==d['saturated'];witness_saturations+=saturated;witness_clippings+=bool(d['clipped'])
   score=reconstruct_score(z,u,goals);check_score(score,e['score']);kind=e['kind'];summaries.setdefault(kind,[]).append(score)
   physical.append(z);commands_all.append(u);episode_rows.append(dict(category=category,index=index,case_index=case_index,kind=kind,score=score,states=z,commands=u))
   if category=='ordinary_episodes':
    key=(case_index,kind);assert key not in originalkeys;originalkeys.add(key)
 assert originalkeys=={(case,kind) for case in range(8) for kind in('nonlinear','local')}
 physical=np.array(physical);command=np.array(commands_all)
 # Same warmup and goals in all method/witness episodes of a paired case.
 for case in range(8):
  group=[x for x in episode_rows if x['case_index']==case];assert len(group)==3
  for x in group[1:]:ck('paired_warmup_physics',x['states'][:17],group[0]['states'][:17],1e-14);ck('paired_goals',np.array([p['goal'] for p in x['score']['phases']]),np.array([p['goal'] for p in group[0]['score']['phases']]),1e-14)
 # All actual physical intervals, plus full rolling trajectories without resets.
 local=mechanics.advance(physical[:,:-1].reshape(-1,4),command.reshape(-1,2),steps=256);ck('ALL4704_local_Hamiltonian_intervals',local,physical[:,1:].reshape(-1,4),1e-8)
 current=physical[:,0].copy();state_max=position_max=0.
 for t in range(196):
  current=mechanics.advance(current,command[:,t],steps=256);state_max=max(state_max,float(abs(current-physical[:,t+1]).max()));position_max=max(position_max,float(abs(mechanics.endpoint(current[:,:2])-mechanics.endpoint(physical[:,t+1,:2])).max()))
 assert state_max<1e-6 and position_max<1e-6,(state_max,position_max);ERROR['ALL24_complete_Hamiltonian_states']=state_max;ERROR['ALL24_complete_Hamiltonian_positions']=position_max
 speed=np.linalg.norm((mechanics.jac(physical[:,:,:2])@physical[:,:,2:,None])[...,0],axis=2);joint=np.linalg.norm(physical[:,:,2:],axis=2);singular=abs(np.sin(physical[:,:,1]));qstep=abs(np.diff(physical[:,:,:2],axis=1)).max(axis=2)
 spots={(i,t) for i in range(24) for t in(0,16,76,136,194)};extremes={}
 for name,x,minimum in[('Cartesian_speed',speed,False),('joint_speed',joint,False),('Jacobian_singularity',singular,True),('joint_step',qstep,False)]:
  flat=int(np.argmin(x) if minimum else np.argmax(x));i,t=np.unravel_index(flat,x.shape);t=min(int(t),195);spots.add((int(i),t));extremes[name]=[int(i),t]
 radau=[]
 for i,t in sorted(spots):
  out=mechanics.radau(physical[i,t],command[i,t]);error=float(abs(out-physical[i,t+1]).max());assert error<1e-8,(i,t,error);radau.append(dict(episode=i,interval=t,error=error))
 ordinary=result['ordinary_episodes'];witnesses=result['witnesses'];resources=dict(ordinary_episodes=16,ordinary_phases=48,ordinary_decisions=1440,ordinary_intervals=3136,ordinary_simulated_seconds=313.6,ordinary_camera_read_opportunities=3152,ordinary_scalar_noise_opportunities=6304,distinct_initial_preparations=4,independent_paired_noise_paths=8,distinct_scalar_noise_rng_draws=3152,warmup_intervals_ordinary=256,privileged_witness_episodes=8,privileged_witness_intervals=1568,privileged_witness_seconds=156.8,new_training_experience_from_witness=0,ordinary_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in ordinary),witness_plant_rhs_evaluations=sum(e['plant_rhs_evaluations'] for e in witnesses),modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for e in ordinary for d in e['decisions']),unsuccessful_optimizer_terminations=sum(not d['optimizer']['success'] for e in ordinary for d in e['decisions']),fallback_decisions=sum(d['fallback_to_initial'] for e in ordinary for d in e['decisions']),actual_local_rows_appended_per_member_per_local_episode=90)
 for key,v in resources.items():ck('resources_'+key,v,result['resources'][key],1e-9)
 compact={}
 for kind,scores in summaries.items():
  compact[kind]=dict(episodes=len(scores),task_costs=[s['task_cost'] for s in scores],combined_charged_costs=[s['combined_charged_cost'] for s in scores],precise_goals=[s['precise_goals'] for s in scores],total_precise_goals=sum(s['precise_goals'] for s in scores),phase_precision_counts=[sum(s['phases'][p]['precision_success'] for s in scores) for p in range(3)],phase_mean_costs=[math.fsum(s['phases'][p]['cost'] for s in scores)/len(scores) for p in range(3)])
 artifacts=[rp,ap,src,fp,wp,op,Path(mechanics.__file__),*[ROOT/name for name in result['dependency_sha256']]]
 out=dict(status='PASS — independent all24episode physics/RNG/witness/score audit; ordinary planner/library audited separately',artifact_sha256={p.name:sha(p) for p in artifacts},audit_source_sha256=sha(Path(__file__)),maximum_reconstruction_errors=ERROR,resources=resources,summaries=compact,witness_saturated_decisions=witness_saturations,witness_goal_clippings=witness_clippings,late_precision_endpoint_indices=[[67,76],[127,136],[187,196]],independent_Radau_checks=radau,maximum_Radau_error=max(x['error'] for x in radau),extreme_interval_indices=extremes,scope=dict(all_physical_intervals=4704,complete_Hamiltonian_trajectories=24,RK4_substeps_per_interval=256,witness_decisions=720,ordinary_decisions_command_timing_only=1440),limitations=['Ordinary chosencommands treated as recorded inputs; optimizer/gradients/localcoefficients/append replay belong to separate independent artifact audit.','Witness is privileged bounded computedtorque evidence, not equally informed baseline or global feasibility/recovery certificate.','Sampled final10endpoints implement declared sustainedwindow, not continuous-time between-sample guarantees.','Four preparations/eight pairednoise paths andone acquiredcorpus, not24independenttrainedsystems.','RHS counters reconstructed from saved arrays, not independent observation of historical walltime/calls.','No physical data or goal/model changes; independent mechanics helper reused rather than another new derivation claim.'],wall_seconds=time.time()-start)
 (ROOT/'audit_cartesian_stream_v17_control_physics_results.json').write_text(json.dumps(out,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','maximum_reconstruction_errors','resources','summaries','witness_saturated_decisions','witness_goal_clippings','maximum_Radau_error','scope','wall_seconds')}))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
