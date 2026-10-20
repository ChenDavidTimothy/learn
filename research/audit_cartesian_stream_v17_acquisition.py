#!/usr/bin/env python3
"""Independent moving-corpus RNG/data/physics replay. No canonical imports.
Uses the previously independent Hamiltonian RK4/Radau implementation; this is
new artifact verification, not a claim of another independent physics derivation.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as mechanics
ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-10):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)

def main():
 start=time.time();rp=ROOT/'cartesian_stream_v17_rep0_acquisition.json';op=ROOT/'cartesian_stream_v17_rep0_ordinary.npz';ep=ROOT/'cartesian_stream_v17_rep0_evaluator.npz';meta=json.loads(rp.read_text())
 src=ROOT/'cartesian_stream_v17_acquire.py';design=ROOT/'cartesian_stream_v17_acquisition_design.md';plant=ROOT/'coupled_context_v14_pilot.py'
 assert sha(src)==meta['source_sha256'] and sha(design)==meta['design_sha256'] and sha(plant)==meta['plant_source_sha256']
 assert all(sha(ROOT/k)==v for k,v in meta['artifacts'].items())
 with np.load(op) as d:ordinary={k:d[k] for k in d.files}
 with np.load(ep) as d:evaluator={k:d[k] for k in d.files}
 assert set(ordinary)=={'observed','commands','split','trajectory_index','dt','sensor_sd'}
 assert set(evaluator)=={'states','sensor_noise','plant_nfev','seed_paths'}
 states=evaluator['states'];actions=ordinary['commands'];observed=ordinary['observed'];noise=evaluator['sensor_noise'];nfev=evaluator['plant_nfev'];paths=evaluator['seed_paths']
 assert states.shape==(40,257,4) and actions.shape==(40,256,2) and observed.shape==noise.shape==(40,257,2) and paths.shape==(40,3,7) and nfev.shape==(40,256)
 assert np.isfinite(states).all() and np.isfinite(observed).all() and np.isfinite(actions).all()
 assert np.max(abs(actions))<=.5 and np.all(nfev>0) and np.issubdtype(nfev.dtype,np.integer)
 assert ordinary['dt']==.1 and ordinary['sensor_sd']==.0002
 k=0;seen=set();initial_support=[]
 for split,count in enumerate((24,8,8)):
  for index in range(count):
   rngs=[];expectedpaths=[]
   for role in range(3):
    path=[20261020,17,650,0,split,index,role];assert tuple(path) not in seen;seen.add(tuple(path));expectedpaths.append(path);rngs.append(np.random.default_rng(np.random.SeedSequence(path)))
   init,commandrng,sensorrng=rngs;theta=init.uniform(-np.pi,np.pi);phi=init.uniform(.4,np.pi-.4);sign=init.choice([-1.,1.]);speed=init.uniform(.02,.12);direction=init.uniform(-np.pi,np.pi)
   q=np.array([theta-sign*phi/2,sign*phi]);V=speed*np.array([np.cos(direction),np.sin(direction)]);v=np.linalg.solve(mechanics.jac(q),V)
   ck('initial_state_rng',np.r_[q,v],states[k,0],2e-15);ck('initial_cartesian_velocity',mechanics.jac(states[k,0,:2])@states[k,0,2:],V,2e-15)
   expected=np.zeros((256,2));expected[8:]=np.repeat(commandrng.uniform(-.5,.5,(124,2)),2,axis=0)
   ck('all_command_rng',expected,actions[k],1e-15);ck('all_sensor_noise_rng',sensorrng.normal(0,.0002,(257,2)),noise[k],1e-15);ck('all_seed_paths',expectedpaths,paths[k],1e-15)
   assert ordinary['split'][k]==split and ordinary['trajectory_index'][k]==index
   initial_support.append(dict(split=split,index=index,theta=theta,phi=phi,sign=sign,speed=speed,direction=direction));k+=1
 assert k==40 and len(seen)==120
 ck('held_two_command_commitment',actions[:,8::2],actions[:,9::2],1e-15);assert np.array_equal(actions[:,:8],np.zeros((40,8,2)))
 positions=mechanics.endpoint(states[:,:,:2]);ck('all_noisy_camera_readings',positions+noise,observed,1e-15)
 # ALL actual intervals independently advanced from their recorded start states.
 local=mechanics.advance(states[:,:-1].reshape(-1,4),actions.reshape(-1,2),steps=128)
 ck('all_local_Hamiltonian_RK4_intervals',local,states[:,1:].reshape(-1,4),1e-9)
 # Also replay complete trajectories from their true initial conditions, no interval resetting.
 current=states[:,0].copy();rolling_max=0.;rolling_position_max=0.
 for t in range(256):
  current=mechanics.advance(current,actions[:,t],steps=128)
  rolling_max=max(rolling_max,float(np.max(abs(current-states[:,t+1]))))
  rolling_position_max=max(rolling_position_max,float(np.max(abs(mechanics.endpoint(current[:,:2])-positions[:,t+1]))))
 assert rolling_max<1e-7 and rolling_position_max<1e-7,(rolling_max,rolling_position_max)
 ERROR['complete_Hamiltonian_trajectory_states']=rolling_max;ERROR['complete_Hamiltonian_trajectory_positions']=rolling_position_max
 # Reconstruct numerical/support diagnostics independently from arrays.
 M=mechanics.mass(states[:,:,:2]);energy=.5*np.einsum('nti,ntij,ntj->nt',states[:,:,2:],M,states[:,:,2:]);work=np.sum(actions*np.diff(states[:,:,:2],axis=1),axis=2);resid=np.diff(energy,axis=1)-work
 J=mechanics.jac(states[:,:,:2]);velocity=(J@states[:,:,2:,None])[...,0];speed=np.linalg.norm(velocity,axis=2);det=abs(np.sin(states[:,:,1]));qstep=abs(np.diff(states[:,:,:2],axis=1));phase=abs(np.diff(states[:,:,:2].sum(axis=2),axis=1));pstep=np.linalg.norm(np.diff(positions,axis=1),axis=2)
 quantiles=[0,.01,.1,.5,.9,.99,1]
 def quant(x):return {str(q):float(np.quantile(x,q)) for q in quantiles}
 coverage=dict(maximum_energy_work_residual=float(np.max(abs(resid))),minimum_mass_eigenvalue=float(np.linalg.eigvalsh(M).min()),joint_coordinate_step_abs=quant(qstep.ravel()),distal_phase_step_abs=quant(phase.ravel()),camera_true_step_norm=quant(pstep),cartesian_speed=quant(speed),joint_speed_norm=quant(np.linalg.norm(states[:,:,2:],axis=2)),absolute_jacobian_determinant=quant(det),sampled_near_singular_abs_det_below_point01=int(np.sum(det<.01)))
 for key,value in coverage.items():
  if isinstance(value,dict):
   for q,v in value.items():ck('coverage_'+key+'_'+q,v,meta['coverage_evaluator_only'][key][q],1e-11)
  else:ck('coverage_'+key,value,meta['coverage_evaluator_only'][key],1e-11)
 # Representative/exreme interval starts selected before comparing numerical errors.
 candidates={(i,t) for i in range(40) for t in(0,7,8,127,255)}
 extrema={}
 for name,array,axis in [('cartesian_speed',speed,0),('joint_speed',np.linalg.norm(states[:,:,2:],axis=2),0),('near_singular',det,1),('qstep',qstep.max(axis=2),0),('distal_phase',phase,0)]:
  flat=int(np.argmin(array) if axis else np.argmax(array));i,t=np.unravel_index(flat,array.shape);t=min(t,255);candidates.add((int(i),int(t)));extrema[name]=[int(i),int(t)]
 # Independent adaptive Radau at all40 initial intervals and10 evenly spread samples plus extrema.
 radau_indices={(i,0) for i in range(40)}|{(i,127) for i in range(0,40,4)}|set(map(tuple,extrema.values()));radau_errors=[]
 for i,t in sorted(radau_indices):
  out=mechanics.radau(states[i,t],actions[i,t]);err=float(np.max(abs(out-states[i,t+1])));radau_errors.append(dict(trajectory=i,interval=t,maximum_state_difference=err));assert err<1e-9,(i,t,err)
 signs=np.sin(states[:,:,1]);cross=signs[:,:-1]*signs[:,1:]<0
 resources=dict(trajectories=40,physical_intervals=10240,physical_seconds=1024.,passive_prefix_intervals=320,passive_prefix_seconds=32.,intervention_intervals=9920,intervention_seconds=992.,scalar_noise_draws=20560,cartesian_camera_reads=10280,plant_rhs_evaluations=int(nfev.sum()))
 assert resources==meta['resources']
 artifacts=[rp,op,ep,src,design,plant,Path(mechanics.__file__)]
 out=dict(status='PASS — independent complete moving-acquisition RNG/camera/Hamiltonian/coverage audit; no learner or control result',artifact_sha256={p.name:sha(p) for p in artifacts},audit_source_sha256=sha(Path(__file__)),maximum_reconstruction_errors=ERROR,resources=resources,initial_preparations_evaluator_only=initial_support,coverage_reconstructed=coverage,additional_support=dict(sample_bracketed_jacobian_sign_changes=int(cross.sum()),trajectories_with_bracketed_sign_change=int(cross.any(1).sum())),independent_adaptive_Radau_checks=radau_errors,maximum_Radau_difference=max(q['maximum_state_difference'] for q in radau_errors),extreme_interval_indices=extrema,audit_work=dict(local_Hamiltonian_intervals=10240,complete_Hamiltonian_replayed_trajectories=40,RK4_substeps_per_interval=128,adaptive_Radau_intervals=len(radau_errors)),limitations=['No canonical mechanics/learner implementation imported; physics helper is previously independently derived Hamiltonian code reused here, not a fresh independent derivation claim.','Stored RHS counter sums independently reconstruct; audit does not observe historical walltime or recreate original DOP853 evaluation counts with another solver.','Quantiles/sign changes describe sampled states; they are not uniform continuous-time support/aliasing guarantees or hidden-state filters.','Single exploratory corpus; overlapping rows not independent learned bodies.','Ordinary-file schema separation verified; later learner source still requires dataflow audit.','No learned predictor, heldoutfork forecast, taskcontrol, or milestone claim.'],wall_seconds=time.time()-start)
 path=ROOT/'audit_cartesian_stream_v17_acquisition_results.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','maximum_reconstruction_errors','additional_support','maximum_Radau_difference','audit_work','wall_seconds')}))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
