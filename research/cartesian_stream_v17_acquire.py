#!/usr/bin/env python3
"""Frozen moving acquisition; no learner or controller. See acquisition design."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
from coupled_context_v14_pilot import flow,mass

DT=.1
SD=.0002
STEPS=256
COUNTS=(24,8,8)

def endpoint(q):
    q=np.asarray(q)
    return np.stack((np.cos(q[...,0])+np.cos(q.sum(axis=-1)),
        np.sin(q[...,0])+np.sin(q.sum(axis=-1))),axis=-1)

def jacobian(q):
    return np.array([[-np.sin(q[0])-np.sin(q.sum()),-np.sin(q.sum())],
        [np.cos(q[0])+np.cos(q.sum()),np.cos(q.sum())]])

def role_rng(rep,split,index,role):
    path=[20261020,17,650,int(rep),int(split),int(index),int(role)]
    return np.random.default_rng(np.random.SeedSequence(path)),path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,default=0)
    parser.add_argument('--output-prefix',default='research/cartesian_stream_v17_rep0')
    args=parser.parse_args();prefix=Path(args.output_prefix)
    targets=[Path(str(prefix)+s) for s in ('_ordinary.npz','_evaluator.npz','_acquisition.json')]
    failure_path=Path(str(prefix)+'_failure.npz')
    if any(p.exists() for p in targets+[failure_path]):raise FileExistsError('Refusing to overwrite acquisition artifacts')
    start=time.time();states=[];observed=[];commands=[];noise=[];splits=[];indices=[];paths=[];nfevs=[]
    with threadpool_limits(limits=1):
        for split,count in enumerate(COUNTS):
            for index in range(count):
                init,p0=role_rng(args.replicate,split,index,0)
                actions,p1=role_rng(args.replicate,split,index,1)
                sensor,p2=role_rng(args.replicate,split,index,2)
                theta=init.uniform(-np.pi,np.pi);phi=init.uniform(.4,np.pi-.4)
                sign=init.choice([-1.,1.]);speed=init.uniform(.02,.12);direction=init.uniform(-np.pi,np.pi)
                q=np.array([theta-sign*phi/2,sign*phi]);V=speed*np.array([np.cos(direction),np.sin(direction)])
                z=np.r_[q,np.linalg.solve(jacobian(q),V)]
                a=np.vstack((np.zeros((8,2)),np.repeat(actions.uniform(-.5,.5,((STEPS-8)//2,2)),2,axis=0)))
                zs=[z.copy()];counts=[]
                for u in a:
                    try:z,n=flow(z[:2],z[2:],u,DT)
                    except Exception:
                        # Keep every completed trajectory and current partial path.
                        prefix.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(failure_path,completed_states=np.array(states),
                            completed_observed=np.array(observed),completed_commands=np.array(commands),
                            current_partial_states=np.array(zs),current_commands=a,
                            current_seed_paths=np.array([p0,p1,p2]),current_completed_nfev=np.array(counts))
                        raise
                    zs.append(z.copy());counts.append(n)
                zs=np.array(zs);eps=sensor.normal(0,SD,(STEPS+1,2))
                states.append(zs);noise.append(eps);observed.append(endpoint(zs[:,:2])+eps)
                commands.append(a);splits.append(split);indices.append(index);paths.append([p0,p1,p2]);nfevs.append(counts)
    states=np.array(states);observed=np.array(observed);commands=np.array(commands);noise=np.array(noise)
    nfevs=np.array(nfevs);positions=endpoint(states[:,:,:2]);vel=np.empty_like(positions)
    min_mass=1e10;work_errors=[]
    for zseq,aseq in zip(states,commands):
        energies=np.array([.5*z[2:]@mass(z[:2])@z[2:] for z in zseq])
        work_errors.extend(np.diff(energies)-np.sum(aseq*np.diff(zseq[:,:2],axis=0),axis=1))
        min_mass=min(min_mass,min(np.linalg.eigvalsh(mass(z[:2]))[0] for z in zseq))
    for i,zseq in enumerate(states):
        for t,z in enumerate(zseq):vel[i,t]=jacobian(z[:2])@z[2:]
    quantiles=[0,.01,.1,.5,.9,.99,1]
    def qs(a):return dict(zip(map(str,quantiles),np.quantile(a,quantiles).tolist()))
    diagnostics=dict(maximum_energy_work_residual=float(np.max(abs(np.array(work_errors)))),
        minimum_mass_eigenvalue=float(min_mass),joint_coordinate_step_abs=qs(abs(np.diff(states[:,:,:2],axis=1)).ravel()),
        distal_phase_step_abs=qs(abs(np.diff(states[:,:,:2].sum(axis=2),axis=1)).ravel()),
        camera_true_step_norm=qs(np.linalg.norm(np.diff(positions,axis=1),axis=2)),
        cartesian_speed=qs(np.linalg.norm(vel,axis=2)),joint_speed_norm=qs(np.linalg.norm(states[:,:,2:],axis=2)),
        absolute_jacobian_determinant=qs(abs(np.sin(states[:,:,1]))),
        sampled_near_singular_abs_det_below_point01=int(np.sum(abs(np.sin(states[:,:,1]))<.01)))
    checks_ok=(states.shape==(40,257,4) and commands.shape==(40,256,2) and
        bool(np.all(np.isfinite(states))) and diagnostics['maximum_energy_work_residual']<1e-8)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(targets[0],observed=observed,commands=commands,split=np.array(splits),trajectory_index=np.array(indices),dt=DT,sensor_sd=SD)
    np.savez_compressed(targets[1],states=states,sensor_noise=noise,plant_nfev=nfevs,seed_paths=np.array(paths))
    result=dict(status=('EXPLORATORY MOVING ACQUISITION; NO LEARNER OR RELEASE' if checks_ok else
        'FAILED ACQUISITION CONSISTENCY; RAW ARRAYS PRESERVED'),replicate=args.replicate,
        source_sha256=digest(__file__),plant_source_sha256=digest(Path(__file__).with_name('coupled_context_v14_pilot.py')),
        design_sha256=digest(Path(__file__).with_name('cartesian_stream_v17_acquisition_design.md')),
        artifacts={p.name:digest(p) for p in targets[:2]},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        dt=DT,sensor_sd=SD,counts=list(COUNTS),steps=STEPS,decision_interval=.2,resources=dict(trajectories=40,
            physical_intervals=40*STEPS,physical_seconds=40*STEPS*DT,passive_prefix_intervals=40*8,passive_prefix_seconds=40*8*DT,
            intervention_intervals=40*(STEPS-8),intervention_seconds=40*(STEPS-8)*DT,
            scalar_noise_draws=int(noise.size),cartesian_camera_reads=40*(STEPS+1),plant_rhs_evaluations=int(nfevs.sum())),
        coverage_evaluator_only=diagnostics,wall_seconds=time.time()-start,
        limitations='Ordinary data contain only noisy Cartesian positions, commanded torques, dt/noise scale and trajectory/split IDs. Evaluator data and coverage are not learner inputs. Initial support is not a later trajectory envelope. Renewed actions only at even t>=8.')
    targets[2].write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    assert checks_ok,'Consistency check failed; inspect preserved raw artifacts'

if __name__=='__main__':main()
