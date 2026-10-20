#!/usr/bin/env python3
"""Exploratory cross-body learning screen; measured transitions, no oracle labels.

Full q/v sensing is deliberately supplied to isolate learning an update rule.
The mass/length/damping/motor values remain evaluator-only. Vectorized standard
horizontal two-link point-mass mechanics extends the validated unit-body plant.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DT=.05
SEED=20261020


def rhs(z,u,p):
    m1,m2,l1,l2,d1,d2,g1,g2,c=np.moveaxis(p,-1,0)
    q2=z[...,1];v1=z[...,2];v2=z[...,3]
    a=(m1+m2)*l1*l1+m2*l2*l2+2*m2*l1*l2*np.cos(q2)
    b=m2*l2*l2+m2*l1*l2*np.cos(q2);d=m2*l2*l2
    h=m2*l1*l2*np.sin(q2)
    t1=g1*u[...,0]+c*u[...,1]-d1*v1+h*(2*v1*v2+v2*v2)
    t2=c*u[...,0]+g2*u[...,1]-d2*v2-h*v1*v1
    det=a*d-b*b
    assert np.all(det>0)
    return np.stack([v1,v2,(d*t1-b*t2)/det,(a*t2-b*t1)/det],axis=-1)


def step(z,u,p,substeps=8):
    z=z.copy();h=DT/substeps
    for _ in range(substeps):
        a=rhs(z,u,p);b=rhs(z+.5*h*a,u,p)
        c=rhs(z+.5*h*b,u,p);d=rhs(z+h*c,u,p)
        z+=h/6*(a+2*b+2*c+d)
    return z


def features(z,u):
    return np.concatenate([np.sin(z[...,:2]),np.cos(z[...,:2]),z[...,2:],u],axis=-1)


def parameters(rng,n,ood=False):
    if ood:
        # Every dimension shifts, with signs independent; not just a nuisance drift.
        choices=rng.integers(0,2,(n,6))
        masses_lengths=np.where(choices[:,:4],rng.uniform(1.35,1.65,(n,4)),rng.uniform(.5,.7,(n,4)))
        gains=np.where(choices[:,4:],rng.uniform(1.35,1.65,(n,2)),rng.uniform(.5,.7,(n,2)))
        damping=rng.uniform(.01,.3,(n,2));coupling=rng.uniform(-.3,.3,(n,1))
    else:
        masses_lengths=rng.uniform(.75,1.25,(n,4))
        damping=rng.uniform(.05,.2,(n,2));gains=rng.uniform(.8,1.2,(n,2));coupling=rng.uniform(-.15,.15,(n,1))
    return np.c_[masses_lengths,damping,gains,coupling]


def checks():
    # Reuse already validated unit-body implementation as the instrument check.
    from coupled_context_v14_pilot import rhs as parent_rhs,flow
    rng=np.random.default_rng(np.random.SeedSequence([SEED,18,11,0]))
    p=np.array([1,1,1,1,0,0,1,1,0.]);errors=[];refinement=[]
    for _ in range(6):
        z=rng.uniform(-1,1,4);u=rng.uniform(-1,1,2)
        errors.append(np.max(abs(rhs(z,u,p)-parent_rhs(0,z,u))))
        exact,_=flow(z[:2],z[2:],u,DT)
        refinement.append([float(np.max(abs(step(z,u,p,k)-exact))) for k in (4,8,16)])
    assert max(errors)<1e-12 and np.max(np.array(refinement)[:,1])<1e-8
    # Generalized cases: independently solve DOP853 only a few representative flows.
    from scipy.integrate import solve_ivp
    errs=[]
    for pi in parameters(rng,6,True):
        z=rng.uniform(-1,1,4);u=rng.uniform(-1,1,2)
        sol=solve_ivp(lambda _,a:rhs(a,u,pi),(0,DT),z,method='DOP853',rtol=1e-11,atol=1e-13)
        assert sol.success;errs.append(float(np.max(abs(step(z,u,pi)-sol.y[:,-1]))))
    assert max(errs)<1e-7
    return dict(unit_rhs_max=float(max(errors)),unit_step_refinement=refinement,generalized_flow_spot_errors=errs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-prefix',default=str(ROOT/'error_transport_screen_data'))
    args=parser.parse_args();prefix=Path(args.output_prefix)
    ordinary=prefix.with_suffix('.npz');truth=Path(str(prefix)+'_evaluator.npz');report=Path(str(prefix)+'_report.json')
    assert not any(p.exists() for p in (ordinary,truth,report))
    start=time.time();verification=checks();splits=[('train',32),('validation',8),('test_id',8),('test_ood',8)]
    allx=[];ally=[];allstate=[];allaction=[];allp=[];split=[]
    for si,(name,n) in enumerate(splits):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,18,10,si,0]));p=parameters(rng,n,si==3)
        init=np.random.default_rng(np.random.SeedSequence([SEED,18,10,si,1]))
        z=np.concatenate([init.uniform(-np.pi,np.pi,(n,8,2)),init.uniform(-.5,.5,(n,8,2))],axis=-1)
        commands=np.random.default_rng(np.random.SeedSequence([SEED,18,10,si,2])).uniform(-1,1,(n,8,48,2)).repeat(2,axis=2)
        states=[z.copy()]
        for t in range(96):
            z=step(z,commands[:,:,t],p[:,None,:]);assert np.isfinite(z).all();states.append(z.copy())
        states=np.stack(states,axis=2);x=features(states[:,:,:-1],commands);y=np.diff(states[...,2:],axis=2)/DT
        allx.append(x);ally.append(y);allstate.append(states);allaction.append(commands);allp.append(p);split += [si]*n
        print('sampled',name,n,'physical bodies',flush=True)
    X=np.concatenate(allx);Y=np.concatenate(ally);states=np.concatenate(allstate);commands=np.concatenate(allaction)
    np.savez_compressed(ordinary,x=X,y=Y,split=np.array(split),dt=np.array(DT))
    np.savez_compressed(truth,states=states,commands=commands,parameters=np.concatenate(allp))
    out=dict(status='COMPLETE EXPLORATORY CROSS-BODY DATA; NO FIT/TEST VALUE INSPECTED',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ordinary_sha256=hashlib.sha256(ordinary.read_bytes()).hexdigest(),evaluator_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),seed_roles=[SEED,18,10,'split','parameter0/initial1/action2'],
        splits=dict(splits),trajectories_per_body=8,steps=96,dt=DT,physical_intervals=int(np.prod(X.shape[:3])),simulated_seconds=float(np.prod(X.shape[:3])*DT),
        measurement='Exact observed q/v and issued actions; finite observed delta-v/dt targets, no derivative/model labels. Simplified noiseless exploratory screen.',
        supplied_features='sin/cos both q, both v, issued u; grouped same-body trajectories',checks=verification,wall_seconds=time.time()-start)
    report.write_text(json.dumps(out,indent=2)+'\n');print('saved',ordinary,flush=True)


if __name__=='__main__':main()
