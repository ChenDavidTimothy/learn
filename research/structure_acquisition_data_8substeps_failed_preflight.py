#!/usr/bin/env python3
"""New experience and later bodies for the frozen structure-acquisition screen.

Plant constants/parameters below are evaluator-only, not learner inputs.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from error_transport_data import rhs as old_rhs,step as old_step,parameters,features,DT
from error_transport_screen import digest,save_json

ROOT=Path(__file__).resolve().parent

def rhs(z,u,p,alpha,beta):
    pp=np.broadcast_to(p,(*z.shape[:-1],9)).copy()
    pp[...,4:6]+=beta*np.abs(z[...,2:])
    return old_rhs(z,u+alpha*u**3,pp)

def step(z,u,p,alpha,beta):
    z=z.copy();h=DT/8
    for _ in range(8):
        a=rhs(z,u,p,alpha,beta);b=rhs(z+.5*h*a,u,p,alpha,beta)
        c=rhs(z+.5*h*b,u,p,alpha,beta);d=rhs(z+h*c,u,p,alpha,beta)
        z+=h/6*(a+2*b+2*c+d)
    return z

def check():
    rng=np.random.default_rng(np.random.SeedSequence([20261020,18,40,99]))
    out=[];zero=[]
    for p in parameters(rng,6,True):
        z=rng.uniform(-1,1,4);u=rng.uniform(-1,1,2);a=rng.uniform(.5,1.5,2);b=rng.uniform(.1,.3,2)
        reference=solve_ivp(lambda _,zz:rhs(zz,u,p,a,b),(0,DT),z,method='DOP853',rtol=1e-11,atol=1e-13)
        assert reference.success
        out.append(float(np.max(abs(step(z,u,p,a,b)-reference.y[:,-1]))))
        zero.append(float(np.max(abs(step(z,u,p,0,0)-old_step(z,u,p)))))
    assert max(out)<1e-7 and max(zero)<1e-13,(out,zero)
    return dict(new_flow_DOP853_spot_error=out,zero_change_old_flow_error=zero)

def main():
    ordinary=ROOT/'structure_acquisition_data.npz';truth=ROOT/'structure_acquisition_data_evaluator.npz';report=ROOT/'structure_acquisition_data_report.json'
    assert not any(p.exists() for p in (ordinary,truth,report))
    start=time.time();checks=check();xx=[];yy=[];zz=[];uu=[];ps=[];aa=[];bb=[];splits=[]
    for si,n in enumerate((8,4,8,8,8)):
        rng=lambda role:np.random.default_rng(np.random.SeedSequence([20261020,18,40,si,role]))
        p=parameters(rng(0),n,si==3);r=rng(1)
        z=np.concatenate([r.uniform(-np.pi,np.pi,(n,8,2)),r.uniform(-.5,.5,(n,8,2))],axis=-1)
        u=rng(2).uniform(-1,1,(n,8,48,2)).repeat(2,axis=2);r=rng(3)
        alpha=r.uniform(.5,1.5,(n,2)) if si!=4 else np.zeros((n,2))
        beta=r.uniform(.1,.3,(n,2)) if si!=4 else np.zeros((n,2))
        states=[z.copy()]
        for t in range(96):
            z=step(z,u[:,:,t],p[:,None],alpha[:,None],beta[:,None]);assert np.isfinite(z).all();states.append(z.copy())
        states=np.stack(states,axis=2)
        xx.append(features(states[:,:,:-1],u));yy.append(np.diff(states[...,2:],axis=2)/DT)
        zz.append(states);uu.append(u);ps.append(p);aa.append(alpha);bb.append(beta);splits += [si]*n
    np.savez_compressed(ordinary,x=np.concatenate(xx),y=np.concatenate(yy),split=np.array(splits),dt=np.array(DT))
    np.savez_compressed(truth,states=np.concatenate(zz),commands=np.concatenate(uu),parameters=np.concatenate(ps),alpha=np.concatenate(aa),beta=np.concatenate(bb))
    save_json(report,dict(status='FRESH ACQUISITION/VALIDATION/LATER TEST GENERATED BEFORE LEARNER FITTING',
        source_sha256=digest(__file__),physics_parent_sha256=digest(ROOT/'error_transport_data.py'),plan_sha256=digest(ROOT/'structure_acquisition_plan.md'),
        ordinary_sha256=digest(ordinary),evaluator_sha256=digest(truth),seed_roles=[20261020,18,40,'split','role'],
        bodies=36,trajectories=288,physical_intervals=27648,simulated_seconds=1382.4,checks=checks,wall_seconds=time.time()-start,
        splits=dict(acquisition=8,validation=4,new_id=8,new_ood=8,old_id=8),scope='Exact q/v and finite velocity differences. New nonlinearities align with supplied grammar. Evaluator coefficients never learner features.'))
    print('Saved new physical experience and frozen later bodies',flush=True)

if __name__=='__main__':main()
