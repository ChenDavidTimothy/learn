#!/usr/bin/env python3
"""Fresh holdout only; frozen sampling law from implicit_relation_plan.md."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from error_transport_data import DT,SEED,parameters,step,features

ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    prefix=ROOT/'implicit_relation_holdout'
    ordinary=prefix.with_suffix('.npz');truth=ROOT/'implicit_relation_holdout_evaluator.npz';report=ROOT/'implicit_relation_holdout_report.json'
    assert not any(p.exists() for p in (ordinary,truth,report))
    start=time.time();allx=[];ally=[];allz=[];allu=[];allp=[];split=[]
    for si in (2,3):
        role=lambda r:np.random.default_rng(np.random.SeedSequence([SEED,18,30,1,si,r]))
        p=parameters(role(0),8,si==3);init=role(1)
        z=np.concatenate([init.uniform(-np.pi,np.pi,(8,8,2)),init.uniform(-.5,.5,(8,8,2))],axis=-1)
        u=role(2).uniform(-1,1,(8,8,48,2)).repeat(2,axis=2);states=[z.copy()]
        for t in range(96):
            z=step(z,u[:,:,t],p[:,None,:]);assert np.isfinite(z).all();states.append(z.copy())
        states=np.stack(states,axis=2)
        allx.append(features(states[:,:,:-1],u));ally.append(np.diff(states[...,2:],axis=2)/DT)
        allz.append(states);allu.append(u);allp.append(p);split += [si]*8
    np.savez_compressed(ordinary,x=np.concatenate(allx),y=np.concatenate(ally),split=np.array(split),dt=np.array(DT))
    np.savez_compressed(truth,states=np.concatenate(allz),commands=np.concatenate(allu),parameters=np.concatenate(allp))
    out=dict(status='COMPLETE FRESH HOLDOUT; NO PREDICTION SCORES INSPECTED',seed_roles=[SEED,18,30,1,'split','parameter0/initial1/action2'],
        ordinary_sha256=digest(ordinary),evaluator_sha256=digest(truth),source_sha256=digest(__file__),physics_source_sha256=digest(ROOT/'error_transport_data.py'),
        plan_sha256=digest(ROOT/'implicit_relation_plan.md'),physical_bodies=16,trajectories=128,physical_intervals=12288,simulated_seconds=614.4,
        dt=DT,wall_seconds=time.time()-start,scope='Reuses validated unchanged acquisition instrument. No new full physics replay. True parameters evaluator-only.')
    report.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
