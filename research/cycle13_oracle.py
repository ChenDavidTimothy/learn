"""Cycle 13b: evaluate privileged exact Bayesian ceiling for partial observability.

Oracle knows true latent factor alignment, transitions, and observation noise.
Not a legitimate learning comparator, only an information-theoretic reference.
Uses target and first nuisance observation history, which is sufficient
for this specific synthetic generator. Other nuisance channels irrelevant.
"""
import json
import math
import statistics
import numpy as np
from cycle12_partial_obs import (
    ROOT, N_TEST, make_task, sample, log_loss, NOISE
)

CONDITIONS=("stationary","nuisance_shift","coupled_target",
            "target_rate_shift")
NUIS=(3,7,11)
TRIALS=20
STATES=[(u,z) for u in (0,1) for z in (0,1)]

def prob_oracle(task,x,regime,ignore_nuisance=False):
    n=x.shape[0]
    k=task["k"]
    j0=int(np.where(task["perm"]==0)[0][0])
    j1=int(np.where(task["perm"]==1)[0][0])
    eps=NOISE
    # Marginal over four latent states at t=1.
    b=np.zeros((n,4),dtype=float)
    def emission(observed,actual):
        return np.where(observed==actual,1-eps,eps)
    def rate_u(action,z):
        if regime=="coupled_target":
            return (np.where(z==1,.42,.02) if not ignore_nuisance
                    else np.full(len(action),.22))
        if regime=="contextual_action":
            return (np.where(action==z,.08,.32) if not ignore_nuisance
                    else np.full(len(action),.20))
        if regime=="target_rate_shift":base=.32
        else:base=.07
        return np.where(action==0,base,np.minimum(.49,base+.08))
    zrate=.45 if regime=="nuisance_shift" else task["flip"][1]
    prev_action=x[:,2*k]
    next_action=x[:,2*k+1]
    for i,(u0,z0) in enumerate(STATES):
        e0=emission(x[:,j0],u0)*emission(x[:,j1],z0)
        for j,(u1,z1) in enumerate(STATES):
            pu=rate_u(prev_action,z0)
            su=np.where(u1==u0,1-pu,pu)
            sz=(1-zrate if z0==z1 else zrate)
            e1=emission(x[:,k+j0],u1)*emission(x[:,k+j1],z1)
            b[:,j]+= .25*e0*su*sz*e1
    b/=b.sum(axis=1,keepdims=True)
    probs=np.zeros(n,dtype=float)
    for i,(u1,z1) in enumerate(STATES):
        pu=rate_u(next_action,z1)
        p2=pu if u1==0 else 1-pu
        py=eps+(1-2*eps)*p2
        probs+=b[:,i]*py
    return np.clip(probs,1e-8,1-1e-8)

def main():
    output={"meta":{"privileged_ceiling":True,
                     "true_parameters_and_indices_revealed_to_oracle":True,
                     "conditions":CONDITIONS,
                     "samples_test":N_TEST,"trials":TRIALS},
            "cases":{}}
    for nui in NUIS:
        for regime in CONDITIONS:
            vals=[]
            for i in range(TRIALS):
                seed=ROOT+713*i
                task=make_task(seed,nui)
                x,y=sample(task,seed+4,N_TEST,regime)
                full=prob_oracle(task,x,regime)
                reduced=prob_oracle(task,x,regime,ignore_nuisance=True)
                vals.append((log_loss(full,y),log_loss(reduced,y)))
            key=f"nuisance{nui}_{regime}"
            output["cases"][key]={
                "full_oracle":statistics.mean(a for a,b in vals),
                "reduced_misspecified":statistics.mean(b for a,b in vals),
                "paired_reduced_minus_full":statistics.mean(b-a for a,b in vals),
                "SE":statistics.stdev(b-a for a,b in vals)/TRIALS**.5}
            print(key,output["cases"][key],flush=True)
    path="/home/dtdav/dev/learn/research/cycle13_oracle.json"
    with open(path,"w") as f:json.dump(output,f,indent=2)
    print("SAVED",path)
if __name__=="__main__":
    main()
