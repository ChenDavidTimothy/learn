"""Cycle 13: trained factored latent-state HMM, strong structured baseline.

No latent states, transition rates, emission noise, or channel indices
are revealed. Learner identifies a candidate target sensor on PRE data,
and fits a controlled two-state HMM's conditional future observation
likelihood by weighted maximum likelihood. It sees the same supervised
input/target samples as Cycle 12b. It does know its two-state HMM family.
"""
import json, math, statistics, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from cycle12_partial_obs import (
    ROOT, N_PRE, N_POST, N_TEST, NOISE,
    make_task, sample, selected_channel, METHODS, log_loss
)
REGIMES=("stationary","nuisance_shift","coupled_target",
         "target_rate_shift")
NUI=(3,7,11)
TRIALS=20

def p_model(x,k,channel,parameters):
    eps, p0, p1=parameters
    x0=x[:,channel]
    x1=x[:,k+channel]
    a_prev=x[:,2*k]
    a=x[:,2*k+1]
    flip_prev=np.where(a_prev==0,p0,p1)
    flip_next=np.where(a==0,p0,p1)
    numer=np.zeros(len(x),dtype=float)
    denom=np.zeros(len(x),dtype=float)
    for s0 in (0,1):
        e0=np.where(x0==s0,1-eps,eps)
        for s1 in (0,1):
            e1=np.where(x1==s1,1-eps,eps)
            t=np.where(s0==s1,1-flip_prev,flip_prev)
            weight=.5*e0*t*e1
            p_s2=flip_next if s1==0 else (1-flip_next)
            py=eps+(1-2*eps)*p_s2
            numer+=weight*py
            denom+=weight
    return np.clip(numer/denom,1e-7,1-1e-7)

def train_hmm(x,y,weights,k,channel):
    y=y.astype(float)
    def objective(theta):
        p=p_model(x,k,channel,theta)
        return float(np.average(-y*np.log(p)-(1-y)*np.log1p(-p),
                                weights=weights))
    starts=((.16,.07,.15),(.28,.20,.27))
    fits=[]
    for start in starts:
        opt=minimize(objective,start,method="L-BFGS-B",
                     bounds=((.025,.475),(.001,.49),(.001,.49)),
                     options={"maxiter":90,"ftol":1e-9})
        fits.append((opt.fun,opt.x))
    return min(fits,key=lambda r:r[0])[1]

def one(seed,nui,condition):
    task=make_task(seed,nui)
    k=task["k"]
    pre,prey=sample(task,seed+1,N_PRE,"stationary")
    post,posty=sample(task,seed+3,N_POST,condition)
    test,testy=sample(task,seed+4,N_TEST,condition)
    channel=selected_channel(pre,prey,k)
    results={}
    for step in (0,N_POST):
        x=np.concatenate((pre,post[:step]),axis=0)
        y=np.r_[prey,posty[:step]]
        w=np.r_[np.full(len(pre),1 if step==0 else .2),
                 np.ones(step)]
        fitted=train_hmm(x,y,w,k,channel)
        pred=p_model(test,k,channel,fitted)
        results[str(step)]={"test_loss":log_loss(pred,testy),
                             "params":fitted.tolist()}
    return results

def main():
    started=time.perf_counter()
    out={"meta":{"pre":N_PRE,"post":N_POST,"test":N_TEST,
                 "trials":TRIALS,"all_method_inputs_equal":True,
                 "assumed_two_state_controlled_HMM":True,
                 "true_params_not_supplied":True,
                 "two_initializations":True,
                 "target_channel_chosen_from_pre_training":True},
         "conditions":{}}
    for nui in NUI:
        for condition in REGIMES:
            trials=[one(ROOT+713*i,nui,condition)
                    for i in range(TRIALS)]
            label=f"nuisance{nui}_{condition}"
            out["conditions"][label]={
                step:{"loss_mean":statistics.mean(t[step]["test_loss"]
                                                  for t in trials),
                      "loss_SE":statistics.stdev(t[step]["test_loss"]
                                                for t in trials)/TRIALS**.5}
                for step in ("0",str(N_POST))}
            print(label,"trainedHMM",round(
                out["conditions"][label][str(N_POST)]["loss_mean"],5),
                "runtime",round(time.perf_counter()-started,1),flush=True)
    out["runtime_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle13_latent_hmm.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("saved",path)
if __name__=="__main__":
    main()
