"""Cycle 12b: action-conditional prediction in a partially observed factor world.

All learners train from the same labeled action-observation histories.
True latent bits are never supplied to learners. Output is the next
noisy observation of the task-relevant factor, but its channel index
is not supplied; it must be learned from pre-shift examples.
The observer sees two noisy consecutive observations and both actions.
Environment is a controlled hidden Markov process with independent
nuisance factors unless the postshift coupled regime is selected.
"""
import json
import math
import os
import random
import statistics
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

N_PRE=960
N_VAL=240
N_POST=192
N_TEST=3000
REPS=20
NOISE=.18
NUI_COUNTS=(3,7,11)
CONDITIONS=("stationary","nuisance_shift","coupled_target",
            "target_rate_shift")
METHODS=("target_predictive","full_logistic_l2",
         "full_logistic_l1","full_interactions","random_forest")
ROOT=20261011

def make_task(seed, nuisance):
    rg=np.random.default_rng(seed+30111)
    k=nuisance+1
    perm=rg.permutation(k)
    # Channel from target latent index zero to raw observation index.
    obs_target=int(np.where(perm==0)[0][0])
    phi=np.r_[0.07,rg.uniform(.03,.26,size=nuisance)]
    return {"k":k,"perm":perm,"target_channel":obs_target,
            "flip":phi}

def sample(task, seed, n, regime):
    rg=np.random.default_rng(seed)
    k=task["k"]
    state=rg.integers(0,2,size=(n,k),dtype=np.uint8)
    ap=rg.integers(0,2,size=n,dtype=np.uint8)
    a=rg.integers(0,2,size=n,dtype=np.uint8)
    phi=task["flip"].copy()
    if regime=="nuisance_shift":
        phi[1:]=.45
    elif regime=="target_rate_shift":
        phi[0]=.32
    def evolve(st,action):
        rates=np.broadcast_to(phi,(n,k)).copy()
        rates[:,0] = np.where(action==0,phi[0],min(.49,phi[0]+.08))
        if regime=="coupled_target":
            # Action-conditional hidden nuisance influence on target flip.
            rates[:,0]=np.where(st[:,1]==1, .42, .02)
        elif regime=="contextual_action":
            # Best action depends jointly on hidden target and nuisance.
            rates[:,0]=np.where(action==st[:,1],.08,.32)
        flips=(rg.random((n,k))<rates).astype(np.uint8)
        return st^flips
    s0=state
    s1=evolve(s0,ap)
    s2=evolve(s1,a)
    noise=(rg.random((3,n,k))<NOISE).astype(np.uint8)
    obs0=(s0^noise[0])[:,task["perm"]]
    obs1=(s1^noise[1])[:,task["perm"]]
    obs2=(s2^noise[2])[:,task["perm"]]
    x=np.column_stack((obs0,obs1,ap,a)).astype(np.uint8)
    target=obs2[:,task["target_channel"]]
    return x,target

def selected_channel(x,y,k):
    # Selection done exclusively on pre-shift training data.
    # No target-channel index or true hidden state is provided.
    last_obs=x[:,k:2*k].astype(float)
    means=last_obs.mean(axis=0)
    yy=y.astype(float)
    score=np.abs(((last_obs-means)*(yy-yy.mean())[:,None]).mean(axis=0))
    return int(np.argmax(score))

def features_small(x,k,index):
    return (x[:,[index,k+index,2*k,2*k+1]]
            .astype(np.uint8))

def fit_table(train_x, train_y, train_w, k, index):
    z=features_small(train_x,k,index)
    ids=(z[:,0] | (z[:,1]<<1) | (z[:,2]<<2) | (z[:,3]<<3))
    positive=np.bincount(ids,weights=train_y*train_w,minlength=16)
    count=np.bincount(ids,weights=train_w,minlength=16)
    return (positive+1)/(count+2)

def predict_table(prob, x,k,index):
    z=features_small(x,k,index)
    ids=(z[:,0] | (z[:,1]<<1) | (z[:,2]<<2) | (z[:,3]<<3))
    return prob[ids]

def make_prediction(method,xold,yold,weights,xquery,index,k,seed):
    if method=="target_predictive":
        p=fit_table(xold,yold,weights,k,index)
        return predict_table(p,xquery,k,index)
    if method=="full_interactions":
        trans=PolynomialFeatures(2,interaction_only=True,
                                 include_bias=False)
        train=trans.fit_transform(xold)
        query=trans.transform(xquery)
        classifier=LogisticRegression(
            C=.15,max_iter=300,solver="lbfgs")
    elif method=="random_forest":
        train=xold; query=xquery
        classifier=RandomForestClassifier(
            n_estimators=100,max_depth=12,min_samples_leaf=5,
            random_state=seed,n_jobs=1)
    elif method=="full_logistic_l1":
        train=xold; query=xquery
        classifier=LogisticRegression(
            C=.5,penalty="l1",solver="liblinear",max_iter=300)
    else:
        train=xold;query=xquery
        classifier=LogisticRegression(C=1.0,solver="lbfgs",max_iter=300)
    if len(np.unique(yold))<2:
        return np.full(len(xquery),float(yold.mean()))
    classifier.fit(train,yold,sample_weight=weights)
    return classifier.predict_proba(query)[:,1]

def log_loss(p,y):
    p=np.clip(p,.002, .998)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))))

def trial(seed,nui,condition):
    task=make_task(seed,nui)
    k=task["k"]
    xpre,ypre=sample(task,seed+1,N_PRE,"stationary")
    xval,yval=sample(task,seed+2,N_VAL,"stationary")
    xpost,ypost=sample(task,seed+3,N_POST,condition)
    xeval,yeval=sample(task,seed+4,N_TEST,condition)
    channel=selected_channel(xpre,ypre,k)
    out={}
    for method in METHODS:
        # 0 post: unchanged pretrained model;
        # 192 post: refit on downweighted old samples plus new samples.
        # Same supervision and nominal old-sample discount for all methods.
        predictions={}
        for npost in (0,N_POST):
            xx=np.concatenate((xpre,xpost[:npost]),axis=0)
            yy=np.r_[ypre,ypost[:npost]]
            weights=np.r_[np.full(len(xpre),1. if npost==0 else .2),
                          np.ones(npost)]
            p=make_prediction(method,xx,yy,weights,xeval,channel,k,seed)
            predictions[str(npost)]=log_loss(p,yeval)
        out[method]=predictions
    return out,int(channel==task["target_channel"])

def main():
    timer=time.perf_counter()
    results={"config":{"pre":N_PRE,"post":N_POST,"validation":N_VAL,
                       "evaluation":N_TEST,"reps":REPS,
                       "observation_noise":NOISE,
                       "test_task_channel_unknown_to_models":True,
                       "nuisance_counts":NUI_COUNTS,
                       "conditions":CONDITIONS,
                       "same_labeled_data_access":True,
                       "train_new_old_weight":.2,
                       "pretrain_validation_unused":True},
             "conditions":{}}
    for nui in NUI_COUNTS:
        for condition in CONDITIONS:
            outcomes=[trial(ROOT+713*i,nui,condition) for i in range(REPS)]
            label=f"nuisance{nui}_{condition}"
            results["conditions"][label]={
              "channel_identification":sum(a[1] for a in outcomes),
              "methods":{
                  name:{step:{
                    "mean":statistics.mean(a[0][name][step] for a in outcomes),
                    "SE":statistics.stdev(a[0][name][step] for a in outcomes)/(REPS**.5)
                  }for step in ("0",str(N_POST))}
                  for name in METHODS}}
            print(label,"match",results["conditions"][label]["channel_identification"],
                  "target",round(results["conditions"][label]["methods"]["target_predictive"][str(N_POST)]["mean"],4),
                  "full-L1",round(results["conditions"][label]["methods"]["full_logistic_l1"][str(N_POST)]["mean"],4),
                  "time",round(time.perf_counter()-timer,1),flush=True)
    results["runtime_seconds"]=time.perf_counter()-timer
    path="/home/dtdav/dev/learn/research/cycle12_partial_obs.json"
    with open(path,"w") as f:json.dump(results,f,indent=2)
    print("SAVED",path,"seconds",results["runtime_seconds"])
if __name__=="__main__":
    main()
