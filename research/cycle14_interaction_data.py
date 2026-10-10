"""Cycle 14: learning a newly consequential interaction after partial observation.

Compare a task-only learned HMM to sparse logistic, all degree-2
interactions, and supervised target-conditioned interaction selection.
All models receive the SAME noisy supervised post-change examples.
The target sensor must be discovered from prechange observations.
Fixed regularization and prechange discount, no target test tuning.
This is a structured feature baseline, NOT a new learning principle.
"""
import json, math, statistics, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from cycle12_partial_obs import (
    ROOT, make_task, sample, selected_channel, log_loss, N_PRE
)
from cycle13_latent_hmm import train_hmm, p_model

NUI=11
REPS=16
REGIMES=("stationary","nuisance_shift","coupled_target")
POST_BUDGETS=(192,768,3072,12288)
TEST=5000
C=.15

def target_interaction_features(x,k,j):
    yy=x[:,k+j].astype(float)*2-1
    zz=x[:,j].astype(float)*2-1
    action=x[:,2*k:].astype(float)
    obs0=x[:,:k].astype(float)*2-1
    obs1=x[:,k:2*k].astype(float)*2-1
    features=[zz, yy, action[:,0]*2-1, action[:,1]*2-1,
              zz*(action[:,0]*2-1),yy*(action[:,1]*2-1)]
    for idx in range(k):
        if idx!=j:
            features.extend((yy*obs1[:,idx],zz*obs0[:,idx]))
    return np.stack(features,axis=1)

def fit_predict(method,train_x,train_y,weights,test_x,k,j,seed):
    if method=="hmm":
        params=train_hmm(train_x,train_y,weights,k,j)
        return p_model(test_x,k,j,params)
    if method=="sparse_logistic":
        a,b=train_x,test_x
        c=.5
    elif method=="target_interactions":
        a=target_interaction_features(train_x,k,j)
        b=target_interaction_features(test_x,k,j)
        c=C
    elif method=="all_interactions":
        poly=PolynomialFeatures(2,interaction_only=True,
                                include_bias=False)
        a=poly.fit_transform(train_x)
        b=poly.transform(test_x)
        c=C
    else:
        raise ValueError(method)
    model=LogisticRegression(C=c,penalty="l1",solver="liblinear",
                             max_iter=1000)
    model.fit(a,train_y,sample_weight=weights)
    return np.clip(model.predict_proba(b)[:,1],.002,.998)

METHODS=("hmm","sparse_logistic","target_interactions",
         "all_interactions")

def experiment(seed,condition):
    task=make_task(seed,NUI)
    k=task["k"]
    pre_x,pre_y=sample(task,seed+1,N_PRE,"stationary")
    maximum=POST_BUDGETS[-1]
    post_x,post_y=sample(task,seed+765,maximum,condition)
    test_x,test_y=sample(task,seed+900,TEST,condition)
    j=selected_channel(pre_x,pre_y,k)
    assert j==task["target_channel"],"selected the wrong target channel"
    answers={}
    for budget in POST_BUDGETS:
        # Discount prechange evidence, allow learning new mechanism.
        train_x=np.concatenate((pre_x,post_x[:budget]))
        train_y=np.r_[pre_y,post_y[:budget]]
        w=np.r_[np.full(N_PRE,.1),np.ones(budget)]
        answers[str(budget)]={}
        for method in METHODS:
            before=time.perf_counter()
            p=fit_predict(method,train_x,train_y,w,test_x,k,j,seed)
            answers[str(budget)][method]={
                "loss":log_loss(p,test_y),
                "train_seconds":time.perf_counter()-before
            }
    return answers

def main():
    started=time.perf_counter()
    output={"design":{"nui":NUI,"pre":N_PRE,
                      "post_budgets":POST_BUDGETS,
                      "test_samples":TEST,"trials":REPS,
                      "pre_weight":.1,
                      "C_target_interactions":C,
                      "new_coupling_index_hidden_from_models":True,
                      "future_target_channel_learned_from_pre":True,
                      "models_train_on_matched_data":True},
            "conditions":{}}
    for cond in REGIMES:
        runs=[experiment(ROOT+713*i,cond) for i in range(REPS)]
        output["conditions"][cond]={}
        for budget in POST_BUDGETS:
            output["conditions"][cond][str(budget)]={}
            for method in METHODS:
                scores=[run[str(budget)][method] for run in runs]
                output["conditions"][cond][str(budget)][method]={
                    key:{"mean":statistics.mean(s[key] for s in scores),
                         "SE":statistics.stdev(s[key] for s in scores)
                              /math.sqrt(REPS)}
                    for key in ("loss","train_seconds")}
        print("DONE",cond,"duration",
              round(time.perf_counter()-started,1),flush=True)
    output["runtime_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle14_interaction_data.json"
    with open(path,"w") as f:json.dump(output,f,indent=2)
    print("saved",path)
    for c,vals in output["conditions"].items():
        print("CONDITION",c)
        for budget in POST_BUDGETS:
            print("budget",budget,{name:round(
                vals[str(budget)][name]["loss"]["mean"],5)
                for name in METHODS})
if __name__=="__main__":
    main()
