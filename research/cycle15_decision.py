"""Cycle 15: decision regret under hidden, action-dependent coupling.

Post-change, target flip rate is 0.08 when action matches nuisance bit,
otherwise 0.32. This makes optimal intervention depend on BOTH
partially observed target and nuisance states.

No model receives hidden states, relevant nuisance index, or oracle
parameters. Each learns from the same labeled histories and random actions.
Oracle is used ONLY to evaluate expected policy regret, not train policies.
"""
import json, math, statistics, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from cycle12_partial_obs import (
    ROOT,N_PRE, make_task,sample,selected_channel,log_loss
)
from cycle13_latent_hmm import p_model, train_hmm
from cycle13_oracle import prob_oracle

NUIS=7
REPS=12
BUDGETS=(192,768,3072,8192)
TEST=4000
C=.35
METHODS=("two_state_hmm","target_action_features",
         "full_degree3","full_l1")

def targeted(x,k,t):
    raw=x[:,:2*k].astype(float)*2-1
    a0=x[:,2*k].astype(float)*2-1
    a1=x[:,2*k+1].astype(float)*2-1
    u0=raw[:,t]
    u1=raw[:,k+t]
    cols=[u0,u1,a0,a1,u0*a0,u1*a1]
    for j in range(k):
        if j==t:continue
        v0=raw[:,j];v1=raw[:,k+j]
        cols += [v0,v1,u0*v0,u1*v1,a0*v0,a1*v1,
                 u0*a0*v0,u1*a1*v1]
    return np.column_stack(cols)

def train(method,x,y,w,k,t):
    if method=="two_state_hmm":
        return train_hmm(x,y,w,k,t)
    if method=="target_action_features":
        xx=targeted(x,k,t)
    elif method=="full_degree3":
        poly=PolynomialFeatures(3,interaction_only=True,include_bias=False)
        xx=poly.fit_transform(x)
    else:xx=x
    if method=="full_degree3":
        reg=.2
    else:reg=C
    clf=LogisticRegression(solver="liblinear",max_iter=800,
                           l1_ratio=1.0,C=reg)
    clf.fit(xx,y,sample_weight=w)
    return clf

def predict(method,model,x,k,t):
    if method=="two_state_hmm":
        return p_model(x,k,t,model)
    if method=="target_action_features":
        xx=targeted(x,k,t)
    elif method=="full_degree3":
        xx=PolynomialFeatures(
            3,interaction_only=True,include_bias=False).fit_transform(x)
    else:xx=x
    return np.clip(model.predict_proba(xx)[:,1],.002,.998)

def evaluate(method,model,task,x,y,learned_channel):
    k=task["k"]; t=learned_channel
    actual_p=predict(method,model,x,k,t)
    test_loss=log_loss(actual_p,y)
    x0=x.copy(); x0[:,-1]=0
    x1=x.copy(); x1[:,-1]=1
    policy=(predict(method,model,x1,k,t)>
            predict(method,model,x0,k,t)).astype(np.uint8)
    oracle_p0=prob_oracle(task,x0,"contextual_action")
    oracle_p1=prob_oracle(task,x1,"contextual_action")
    best=np.maximum(oracle_p0,oracle_p1)
    selected=np.where(policy==1,oracle_p1,oracle_p0)
    regret=float(np.mean(best-selected))
    random_regret=float(np.mean(
        best-(oracle_p0+oracle_p1)/2))
    return {"predictive_loss":test_loss,
            "policy_regret":regret,
            "random_policy_regret":random_regret,
            "fraction_action_one":float(np.mean(policy))}
def run(seed):
    task=make_task(seed,NUIS)
    k=task["k"]; t=task["target_channel"]
    pre_x,pre_y=sample(task,seed+1,N_PRE,"stationary")
    chosen=selected_channel(pre_x,pre_y,k)
    if chosen !=t:
        raise AssertionError("channel selection failed")
    post_x,post_y=sample(task,seed+883,BUDGETS[-1],
                         "contextual_action")
    x_test,y_test=sample(task,seed+991,TEST,
                         "contextual_action")
    results={}
    for budget in BUDGETS:
        xx=np.r_[pre_x,post_x[:budget]]
        yy=np.r_[pre_y,post_y[:budget]]
        weights=np.r_[np.full(N_PRE,.1),np.ones(budget)]
        results[str(budget)]={}
        for method in METHODS:
            tic=time.perf_counter()
            fitted=train(method,xx,yy,weights,k,chosen)
            metrics=evaluate(method,fitted,task,x_test,y_test,chosen)
            metrics["seconds"]=time.perf_counter()-tic
            results[str(budget)][method]=metrics
    return results

def main():
    tic=time.perf_counter()
    out={"protocol":{"nui":NUIS,"trials":REPS,
                     "pre":N_PRE,"post_budgets":BUDGETS,
                     "test":TEST,"C_target":C,
                     "oracle_only_for_regret_evaluation":True,
                     "task_index_learned":True,
                     "coupled_nuisance_index_not_revealed":True,
                     "equal_observation_budgets":True},
         "results":{}}
    records=[]
    for i in range(REPS):
        records.append(run(ROOT+1199*i))
        print("finished",i+1,"/",REPS,
              "elapsed",round(time.perf_counter()-tic,1),flush=True)
    for budget in BUDGETS:
        out["results"][str(budget)]={}
        for method in METHODS:
            each=[r[str(budget)][method] for r in records]
            out["results"][str(budget)][method]={
                key:{"mean":statistics.mean(a[key] for a in each),
                     "SE":statistics.stdev(a[key] for a in each)/REPS**.5}
                for key in each[0]}
    out["elapsed_seconds"]=time.perf_counter()-tic
    path="/home/dtdav/dev/learn/research/cycle15_decision.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("SAVED",path)
    for budget in BUDGETS:
        print("BUDGET",budget)
        for name,item in out["results"][str(budget)].items():
            print(name,"loss",round(item["predictive_loss"]["mean"],5),
                  "regret",round(item["policy_regret"]["mean"],5),
                  "random",round(item["random_policy_regret"]["mean"],5))
if __name__=="__main__":
    main()
