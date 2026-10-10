"""Cycle 16: direct action-contrast learners, strong causal baselines.

Randomized action assignment a ~ Bernoulli(.5) makes
Z=(2a-1)*(2Y-1) a valid transformed outcome:
E[Z | history] = E[Y|a=1,history]-E[Y|a=0,history].
A regression of Z on history features can therefore optimize
action choice directly without fitting both outcome distributions.
This is classical treatment-effect / policy learning, not new.
"""
import json, math, statistics, time
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from cycle12_partial_obs import ROOT,N_PRE,make_task,sample,selected_channel
from cycle13_oracle import prob_oracle

NUI=7
TRIALS=12
BUDGETS=(192,768,3072,8192)
TEST=4000
METHODS=("direct_ridge","direct_lasso","t_learner")

def contrast_features(x,k,t):
    v=x[:,:2*k].astype(float)*2-1
    ap=x[:,2*k].astype(float)*2-1
    u0=v[:,t];u1=v[:,k+t]
    cols=[np.ones(len(x)),u1,u0,ap,u1*ap,u0*ap]
    for j in range(k):
        if j==t:continue
        v0,v1=v[:,j],v[:,k+j]
        cols.extend((u1*v1,u0*v1,u1*v0,u0*v0,
                     u1*v1*ap,u0*v0*ap))
    return np.column_stack(cols)

def train_predict(method,x,y,w,xtest,k,t):
    f=contrast_features(x,k,t)
    ft=contrast_features(xtest,k,t)
    a=x[:,-1]
    if method=="direct_ridge":
        z=(2*a.astype(float)-1)*(2*y.astype(float)-1)
        model=Ridge(alpha=90.,fit_intercept=False)
        model.fit(f,z,sample_weight=w)
        return model.predict(ft)
    if method=="direct_lasso":
        z=(2*a.astype(float)-1)*(2*y.astype(float)-1)
        model=Lasso(alpha=.004,fit_intercept=False,max_iter=2000)
        model.fit(f,z,sample_weight=w)
        return model.predict(ft)
    # Strong meta-learner baseline: fit separate potential outcome models.
    p=[]
    for arm in (0,1):
        chosen=a==arm
        model=LogisticRegression(C=.4,l1_ratio=1.0,solver="liblinear",
                                 max_iter=1000)
        model.fit(f[chosen],y[chosen],sample_weight=w[chosen])
        p.append(model.predict_proba(ft)[:,1])
    return p[1]-p[0]

def regret(task,x,action):
    x0=x.copy();x0[:,-1]=0
    x1=x.copy();x1[:,-1]=1
    o0=prob_oracle(task,x0,"contextual_action")
    o1=prob_oracle(task,x1,"contextual_action")
    optimal=np.maximum(o0,o1)
    prediction=np.where(action==1,o1,o0)
    return float(np.mean(optimal-prediction))

def one(seed):
    task=make_task(seed,NUI);k=task["k"]
    pre,py=sample(task,seed+1,N_PRE,"stationary")
    target=selected_channel(pre,py,k)
    assert target==task["target_channel"]
    post,qy=sample(task,seed+883,BUDGETS[-1],"contextual_action")
    test,ty=sample(task,seed+991,TEST,"contextual_action")
    results={}
    for budget in BUDGETS:
        xx=np.r_[pre,post[:budget]]
        yy=np.r_[py,qy[:budget]]
        weights=np.r_[np.full(N_PRE,.1),np.ones(budget)]
        results[str(budget)]={}
        for method in METHODS:
            started=time.perf_counter()
            contrast=train_predict(method,xx,yy,weights,
                                   test,k,target)
            action=(contrast>0).astype(np.uint8)
            results[str(budget)][method]={
                "regret":regret(task,test,action),
                "training_seconds":time.perf_counter()-started,
                "action1_rate":float(np.mean(action))}
    return results

def main():
    started=time.perf_counter()
    out={"meta":{"pre":N_PRE,"budget":BUDGETS,"trials":TRIALS,
                 "randomized_actions":True,
                 "known_treatment_propensity":.5,
                 "no_oracle_in_learning":True,
                 "pre_old_weight":.1,
                 "same_data_seeds_as_cycle15":True},
         "results":{}}
    allresults=[]
    for i in range(TRIALS):
        allresults.append(one(ROOT+1199*i))
        print("finished",i+1,"/",TRIALS,
              "runtime",round(time.perf_counter()-started,1),flush=True)
    for budget in BUDGETS:
        out["results"][str(budget)]={}
        for method in METHODS:
            rows=[r[str(budget)][method] for r in allresults]
            out["results"][str(budget)][method]={
                key:{"mean":statistics.mean(a[key] for a in rows),
                     "SE":statistics.stdev(a[key] for a in rows)
                          /math.sqrt(TRIALS)}
                for key in rows[0]}
    out["runtime_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle16_treatment_effect.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("saved",path)
    for budget in BUDGETS:
        print("BUDGET",budget,
              {method:round(out["results"][str(budget)][method]["regret"]["mean"],5)
               for method in METHODS})
if __name__=="__main__":
    main()
