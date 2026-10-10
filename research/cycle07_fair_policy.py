"""Cycle 06: honest selection between algebraic probes and generic learners.

Every method receives exactly 1024 oracle observations.
The adaptive policy spends 297 on parity basis probes, then 727
on independent uniform observations, using 180 held-out to decide.
Generic competitors receive 1024 uniform observations and select
between logistic and random forest using 180 held-out.
Selection uses noisy empirical log loss, never noiseless test truth.
"""
import json, math, statistics, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from cycle07_ransac import solve_passive
from cycle05_interventions import (
    D, EPS, K, REPEATS, bit_parity, world, draw_noise,
    active_basis_predictor, HOLDOUT, ROOT
)

BUDGET=1024
BASIS=(D+1)*REPEATS
VALID=180
TRIALS=36
SEED=31092026
MODES=("sparse_parity4","dense_parity12","majority11",
       "conjunction5","xor_and")
assert BASIS < BUDGET-VALID

def target(seed,mode):
    rng=np.random.default_rng(seed)
    if mode!="xor_and":
        return world(rng,mode)[0]
    mask=np.uint64(sum(1<<int(i) for i in rng.choice(D,4,replace=False)))
    conj=np.uint64(sum(1<<int(i) for i in rng.choice(D,5,replace=False)))
    intercept=int(rng.integers(0,2))
    return lambda x:(bit_parity(x&mask)^intercept^
                     ((x&conj)==conj).astype(np.uint8))

def as_binary(x):
    return ((np.asarray(x,dtype=np.uint64)[:,None] >>
             np.arange(D,dtype=np.uint64))&1).astype(np.uint8)

def train_generic(x,y,kind,seed):
    xx=as_binary(x)
    if len(np.unique(y))==1:
        return ("constant",float(np.mean(y)))
    if kind=="logistic":
        model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=500)
    elif kind=="forest":
        model=RandomForestClassifier(n_estimators=90,min_samples_leaf=3,
                                     n_jobs=1,random_state=seed)
    else:raise ValueError(kind)
    model.fit(xx,y)
    return ("model",model)

def predict_generic(obj,x):
    typ,m=obj
    if typ=="constant":return np.full(len(x),m)
    return m.predict_proba(as_binary(x))[:,1]

def losses(p,truth):
    p=np.clip(np.asarray(p,dtype=float),0.01,0.99)
    q=EPS+(1-2*EPS)*np.asarray(truth,dtype=float)
    ex=-q*np.log(p)-(1-q)*np.log1p(-p)
    return ex-( -EPS*math.log(EPS)-(1-EPS)*math.log1p(-EPS))

def predict_parity(mask,intercept,x):
    return EPS+(1-2*EPS)*(bit_parity(x&mask)^intercept)

def trial(seed,mode):
    f=target(seed,mode)
    rng=np.random.default_rng(seed+11001)
    xtest=rng.integers(0,1<<D,size=HOLDOUT,dtype=np.uint64)
    t0=time.perf_counter_ns()
    amask,aintercept,_,_=active_basis_predictor(
        f,np.random.default_rng(seed+991))
    xextra=rng.integers(0,1<<D,size=BUDGET-BASIS,dtype=np.uint64)
    yextra=draw_noise(xextra,f,rng)
    train_x,valid_x=xextra[:-VALID],xextra[-VALID:]
    train_y,valid_y=yextra[:-VALID],yextra[-VALID:]
    options={}
    for name in ("logistic","forest"):
        obj=train_generic(train_x,train_y,name,seed)
        options[name]=(obj,float(np.mean(
            -(valid_y*np.log(np.clip(predict_generic(obj,valid_x),0.01,0.99))+
              (1-valid_y)*np.log1p(-np.clip(predict_generic(obj,valid_x),0.01,0.99))))))
    pparity=predict_parity(amask,aintercept,valid_x)
    score_parity=float(np.mean(
        -(valid_y*np.log(pparity)+(1-valid_y)*np.log1p(-pparity))))
    best_generic=min(options,key=lambda s:options[s][1])
    # Minimum hold-out log score; no manual threshold or oracle.
    choice=("parity" if score_parity<=options[best_generic][1]
            else best_generic)
    if choice=="parity":
        pred=predict_parity(amask,aintercept,xtest)
    else:
        obj=train_generic(xextra,yextra,choice,seed)
        pred=predict_generic(obj,xtest)
    adaptive_time=(time.perf_counter_ns()-t0)/1e9
    adaptive_loss=float(np.mean(losses(pred,f(xtest))))

    t1=time.perf_counter_ns()
    # Passive competitor: same exact budget, but all uniform inputs.
    xpass=np.random.default_rng(seed+8802).integers(
        0,1<<D,size=BUDGET,dtype=np.uint64)
    ypass=draw_noise(xpass,f,np.random.default_rng(seed+77214))
    hold=(xpass[-VALID:],ypass[-VALID:])
    candidate={}
    for name in ("logistic","forest"):
        obj=train_generic(xpass[:-VALID],ypass[:-VALID],name,seed)
        pp=np.clip(predict_generic(obj,hold[0]),.01,.99)
        candidate[name]=float(np.mean(
            -(hold[1]*np.log(pp)+(1-hold[1])*np.log1p(-pp))))
    # Stronger passive comparator knows the same parity algebra.
    # It must infer the support from noisy, UNCONTROLLED observations.
    support,intercept,_,attempts=solve_passive(
        xpass[:-VALID],ypass[:-VALID],seed+921101)
    if support is not None:
        pval=np.clip(predict_parity(support,intercept,hold[0]),.01,.99)
        candidate["parity"]=float(np.mean(
            -(hold[1]*np.log(pval)+(1-hold[1])*np.log1p(-pval))))
    bestpass=min(candidate,key=candidate.get)
    if bestpass=="parity":
        support,intercept,_,_=solve_passive(xpass,ypass,seed+921101)
        passive_pred=predict_parity(support,intercept,xtest)
    else:
        passive_obj=train_generic(xpass,ypass,bestpass,seed)
        passive_pred=predict_generic(passive_obj,xtest)
    passive_time=(time.perf_counter_ns()-t1)/1e9
    passive_loss=float(np.mean(losses(passive_pred,f(xtest))))
    return {"adaptive_excess":adaptive_loss,
            "passive_excess":passive_loss,
            "parity_selected":int(choice=="parity"),
            "adaptive_ms":1000*adaptive_time,
            "passive_ms":1000*passive_time,
            "choice":choice,"passive_choice":bestpass}

def main():
    started=time.perf_counter()
    out={"meta":{"budget":BUDGET,"basis_budget":BASIS,
                 "other_queries":BUDGET-BASIS,"validation":VALID,
                 "modes":MODES,"trials":TRIALS,"forest_trees":90,
                 "evaluation_samples":HOLDOUT,
                 "query_budget_matched":True},
         "conditions":{}}
    for mode in MODES:
        rows=[]
        for i in range(TRIALS):
            rows.append(trial(SEED+i*107,mode))
        num=len(rows)
        out["conditions"][mode]={
            k:{"mean":statistics.mean(float(row[k]) for row in rows),
               "SE":statistics.stdev(float(row[k]) for row in rows)/num**.5}
            for k in ("adaptive_excess","passive_excess",
                      "parity_selected","adaptive_ms","passive_ms")}
        out["conditions"][mode]["policy_choices"]={
            name:sum(row["choice"]==name for row in rows)
            for name in ("parity","logistic","forest")}
        out["conditions"][mode]["passive_choices"]={
            name:sum(row["passive_choice"]==name for row in rows)
            for name in ("parity","logistic","forest")}
        diffs=[r["adaptive_excess"]-r["passive_excess"] for r in rows]
        out["conditions"][mode]["paired_excess_difference"]={
            "mean":statistics.mean(diffs),
            "SE":statistics.stdev(diffs)/num**.5}
        print("COMPLETED",mode,
              "adaptive",round(out["conditions"][mode]["adaptive_excess"]["mean"],4),
              "passive",round(out["conditions"][mode]["passive_excess"]["mean"],4),
              "choices",out["conditions"][mode]["policy_choices"],
              "elapsed",round(time.perf_counter()-started,1),flush=True)
    out["wall_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle07_fair_results.json"
    with open(path,"w") as fp:json.dump(out,fp,indent=2)
    print("SAVED:",path)
if __name__=="__main__":
    main()
