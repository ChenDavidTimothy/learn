"""Cycle 11: Select old-mechanism fingerprints, test on fresh data.

Signature discovery and testing use disjoint prechange observations.
The test is conditionally valid: after selecting feature indices from
independent training observations, compare calibration to postchange.
Hoeffding and Bonferroni control false alerts for arbitrary functions.
This is ordinary split-sample two-sample testing, not novel theory.
"""
import itertools, json, math, random, statistics, time
import numpy as np
from cycle02_unknown_structure import (
    D,OUT,NOISE,XALL,rule_world,observe)
PRE_TRAIN=48
PRE_CAL=48
POST=192
N=160
ROOT=20261110
ALPHA=.05
CONFIGS=((1,1),(3,1),(4,1),(4,2),(4,4))
FEATURE_CACHE={}
for degree in sorted(set(c[0] for c in CONFIGS)):
    tuples=[s for k in range(degree+1)
            for s in itertools.combinations(range(D),k)]
    matrix=np.array([
        np.prod(1-2*XALL[:,idx],axis=1) if idx
        else np.ones(len(XALL))
        for idx in tuples],dtype=float).T
    FEATURE_CACHE[degree]=matrix

def one(seed,condition,degree,topk):
    py=random.Random(seed)
    old,new,changed=rule_world(py,condition.replace("stable_",""))
    if condition.startswith("stable_"):
        new=old.copy()
        changed=np.array([],dtype=int)
    rg=np.random.default_rng(seed+800)
    b=rg.integers(0,len(XALL),size=PRE_TRAIN+PRE_CAL)
    a=rg.integers(0,len(XALL),size=POST)
    by=np.array([observe(old,int(x),rg) for x in b])
    ay=np.array([observe(new,int(x),rg) for x in a])
    feature=FEATURE_CACHE[degree]
    f_train=feature[b[:PRE_TRAIN]]
    f_cal=feature[b[PRE_TRAIN:]]
    f_post=feature[a]
    y_train=2*by[:PRE_TRAIN].astype(np.int8)-1
    y_cal=2*by[PRE_TRAIN:].astype(np.int8)-1
    y_post=2*ay.astype(np.int8)-1
    train_means=y_train.T@f_train/PRE_TRAIN
    top=np.argsort(np.abs(train_means),axis=1)[:,-topk:]
    threshold=math.sqrt(
        2*(1/PRE_CAL+1/POST)*math.log(2*OUT*topk/ALPHA))
    flags=[]
    scores=[]
    for j in range(OUT):
        select=top[j]
        mu0=y_cal[:,j]@f_cal[:,select]/PRE_CAL
        mu1=y_post[:,j]@f_post[:,select]/POST
        score=np.max(np.abs(mu1-mu0))
        scores.append(score)
        flags.append(score>=threshold)
    flags={i for i,val in enumerate(flags) if val}
    changed=set(map(int,changed))
    return {"tp":len(flags&changed),"fp":len(flags-changed),
            "fn":len(changed-flags),"alerts":len(flags),
            "threshold":threshold}
def main():
    beginning=time.perf_counter()
    results={"meta":{"pre_selection":PRE_TRAIN,
                     "pre_calibration":PRE_CAL,
                     "post_observations":POST,
                     "trials":N,"alpha_familywise":ALPHA,
                     "selection_and_calibration_independent":True},
             "conditions":{}}
    for degree,topk in CONFIGS:
        label=f"d{degree}_top{topk}"
        results["conditions"][label]={}
        for condition in ("in_class","novel_parity4","dense_threshold",
                          "stable_in_class","stable_dense_threshold"):
            rows=[one(ROOT+i*97,condition,degree,topk)
                  for i in range(N)]
            stats={metric:statistics.mean(r[metric] for r in rows)
                   for metric in ("tp","fp","fn","alerts","threshold")}
            stats["exact"]=sum(r["fn"]==0 and r["fp"]==0
                               for r in rows)
            stats["any_false"]=sum(r["fp"]>0 for r in rows)
            results["conditions"][label][condition]=stats
        print(label,"novel_exact",
              results["conditions"][label]["novel_parity4"]["exact"],
              "dense_exact",
              results["conditions"][label]["dense_threshold"]["exact"],
              "elapsed",round(time.perf_counter()-beginning,2),
              flush=True)
    results["time_seconds"]=time.perf_counter()-beginning
    path="/home/dtdav/dev/learn/research/cycle11_fingerprint.json"
    with open(path,"w") as f:json.dump(results,f,indent=2)
    print("SAVED",path,"time",round(results["time_seconds"],2))
if __name__=="__main__":
    main()
