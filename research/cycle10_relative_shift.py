"""Relative mechanism-shift test via observable Walsh features.
This is an established finite-feature two-sample procedure.
"""
import itertools, json, math, random, statistics, time, os
import numpy as np
from cycle02_unknown_structure import D,OUT,NOISE,XALL,rule_world,observe
PRE=96
POST=192
REPS=160
ALPHA=.05
ROOT=20261110
DEG=int(os.getenv("FEATURE_DEGREE","4"))
SETS=[c for k in range(DEG+1)
      for c in itertools.combinations(range(D),k)]
FEATURES=np.array(
    [np.prod(1-2*XALL[:,c],axis=1) if c else
     np.ones(len(XALL)) for c in SETS],
    dtype=np.float64).T
THRESHOLD=math.sqrt(
    2*(1/PRE+1/POST)*math.log(2*OUT*len(SETS)/ALPHA))
def one(seed,condition):
    py=random.Random(seed)
    old,new,changed=rule_world(py,condition.replace("stable_",""))
    if condition.startswith("stable_"):
        new=old.copy()
        changed=np.array([],dtype=int)
    rg=np.random.default_rng(seed+800)
    bx=rg.integers(0,len(XALL),size=PRE)
    ax=rg.integers(0,len(XALL),size=POST)
    by=np.array([observe(old,int(x),rg) for x in bx])
    ay=np.array([observe(new,int(x),rg) for x in ax])
    b=(2*by.astype(np.int8)-1).astype(float)
    a=(2*ay.astype(np.int8)-1).astype(float)
    mu0=b.T@FEATURES[bx]/PRE
    mu1=a.T@FEATURES[ax]/POST
    scores=np.max(np.abs(mu1-mu0),axis=1)
    flags=set(np.where(scores>=THRESHOLD)[0].tolist())
    changed=set(map(int,changed))
    return {"tp":len(changed&flags),
            "fp":len(flags-changed),
            "fn":len(changed-flags),
            "alerts":len(flags),
            "max_score":float(max(scores))}
def main():
    started=time.perf_counter()
    out={"meta":{"pre":PRE,"post":POST,"replications":REPS,
                 "noise":NOISE,"alpha":ALPHA,
                 "features":len(SETS),"max_degree":DEG,
                 "threshold":THRESHOLD,
                 "same_pre_post_input_distribution":True,
                 "no_known_rule_family_assumed":True},
         "conditions":{}}
    for cond in ("in_class","novel_parity4","dense_threshold",
                 "stable_in_class","stable_dense_threshold"):
        rows=[one(ROOT+i*97,cond) for i in range(REPS)]
        info={key:{"mean":statistics.mean(r[key] for r in rows),
                   "SE":statistics.stdev(r[key] for r in rows)/REPS**.5}
              for key in rows[0]}
        info["any_false"]=sum(r["fp"]>0 for r in rows)
        info["exact_localization"]=sum(
            r["fn"]==0 and r["fp"]==0 for r in rows)
        out["conditions"][cond]=info
        print(cond,"true hits",info["tp"]["mean"],
              "false hits",info["fp"]["mean"],
              "exact",info["exact_localization"],"/",REPS,
              flush=True)
    out["time_seconds"]=time.perf_counter()-started
    path=f"/home/dtdav/dev/learn/research/cycle10_relative_shift_d{DEG}.json"
    with open(path,"w") as f:
        json.dump(out,f,indent=2)
    print("saved",path,"time",round(out["time_seconds"],2))
if __name__=="__main__":
    main()
