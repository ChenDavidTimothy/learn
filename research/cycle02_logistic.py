"""Cycle 02 linear logistic classification baseline with calibration."""
import json,random,statistics,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from cycle02_unknown_structure import (
    OUT,NOISE,PRE,POST,CHECKPOINTS,SEED0,XALL,
    ENTROPY_NOISE,rule_world,observe)
N=24
METHODS=("logistic_replay","logistic_recent")
def evaluation(p,new,changed,indices):
    target=NOISE+(1-2*NOISE)*new[:,indices]
    pred=np.clip(p.T,1e-5,1-1e-5)
    err=(-(target*np.log(pred)+(1-target)*np.log1p(-pred))
         .mean(axis=1)-ENTROPY_NOISE)
    return {"all":float(err.mean()),"changed":float(err[changed].mean()),
            "retained":float(np.delete(err,changed).mean())}
def run(seed,cond):
    rng=np.random.default_rng(seed)
    old,new,changed=rule_world(random.Random(seed),cond)
    ix0=rng.integers(0,len(XALL),size=PRE)
    y0=np.array([observe(old,int(i),rng) for i in ix0])
    ix1=np.random.default_rng(seed+221133).integers(0,len(XALL),size=POST)
    y1=np.array([observe(new,int(ix),np.random.default_rng(
        seed*1000003+(t+1)*10009+int(ix)*71))
        for t,ix in enumerate(ix1)])
    test=(np.arange(len(XALL)) if cond=="novel_parity4"
          else np.random.default_rng(seed+998800).choice(
              len(XALL),size=512,replace=True))
    results={}
    for method in METHODS:
        results[method]={}
        for t in CHECKPOINTS:
            if t==0 or method.endswith("replay"):
                x=np.r_[ix0,ix1[:t]]
                y=np.concatenate((y0,y1[:t]),axis=0)
                weights=np.r_[np.full(PRE,1 if t==0 else 0.20),
                              np.ones(t)]
            else:
                x,y=ix1[:t],y1[:t]
                weights=np.ones(t)
            probs=np.empty((len(test),OUT))
            for j in range(OUT):
                labels=y[:,j]
                if len(np.unique(labels))<2:
                    probs[:,j]=float(np.average(labels,weights=weights))
                else:
                    model=LogisticRegression(
                        C=1.0,solver="lbfgs",max_iter=200)
                    model.fit(XALL[x],labels,sample_weight=weights)
                    probs[:,j]=model.predict_proba(XALL[test])[:,1]
            results[method][t]=evaluation(probs,new,changed,test)
    return results
def main():
    start=time.perf_counter()
    data={"meta":{"trials":N,"C":1.0,"replay_pre_weight":0.2,
                  "features":"raw 10 binary inputs",
                  "query_policy":"matched random observations"},
          "conditions":{}}
    for cond in ("in_class","dense_threshold","novel_parity4"):
        rows=[run(SEED0+i,cond) for i in range(N)]
        data["conditions"][cond]={
            method:{str(t):{k:statistics.mean(
                r[method][t][k] for r in rows)
                for k in ("all","changed","retained")}
                for t in CHECKPOINTS}
            for method in METHODS}
        print("done",cond,round(time.perf_counter()-start,1),flush=True)
    data["runtime_seconds"]=time.perf_counter()-start
    p="/home/dtdav/dev/learn/research/cycle02_logistic_results.json"
    with open(p,"w") as f:json.dump(data,f,indent=2)
    for cond,r in data["conditions"].items():
        print(cond)
        for name,record in r.items():
            print(name,[round(record[str(t)]["all"],5)
                        for t in CHECKPOINTS],
                  "changed48",round(record["48"]["changed"],5))
    print("saved:",p)
if __name__=="__main__":
    main()
