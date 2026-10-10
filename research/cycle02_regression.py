"""Cycle 02 strong linear and polynomial controls on matched random queries."""
import json, random, statistics, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from cycle02_unknown_structure import (
    D, OUT, NOISE, PRE, POST, CHECKPOINTS, SEED0, XALL,
    ENTROPY_NOISE, rule_world, observe
)
FEATURES = {
    "linear": XALL.astype(float),
    "poly3": PolynomialFeatures(
        degree=3, interaction_only=True, include_bias=False
    ).fit_transform(XALL).astype(float)
}
METHODS = ("linear_replay","linear_recent","poly3_replay","poly3_recent")
N = 24
def score(pred, new, changed, test):
    p = np.clip(pred.T, 1e-5, 1-1e-5)
    q = NOISE+(1-2*NOISE)*new[:,test]
    each = (-(q*np.log(p)+(1-q)*np.log1p(-p))
            .mean(axis=1)-ENTROPY_NOISE)
    return {"all":float(each.mean()),
            "changed":float(each[changed].mean()),
            "retained":float(np.delete(each,changed).mean())}
def one(seed, condition):
    py = random.Random(seed)
    rg = np.random.default_rng(seed)
    old,new,changed = rule_world(py,condition)
    pre_x = rg.integers(0,len(XALL),size=PRE)
    pre_y = np.array([observe(old,int(i),rg) for i in pre_x])
    post_x = np.random.default_rng(seed+221133).integers(
        0,len(XALL),size=POST)
    post_y = np.array([observe(new,int(ix),np.random.default_rng(
        seed*1000003+(t+1)*10009+int(ix)*71))
        for t,ix in enumerate(post_x)])
    test = (np.arange(len(XALL)) if condition=='novel_parity4'
            else np.random.default_rng(seed+998800).choice(
                len(XALL),size=512,replace=True))
    result={}
    for name in METHODS:
        phi = FEATURES[name.split("_")[0]]
        result[name]={}
        for t in CHECKPOINTS:
            if t==0 or name.endswith("replay"):
                ids=np.r_[pre_x,post_x[:t]]
                labels=np.r_[pre_y,post_y[:t]]
                weights=np.r_[np.full(PRE,1 if t==0 else 0.2),
                              np.full(t,1.0)]
            else:
                ids,labels=post_x[:t],post_y[:t]
                weights=np.ones(t)
            model=Ridge(alpha=2.0)
            model.fit(phi[ids],labels,sample_weight=weights)
            result[name][t]=score(model.predict(phi[test]),new,changed,test)
    return result
def main():
    started=time.perf_counter()
    out={"metadata":{"trials":N,"alpha":2,"old_weight":0.2,
                     "features":{k:v.shape[1] for k,v in FEATURES.items()},
                     "policy":"matched random queries"},
         "conditions":{}}
    for condition in ("in_class","dense_threshold","novel_parity4"):
        print("starting",condition,flush=True)
        runs=[one(SEED0+i,condition) for i in range(N)]
        out["conditions"][condition]={
            name:{str(t):{key:statistics.mean(
                r[name][t][key] for r in runs)
                for key in ("all","changed","retained")}
                for t in CHECKPOINTS}
            for name in METHODS}
    out["time_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle02_regression_results.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("saved:",path,"elapsed:",round(out["time_seconds"],2))
    for condition,methods in out["conditions"].items():
        print("CONDITION",condition)
        for name,r in methods.items():
            print(name,[round(r[str(t)]["all"],5) for t in CHECKPOINTS],
                  "changed",round(r["48"]["changed"],5))
if __name__=="__main__":
    main()
