"""Cycle 03: Bayes with candidate class expanded to fourth-order parity.

The learner does NOT receive the identity or location of changed rules.
Its richer grammar is prespecified, so success is not autonomous discovery.
"""
import itertools, json, math, random, statistics, time
import numpy as np
from cycle02_unknown_structure import (
    D, OUT, CHANGED, NOISE, PRE, POST, TRIALS, CHECKPOINTS,
    SEED0, XALL, GRAMMAR, PRIOR, ENTROPY_NOISE,
    observe, rule_world
)
EPS=1e-10
FOURTH=np.stack([
    np.bitwise_xor.reduce(XALL[:,c].astype(np.uint8),axis=1)
    for c in itertools.combinations(range(D),4)
],axis=1)
G4=np.concatenate((GRAMMAR,FOURTH),axis=1).astype(np.float32)
P4=np.r_[0.90*PRIOR, np.full(FOURTH.shape[1],0.10/FOURTH.shape[1])]
assert abs(P4.sum()-1)<1e-12
def update(p,x,y):
    likelihood=np.where(G4[x][None,:]==y[:,None],1-NOISE,NOISE)
    w=p*likelihood
    return w/w.sum(axis=1,keepdims=True)
def predict(p,indices):
    return NOISE+(1-2*NOISE)*(p@G4[indices].T)
def score(p,world,changed):
    ids=np.arange(len(XALL))
    pred=np.clip(predict(p,ids),EPS,1-EPS)
    q=NOISE+(1-2*NOISE)*world[:,ids]
    loss=(-(q*np.log(pred)+(1-q)*np.log1p(-pred)))
    excess=loss.mean(axis=1)-ENTROPY_NOISE
    return {"all":float(excess.mean()),
            "changed":float(excess[changed].mean()),
            "retained":float(np.delete(excess,changed).mean())}
def choose(p,method,rng):
    pool=rng.integers(0,len(XALL),size=64)
    if method.endswith("random"):
        return int(pool[0])
    pred=np.clip(predict(p,pool),EPS,1-EPS)
    entropy=-(pred*np.log(pred)+(1-pred)*np.log1p(-pred))
    return int(pool[entropy.sum(axis=0).argmax()])
def trial(seed,condition):
    py=random.Random(seed)
    rng=np.random.default_rng(seed)
    old,new,changed=rule_world(py,condition)
    old_indices=rng.integers(0,len(XALL),size=PRE)
    old_y=np.stack([observe(old,int(ix),rng) for ix in old_indices])
    base=np.tile(P4,(OUT,1))
    for ix,y in zip(old_indices,old_y):
        base=update(base,int(ix),y)
    result={}
    for i,method in enumerate(("expanded_active","expanded_random")):
        # Before the change, all output rules have been learned.
        # At the known shift time, each output has independent change
        # prior probability CHANGED/OUT; no change identities are supplied.
        prior=(1-CHANGED/OUT)*base+(CHANGED/OUT)*P4[None,:]
        p=prior.copy()
        policy_rng=np.random.default_rng(seed+31337+227*i)
        hist={}
        for t in range(POST+1):
            if t in CHECKPOINTS:
                hist[t]=score(p,new,changed)
            if t==POST:
                break
            ix=choose(p,method,policy_rng)
            y=observe(new,ix,np.random.default_rng(
                seed*1000003+(t+1)*10009+ix*71))
            p=update(p,ix,y)
        result[method]=hist
    return result
def main():
    timer=time.perf_counter()
    outcomes={"meta":{"trials":TRIALS,"pre":PRE,"post":POST,
              "old_grammar":int(GRAMMAR.shape[1]),
              "added_fourth_parity":int(FOURTH.shape[1]),
              "new_grammar":int(G4.shape[1]),
              "fourth_order_prior_mass":0.10,
              "shift_time_known":True},
              "conditions":{}}
    for cond in ("in_class","novel_parity4","dense_threshold"):
        rows=[trial(SEED0+i,cond) for i in range(TRIALS)]
        outcomes["conditions"][cond]={
            name:{str(t):{key:statistics.mean(
                row[name][t][key] for row in rows)
                for key in ("all","changed","retained")}
                for t in CHECKPOINTS}
            for name in ("expanded_active","expanded_random")}
    outcomes["time_seconds"]=time.perf_counter()-timer
    path="/home/dtdav/dev/learn/research/cycle03_extended_results.json"
    with open(path,"w") as f:json.dump(outcomes,f,indent=2)
    print("Saved:",path,"runtime:",round(outcomes["time_seconds"],2))
    for cond,methods in outcomes["conditions"].items():
        print("CONDITION",cond)
        for name,row in methods.items():
            print(name,[round(row[str(t)]["all"],5)
                         for t in CHECKPOINTS],
                  "changed",round(row["48"]["changed"],5),
                  "retained",round(row["48"]["retained"],5))
if __name__=="__main__":
    main()
