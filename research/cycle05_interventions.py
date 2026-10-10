"""Cycle 05: learn an unenumerated parity mask with noisy membership queries.

Predeclared question: at equal query counts, how much computation is saved
by algebraically targeted interventions relative to exhaustive maximum
likelihood over known k-sparse parity masks? How brittle is the strategy
when the target is not a parity? This is classical parity learning,
NOT a new algorithm.
"""
import itertools
import json
import math
import os
import random
import time
import statistics
import numpy as np

D=32
K=4
EPS=0.10
REPEATS=9
BUDGET=(D+1)*REPEATS
TRIALS=48
ROOT=20261010
HOLDOUT=4096
EXHAUSTIVE_BATCH=4096
MODES=("sparse_parity4","dense_parity12","majority11","conjunction5")
SEED_START=998122
ALL_MASKS=np.array(
    [sum(1<<i for i in c) for c in itertools.combinations(range(D),K)],
    dtype=np.uint64
)
assert len(ALL_MASKS)==math.comb(D,K)

def bit_parity(x):
    return (np.bitwise_count(np.asarray(x,dtype=np.uint64))&1).astype(np.uint8)

def world(rng,mode):
    coords=rng.choice(D,size={"sparse_parity4":4,"dense_parity12":12,
                              "majority11":11,"conjunction5":5}[mode],
                      replace=False)
    mask=np.uint64(sum(1<<int(i) for i in coords))
    offset=int(rng.integers(0,2))
    if mode.endswith("parity4") or mode=="dense_parity12":
        fun=lambda x: bit_parity(x&mask)^offset
    elif mode=="majority11":
        fun=lambda x: (np.bitwise_count(x&mask)>=6).astype(np.uint8)
    else:
        fun=lambda x: ((x&mask)==mask).astype(np.uint8)
    return fun,mask,offset

def draw_noise(query_x,world_fun,rng):
    x=np.asarray(query_x,dtype=np.uint64)
    return world_fun(x)^(rng.random(len(x))<EPS).astype(np.uint8)

def active_basis_predictor(fun,rng):
    # Every bit query e_i and all-zero query is repeated REPEATS times.
    # The target mask is NOT enumerated or specified.
    x=np.repeat(np.array([0]+[1<<i for i in range(D)],
                         dtype=np.uint64),REPEATS)
    y=draw_noise(x,fun,rng).reshape(D+1,REPEATS)
    majority=(np.sum(y,axis=1)>REPEATS//2).astype(np.uint8)
    intercept=int(majority[0])
    support=majority[1:]^intercept
    mask=np.uint64(sum(1<<i for i,v in enumerate(support) if v))
    return mask,intercept,x,y.ravel()

def passive_sparse_ml(x,y):
    # Conventional exhaustive ML over exactly K selected input bits and
    # a free intercept. Model knows K=4, not which bits.
    best_error=len(x)+1
    best_mask=0
    best_intercept=0
    for start in range(0,len(ALL_MASKS),EXHAUSTIVE_BATCH):
        masks=ALL_MASKS[start:start+EXHAUSTIVE_BATCH]
        parity=(np.bitwise_count(masks[:,None]&x[None,:])&1)
        errors=np.count_nonzero(parity != y[None,:],axis=1)
        choice=np.minimum(errors,len(y)-errors)
        idx=int(np.argmin(choice))
        if choice[idx]<best_error:
            best_error=int(choice[idx])
            best_mask=int(masks[idx])
            best_intercept=int(errors[idx]>(len(y)-errors[idx]))
    return np.uint64(best_mask),best_intercept

def score(mask,intercept,fun,test_x):
    pred=bit_parity(test_x&mask)^intercept
    real=fun(test_x)
    # Excluding irreducible Bernoulli noise, 10% error adds
    # (1-2 EPS) log((1-EPS)/EPS) nats.
    mismatch=float(np.mean(pred!=real))
    excess=mismatch*(1-2*EPS)*math.log((1-EPS)/EPS)
    return mismatch,excess

def trial(seed,mode):
    rng=np.random.default_rng(seed)
    f,truth_mask,truth_intercept=world(rng,mode)
    # Independent oracle noise for each method; same query budget.
    test_x=np.random.default_rng(seed+33444).integers(
        0,1<<D,size=HOLDOUT,dtype=np.uint64)
    r_active=np.random.default_rng(seed+991)
    start=time.perf_counter_ns()
    amask,ai,ax,ay=active_basis_predictor(f,r_active)
    active_time=(time.perf_counter_ns()-start)/1e9
    uniform_rng=np.random.default_rng(seed+773)
    px=uniform_rng.integers(0,1<<D,size=BUDGET,dtype=np.uint64)
    py=draw_noise(px,f,uniform_rng)
    start=time.perf_counter_ns()
    pmask,pi=passive_sparse_ml(px,py)
    passive_time=(time.perf_counter_ns()-start)/1e9
    ae,al=score(amask,ai,f,test_x)
    pe,pl=score(pmask,pi,f,test_x)
    return {"active_error":ae,"passive_error":pe,
            "active_excess_nats":al,"passive_excess_nats":pl,
            "active_ms":1000*active_time,"passive_ms":1000*passive_time,
            "active_exact":bool(amask==truth_mask and ai==truth_intercept)
                           if "parity" in mode else None,
            "passive_exact":bool(pmask==truth_mask and pi==truth_intercept)
                           if mode=="sparse_parity4" else None}

def summary(rows):
    d={}
    for k in rows[0]:
        x=[r[k] for r in rows if r[k] is not None]
        if not x:continue
        d[k]=dict(mean=statistics.mean(map(float,x)),
                  SE=statistics.stdev(map(float,x))/math.sqrt(len(x))
                     if len(x)>1 else 0)
    return d

def main():
    stamp=time.perf_counter()
    output={"config":{"dimensions":D,"known_sparse_degree_ml":K,
                      "flipped_noise":EPS,"repeats":REPEATS,
                      "query_budget":BUDGET,"heldout_uniform":HOLDOUT,
                      "trials_per_mode":TRIALS,
                      "enumerated_sparse_masks":len(ALL_MASKS),
                      "oracle_queries_equal":True,
                      "active_is_classical_membership_query_learning":True},
            "modes":{}}
    for mode in MODES:
        rows=[]
        for i in range(TRIALS):
            rows.append(trial(SEED_START+77*i,mode))
        output["modes"][mode]=summary(rows)
        print("DONE",mode,"elapsed",round(time.perf_counter()-stamp,1),
              "error(active/passive)",
              round(output["modes"][mode]["active_error"]["mean"],4),
              round(output["modes"][mode]["passive_error"]["mean"],4),
              flush=True)
    output["total_seconds"]=time.perf_counter()-stamp
    out="/home/dtdav/dev/learn/research/cycle05_results.json"
    with open(out,"w") as f:json.dump(output,f,indent=2)
    print("SAVED",out)
    print("TIME",round(output["total_seconds"],2))
    for mode,rec in output["modes"].items():
        print(mode,"active_err",rec["active_error"],
              "passive_err",rec["passive_error"],
              "speedup passive/active",
              rec["passive_ms"]["mean"]/rec["active_ms"]["mean"])
if __name__=="__main__":
    main()
