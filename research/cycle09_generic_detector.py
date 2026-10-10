"""Cycle 09: universal, candidate-free model inadequacy detection.

Known null: ONE fixed rule among 1335 low-degree grammar candidates.
Alternative e-process numerator Q is an iid fair-coin predictive model.
No new operator family is preloaded. Any q_t (conditional density)
could be used without invalidating null anytime-type-I control.

E_t = Q(y1:t|x1:t) / max_{h in H0} P_h(y1:t|x1:t)
For fixed true h*, E_t <= Q/P_h*, which is a nonnegative martingale
for predictable interventions under correct known observation noise.
This tests representation adequacy; it DOES NOT learn a replacement.
"""
import concurrent.futures
import json, math, random, statistics, time
import numpy as np
from cycle02_unknown_structure import (
    D, OUT, NOISE, GRAMMAR, rule_world, observe, SEED0
)
T=192
CHECKPOINTS=(16,32,48,64,96,128,192)
REPLICATIONS=150
ALPHA=.05
THRESHOLD=math.log(OUT/ALPHA)
MODES=("in_class","novel_parity4","dense_threshold")
STARTSEED=19112027
NWORKERS=6

def trial(payload):
    mode,seed=payload
    rng=random.Random(seed)
    world_rng=np.random.default_rng(seed+333)
    _,new,changed=rule_world(rng,mode)
    log_h=np.zeros((OUT,GRAMMAR.shape[1]),dtype=np.float64)
    first=np.full(OUT,-1,dtype=np.int16)
    signal={}
    for t in range(1,T+1):
        x=int(world_rng.integers(0,2**D))
        labels=observe(new,x,world_rng)
        # The whole null class is tested, not a posterior with a prior.
        lik=np.where(GRAMMAR[x][None,:]==labels[:,None],
                     1-NOISE,NOISE)
        log_h+=np.log(lik)
        stat=-t*math.log(2)-log_h.max(axis=1)
        fired=(first<0)&(stat>=THRESHOLD)
        first[fired]=t
        if t in CHECKPOINTS:
            signal[str(t)]=int(sum(first>=0))
    changed=set(map(int,changed))
    return {"true":sum(first[j]>0 for j in changed),
            "false":sum(first[j]>0 for j in range(OUT)
                        if j not in changed),
            "all_alerts":int(sum(first>0)),
            "first_changed":min([int(first[j]) for j in changed
                                 if first[j]>0],default=None),
            "last_changed":max([int(first[j]) for j in changed
                                if first[j]>0],default=None),
            "counts":signal,
            "each_detection":list(map(int,first))}

def main():
    started=time.perf_counter()
    out={"meta":{"samples_post":T,"checkpoints":CHECKPOINTS,
                 "trials_per_regime":REPLICATIONS,"seed_start":STARTSEED,
                 "alpha_familywise":ALPHA,"workers":NWORKERS,
                 "all_test_queries_random":True,
                 "does_not_require_alternative_grammar":True,
                 "no_adaptation_performed":True,
                 "null_assumes_fixed_postchange_rule":True,
                 "i.i.d._noise_known":True},
         "conditions":{}}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NWORKERS) as pool:
        for mode in MODES:
            tasks=((mode,STARTSEED+i*107) for i in range(REPLICATIONS))
            rows=list(pool.map(trial,tasks,chunksize=4))
            times=[t for r in rows for t in r["each_detection"] if t>0]
            changed_times=[r["last_changed"] for r in rows
                           if r["true"]==2]
            out["conditions"][mode]={
                "mean_true_alert_count":statistics.mean(r["true"] for r in rows),
                "mean_false_alert_count":statistics.mean(r["false"] for r in rows),
                "trials_any_alert":sum(r["all_alerts"]>0 for r in rows),
                "trials_any_false_alert":sum(r["false"]>0 for r in rows),
                "trials_both_changed_alert":sum(r["true"]==2 for r in rows),
                "mean_first_all_alert_time":(
                    statistics.mean(times) if times else None),
                "mean_time_both_detected":(
                    statistics.mean(changed_times) if changed_times else None),
                "mean_total_alerts_by_checkpoint":{
                    str(t):statistics.mean(r["counts"][str(t)] for r in rows)
                    for t in CHECKPOINTS},
            }
            print(mode,out["conditions"][mode],
                  "elapsed",round(time.perf_counter()-started,2),
                  flush=True)
    out["runtime_seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle09_generic_detector.json"
    with open(path,"w") as fp:json.dump(out,fp,indent=2,default=lambda x:x.item())
    print("SAVED:",path,"runtime:",round(out["runtime_seconds"],2))
if __name__=="__main__":
    main()
