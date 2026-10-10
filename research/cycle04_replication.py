"""Cycle 04 independent-seed replication; no tuning on replication seeds."""
import concurrent.futures
import json
import os
import statistics
import time
import numpy as np
from cycle04_certified_expansion import execute
SEEDS=240
WORKERS=6
BASE=97310621
CONDITIONS=("in_class","novel_parity4","dense_threshold")
METHODS=("restricted","always_expanded","certificate_expand")

def run(task):
    condition,seed=task
    return execute(seed,condition)

def summarize(rows):
    data={}
    for name in METHODS:
        data[name]={}
        for t in (0,8,24,48):
            for key in ("all","changed","retained"):
                vals=[r[0][t][name][key] for r in rows]
                data[name][f"{t}_{key}"]={
                    "mean":statistics.mean(vals),
                    "SE":statistics.stdev(vals)/(len(vals)**0.5)
                }
    stats=[]
    for row in rows:
        hits=row[2]
        changed=set(map(int,row[3]))
        true=[int(hits[j]) for j in changed if hits[j]>=0]
        false=[int(hits[j]) for j in range(len(hits))
               if j not in changed and hits[j]>=0]
        stats.append({"true_hits":len(true),"false_hits":len(false),
                      "times":true})
    times=[t for s in stats for t in s["times"]]
    data["detections"]={
        "trials":len(rows),
        "changed_hit_count":sum(s["true_hits"] for s in stats),
        "changed_possible":2*len(rows),
        "familywise_false_count":sum(s["false_hits"]>0 for s in stats),
        "false_output_count":sum(s["false_hits"] for s in stats),
        "detection_delay_mean":statistics.mean(times) if times else None,
        "detection_delay_median":statistics.median(times) if times else None,
        "detection_delay_p90":float(np.percentile(times,90)) if times else None,
    }
    delta=[r[0][48]["certificate_expand"]["all"] -
           r[0][48]["always_expanded"]["all"] for r in rows]
    data["cert_minus_always_final"]={
        "mean":statistics.mean(delta),
        "SE":statistics.stdev(delta)/len(delta)**0.5,
    }
    return data

def main():
    start=time.perf_counter()
    out={"meta":{"seeds_per_condition":SEEDS,
                 "seed_start":BASE,"workers":WORKERS,
                 "independent_from_initial_24_seeds":True,
                 "algorithm_parameters_unchanged":True},
         "conditions":{}}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS) as pool:
        for cond in CONDITIONS:
            tasks=((cond,BASE+i) for i in range(SEEDS))
            runs=list(pool.map(run,tasks,chunksize=4))
            out["conditions"][cond]=summarize(runs)
            print("completed",cond,round(time.perf_counter()-start,2),
                  "sec",flush=True)
    out["runtime_seconds"]=time.perf_counter()-start
    path="/home/dtdav/dev/learn/research/cycle04_replication.json"
    with open(path,"w") as f:json.dump(out,f,indent=2,default=lambda v:v.item())
    for cond,info in out["conditions"].items():
        print("\n",cond,info["detections"])
        for name in METHODS:
            vals=info[name]["48_all"]
            print(name,"final",round(vals["mean"],6),
                  "SE",round(vals["SE"],6))
    print("saved:",path)
if __name__=="__main__":
    main()
