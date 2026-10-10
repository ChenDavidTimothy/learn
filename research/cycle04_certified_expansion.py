"""Cycle 04: certificate-triggered model expansion under adaptive-safe tests.

Evidence e_t = alternative-mixture likelihood / maximum null likelihood.
For every fixed null rule h*, e_t <= Q(data)/P_h*(data), a nonnegative
likelihood-ratio martingale even for predictable adaptive inputs.
Therefore P_h*(ever e_t >= 1/alpha) <= alpha by Ville's inequality.
Six outputs use Bonferroni alpha_j=0.05/6.
Based on universal inference; not a novel testing principle.
"""
import json, math, random, statistics, time
import numpy as np
from scipy.special import logsumexp
from cycle02_unknown_structure import (
    D, OUT, CHANGED, NOISE, PRE, POST, TRIALS, CHECKPOINTS, SEED0,
    XALL, GRAMMAR, PRIOR, ENTROPY_NOISE, rule_world, observe,
    posterior_update
)
from cycle03_extended_grammar import FOURTH, G4, P4
EPS=1e-9
ALPHA=0.05
THRESHOLD=math.log(OUT/ALPHA)
METHODS=("restricted","always_expanded","certificate_expand")

def model_predictions(base,extended,flags):
    p=np.empty((OUT,len(XALL)))
    for j in range(OUT):
        if flags[j]:
            p[j] = extended[j]@G4.T
        else:
            p[j] = base[j]@GRAMMAR.T
    return NOISE+(1-2*NOISE)*p

def score(base,ext,flags,truth,changed):
    pred=np.clip(model_predictions(base,ext,flags),EPS,1-EPS)
    q=NOISE+(1-2*NOISE)*truth
    by_output=(-(q*np.log(pred)+(1-q)*np.log1p(-pred))
               .mean(axis=1)-ENTROPY_NOISE)
    return {"all":float(by_output.mean()),
            "changed":float(by_output[changed].mean()),
            "retained":float(np.delete(by_output,changed).mean())}

def normalize(weights):
    return weights/weights.sum(axis=1,keepdims=True)

def update_extended(weights,x,labels):
    factor=np.where(G4[x][None,:]==labels[:,None],1-NOISE,NOISE)
    return normalize(weights*factor)

def replay_extended(prior,post_inputs,post_outputs,j):
    weights=prior.copy()
    for x,y in zip(post_inputs,post_outputs):
        factor=np.where(G4[x]==y[j],1-NOISE,NOISE)
        weights=weights*factor
        weights/=weights.sum()
    return weights

def execute(seed,condition):
    py=random.Random(seed)
    rng=np.random.default_rng(seed)
    old,new,changed=rule_world(py,condition)
    pre_inputs=rng.integers(0,len(XALL),size=PRE)
    pre_labels=[observe(old,int(x),rng) for x in pre_inputs]
    base_old=np.tile(PRIOR,(OUT,1))
    for x,y in zip(pre_inputs,pre_labels):
        base_old=posterior_update(base_old,int(x),y)
    q=CHANGED/OUT
    prior_null=(1-q)*base_old+q*PRIOR[None,:]
    prior_extended=np.concatenate((
        (1-q)*base_old+q*0.90*PRIOR[None,:],
        np.tile(q*0.10/FOURTH.shape[1],(OUT,FOURTH.shape[1]))
    ),axis=1)
    # Same experiment observations for every method, no query advantage.
    post_inputs=np.random.default_rng(seed+221133).integers(
        0,len(XALL),size=POST)
    post_outputs=[observe(new,int(x),np.random.default_rng(
        seed*1000003+(t+1)*10009+int(x)*71))
        for t,x in enumerate(post_inputs)]
    post_outputs=np.array(post_outputs)
    low=prior_null.copy()
    full=prior_extended.copy()
    detect_low=prior_null.copy()
    detect_full=prior_extended.copy()
    flags=np.zeros(OUT,dtype=bool)
    # Relative log likelihood of each fixed candidate from shift onset.
    log_null=np.zeros((OUT,GRAMMAR.shape[1]))
    log_alt=np.zeros((OUT,FOURTH.shape[1]))
    first_hit=np.full(OUT,-1,dtype=int)
    records={}
    counts={}
    for t in range(POST+1):
        if t in CHECKPOINTS:
            records[t]={
                "restricted":score(low,full,np.zeros(OUT,dtype=bool),new,changed),
                "always_expanded":score(low,full,np.ones(OUT,dtype=bool),new,changed),
                "certificate_expand":score(detect_low,detect_full,flags,new,changed)
            }
            counts[t]={"flags":int(flags.sum()),
                       "true_hits":int(flags[changed].sum()),
                       "false_hits":int(flags.sum()-flags[changed].sum())}
        if t==POST: break
        x=int(post_inputs[t]); y=post_outputs[t]
        low=posterior_update(low,x,y)
        full=update_extended(full,x,y)
        detect_low=posterior_update(detect_low,x,y)
        if flags.any():
            detect_full[flags]=update_extended(detect_full[flags],x,y[flags])
        null_l=np.where(GRAMMAR[x][None,:]==y[:,None],1-NOISE,NOISE)
        alt_l=np.where(FOURTH[x][None,:]==y[:,None],1-NOISE,NOISE)
        log_null+=np.log(null_l)
        log_alt+=np.log(alt_l)
        log_e=logsumexp(log_alt,axis=1)-math.log(FOURTH.shape[1])-log_null.max(axis=1)
        for j in np.flatnonzero((~flags)&(log_e>=THRESHOLD)):
            flags[j]=True
            first_hit[j]=t+1
            detect_full[j]=replay_extended(
                prior_extended[j],post_inputs[:t+1],post_outputs[:t+1],j)
    return records,counts,first_hit,changed

def main():
    before=time.perf_counter()
    data={"meta":{"pre":PRE,"post":POST,"trials":TRIALS,
                  "alpha_familywise":ALPHA,"log_threshold":THRESHOLD,
                  "null_candidates":int(GRAMMAR.shape[1]),
                  "alternate_candidates":int(FOURTH.shape[1]),
                  "all_methods_see_same_post_data":True,
                  "all_post_inputs_random":True,
                  "change_onset_known":True,
                  "change_locations_unknown":True},
          "conditions":{}}
    for cond in ("in_class","novel_parity4","dense_threshold"):
        rows=[execute(SEED0+i,cond) for i in range(TRIALS)]
        out={}
        for name in METHODS:
            out[name]={str(t):{key:statistics.mean(
                row[0][t][name][key] for row in rows)
                for key in ("all","changed","retained")}
                for t in CHECKPOINTS}
        observed={
            "mean_true_detected":statistics.mean(
                sum(t>=0 for t in row[2][row[3]]) for row in rows),
            "mean_false_detected":statistics.mean(
                sum(t>=0 for i,t in enumerate(row[2])
                    if i not in row[3]) for row in rows),
            "trials_with_any_false_hit":sum(
                any(t>=0 for i,t in enumerate(row[2])
                    if i not in row[3]) for row in rows),
            "mean_detection_time_over_detected_changed":None,
        }
        times=[int(row[2][j]) for row in rows for j in row[3]
               if row[2][j]>=0]
        if times:observed["mean_detection_time_over_detected_changed"]=statistics.mean(times)
        data["conditions"][cond]={"methods":out,"detections":observed}
    data["time_seconds"]=time.perf_counter()-before
    path="/home/dtdav/dev/learn/research/cycle04_results.json"
    with open(path,"w") as f:json.dump(data,f,indent=2,default=lambda a:a.item())
    print("saved:",path,"duration",round(data["time_seconds"],2))
    for cond,c in data["conditions"].items():
        print("CONDITION",cond)
        for name,r in c["methods"].items():
            print(name,[round(r[str(t)]["all"],6) for t in CHECKPOINTS],
                  "changed48:",round(r["48"]["changed"],6))
        print("detections:",c["detections"])
if __name__=="__main__":
    main()
