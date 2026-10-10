"""Cycle 20: independent confirmation of predictive-loss versus policy-regret gap.

Predeclared and frozen models from Cycles 13, 15, 16.
120 NEW seeds per condition, 8192 post-change randomized-action labels.
No tuning on test data; all methods get same training observations.
Methods: trained two-state HMM, task-targeted S learner, R-style
transformed-outcome Lasso, T meta-learner. These are established methods.
"""
import json,math,statistics,time
import numpy as np
from cycle12_partial_obs import ROOT,N_PRE,make_task,sample,selected_channel,log_loss
from cycle15_decision import train,predict
from cycle16_treatment_effect import train_predict
from cycle13_oracle import prob_oracle

NUI=7
NPOST=8192
NTEST=3500
NSEEDS=120
SEED0=74111026
REGIMES=("stationary","nuisance_shift","contextual_action",
         "target_rate_shift")
METHODS=("two_state_hmm","target_action_features",
         "direct_lasso","t_learner")

def eval_one(seed,regime):
    task=make_task(seed,NUI)
    k=task["k"]
    pre,py=sample(task,seed+1,N_PRE,"stationary")
    index=selected_channel(pre,py,k)
    if index!=task["target_channel"]:
        raise ValueError("Wrong target feature selected")
    post,qy=sample(task,seed+883,NPOST,regime)
    test,y=sample(task,seed+991,NTEST,regime)
    x=np.r_[pre,post]
    labels=np.r_[py,qy]
    weight=np.r_[np.full(N_PRE,.1),np.ones(NPOST)]
    x0=test.copy();x0[:,-1]=0
    x1=test.copy();x1[:,-1]=1
    p0_true=prob_oracle(task,x0,regime)
    p1_true=prob_oracle(task,x1,regime)
    oracle_best=np.maximum(p0_true,p1_true)
    reference=float(np.mean(oracle_best-(p0_true+p1_true)/2))
    outputs={"random_reference":reference}
    for name in METHODS:
        clock=time.perf_counter()
        if name in ("two_state_hmm","target_action_features"):
            model=train(name,x,labels,weight,k,index)
            pred=predict(name,model,test,k,index)
            cf0=predict(name,model,x0,k,index)
            cf1=predict(name,model,x1,k,index)
            outcomes={"loss":log_loss(pred,y)}
        else:
            cf0=None;cf1=None
            # Contrast-only meta learners do not model baseline outcomes.
            effect=train_predict(name,x,labels,weight,test,k,index)
            outcomes={}
        if cf0 is not None:
            action=(cf1>cf0).astype(np.uint8)
        else:
            action=(effect>0).astype(np.uint8)
        outcomes["regret"]=float(np.mean(
            oracle_best-np.where(action==1,p1_true,p0_true)))
        outcomes["fit_seconds"]=time.perf_counter()-clock
        outputs[name]=outcomes
    return outputs

def avg_stats(xs):
    return {"mean":statistics.mean(xs),
            "SE":statistics.stdev(xs)/math.sqrt(len(xs))}

def main():
    started=time.perf_counter()
    out={"config":{"new_seeds":NSEEDS,"seed_start":SEED0,
                   "post":NPOST,"pre":N_PRE,"test":NTEST,
                   "nui":NUI,"same_observations":True,
                   "hyperparameters_fixed_from_prior_cycles":True,
                   "no_test_hyperparam_selection":True},
         "conditions":{}}
    for regime in REGIMES:
        rows=[]
        for i in range(NSEEDS):
            rows.append(eval_one(SEED0+911*i,regime))
            if (i+1)%40==0:
                print(regime,i+1,"/",NSEEDS,
                      "elapsed",round(time.perf_counter()-started,1),
                      flush=True)
        stats={"random_reference":avg_stats(
            [r["random_reference"] for r in rows])}
        for method in METHODS:
            stats[method]={
                key:avg_stats([r[method][key] for r in rows])
                for key in rows[0][method]}
        for a,b in (("target_action_features","two_state_hmm"),
                    ("target_action_features","direct_lasso"),
                    ("target_action_features","t_learner")):
            dif=[r[a]["regret"]-r[b]["regret"] for r in rows]
            stats[a+"_minus_"+b]=avg_stats(dif)
        out["conditions"][regime]=stats
        print("COND",regime,"random",
              round(stats["random_reference"]["mean"],5),
              "target",round(stats["target_action_features"]["regret"]["mean"],5),
              "hmm",round(stats["two_state_hmm"]["regret"]["mean"],5),
              "lasso",round(stats["direct_lasso"]["regret"]["mean"],5),
              flush=True)
    out["seconds"]=time.perf_counter()-started
    path="/home/dtdav/dev/learn/research/cycle20_independent_replication.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("SAVED",path,"seconds",round(out["seconds"],1))
if __name__=="__main__":
    main()
