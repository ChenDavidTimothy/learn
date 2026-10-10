"""Cycle 18: unknown-onset representation selection by decision value.

Streams contain independent POMDP three-step episodes with randomized
actions; they are NOT single unbroken trajectories. One unannounced
change occurs after a randomly chosen episode. Algorithms only see
time-ordered observed action/history/outcome tuples.

Candidates: learned two-state HMM and sparse target-action interaction
model. Two gates choose via held-out log loss or doubly robust policy
value (known exploration propensity 0.5). DR selection is classic
off-policy evaluation, not an original algorithm.
"""
import json,math,statistics,time
import numpy as np
from cycle12_partial_obs import ROOT,N_PRE,make_task,sample,selected_channel
from cycle15_decision import train,predict
from cycle13_oracle import prob_oracle

NUI=7
REPS=48
STREAM=4096
WINDOW=2048
CHECKPOINTS=(0,512,1024,1536,2048,2560,3072,3584,4096)
REGIMES=("stationary","nuisance_shift","contextual_action",
         "target_rate_shift")
CANDIDATES=("two_state_hmm","target_action_features")

def candidate_probs(method,model,x,k,j):
    x0=x.copy();x0[:,-1]=0
    x1=x.copy();x1[:,-1]=1
    p0=np.clip(predict(method,model,x0,k,j),.002,.998)
    p1=np.clip(predict(method,model,x1,k,j),.002,.998)
    return p0,p1

def expected_regret(choice,oracle):
    p0,p1=oracle
    return float(np.mean(
        np.maximum(p0,p1)-np.where(choice==1,p1,p0)))

def holdout_select(xtrain,ytrain,xval,yval,k,j,seed):
    models={method:train(method,xtrain,ytrain,np.ones(len(xtrain)),
                         k,j)
            for method in CANDIDATES}
    estimate={m:candidate_probs(m,model,xval,k,j)
              for m,model in models.items()}
    a=xval[:,-1].astype(np.uint8)
    outcome=yval.astype(float)
    lls={}
    for m,(p0,p1) in estimate.items():
        pred=np.where(a==1,p1,p0)
        lls[m]=float(np.mean(-outcome*np.log(pred)
                             -(1-outcome)*np.log1p(-pred)))
    # Model-averaged fixed nuisance reference for independent validation.
    m0=np.mean([vals[0] for vals in estimate.values()],axis=0)
    m1=np.mean([vals[1] for vals in estimate.values()],axis=0)
    mr=np.where(a==1,m1,m0)
    values={}
    for m,(p0,p1) in estimate.items():
        chosen=(p1>p0).astype(np.uint8)
        mp=np.where(chosen==1,m1,m0)
        values[m]=float(np.mean(
            mp+2*(a==chosen)*(outcome-mr)))
    pi_h=(estimate["two_state_hmm"][1]>
          estimate["two_state_hmm"][0]).astype(np.uint8)
    pi_i=(estimate["target_action_features"][1]>
          estimate["target_action_features"][0]).astype(np.uint8)
    v_h=np.where(pi_h==1,m1,m0)
    v_i=np.where(pi_i==1,m1,m0)
    diff=(v_i-v_h)+2*((a==pi_i).astype(float)-
                     (a==pi_h).astype(float))*(outcome-mr)
    margin=1.645*np.std(diff,ddof=1)/np.sqrt(len(diff))
    confident=("target_action_features" if diff.mean()>margin
               else "two_state_hmm")
    return (min(lls,key=lls.get),max(values,key=values.get),
            confident,lls,values)

def perform(seed,regime):
    task=make_task(seed,NUI)
    k=task["k"]
    pre,py=sample(task,seed+1,N_PRE,"stationary")
    index=selected_channel(pre,py,k)
    assert index==task["target_channel"]
    # Randomized time of change, never passed into model-selection code.
    tchange= int(np.random.default_rng(seed+7738).integers(1200,1750))
    if regime=="stationary":tchange=STREAM+1
    first,fy=sample(task,seed+102,tchange,"stationary")
    if tchange<STREAM:
        second,sy=sample(task,seed+103,STREAM-tchange,regime)
        xs=np.concatenate((first,second))
        ys=np.r_[fy,sy]
    else:xs,ys=first,fy
    test={}
    for mode in ("stationary",regime):
        xx,_=sample(task,seed+4000,2500,mode)
        z0=xx.copy();z0[:,-1]=0
        z1=xx.copy();z1[:,-1]=1
        test[mode]=(xx,(prob_oracle(task,z0,mode),
                        prob_oracle(task,z1,mode)))
    records={}
    for t in CHECKPOINTS:
        h=np.r_[pre,xs[:t]]
        labels=np.r_[py,ys[:t]]
        x=h[-WINDOW:];y=labels[-WINDOW:]
        valid_count=max(160,min(384,len(x)//4))
        xtrain,ytrain=x[:-valid_count],y[:-valid_count]
        xval,yval=x[-valid_count:],y[-valid_count:]
        gate_loss,gate_dr,gate_conf,_,_=holdout_select(
            xtrain,ytrain,xval,yval,k,index,seed)
        models={m:train(m,x,y,np.ones(len(x)),k,index)
                for m in CANDIDATES}
        current=regime if t>tchange else "stationary"
        inputs,oracle=test[current]
        losses={}
        for m,model in models.items():
            p0,p1=candidate_probs(m,model,inputs,k,index)
            losses[m]=expected_regret((p1>p0),oracle)
        records[str(t)]={
            "time_after_change":(t-tchange if t>tchange else None),
            "mode":current,
            "regret_hmm":losses["two_state_hmm"],
            "regret_interaction":losses["target_action_features"],
            "regret_gate_logloss":losses[gate_loss],
            "regret_gate_dr":losses[gate_dr],
            "regret_gate_confident":losses[gate_conf],
            "gate_logloss_selected":gate_loss,
            "gate_dr_selected":gate_dr,
            "gate_confident_selected":gate_conf}
    return {"tchange":tchange,"regime":regime,"records":records}

def main():
    start=time.perf_counter()
    all_data={"design":{"stream":STREAM,"window":WINDOW,
               "trials":REPS,"checkpoints":CHECKPOINTS,
               "shift_unknown_to_learners":True,
               "randomized_action_probability":.5,
               "independent_short_episodes_not_continuous_POMDP":True,
               "validation_separate_from_candidate_training":True},
          "regimes":{}}
    for regime in REGIMES:
        trials=[]
        for i in range(REPS):
            trials.append(perform(ROOT+100000000+i*811,regime))
            if (i+1)%8==0:
                print(regime,i+1,"/",REPS,
                      "elapsed",round(time.perf_counter()-start,1),
                      flush=True)
        out={}
        for t in CHECKPOINTS:
            rows=[trial["records"][str(t)] for trial in trials]
            out[str(t)]={
                metric:{"mean":statistics.mean(row[metric] for row in rows),
                        "SE":statistics.stdev(row[metric] for row in rows)
                             /math.sqrt(REPS)}
                for metric in ("regret_hmm","regret_interaction",
                               "regret_gate_logloss","regret_gate_dr",
                               "regret_gate_confident")}
            for gate in ("gate_logloss_selected","gate_dr_selected",
                         "gate_confident_selected"):
                out[str(t)][gate+"_rate_interaction"]=(
                  sum(r[gate]=="target_action_features" for r in rows)/REPS)
            out[str(t)]["n_after_change"]=sum(
                r["mode"]==regime and
                r["time_after_change"] is not None for r in rows)
        all_data["regimes"][regime]={
            "aggregate":out,
            "raw_trial_records":trials}
    all_data["runtime_seconds"]=time.perf_counter()-start
    path="/home/dtdav/dev/learn/research/cycle19_confident_gate.json"
    with open(path,"w") as f:json.dump(all_data,f,indent=2)
    print("SAVED",path,"SECONDS",round(all_data["runtime_seconds"],1))
    for regime,rec in all_data["regimes"].items():
        print("REGIME",regime)
        for t in (1536,2560,3584,4096):
            row=rec["aggregate"][str(t)]
            print(t,{m:round(row[m]["mean"],5)
                     for m in ("regret_hmm","regret_interaction",
                               "regret_gate_logloss","regret_gate_dr",
                               "regret_gate_confident")},
                  "loss_choose_interaction",round(
                   row["gate_logloss_selected_rate_interaction"],2),
                  "dr_choose_interaction",round(
                   row["gate_dr_selected_rate_interaction"],2))
if __name__=="__main__":
    main()
