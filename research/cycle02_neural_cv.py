"""Cycle 02 neural check: tune PRETRAIN epochs only by held-out prechange data.

No evaluation-set tuning. Post-change updates and replay weights fixed.
This is a modest neural baseline, not a state-of-the-art neural method.
"""
import json,random,statistics,time,copy
import numpy as np
import torch
from torch import nn
from cycle02_unknown_structure import (
    NOISE,PRE,POST,OUT,CHECKPOINTS,SEED0,XALL,observe,rule_world)
from cycle02_neural import new_model,fit_steps,score_nn,DEVICE
N=24
PRE_BLOCK=30
PRE_MAX=240
POST_STEPS=60
LR=0.002
WD=0.1
MODELS=("shared_replay","modular_replay")

def net_seed(seed,mode):
    torch.manual_seed(seed*13+123)
    if DEVICE.type=="cuda":torch.cuda.manual_seed_all(seed*13+123)
    return new_model(mode)
def make_tensors(index,labels):
    return (torch.from_numpy(XALL[index]).to(DEVICE),
            torch.from_numpy(labels.astype(np.float32)).to(DEVICE))
def validation_epoch(seed,mode,x,y):
    rg=np.random.default_rng(seed+770007)
    perm=rg.permutation(len(x))
    train,valid=perm[24:],perm[:24]
    tr_x,tr_y=x[train],y[train]
    v_x,v_y=x[valid],y[valid]
    model=net_seed(seed,mode)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    step_rng=np.random.default_rng(seed+77991)
    best,ep=1e100,PRE_BLOCK
    for n in range(PRE_BLOCK,PRE_MAX+1,PRE_BLOCK):
        fit_steps(model,optimizer,tr_x,tr_y,None,None,
                  PRE_BLOCK,False,step_rng)
        model.eval()
        with torch.no_grad():
            loss=nn.functional.binary_cross_entropy_with_logits(
                model(v_x),v_y).item()
        if loss<best:
            best,ep=loss,n
    return ep
def run(seed,condition):
    rng=np.random.default_rng(seed)
    old,new,changed=rule_world(random.Random(seed),condition)
    pre_ids=rng.integers(0,len(XALL),size=PRE)
    y_pre=np.array([observe(old,int(i),rng) for i in pre_ids])
    test=(np.arange(len(XALL)) if condition=="novel_parity4"
          else np.random.default_rng(seed+998800).choice(
              len(XALL),size=512,replace=True))
    post_ids=np.random.default_rng(seed+221133).integers(
        0,len(XALL),size=POST)
    y_post=np.array([observe(new,int(i),np.random.default_rng(
        seed*1000003+(t+1)*10009+int(i)*71))
        for t,i in enumerate(post_ids)])
    old_x,old_y=make_tensors(pre_ids,y_pre)
    post_x,post_y=make_tensors(post_ids,y_post)
    histories={}
    epochs={}
    for mode in MODELS:
        n=validation_epoch(seed,mode,old_x,old_y)
        epochs[mode]=n
        model=net_seed(seed,mode)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        step_rng=np.random.default_rng(seed+77991)
        fit_steps(model,opt,old_x,old_y,None,None,n,False,step_rng)
        records={0:score_nn(model,new,changed,test)}
        for prev,t in zip(CHECKPOINTS[:-1],CHECKPOINTS[1:]):
            fit_steps(model,opt,old_x,old_y,
                      post_x[:t],post_y[:t],POST_STEPS,True,step_rng)
            records[t]=score_nn(model,new,changed,test)
        histories[mode]=records
    return histories,epochs
def main():
    t0=time.perf_counter()
    output={"meta":{"N":N,"pre_block":PRE_BLOCK,"pre_max":PRE_MAX,
                   "post_steps_per_checkpoint":POST_STEPS,
                   "lr":LR,"weight_decay":WD,
                   "device":str(DEVICE),
                   "pretrain_epochs_selected_on_prechange_validation":True,
                   "no_eval_tuning":True},"conditions":{}}
    for condition in ("in_class","dense_threshold","novel_parity4"):
        cases=[]
        for i in range(N):
            cases.append(run(SEED0+i,condition))
            if (i+1)%8==0:
                print(condition,i+1,"cases",round(time.perf_counter()-t0,1),
                      "seconds",flush=True)
        report={}
        for mode in MODELS:
            report[mode]={
                str(t):{metric:statistics.mean(
                    res[0][mode][t][metric] for res in cases)
                    for metric in ("all","changed","retained")}
                for t in CHECKPOINTS}
            report[mode]["mean_selected_pre_epochs"]=statistics.mean(
                case[1][mode] for case in cases)
        output["conditions"][condition]=report
    output["runtime_seconds"]=time.perf_counter()-t0
    path="/home/dtdav/dev/learn/research/cycle02_neural_cv_results.json"
    with open(path,"w") as f:json.dump(output,f,indent=2)
    print("saved",path,"runtime",output["runtime_seconds"])
    for cond,models in output["conditions"].items():
        print("CONDITION:",cond)
        for name,val in models.items():
            print(name,[round(val[str(t)]["all"],5) for t in CHECKPOINTS],
                  "epochs",val["mean_selected_pre_epochs"])
if __name__=="__main__":
    main()
