"""Cycle 17: matched-information neural and boosted-tree action baselines.

Each learns P(next task observation | two noisy histories, randomized action)
from the same data as Cycles 15/16. Task channel is NOT supplied:
learner uses all observation channels. Target index appears only
in oracle policy regret evaluation, never neural or boosting input.
Neural early stopping uses independent weighted training validation.
"""
import copy,json,math,statistics,time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from cycle12_partial_obs import (
    ROOT,N_PRE,make_task,sample,log_loss
)
from cycle15_decision import evaluate

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
NUI=7
REPS=12
BUDGETS=(192,768,3072,8192)
TEST=4000
REGIMES=("stationary","contextual_action")
METHODS=("neural_mlp","hist_boost")

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)

def mlp_fit_predict(x,y,w,xtest,seed):
    rg=np.random.default_rng(seed+197)
    perm=rg.permutation(len(x))
    nval=max(64,int(.2*len(x)))
    train_idx=perm[nval:]
    val_idx=perm[:nval]
    x_t=torch.from_numpy((x.astype(np.float32)*2-1)).to(DEVICE)
    y_t=torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    w_t=torch.from_numpy(w.astype(np.float32)).to(DEVICE)
    torch.manual_seed(seed+491)
    if DEVICE=="cuda":torch.cuda.manual_seed_all(seed+491)
    model=MLP(x.shape[1]).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,
                                weight_decay=.01)
    ix=torch.tensor(train_idx,device=DEVICE)
    val=torch.tensor(val_idx,device=DEVICE)
    best=float("inf");patience=0;best_state=None
    for step in range(350):
        sampled=ix[torch.multinomial(w_t[ix],512,replacement=True)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores=model(x_t[sampled])
        loss=nn.functional.binary_cross_entropy_with_logits(
            scores,y_t[sampled])
        loss.backward()
        optimizer.step()
        if (step+1)%10==0:
            model.eval()
            with torch.no_grad():
                err=nn.functional.binary_cross_entropy_with_logits(
                    model(x_t[val]),y_t[val],reduction="none")
                val_loss=float((err*w_t[val]).sum()/(w_t[val].sum()))
            if val_loss<best-0.00005:
                best=val_loss;patience=0
                best_state=copy.deepcopy(model.state_dict())
            else:
                patience+=1
            if patience>=10:break
    model.load_state_dict(best_state)
    model.eval()
    def predictor(inputs):
        with torch.no_grad():
            t=torch.from_numpy(inputs.astype(np.float32)*2-1).to(DEVICE)
            return model(t).sigmoid().cpu().numpy()
    return predictor

def boost_fit_predict(x,y,w,xtest,seed):
    model=HistGradientBoostingClassifier(
        max_iter=250,max_leaf_nodes=15,
        max_depth=5,min_samples_leaf=16,
        learning_rate=.045,l2_regularization=6,
        early_stopping=False,random_state=seed)
    model.fit(x,y,sample_weight=w)
    return lambda inputs:model.predict_proba(inputs)[:,1]

def regret_from_probs(predictor,task,test,y,regime):
    # Oracle reference uses complete latent model only in evaluation.
    k=task["k"]
    p=predictor(test)
    loss=log_loss(p,y)
    x0=test.copy();x0[:,-1]=0
    x1=test.copy();x1[:,-1]=1
    candidate=(predictor(x1)>predictor(x0))
    from cycle13_oracle import prob_oracle
    prob0=prob_oracle(task,x0,regime)
    prob1=prob_oracle(task,x1,regime)
    gap=np.maximum(prob0,prob1)-np.where(candidate,prob1,prob0)
    return {"predictive_loss":loss,
            "policy_regret":float(np.mean(gap)),
            "random_regret":float(np.mean(
                np.maximum(prob0,prob1)-(prob0+prob1)/2))}

def run(seed,regime):
    task=make_task(seed,NUI)
    pre,py=sample(task,seed+1,N_PRE,"stationary")
    post,qy=sample(task,seed+883,BUDGETS[-1],regime)
    test,ty=sample(task,seed+991,TEST,regime)
    data={}
    for budget in BUDGETS:
        x=np.r_[pre,post[:budget]]
        y=np.r_[py,qy[:budget]]
        w=np.r_[np.full(N_PRE,.1),np.ones(budget)]
        data[str(budget)]={}
        for method in METHODS:
            start=time.perf_counter()
            if method=="neural_mlp":
                predictor=mlp_fit_predict(x,y,w,test,seed+budget)
            else:
                predictor=boost_fit_predict(x,y,w,test,seed+budget)
            values=regret_from_probs(predictor,task,test,ty,regime)
            values["fit_eval_seconds"]=time.perf_counter()-start
            data[str(budget)][method]=values
    return data

def main():
    start=time.perf_counter()
    out={"protocol":{"gpu":DEVICE,"trials":REPS,"nui":NUI,
                     "pre":N_PRE,"post_budgets":BUDGETS,
                     "test":TEST,"weighted_pre":.1,
                     "no_hidden_state_or_target_index_used_in_training":True,
                     "neural_validation_early_stop":True},
         "results":{}}
    for regime in REGIMES:
        trials=[]
        for i in range(REPS):
            trials.append(run(ROOT+1199*i,regime))
            if (i+1)%4==0:
                print(regime,i+1,"/",REPS,
                      "elapsed",round(time.perf_counter()-start,1),
                      flush=True)
        out["results"][regime]={}
        for budget in BUDGETS:
            out["results"][regime][str(budget)]={}
            for method in METHODS:
                rows=[r[str(budget)][method] for r in trials]
                out["results"][regime][str(budget)][method]={
                    key:{"mean":statistics.mean(t[key] for t in rows),
                         "SE":statistics.stdev(t[key] for t in rows)
                              /math.sqrt(REPS)}
                    for key in rows[0]}
    out["runtime_seconds"]=time.perf_counter()-start
    path="/home/dtdav/dev/learn/research/cycle17_neural_tree.json"
    with open(path,"w") as f:json.dump(out,f,indent=2)
    print("SAVED",path)
    for regime,rr in out["results"].items():
        print("REGIME",regime)
        for budget,row in rr.items():
            print("BUDGET",budget,{
                method:round(v["policy_regret"]["mean"],5)
                for method,v in row.items()})
if __name__=="__main__":
    main()
