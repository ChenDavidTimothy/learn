"""Cycle 44: early-stopping confound audit of Cycle40 AutoencodeMedium.

Same POPGym 1.0.6 observation episodes, seed streams and reward-only labels.
Conventional GRU, TCN and causal Transformer with a much longer patience.
Tests whether apparent architecture failures are actually premature halting.
No new learning algorithm; compute and epoch disparity disclosed.
"""
import time,json,copy
import numpy as np,torch,torch.nn.functional as F
from cycle40_raw_memory_baselines import (
    collect,tokens,live_policy_eval,MODELS,mask_bce,DEVICE)
torch.set_num_threads(2)
TASK='AutoencodeMedium'
SEEDS=[40111026+i*66791 for i in range(6)]
METHODS=('GRU','TCN','causal_transformer')
def fit_strong(method,train,val,seed,max_epochs=140,patience=45):
    torch.manual_seed(seed)
    model=MODELS[method]().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001)
    best=1e9;best_state=None;stale=0
    x,a,y,w,_=train
    vx,va,vy,vw,_=val
    history=[]
    for epoch in range(max_epochs):
        order=np.random.default_rng(seed+epoch*313).permutation(len(x))
        model.train()
        for i in range(0,len(order),24):
            ids=order[i:i+24]
            xb=x[ids].to(DEVICE);ab=a[ids].to(DEVICE)
            yy=y[ids].to(DEVICE);mask=w[ids].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=mask_bce(model(xb),ab,yy,mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval();losses=[]
        with torch.no_grad():
            for i in range(0,len(vx),24):
                ids=slice(i,i+24)
                val_loss=float(mask_bce(model(vx[ids].to(DEVICE)),
                    va[ids].to(DEVICE),vy[ids].to(DEVICE),vw[ids].to(DEVICE)))
                losses.append((val_loss,float(vw[ids].sum())))
        m=sum(z*n for z,n in losses)/sum(n for z,n in losses)
        history.append(m)
        if m<best-.0003:
            best=m;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state);model.eval()
    return model,{'epochs':len(history),'heldout_reward_bce':best,
        'initial_val_bce':history[0],
        'late_min_bce':min(history[-20:])}
def main():
    start=time.perf_counter()
    output={'protocol':{'environment':'native POPGym AutoencodeMedium',
       'trials':6,'seeds':SEEDS,'training_episodes':700,
       'eval_episodes':140,'inputs':'observations only; randomized chosen-action signed reward labels',
       'model_classes':METHODS,'max_epochs':140,'early_stop_patience':45,
       'same_labeled_data_as_cycle40':True,
       'intention':'falsify early-stop driven apparent learning failure',
       'not_compute_matched_to_cycle40':True},'results':{}}
    for m in METHODS:output['results'][m]=[]
    for seed in SEEDS:
        records=collect(TASK,700,seed)
        test=collect(TASK,140,seed+900000)
        perm=np.random.default_rng(seed+813).permutation(700)
        nval=140
        tr=tokens([records[i] for i in perm[nval:]])
        val=tokens([records[i] for i in perm[:nval]])
        for m in METHODS:
            tic=time.perf_counter()
            model,fit=fit_strong(m,tr,val,seed+len(m)*271)
            runtime=time.perf_counter()-tic
            live,ips=live_policy_eval(TASK,model,test,seed+900000)
            data={'seed':seed,'return':float(live.mean()),
                  'fit_seconds':runtime,**fit}
            output['results'][m].append(data)
            print('TRIAL',m,seed,'return',round(data['return'],4),
                  'epochs',data['epochs'],'wall',round(runtime,2),flush=True)
    output['summary']={}
    for m in METHODS:
        scores=np.array([r['return'] for r in output['results'][m]])
        runtime=np.array([r['fit_seconds'] for r in output['results'][m]])
        output['summary'][m]={'mean_return':float(scores.mean()),
         'SE_return':float(scores.std(ddof=1)/np.sqrt(len(scores))),
         'mean_fit_seconds':float(runtime.mean()),
         'epochs':[r['epochs'] for r in output['results'][m]]}
    output['seconds']=time.perf_counter()-start
    with open('research/cycle44_early_stop_results.json','w') as f:json.dump(output,f,indent=2)
    print('SAVED',output['summary'],'TOTAL',round(output['seconds'],1),flush=True)
if __name__=='__main__':main()
