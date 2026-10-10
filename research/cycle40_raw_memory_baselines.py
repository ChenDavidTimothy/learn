"""Cycle 40: reward-only history representation learning on independent POPGym.

Unlike Cycles 24-31, no known lag/reverse/first index is supplied to
any model. Identical observation-only inputs and random-action rewards.
Strong conventional baselines: GRU, dilated causal convolution (TCN),
causal Transformer, memory-free MLP. No proposed novel learner.

Train/evaluate on action-independent POPGym card-memory tasks. We verify
observation trajectory invariance to logged actions before using unbiased
heldout inverse-propensity (IPW) policy evaluation.
"""
import argparse,json,time,copy,hashlib
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
import popgym
from scipy.stats import t as tdist
torch.set_num_threads(2)
TASKS=('RepeatFirstEasy','RepeatPreviousMedium','AutoencodeEasy','AutoencodeMedium')
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A=4
def env(task):return getattr(popgym.envs,task)()
def observation(x):
    if isinstance(x,tuple):return int(x[1]),int(x[0])
    return int(x),1
def collect(task,n,seed):
    e=env(task);random=np.random.default_rng(seed);rows=[]
    for i in range(n):
        ob,_=e.reset(seed=seed+37*i)
        seq=[];done=False
        while not done:
            card,mode=observation(ob)
            act=int(random.integers(4))
            new,r,t,trunc,_=e.step(act)
            seq.append((card,mode,act,float(r)))
            ob=new;done=t or trunc
        rows.append(seq)
    e.close();return rows
def check_action_independence(task,seed=10100):
    e=env(task)
    for i in range(10):
        outcomes=[]
        for action in (0,1,2,3):
            obs,_=e.reset(seed=seed+13*i)
            seq=[];done=False
            while not done:
                new,r,t,tr,_=e.step(action)
                seq.append(observation(new));done=t or tr
            outcomes.append(seq)
        assert outcomes[0]==outcomes[1]==outcomes[2]==outcomes[3],task
    e.close()
def tokens(rows):
    n=len(rows);T=max(len(r) for r in rows)
    x=np.zeros((n,T,8),dtype=np.float32)
    actions=np.zeros((n,T),np.int64)
    y=np.zeros((n,T),np.float32)
    weights=np.zeros((n,T),np.float32)
    valid=np.zeros((n,T),np.float32)
    for i,row in enumerate(rows):
        for t,(card,mode,action,reward) in enumerate(row):
            x[i,t,card]=1.
            x[i,t,4+mode]=1.
            x[i,t,6]=t/310.
            x[i,t,7]=1.
            actions[i,t]=action
            y[i,t]=float(reward>0)
            weights[i,t]=float(reward!=0)
            valid[i,t]=1.
    return tuple(torch.from_numpy(v) for v in (x,actions,y,weights,valid))
class MemoryFree(nn.Module):
    def __init__(self,d=8,h=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),
            nn.Linear(h,h),nn.ReLU(),nn.Linear(h,A))
    def forward(self,x):return self.net(x)
class GRU(nn.Module):
    def __init__(self,d=8,h=96):
        super().__init__();self.enc=nn.Linear(d,h)
        self.gru=nn.GRU(h,h,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,A))
    def forward(self,x):return self.head(self.gru(F.relu(self.enc(x)))[0])
class ConvBlock(nn.Module):
    def __init__(self,h,dil):
        super().__init__();self.dil=dil;self.fc1=nn.Conv1d(h,h,3,dilation=dil)
        self.fc2=nn.Conv1d(h,h,1);self.ln=nn.LayerNorm(h)
    def forward(self,x):
        z=x.transpose(1,2)
        z=F.pad(z,(2*self.dil,0))
        z=F.relu(self.fc1(z))
        z=self.fc2(z).transpose(1,2)
        return self.ln(x+z)
class TCN(nn.Module):
    def __init__(self,d=8,h=64):
        super().__init__();self.enc=nn.Linear(d,h)
        self.blocks=nn.ModuleList([ConvBlock(h,2**i) for i in range(8)])
        self.head=nn.Linear(h,A)
    def forward(self,x):
        x=F.relu(self.enc(x))
        for b in self.blocks:x=b(x)
        return self.head(x)
class CausalAttention(nn.Module):
    def __init__(self,d=8,h=64):
        super().__init__();self.enc=nn.Linear(d,h)
        self.pos=nn.Embedding(320,h)
        layer=nn.TransformerEncoderLayer(h,4,dim_feedforward=128,
                  dropout=.05,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(layer,num_layers=2,
               enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,A))
    def forward(self,x):
        B,T,_=x.shape
        pos=torch.arange(T,device=x.device)
        h=self.enc(x)+self.pos(pos)[None,:,:]
        causal=torch.ones((T,T),device=x.device,dtype=torch.bool).triu(1)
        return self.head(self.tr(h,mask=causal))
MODELS={'memory_free':MemoryFree,'GRU':GRU,'TCN':TCN,'causal_transformer':CausalAttention}
def mask_bce(z,a,y,mask):
    chosen=z.gather(-1,a.unsqueeze(-1)).squeeze(-1)
    return (F.binary_cross_entropy_with_logits(chosen,y,reduction='none')*mask).sum()/mask.sum().clamp(min=1)
def fit(method,train,val,seed,epochs=45):
    torch.manual_seed(seed)
    model=MODELS[method]().to(DEVICE)
    optim=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001)
    best_loss=float('inf');best_weights=None;stale=0;used=0
    x,a,y,weights,_=train
    vx,va,vy,vweights,_=val
    for epoch in range(epochs):
        order=np.random.default_rng(seed+epoch*313).permutation(len(x))
        model.train()
        for i in range(0,len(order),24):
            ids=order[i:i+24]
            xb=x[ids].to(DEVICE);ab=a[ids].to(DEVICE)
            yy=y[ids].to(DEVICE);mask=weights[ids].to(DEVICE)
            optim.zero_grad(set_to_none=True)
            error=mask_bce(model(xb),ab,yy,mask)
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        model.eval();losses=[]
        with torch.no_grad():
            for j in range(0,len(vx),24):
                ids=slice(j,j+24)
                ll=mask_bce(model(vx[ids].to(DEVICE)),
                    va[ids].to(DEVICE),vy[ids].to(DEVICE),vweights[ids].to(DEVICE))
                losses.append((float(ll),float(vweights[ids].sum())))
        dev=sum(l*w for l,w in losses)/sum(w for l,w in losses)
        used=epoch+1
        if dev<best_loss-.0003:
            best_loss=dev;best_weights=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=9:break
    model.load_state_dict(best_weights);model.eval()
    return model,{'epochs':used,'heldout_reward_bce':best_loss}
def live_policy_eval(task,model,eval_rows,seed):
    tensor=tokens(eval_rows)[0]
    all_predictions=[]
    with torch.no_grad():
        for j in range(0,len(tensor),32):
            yhat=model(tensor[j:j+32].to(DEVICE))
            all_predictions.extend(yhat.argmax(dim=2).cpu().tolist())
    e=env(task);returns=[];rewards=[];episodes=[]
    for i,(row,chosen) in enumerate(zip(eval_rows,all_predictions)):
        ob,_=e.reset(seed=seed+37*i)
        ret=0.;matches=0;scored=0;ips=0.
        for j,(_,_,logged_act,logged_reward) in enumerate(row):
            a=int(chosen[j])
            # Strong validation of counterfactual evaluability
            observed=observation(ob)
            assert observed==row[j][:2],(task,i,j,observed,row[j][:2])
            ob,r,term,trunc,_=e.step(a)
            ret+=float(r)
            if r: scored+=1;matches+=int(r>0)
            ips+=4*(logged_act==a)*logged_reward
        returns.append(ret);rewards.append(ips)
    e.close()
    return np.asarray(returns),np.asarray(rewards)
def trial(task,seed,ntrain,ntest,epochs):
    check_action_independence(task)
    observed=collect(task,ntrain,seed)
    eval_rows=collect(task,ntest,seed+900000)
    perm=np.random.default_rng(seed+813).permutation(ntrain)
    nv=max(32,int(ntrain*.2))
    tr=tokens([observed[i] for i in perm[nv:]])
    val=tokens([observed[i] for i in perm[:nv]])
    out={}
    for method in MODELS:
        t0=time.perf_counter()
        model,stats=fit(method,tr,val,seed+len(method)*271,epochs)
        seconds=time.perf_counter()-t0
        returns,ips=live_policy_eval(task,model,eval_rows,seed+900000)
        out[method]={'return_mean':float(returns.mean()),
             'return_SE':float(returns.std(ddof=1)/np.sqrt(len(returns))),
             'fit_seconds':seconds,**stats,
             'ips_return_mean':float(ips.mean()),
             'returns':returns.tolist()}
        print('METHOD',task,method,'seed',seed,'return',round(out[method]['return_mean'],4),
            'train_time',round(seconds,2),'epochs',stats['epochs'],flush=True)
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=4)
    p.add_argument('--train',type=int,default=500)
    p.add_argument('--test',type=int,default=120)
    p.add_argument('--epochs',type=int,default=45)
    p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--output',default='research/cycle40_raw_memory_results.json')
    args=p.parse_args();start=time.perf_counter()
    out={'protocol':{'external_environment':'POPGym 1.0.6',
      'methods':list(MODELS),'tasks':args.tasks,'train_episodes':args.train,
      'test_episodes':args.test,'trials_per_task':args.trials,
      'maximum_epochs':args.epochs,'reward_only_training':True,
      'input':'observation card, mode, normalized timestep, constant; NO exact memory lag or reverse index; no hidden state, no true reward function',
      'training_actions':'uniform random; response label reward sign for chosen action only',
      'model_early_stop':'20 pct of training episodes, heldout BCE',
      'test':'actual live returns by replay of exactly matched observation histories; verified action independence',
      'separate_test_seeds':True,'methods_not_compute_matched':True,
      'no_novel_algorithm':True,'device':str(DEVICE)},'tasks':{}}
    for task in args.tasks:
        records=[]
        for i in range(args.trials):
            rec=trial(task,40111026+i*66791,args.train,args.test,args.epochs)
            records.append(rec)
            print('TRIAL_FINISHED',task,i+1,'of',args.trials,
                  'elapsed',round(time.perf_counter()-start,1),flush=True)
        result={}
        for method in MODELS:
            for field in ('return_mean','fit_seconds','heldout_reward_bce'):
                vals=np.array([r[method][field] for r in records])
                result.setdefault(method,{})[field]={
                   'mean':float(vals.mean()),'SE':float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None}
            result[method]['epochs']=[r[method]['epochs'] for r in records]
        for m in MODELS:
            if m=='GRU':continue
            diff=np.array([r[m]['return_mean']-r['GRU']['return_mean'] for r in records])
            result['paired_'+m+'_minus_GRU']={'mean':float(diff.mean()),
                'SE':float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else None}
        out['tasks'][task]=result
        with open(args.output,'w') as f:json.dump(out,f,indent=2)
    out['seconds']=time.perf_counter()-start
    with open(args.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',args.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':main()
