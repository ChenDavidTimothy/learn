"""Cycle 24: external POPGym reward-only memory discovery audit.
Program expert priors are given, NOT autonomously invented.
Compare IPS-selected index operators, tabular reward bandit, and GRU.
All train on same uniformly randomized reward/action histories.
"""
import argparse,json,time,copy
import popgym,numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)
TASKS=['RepeatFirstEasy','RepeatPreviousEasy','RepeatPreviousMedium']
EXPERTS=['first']+[f'lag{i}' for i in range(1,65)]+['mode']
def env(task):return getattr(popgym.envs,task)()
def collect(task,n,seed):
    e=env(task);rg=np.random.default_rng(seed);out=[]
    for j in range(n):
        obs,_=e.reset(seed=seed+37*j);seq=[];done=False
        while not done:
            a=int(rg.integers(4));nxt,r,t,tr,_=e.step(a)
            seq.append((int(obs),a,float(r)));obs=nxt;done=t or tr
        out.append(seq)
    e.close();return out
def choices(hist):
    return [hist[0]]+[hist[-i] if len(hist)>=i else hist[-1] for i in range(1,65)]+[int(np.argmax(np.bincount(hist,minlength=4)))]
def program_fit(seqs):
    score=np.zeros(len(EXPERTS));count=0
    for seq in seqs:
        hist=[]
        for o,a,r in seq:
            hist.append(o)
            if r==0:continue
            score+=4*(np.array(choices(hist))==a)*np.sign(r);count+=1
    return int(np.argmax(score)),(score/max(1,count)).tolist()
def table_fit(seqs):
    pos=np.ones((4,4));tot=np.full((4,4),2.)
    for seq in seqs:
        for o,a,r in seq:
            if r==0:continue
            pos[o,a]+=int(r>0);tot[o,a]+=1
    return pos/tot
def input_data(seq):
    out=np.zeros((len(seq),10),np.float32);prev=4;pr=0
    for i,(o,a,r) in enumerate(seq):
        out[i,o]=1
        if prev<4:out[i,4+prev]=1
        out[i,8]=pr;out[i,9]=min(1.,i/104.)
        prev=a;pr=np.sign(r)
    return out
def tensor_data(seqs):
    n=len(seqs);m=max(map(len,seqs));x=np.zeros((n,m,10),np.float32)
    a=np.zeros((n,m),np.int64);y=np.zeros((n,m),np.float32);mask=np.zeros((n,m),np.float32)
    for j,seq in enumerate(seqs):
        l=len(seq);x[j,:l]=input_data(seq)
        a[j,:l]=[s[1] for s in seq]
        y[j,:l]=[int(s[2]>0) for s in seq]
        mask[j,:l]=[int(s[2]!=0) for s in seq]
    return [torch.from_numpy(v) for v in (x,a,y,mask)]
class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.recur=nn.GRU(10,48,batch_first=True);self.head=nn.Linear(48,4)
    def forward(self,x,state=None):
        out,h=self.recur(x,state);return self.head(out),h
def bce(model,batch):
    x,a,y,m=batch;z=model(x)[0].gather(-1,a[:,:,None]).squeeze(-1)
    return (F.binary_cross_entropy_with_logits(z,y,reduction='none')*m).sum()/m.sum()
def gru_fit(seqs,seed,epochs):
    rng=np.random.default_rng(seed);idx=rng.permutation(len(seqs));nv=max(8,len(idx)//5)
    tr=idx[nv:];val=idx[:nv];all_t=tensor_data(seqs)
    torch.manual_seed(seed);model=GRU();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    best=1e9;state=None;stall=0;used=0
    for epoch in range(epochs):
        model.train()
        for st in range(0,len(tr),24):
            ix=rng.permutation(tr)[st:st+24]
            batch=[v[ix] for v in all_t]
            opt.zero_grad();loss=bce(model,batch);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        model.eval()
        with torch.no_grad():v=float(bce(model,[v[val] for v in all_t]))
        used=epoch+1
        if v<best-0.0002:best=v;state=copy.deepcopy(model.state_dict());stall=0
        else:stall+=1
        if stall>=7:break
    model.load_state_dict(state);model.eval()
    return model,used,best
def run_policy(task,kind,model,seed,episodes):
    e=env(task);scores=[];acc=[]
    with torch.no_grad():
        for j in range(episodes):
            ob,_=e.reset(seed=seed+829*j);hist=[];done=False;cum=0;hits=0;n=0
            prev=4;pr=0;hidden=None;t=0
            while not done:
                hist.append(int(ob))
                if kind=='program':a=choices(hist)[model]
                elif kind=='table':a=int(model[int(ob)].argmax())
                else:
                    x=np.zeros((1,1,10),np.float32);x[0,0,int(ob)]=1
                    if prev<4:x[0,0,4+prev]=1
                    x[0,0,8]=pr;x[0,0,9]=min(1.,t/104.)
                    logits,hidden=model(torch.from_numpy(x),hidden)
                    a=int(logits[0,-1].argmax().item())
                ob,r,end,trunc,_=e.step(int(a));done=end or trunc
                cum+=float(r)
                if r!=0:n+=1;hits+=int(r>0)
                prev=a;pr=np.sign(r);t+=1
            scores.append(cum);acc.append(hits/max(1,n))
    e.close();return float(np.mean(scores)),float(np.mean(acc))
def once(task,seed,ntrain,neval,epochs):
    tic=time.perf_counter();seqs=collect(task,ntrain,seed)
    expert,sc=program_fit(seqs);tb=table_fit(seqs)
    training_program=time.perf_counter()-tic
    start=time.perf_counter();gru,used,val=gru_fit(seqs,seed+900000,epochs)
    training_gru=time.perf_counter()-start
    vals={}
    for name,model in [('program',expert),('table',tb),('gru',gru)]:
        reward,acc=run_policy(task,name,model,seed+1000000,neval)
        vals[name]={'return':reward,'action_accuracy':acc,
                     'fit_seconds':training_gru if name=='gru' else training_program}
    vals['program']['selected']=EXPERTS[expert]
    vals['gru']['epochs']=used;vals['gru']['validation_bce']=val
    return vals
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=6)
    p.add_argument('--train',type=int,default=256)
    p.add_argument('--test',type=int,default=64)
    p.add_argument('--epochs',type=int,default=28)
    p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--output',default='research/cycle24_popgym_memory_results.json')
    a=p.parse_args();t0=time.perf_counter()
    result={'protocol':{'benchmark':'POPGym 1.0.6','trials':a.trials,
       'training_episodes':a.train,'test_episodes':a.test,
       'tasks':a.tasks,'max_epochs':a.epochs,
       'training':'uniform randomized actions; only observed actions and signed rewards; same log',
       'prior':'first, lag1..64, mode operators pre-specified; no hidden state',
       'model_selection':'IPS for program, validation BCE for GRU; no test tuning',
       'baseline_cost':'GRU training more costly; not compute matched'},'tasks':{}}
    for task in a.tasks:
        records=[]
        for trial in range(a.trials):
            rec=once(task,24681026+trial*71231,a.train,a.test,a.epochs)
            records.append(rec)
            print(task,trial+1,'/',a.trials,rec['program']['selected'],
                  [round(rec[x]['return'],3) for x in ('program','gru','table')],
                  'elapsed',round(time.perf_counter()-t0,1),flush=True)
        summary={}
        for method in ('program','gru','table'):
            summary[method]={}
            for field in ('return','action_accuracy','fit_seconds'):
                values=np.array([v[method][field] for v in records])
                summary[method][field]={'mean':float(values.mean()),
                   'SE':float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None}
        summary['selected_programs']=[r['program']['selected'] for r in records]
        result['tasks'][task]=summary
    result['runtime_seconds']=time.perf_counter()-t0
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',a.output,round(result['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
