"""Cycle 26: structural program induction on external POPGym memory tasks.

No named lag/reverse/first candidate is supplied. Candidate family:
index(t,W) = a*t + b*W + c, with bounded integer coefficients,
where W is the observable number of initial cards before a phase switch
(and 0 if no switch). Induce coefficients using only logged randomized
actions, observed signed rewards, and observation history.

This is classical enumerable program synthesis with a researcher-given DSL.
It is NOT unrestricted autonomous operator invention or a new paradigm.
"""
import time,json,argparse,itertools
import numpy as np,popgym
from cycle24_popgym_memory import collect as collect_old,env

A=np.array([a for a,b,c in itertools.product((-1,0,1),range(-2,3),range(-70,71))],np.int16)
B=np.array([b for a,b,c in itertools.product((-1,0,1),range(-2,3),range(-70,71))],np.int16)
C=np.array([c for a,b,c in itertools.product((-1,0,1),range(-2,3),range(-70,71))],np.int16)
GRAMMAR=int(len(A))
TASKS=['RepeatFirstEasy','RepeatPreviousEasy','RepeatPreviousMedium','AutoencodeEasy','AutoencodeMedium','AutoencodeHard']
def collect_data(task,n,seed):
    if not task.startswith('Autoencode'):
        seqs=collect_old(task,n,seed)
        return [[(1,o,a,r) for o,a,r in seq] for seq in seqs]
    e=env(task);rg=np.random.default_rng(seed);out=[]
    for j in range(n):
        obs,_=e.reset(seed=seed+37*j);seq=[];done=False
        while not done:
            action=int(rg.integers(4))
            nxt,r,end,trunc,_=e.step(action)
            seq.append((int(obs[0]),int(obs[1]),action,float(r)))
            obs=nxt;done=end or trunc
        out.append(seq)
    e.close();return out
def index_choices(t,W,hist):
    indices=A*t+B*W+C
    valid=(indices>=0)&(indices<=t)
    clipped=np.clip(indices,0,t)
    ans=np.asarray(hist,dtype=np.int16)[clipped]
    # invalid formula defaults to action 0; this costs expected reward
    ans[~valid]=0
    return ans
def learn(data):
    score=np.zeros(GRAMMAR,dtype=np.float64);count=0
    for seq in data:
        hist=[];W=0
        for t,(mode,obs,act,r) in enumerate(seq):
            hist.append(obs)
            if mode==0 and W==0:W=t+1
            if r==0:continue
            outputs=index_choices(t,W,hist)
            score+=4.*(outputs==act)*np.sign(r)
            count+=1
    ix=int(np.argmax(score))
    return (int(A[ix]),int(B[ix]),int(C[ix])),float(score[ix]/count)
def predict(rule,hist,W):
    a,b,c=rule;t=len(hist)-1;idx=a*t+b*W+c
    return int(hist[idx]) if 0<=idx<=t else 0
def evaluate(task,rule,seed,episodes=72):
    e=env(task);scores=[];valid_steps=[]
    for j in range(episodes):
        obs,_=e.reset(seed=seed+j*829)
        hist=[];W=0;done=False;cum=0;pos=0;count=0
        while not done:
            if task.startswith('Autoencode'):
                mode,card=int(obs[0]),int(obs[1])
            else:mode,card=1,int(obs)
            hist.append(card)
            if mode==0 and W==0:W=len(hist)
            a=predict(rule,hist,W)
            obs,r,end,trunc,_=e.step(a)
            cum+=float(r);done=end or trunc
            if r!=0:count+=1;pos+=int(r>0)
        scores.append(cum);valid_steps.append(pos/max(1,count))
    e.close()
    return float(np.mean(scores)),float(np.mean(valid_steps))
def compare(task,seed,ntrain,ntest):
    started=time.perf_counter()
    d=collect_data(task,ntrain,seed)
    rule,ips=learn(d)
    secs=time.perf_counter()-started
    value,accuracy=evaluate(task,rule,seed+1000000,ntest)
    return dict(rule=list(rule),ips_score=ips,return_=value,
                action_accuracy=accuracy,training_seconds=secs)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--train',type=int,default=160)
    p.add_argument('--test',type=int,default=100)
    p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--output',default='research/cycle26_memory_synthesis_results.json')
    args=p.parse_args()
    now=time.perf_counter()
    result={'protocol':{'external_benchmark':'POPGym 1.0.6',
       'grammarsize':GRAMMAR,'expression':'index = a*t+b*W+c',
       'a_range':[-1,0,1],'b_range':list(range(-2,3)),'c_range':[-70,70],
       'operator_names_not_provided_to_model':True,
       'researcher_furnished_expression_grammar':True,
       'actions':'uniform random in observed training trajectories',
       'training_labels':'reward from chosen action only; no correct action or hidden state',
       'testing':'held-out independently seeded live environment episodes',
       'train_episodes':args.train,'test_episodes':args.test,'trials':args.trials},
       'results':{}}
    for task in args.tasks:
        rows=[]
        for i in range(args.trials):
            row=compare(task,25681026+i*77231,args.train,args.test)
            rows.append(row)
            print(task,i+1,row,'elapsed',round(time.perf_counter()-now,1),flush=True)
        result['results'][task]={'chosen_rules':[r['rule'] for r in rows]}
        for key in ('return_','action_accuracy','training_seconds'):
            vals=np.array([r[key] for r in rows])
            result['results'][task][key]={'mean':float(vals.mean()),
                'SE':float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None}
    result['seconds']=time.perf_counter()-now
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',args.output,round(result['seconds'],1),flush=True)
if __name__=='__main__':main()
