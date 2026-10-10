"""Cycle 25: strong conventional sparse linear reward predictor on the
same POPGym histories and reward-only labels as Cycle 24.

Temporal action-match features are given; unlike Cycle 24 program argmax,
their coefficients are learned via ordinary logistic regression.
A reduced 1..16-lag family is a preregistered misspecification control.
"""
import json,argparse,time,statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from cycle24_popgym_memory import collect,env
TASKS=['RepeatFirstEasy','RepeatPreviousEasy','RepeatPreviousMedium']

def ft(hist,act,maxlag,include_first):
    c=[int(act==hist[0])] if include_first else []
    c.extend(int(act==hist[-j]) if len(hist)>=j else 0 for j in range(1,maxlag+1))
    return c

def fit(seq,maxlag,first):
    x=[];y=[]
    for episode in seq:
        hist=[]
        for o,a,r in episode:
            hist.append(o)
            if r==0:continue
            x.append(ft(hist,a,maxlag,first));y.append(int(r>0))
    n=len(y)
    rng=np.random.default_rng(417)
    order=rng.permutation(n);nv=max(50,int(.2*n))
    train=order[nv:];val=order[:nv]
    x=np.array(x,np.float64);y=np.array(y,np.int64)
    results=[]
    for C in (.1,1.,10.):
        model=LogisticRegression(C=C,solver='liblinear',max_iter=100)
        model.fit(x[train],y[train])
        p=model.predict_proba(x[val])[:,1]
        loss=-np.mean(y[val]*np.log(np.maximum(p,1e-9))+
                   (1-y[val])*np.log(np.maximum(1-p,1e-9)))
        results.append((loss,C))
    C=min(results)[1]
    model=LogisticRegression(C=C,solver='liblinear',max_iter=100)
    model.fit(x,y)
    return model,C

def test(task,model,maxlag,first,seed,n=80):
    e=env(task);scores=[]
    for j in range(n):
        obs,_=e.reset(seed=seed+j*829);hist=[];total=0.;done=False
        while not done:
            hist.append(int(obs))
            feats=np.array([ft(hist,a,maxlag,first) for a in range(4)])
            choice=int(model.predict_proba(feats)[:,1].argmax())
            obs,r,end,trunc,_=e.step(choice);done=end or trunc;total+=float(r)
        scores.append(total)
    e.close()
    return float(np.mean(scores))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--trials',type=int,default=8)
    a.add_argument('--train',type=int,default=350)
    a.add_argument('--test',type=int,default=100)
    a.add_argument('--output',default='research/cycle25_linear_memory_results.json')
    arg=a.parse_args();now=time.perf_counter()
    out={'meta':{'environment':'external POPGym 1.0.6','trials':arg.trials,
         'train_episodes':arg.train,'test_episodes':arg.test,
         'observations':'only observed history, logged randomized action and reward',
         'families':{'adequate':'first + all lag1..64 matches','misspecified':'first + lags1..16'},
         'cross_val':'20pct randomized training steps for C, refit all; test independent'},'tasks':{}}
    for task in TASKS:
        rows=[]
        for i in range(arg.trials):
            seed=41881026+11011*i
            seq=collect(task,arg.train,seed)
            d={}
            for name,maxlag in [('full',64),('limited',16)]:
                tic=time.perf_counter()
                model,C=fit(seq,maxlag,True)
                d[name]={'C':C,'fit_seconds':time.perf_counter()-tic,
                  'return':test(task,model,maxlag,True,seed+1000000,arg.test)}
            rows.append(d)
            print(task,i+1,[round(d[k]['return'],3) for k in ('full','limited')],flush=True)
        out['tasks'][task]={}
        for name in ('full','limited'):
            out['tasks'][task][name]={}
            for key in ('return','fit_seconds'):
                vals=[r[name][key] for r in rows]
                out['tasks'][task][name][key]={
                    'mean':float(np.mean(vals)),
                    'SE':float(np.std(vals,ddof=1)/np.sqrt(len(vals)))}
    out['runtime_seconds']=time.perf_counter()-now
    with open(arg.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',arg.output,round(out['runtime_seconds'],2))
if __name__=='__main__':main()
