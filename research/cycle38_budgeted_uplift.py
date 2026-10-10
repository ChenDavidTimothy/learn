"""Cycle 38: real RCT, capacity-constrained treatment-effect ranking.

Under a predeclared 20% (and secondary 40%) Mens Email send budget,
compare causal ranking models with uniform randomized mailing and
simple covariate rankings. Estimate heldout randomized policy value using
known 1/3 propensities, independent nuisance model, paired bootstrap.

This is an established uplift-modeling evaluation, NOT a novel algorithm.
No person-level counterfactual truth available. The send budget is
hypothetical, not a documented constraint on the 2008 campaign.
"""
import argparse,json,time
import numpy as np
from sklearn.model_selection import train_test_split
from cycle37_hillstrom_policy import (
    load, t_hgb,t_logit,s_logit,dr_pseudoutcome,bootstrap_diff
)
BUDGETS=(.2,.4)
def dr_random_scores(a,y,m,q):
    n=len(a)
    reward=q*m[:,1]+(1-q)*m[:,0]
    weights=np.where(a==1,q*3,np.where(a==0,(1-q)*3,0.))
    return reward+weights*(y-m[np.arange(n),a])
def dr_deterministic(a,y,m,pi):
    n=len(a)
    return m[np.arange(n),pi]+3*(a==pi)*(y-m[np.arange(n),a])
def rank_policy(values,q):
    n=len(values);k=int(round(n*q))
    chosen=np.zeros(n,dtype=np.int64)
    selected=np.argsort(values,kind='stable')[-k:]
    chosen[selected]=1
    return chosen
def run(seed,bootstrap=300):
    x,y,a,names,_=load()
    idx=np.arange(len(a))
    t,tmp=train_test_split(idx,test_size=.4,stratify=a,random_state=seed)
    v,h=train_test_split(tmp,test_size=.5,stratify=a[tmp],random_state=seed+13)
    xt,at,yt=x[t],a[t],y[t]
    xv,av,yv=x[v],a[v],y[v]
    xh,ah,yh=x[h],a[h],y[h]
    mv,mh=t_hgb(xt,yt,at,[xv,xh],seed+11919)
    predictions={
       'S_logistic':s_logit(xt,yt,at,[xv,xh],seed),
       'T_logistic':t_logit(xt,yt,at,[xv,xh],seed),
       'T_boosted':t_hgb(xt,yt,at,[xv,xh],seed),
       'DR_direct':dr_pseudoutcome(xt,yt,at,[xv,xh],seed)}
    # raw covariate ranks, fixed a priori:
    nameslist=list(names)
    history_idx=nameslist.index('history')
    recency_idx=nameslist.index('recency')
    rankers={
       'history_high':(xv[:,history_idx],xh[:,history_idx]),
       'recency_recent':(-xv[:,recency_idx],-xh[:,recency_idx])}
    for name,(pv,ph) in predictions.items():
        rankers[name]=(pv[:,1]-pv[:,0],ph[:,1]-ph[:,0])
    result={}
    for q in BUDGETS:
        baseval=dr_random_scores(av,yv,mv,q).mean()
        base=dr_random_scores(ah,yh,mh,q)
        # Strong fallback: choose uniform random allocation on validation
        # if no learned ranker exceeds it. This is a stochastic reference,
        # spending budget q in expectation rather than exactly per batch.
        pairs={'uniform_random':base}
        choices={'uniform_random':float(baseval)}
        for name,(sv,sh) in rankers.items():
            pi_v=rank_policy(sv,q);pi_h=rank_policy(sh,q)
            choices[name]=float(dr_deterministic(av,yv,mv,pi_v).mean())
            pairs[name]=dr_deterministic(ah,yh,mh,pi_h)
        selected=max(choices,key=choices.get)
        result[str(q)]={'selected':selected,
            'validation_random_value':float(baseval),
            'validation_values':choices,
            'heldout_random_value':float(base.mean()),
            'heldout_values':{name:float(v.mean()) for name,v in pairs.items()},
            'heldout_selected_minus_random':bootstrap_diff(
                  pairs[selected],base,seed+701,bootstrap),
            'heldout_all_minus_random':{
               k:bootstrap_diff(v,base,seed+931+i,bootstrap)
               for i,(k,v) in enumerate(pairs.items())}}
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--splits',type=int,default=6)
    p.add_argument('--bootstrap',type=int,default=800)
    p.add_argument('--output',default='research/cycle38_budgeted_uplift_results.json')
    arg=p.parse_args();started=time.perf_counter()
    out={'protocol':{'source':'Hillstrom MineThatData 2008 randomized 3-arm, 64000 customers',
       'treatment_policy':'send Mens E-Mail to budgeted portion, No E-Mail otherwise',
       'budgets':BUDGETS,'propensity_each_arm':1/3,
       'raw_dataset_not_committed':True,
       'learners':['S logistic with arm interactions','T logistic',
                   'T gradient boosted','DR direct pseudooutcome',
                   'rank past spend','rank recency'],
       'reference_policy':'stochastic population random send at budget q',
       'policy_value':'known-propensity doubly robust using independent train-only nuisance',
       'val_selection':'best val DR effect among rankers for each budget',
       'evaluation':'strict disjoint heldout customers, paired bootstraps',
       'splits':arg.splits,'seed_start':38011026,
       'split_reuse':'sensitivity, NOT independent experiments',
       'economic_assumption':'hypothetical fixed send proportion, no measured email cost',
       'scientific_claim':'testing established treatment-effect learning, not new algorithm'},
       'runs':[]}
    for i in range(arg.splits):
        r=run(38011026+1729*i,arg.bootstrap)
        out['runs'].append(r)
        print('RUN',i+1,'/',arg.splits,
            [(q,r[str(q)]['selected'],
              round(r[str(q)]['heldout_selected_minus_random']['mean'],5),
              r[str(q)]['heldout_selected_minus_random']['CI95']) for q in BUDGETS],
            'elapsed',round(time.perf_counter()-started,1),flush=True)
    out['seconds']=time.perf_counter()-started
    with open(arg.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',arg.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':main()
