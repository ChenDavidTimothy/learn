"""Cycle 33: falsify robustness claims for privileged-model information design.

All selectors use the same pre-shift finite hypothesis tables. Only the
simulator applies observation-channel corruption after the tables have been
frozen. This isolates the effect of a *wrong likelihood model*, NOT unknown
reward mechanisms, and does not simulate a lifelong trajectory.
"""
import argparse,json,time
import numpy as np
from cycle32_decision_design import (
    METHODS,BUDGET,sample_world,updated,design_scores,
    optimal_probe,summarize
)
def distort(lik,seed,mode):
    r=np.random.default_rng(seed+456341)
    q=lik.copy()
    if mode=='correct':pass
    elif mode=='nuisance_corruption':
        # Mixed corruption on randomly selected half of probe channels,
        # not necessarily reward-relevant.
        ix=r.choice(q.shape[1],q.shape[1]//2,replace=False)
        q[:,ix]=.75-q[:,ix]/2
    elif mode=='full_observation_flip':q=1-q
    elif mode=='partial_flip':
        ix=r.choice(q.shape[1],q.shape[1]//2,replace=False)
        q[:,ix]=1-q[:,ix]
    elif mode=='uninformative_shift':q=.5*np.ones_like(q)
    else:raise ValueError(mode)
    return np.clip(q,.005,.995)
MODES=('correct','nuisance_corruption','partial_flip',
       'full_observation_flip','uninformative_shift')
def trial(seed,mode,cost):
    mu,lik,truth=sample_world(seed,'nuisance')
    true_lik=distort(lik,seed,mode)
    H,E=lik.shape
    rng=np.random.default_rng(seed+818888)
    uniforms=rng.random((BUDGET,E))
    out={}
    for method in METHODS+('no_probe',):
        p=np.full(H,1/H);n=0
        for step in range(BUDGET):
            if method=='no_probe':break
            if method=='dp_optimal':
                _,j=optimal_probe(p,lik,mu,BUDGET-step,cost)
                if j is None:break
            elif method=='random':
                j=int(np.random.default_rng(seed+99887+step).integers(E))
            else:
                kind='evi_greedy' if method=='evi_lookahead' and step>0 else method
                scores=design_scores(p,lik,mu,kind)
                j=int(np.argmax(scores))
                if method.startswith('evi_'):
                    stop_cost=cost*(BUDGET-step if method=='evi_lookahead' and step==0 else 1)
                    if scores[j]<=stop_cost+1e-12:break
            # real data generated using a potentially incorrect channel
            outcome=int(uniforms[step,j]<true_lik[truth,j])
            _,p1,p0=updated(p,lik[:,j]) # learner incorrectly trusts assumed model
            p=p1 if outcome else p0
            n+=1
        act=int(np.argmax(p@mu))
        regret=float(max(mu[truth])-mu[truth,act])
        out[method]={'regret':regret,'net_cost':regret+cost*n,'probes':n}
    return out
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=1600)
    parser.add_argument('--seed-start',type=int,default=33101026)
    parser.add_argument('--probe-cost',type=float,default=.012)
    parser.add_argument('--output',default='research/cycle33_misspecification_results.json')
    a=parser.parse_args();t0=time.perf_counter()
    output={'protocol':{'world':'cycle32 nuisance bit, finite H=64 E=8',
         'training':'not learned; candidates supplied, unannounced observation-model shift',
         'modes':MODES,'seeds_per_mode':a.seeds,'seed_start':a.seed_start,
         'probe_cost':a.probe_cost,'reward_semantics_unchanged':True,
         'all_methods_use_frozen_original_likelihoods':True,
         'objective':'post-probe action regret plus probe cost',
         'significance':'stress-test *incorrect* model assumptions, not a new algorithm'},
         'results':{}}
    for idx,mode in enumerate(MODES):
        records=[]
        for i in range(a.seeds):
            records.append(trial(a.seed_start+idx*500000+211*i,mode,a.probe_cost))
            if (i+1)%400==0:
                print(mode,i+1,'/',a.seeds,'elapsed',round(time.perf_counter()-t0,1),flush=True)
        summary={}
        for method in METHODS+('no_probe',):
            summary[method]={k:summarize([r[method][k] for r in records])
                             for k in records[0][method]}
        for method in METHODS:
            summary['paired_'+method+'_minus_no_probe_net']=summarize(
               [r[method]['net_cost']-r['no_probe']['net_cost'] for r in records])
        output['results'][mode]=summary
        print('RESULT',mode,[(m,round(summary[m]['net_cost']['mean'],5)) for m in METHODS+('no_probe',)],flush=True)
    output['runtime_seconds']=time.perf_counter()-t0
    with open(a.output,'w') as f:json.dump(output,f,indent=2)
    print('SAVED',a.output,round(output['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
