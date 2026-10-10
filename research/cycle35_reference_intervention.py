"""Cycle 35: a symmetry-breaking reference probe versus reward-only learning.

A sensor's global polarity is unknown. Normal sensors alone cannot
distinguish (latent bits, polarity) from (complement bits, inverse polarity).
An explicit known-state calibration reference (latent reference bit=0)
is added as a third kind of intervention, distinguished from normal
unknown-state observation probes and randomized reward probes.

All likelihood tables and the exact reference guarantee are supplied.
This is model-based active experiment design, not autonomous invention.
"""
import argparse,json,time
import numpy as np
from cycle32_decision_design import optimal_probe,updated,summarize
from cycle34_reward_anchor import generate,B,C

METHODS=('oracle_polarity','unknown_probe_only','unknown_reward_only',
         'unknown_reference_only','unknown_joint','no_probe')
REF_ERROR=.08
def trial(seed,polarity,broken=False):
    mu,base,_=generate(seed)
    theta=int(np.random.default_rng(seed+71).integers(len(mu)))
    qtrue=base if polarity==0 else 1-base
    reward_true=mu
    ref_true=.5 if broken else (REF_ERROR if polarity==0 else 1-REF_ERROR)
    likelihood_real=np.column_stack((qtrue,reward_true,
                                    np.full((len(mu),1),ref_true)))
    noise=np.random.default_rng(seed+371010).random((B,9))
    results={}
    for method in METHODS:
        if method=='no_probe':
            regret=float(np.max(mu[theta])-mu[theta,0])
            results[method]={'regret':regret,'net_cost':regret,
                'queries':0,'reference_queries':0,'reward_queries':0}
            continue
        if method=='oracle_polarity':
            rewards=mu;likelihood=qtrue
            real_ids=np.arange(6)
        else:
            rewards=np.vstack([mu,mu])
            sensors=np.vstack([base,1-base])
            ref=np.r_[np.full(len(mu),REF_ERROR),np.full(len(mu),1-REF_ERROR)][:,None]
            reward_lik=rewards
            if method=='unknown_probe_only':
                likelihood=sensors;real_ids=np.arange(6)
            elif method=='unknown_reward_only':
                likelihood=reward_lik;real_ids=np.array([6,7])
            elif method=='unknown_reference_only':
                likelihood=np.column_stack((sensors,ref))
                real_ids=np.r_[np.arange(6),8]
            else:
                likelihood=np.column_stack((sensors,reward_lik,ref))
                real_ids=np.arange(9)
        post=np.full(len(rewards),1/len(rewards))
        count=0;nr=0;nref=0
        for step in range(B):
            _,j=optimal_probe(post,likelihood,rewards,B-step,C)
            if j is None:break
            e=int(real_ids[j])
            y=int(noise[step,e]<likelihood_real[theta,e])
            _,p1,p0=updated(post,likelihood[:,j])
            post=p1 if y else p0
            count+=1;nr+=(e in (6,7));nref+=(e==8)
        a=int(np.argmax(post@rewards))
        reg=float(np.max(mu[theta])-mu[theta,a])
        results[method]={'regret':reg,'net_cost':reg+C*count,
            'queries':count,'reference_queries':nref,'reward_queries':nr}
    return results
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=1000)
    parser.add_argument('--output',default='research/cycle35_reference_results.json')
    a=parser.parse_args();t0=time.perf_counter()
    output={'protocol':{'world':'finite binary factor model, H=64',
       'probe_types':['6 sensors with unknown global polarity',
                      '2 randomized action reward observations',
                      'known latent-state=0 reference reading'],
       'reference_error':REF_ERROR,'query_budget':B,'cost_each':C,
       'model_known_to_all':'all hypothesis likelihood/reward tables assumed known',
       'true_polarity_hidden_except_oracle':True,
       'blind_breakage':'reference silently loses meaning under broken condition',
       'samples_per_condition':a.seeds,'seed_start':35111026,
       'methods':METHODS,
       'interpretation':'structural identifiability test, NOT new algorithm'},
       'conditions':{}}
    for polarity in [0,1]:
        for broken in [False,True]:
            key=f'polarity{polarity}_reference_{"broken" if broken else "valid"}'
            records=[]
            for i in range(a.seeds):
                records.append(trial(35111026+polarity*100000+int(broken)*700000+i*419,polarity,broken))
                if (i+1)%250==0:
                    print(key,i+1,'/',a.seeds,'elapsed',round(time.perf_counter()-t0,1),flush=True)
            summary={}
            for method in METHODS:
                summary[method]={k:summarize([r[method][k] for r in records])
                                for k in records[0][method]}
            for method in METHODS:
                if method=='unknown_joint':continue
                summary['paired_joint_minus_'+method+'_net']=summarize(
                    [r['unknown_joint']['net_cost']-r[method]['net_cost'] for r in records])
            output['conditions'][key]=summary
            print('RESULT',key,[(m,round(summary[m]['net_cost']['mean'],4),
                     round(summary[m]['reference_queries']['mean'],2))
                     for m in METHODS],flush=True)
    output['seconds']=time.perf_counter()-t0
    with open(a.output,'w') as f:json.dump(output,f,indent=2)
    print('SAVED',a.output,round(output['seconds'],1),flush=True)
if __name__=='__main__':main()
