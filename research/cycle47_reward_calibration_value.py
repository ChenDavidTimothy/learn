"""Cycle 47: recoverability when sensor and transition changes are aliased.

One observed postchange system has old transitions and inverted sensor;
another has swapped transitions and normal sensor. They have IDENTICAL
action/observation transcript laws, but opposite optimal terminal action.
Comparison: no adaptation, prechange action-conditioned likelihood,
minimax random action, calibration from 1/3/5 *observed reward queries*,
privileged polarity oracle. All methods share actions, stimuli and terminal
evaluation. Extra reward queries cost 0.025 each.
Nothing here is a novel learning algorithm; it tests a specific
operational condition for external supervision/interventions.
"""
import argparse,json,time
import numpy as np
from cycle45_sensor_self_calibration import DYNAMICS,generate,train_statistics,predict_polarity,NOISE

COST=.025
METHODS=('no_adapt','action_mean','pair_model','minimax_random','reward1',
         'reward3','reward5','oracle')
CONDITIONS=(('invert_only',DYNAMICS['asymmetric'],1),
           ('transition_swap_only',DYNAMICS['asymmetric'][::-1],0),
           ('normal',DYNAMICS['asymmetric'],0),
           ('swap_and_invert',DYNAMICS['asymmetric'][::-1],1))
def single(seed,source_steps,probe_steps,episodes):
    rg=np.random.default_rng(seed)
    model=train_statistics(rg,DYNAMICS['asymmetric'],source_steps)
    record={}
    for label,dynamics,polarity in CONDITIONS:
        results={m:[] for m in METHODS}
        for ep in range(episodes):
            ob,ac,nuis,x=generate(rg,dynamics,probe_steps,polarity,.2)
            for method in METHODS:
                if method=='oracle':hat=polarity
                elif method=='minimax_random':hat=int(rg.integers(2))
                elif method.startswith('reward'):
                    count=int(method[6:])
                    rewarded=(ac[:count]==x[:count]).astype(np.int8)
                    state=ac[:count]^(1-rewarded)
                    # Infer polarity from selected-action reward feedback;
                    # the state is NOT separately observed.
                    mismatch=ob[:count]^state
                    hat=int(mismatch.sum()>count/2)
                else:hat=predict_polarity(ob,ac,model,method)
                a=int(ob[-1]^hat)
                reward=int(a==x[-1])
                n_queries=int(method[6:]) if method.startswith('reward') else 0
                results[method].append((reward,reward-COST*n_queries,
                                        int(hat==polarity)))
        record[label]={m:np.mean(v,axis=0).tolist() for m,v in results.items()}
    return record
def mean_se(xs):
    xs=np.asarray(xs,float)
    return {'mean':float(xs.mean()),
            'SE':float(xs.std(ddof=1)/np.sqrt(len(xs)))}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=128)
    parser.add_argument('--episodes',type=int,default=40)
    parser.add_argument('--source-steps',type=int,default=8000)
    parser.add_argument('--probe-steps',type=int,default=64)
    parser.add_argument('--output',default='research/cycle47_reward_calibration_results.json')
    args=parser.parse_args();start=time.perf_counter()
    report={'protocol':{'model':'action-conditioned hidden two-state Markov sensor model',
       'control':'postchange sensor-only transcript laws aliased across inversion and swapped dynamics',
       'seeds':args.seeds,'source_steps':args.source_steps,
       'episodes_each_condition_per_seed':args.episodes,
       'post_sensor_steps':args.probe_steps,'extra_reward_query_cost':COST,
       'reward_querier_gets_chosen_action_reward_only':True,
       'other_methods_get_no_postchange_reward_until_terminal_test':True,
       'method_families':METHODS,'source_statistics_learned':True,
       'reward_noise':'deterministic terminal reward, noisy sensor with flip 0.12',
       'strong_privileged_control':'oracle knows true polarity',
       'scientific_novelty':'known experimental identifiability and supervised sensing, not new algorithm'},'conditions':{}}
    rows=[]
    for i in range(args.seeds):
        rows.append(single(47111026+8191*i,args.source_steps,
                           args.probe_steps,args.episodes))
        if (i+1)%32==0:
            print('SEEDS',i+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
    for label,_,polarity in CONDITIONS:
        report['conditions'][label]={}
        for m in METHODS:
            per=np.array([row[label][m] for row in rows])
            report['conditions'][label][m]={
                'raw_terminal_reward':mean_se(per[:,0]),
                'net_utility':mean_se(per[:,1]),
                'polarity_correct':mean_se(per[:,2])}
        print('RESULT',label,[(m,round(report['conditions'][label][m]['net_utility']['mean'],3))
                for m in METHODS],flush=True)
    report['runtime_seconds']=time.perf_counter()-start
    with open(args.output,'w') as f:json.dump(report,f,indent=2)
    print('SAVED',args.output,round(report['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
