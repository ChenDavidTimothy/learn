"""Cycle 34: observational gauge ambiguity and the value of reward anchors.

Finite latent model x unknown; probe sensors may be globally inverted
without warning. The true sensor polarity is not observed. Under uniform
prior on latent bits and polarity, probe-only histories are invariant
under (x,polarity)->(complement(x),opposite(polarity)), while optimal
decision flips. Thus no probe-only algorithm can identify the optimal
action without reward feedback, asymmetry or additional assumptions.

We compare exact Bayes adaptive two-experiment design (with and without
randomized action-reward calibration), misspecified known-polarity planner,
oracle polarity-conditioned Bayes, and abstain/zero-probe baseline.
This is an established identifiability/information-design concept.
"""
import argparse,json,time
import numpy as np
from cycle32_decision_design import optimal_probe,updated,summarize,model_value
B=2; C=.012
METHODS=('oracle_polarity','naive_normal','uncertain_probe_only',
         'uncertain_with_reward','reward_only','no_probe')
def generate(seed):
    rng=np.random.default_rng(seed)
    H=64; E=6
    bits=((np.arange(H)[:,None]>>np.arange(6))&1)
    decision=int(rng.integers(6))
    order=rng.permutation(6)
    eps=rng.uniform(.06,.18,size=6)
    base=eps[None,:]+(1-2*eps[None,:])*bits[:,order]
    gap=float(rng.uniform(.3,.48))
    delta=(2*bits[:,decision]-1)*gap
    mu=np.column_stack((.5-delta/2,.5+delta/2))
    return mu,base,decision
def learner_tables(mu,base,method,truth_polarity):
    n=len(mu)
    if method in ('oracle_polarity','naive_normal'):
        polarity=truth_polarity if method=='oracle_polarity' else 0
        probe=base if polarity==0 else 1-base
        model_reward=mu
    else:
        probe=np.vstack((base,1-base))
        model_reward=np.vstack((mu,mu))
    if method in ('uncertain_with_reward','reward_only'):
        reward=model_reward.copy()
        if method=='reward_only':probe=reward
        else:probe=np.column_stack((probe,reward))
    return model_reward,probe
def run(seed,true_polarity):
    mu,base,decision=generate(seed)
    true_model=int(np.random.default_rng(seed+71).integers(64))
    truth_probe=base if true_polarity==0 else 1-base
    true_probs=np.column_stack((truth_probe,mu))
    # Shared random noise for each potential observed feedback.
    rng=np.random.default_rng(seed+741008)
    u=rng.random((B,8))
    outputs={}
    for method in METHODS:
        if method=='no_probe':
            outputs[method]={'regret':float(max(mu[true_model])-mu[true_model,0]),
                 'net_cost':float(max(mu[true_model])-mu[true_model,0]),'probes':0,
                 'calibration':0}
            continue
        candidate_rewards,lik=learner_tables(mu,base,method,true_polarity)
        post=np.full(len(candidate_rewards),1/len(candidate_rewards))
        count=0;calibration=0
        # map likelihood columns to actual outcome columns: sensor0..5,
        # reward actions 6 and 7.
        for step in range(B):
            _,j=optimal_probe(post,lik,candidate_rewards,B-step,C)
            if j is None:break
            observed_column=(j if method not in ('reward_only',) else j+6)
            if observed_column>=6:calibration+=1
            y=int(u[step,observed_column]<true_probs[true_model,observed_column])
            _,p1,p0=updated(post,lik[:,j])
            post=p1 if y else p0
            count+=1
        a=int(np.argmax(post@candidate_rewards))
        loss=float(np.max(mu[true_model])-mu[true_model,a])
        outputs[method]={'regret':loss,'net_cost':loss+C*count,'probes':count,
                         'calibration':calibration}
    return outputs
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=750)
    p.add_argument('--output',default='research/cycle34_reward_anchor_results.json')
    args=p.parse_args();t0=time.perf_counter()
    out={'protocol':{'hidden_models':64,'sensor_polarity_states':2,
        'query_budget':B,'query_cost':C,'seeds_each':args.seeds,
        'seed_start':34121026,
        'true_polarity_known_to_oracle_only':True,
        'reward_channel':'randomized action outcomes, generative means known per candidate',
        'no_reward_observation_except_when_calibrating':True,
        'reward_mean_model_supplied_to_all_methods':True,
        'likelihood_model_supplied_to_all_methods':True,
        'exact_bayesian_planning_over_two_steps':True,
        'methods':METHODS,
        'interpretation':'gauge ambiguity, standard Bayes information design; NOT representation invention'},
        'results':{}}
    for polarity in (0,1):
        trials=[]
        for i in range(args.seeds):
            trials.append(run(34121026+polarity*1000000+i*271,polarity))
            if (i+1)%250==0:
                print('polarity',polarity,'completed',i+1,'/',args.seeds,
                      'elapsed',round(time.perf_counter()-t0,1),flush=True)
        summary={}
        for m in METHODS:
            summary[m]={k:summarize([r[m][k] for r in trials])
                        for k in trials[0][m]}
        for m in METHODS:
            if m=='uncertain_with_reward':continue
            summary['paired_anchor_minus_'+m+'_net']=summarize(
                [r['uncertain_with_reward']['net_cost']-r[m]['net_cost'] for r in trials])
        out['results'][str(polarity)]=summary
        print('RESULT',polarity,[(m,round(summary[m]['net_cost']['mean'],4),
            round(summary[m]['calibration']['mean'],2)) for m in METHODS],flush=True)
    out['runtime_seconds']=time.perf_counter()-t0
    with open(args.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',args.output,round(out['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
