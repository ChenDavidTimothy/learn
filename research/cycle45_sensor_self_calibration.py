"""Cycle 45: controlled sensor-polarity identifiability without a reference.

Genuine action-dependent binary latent transition dynamics, independent
heldout trajectories, *observation-only* post-change calibration.
Pre-change sensors/reward identify the source action convention; a
candidate observation inversion is allowed, but true inversion is HIDDEN.
Learn action-conditioned observation statistics from SOURCE trajectories.
Competing ordinary controls: raw sensor, unconditional sensor marginal,
action-conditioned first moment, pair distribution, privileged oracle.

Terminal reward observed only after all diagnostic observations, so this
isolates sensor self-calibration from immediate reward feedback. No new
general representation algorithm is proposed.
"""
import json,argparse,time
import numpy as np
DYNAMICS={
 'asymmetric':np.array([[.08,.28],[.28,.08]]), # actions x [P(0->1), P(1->0)]
 'symmetric':np.array([[.18,.18],[.28,.28]]),
}
NOISE=.12
def generate(rg,dyn,steps,invert=False,nuisance_prob=.2):
    x=int(rg.integers(2))
    src=[];act=[];nuis=[];latent=[]
    for t in range(steps):
        latent.append(x)
        src.append(x^int(rg.random()<NOISE)^int(invert))
        nuis.append(int(rg.random()<nuisance_prob))
        a=int(rg.integers(2));act.append(a)
        flip=float(dyn[a,x])
        x=x^int(rg.random()<flip)
    return (np.asarray(src,np.int8),np.asarray(act,np.int8),
            np.asarray(nuis,np.int8),np.asarray(latent,np.int8))
def train_statistics(rg,dyn,steps=10000):
    o,a,nuis,x=generate(rg,dyn,steps,False,.2)
    # source "label" conventional orientation can be learned from
    # outcome rewards for random actions, but here source labels are
    # conceptually available through deterministic randomized rewards:
    # reward=1[chosen_action==x], and x=action xor not reward.
    # use the computed label ONLY for source orientation, not the learned
    # diagnostic response density. This is equivalent to observed reward.
    r=(a==x)
    recovered_x=np.bitwise_xor(a,~r.astype(np.int8)&1)
    assert np.array_equal(recovered_x,x)
    sensor_inverted=(np.mean(o!=recovered_x)>.5)
    assert not sensor_inverted
    # pre-change observational signature, no ground truth x
    next_o=o[1:];prev_o=o[:-1];prev_a=a[:-1]
    per_a=np.zeros(2);pairs=np.zeros((2,2,2))
    for aa in (0,1):
        inds=prev_a==aa
        per_a[aa]=(next_o[inds].sum()+.5)/(inds.sum()+1)
        for b in (0,1):
            for c in (0,1):
                pairs[aa,b,c]=np.sum(inds&(prev_o==b)&(next_o==c))
        pairs[aa]=(pairs[aa]+.5)/(pairs[aa].sum()+2.)
    marginal=(o.sum()+.5)/(len(o)+1.)
    return {'next_given_action':per_a,'joint_pairs':pairs,'marginal':marginal,
            'source_observation_error':float(np.mean(o!=recovered_x))}
def predict_polarity(obs,act,model,mode):
    # log scores compare the source observation distribution with its
    # global binary inversion. Source observational statistics learned,
    # not true transition parameters.
    if mode=='no_adapt':return 0
    if mode=='unconditional':
        p=float(model['marginal'])
        vals=obs[1:]
        score=np.sum(np.where(vals==1,np.log(p),np.log1p(-p)))
        score_inv=np.sum(np.where(vals==0,np.log(p),np.log1p(-p)))
    elif mode=='action_mean':
        p=np.clip(model['next_given_action'][act[:-1]],1e-8,1-1e-8)
        vals=obs[1:]
        score=np.sum(np.where(vals,np.log(p),np.log1p(-p)))
        score_inv=np.sum(np.where(1-vals,np.log(p),np.log1p(-p)))
    elif mode=='pair_model':
        p=np.clip(model['joint_pairs'],1e-8,1.)
        v=obs[:-1];w=obs[1:];aa=act[:-1]
        score=np.log(p[aa,v,w]).sum()
        score_inv=np.log(p[aa,1-v,1-w]).sum()
    else:raise ValueError(mode)
    return int(score_inv>score)
def trial(seed,source_len,post_len,episodes,regime,invert,nuisance_shift,reward_probes):
    rg=np.random.default_rng(seed)
    learned=train_statistics(rg,DYNAMICS['asymmetric' if regime!='symmetric' else 'symmetric'],source_len)
    if regime=='symmetric':dyn=DYNAMICS['symmetric']
    elif regime=='action_swap':dyn=DYNAMICS['asymmetric'][::-1]
    else:dyn=DYNAMICS['asymmetric']
    outcomes={name:[] for name in ('no_adapt','unconditional','action_mean','pair_model',
                                 'reward_calibrate','oracle')}
    for k in range(episodes):
        # Balanced, randomized *open-loop* probe actions, same physical
        # environment and random stream for all policies.
        ob,ac,nuis,latent=generate(rg,dyn,post_len,invert,
                                    .8 if nuisance_shift else .2)
        for name in outcomes:
            if name=='oracle':pol=int(invert)
            elif name=='reward_calibrate':
                # Extra service: random-action reward observations for a
                # handful of probe steps, unavailable to sensor-only methods.
                observed_state=ac[:reward_probes]^(
                    1-(ac[:reward_probes]==latent[:reward_probes]).astype(np.int8))
                sensor_mismatch=ob[:reward_probes]^observed_state
                pol=int(sensor_mismatch.sum()>reward_probes/2)
            else:
                pol=predict_polarity(ob,ac,learned,name)
            reward=int((ob[-1]^pol)==latent[-1])
            outcomes[name].append((int(pol==invert),reward))
    return {name:{'polarity_correct':float(np.mean([x[0] for x in vals])),
                  'decision_reward':float(np.mean([x[1] for x in vals]))}
            for name,vals in outcomes.items()}
def aggregate(vals):
    vals=np.asarray(vals,float)
    return {'mean':float(vals.mean()),
        'SE':float(vals.std(ddof=1)/np.sqrt(len(vals)))}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=96)
    p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--source-len',type=int,default=7000)
    p.add_argument('--probes',type=int,default=64)
    p.add_argument('--reward-probes',type=int,default=3)
    p.add_argument('--output',default='research/cycle45_self_calibration_results.json')
    args=p.parse_args();begin=time.perf_counter()
    conditions=[('asymmetric',False,False),('asymmetric',True,False),
       ('asymmetric',False,True),('asymmetric',True,True),
       ('symmetric',False,False),('symmetric',True,False),
       ('action_swap',False,False),('action_swap',True,False)]
    output={'protocol':{'environment':'genuine action-dependent controlled two-state Markov model',
       'source':'pre-shift collected sensor/action histories and randomized reward for convention orientation',
       'source_len':args.source_len,'post_probe_steps':args.probes,
       'test_episodes_per_seed':args.episodes,'source_seeds':args.seeds,
       'sensor_noise':NOISE,'binary_actions':2,'nuisance_rate_pre':.2,'nuisance_rate_drift':.8,
       'post_test_feedback':'terminal reward only, no reward during sensor-only probes',
       'privileged_oracle':'knows true inversion',
       'reward_calibration':'uses '+str(args.reward_probes)+' EXTRA action reward readouts on diagnostic steps, not available to other methods',
       'known_candidate_model_family':'normal vs globally inverted observations; no unknown transformation construction',
       'possible_shift':'sensor inversion, nuisance-only, or action transition swap',
       'no_general_novel_algorithm':True},'conditions':{}}
    for regime_index,(regime,invert,nuisance_shift) in enumerate(conditions):
        key=f'{regime}__sensor_invert{int(invert)}__nuisance_drift{int(nuisance_shift)}'
        raws=[]
        for i in range(args.seeds):
            raws.append(trial(45111026+regime_index*1000000+i*9187,
                         args.source_len,args.probes,args.episodes,
                         regime,invert,nuisance_shift,args.reward_probes))
        summary={}
        for method in raws[0]:
            summary[method]={stat:aggregate([r[method][stat] for r in raws])
                 for stat in ('polarity_correct','decision_reward')}
        output['conditions'][key]=summary
        print('RESULT',key,[(m,round(summary[m]['decision_reward']['mean'],3),
          round(summary[m]['polarity_correct']['mean'],3)) for m in summary],
          'seconds',round(time.perf_counter()-begin,1),flush=True)
        with open(args.output,'w') as f:json.dump(output,f,indent=2)
    output['runtime_seconds']=time.perf_counter()-begin
    with open(args.output,'w') as f:json.dump(output,f,indent=2)
    print('SAVED',args.output,round(output['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
