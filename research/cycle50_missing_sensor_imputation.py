"""Cycle 50: inference-only recovery from a failed RGB sensor channel.

Separate from signed-permutation palette correction: a channel is erased,
so no pixelwise invertible transform can restore it. Source-domain
unlabeled RGB observations can nonetheless train a conditional estimator
E[missing channel | two surviving channels], exploiting *redundancy*.

Standard baselines: leave channel zero, fill with global mean, linear
conditional regression, nonparametric 2-channel histogram/colorization,
and privileged intact-source observation upper bound. The missing
channel identity is diagnosed by tiny empirical variance in shifted
unlabeled observations; original training has no missing channels.

This is established feature imputation/test-time restoration, NOT a new
learning algorithm. Test on native action-dependent POPGym Arcade pixel
POMDP, heldout live reward, multiple independent training seeds.
"""
import argparse,json,time
import numpy as np
import jax,jax.numpy as jnp,popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from cycle48_arcade_observation_adapter import trained,config_for
jax.config.update('jax_platform_name','gpu')
CHANNELS=('red','green','blue')
def get_frames(env,p,seed,n_envs=8,n_steps=64):
    ob,st=jax.vmap(env.reset,in_axes=(0,None))(
        jax.random.split(jax.random.PRNGKey(seed),n_envs),p)
    def step(carry,_):
        obs,state,rng=carry
        rng,ar,kr=jax.random.split(rng,3)
        acts=jax.random.randint(ar,(n_envs,),0,5)
        new,ns,reward,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
           jax.random.split(kr,n_envs),state,acts,p)
        return (new,ns,rng),obs
    _,collected=jax.lax.scan(step,(ob,st,jax.random.PRNGKey(seed+70017)),
                             None,length=n_steps)
    return np.asarray(collected)
def fit_source(frames):
    rgb=frames.reshape(-1,3).astype(np.float32)
    n=rgb.shape[0]
    mu=rgb.mean(axis=0)
    models={}
    for c in range(3):
        other=[j for j in range(3) if j!=c]
        # A simple linear feature-completion model.
        sub=rgb[::40]
        D=np.column_stack((np.ones(len(sub),np.float32),sub[:,other]))
        beta=np.linalg.lstsq(D,sub[:,c],rcond=None)[0]
        # Ordinary nonparametric discrete color palette conditional mean.
        key=frames[...,other[0]].astype(np.int32)*256+frames[...,other[1]].astype(np.int32)
        idx=key.ravel()
        count=np.bincount(idx,minlength=65536)
        weighted=np.bincount(idx,weights=frames[...,c].ravel(),minlength=65536)
        # Smoothed conditional estimate (source learned, no target labels).
        lookup=((weighted+2*mu[c])/(count+2)).astype(np.float32)
        models[c]={'other':other,'linear':beta.tolist(),
                   'lookup':lookup.tolist(),'nonzero_palette_bins':int((count>0).sum())}
    return mu.tolist(),models
def corrupt(obs,removed):
    if removed<0:return obs
    return obs.at[...,removed].set(0)
def diagnose_shift(frames,source_mu):
    flat=frames.reshape(-1,3).astype(float)
    observed_sd=flat.std(axis=0)
    # Target channel erased exactly => std zero. Explicitly thresholded,
    # not a general detector for arbitrary visual domain changes.
    candidates=np.flatnonzero(observed_sd<1.)
    return int(candidates[0]) if len(candidates)==1 else -1,observed_sd.tolist()
def evaluate(env,p,model,seed,removed,method,mu,models,
             n_envs=24,n_steps=512):
    obs,st=jax.vmap(env.reset,in_axes=(0,None))(
        jax.random.split(jax.random.PRNGKey(seed),n_envs),p)
    means=jnp.asarray(mu,jnp.float32)
    if removed>=0:
        chosen=models[removed]
        keep=jnp.asarray(chosen['other'])
        beta=jnp.asarray(chosen['linear'],jnp.float32)
        lut=jnp.asarray(chosen['lookup'],jnp.float32)
    def repair(image):
        if method=='intact_oracle':return image.astype(jnp.float32)
        z=corrupt(image,removed).astype(jnp.float32)
        if removed<0 or method=='unrepaired':return z
        if method=='constant_mean':
            return z.at[...,removed].set(means[removed])
        if method=='linear':
            mapped=beta[0]+beta[1]*z[...,keep[0]]+beta[2]*z[...,keep[1]]
            return z.at[...,removed].set(jnp.clip(mapped,0,255.))
        if method=='palette_lut':
            idx=z[...,keep[0]].astype(jnp.int32)*256+z[...,keep[1]].astype(jnp.int32)
            return z.at[...,removed].set(lut[idx])
        raise ValueError(method)
    def step(carry,_):
        ob,states,rng=carry
        a=model(repair(ob)).argmax(-1)
        rng,k=jax.random.split(rng)
        new,ns,reward,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(k,n_envs),states,a,p)
        return (new,ns,rng),(info['returned_episode'].astype(jnp.float32),
               info['returned_episode_returns'].astype(jnp.float32),
               reward)
    _,(done,rewarded,stepreward)=jax.lax.scan(step,
          (obs,st,jax.random.PRNGKey(seed+333)),None,length=n_steps)
    n=float(done.sum())
    return {'episodes_completed':int(n),
       'episode_return_mean':float((done*rewarded).sum()/max(n,1.)),
       'step_reward_mean':float(stepreward.mean())}
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--seeds',type=int,default=3)
    a.add_argument('--train-steps',type=int,default=65536)
    a.add_argument('--source-envs',type=int,default=8)
    a.add_argument('--source-steps',type=int,default=64)
    a.add_argument('--target-envs',type=int,default=8)
    a.add_argument('--target-steps',type=int,default=16)
    a.add_argument('--eval-envs',type=int,default=32)
    a.add_argument('--eval-steps',type=int,default=512)
    a.add_argument('--task',default='NoisyCartPoleEasy')
    a.add_argument('--output',default='research/cycle50_missing_sensor_results.json')
    args=a.parse_args();start=time.perf_counter()
    env,p=popgym_arcade.make(args.task,partial_obs=True,obs_size=128)
    env=LogWrapper(env)
    result={'protocol':{'external':'native POPGym Arcade 0.0.6',
      'task':args.task,'training':'published memory-free PQN',
      'train_episodes_steps':args.train_steps,
      'source_unlabeled_RGB_frames':args.source_envs*args.source_steps,
      'target_unlabeled_frames':args.target_envs*args.target_steps,
      'reward_labels_for_imputation':False,
      'model_family':'per-color linear regression and discrete conditional palette LUT',
      'candidate_missing_channels':CHANNELS,
      'detection':'one target channel with empirical pixel SD < 1; no generic sensor corruption diagnosis',
      'no_new_algorithm':True,'test_seed_trials':args.seeds,
      'test_actors':args.eval_envs,'eval_steps_per_actor':args.eval_steps,
      'oracle_privilege':'unmodified source observation, impossible after erasure'},'trials':[]}
    for i in range(args.seeds):
        seed=50111026+12131*i
        actor,tsec=trained(config_for(seed,args.train_steps,args.task),seed)
        print('TRAINED',seed,'seconds',round(tsec,1),flush=True)
        source=get_frames(env,p,seed+100000,args.source_envs,args.source_steps)
        mean,models=fit_source(source)
        source=None
        records={}
        for lost in (-1,0,1,2):
            shifted=get_frames(env,p,seed+200000+311*(lost+1),
                               args.target_envs,args.target_steps).copy()
            if lost>=0:shifted[...,lost]=0
            detected,sds=diagnose_shift(shifted,mean)
            assert detected==lost,(lost,detected,sds)
            names=['intact_oracle','unrepaired']
            if lost>=0:names+=['constant_mean','linear','palette_lut']
            test={}
            for m in names:
                test[m]=evaluate(env,p,actor,seed+600000,
                      lost,m,mean,models,args.eval_envs,args.eval_steps)
            records[str(lost)]={'detected':detected,
                'target_channel_sds':sds,'scores':test}
            print('CHANNEL',seed,lost,'detected',detected,
                  [(name,round(v['episode_return_mean'],3)) for name,v in test.items()],
                  flush=True)
        result['trials'].append({'seed':seed,'train_seconds':tsec,
                                'color_palette_bins':[models[k]['nonzero_palette_bins'] for k in range(3)],
                                'cases':records})
        with open(args.output,'w') as f:json.dump(result,f,indent=2)
    result['walltime_seconds']=time.perf_counter()-start
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',args.output,round(result['walltime_seconds'],1),flush=True)
if __name__=='__main__':main()
