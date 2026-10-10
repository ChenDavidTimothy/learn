"""Cycle 48: unsupervised pixel-sensor adaptation in native POPGym Arcade.

A frozen official PQN policy is trained on pixel partial-observation
NoisyCartPoleEasy. Held-out test observations undergo *artificial*
shift (identity, inverted RGB, brightening, channel permutation,
irrecoverable channel erasure), without altering actual dynamics.

Standard AdaBN-style per-channel moment alignment learns an input
normalization from heldout random-action, rewardless calibration images.
It does NOT know the target shift type. Strong controls: frozen source
policy and shift-specific true inverse (privileged, when invertible).

This tests raw-input distribution alignment, NOT autonomous invention of
latent model structure, a new adaptation algorithm, or native
nonstationarity provided by the POPGym Arcade authors.
"""
import os,sys,argparse,json,time
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
import popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from popgym_arcade.train import get_args
from popgym_arcade.baselines.pqn import make_train
jax.config.update("jax_platform_name","gpu")
SHIFTS=('identity','invert_rgb','brighten_48','permute_rgb','erase_blue')
def config_for(seed,total,task='NoisyCartPoleEasy'):
    old=sys.argv
    sys.argv=['official','PQN','--SEED',str(seed),
      '--NUM_SEEDS','1','--ENV_NAME',task,'--OBS_SIZE','128',
      '--NUM_ENVS','8','--NUM_STEPS','32','--TOTAL_TIMESTEPS',str(total),
      '--TOTAL_TIMESTEPS_DECAY',str(total),'--NUM_MINIBATCHES','4',
      '--NUM_EPOCHS','2','--HIDDEN_SIZE','64','--NUM_LAYERS','1',
      '--WANDB_MODE','disabled','--MEMORY_TYPE','MLP','--PARTIAL']
    try:cfg=vars(get_args())
    finally:sys.argv=old
    cfg['WANDB_MODE']='disabled'
    cfg['TEST_DURING_TRAINING']=False
    cfg['TEST_NUM_STEPS']=256
    cfg['TEST_NUM_ENVS']=8
    return cfg
def trained(cfg,seed):
    tick=time.perf_counter()
    train_fn=eqx.filter_jit(eqx.filter_vmap(make_train(cfg)))
    result=jax.block_until_ready(train_fn(jax.random.split(jax.random.PRNGKey(seed),1)))
    leaves=result['runner_state'][0].model
    model=jax.tree_util.tree_map(
        lambda x: x[0] if eqx.is_array(x) and x.ndim>0 and x.shape[0]==1 else x,leaves)
    return model,time.perf_counter()-tick
def change(obs,shift):
    x=obs.astype(jnp.float32)
    if shift=='identity':return x
    if shift=='invert_rgb':return 255.-x
    if shift=='brighten_48':return jnp.minimum(255.,x+48.)
    if shift=='permute_rgb':return x[...,jnp.array([2,0,1])]
    if shift=='erase_blue':
        return jnp.stack((x[...,0],x[...,1],jnp.zeros_like(x[...,2])),axis=-1)
    raise ValueError(shift)
def privileged_restore(obs,shift):
    if shift=='identity':return obs
    if shift=='invert_rgb':return 255.-obs
    if shift=='brighten_48':return jnp.maximum(0.,obs-48.)
    if shift=='permute_rgb':return obs[...,jnp.array([1,2,0])]
    if shift=='erase_blue':return obs # deliberately noninvertible
    raise ValueError(shift)
def observations_moments(env,p,seed,shift,nenv=16,nsteps=32):
    keys=jax.random.split(jax.random.PRNGKey(seed),nenv)
    obs,states=jax.vmap(env.reset,in_axes=(0,None))(keys,p)
    def step(carry,_):
        obs,states,rng,mom1,mom2,count=carry
        y=change(obs,shift)
        moment1=y.sum(axis=(0,1,2))
        moment2=jnp.square(y).sum(axis=(0,1,2))
        rng,ar,kr=jax.random.split(rng,3)
        actions=jax.random.randint(ar,(nenv,),0,5)
        k=jax.random.split(kr,nenv)
        next_obs,next_st,r,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
           k,states,actions,p)
        return (next_obs,next_st,rng,mom1+moment1,mom2+moment2,
            count+nenv*y.shape[1]*y.shape[2]),None
    zero=jnp.zeros(3,jnp.float32)
    init=(obs,states,jax.random.PRNGKey(seed+9999),zero,zero,
          jnp.array(0,jnp.int32))
    final,_=jax.lax.scan(step,init,None,length=nsteps)
    m1,m2,n=final[3],final[4],final[5]
    mean=m1/n
    sd=jnp.sqrt(jnp.maximum(m2/n-jnp.square(mean),1e-3))
    return mean,sd
def adaptation(img,shift,method,source,target):
    if method=='raw':return img
    if method=='oracle_restore':return privileged_restore(img,shift)
    if method=='moments':
        mean_s,sd_s=source;mean_t,sd_t=target
        aligned=(img-mean_t)*(sd_s/jnp.maximum(sd_t,1.))-0.+mean_s
        return jnp.clip(aligned,0,255.)
    raise ValueError(method)
def evaluate(env,p,model,seed,shift,method,source,target,
             nenv=24,nsteps=512):
    obs,states=jax.vmap(env.reset,in_axes=(0,None))(
        jax.random.split(jax.random.PRNGKey(seed),nenv),p)
    def one(carry,_):
        obs,states,rng=carry
        corrected=adaptation(change(obs,shift),shift,method,source,target)
        logits=model(corrected)
        a=logits.argmax(-1)
        rng,k=jax.random.split(rng)
        nextobs,nextstates,reward,done,info=jax.vmap(
            env.step,in_axes=(0,0,0,None))(
              jax.random.split(k,nenv),states,a,p)
        return (nextobs,nextstates,rng),(
            info['returned_episode'].astype(jnp.float32),
            info['returned_episode_returns'].astype(jnp.float32),
            reward)
    _,(done,epreturns,reward)=jax.lax.scan(one,
          (obs,states,jax.random.PRNGKey(seed+333)),None,length=nsteps)
    completed=done.sum()
    sum_returns=(done*epreturns).sum()
    return {'completed_episodes':int(completed),
       'mean_completed_episode_return':float(sum_returns/jnp.maximum(completed,1)),
       'mean_step_reward':float(reward.mean())}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--train-steps',type=int,default=131072)
    parser.add_argument('--calib-envs',type=int,default=16)
    parser.add_argument('--calib-steps',type=int,default=32)
    parser.add_argument('--eval-envs',type=int,default=24)
    parser.add_argument('--eval-steps',type=int,default=512)
    parser.add_argument('--task',default='NoisyCartPoleEasy')
    parser.add_argument('--output',default='research/cycle48_pixel_adapter_results.json')
    args=parser.parse_args();t0=time.perf_counter()
    env,p=popgym_arcade.make(args.task,partial_obs=True,obs_size=128)
    env=LogWrapper(env)
    report={'protocol':{'benchmark':'POPGym Arcade 0.0.6, actual action-dependent POMDP',
       'task':args.task,'training_algorithm':'authors PQN memory-free control, unmodified learning rules',
       'seeds':args.seeds,'train_env_steps_per_seed':args.train_steps,
       'source_calib_steps':args.calib_envs*args.calib_steps,
       'shifted_calib_steps_each_shift':args.calib_envs*args.calib_steps,
       'calib_reward_access':False,'eval_actors':args.eval_envs,
       'eval_steps_per_actor':args.eval_steps,
       'transformations':SHIFTS,
       'source_model_held_frozen':True,
       'adapter':'AdaBN-style unlabeled per-channel mean and standard-deviation alignment',
       'privileged':'shift_specific_true_inverse knows shift, including irreversible erase control',
       'no_new_algorithm':True,'no_mid_episode_unknown_onset':True},'results':[]}
    for i in range(args.seeds):
        seed=48111026+i*11291
        cfg=config_for(seed,args.train_steps,args.task)
        model,train_seconds=trained(cfg,seed)
        print('TRAINED',i+1,'/',args.seeds,'seconds',round(train_seconds,1),flush=True)
        source=observations_moments(env,p,seed+510000,'identity',
                                 args.calib_envs,args.calib_steps)
        cases={}
        for j,shift in enumerate(SHIFTS):
            target=observations_moments(env,p,seed+700000+j*173,shift,
                                args.calib_envs,args.calib_steps)
            cases[shift]={}
            for method in ('raw','moments','oracle_restore'):
                tick=time.perf_counter()
                res=evaluate(env,p,model,seed+900000,shift,method,source,target,
                           args.eval_envs,args.eval_steps)
                res['evaluation_seconds']=time.perf_counter()-tick
                cases[shift][method]=res
            print('SHIFT',seed,shift,[(m,round(v['mean_completed_episode_return'],3),
                       v['completed_episodes']) for m,v in cases[shift].items()],flush=True)
        report['results'].append({'seed':seed,'train_seconds':train_seconds,'cases':cases})
        with open(args.output,'w') as f:json.dump(report,f,indent=2)
    report['total_seconds']=time.perf_counter()-t0
    with open(args.output,'w') as f:json.dump(report,f,indent=2)
    print('SAVED',args.output,round(report['total_seconds'],1),flush=True)
if __name__=='__main__':main()
