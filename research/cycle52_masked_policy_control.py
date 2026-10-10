"""Cycle 52: strong ordinary masked-training control for failed RGB sensors.

Use authors' POPGym Arcade PQN algorithm with observational augmentation:
each training episode independently has no channel failure, or R/G/B
channel zeroed, with 25% probability each. This is established random
sensor-dropout training, similar to Skand et al. CoRL 2025, and is a
strong alternative to postfailure conditional color reconstruction.

Compare actual live policy returns on heldout NoisyCartPoleEasy with any
one channel erased, and the no-dropout case, at the SAME 65,536 online
RL interaction budget as the source policies of Cycle 50. Source policy
training in Cycle 50 does not see masking but receives unlabeled source
images for a post-hoc imputer; masked policy needs prior knowledge of
the possible dropout family. Privileges and calibration cost differ.
"""
import argparse,json,time,sys
from typing import NamedTuple
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
import popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from popgym_arcade.baselines.pqn import make_train
from cycle48_arcade_observation_adapter import config_for
jax.config.update('jax_platform_name','gpu')
class FailureState(NamedTuple):
    inner: object
    missing: jax.Array
class MissingChannelEnv:
    def __init__(self,env):self.env=env
    def __getattr__(self,name):return getattr(self.env,name)
    @staticmethod
    def masked(image,missing):
        keep=jnp.logical_or(jnp.arange(3)!=missing,missing<0)
        return image*keep.astype(image.dtype)
    def reset(self,key,params):
        k1,k2=jax.random.split(key)
        obs,state=self.env.reset(k1,params)
        # -1 = intact, 0/1/2 = erased R/G/B
        idx=jax.random.randint(k2,(),0,4)-1
        return self.masked(obs,idx), FailureState(state,idx)
    def step(self,key,state,action,params):
        ks,ki=jax.random.split(key)
        obs,inner,reward,done,info=self.env.step(ks,state.inner,action,params)
        idx=jnp.where(done,jax.random.randint(ki,(),0,4)-1,state.missing)
        return self.masked(obs,idx),FailureState(inner,idx),reward,done,info
def learn(seed,total,task):
    cfg=config_for(seed,total,task)
    original=popgym_arcade.make
    def patched(name,**kwargs):
        env,params=original(name,**kwargs)
        return MissingChannelEnv(env),params
    popgym_arcade.make=patched
    try:fn=make_train(cfg)
    finally:popgym_arcade.make=original
    begin=time.perf_counter()
    output=jax.block_until_ready(eqx.filter_jit(eqx.filter_vmap(fn))(
        jax.random.split(jax.random.PRNGKey(seed),1)))
    model=output['runner_state'][0].model
    model=jax.tree_util.tree_map(
      lambda x:x[0] if eqx.is_array(x) and x.ndim>0 and x.shape[0]==1 else x,
      model)
    return model,time.perf_counter()-begin
def evaluate(env,p,actor,seed,missing,n_envs,n_steps):
    obs,states=jax.vmap(env.reset,in_axes=(0,None))(
        jax.random.split(jax.random.PRNGKey(seed),n_envs),p)
    def step(carry,_):
        ob,st,rng=carry
        inp=MissingChannelEnv.masked(ob,missing)
        a=actor(inp).argmax(-1)
        rng,k=jax.random.split(rng)
        nextob,nextst,reward,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(k,n_envs),st,a,p)
        return (nextob,nextst,rng),(
          info['returned_episode'].astype(jnp.float32),
          info['returned_episode_returns'].astype(jnp.float32),reward)
    _,(done,episodic,r)=jax.lax.scan(step,
        (obs,states,jax.random.PRNGKey(seed+333)),None,length=n_steps)
    n=float(done.sum())
    return {'completed_episodes':int(n),'mean_return':float(
            (done*episodic).sum()/max(n,1.)),
            'mean_step_reward':float(r.mean())}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--train-steps',type=int,default=65536)
    parser.add_argument('--eval-envs',type=int,default=32)
    parser.add_argument('--eval-steps',type=int,default=512)
    parser.add_argument('--task',default='NoisyCartPoleEasy')
    parser.add_argument('--output',default='research/cycle52_masked_policy_results.json')
    a=parser.parse_args();begin=time.perf_counter()
    env,p=popgym_arcade.make(a.task,partial_obs=True,obs_size=128)
    env=LogWrapper(env)
    result={'protocol':{'benchmark':'POPGym Arcade 0.0.6 pixel partial control',
      'task':a.task,'algorithm':'official PQN, observation mask augmentation only',
      'training_episodes_steps':a.train_steps,
      'mask_distribution':'per training episode independently intact/R/G/B erased each with 25% probability',
      'source_train_rights':'trained on all three known missing-channel corruption families, unlike post-failure source model',
      'eval_actors':a.eval_envs,'eval_steps_per_actor':a.eval_steps,
      'training_seeds':a.seeds,'no_test_adaptation_or_calibration':True,
      'method_class':'standard channel dropout domain randomization, not novel',
      'closest_reference':'Skand et al. 2025 Simple Masked Training Strategies Yield Control Policies That Are Robust to Sensor Failure',
      'evaluation':'independent native simulator trajectories, intact/masked raw RGB'},'runs':[]}
    for i in range(a.seeds):
        seed=52111026+11173*i
        actor,seconds=learn(seed,a.train_steps,a.task)
        trials={}
        for missing in (-1,0,1,2):
            trials[str(missing)]=evaluate(env,p,actor,seed+700000,
                   missing,a.eval_envs,a.eval_steps)
        result['runs'].append({'seed':seed,'train_seconds':seconds,'scores':trials})
        print('RUN',seed,'train_seconds',round(seconds,2),[
           (k,round(v['mean_return'],3)) for k,v in trials.items()],flush=True)
        with open(a.output,'w') as f:json.dump(result,f,indent=2)
    result['wall_seconds']=time.perf_counter()-begin
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',a.output,round(result['wall_seconds'],1),flush=True)
if __name__=='__main__':main()
