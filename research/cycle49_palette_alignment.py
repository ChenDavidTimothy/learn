"""Cycle 49: automatic signed-channel/affine sensor recovery from unlabeled RGB.

Follows Cycle 48 where ordinary positive-scale channel moment matching
fixed brightness changes but NOT global inversion or RGB permutation.
Unlike Cycle 48's privileged oracle, this standard group-search adapter
does NOT know the applied shift. It has a *researcher-supplied family*
of 48 signed RGB permutations and estimates per-channel affine offsets
from unrelated unlabeled random-action calibration trajectories.

Optimize a fixed source-vs-target spatial-mean + skewness discrepancy,
not rewards, correct actions, model loss or shift labels. Compare to the
original strong frozen model, plain per-channel moments, and privileged
shift-specific inverse. This is established sensor normalization / domain
alignment, NOT autonomous invention of an unrestricted transformation.
"""
import argparse,json,time,itertools
import numpy as np,jax,jax.numpy as jnp
import popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from cycle48_arcade_observation_adapter import (
    trained,config_for,change,SHIFTS,evaluate,privileged_restore)
jax.config.update('jax_platform_name','gpu')
def calibration(env,p,seed,shift,nenv,steps):
    keys=jax.random.split(jax.random.PRNGKey(seed),nenv)
    obs,state=jax.vmap(env.reset,in_axes=(0,None))(keys,p)
    def next_step(carry,_):
        ob,st,rng,img_sum,s1,s2,s3,n=carry
        x=change(ob,shift).astype(jnp.float32)
        img_sum+=x.sum(axis=0)
        s1+=x.sum(axis=(0,1,2))
        s2+=jnp.square(x).sum(axis=(0,1,2))
        s3+=jnp.power(x,3).sum(axis=(0,1,2))
        n+=nenv*x.shape[1]*x.shape[2]
        rng,ra,rs=jax.random.split(rng,3)
        a=jax.random.randint(ra,(nenv,),0,5)
        new,st,r,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(rs,nenv),st,a,p)
        return (new,st,rng,img_sum,s1,s2,s3,n),None
    init=(obs,state,jax.random.PRNGKey(seed+1711),
          jnp.zeros(obs.shape[1:],jnp.float32),
          jnp.zeros(3,jnp.float32),jnp.zeros(3,jnp.float32),
          jnp.zeros(3,jnp.float32),jnp.array(0,jnp.int32))
    finished,_=jax.lax.scan(next_step,init,None,length=steps)
    image=np.asarray(finished[3]/(nenv*steps))
    n=float(finished[7])
    mu=np.asarray(finished[4]/n);second=np.asarray(finished[5]/n)
    sd=np.sqrt(np.maximum(second-mu**2,1e-3))
    third=np.asarray(finished[6]/n)
    central3=third-3*mu*second+2*mu**3
    skew=central3/(sd**3)
    return {'spatial':image,'mean':mu,'sd':sd,'skew':skew}
def align_coefficients(src,tgt):
    # Fixed architecture-agnostic group search; no shift labels.
    best=None
    ref=(src['spatial']-src['mean'])/src['sd']
    for perm in itertools.permutations((0,1,2)):
        perm=np.array(perm,dtype=int)
        standardized=(tgt['spatial'][...,perm]-tgt['mean'][perm])/tgt['sd'][perm]
        for signs in itertools.product((-1,1),repeat=3):
            s=np.array(signs)
            expected_skew=s*tgt['skew'][perm]
            template=float(np.mean((ref-standardized*s)**2))
            skew=float(np.mean((src['skew']-expected_skew)**2))
            criterion=template+0.25*skew
            if best is None or criterion<best[0]:
                best=(criterion,perm.tolist(),list(signs),template,skew)
    return {'fit_objective':best[0],'permutation':best[1],
      'signs':best[2],'spatial_loss':best[3],'skew_loss':best[4],
      'source_mean':src['mean'].tolist(),'source_sd':src['sd'].tolist(),
      'target_mean':tgt['mean'].tolist(),'target_sd':tgt['sd'].tolist()}
def eval_palette(env,p,model,seed,shift,coeff,nenv=24,nsteps=512):
    perm=jnp.asarray(coeff['permutation'])
    signs=jnp.asarray(coeff['signs'],jnp.float32)
    mus=jnp.asarray(coeff['source_mean'])
    sds=jnp.asarray(coeff['source_sd'])
    mut=jnp.asarray(coeff['target_mean'])[perm]
    sdt=jnp.asarray(coeff['target_sd'])[perm]
    obs,states=jax.vmap(env.reset,in_axes=(0,None))(
        jax.random.split(jax.random.PRNGKey(seed),nenv),p)
    def step(carry,_):
        ob,st,rng=carry
        seen=change(ob,shift).astype(jnp.float32)
        canonical=jnp.clip(
            mus+signs*(seen[...,perm]-mut)*sds/jnp.maximum(sdt,1.),0,255.)
        choice=model(canonical).argmax(-1)
        rng,k=jax.random.split(rng)
        nextob,nextst,r,done,info=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(k,nenv),st,choice,p)
        return (nextob,nextst,rng),(
            info['returned_episode'].astype(jnp.float32),
            info['returned_episode_returns'].astype(jnp.float32),r)
    _,(done,ret,rew)=jax.lax.scan(step,
           (obs,states,jax.random.PRNGKey(seed+333)),None,length=nsteps)
    completed=done.sum()
    return {'completed_episodes':int(completed),
       'mean_completed_episode_return':float(
             (done*ret).sum()/jnp.maximum(completed,1)),
       'mean_step_reward':float(rew.mean())}
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--seeds',type=int,default=3)
    pa.add_argument('--train-steps',type=int,default=65536)
    pa.add_argument('--calib-envs',type=int,default=16)
    pa.add_argument('--calib-steps',type=int,default=32)
    pa.add_argument('--eval-envs',type=int,default=32)
    pa.add_argument('--eval-steps',type=int,default=512)
    pa.add_argument('--task',default='NoisyCartPoleEasy')
    pa.add_argument('--output',default='research/cycle49_palette_alignment_results.json')
    a=pa.parse_args();started=time.perf_counter()
    env,p=popgym_arcade.make(a.task,partial_obs=True,obs_size=128)
    env=LogWrapper(env)
    out={'protocol':{'source':'native action-dependent POPGym Arcade 0.0.6',
        'task':a.task,'training':'authors official PQN, no learning update modifications',
        'source_train_steps':a.train_steps,
        'calibration_steps_per_domain':a.calib_envs*a.calib_steps,
        'calibration_rewards_used':False,'eval_envs':a.eval_envs,
        'eval_steps':a.eval_steps,'training_seeds':a.seeds,
        'source_shift_candidate_family':'signed RGB channel permutation, per-channel affine transform, 48 structural choices + estimated scale/offset',
        'selection':'unlabeled source-vs-target spatial mean template + skewness mismatch, fixed coefficient 0.25; no reward/test tuning',
        'reference':'identity raw policy, ordinary per-channel moments, privileged exact shift inverse',
        'no_shift_class_supplied_to_selector':True,
        'candidate_transform_group_researcher_supplied':True,
        'irrecoverable_negative_control':'channel erased cannot be reconstructed from RGB image alone',
        'novelty':'established image standardization/group alignment, no new algorithm',
        'external_shift':'artificial after native simulator render; NOT authors nonstationary benchmark'},'runs':[]}
    for ii in range(a.seeds):
        seed=49111026+21131*ii
        model,seconds=trained(config_for(seed,a.train_steps,a.task),seed)
        print('TRAINED',seed,'seconds',round(seconds,1),flush=True)
        source=calibration(env,p,seed+100000,'identity',a.calib_envs,a.calib_steps)
        src_t=(jnp.asarray(source['mean']),jnp.asarray(source['sd']))
        cases={}
        for si,shift in enumerate(SHIFTS):
            target=calibration(env,p,seed+300000+173*si,shift,
                                    a.calib_envs,a.calib_steps)
            target_t=(jnp.asarray(target['mean']),jnp.asarray(target['sd']))
            params=align_coefficients(source,target)
            checks={}
            for method in ('raw','moments','oracle_restore'):
                checks[method]=evaluate(env,p,model,seed+700000,
                     shift,method,src_t,target_t,a.eval_envs,a.eval_steps)
            checks['spatial_signed_align']=eval_palette(env,p,model,
                  seed+700000,shift,params,a.eval_envs,a.eval_steps)
            cases[shift]={'selected_transform':params,'scores':checks}
            print('SHIFT',seed,shift,
                  'signs',params['signs'],'perm',params['permutation'],
                  [(m,round(v['mean_completed_episode_return'],3))
                   for m,v in checks.items()],flush=True)
        out['runs'].append({'seed':seed,'training_seconds':seconds,'cases':cases})
        with open(a.output,'w') as f:json.dump(out,f,indent=2)
    out['seconds']=time.perf_counter()-started
    with open(a.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',a.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':main()
