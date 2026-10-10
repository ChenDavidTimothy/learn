"""Cycle 43: official POPGym Arcade benchmark training audit.

Calls the published package's own PQN and recurrent PQN implementation.
This is a reproducibility and feasibility pilot, NOT evidence of a new
algorithm. Uses JAX CPU in isolated .venv_arcade and tracks on-training
completed-episode returns and, if enabled, periodic heldout tests.

Published benchmark: https://github.com/bolt-research/popgym-arcade
Any resulting scores at 20k steps are *undertrained pilots*, NOT SOTA.
"""
import argparse,os,sys,time,json,importlib.metadata
import numpy as np
import jax,equinox as eqx
from popgym_arcade.train import get_args

def summarize_curves(values,complete):
    values=np.asarray(values,float)
    done=np.asarray(complete,float)
    if values.ndim==2:values=values[0]
    if done.ndim==2:done=done[0]
    out={}
    for name,sl in [('early',slice(0,max(1,len(values)//4))),
                    ('late',slice(max(0,3*len(values)//4),len(values)))]:
        w=done[sl]
        out[name]={'completed_episode_count_effective':float(w.sum()),
           'weighted_mean_return':float(values[sl].sum()/w.sum()) if w.sum()>0 else None}
    return out

def run(algorithm,task,seed,total,partial,obs_size=128,parallel_envs=8,rollout=32,epochs=2,minibatches=4,periodic_test=True,test_envs=8):
    old=sys.argv
    # Use same official parser defaults, overriding only declared controls.
    params=[
       algorithm,'--SEED',str(seed),'--NUM_SEEDS','1',
       '--ENV_NAME',task,'--OBS_SIZE',str(obs_size),
       '--NUM_ENVS',str(parallel_envs),'--NUM_STEPS',str(rollout),
       '--TOTAL_TIMESTEPS',str(total),
       '--TOTAL_TIMESTEPS_DECAY',str(max(total,2048)),
       '--NUM_MINIBATCHES',str(minibatches),'--NUM_EPOCHS',str(epochs),
       '--HIDDEN_SIZE','64','--NUM_LAYERS','1',
       '--WANDB_MODE','disabled','--MEMORY_TYPE',
       'LRU' if algorithm=='PQN_RNN' else 'MLP']
    if partial:params.append('--PARTIAL')
    sys.argv=['popgym_arcade.train']+params
    try:cfg=vars(get_args())
    finally:sys.argv=old
    cfg['TEST_DURING_TRAINING']=periodic_test
    cfg['TEST_INTERVAL']=.25
    cfg['TEST_NUM_ENVS']=test_envs
    cfg['TEST_NUM_STEPS']=256
    cfg['EPS_TEST']=0.0
    cfg['WANDB_MODE']='disabled'
    if algorithm=='PQN':
        from popgym_arcade.baselines.pqn import make_train
    else:
        from popgym_arcade.baselines.pqn_rnn import make_train
    begin=time.perf_counter()
    results=jax.block_until_ready(eqx.filter_jit(eqx.filter_vmap(make_train(cfg)))(
        jax.random.split(jax.random.PRNGKey(seed),1)))
    elapsed=time.perf_counter()-begin
    metrics=results['metrics']
    result={'algorithm':algorithm,'memory_type':cfg['MEMORY_TYPE'],'task':task,
       'seed':seed,'total_env_steps':total,'wallclock_seconds':elapsed,
       'partial_obs':partial,'obs_size':obs_size,
       'source':'official popgym-arcade 0.0.6 published package training implementation',
       'jax_platform':str(jax.devices()[0]),
       'training_config':{'NUM_ENVS':parallel_envs,'NUM_STEPS':rollout,'NUM_EPOCHS':epochs,
           'NUM_MINIBATCHES':minibatches,'TEST_NUM_ENVS':test_envs,'TEST_DURING_TRAINING':periodic_test},
       'train_metrics_keys':list(metrics.keys()),'return_means':None}
    if 'returned_episode_returns' in metrics and 'returned_episode' in metrics:
        result['return_means']=summarize_curves(metrics['returned_episode_returns'],
                                                metrics['returned_episode'])
    else:
        result['available_metrics']={key:float(np.asarray(v).mean()) for key,v in metrics.items()
                  if np.asarray(v).size and np.isfinite(np.asarray(v)).all()}
    if 'test_returned_episode_returns' in metrics:
        zz=np.asarray(metrics['test_returned_episode_returns'],float)
        if zz.ndim==2:zz=zz[0]
        result['heldout_test_return_series']=zz.tolist()
        result['heldout_test_final']=float(zz[-1]) if np.isfinite(zz[-1]) else None
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=['PQN','PQN_RNN'],default='PQN')
    p.add_argument('--task',default='NoisyCartPoleEasy')
    p.add_argument('--seed',type=int,default=43211026)
    p.add_argument('--total',type=int,default=20480)
    p.add_argument('--parallel-envs',type=int,default=8)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--test-envs',type=int,default=8)
    p.add_argument('--no-test',action='store_true')
    p.add_argument('--partial',action='store_true')
    p.add_argument('--output',default='research/cycle43_arcade_pilot.json')
    args=p.parse_args()
    outcome=run(args.algorithm,args.task,args.seed,args.total,args.partial,
          parallel_envs=args.parallel_envs,epochs=args.epochs,rollout=args.rollout,
          minibatches=args.minibatches,test_envs=args.test_envs,periodic_test=not args.no_test)
    with open(args.output,'w') as f:json.dump(outcome,f,indent=2)
    print('RESULT',outcome,flush=True)
if __name__=='__main__':main()
