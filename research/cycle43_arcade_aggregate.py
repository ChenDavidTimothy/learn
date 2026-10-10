"""Aggregate Phase VI official POPGym Arcade short-budget pilot.

Three nominal training seeds per environment/algorithm, 32,768 real
environment interactions each. Test RNG streams are implementation-specific
and NOT synchronized across algorithms. This is a feasibility/stability
audit, *not* a statistically powered or compute-matched SOTA comparison.
"""
import json,statistics,math
from pathlib import Path
TASKS=('CountRecallEasy','NoisyCartPoleEasy')
METHODS=('PQN','PQN_RNN')
SEEDS=(43221026,43221027,43221028)
def path(task,m,seed):
    if seed==SEEDS[0]:return Path(f'research/cycle43_{task}_{m}_pilot.json')
    return Path(f'research/cycle43_{task}_{m}_{seed}.json')
def summarize(a):
    if len(a)<2:return {'mean':float(statistics.mean(a)),'SE':None}
    return {'mean':float(statistics.mean(a)),
            'SE':float(statistics.stdev(a)/math.sqrt(len(a)))}
def main():
    summary={'protocol':{
       'source':'authors official popgym-arcade==0.0.6 training implementation',
       'baseline_methods':METHODS,'tasks':TASKS,'seeds':SEEDS,
       'env_steps_per_seed':32768,'partial_obs':True,'pixel_size':128,
       'parallel_environments':8,'steps_per_update':32,
       'update_epochs':2,'minibatches':4,'eval_environments':8,
       'evaluation_steps':256,'test_interval_fraction_of_updates':.25,
       'gpu':'JAX CUDA13 RTX3060Ti 8GiB',
       'metric':'final fresh-rollout mean returned_episode_returns from official test function',
       'caution':'short undertrained pilot; no test RNG pairing; no tuned learning-rate sweeps',
       'diagnostic_patch':'one-line zeroing of invalid RNN action-churn logging only',
       'not_novel':True},'tasks':{}}
    for task in TASKS:
        bymethod={}
        for m in METHODS:
            rows=[]
            for seed in SEEDS:
                f=path(task,m,seed)
                assert f.exists(),f
                row=json.load(f.open())
                assert row['task']==task and row['algorithm']==m
                assert row['total_env_steps']==32768 and row['partial_obs'] and row['heldout_test_final'] is not None
                rows.append({'seed':seed,'heldout_test_final':row['heldout_test_final'],
                             'wallclock_seconds':row['wallclock_seconds']})
            bymethod[m]={'seed_results':rows,
                'heldout_final':summarize([q['heldout_test_final'] for q in rows]),
                'wallclock_seconds':summarize([q['wallclock_seconds'] for q in rows])}
        summary['tasks'][task]=bymethod
        print(task,[(m,round(bymethod[m]['heldout_final']['mean'],4),
                          round(bymethod[m]['heldout_final']['SE'],4)) for m in METHODS],flush=True)
    with open('research/cycle43_arcade_pilot_summary.json','w') as f:json.dump(summary,f,indent=2)
    print('SAVED')
if __name__=='__main__':main()
