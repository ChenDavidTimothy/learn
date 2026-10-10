"""Cycle 29 robustness: randomized reward sign measurement errors, not native POPGym."""
import json
import numpy as np
from cycle26_memory_program_synthesis import collect_data,evaluate
from cycle27_transfer_identifiability import fit
NOISE_LEVELS=(0.,.2,.4,.45)
MIX_COUNTS=(0,1,5,20,30)
def logged_measurements(seqs,rng,error_probability):
    if error_probability==0:return seqs
    changed=[]
    for seq in seqs:
        fixed=[]
        for mode,o,a,r in seq:
            if r and rng.random()<error_probability:r=-r
            fixed.append((mode,o,a,r))
        changed.append(fixed)
    return changed

def trial(seed,neval):
    easy0=collect_data('AutoencodeEasy',120,seed)
    medium0=collect_data('AutoencodeMedium',30,seed+229191)
    output={}
    for noise in NOISE_LEVELS:
        rg=np.random.default_rng(seed+3471777+round(noise*100))
        easy=logged_measurements(easy0,rg,noise)
        medium=logged_measurements(medium0,rg,noise)
        for m in MIX_COUNTS:
            dataset=easy[:120-2*m]+medium[:m]
            rule,meta=fit(dataset)
            assert meta['labels']==6240
            ret,acc=evaluate('AutoencodeHard',rule,seed+2000000,neval)
            output[str((noise,m))]={'rule':rule,'return':ret,'accuracy':acc}
    return output
def main():
    import argparse,time
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--eval',type=int,default=30)
    p.add_argument('--output',default='research/cycle29_noise_diversity_results.json')
    a=p.parse_args();begin=time.perf_counter()
    out={'protocol':{'benchmark':'POPGym 1.0.6 Autoencode',
          'measurement_noise':'training reward signs independently flipped, native benchmark is deterministic',
          'flip_probability':NOISE_LEVELS,'medium_episode_counts':MIX_COUNTS,
          'fixed_reward_bearing_training_steps':6240,'train_policy':'uniform random',
          'heldout':'clean AutoencodeHard',
          'trials':a.trials,'eval_episodes':a.eval,'seed_start':29111026},'results':{}}
    rows=[]
    for i in range(a.trials):
        rows.append(trial(29111026+71113*i,a.eval))
        if (i+1)%8==0:
            print('TRIAL',i+1,'elapsed',round(time.perf_counter()-begin,1),flush=True)
    for noise in NOISE_LEVELS:
        for m in MIX_COUNTS:
            key=str((noise,m));scores=np.array([v[key]['return'] for v in rows])
            exact=np.array([v[key]['rule']==[-1,2,-2] for v in rows])
            out['results'][key]={'noise':noise,'medium_episodes':m,
              'exact_rate':float(exact.mean()),'return_mean':float(scores.mean()),
              'return_SE':float(scores.std(ddof=1)/np.sqrt(len(scores)))}
    out['seconds']=time.perf_counter()-begin
    with open(a.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',a.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':main()
