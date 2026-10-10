"""Cycle 28: how little intervention diversity resolves program aliasing?

Keep number of rewarded learning decisions exactly fixed:
120 easy episodes = (120-2*m) easy + m medium episodes,
as medium contributes twice as many rewarded decisions.
Compare held-out Hard return with different m. No test data used to fit.
This is classical experiment design/persistent excitation, not novelty.
"""
import json,argparse,time
import numpy as np
from cycle26_memory_program_synthesis import collect_data,evaluate
from cycle27_transfer_identifiability import fit
M=[0,1,2,5,10,20,30]
def trial(seed,n_eval):
    easy=collect_data('AutoencodeEasy',120,seed)
    medium=collect_data('AutoencodeMedium',30,seed+229191)
    out={}
    for m in M:
        train=easy[:120-2*m]+medium[:m]
        rule,details=fit(train)
        result,accuracy=evaluate('AutoencodeHard',rule,seed+2000000,n_eval)
        out[str(m)]={'rule':rule,'test_return':result,'accuracy':accuracy,
                     'labels':details['labels'],'ties':details['tied_programs']}
        assert details['labels']==6240,details
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=48)
    p.add_argument('--test',type=int,default=40)
    p.add_argument('--output',default='research/cycle28_diversity_results.json')
    a=p.parse_args();now=time.perf_counter()
    out={'protocol':{'fixed_reward_labels':6240,'environments':'POPGym Autoencode Easy and Medium train; Hard heldout',
          'medium_episodes_m':M,'easy_episodes': [120-2*m for m in M],
          'novel_trial_seed_start':28111026,
          'experimenter_supplied_grammar':True,
          'data_diversity_varied_with_identical_reward_label_count':True,
          'test_episodes':a.test,'trials':a.trials},'conditions':{}}
    raw=[]
    for i in range(a.trials):
        r=trial(28111026+91793*i,a.test)
        raw.append(r)
        if (i+1)%12==0:
            print('finished',i+1,'/',a.trials,'elapsed',round(time.perf_counter()-now,1),
                  [(m,round(r[str(m)]['test_return'],3),r[str(m)]['rule']) for m in M],flush=True)
    for m in M:
        rs=[q[str(m)] for q in raw]
        x=np.array([r['test_return'] for r in rs])
        out['conditions'][str(m)]={'mean_return':float(x.mean()),
            'SE':float(x.std(ddof=1)/np.sqrt(len(x))),
            'correct_rule_rate':float(np.mean([r['rule']==[-1,2,-2] for r in rs])),
            'mean_number_tied':float(np.mean([r['ties'] for r in rs])),
            'selected_rules':[r['rule'] for r in rs]}
    out['total_seconds']=time.perf_counter()-now
    with open(a.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',a.output,round(out['total_seconds'],1),flush=True)
if __name__=='__main__':main()
