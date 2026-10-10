"""Audit whether mandatory success assertions in old Cycle 15/20
could be concealing target-sensor selection failures.
Check genuinely fresh task seeds; never drop failed identification.
"""
import json,time
from cycle12_partial_obs import make_task,sample,selected_channel,N_PRE
def main():
    report={'pre':N_PRE,'seed_start':98111026,'replications_per_nuisance_count':1000,'counts':{}}
    for nuis in [3,7,11]:
        failed=[]
        for i in range(1000):
            seed=98111026+100013*nuis+731*i
            task=make_task(seed,nuis)
            pre,py=sample(task,seed+1,N_PRE,'stationary')
            chosen=selected_channel(pre,py,task['k'])
            if chosen!=task['target_channel']:
                failed.append(dict(seed=seed,chosen=chosen,true=task['target_channel']))
        report['counts'][str(nuis)]={'failed':len(failed),'passed':1000-len(failed),'first_20_failures':failed[:20]}
        print('nuis',nuis,'failed',len(failed),flush=True)
    with open('research/cycle23_channel_audit.json','w') as f:json.dump(report,f,indent=2)
    print('SAVED')
if __name__=='__main__':main()
