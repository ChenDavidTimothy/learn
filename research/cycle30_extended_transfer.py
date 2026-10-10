"""Cycle 30: independent extrapolation across unseen POPGym Autoencode lengths.

Train two environments Easy/Medium, never inspect heldout simulator state.
Evaluate same frozen learned arithmetic expression on 3,4,6,8 decks.
This tests external extrapolation, not an independent operator family.
"""
import numpy as np,json,time
import popgym
from cycle26_memory_program_synthesis import collect_data
from cycle27_transfer_identifiability import fit

DECKS=(3,4,6,8)
def evaluate(rule,num_decks,seed,episodes=24):
    e=popgym.envs.Autoencode(num_decks=num_decks)
    returns=[]
    for ep in range(episodes):
        ob,_=e.reset(seed=seed+829*ep)
        hist=[];W=0;cum=0;done=False
        while not done:
            mode,card=int(ob[0]),int(ob[1])
            hist.append(card)
            if mode==0 and W==0:W=len(hist)
            t=len(hist)-1
            index=rule[0]*t+rule[1]*W+rule[2]
            action=int(hist[index]) if 0<=index<=t else 0
            ob,r,end,trunc,_=e.step(action)
            done=end or trunc;cum+=float(r)
        returns.append(cum)
    e.close()
    return float(np.mean(returns))
def main():
    now=time.perf_counter();n=24
    report={'protocol':{'benchmark':'POPGym Autoencode generic num_decks',
       'training':'60 Easy plus 30 Medium episodes; 6240 reward steps',
       'eval_decks':DECKS,'eval_episodes_each':24,'fresh_seeds':n,
       'no_test_specific_fitting':True},'results':{}}
    rows={str(d):[] for d in DECKS}
    rules=[]
    for i in range(n):
        seed=30781026+44119*i
        d=collect_data('AutoencodeEasy',60,seed)
        d+=collect_data('AutoencodeMedium',30,seed+239191)
        rule,meta=fit(d);rules.append(rule)
        for deck in DECKS:
            score=evaluate(rule,deck,seed+3000000,24)
            rows[str(deck)].append(score)
        if (i+1)%8==0:print('done',i+1,'elapsed',round(time.perf_counter()-now,1),flush=True)
    report['chosen_rules']=rules
    for deck in DECKS:
        x=np.array(rows[str(deck)])
        report['results'][str(deck)]={'mean_return':float(x.mean()),
           'SE':float(x.std(ddof=1)/np.sqrt(len(x)))}
    report['runtime_seconds']=time.perf_counter()-now
    with open('research/cycle30_extended_transfer_results.json','w') as f:
        json.dump(report,f,indent=2)
    print('SAVED',round(report['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
