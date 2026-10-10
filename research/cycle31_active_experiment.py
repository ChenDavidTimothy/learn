"""Cycle 31: autonomous choice of an informative training environment.

Given a researcher-supplied candidate affine index grammar, infer the
empirically tied version space from logged easy episodes. Select the next
task length using *only predicted action/index disagreement* across programs,
without reward oracle or knowledge of correct target expression.
Compare with known, standard information/disagreement experiment design.
This is active version-space learning, NOT a new algorithmic principle.
"""
import json,time,argparse
import numpy as np
from cycle26_memory_program_synthesis import collect_data,evaluate
from cycle27_transfer_identifiability import A,B,C,COEF,fit
N=len(A)
CHOICES=(52,104)
def version_space(seqs):
    scores=np.zeros(N);count=0
    for seq in seqs:
        hist=[];W=0
        for t,(mode,o,act,r) in enumerate(seq):
            hist.append(o)
            if mode==0 and W==0:W=t+1
            if r==0:continue
            idx=A*t+B*W+C
            valid=(idx>=0)&(idx<=t)
            action=np.asarray(hist,np.int16)[np.clip(idx,0,t)]
            action[~valid]=0
            scores+=4.*(action==act)*np.sign(r)
            count+=1
    ix=np.flatnonzero(scores>=scores.max()-1e-9)
    return COEF[ix],len(ix)
def distinguish(versions,W):
    # Each candidate's predicted observable memory index on the proposed
    # length; compares programs only, never real correct actions or reward.
    t=np.arange(W-1,2*W-1,dtype=int)
    idx=(versions[:,0,None]*t[None,:]+
         versions[:,1,None]*W+versions[:,2,None])
    idx=np.where((idx>=0)&(idx<=t[None,:]),idx,-1)
    different=0.;pairs=0
    for i in range(len(idx)):
        for j in range(i+1,len(idx)):
            different+=float(np.mean(idx[i]!=idx[j]))
            pairs+=1
    return different/max(1,pairs)
def choose(versions):
    values={W:distinguish(versions,W)/(2*W-1) for W in CHOICES}
    return max(values,key=values.get),values
def one(seed,ntest):
    easy=collect_data('AutoencodeEasy',120,seed)
    med=collect_data('AutoencodeMedium',1,seed+324510)
    version,n=version_space(easy[:118])
    chosen,values=choose(version)
    active_train=easy[:118]+(med if chosen==104 else easy[118:])
    selected,meta=fit(active_train)
    hard_value,_=evaluate('AutoencodeHard',selected,seed+1000000,ntest)
    return {'version_size':n,'scores':values,'selected_length':chosen,
           'chosen_rule':selected,'hard_return':hard_value,
           'reward_labels':meta['labels']}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=48)
    p.add_argument('--test',type=int,default=80)
    p.add_argument('--output',default='research/cycle31_active_experiment_results.json')
    args=p.parse_args();now=time.perf_counter()
    out={'protocol':{'algorithm':'version-space index disagreement per cost',
          'benchmark':'POPGym Autoencode',
          'choices':'next one Medium or two Easy episodes',
          'rewarded_train_steps':6240,'predata':'118 Easy episodes',
          'prior':'researcher-defined affine expression grammar',
          'candidate_lengths':[52,104],'test':'AutoencodeHard',
          'seed_start':31111026,'trials':args.trials},'results':{}}
    rows=[]
    for i in range(args.trials):
        r=one(31111026+77117*i,args.test)
        rows.append(r)
        if (i+1)%12==0:print('finished',i+1,'/',args.trials,r,flush=True)
    rewards=np.array([r['hard_return'] for r in rows])
    out['results']={
        'fraction_selected_medium':float(np.mean([r['selected_length']==104 for r in rows])),
        'mean_test_return':float(rewards.mean()),
        'SE_test_return':float(rewards.std(ddof=1)/np.sqrt(len(rewards))),
        'correct_formula_fraction':float(np.mean([r['chosen_rule']==[-1,2,-2] for r in rows])),
        'mean_version_size':float(np.mean([r['version_size'] for r in rows])),
        'sample_scores':rows[0]['scores'],
        'selected_rules':[r['chosen_rule'] for r in rows]}
    out['runtime_seconds']=time.perf_counter()-now
    with open(args.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',args.output,round(out['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
