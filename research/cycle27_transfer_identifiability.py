"""Cycle 27: identifiability of transferable temporal programs.

Same logged randomized reward budget, different episode-length diversity.
POPGym Autoencode Easy/Medium training and unseen Hard evaluation.
Restricted affine program grammar is specified by the researcher.
"""
import json,time,argparse
import numpy as np
from cycle26_memory_program_synthesis import collect_data,evaluate

COEF=np.array([(a,b,c) for a in (-1,0,1) for b in (0,1,2)
               for c in range(-325,326)],np.int32)
A,B,C=COEF.T
N=len(COEF)
def fit(seqs,tie='lexical'):
    scores=np.zeros(N,dtype=float);labels=0
    for seq in seqs:
        history=[];W=0
        for t,(mode,o,a,r) in enumerate(seq):
            history.append(o)
            if mode==0 and W==0:W=t+1
            if r==0:continue
            pos=A*t+B*W+C
            valid=(pos>=0)&(pos<=t)
            guess=np.asarray(history,dtype=np.int16)[np.clip(pos,0,t)]
            guess[~valid]=0
            scores+=4.*(guess==a)*np.sign(r)
            labels+=1
    best=np.flatnonzero(scores>=scores.max()-1e-9)
    if tie=='mdl':
        pen=np.abs(COEF[best]).sum(axis=1)
        best=best[pen==pen.min()]
    winner=int(best[0])
    return list(map(int,COEF[winner])),dict(
        labels=labels,max_ips=float(scores[winner]/labels),
        tied_programs=int(len(np.flatnonzero(scores>=scores.max()-1e-9))))
def trial(seed,n_eval):
    easy=collect_data('AutoencodeEasy',120,seed)
    medium=collect_data('AutoencodeMedium',60,seed+110001)
    groups={
      'easy_only':easy,
      'easy_only_MDL_prior':easy,
      'medium_only':medium,
      'mixed_lengths':easy[:60]+medium[:30]}
    out={}
    for name,seqs in groups.items():
        start=time.perf_counter()
        rule,stats=fit(seqs,'mdl' if name=='easy_only_MDL_prior' else 'lexical')
        secs=time.perf_counter()-start
        ret,acc=evaluate('AutoencodeHard',rule,seed+2000000,n_eval)
        out[name]=dict(rule=rule,**stats,return_hard=ret,
                       accuracy_hard=acc,training_seconds=secs)
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=40)
    p.add_argument('--eval',type=int,default=80)
    p.add_argument('--output',default='research/cycle27_identifiability_results.json')
    args=p.parse_args();t0=time.perf_counter()
    result={'protocol':{
      'benchmark':'external POPGym 1.0.6 Autoencode',
      'program_grammar':'index(t,W)=a*t+b*W+c',
      'a':[-1,0,1],'b':[0,1,2],'c':[-325,325],
      'number_programs':int(N),
      'training_groups':{
        'easy_only':'120 AutoencodeEasy episodes',
        'easy_only_MDL_prior':'same as easy, L1 coefficient penalty only for equal IPS',
        'medium_only':'60 AutoencodeMedium episodes',
        'mixed_lengths':'60 Easy + 30 Medium episodes'},
      'uniform_random_logged_actions':True,
      'equal_scored_training_steps':True,
      'test':'independent AutoencodeHard episodes',
      'train_seed_start':27771026,
      'test_episode_count':args.eval,'trials':args.trials,
      'oracle_not_used_for_training':True,
      'interpretation':'observational confounding and prior sensitivity, not novel algorithm'},'results':{}}
    raw=[]
    for i in range(args.trials):
        record=trial(27771026+33173*i,args.eval)
        raw.append(record)
        if (i+1)%10==0:
            print('done',i+1,'elapsed',round(time.perf_counter()-t0,1),
                  [(m,round(record[m]['return_hard'],3),record[m]['rule'])
                   for m in record],flush=True)
    for key in raw[0]:
        values=np.array([r[key]['return_hard'] for r in raw])
        correct=np.array([r[key]['accuracy_hard'] for r in raw])
        result['results'][key]={
            'mean_test_return':float(values.mean()),
            'SE_test_return':float(values.std(ddof=1)/np.sqrt(len(values))),
            'mean_correct_rate':float(correct.mean()),
            'selected_rules':[r[key]['rule'] for r in raw],
            'mean_training_seconds':float(np.mean([r[key]['training_seconds'] for r in raw])),
            'mean_tied_programs':float(np.mean([r[key]['tied_programs'] for r in raw])),
            'training_scored_steps':[r[key]['labels'] for r in raw]}
    result['elapsed_seconds']=time.perf_counter()-t0
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',args.output,round(result['elapsed_seconds'],1),flush=True)
if __name__=='__main__':main()
