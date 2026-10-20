#!/usr/bin/env python3
"""Independent prescribed q1-orbit/paired-risk diagnostic from retained commands."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'coupled_learning_v14_results.json').read_text());conditions=[]
    labels=('constant_forward','context_forward','naive_inverse','corrected_inverse','oracle_endpoint')
    for condition in (0,1):
        rows=[r for r in data['runs'] if r['condition']==condition];reports={}
        paired=np.array([np.mean([t['controllers']['naive_inverse']['squared_error']-
            t['controllers']['corrected_inverse']['squared_error'] for t in r['tasks']]) for r in rows])
        for label in labels:
            per_seed=[]
            for r in rows:
                groups={}
                for t in r['tasks']:
                    key=(t['known_preparation'][1],*t['desired_normalized_displacement'])
                    groups.setdefault(key,[]).append(t['controllers'][label]['command'])
                assert len(groups)==40 and all(len(u)==4 for u in groups.values())
                centered=[];max_pair=0.
                for u in groups.values():
                    values=np.array(u);centered.extend(np.sum((values-values.mean(axis=0))**2,axis=1))
                    max_pair=max(max_pair,float(np.max(np.linalg.norm(values[:,None,:]-values[None,:,:],axis=2))))
                per_seed.append({'seed':r['seed'],'mean_squared_within_q1_orbit_variation':float(np.mean(centered)),
                                 'maximum_q1_pair_command_distance':max_pair})
            reports[label]={'RMS_within_q1_orbit_command_variation':float(np.sqrt(np.mean([
                r['mean_squared_within_q1_orbit_variation'] for r in per_seed]))),
                'mean_seed_componentwise_RMS_q1_variation':float(np.mean([np.sqrt(r['mean_squared_within_q1_orbit_variation']/2) for r in per_seed])),
                'max_q1_pair_command_distance':max(r['maximum_q1_pair_command_distance'] for r in per_seed),
                'per_seed':per_seed}
        conditions.append({'condition':condition,'models':reports,'naive_minus_corrected_task_MSE':{
            'per_seed':paired.tolist(),'mean':float(paired.mean()),
            'clustered_mc_se':float(paired.std(ddof=1)/np.sqrt(len(paired)))}})
    print(json.dumps({'scope':'Prescribed q1 invariance diagnostic; actual retained commands, '
        'same q2/goal across4q1; no new learner or physical experiments',
        'definition':'Pooled EuclideanRMS=sqrt(mean_seed,context,goal,q1 ||u-mean_q1u||^2); alternative=mean_seed sqrt(mean_context,goal,q1,component (u-mean_q1u)^2). Nonlinear pooling can change crossmodel order; neither denotes exact invariance.',
        'conditions':conditions},indent=2,allow_nan=False))


if __name__=='__main__':main()
