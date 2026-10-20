#!/usr/bin/env python3
"""Prescribed same-family diagnostic. Never tune from held-out true outcomes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from shared_experience_v12 import train,direct_episode,targets,score,TEST_GOALS,MOTOR,stat


def prefix_cost(s,u,g):
    s=np.asarray(s)[1:71];u=np.asarray(u)[:70];g=targets(g)[:70]
    return float(np.mean((s[:,0]-g)**2+.3*s[:,1]**2+.03*u*u)/(1+np.mean(g*g)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-prefix',default='research/model_reuse_v12');p.add_argument('--shared-prefix',default='research/shared_experience_v12');a=p.parse_args()
    shared_path=Path(a.shared_prefix+'_results.json');shared=json.loads(shared_path.read_text());rows=[];arrays={};timing=[]
    with threadpool_limits(limits=1):
        for r in shared['runs']:
            final=r['checkpoints'][-1];seed=r['seed']+10000000000;model=final['models']['nonlinear'];true=r['true_theta_evaluator_only']
            t0=time.perf_counter();cem,cp,data=train(model,seed);timing.append(dict(seed=seed,virtual_training_seconds=time.perf_counter()-t0));p=cp[8]
            key=f"p{r['plant_index']}_s{r['replicate']}";arrays.update({key+'_'+k:v for k,v in data.items()})
            tests=[]
            for j,g in enumerate(TEST_GOALS):
                stream=r['seed']+500000+j;s,o,u=direct_episode(true,p,g,stream,MOTOR)
                result=dict(**score(s,o,u,targets(g)),states=s.tolist(),observations=o.tolist(),commands=u.tolist(),seed=stream)
                originals=final['tests'][j]['controllers'];prefix={k:prefix_cost(v['states'],v['commands'],g) for k,v in originals.items()}
                prefix['reused_direct']=prefix_cost(s,u,g)
                tests.append(dict(goal=g,reused_direct=result,prefix70_costs=prefix))
            rows.append(dict(plant=r['plant_index'],replicate=r['replicate'],physical_seed=r['seed'],virtual_seed=seed,
                learned_model=model,cem=cem,selected_coordinates=p.tolist(),tests=tests))
    summary=[]
    for pi in range(4):
        current=[r for r in rows if r['plant']==pi];original=[r['checkpoints'][-1] for r in shared['runs'] if r['plant_index']==pi]
        paired=[np.mean([old['controllers']['direct']['actual_cost']-new['reused_direct']['actual_cost'] for old,new in zip(o['tests'],r['tests'])]) for o,r in zip(original,current)]
        prefix={label:stat([np.mean([t['prefix70_costs'][label] for t in r['tests']]) for r in current]) for label in ('direct','linear','nonlinear','reused_direct')}
        summary.append(dict(plant=pi,reused_cost=stat([np.mean([t['reused_direct']['actual_cost'] for t in r['tests']]) for r in current]),
            stationary_successes=sum(t['reused_direct']['stationary_success'] for r in current for t in r['tests']),trials=16,
            paired_original_direct_minus_reused=stat(paired),prefix70_cost=prefix))
    prefix=a.output_prefix;np.savez_compressed(prefix+'_data.npz',**arrays);digest=hashlib.sha256(Path(prefix+'_data.npz').read_bytes()).hexdigest()
    out=dict(meta=dict(status='PRESCRIBED SAME-FAMILY DIAGNOSTIC; audit pending',design='model_reuse_v12_design.md',
        shared_result_sha256=hashlib.sha256(shared_path.read_bytes()).hexdigest(),data_sha256=digest,extra_physical_training_steps=0,
        virtual_steps_per_seed=33024,virtual_resets_per_seed=384,physical_test_steps=32*2*86,physical_test_resets=64,
        limitations='samecontrollerfamily; extra virtualcompute; suppliedpredictivefamily; exactmodelnotgiven; notoptimal'),runs=rows,summary=summary)
    Path(prefix+'_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');Path(prefix+'_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
