#!/usr/bin/env python3
"""POSTHOC ordinary-validation diagnostic, never a deployed candidate selector."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cartesian_stream_v17_learn import Network,LOCAL_CANDIDATES,rows,local_predictor,digest,encode

def stats(x):
    x=np.asarray(x)
    return dict(mean=float(np.mean(x)),trajectory_se=float(np.std(x,ddof=1)/np.sqrt(len(x))),trajectory_values=x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='research/cartesian_stream_v17_rep0');args=p.parse_args()
    prefix=Path(args.prefix);out=Path(str(prefix)+'_action_risk_posthoc.json');arr=Path(str(prefix)+'_action_risk_posthoc_arrays.npz')
    if out.exists() or arr.exists():raise FileExistsError('Refusing to overwrite posthoc diagnostic')
    start=time.time();torch.set_num_threads(1)
    ordinary=Path(str(prefix)+'_ordinary.npz');fit=Path(str(prefix)+'_fit.json');weight_path=Path(str(prefix)+'_weights.npz')
    meta=json.loads(fit.read_text());assert digest(ordinary)==meta['ordinary_sha256'] and digest(weight_path)==meta['weights_sha256']
    assert digest(Path(__file__).with_name('cartesian_stream_v17_learn.py'))==meta['source_sha256']
    assert digest(Path(__file__).with_name('cartesian_stream_v17_local.py'))==meta['local_source_sha256']
    d=np.load(ordinary,allow_pickle=False);split=d['split'];keep=np.flatnonzero(split!=2)
    # NPZ stores one combined array; no test row contributes to any computation.
    o=d['observed'][keep];a=d['commands'][keep];split=split[keep]
    train=np.flatnonzero(split==0);val=np.flatnonzero(split==1)
    X,U,Y,_=rows(o,a,train);VX,VU,VY,origins=rows(o,a,val)
    s={k:np.array(v) for k,v in meta['scale'].items()};X=(X-s['context_mean'])/s['context_std'];Y=(Y-s['target_mean'])/s['target_std']
    VX=(VX-s['context_mean'])/s['context_std'];bootstrap=[np.array(b) for b in meta['bootstrap_rows']]
    auxiliaries=np.zeros((len(origins),8,2));seed_paths=[]
    for i in val:
        mask=origins[:,0]==i;path=[20261020,17,680,0,1,int(d['trajectory_index'][keep[i]])]
        auxiliaries[mask]=np.random.default_rng(np.random.SeedSequence(path)).uniform(-.5,.5,(int(mask.sum()),8,2));seed_paths.append(path)
    proposed=np.concatenate((VU[:,None,:],auxiliaries/.5,np.zeros((len(VX),1,2))),axis=1)
    weights=np.load(weight_path,allow_pickle=False);predictions=[];labels=[]
    with threadpool_limits(limits=1),torch.no_grad():
        for ci,record in enumerate(meta['nonlinear_candidates']):
            members=[]
            for mi in range(3):
                net=Network(record['width']);net.load_state_dict({k:torch.tensor(weights[f'nn{ci}_member{mi}_{k}']) for k in net.state_dict()})
                inp=np.concatenate((np.repeat(VX[:,None,:],10,axis=1),proposed),axis=2)
                y=net(torch.tensor(inp.reshape(-1,68),dtype=torch.float64)).numpy().reshape(len(VX),10,4)
                members.append(s['target_mean']+s['target_std']*y)
            predictions.append(np.array(members));labels.append(f'nonlinear_{ci}')
        for ci,config in enumerate(LOCAL_CANDIDATES):
            model=local_predictor(X,U,Y,bootstrap,s,config);members=[]
            for member in model.members:
                pred=[]
                for x,commands in zip(VX,proposed):
                    coefficients,_=member.fit(x)
                    normalized=coefficients[0]+commands@coefficients[-2:]
                    pred.append(s['target_mean']+s['target_std']*normalized)
                members.append(np.array(pred))
            predictions.append(np.array(members));labels.append(f'local_{ci}')
    predictions=np.array(predictions)
    # Common drift control variate, independent of actual validation outcomes.
    common_drift=predictions[:4,:,:,9,:].mean(axis=(0,1))
    baselines=dict(zero=np.zeros_like(common_drift),all_twelve_nonlinear_zero_command=common_drift)
    summaries={};all_scores={};all_ensemble_scores={};ordinary_scores=[]
    for ci,label in enumerate(labels):
        pred=predictions[ci];delta=pred[:,:,None,0,:]-pred[:,:,1:9,:]
        ordinary_row_mse=np.mean((pred[:,:,0,:]-VY[None,:,:])**2,axis=(0,2));ordinary_scores.append(ordinary_row_mse)
        cluster_mse=np.array([ordinary_row_mse[origins[:,0]==i].mean() for i in val])
        summary=dict(ordinary_one_block_member_coordinate_mse=stats(cluster_mse),action_scores={})
        for baseline,b in baselines.items():
            sq=.5*np.mean(delta**2,axis=3)
            cross=2*np.mean((VY-b)[None,:,None,:]*delta,axis=3)
            perrow=np.mean(sq-cross,axis=(0,2))
            ensemble_delta=delta.mean(0)
            ens_perrow=np.mean(.5*np.mean(ensemble_delta**2,axis=2)-2*np.mean((VY-b)[:,None,:]*ensemble_delta,axis=2),axis=1)
            clusters=np.array([perrow[origins[:,0]==i].mean() for i in val])
            ens_clusters=np.array([ens_perrow[origins[:,0]==i].mean() for i in val])
            summary['action_scores'][baseline]=dict(member_score=stats(clusters),ensemble_mean_score=stats(ens_clusters))
            all_scores.setdefault(baseline,[]).append(perrow);all_ensemble_scores.setdefault(baseline,[]).append(ens_perrow)
        summaries[label]=summary
    paired={}
    for baseline,values in all_scores.items():
        cluster=np.array([[v[origins[:,0]==i].mean() for i in val] for v in values])
        diff=cluster[:,None,:]-cluster[None,:,:]
        paired[baseline]=dict(mean_candidate_row_minus_column=diff.mean(2),trajectory_se=np.std(diff,axis=2,ddof=1)/np.sqrt(len(val)))
    np.savez_compressed(arr,predicted_corrections=predictions,auxiliary_commands=auxiliaries,actual_commands=VU*.5,
        target_corrections=VY,common_drift_baseline=common_drift,origins=origins,ordinary_member_mse_by_row=np.array(ordinary_scores),
        **{f'member_score_{k}':np.array(v) for k,v in all_scores.items()},
        **{f'ensemble_score_{k}':np.array(v) for k,v in all_ensemble_scores.items()})
    result=dict(status='EXPLORATORY POSTHOC ORDINARY-VALIDATION DIAGNOSTIC; NO CANDIDATE RESELECTION OR RELEASE',
        source_sha256=digest(__file__),ordinary_sha256=digest(ordinary),fit_sha256=digest(fit),weights_sha256=digest(weight_path),
        design_sha256=digest(Path(__file__).with_name('cartesian_stream_v17_action_risk_posthoc_design.md')),
        arrays_sha256=digest(arr),candidate_order=labels,auxiliary_seed_paths=seed_paths,summaries=summaries,paired_member_scores=paired,
        resources=dict(validation_trajectories=8,validation_rows=960,virtual_auxiliary_commands=960*8,
            candidate_member_block_queries=8*3*960*10,local_context_member_fits=4*3*960,new_physical_experience=0),
        limitations='POSTHOC after original selected testfork negative. No test observations/evaluator/fork/control outputs used. Existing selection unchanged. Eight correlated-trajectory clusters; mean-effect risk misses intrinsic/coherent distributional ambiguity. A poor drift baseline can increase variance. No post-selection or policy-transfer guarantee.',wall_seconds=time.time()-start)
    out.write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n')
    for label in labels:print(label,json.dumps(summaries[label],default=encode))

if __name__=='__main__':main()
