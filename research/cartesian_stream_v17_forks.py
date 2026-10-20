#!/usr/bin/env python3
"""External heldout intervention diagnostic; never fitting/selection data."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cartesian_stream_v17_learn import load_selected,digest,encode
from cartesian_stream_v17_acquire import endpoint
from coupled_context_v14_pilot import flow

BANK=np.array([[0.,0.],[.4,0.],[-.4,0.],[0.,.4],[0.,-.4]])
TIMES=(16,40,120,240)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='research/cartesian_stream_v17_rep0');args=p.parse_args()
    prefix=Path(args.prefix);ordinary=Path(str(prefix)+'_ordinary.npz');evaluator=Path(str(prefix)+'_evaluator.npz')
    fitfile=Path(str(prefix)+'_fit.json');target=Path(str(prefix)+'_forks.json')
    if target.exists():raise FileExistsError('Refusing to overwrite fork diagnostics')
    start=time.time();torch.set_num_threads(1);predictors,meta=load_selected(prefix,ordinary)
    data=np.load(ordinary,allow_pickle=False);ev=np.load(evaluator,allow_pickle=False)
    rows=[];nfev=0
    with threadpool_limits(limits=1),torch.no_grad():
        for i in np.flatnonzero(data['split']==2):
            for t in TIMES:
                o=data['observed'][i,:t+1];a=data['commands'][i,:t]
                forecasts={}
                for kind,predictor in predictors.items():
                    prepared=predictor.prepare(o,a)
                    forecasts[kind]=np.array([predictor.paths(o,a,torch.tensor(u[None,:],dtype=torch.float64),prepared).numpy() for u in BANK])
                truths=[];states=[];evaluations=[]
                for u in BANK:
                    z=ev['states'][i,t].copy();seq=[];counts=[]
                    for _ in range(2):
                        z,k=flow(z[:2],z[2:],u,.1);nfev+=k;seq.append(z.copy());counts.append(k)
                    states.append(seq);evaluations.append(counts);truths.append(endpoint(np.array(seq)[:,:2]))
                rows.append(dict(trajectory=int(i),time_index=t,predicted_positions_by_method=forecasts,
                    actual_positions_evaluator_only=np.array(truths),fork_states_evaluator_only=np.array(states),plant_nfev=evaluations))
    summary={}
    truths=np.array([r['actual_positions_evaluator_only'] for r in rows]);contrasts=truths[:,1:]-truths[:,:1]
    signal=float(np.mean(contrasts**2))
    for kind in predictors:
        pred=np.array([r['predicted_positions_by_method'][kind] for r in rows]);means=pred.mean(2)
        predicted_contrasts=means[:,1:]-means[:,:1]
        error=predicted_contrasts-contrasts
        summary[kind]=dict(ensemble_mean_absolute_coordinate_mse=float(np.mean((means-truths)**2)),
            member_absolute_coordinate_mse=float(np.mean((pred-truths[:,:,None,:,:])**2)),
            ensemble_mean_contrast_coordinate_mse=float(np.mean(error**2)),
            contrast_coordinate_mse_by_future_sample=np.mean(error**2,axis=(0,1,3)),
            zero_contrast_coordinate_mse=signal,contrast_mse_over_zero_contrast=float(np.mean(error**2)/signal),
            mean_true_contrast_norm_by_future_sample=np.mean(np.linalg.norm(contrasts,axis=3),axis=(0,1)),
            mean_predicted_contrast_norm_by_future_sample=np.mean(np.linalg.norm(predicted_contrasts,axis=3),axis=(0,1)))
    result=dict(status='HELDOUT EVALUATOR INTERVENTION CONTRASTS; NOT TRAINING OR A RELEASE',source_sha256=digest(__file__),
        learning_source_sha256=meta['source_sha256'],ordinary_sha256=digest(ordinary),evaluator_sha256=digest(evaluator),fit_sha256=digest(fitfile),
        design_sha256=digest(Path(__file__).with_name('cartesian_stream_v17_learning_design.md')),
        command_bank=BANK,time_indices=TIMES,selected=meta['selected'],summary=summary,contexts=rows,
        resources=dict(contexts=32,counterfactual_preparations=160,plant_flows=320,privileged_physical_seconds=32.,
            plant_rhs_evaluations=nfev,ordinary_new_experience=0,sensor_draws=0),wall_seconds=time.time()-start,
        limitations='32 histories from8 fixedtest trajectories, not32 independentbodies. Exactstates and forkmeans evaluator-only. No actual forkcamera noise or achieved control performance. Models selected only by prior ordinaryvalidation; forksnever update learners.')
    target.write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,default=encode,indent=2))

if __name__=='__main__':main()
