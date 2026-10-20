#!/usr/bin/env python3
"""Reproduce preserved206 coefficient-tolerance failure; exact inputs retained.

No canonical code imports/changes. Third pivoted QR, SVD and existing mpmath
80-digit weighted normal equations diagnose the declared ridge objective.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

from audit_cartesian_stream_v17_fit import ROOT,sha,rows,context,paths
from audit_cartesian_stream_v17_control import fit_local
from audit_cartesian_stream_v17_precontrol import cost_gradient


def main():
    prefix=ROOT/'cartesian_stream_v17_rep206';fitpath=Path(str(prefix)+'_local_fit.json');rp=Path(str(prefix)+'_study_results.json');op=Path(str(prefix)+'_ordinary.npz');ap=Path(str(prefix)+'_study_arrays.npz')
    fit=json.loads(fitpath.read_text());result=json.loads(rp.read_text());data=np.load(op,allow_pickle=False);arrays=np.load(ap,allow_pickle=False)
    o=data['observed'];a=data['commands'];X,U,Yraw,origins=rows(o,a,np.arange(24))
    independent_scale=dict(context_mean=X.mean(0),context_std=np.maximum(X.std(0),.01),target_mean=Yraw.mean(0),target_std=np.maximum(Yraw.std(0),.001))
    scale={k:np.array(v) for k,v in fit['scale'].items()};X=(X-scale['context_mean'])/scale['context_std']
    Y=(Yraw-independent_scale['target_mean'])/independent_scale['target_std']
    # Source's residual arithmetic, separately written without importing it.
    Ycanonical=[]
    for i in range(24):
        for t in range(16,256,2):
            baseline=o[i,t]+np.arange(1,3)[:,None]*(o[i,t]-o[i,t-1])
            Ycanonical.append((o[i,t+1:t+3]-baseline).ravel())
    Ycanonical=(np.array(Ycanonical)-scale['target_mean'])/scale['target_std']
    ei=7;di=88;member=1;t=16+2*di;record=result['ordinary_episodes'][ei];decision=record['decisions'][di];keys=record['array_keys']
    observed=arrays[keys['observed']][:t+1];commands=arrays[keys['commands']][:t]
    query=(context(observed,commands)-scale['context_mean'])/scale['context_std']
    saved_models=list(arrays[keys['local_fit_coefficients']][di]);b=np.array(fit['bootstrap_rows'][member]);cfg=fit['candidates'][fit['selected_local_candidate']]
    delta=X[b]-query;distance=np.mean(delta[:,:2]**2,axis=1)+np.mean(delta[:,2:34]**2,axis=1)+np.mean(delta[:,34:]**2,axis=1)
    ix=np.argsort(distance,kind='stable')[:cfg['neighbors']];w=np.exp(-distance[ix]/max(distance[ix[-1]],1e-12));w/=w.sum()
    D=np.c_[np.ones(len(ix)),delta[ix],U[b][ix]];target=Y[b][ix];target_canonical=Ycanonical[b][ix]
    reg=np.c_[np.zeros(68),np.eye(68)]*np.sqrt(cfg['ridge']);A=np.r_[np.sqrt(w)[:,None]*D,reg]
    B=np.r_[np.sqrt(w)[:,None]*target,np.zeros((68,4))];Bc=np.r_[np.sqrt(w)[:,None]*target_canonical,np.zeros((68,4))]
    with threadpool_limits(limits=1):
        svd=np.linalg.lstsq(A,B,rcond=None)[0];svd_canonical=np.linalg.lstsq(A,Bc,rcond=None)[0]
        qr=lstsq(A,Bc,lapack_driver='gelsy')[0]
        G=D.T@(w[:,None]*D)+cfg['ridge']*np.diag(np.r_[0,np.ones(68)])
        rhs=D.T@(w[:,None]*target_canonical);normal=np.linalg.solve(G,rhs)
        saved=saved_models[member]
        print('double precision solves done; constructing80-digit exact weighted objective',flush=True)
        mp.mp.dps=80
        md=mp.matrix(D.tolist());my=mp.matrix(target_canonical.tolist())
        wd=mp.matrix([[mp.mpf(float(w[i]))*md[i,j] for j in range(69)] for i in range(len(w))])
        wy=mp.matrix([[mp.mpf(float(w[i]))*my[i,j] for j in range(4)] for i in range(len(w))])
        mg=md.T*wd;mrhs=md.T*wy
        for j in range(1,69):mg[j,j]+=mp.mpf(float(cfg['ridge']))
        columns=[mp.lu_solve(mg,mrhs[:,j]) for j in range(4)]
        high=np.array([[float(columns[j][i]) for j in range(4)] for i in range(69)])
        gap=mp.mpf('0')
        for j in range(4):
            difference=mp.matrix([mp.mpf(float(saved[i,j]))-columns[j][i] for i in range(69)])
            gap+=(difference.T*mg*difference)[0]
        print('80-digit solve done',flush=True)
        def objective(coef):return float(np.sum(w[:,None]*(D@coef-target_canonical)**2)+cfg['ridge']*np.sum(coef[1:]**2))
        def stationarity(coef):return float(np.max(abs(G@coef-rhs)))
        solutions=dict(saved_normal=saved,reconstructed_normal=normal,independent_target_svd=svd,canonical_target_svd=svd_canonical,pivoted_qr=qr,mpmath80=high)
        objective_values={name:objective(coef) for name,coef in solutions.items()}
        residuals={name:stationarity(coef) for name,coef in solutions.items()}
        differences={name:float(np.max(abs(coef-high))) for name,coef in solutions.items()}
        independent_models=[]
        for bs in fit['bootstrap_rows']:
            bs=np.array(bs);coef,_,_=fit_local(X[bs],U[bs],Y[bs],query,cfg['neighbors'],cfg['ridge']);independent_models.append(coef)
        # Compare actual prepared plans against the unchanged saved records,
        # separately from coefficient coordinates.
        downstream=[];saved_arrays={}
        for label in ['shifted_plan','zero_plan','initial_plan','selected_plan']:
            plan=np.array(decision[label]);ps,js=paths(observed,commands,plan,scale,saved_models,'local',True)
            pi,ji=paths(observed,commands,plan,independent_scale,independent_models,'local',True)
            cs,gs=cost_gradient(ps,js,observed[-1],np.array(decision['current_goal']),plan)
            ci,gi=cost_gradient(pi,ji,observed[-1],np.array(decision['current_goal']),plan)
            downstream.append(dict(label=label,path_max_difference=float(np.max(abs(ps-pi))),cost_saved_coefficients=cs,cost_independent_coefficients=ci,cost_difference=abs(cs-ci),gradient_max_difference=float(np.max(abs(gs-gi)))))
            if label=='selected_plan':
                assert np.isclose(ci,decision['selected_cost'],rtol=2e-8,atol=2e-9)
                assert np.allclose(gi,decision['selected_gradient'],rtol=5e-7,atol=2e-8)
                saved_arrays=dict(selected_plan=plan,source_selected_paths=ps,independent_selected_paths=pi,source_selected_gradient=gs,independent_selected_gradient=gi)
    arraypath=ROOT/'audit_cartesian_stream_v17_study_roundoff_inputs.npz'
    np.savez_compressed(arraypath,design=D,weights=w,canonical_targets=target_canonical,independent_targets=target,
        augmented_design=A,canonical_augmented_targets=Bc,independent_augmented_targets=B,query=query,
        source_all_member_coefficients=np.array(saved_models),independent_all_member_coefficients=np.array(independent_models),
        mpmath80_coefficients=high,pivoted_qr_coefficients=qr,independent_svd_coefficients=svd,
        ordinary_observed_prefix=observed,issued_command_prefix=commands,**saved_arrays)
    worst=np.unravel_index(np.argmax(abs(svd-saved)),svd.shape)
    out=dict(status='ROUND-OFF DIAGNOSTIC; ORIGINAL AUDIT FAILURE PRESERVED; CANONICAL STUDY UNCHANGED',replicate=206,episode=ei,variant='frozen',renewal=di,member=member,
        original_atol=2e-10,original_rtol=2e-8,max_original_coefficient_difference=float(np.max(abs(svd-saved))),largest_difference_index=list(map(int,worst)),
        source_auditor_sha256=sha(ROOT/'audit_cartesian_stream_v17_study_coefficient_tolerance_pilot.py'),diagnostic_source_sha256=sha(Path(__file__)),
        artifact_sha256={p.name:sha(p) for p in [fitpath,rp,op,ap,arraypath]},mpmath_decimal_digits=80,
        condition_augmented=float(np.linalg.cond(A)),condition_normal=float(np.linalg.cond(G)),
        max_normalized_target_rounding_difference=float(np.max(abs(target-target_canonical))),
        objectives=objective_values,normal_stationarity=residuals,max_difference_vs_mpmath80=differences,
        exact_quadratic_objective_gap_saved_vs_mpmath80=str(gap),
        weighted_augmented_saved_vs_mpmath_prediction_norm=float(np.linalg.norm(A@(saved-high))),
        weighted_augmented_saved_vs_independent_svd_prediction_norm=float(np.linalg.norm(A@(saved-svd))),
        downstream_actual_prepared_plans=downstream,
        limitations='Only preserved coefficient assertion diagnosed. All206/207 records still require completed unchanged-scope replay after any explicitly documented numerical criterion repair. No controller/model/test changes are authorized by this diagnostic.')
    (ROOT/'audit_cartesian_stream_v17_study_roundoff_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
