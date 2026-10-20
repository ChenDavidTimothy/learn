#!/usr/bin/env python3
"""Independent ordinary-decision artifact audit, prepared before control outcomes.

No canonical implementation imports, physical state inputs or optimizer replay.
Reconstructs all online local libraries/QR fits and every recorded plan's modeled
cost, selected gradient and acceptance policy from ordinary prefixes only.
Physical/RNG/actual-score replay belongs to the separate theory auditor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cartesian_stream_v17_fit import ROOT, sha, rows, context, paths
from audit_cartesian_stream_v17_precontrol import cost_gradient


def fit_local(X,U,Y,query,k,ridge):
    centered=X-query
    distances=np.sum(centered[:,:2]**2,axis=1)/2+np.sum(centered[:,2:34]**2,axis=1)/32+np.sum(centered[:,34:]**2,axis=1)/32
    ix=np.argsort(distances,kind='stable')[:min(k,len(X))]
    bandwidth=max(float(distances[ix[-1]]),1e-12)
    weights=np.exp(-distances[ix]/bandwidth);weights/=weights.sum()
    design=np.c_[np.ones(len(ix)),centered[ix],U[ix]]
    regularizer=np.c_[np.zeros(68),np.eye(68)]*np.sqrt(ridge)
    augmented=np.r_[np.sqrt(weights)[:,None]*design,regularizer]
    targets=np.r_[np.sqrt(weights)[:,None]*Y[ix],np.zeros((68,4))]
    coefficient=np.linalg.lstsq(augmented,targets,rcond=None)[0]
    residual=design@coefficient-Y[ix]
    mse=np.sum(weights[:,None]*residual**2,axis=0)
    normal_residual=design.T@(weights[:,None]*residual);normal_residual[1:]+=ridge*coefficient[1:]
    diagnostic=dict(n_rows=len(X),neighbors_requested=k,neighbors_used=len(ix),ridge=ridge,
        bandwidth_d2=bandwidth,nearest_d2=float(distances[ix[0]]),farthest_selected_d2=float(distances[ix[-1]]),
        weighted_mean_d2=float(weights@distances[ix]),effective_neighbors=float(1/np.sum(weights**2)),
        weighted_mse_per_output=mse,weighted_mse=float(mse.mean()),
        penalized_objective_per_output=mse+ridge*np.sum(coefficient[1:]**2,axis=0))
    return coefficient,diagnostic,float(np.max(abs(normal_residual)))


def goals_from_reading(reading):
    g=.95*reading
    def rotate(angle):
        c=np.cos(angle);s=np.sin(angle)
        return np.array([c*g[0]-s*g[1],s*g[0]+c*g[1]])
    return np.array([g,rotate(.04),rotate(-.04)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default=str(ROOT/'cartesian_stream_v17_rep0_control'))
    parser.add_argument('--fit-prefix',default=str(ROOT/'cartesian_stream_v17_rep0'))
    args=parser.parse_args();prefix=Path(args.prefix);fitprefix=Path(args.fit_prefix)
    resultpath=Path(str(prefix)+'_results.json');arraypath=Path(str(prefix)+'_arrays.npz')
    fitpath=Path(str(fitprefix)+'_fit.json');weightpath=Path(str(fitprefix)+'_weights.npz');ordinarypath=Path(str(fitprefix)+'_ordinary.npz')
    saved=json.loads(resultpath.read_text())
    assert saved['status']=='COMPLETE EXPLORATORY MOVING WITNESS/CONTROL PILOT; INDEPENDENT AUDIT PENDING','Refusing incomplete control artifact'
    assert saved['arrays_sha256']==sha(arraypath)
    assert saved['source_sha256']==sha(ROOT/'cartesian_stream_v17_control.py')
    assert saved['ordinary_sha256']==sha(ordinarypath) and saved['fit_sha256']==sha(fitpath) and saved['weights_sha256']==sha(weightpath)
    for name,value in saved['dependency_sha256'].items():assert value==sha(ROOT/name),(name,'dependency changed')
    fit=json.loads(fitpath.read_text());data=np.load(ordinarypath,allow_pickle=False);weights=np.load(weightpath,allow_pickle=False);arrays=np.load(arraypath,allow_pickle=False)
    X,U,Y,origins=rows(data['observed'],data['commands'],np.flatnonzero(data['split']==0))
    scale={k:np.array(v) for k,v in fit['scale'].items()}
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    bootstraps=[np.array(b) for b in fit['bootstrap_rows']]
    nn_index=fit['selected']['nonlinear']
    nn_models=[[weights[f'nn{nn_index}_member{member}_layers.{layer}.{name}'] for layer in [0,2,4] for name in ['weight','bias']] for member in range(3)]
    local_config=fit['local_candidates'][fit['selected']['local']]
    max_errors=dict(local_coefficients=0.,local_query=0.,local_diagnostics=0.,local_stationarity=0.,plan_cost=0.,selected_gradient=0.,selected_projected_gradient=0.,goal=0.)
    episodes=[];stationarity_by_family={'nonlinear':[],'local':[]};worst_stationarity={}
    totals=dict(decisions=0,local_member_fits=0,modeled_value_gradient_calls=0,unsuccessful_optimizer_terminations=0,fallback_decisions=0)
    assert len(saved['ordinary_episodes'])==16
    with threadpool_limits(limits=1):
        for episode_index,record in enumerate(saved['ordinary_episodes']):
            kind=record['kind'];case_index=record['case_index'];keys=record['array_keys']
            assert kind==('nonlinear' if episode_index%2==0 else 'local') and case_index==episode_index//2
            assert record['selected_candidate_index']==fit['selected'][kind]
            observed=arrays[keys['observed']];commands=arrays[keys['commands']]
            assert observed.shape==(197,2) and commands.shape==(196,2)
            assert np.isfinite(observed).all() and np.isfinite(commands).all() and np.max(abs(commands))<=.5
            goals=goals_from_reading(observed[16]);ge=float(np.max(abs(goals-record['goals'])));max_errors['goal']=max(max_errors['goal'],ge);assert ge<1e-13
            assert len(record['decisions'])==90
            libraries=[(X[b].copy(),U[b].copy(),Y[b].copy()) for b in bootstraps] if kind=='local' else None
            if kind=='local':
                coefficients_saved=arrays[keys['local_fit_coefficients']];queries_saved=arrays[keys['local_fit_queries']]
                assert coefficients_saved.shape==(90,3,69,4) and queries_saved.shape==(90,66)
                assert record['local_library_initial_rows']==[len(b) for b in bootstraps]
            previous=np.zeros((6,2));episode_max_pg=0.;episode_fallbacks=0;episode_failures=0
            for index,decision in enumerate(record['decisions']):
                t=16+2*index;phase=index//30
                assert (decision['time_index'],decision['renewal'],decision['phase'])==(t,index,phase)
                assert np.allclose(decision['current_goal'],goals[phase],rtol=0,atol=1e-13)
                obs=observed[:t+1];a=commands[:t]
                if kind=='local':
                    query=(context(obs,a)-scale['context_mean'])/scale['context_std']
                    qe=float(np.max(abs(query-queries_saved[index])));max_errors['local_query']=max(max_errors['local_query'],qe);assert qe<1e-12
                    models=[]
                    assert decision['local_fit_array_index']==index and len(decision['local_fit_diagnostics'])==3
                    for member,(xl,ul,yl) in enumerate(libraries):
                        coefficient,diagnostic,stationarity=fit_local(xl,ul,yl,query,local_config['neighbors'],local_config['ridge'])
                        models.append(coefficient);max_errors['local_stationarity']=max(max_errors['local_stationarity'],stationarity)
                        ce=float(np.max(abs(coefficient-coefficients_saved[index,member])));max_errors['local_coefficients']=max(max_errors['local_coefficients'],ce)
                        assert np.allclose(coefficient,coefficients_saved[index,member],rtol=2e-8,atol=2e-10),(episode_index,index,member,'local coefficients',ce)
                        saved_d=decision['local_fit_diagnostics'][member]
                        for key,value in diagnostic.items():
                            err=float(np.max(abs(np.array(value)-np.array(saved_d[key]))));max_errors['local_diagnostics']=max(max_errors['local_diagnostics'],err)
                            assert np.allclose(value,saved_d[key],rtol=2e-8,atol=2e-10),(episode_index,index,member,key,err)
                        assert saved_d['normal_equation_max_residual']<1e-8
                        totals['local_member_fits']+=1
                else:
                    models=nn_models;assert decision['local_fit_diagnostics']==[]

                def evaluate(plan):
                    p=np.array(plan,dtype=float);assert p.shape==(6,2)
                    yp,jp=paths(obs,a,p,scale,models,kind,True)
                    return cost_gradient(yp,jp,obs[-1],goals[phase],p)

                shifted=np.r_[previous[1:],np.zeros((1,2))];zero=np.zeros((6,2))
                assert np.array_equal(shifted,np.array(decision['shifted_plan'])) and np.array_equal(zero,np.array(decision['zero_plan']))
                sc,sg=evaluate(shifted);zc,zg=evaluate(zero)
                for value,gradient,key,flag in [(sc,sg,'shifted_cost','shifted_finite'),(zc,zg,'zero_cost','zero_finite')]:
                    finite=bool(np.isfinite(value) and np.isfinite(gradient).all())
                    assert finite==decision[flag]
                    if finite:
                        err=abs(value-decision[key]);max_errors['plan_cost']=max(max_errors['plan_cost'],err)
                        assert np.isclose(value,decision[key],rtol=2e-8,atol=2e-9),(episode_index,index,key,err)
                # Program's tie/accept decisions use its recorded values, while
                # their independently recomputed values are checked above/below.
                expected_name=('shifted_previous_plan' if decision['shifted_finite'] and
                    (not decision['zero_finite'] or decision['shifted_cost']<=decision['zero_cost']) else 'zero_plan')
                initial=shifted if expected_name=='shifted_previous_plan' else zero
                assert decision['initial_competitor']==expected_name and np.array_equal(initial,np.array(decision['initial_plan']))
                initial_cost=sc if expected_name=='shifted_previous_plan' else zc
                assert np.isclose(initial_cost,decision['initial_cost'],rtol=2e-8,atol=2e-9)
                optimizer=decision['optimizer'];candidate=optimizer['final_plan']
                if candidate is None:
                    assert optimizer['success'] is False and optimizer['final_cost'] is None
                    accept=False
                else:
                    candidate=np.array(candidate);feasible=bool(np.isfinite(candidate).all() and np.max(abs(candidate))<=.5)
                    assert feasible==optimizer['final_feasible']
                    cc,cg=evaluate(candidate)
                    if optimizer['final_finite']:
                        assert np.isfinite(cc) and np.isfinite(cg).all()
                        err=abs(cc-optimizer['final_cost']);max_errors['plan_cost']=max(max_errors['plan_cost'],err)
                        assert np.isclose(cc,optimizer['final_cost'],rtol=2e-8,atol=2e-9),(episode_index,index,'candidate cost',err)
                    accept=optimizer['final_feasible'] and optimizer['final_finite'] and optimizer['final_cost']<=decision['initial_cost']+1e-12
                    assert decision['modeled_value_gradient_calls']==optimizer['nfev']+3
                selected=np.array(decision['selected_plan']);expected=candidate if accept else initial
                assert decision['fallback_to_initial']==(not accept) and np.array_equal(selected,expected)
                assert np.max(abs(selected))<=.5 and np.array_equal(commands[t],selected[0]) and np.array_equal(commands[t+1],selected[0])
                assert np.array_equal(np.array(decision['command']),selected[0])
                cost,gradient=evaluate(selected)
                error=abs(cost-decision['selected_cost']);max_errors['plan_cost']=max(max_errors['plan_cost'],error)
                assert np.isclose(cost,decision['selected_cost'],rtol=2e-8,atol=2e-9)
                assert cost<=initial_cost+2e-8*max(1.,abs(initial_cost))
                gerror=float(np.max(abs(gradient-decision['selected_gradient'])));max_errors['selected_gradient']=max(max_errors['selected_gradient'],gerror)
                assert np.allclose(gradient,decision['selected_gradient'],rtol=5e-7,atol=2e-8),(episode_index,index,'gradient',gerror)
                pg=selected-np.clip(selected-gradient,-.5,.5)
                pgerror=float(np.max(abs(pg-decision['selected_projected_gradient'])));max_errors['selected_projected_gradient']=max(max_errors['selected_projected_gradient'],pgerror)
                assert np.allclose(pg,decision['selected_projected_gradient'],rtol=5e-7,atol=2e-8)
                assert np.isclose(np.max(abs(pg)),decision['selected_projected_gradient_max'],rtol=5e-7,atol=2e-8)
                episode_max_pg=max(episode_max_pg,float(np.max(abs(pg))))
                pgmax=float(np.max(abs(pg)));stationarity_by_family[kind].append(pgmax)
                if kind not in worst_stationarity or pgmax>worst_stationarity[kind]['projected_gradient_max']:
                    worst_stationarity[kind]=dict(episode_index=episode_index,case_index=case_index,renewal=index,phase=phase,time_index=t,
                        projected_gradient_max=pgmax,selected_cost=cost,initial_cost=initial_cost,
                        optimizer_success=optimizer['success'],optimizer_message=optimizer['message'],optimizer_nfev=optimizer['nfev'],optimizer_nit=optimizer['nit'])
                if kind=='local':
                    target=np.concatenate([observed[t+k]-observed[t]-k*(observed[t]-observed[t-1]) for k in [1,2]])
                    y=(target-scale['target_mean'])/scale['target_std']
                    for member,(xl,ul,yl) in enumerate(libraries):libraries[member]=(np.r_[xl,query[None]],np.r_[ul,(2*commands[t])[None]],np.r_[yl,y[None]])
                    assert decision['local_library_rows_after_append']==[len(b)+index+1 for b in bootstraps]
                previous=selected.copy();totals['decisions']+=1
                totals['modeled_value_gradient_calls']+=decision['modeled_value_gradient_calls']
                episode_fallbacks+=int(decision['fallback_to_initial']);episode_failures+=int(not optimizer['success'])
            if kind=='local':assert record['local_library_final_rows']==[len(b)+90 for b in bootstraps]
            totals['fallback_decisions']+=episode_fallbacks;totals['unsuccessful_optimizer_terminations']+=episode_failures
            episodes.append(dict(index=episode_index,case_index=case_index,kind=kind,decisions=90,optimizer_unsuccessful=episode_failures,fallbacks=episode_fallbacks,max_selected_projected_gradient=episode_max_pg))
            print('audited ordinary episode',episode_index,kind,'decisions90',flush=True)
    assert totals['decisions']==1440 and totals['local_member_fits']==2160
    for key in ['modeled_value_gradient_calls','unsuccessful_optimizer_terminations','fallback_decisions']:assert totals[key]==saved['resources'][key]
    artifact_files=[resultpath,arraypath,fitpath,ordinarypath,weightpath,ROOT/'cartesian_stream_v17_control.py',ROOT/'cartesian_stream_v17_planner_contract.md',ROOT/'audit_cartesian_stream_v17_fit.py',ROOT/'audit_cartesian_stream_v17_precontrol.py']
    quantiles=[0,.5,.9,.99,1.]
    stationarity_summary={kind:dict(quantiles=dict(zip(map(str,quantiles),np.quantile(values,quantiles).tolist())),
        counts_above={str(tol):int(np.sum(np.array(values)>tol)) for tol in [1e-6,1e-5,1e-4,1e-3,.01]},worst=worst_stationarity[kind]) for kind,values in stationarity_by_family.items()}
    result=dict(status='PASS INDEPENDENT ACTUAL ORDINARY DECISION/ONLINE-LIBRARY REPLAY',artifact_sha256={p.name:sha(p) for p in artifact_files},maximum_absolute_errors=max_errors,totals=totals,episodes=episodes,selected_plan_stationarity=stationarity_summary,
        limitations='No canonical model/local/control imports or optimizer rerun. Every plan cost/selected derivative reconstructed from ordinary prefixes; all local libraries fitted by independent augmented least squares. Optimizer internal trajectory and unsaved original final jac are not independently certified. Physical trajectory/RNG/actual-score/witness replay belongs to separate audit. One corpus/four preparations: no population superiority claim.')
    (ROOT/'audit_cartesian_stream_v17_control_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
