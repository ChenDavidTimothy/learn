#!/usr/bin/env python3
"""Fresh-corpus independent fit/ordinary-decision audit, frozen before outcomes.

No canonical implementation imports or physical-state features. Reuses our
independent augmented least squares, causal NumPy paths and analytic derivatives.
Physics/RNG/true score replay is a separate auditor's responsibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

from audit_cartesian_stream_v17_fit import ROOT,sha,rows,context,paths
from audit_cartesian_stream_v17_control import fit_local,goals_from_reading
from audit_cartesian_stream_v17_precontrol import cost_gradient

REPLICATES=tuple(range(200,208))
CONFIGS=[(k,r) for k in [128,512] for r in [.001,.1]]


def assert_close(actual,expected,rtol=2e-8,atol=2e-10,label='numeric'):
    assert np.allclose(actual,expected,rtol=rtol,atol=atol),(label,float(np.max(abs(np.asarray(actual)-np.asarray(expected)))))
    return float(np.max(abs(np.asarray(actual)-np.asarray(expected))))


def coefficient_roundoff_certificate(X,U,Y_source,query,config,saved,reference):
    """Secondary certificate, NEVER a silent widening of the original test.

    Replay exact source normal equations, compare a third pivoted QR, and bound
    the coordinate difference by residuals and an a-posteriori inverse bound.
    All downstream cost/derivative checks remain unchanged.
    """
    k,ridge=config;centered=X-query
    d2=np.mean(centered[:,:2]**2,axis=1)+np.mean(centered[:,2:34]**2,axis=1)+np.mean(centered[:,34:]**2,axis=1)
    ix=np.argsort(d2,kind='stable')[:min(k,len(X))];bandwidth=max(float(d2[ix[-1]]),1e-12)
    w=np.exp(-d2[ix]/bandwidth);w=w/np.sum(w)
    design=np.column_stack((np.ones(len(ix)),centered[ix],U[ix]));target=Y_source[ix]
    normal=design.T@(w[:,None]*design)+ridge*np.diag(np.r_[0.,np.ones(68)])
    rhs=design.T@(w[:,None]*target);replay=np.linalg.solve(normal,rhs)
    assert np.array_equal(replay,saved),'Secondary criterion requires EXACT source normal-equation replay'
    A=np.r_[np.sqrt(w)[:,None]*design,np.c_[np.zeros(68),np.eye(68)]*np.sqrt(ridge)]
    B=np.r_[np.sqrt(w)[:,None]*target,np.zeros((68,4))]
    qr=lstsq(A,B,lapack_driver='gelsy')[0]
    assert_close(qr,reference,label='secondary pivoted-QR versus independent SVD')
    eps=np.finfo(float).eps;gamma=70*eps/(1-70*eps)
    inverse=np.linalg.inv(normal)
    inverse_residual_upper=(np.linalg.norm(np.eye(69)-inverse@normal,np.inf)+
        gamma*(1+np.linalg.norm(abs(inverse)@abs(normal),np.inf)))
    assert inverse_residual_upper<.5,'Cannot certify ill-conditioned inverse'
    inverse_norm_upper=np.linalg.norm(inverse,np.inf)/(1-inverse_residual_upper)
    rs=np.linalg.norm(normal@saved-rhs,np.inf);rr=np.linalg.norm(normal@reference-rhs,np.inf)
    source_eval_roundoff=gamma*(np.linalg.norm(abs(normal)@abs(saved),np.inf)+np.linalg.norm(rhs,np.inf))
    reference_eval_roundoff=gamma*(np.linalg.norm(abs(normal)@abs(reference),np.inf)+np.linalg.norm(rhs,np.inf))
    denominator=np.linalg.norm(normal,np.inf)*np.linalg.norm(saved,np.inf)+np.linalg.norm(rhs,np.inf)
    backward_error=rs/max(denominator,np.finfo(float).tiny)
    assert backward_error<=gamma,'Saved exact normal replay lacks backward-error certificate'
    difference_bound=inverse_norm_upper*(rs+rr+source_eval_roundoff+reference_eval_roundoff)
    difference=float(np.max(abs(saved-reference)))
    assert difference<=difference_bound,'Coefficient discrepancy exceeds condition/residual bound'
    certificate=dict(original_rtol=2e-8,original_atol=2e-10,original_criterion_passed=False,
        exact_source_normal_replay=True,max_coefficient_difference=difference,
        third_pivoted_qr_vs_svd_max=float(np.max(abs(qr-reference))),condition_augmented=float(np.linalg.cond(A)),
        condition_normal=float(np.linalg.cond(normal)),normal_inverse_residual_upper=float(inverse_residual_upper),
        inverse_infinity_norm_upper=float(inverse_norm_upper),source_normal_residual_infinity=float(rs),
        reference_normal_residual_infinity=float(rr),source_backward_error=float(backward_error),
        residual_evaluation_gamma70=float(gamma),coordinate_difference_upper_bound=float(difference_bound),
        weighted_augmented_prediction_difference_norm=float(np.linalg.norm(A@(saved-reference))),
        qualification='Condition/residual floating-point certificate; original coordinate mismatch retained. Downstream criteria unchanged.')
    inputs=dict(normal=normal,rhs=rhs,design=design,weights=w,source_targets=target,
        saved_coefficient=saved,independent_coefficient=reference,pivoted_qr_coefficient=qr)
    return certificate,inputs


def audit_one(replicate,prefix):
    started=time.perf_counter();prefix=Path(prefix)
    files={name:Path(str(prefix)+suffix) for name,suffix in [
        ('ordinary','_ordinary.npz'),('acquisition','_acquisition.json'),('evaluator','_evaluator.npz'),
        ('fit','_local_fit.json'),('forecasts','_local_test_forecasts.npz'),('results','_study_results.json'),('arrays','_study_arrays.npz')]}
    result=json.loads(files['results'].read_text());fit=json.loads(files['fit'].read_text());acquisition=json.loads(files['acquisition'].read_text())
    assert result['status']=='COMPLETE FRESH LOCAL RETENTION CORPUS; INDEPENDENT AUDIT PENDING','Refuse incomplete corpus'
    assert fit['status']=='COMPLETE FRESH LOCAL-ONLY FIT/SELECTION; INDEPENDENT AUDIT PENDING'
    assert result['replicate']==fit['replicate']==acquisition['replicate']==replicate and replicate in REPLICATES
    assert result['source_sha256']==fit['source_sha256']==sha(ROOT/'cartesian_stream_v17_study.py')
    assert result['dependency_sha256']==fit['dependency_sha256']
    for name,digest in result['dependency_sha256'].items():assert digest==sha(ROOT/name),(name,'dependency hash')
    for key,filename in [('ordinary_sha256','ordinary'),('acquisition_sha256','acquisition'),('test_forecasts_sha256','forecasts')]:
        assert result[key]==fit[key]==sha(files[filename]),key
    assert result['evaluator_sha256']==sha(files['evaluator']) and result['fit_sha256']==sha(files['fit']) and result['arrays_sha256']==sha(files['arrays'])
    assert acquisition['artifacts'][files['ordinary'].name]==sha(files['ordinary']) and acquisition['artifacts'][files['evaluator'].name]==sha(files['evaluator'])
    assert fit['resources']['neural_fits']==result['resources']['neural_fits']==0
    data=np.load(files['ordinary'],allow_pickle=False);forecasts=np.load(files['forecasts'],allow_pickle=False);arrays=np.load(files['arrays'],allow_pickle=False)
    assert set(data.files)=={'observed','commands','split','trajectory_index','dt','sensor_sd'}
    o=data['observed'];a=data['commands'];assert o.shape==(40,257,2) and a.shape==(40,256,2)
    assert np.isfinite(o).all() and np.isfinite(a).all() and float(data['dt'])==.1 and float(data['sensor_sd'])==.0002
    train,validation,test=[np.flatnonzero(data['split']==k) for k in range(3)];assert list(map(len,[train,validation,test]))==[24,8,8]
    X,U,Y,origins=rows(o,a,train);assert np.array_equal(origins,fit['training_origins'])
    scale=dict(context_mean=X.mean(0),context_std=np.maximum(X.std(0),.01),target_mean=Y.mean(0),target_std=np.maximum(Y.std(0),.001))
    scale_errors={k:assert_close(v,fit['scale'][k],rtol=0,atol=2e-13,label=k) for k,v in scale.items()}
    source_scale={k:np.array(v) for k,v in fit['scale'].items()}
    assert np.array_equal(scale['context_mean'],source_scale['context_mean']) and np.array_equal(scale['context_std'],source_scale['context_std'])
    current=o[origins[:,0],origins[:,1]];previous=o[origins[:,0],origins[:,1]-1]
    baseline=current[:,None,:]+np.arange(1,3)[None,:,None]*(current-previous)[:,None,:]
    future=np.stack([o[origins[:,0],origins[:,1]+k] for k in [1,2]],axis=1)
    source_Y=((future-baseline).reshape(-1,4)-source_scale['target_mean'])/source_scale['target_std']
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    bootstrap=[]
    for member in range(3):
        seed=[20261020,17,660,replicate,0,member]
        ids=np.random.default_rng(np.random.SeedSequence(seed)).choice(train,len(train),replace=True)
        b=np.concatenate([np.flatnonzero(origins[:,0]==i) for i in ids]);bootstrap.append(b)
        assert seed==fit['bootstrap_seed_paths'][member] and np.array_equal(ids,fit['bootstrap_episode_ids'][member]) and np.array_equal(b,fit['bootstrap_rows'][member])
    errors=dict(offline_score=0.,raw_test_prediction=0.,control_query=0.,control_coefficient=0.,control_fit_diagnostic=0.,local_stationarity=0.,plan_cost=0.,selected_gradient=0.,selected_projected_gradient=0.)
    offline_cache={};score_checks=[]
    def offline_models(ci,i,t):
        key=(ci,int(i),t)
        if key not in offline_cache:
            query=(context(o[i,:t+1],a[i,:t])-scale['context_mean'])/scale['context_std'];models=[]
            for b in bootstrap:
                coefficient,_,stationarity=fit_local(X[b],U[b],Y[b],query,*CONFIGS[ci]);models.append(coefficient)
                errors['local_stationarity']=max(errors['local_stationarity'],stationarity)
            offline_cache[key]=models
        return offline_cache[key]

    def check_forecasts(ci,ids,blocks,stride,saved,label,raw=None):
        values=[];means=[];index=0
        for i in ids:
            for t in range(16,256-2*blocks+1,stride):
                prediction=paths(o[i,:t+1],a[i,:t],a[i,t:t+2*blocks:2],scale,offline_models(ci,i,t),'local')
                target=o[i,t+1:t+2*blocks+1]
                member_mse=float(np.mean((prediction-target)**2));ensemble_mse=float(np.mean((prediction.mean(0)-target)**2))
                by_sample=np.mean((prediction-target)**2,axis=(0,2));record=saved['records'][index]
                assert (record['trajectory'],record['time_index'])==(int(i),t)
                current=np.r_[member_mse,ensemble_mse,by_sample];expected=np.r_[record['member_coordinate_mse'],record['ensemble_mean_coordinate_mse'],record['member_mse_by_future_sample']]
                error=assert_close(current,expected,atol=2e-11,label=(label,index));errors['offline_score']=max(errors['offline_score'],error)
                if raw is not None:
                    assert np.array_equal(forecasts[raw+'_origins'][index],[i,t])
                    assert np.array_equal(forecasts[raw+'_observed_targets'][index],target)
                    pe=assert_close(prediction,forecasts[raw+'_member_predictions'][index],atol=2e-10,label=(raw,index,'raw prediction'));errors['raw_test_prediction']=max(errors['raw_test_prediction'],pe)
                values.append(member_mse);means.append(ensemble_mse);index+=1
        assert index==saved['origins']==len(saved['records'])
        summary=[np.mean(values),np.mean(means)];expected=[saved['member_coordinate_mse'],saved['ensemble_mean_coordinate_mse']]
        errors['offline_score']=max(errors['offline_score'],assert_close(summary,expected,atol=2e-11,label=label))
        score_checks.append(dict(label=label,candidate=ci,origins=index,member_coordinate_mse=float(summary[0]),ensemble_mean_coordinate_mse=float(summary[1])))
        return summary[0]

    with threadpool_limits(limits=1):
        assert len(fit['candidates'])==4;scores=[]
        for ci,(k,ridge) in enumerate(CONFIGS):
            candidate=fit['candidates'][ci];assert (candidate['neighbors'],candidate['ridge'])==(k,ridge)
            scores.append(check_forecasts(ci,validation,6,24,candidate['validation'],'validation'))
        selected=int(np.argmin(scores));assert selected==fit['selected_local_candidate']==result['selected_local_candidate']
        for label,blocks,stride in [('one_block',1,2),('six_block',6,24)]:check_forecasts(selected,test,blocks,stride,fit['test_observed_scores'][label],label,label)
        print('replicate',replicate,'all fit/forecast records verified',flush=True)
        assert len(result['witnesses'])==8 and len(result['ordinary_episodes'])==16
        counts=dict(decisions=0,control_member_fits=0,retained_row_events=0,retained_append_copies=0,frozen_pair_notifications=0,
            modeled_value_gradient_calls=0,unsuccessful_optimizer_terminations=0,fallback_decisions=0)
        episode_checks=[];pg_values=[];worst_pg=None;secondary_certificates=[];secondary_inputs={}
        for ei,record in enumerate(result['ordinary_episodes']):
            variant='retained' if ei%2==0 else 'frozen';case_index=ei//2
            assert (record['variant'],record['case_index'])==(variant,case_index)
            assert record['kind']==('local' if variant=='retained' else 'local_frozen_library')
            assert record['selected_candidate_index']==selected and record['selected_configuration']==dict(neighbors=CONFIGS[selected][0],ridge=CONFIGS[selected][1])
            assert record['local_library_initial_rows']==[2880]*3 and record['local_library_final_rows']==([2970]*3 if variant=='retained' else [2880]*3)
            assert record['real_pair_notifications']==90 and record['actual_local_row_events']==(90 if variant=='retained' else 0)
            keys=record['array_keys'];observed=arrays[keys['observed']];commands=arrays[keys['commands']]
            coef_saved=arrays[keys['local_fit_coefficients']];query_saved=arrays[keys['local_fit_queries']]
            assert observed.shape==(197,2) and commands.shape==(196,2) and coef_saved.shape==(90,3,69,4) and query_saved.shape==(90,66)
            assert np.isfinite(observed).all() and np.isfinite(commands).all() and np.max(abs(commands))<=.5
            goals=goals_from_reading(observed[16]);assert_close(goals,record['goals'],rtol=0,atol=1e-13,label='goals')
            if variant=='frozen':
                paired=result['ordinary_episodes'][ei-1];paired_keys=paired['array_keys']
                assert np.array_equal(observed[:17],arrays[paired_keys['observed']][:17]) and np.array_equal(commands[:16],arrays[paired_keys['commands']][:16])
                assert np.array_equal(record['goals'],paired['goals'])
                for key in ['initial_seed_path','warmup_seed_path','noise_seed_path']:assert record[key]==paired[key]
                assert record['paired_retained_task_cost']==paired['score']['task_cost']
                assert record['frozen_minus_retained_task_cost']==record['score']['task_cost']-paired['score']['task_cost']
            libraries=[(X[b].copy(),U[b].copy(),Y[b].copy()) for b in bootstrap]
            source_targets=[source_Y[b].copy() for b in bootstrap]
            previous=np.zeros((6,2));assert len(record['decisions'])==90
            episode_pg=[];failed=0;fallbacks=0
            for di,d in enumerate(record['decisions']):
                t=16+2*di;phase=di//30;assert (d['time_index'],d['renewal'],d['phase'])==(t,di,phase)
                assert_close(d['current_goal'],goals[phase],rtol=0,atol=1e-13,label='current goal')
                obs=observed[:t+1];past=commands[:t];query=(context(obs,past)-scale['context_mean'])/scale['context_std']
                errors['control_query']=max(errors['control_query'],assert_close(query,query_saved[di],rtol=0,atol=1e-12,label='query'))
                models=[]
                assert len(d['local_fit_diagnostics'])==3
                if variant=='retained':assert d['local_fit_array_index']==di
                for member,(xl,ul,yl) in enumerate(libraries):
                    coefficient,diagnostic,stationarity=fit_local(xl,ul,yl,query,*CONFIGS[selected]);models.append(coefficient)
                    errors['local_stationarity']=max(errors['local_stationarity'],stationarity)
                    saved_coefficient=coef_saved[di,member]
                    errors['control_coefficient']=max(errors['control_coefficient'],float(np.max(abs(coefficient-saved_coefficient))))
                    if not np.allclose(coefficient,saved_coefficient,rtol=2e-8,atol=2e-10):
                        # Original assertion retained. The narrowly secondary
                        # path must produce a logged mathematical certificate.
                        cert,cert_inputs=coefficient_roundoff_certificate(xl,ul,source_targets[member],query,CONFIGS[selected],saved_coefficient,coefficient)
                        cert.update(replicate=replicate,episode=ei,variant=variant,renewal=di,member=member)
                        secondary_certificates.append(cert)
                        for key,value in cert_inputs.items():secondary_inputs[f'episode{ei}_renewal{di}_member{member}_{key}']=value
                    sd=d['local_fit_diagnostics'][member];assert sd['n_rows']==2880+(di if variant=='retained' else 0)
                    for key,value in diagnostic.items():errors['control_fit_diagnostic']=max(errors['control_fit_diagnostic'],assert_close(value,sd[key],label=(replicate,ei,di,member,key)))
                    assert sd['normal_equation_max_residual']<1e-8;counts['control_member_fits']+=1
                def evaluate(plan):
                    plan=np.array(plan);prediction,jac=paths(obs,past,plan,scale,models,'local',True)
                    return cost_gradient(prediction,jac,obs[-1],goals[phase],plan)
                shifted=np.r_[previous[1:],np.zeros((1,2))];zero=np.zeros((6,2))
                assert np.array_equal(shifted,d['shifted_plan']) and np.array_equal(zero,d['zero_plan'])
                sc,sg=evaluate(shifted);zc,zg=evaluate(zero)
                for value,gradient,key,flag in [(sc,sg,'shifted_cost','shifted_finite'),(zc,zg,'zero_cost','zero_finite')]:
                    finite=bool(np.isfinite(value) and np.isfinite(gradient).all());assert finite==d[flag]
                    if finite:errors['plan_cost']=max(errors['plan_cost'],assert_close(value,d[key],atol=2e-9,label=(replicate,ei,di,key)))
                shifted_initial=d['shifted_finite'] and (not d['zero_finite'] or d['shifted_cost']<=d['zero_cost'])
                initial=shifted if shifted_initial else zero;initial_cost=sc if shifted_initial else zc
                assert d['initial_competitor']==('shifted_previous_plan' if shifted_initial else 'zero_plan') and np.array_equal(initial,d['initial_plan'])
                assert_close(initial_cost,d['initial_cost'],atol=2e-9,label='initial cost')
                opt=d['optimizer'];candidate=opt['final_plan']
                if candidate is None:assert opt['success'] is False;accept=False
                else:
                    candidate=np.array(candidate);feasible=bool(np.isfinite(candidate).all() and np.max(abs(candidate))<=.5);assert feasible==opt['final_feasible']
                    cc,cg=evaluate(candidate)
                    if opt['final_finite']:
                        assert np.isfinite(cc) and np.isfinite(cg).all()
                        errors['plan_cost']=max(errors['plan_cost'],assert_close(cc,opt['final_cost'],atol=2e-9,label='candidate cost'))
                    accept=opt['final_feasible'] and opt['final_finite'] and opt['final_cost']<=d['initial_cost']+1e-12
                    assert d['modeled_value_gradient_calls']==opt['nfev']+3
                chosen=np.array(d['selected_plan']);assert np.array_equal(chosen,candidate if accept else initial) and d['fallback_to_initial']==(not accept)
                cost,gradient=evaluate(chosen);errors['plan_cost']=max(errors['plan_cost'],assert_close(cost,d['selected_cost'],atol=2e-9,label='selected cost'))
                assert cost<=initial_cost+2e-8*max(1.,abs(initial_cost))
                errors['selected_gradient']=max(errors['selected_gradient'],assert_close(gradient,d['selected_gradient'],rtol=5e-7,atol=2e-8,label='gradient'))
                pg=chosen-np.clip(chosen-gradient,-.5,.5)
                errors['selected_projected_gradient']=max(errors['selected_projected_gradient'],assert_close(pg,d['selected_projected_gradient'],rtol=5e-7,atol=2e-8,label='PG'))
                assert_close(np.max(abs(pg)),d['selected_projected_gradient_max'],rtol=5e-7,atol=2e-8,label='PG max')
                pgmax=float(np.max(abs(pg)));pg_values.append(pgmax);episode_pg.append(pgmax)
                if worst_pg is None or pgmax>worst_pg['value']:worst_pg=dict(value=pgmax,episode=ei,variant=variant,case_index=case_index,renewal=di,time_index=t,message=opt['message'])
                assert np.array_equal(commands[t],chosen[0]) and np.array_equal(commands[t+1],chosen[0]) and np.array_equal(d['command'],chosen[0])
                if variant=='frozen' and di==0:assert np.array_equal(chosen,paired['decisions'][0]['selected_plan'])
                if variant=='retained':
                    y=np.concatenate([observed[t+k]-observed[t]-k*(observed[t]-observed[t-1]) for k in [1,2]])
                    y=(y-scale['target_mean'])/scale['target_std']
                    libraries=[(np.r_[xl,query[None]],np.r_[ul,(2*commands[t])[None]],np.r_[yl,y[None]]) for xl,ul,yl in libraries]
                    source_baseline=observed[t]+np.arange(1,3)[:,None]*(observed[t]-observed[t-1])
                    source_new_y=((observed[t+1:t+3]-source_baseline).ravel()-source_scale['target_mean'])/source_scale['target_std']
                    source_targets=[np.r_[ys,source_new_y[None]] for ys in source_targets]
                    assert d['local_library_rows_after_append']==[2880+di+1]*3
                    counts['retained_row_events']+=1;counts['retained_append_copies']+=3
                else:counts['frozen_pair_notifications']+=1
                previous=chosen.copy();counts['decisions']+=1;counts['modeled_value_gradient_calls']+=d['modeled_value_gradient_calls'];failed+=int(not opt['success']);fallbacks+=int(d['fallback_to_initial'])
            counts['unsuccessful_optimizer_terminations']+=failed;counts['fallback_decisions']+=fallbacks
            episode_checks.append(dict(index=ei,variant=variant,case_index=case_index,decisions=90,optimizer_unsuccessful=failed,fallbacks=fallbacks,max_projected_gradient=max(episode_pg)))
            print('replicate',replicate,'ordinary episode',ei,variant,'verified',flush=True)
    assert counts['decisions']==1440 and counts['control_member_fits']==4320 and counts['retained_row_events']==720 and counts['retained_append_copies']==2160 and counts['frozen_pair_notifications']==720
    for key in ['modeled_value_gradient_calls','unsuccessful_optimizer_terminations','fallback_decisions']:assert counts[key]==result['resources'][key]
    expected_resources=dict(acquisition_trajectories=40,acquisition_intervals=10240,acquisition_simulated_seconds=1024.,acquisition_independent_scalar_camera_draws=20560,
        ordinary_episodes=16,sustained_goal_windows=48,ordinary_decisions=1440,ordinary_intervals=3136,ordinary_simulated_seconds=313.6,ordinary_camera_read_opportunities=3152,ordinary_scalar_noise_opportunities=6304,
        fresh_initial_preparations=4,fresh_paired_noise_paths=8,distinct_scalar_control_noise_draws=3152,retained_actual_row_events=720,retained_member_append_copies=2160,frozen_real_pair_notifications=720,frozen_actual_append_copies=0,
        reference_episodes=8,privileged_reference_intervals=1568,privileged_reference_seconds=156.8,new_training_experience_from_reference=0)
    for key,value in expected_resources.items():assert result['resources'][key]==value
    recorded_summaries={}
    for variant in ['retained','frozen']:
        records=[e for e in result['ordinary_episodes'] if e['variant']==variant]
        recorded_summaries[variant]=dict(mean_task_cost=float(np.mean([e['score']['task_cost'] for e in records])),
            mean_combined_charged_cost=float(np.mean([e['score']['combined_charged_cost'] for e in records])),precise_goals=int(sum(e['score']['precise_goals'] for e in records)),goal_windows=24)
    differences=[e['frozen_minus_retained_task_cost'] for e in result['ordinary_episodes'] if e['variant']=='frozen']
    diagnostic=ROOT/'audit_cartesian_stream_v17_study_roundoff_results.json';diagnosis=json.loads(diagnostic.read_text())
    assert diagnosis['mpmath_decimal_digits']==80 and diagnosis['max_difference_vs_mpmath80']['canonical_target_svd']<1e-10 and diagnosis['max_difference_vs_mpmath80']['pivoted_qr']<1e-10
    assert diagnosis['source_auditor_sha256']==sha(ROOT/'audit_cartesian_stream_v17_study_coefficient_tolerance_pilot.py')
    assert diagnosis['diagnostic_source_sha256']==sha(ROOT/'audit_cartesian_stream_v17_study_roundoff.py')
    for name,digest in diagnosis['artifact_sha256'].items():assert digest==sha(ROOT/name)
    numerical_files=[diagnostic,ROOT/'audit_cartesian_stream_v17_study_roundoff.py',ROOT/'audit_cartesian_stream_v17_study_roundoff_inputs.npz',ROOT/'audit_cartesian_stream_v17_study_coefficient_tolerance_pilot.py',ROOT/'audit_cartesian_stream_v17_study_coefficient_tolerance_failure.json']
    if secondary_inputs:
        numerical_path=ROOT/f'audit_cartesian_stream_v17_study_rep{replicate}_roundoff_inputs.npz'
        np.savez_compressed(numerical_path,**secondary_inputs);numerical_files.append(numerical_path)
    artifact_files=list(files.values())+[ROOT/'cartesian_stream_v17_study.py',ROOT/'cartesian_stream_v17_study_design.md',Path(__file__),ROOT/'audit_cartesian_stream_v17_fit.py',ROOT/'audit_cartesian_stream_v17_control.py',ROOT/'audit_cartesian_stream_v17_precontrol.py']+numerical_files
    artifact_paths={p.name:str(p.resolve()) for p in artifact_files}
    audit=dict(status='PASS INDEPENDENT FRESH LOCAL FIT/FORECAST/ORDINARY-DECISION REPLAY',replicate=replicate,
        artifact_sha256={p.name:sha(p) for p in artifact_files},artifact_paths=artifact_paths,scale_max_errors=scale_errors,
        complete_episode_bootstraps_reconstructed=True,selected_local_candidate=selected,forecast_checks=score_checks,
        offline_unique_member_fits=3*len(offline_cache),maximum_absolute_errors=errors,counts=counts,episodes=episode_checks,
        secondary_coefficient_roundoff_certificates=secondary_certificates,
        selected_projected_gradient_values=pg_values,worst_projected_gradient=worst_pg,
        recorded_variant_summaries=recorded_summaries,recorded_paired_mean_frozen_minus_retained_task_cost=float(np.mean(differences)),
        wall_seconds=time.perf_counter()-started,
        limitations='No canonical implementation/physical state features. All local fits by independent augmented least squares, all modeled plans/selected derivatives replayed. Optimizer paths not rerun. Reported true task scores are arithmetic summaries awaiting separate physics/RNG/scoring audit. Same fixed body; one independent acquired corpus per replicate.')
    output=ROOT/f'audit_cartesian_stream_v17_study_rep{replicate}_results.json';output.write_text(json.dumps(audit,indent=2)+'\n')
    print('PASS fresh replicate',replicate,flush=True)


def aggregate():
    reports=[];report_paths=[]
    for rep in REPLICATES:
        path=ROOT/f'audit_cartesian_stream_v17_study_rep{rep}_results.json';r=json.loads(path.read_text());assert r['replicate']==rep and r['status'].startswith('PASS')
        for name,digest in r['artifact_sha256'].items():assert digest==sha(Path(r['artifact_paths'][name])),(rep,name,'stale audit')
        reports.append(r);report_paths.append(path)
    def mean_se(values):
        return dict(per_corpus=list(map(float,values)),mean=float(np.mean(values)),descriptive_corpus_se=float(np.std(values,ddof=1)/np.sqrt(8)))
    recorded={variant:dict(mean_task_cost=mean_se([r['recorded_variant_summaries'][variant]['mean_task_cost'] for r in reports]),
        mean_combined_charged_cost=mean_se([r['recorded_variant_summaries'][variant]['mean_combined_charged_cost'] for r in reports]),
        precise_goals=sum(r['recorded_variant_summaries'][variant]['precise_goals'] for r in reports),goal_windows=192) for variant in ['retained','frozen']}
    differences=mean_se([r['recorded_paired_mean_frozen_minus_retained_task_cost'] for r in reports])
    all_pg=np.concatenate([r['selected_projected_gradient_values'] for r in reports])
    out=dict(status='PASS ALL EIGHT INDEPENDENT FIT/FORECAST/ORDINARY-DECISION AUDITS; PHYSICAL SCORES SEPARATE',replicates=list(REPLICATES),
        artifact_sha256={p.name:sha(p) for p in report_paths+[Path(__file__)]},total_decisions=sum(r['counts']['decisions'] for r in reports),
        total_control_member_fits=sum(r['counts']['control_member_fits'] for r in reports),total_offline_member_fits=sum(r['offline_unique_member_fits'] for r in reports),
        maximum_absolute_errors={key:max(r['maximum_absolute_errors'][key] for r in reports) for key in reports[0]['maximum_absolute_errors']},
        total_optimizer_unsuccessful=sum(r['counts']['unsuccessful_optimizer_terminations'] for r in reports),total_fallbacks=sum(r['counts']['fallback_decisions'] for r in reports),
        secondary_coefficient_roundoff_certificates=[c for r in reports for c in r['secondary_coefficient_roundoff_certificates']],
        projected_gradient_quantiles=dict(zip(map(str,[0,.5,.9,.99,1]),np.quantile(all_pg,[0,.5,.9,.99,1]).tolist())),
        recorded_score_arithmetic=recorded,recorded_paired_frozen_minus_retained_task_cost=differences,
        limitations='Independent decisions/fitting verified. True physics/RNG/scoring must have separate audit before interpreting recorded-score summaries. Descriptive SE clusters eight acquired corpora; goals/windows/noise repeats are not independent bodies. No universal superiority or global optimizer claim.')
    (ROOT/'audit_cartesian_stream_v17_study_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,choices=REPLICATES);parser.add_argument('--prefix');parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    if args.aggregate:aggregate()
    else:
        assert args.replicate is not None,'Specify --replicate or --aggregate'
        audit_one(args.replicate,args.prefix or ROOT/f'cartesian_stream_v17_rep{args.replicate}')


if __name__=='__main__':main()
