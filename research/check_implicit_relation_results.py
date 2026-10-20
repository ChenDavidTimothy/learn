#!/usr/bin/env python3
"""Bounded independent second-screen arithmetic/selection/gate check.

No neural forward, fitting, physics or new sampling; stored conditioning flags
are checked against saved singular-value diagnostics, not recomputed networks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
METHODS=('implicit','explicit','alpaca','coda')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rp=ROOT/'implicit_relation_results.json';pp=ROOT/'implicit_relation_predictions.npz';dp=ROOT/'error_transport_screen_data.npz';hp=ROOT/'implicit_relation_holdout.npz';sp=ROOT/'implicit_relation_selection.json'
    saved=json.loads(rp.read_text());selection=json.loads(sp.read_text());preds=np.load(pp,allow_pickle=False);data=np.load(dp,allow_pickle=False);heldout=np.load(hp,allow_pickle=False)
    assert saved['data_sha256']==sha(dp) and saved['holdout_sha256']==sha(hp) and saved['selection_sha256']==sha(sp) and saved['predictions_sha256']==sha(pp) and saved['source_sha256']==sha(ROOT/'implicit_relation_screen.py')
    assert selection['status']=='FROZEN VALIDATION-ONLY SELECTION; TEST NOT SCORED' and selection['choices']==saved['choices']
    report_path=ROOT/'implicit_relation_holdout_report.json';acq=json.loads(report_path.read_text())
    for key,p in [('ordinary_sha256',hp),('evaluator_sha256',ROOT/'implicit_relation_holdout_evaluator.npz'),('source_sha256',ROOT/'implicit_relation_data.py'),('physics_source_sha256',ROOT/'error_transport_data.py'),('plan_sha256',ROOT/'implicit_relation_plan.md')]:assert acq[key]==sha(p)
    assert acq['seed_roles']==[20261020,18,30,1,'split','parameter0/initial1/action2'] and acq['physical_bodies']==16
    train=data['split']==0;assert np.sum(train)==32
    x=data['x'][train].reshape(-1,8);y=data['y'][train].reshape(-1,2)
    scale=dict(x_mean=x.mean(0),x_std=np.maximum(x.std(0),.01),y_mean=y.mean(0),y_std=np.maximum(y.std(0),.01))
    target=((heldout['y']-scale['y_mean'])/scale['y_std']).astype(np.float32)
    assert heldout['x'].shape==(16,8,96,8) and heldout['y'].shape==(16,8,96,2) and np.array_equal(heldout['split'],np.repeat([2,3],8))
    files=[rp,pp,dp,hp,sp,report_path,ROOT/'implicit_relation_holdout_evaluator.npz',ROOT/'implicit_relation_plan.md',ROOT/'implicit_relation_screen.py',ROOT/'implicit_relation_model.py',ROOT/'implicit_relation_data.py',ROOT/'error_transport_data.py',ROOT/'error_transport_screen.py',ROOT/'error_transport_model.py',ROOT/'error_transport_baselines.py']
    fit_checks=[];selected={};max_scale_error=0.;ineligible=[]
    old_screen=json.loads((ROOT/'error_transport_screen_seed0_results.json').read_text());files.append(ROOT/'error_transport_screen_seed0_results.json')
    for method in METHODS:
        options=[]
        for lr in [.001,.003]:
            stem=f'implicit_relation_{method}' if method in ('implicit','explicit') else f'error_transport_screen_{method}'
            path=ROOT/f'{stem}_lr{lr:g}_seed0.json';weight=path.with_suffix('.pt');report=json.loads(path.read_text());files.extend([path,weight])
            assert report['status'].startswith('COMPLETE') and report['data_sha256']==sha(dp) and report['weights_sha256']==sha(weight)
            assert report['method']==method and report['lr']==lr and report['seed']==0
            assert len(report['all_training_losses'])==2000 and np.isfinite(report['all_training_losses']).all()
            assert [r['step'] for r in report['validation_records']]==[500,1000,1500,2000]
            eligible=[]
            if method in ('implicit','explicit'):
                for key,p in [('holdout_sha256',hp),('driver_sha256',ROOT/'implicit_relation_screen.py'),('model_sha256',ROOT/'implicit_relation_model.py'),('plan_sha256',ROOT/'implicit_relation_plan.md')]:assert report[key]==sha(p)
                seed=int(np.random.SeedSequence([20261020,18,31,0,('implicit','explicit').index(method)]).generate_state(1)[0]);assert seed==report['initialization_seed']
                for rec in report['validation_records']:
                    values=np.array([rec['body_scores'][str(n)] for n in [4,16,64]]);assert values.shape==(3,8) and np.isfinite(values).all() and np.isclose(values.mean(),rec['criterion'],atol=1e-14,rtol=1e-14)
                    flags=[]
                    for diag in rec['diagnostics'].values():
                        sv=np.array(diag['min_singular']);flag=np.array(diag['conditioning_failure']);assert sv.shape==(8,) and np.isfinite(sv).all() and np.array_equal(sv<1e-4,flag)
                        flags.extend(flag.tolist())
                    failed=any(flags);assert failed==rec['conditioning_failure']
                    if failed:ineligible.append(dict(method=method,lr=lr,step=rec['step']))
                    else:eligible.append(rec)
            else:
                assert report['driver_sha256']==sha(ROOT/'error_transport_screen.py') and report['model_source_sha256']==sha(ROOT/'error_transport_model.py') and report['baseline_source_sha256']==sha(ROOT/'error_transport_baselines.py')
                for checkpoint in report['validation_records']:
                    for rec in checkpoint['validation']:
                        scores=np.array([rec['body_scores'][str(n)] for n in [4,16,64]])
                        assert scores.shape==(3,8) and np.isfinite(scores).all() and np.isclose(scores.mean(),rec['criterion'],rtol=1e-14,atol=1e-14)
                        eligible.append(dict(rec,step=checkpoint['step']))
            best=min(eligible,key=lambda r:r['criterion']);assert best==report['best_validation']
            payload=torch.load(weight,map_location='cpu',weights_only=False);assert payload['choice']==best and payload['method']==method
            assert all(torch.isfinite(value).all() for value in payload['state_dict'].values())
            for key,value in scale.items():
                err=float(np.max(abs(value-payload['scale'][key])));max_scale_error=max(max_scale_error,err);assert err<1e-12
            count=sum(value.numel() for key,value in payload['state_dict'].items() if key!='noise_variance');assert count==report['parameter_count']
            options.append((report,path,weight));fit_checks.append(dict(method=method,lr=lr,best_step=best['step'],criterion=best['criterion'],eligible_checkpoints=len(eligible),parameter_count=count,wall_seconds=report['wall_seconds']))
        report,path,weight=min(options,key=lambda r:r[0]['best_validation']['criterion']);choice=saved['choices'][method]
        assert Path(choice['report']).resolve()==path.resolve() and choice['report_sha256']==sha(path) and Path(choice['weights']).resolve()==weight.resolve() and choice['weights_sha256']==sha(weight) and choice['validation']==report['best_validation']
        if method in ('alpaca','coda'):
            old=old_screen['choices'][method]
            assert old['weights_sha256']==choice['weights_sha256'] and old['report_sha256']==choice['report_sha256'] and old['validation']==choice['validation'],'Baseline selection differs from frozen first screen'
        selected[method]=dict(training_lr=report['lr'],step=report['best_validation']['step'],adaptation_lr=report['best_validation'].get('adaptation_lr'),parameter_count=report['parameter_count'])
    metrics={};expected=[];max_mse_error=0.;failures=[];minimum=1e300;maximum=0.
    for method in METHODS:
        metrics[method]={}
        for si,label in [(2,'test_id'),(3,'test_ood')]:
            ids=np.flatnonzero(heldout['split']==si);qy=target[ids,1:].reshape(8,672,2);rawy=heldout['y'][ids,1:].reshape(8,672,2)
            for wrong in [False,True]:
                group=label+('_wrong_support' if wrong else '');r=saved['results'][method][group];assert r['body_ids']==ids.tolist();metrics[method][group]={}
                for budget in [0,4,16,64]:
                    key=f'{method}_{group}_support{budget}';expected.append(key);prediction=preds[key];assert prediction.shape==qy.shape and np.isfinite(prediction).all()
                    residual=prediction.astype(float)-qy.astype(float);mse=np.mean(residual**2,axis=(1,2));reported=np.array(r['standardized_body_mse'][str(budget)])
                    max_mse_error=max(max_mse_error,float(np.max(abs(mse-reported))));assert np.allclose(mse,reported,rtol=2e-6,atol=2e-7)
                    raw=np.mean((residual*scale['y_std'])**2,axis=(1,2));original_raw=np.mean((prediction.astype(float)*scale['y_std']+scale['y_mean']-rawy)**2,axis=(1,2))
                    metrics[method][group][str(budget)]=dict(body_ids=ids.tolist(),standardized_body_mse=mse.tolist(),raw_body_mse=raw.tolist(),raw_against_original_measured_targets=original_raw.tolist(),mean=float(mse.mean()),raw_mean=float(raw.mean()))
                    if budget==0:assert np.array_equal(prediction,preds[f'{method}_{label}_support0'])
                    if method in ('implicit','explicit'):
                        diag=r['diagnostics'][str(budget)];sv=np.array(diag['min_singular']);cond=np.array(diag['max_condition']);latent=np.array(diag['latent_norm'])
                        assert sv.shape==cond.shape==latent.shape==(8,) and np.isfinite(sv).all() and np.isfinite(cond).all() and np.isfinite(latent).all()
                        assert np.array_equal(sv<1e-4,diag['conditioning_failure']) and np.all(sv>=0) and np.all(cond>=1-1e-5) and np.all(latent>=0)
                        if method=='implicit':
                            minimum=min(minimum,float(sv.min()));maximum=max(maximum,float(cond.max()))
                            failures.extend(dict(group=group,budget=budget,body=int(ids[i])) for i,flag in enumerate(diag['conditioning_failure']) if flag)
                        if method=='explicit':assert np.array_equal(sv,np.ones(8)) and np.array_equal(cond,np.ones(8))
                        if budget==0:assert np.array_equal(latent,np.zeros(8)) and np.array_equal(sv,np.ones(8)) and np.array_equal(cond,np.ones(8))
    assert set(preds.files)==set(expected) and len(expected)==64
    mean=lambda method,split,n:metrics[method][split][str(n)]['mean']
    strongest=lambda split,n:min(mean(m,split,n) for m in ['alpaca','coda'])
    gates={}
    for n in [16,64]:
        other=64 if n==16 else 16
        gates[str(n)]=dict(advantage_over_both_strong_and_nested=mean('implicit','test_ood',n)<=.9*min(strongest('test_ood',n),mean('explicit','test_ood',n)),
            id_within_5_percent_at_both=all(mean('implicit','test_id',k)<=1.05*strongest('test_id',k) for k in [16,64]),
            opposite_ood_within_5_percent=mean('implicit','test_ood',other)<=1.05*strongest('test_ood',other),
            correct_support_better_than_wrong=mean('implicit','test_ood',n)<mean('implicit','test_ood_wrong_support',n),no_conditioning_failure=not failures)
    accuracy=any(all(g.values()) for g in gates.values())
    parity=all(mean('implicit',s,n)<=1.05*strongest(s,n) for s in ['test_id','test_ood'] for n in [16,64]) and not failures
    outcome='ACCURACY SCREEN SURVIVES; CONFIRMATION OWED' if accuracy else 'ACCURACY PARITY ONLY; REPEATED FAIR TIMING OWED' if parity else 'REJECT SECOND CANDIDATE; NO RESCUE TUNING'
    decision_path=ROOT/'implicit_relation_decision.json'
    decision_check=None
    if decision_path.exists():
        decision=json.loads(decision_path.read_text());files.extend([decision_path,ROOT/'implicit_relation_decision.py'])
        assert decision['results_sha256']==sha(rp) and decision['source_sha256']==sha(ROOT/'implicit_relation_decision.py') and decision['accuracy_gate']==gates and decision['accuracy_gate_pass']==accuracy and decision['efficiency_accuracy_prerequisite']==parity and decision['status']==outcome
        assert len(decision['conditioning_failures'])==len(failures) and decision['minimum_query_singular_value']==minimum and decision['maximum_query_condition']==maximum
        decision_check='Saved declared decision exactly agrees'
    report=dict(status='PASS PROPORTIONAL SECOND-SCREEN RESULT/SELECTION CHECK',artifact_sha256={p.name:sha(p) for p in files},
        train_only_normalization={k:v.tolist() for k,v in scale.items()},max_scale_error=max_scale_error,fit_checks=fit_checks,selected=selected,ineligible_validation_checkpoints=ineligible,
        all_per_body_normalized_raw_metrics=metrics,max_recorded_MSE_reduction_difference=max_mse_error,
        saved_diagnostic_conditioning_failures=failures,minimum_saved_query_singular_value=minimum,maximum_saved_query_condition=maximum,
        accuracy_gate=gates,accuracy_gate_pass=accuracy,efficiency_accuracy_prerequisite=parity,decision=outcome,saved_decision_check=decision_check,
        baseline_original_selection_reused_exactly=True,
        scope='Independent saved64prediction arrays/512body-budget normalized/raw errors; original train-only scales; all8reports/validation eligibility and selection/weights/provenance. Static source dataflow checked: supporttrajectory0/querytrajectories1..7, newholdout only, no evaluatorfeatures. No fitting, neural forward, singular-value recomputation or physics replay.',
        limitations='One trained seed/exact qv/noiseless finite-transition screen in one family. Baseline frozen-noiseALPaCA/transition-lossCoDA scopes/capacity differences disclosed. Conditioning flags verified against stored numerical diagnostics, not independently recomputed matrices. Good conditioning alone is not accuracy/capability. Efficiency continuation requires repeated synchronized cached-support benchmarks only if accuracy prerequisites survive.')
    (ROOT/'check_implicit_relation_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],selected=selected,accuracy_gate=gates,efficiency_accuracy_prerequisite=parity,decision=outcome,conditioning_failures=len(failures),minimum_saved_singular=minimum,max_mse_error=max_mse_error,
        OOD={m:{str(n):mean(m,'test_ood',n) for n in [16,64]} for m in METHODS}),indent=2))


if __name__=='__main__':main()
