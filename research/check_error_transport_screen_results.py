#!/usr/bin/env python3
"""Proportional independent decisive-screen arithmetic/selection check.

No network-forward replay, retraining, canonical model imports or physics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
METHODS=('joint','separable','alpaca','coda','pooled')
BUDGETS=(0,4,16,64)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    resultpath=ROOT/'error_transport_screen_seed0_results.json';predictionpath=ROOT/'error_transport_screen_seed0_predictions.npz';datapath=ROOT/'error_transport_screen_data.npz'
    saved=json.loads(resultpath.read_text());predictions=np.load(predictionpath,allow_pickle=False);data=np.load(datapath,allow_pickle=False)
    assert saved['seed']==0 and saved['source_sha256']==sha(ROOT/'error_transport_screen.py') and saved['data_sha256']==sha(datapath) and saved['predictions_sha256']==sha(predictionpath)
    x=data['x'];y=data['y'];split=data['split'];assert x.shape==(56,8,96,8) and y.shape==(56,8,96,2)
    assert [int(np.sum(split==i)) for i in range(4)]==[32,8,8,8]
    training=split==0
    # Independently concatenate only32training bodies, preserving declared
    # coordinate axes. Normalization is not estimated from any heldout target.
    scale=dict(x_mean=x[training].reshape(-1,8).mean(0),x_std=np.maximum(x[training].reshape(-1,8).std(0),.01),
        y_mean=y[training].reshape(-1,2).mean(0),y_std=np.maximum(y[training].reshape(-1,2).std(0),.01))
    normalized=((y-scale['y_mean'])/scale['y_std']).astype(np.float32)
    files=[resultpath,predictionpath,datapath,ROOT/'error_transport_screen.py',ROOT/'error_transport_data.py',ROOT/'error_transport_model.py',ROOT/'error_transport_baselines.py',ROOT/'campaign_plan.md']
    fit_checks=[];best_by_method={};max_scale_error=0.
    for method in METHODS:
        fits=[]
        for lr in [.001,.003]:
            path=ROOT/f'error_transport_screen_{method}_lr{lr:g}_seed0.json';weight=path.with_suffix('.pt');report=json.loads(path.read_text());files.extend([path,weight])
            assert report['status']=='COMPLETE TRAIN/VALIDATION FIT; TEST NOT SCORED' and report['method']==method and report['lr']==lr and report['seed']==0 and report['steps']==2000
            for field,p in [('data_sha256',datapath),('driver_sha256',ROOT/'error_transport_screen.py'),('model_source_sha256',ROOT/'error_transport_model.py'),('baseline_source_sha256',ROOT/'error_transport_baselines.py'),('weights_sha256',weight)]:assert report[field]==sha(p),(method,lr,field)
            expected_seed=int(np.random.SeedSequence([20261020,18,20,0,METHODS.index(method)]).generate_state(1)[0]);assert report['initialization_seed']==expected_seed
            assert len(report['all_training_losses'])==2000 and np.isfinite(report['all_training_losses']).all()
            assert [r['step'] for r in report['validation_records']]==[500,1000,1500,2000]
            entries=[]
            expected_lrs=[.003,.01,.03] if method=='pooled' else [.01,.03,.1] if method=='coda' else [.03]
            for checkpoint in report['validation_records']:
                assert [r['adaptation_lr'] for r in checkpoint['validation']]==expected_lrs
                for record in checkpoint['validation']:
                    scores=np.array([record['body_scores'][str(n)] for n in [4,16,64]])
                    assert scores.shape==(3,8) and np.isfinite(scores).all()
                    assert np.isclose(scores.mean(),record['criterion'],rtol=1e-14,atol=1e-14)
                    entries.append(dict(record,step=checkpoint['step']))
            best=min(entries,key=lambda e:e['criterion']);assert best==report['best_validation'],(method,lr,'checkpoint/adaptation selection')
            payload=torch.load(weight,map_location='cpu',weights_only=False)
            assert payload['method']==method and payload['choice']==best
            for key,value in scale.items():
                err=float(np.max(abs(value-payload['scale'][key])));max_scale_error=max(max_scale_error,err);assert err<1e-12
            assert all(torch.isfinite(value).all() for value in payload['state_dict'].values())
            parameter_count=sum(v.numel() for key,v in payload['state_dict'].items() if key!='noise_variance')
            assert parameter_count==report['parameter_count']
            fits.append((report,path));fit_checks.append(dict(method=method,lr=lr,checkpoint=best['step'],adaptation_lr=best['adaptation_lr'],validation_criterion=best['criterion'],parameter_count=report['parameter_count'],fit_wall_seconds=report['wall_seconds']))
        report,path=min(fits,key=lambda p:p[0]['best_validation']['criterion']);choice=saved['choices'][method]
        assert Path(choice['report']).resolve()==path.resolve() and choice['report_sha256']==sha(path)
        assert choice['weights_sha256']==report['weights_sha256'] and choice['validation']==report['best_validation'] and choice['parameter_count']==report['parameter_count']
        best_by_method[method]=dict(training_lr=report['lr'],checkpoint=report['best_validation']['step'],adaptation_lr=report['best_validation']['adaptation_lr'])
    arrays_expected=[];summaries={};max_standardized_error=0.;max_raw_report_error=0.;max_raw_target_quantization_effect=0.;body_metrics={}
    for method in METHODS:
        summaries[method]={};body_metrics[method]={}
        for split_id,label in [(2,'test_id'),(3,'test_ood')]:
            ids=np.flatnonzero(split==split_id);target=normalized[ids,1:].reshape(8,672,2);raw_target=y[ids,1:].reshape(8,672,2)
            for wrong in [False,True]:
                group=label+('_wrong_support' if wrong else '');report=saved['results'][method][group];assert report['body_ids']==ids.tolist()
                summaries[method][group]={};body_metrics[method][group]={}
                for n in BUDGETS:
                    key=f'{method}_{group}_support{n}';arrays_expected.append(key);prediction=predictions[key]
                    assert prediction.shape==target.shape and np.isfinite(prediction).all()
                    # Independent binary64 reduction of actual saved float32
                    # predictions/quantized measured targets. GPU reductions
                    # need not be bit-identical; tolerance scaled to float32.
                    residual=prediction.astype(np.float64)-target.astype(np.float64)
                    standardized=np.mean(residual**2,axis=(1,2));raw=np.mean((residual*scale['y_std'])**2,axis=(1,2))
                    physical=np.mean((prediction.astype(np.float64)*scale['y_std']+scale['y_mean']-raw_target)**2,axis=(1,2))
                    declared_standard=np.array(report['standardized_body_mse'][str(n)]);declared_raw=np.array(report['raw_body_mse'][str(n)])
                    max_standardized_error=max(max_standardized_error,float(np.max(abs(standardized-declared_standard))))
                    max_raw_report_error=max(max_raw_report_error,float(np.max(abs(raw-declared_raw))))
                    max_raw_target_quantization_effect=max(max_raw_target_quantization_effect,float(np.max(abs(physical-raw))))
                    assert np.allclose(standardized,declared_standard,rtol=2e-6,atol=2e-7),(key,'standardized score')
                    assert np.allclose(raw,declared_raw,rtol=2e-6,atol=2e-7),(key,'raw score')
                    summaries[method][group][str(n)]=dict(standardized_mean=float(standardized.mean()),raw_mean=float(raw.mean()),raw_measured_target_mean=float(physical.mean()))
                    body_metrics[method][group][str(n)]=dict(body_ids=ids.tolist(),standardized=standardized.tolist(),raw=raw.tolist(),raw_vs_float64_measured_target=physical.tolist())
                    if n==0:assert np.array_equal(prediction,predictions[f'{method}_{label}_support0'])
    assert set(arrays_expected)==set(predictions.files) and len(arrays_expected)==80
    decisions={}
    for n in [16,64]:
        candidate=summaries['joint']['test_ood'][str(n)]['standardized_mean']
        competitors={method:summaries[method]['test_ood'][str(n)]['standardized_mean'] for method in ['separable','alpaca','coda','pooled']}
        strongest=min(competitors,key=competitors.get);baseline=competitors[strongest]
        wrong=summaries['joint']['test_ood_wrong_support'][str(n)]['standardized_mean']
        decisions[str(n)]=dict(candidate_mean=candidate,strongest_comparator=strongest,comparator_mean=baseline,
            relative_improvement_over_strongest=1-candidate/baseline,passes_10_percent_improvement=bool(candidate<=.9*baseline),
            correct_support_beats_wrong=bool(candidate<wrong),wrong_support_mean=wrong,wrong_minus_correct=wrong-candidate)
    passes=any(d['passes_10_percent_improvement'] for d in decisions.values())
    # Both budgets fail the necessary10% condition, independently of how the
    # no-opposite-loss clause would be operationalized after a winning budget.
    assert passes is False
    out=dict(status='PASS PROPORTIONAL DECISIVE-SCREEN ARITHMETIC/SELECTION CHECK; CANDIDATE RETIREMENT SUPPORTED',
        artifact_sha256={p.name:sha(p) for p in files},train_only_scale={k:v.tolist() for k,v in scale.items()},max_checkpoint_scale_error=max_scale_error,
        fit_selection_checks=fit_checks,selected=best_by_method,summaries=summaries,all_per_body_metrics=body_metrics,
        max_standardized_mse_report_error=max_standardized_error,max_raw_mse_report_error=max_raw_report_error,
        max_raw_target_float32_quantization_effect=max_raw_target_quantization_effect,screening_decisions=decisions,
        prospective_continuation_criterion_met=False,decision='REJECT current residual-conditioned joint operator as the main campaign candidate; no control rollout or precision rescue owed.',
        independent_scope='All80prediction arrays/640body-budget comparisons,10fit reports/checkpoints/weight hashes, normalization and validation-only selection checked; no retraining/network-forward/physics replay.',
        limitations='One trained seed; eight heldout bodies per split from one fully observed noiseless family. No inferential proof or control capability claim. ALPaCA fixed-noise NLL and CoDA transition-loss core have disclosed scope; capacities/adaptation/training compute differ. Correct support benefit does not establish advantage over standard adaptation.')
    Path(__file__).with_name('check_error_transport_screen_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],decisions=decisions,selected=best_by_method,max_standardized_error=max_standardized_error,max_raw_report_error=max_raw_report_error,max_raw_target_quantization_effect=max_raw_target_quantization_effect),indent=2))


if __name__=='__main__':main()
