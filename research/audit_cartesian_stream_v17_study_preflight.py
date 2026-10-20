#!/usr/bin/env python3
"""Synthetic-only fresh-study preflight: no fresh study RNG draws/plant flows.

Tests wrapper transparency/selection/failure recording and seed-path placement
using temporary synthetic corpora, a deterministic fake predictor and fake RNG.
Actual physical/model performance is deliberately outside this fixture.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from audit_cartesian_stream_v17_fit import ROOT,sha,rows


def main():
    source=ROOT/'cartesian_stream_v17_study.py'
    assert sha(source)=='a213d5eb27ab1736fe0a855787f8d61293a5a15f52c7d773ba4d3634454a1754'
    spec=importlib.util.spec_from_file_location('fresh_wrapper_under_test',source)
    unit=importlib.util.module_from_spec(spec);spec.loader.exec_module(unit)
    import cartesian_stream_v17_control as control
    def forbidden(*args,**kwargs):raise AssertionError('No physical execution in fresh preflight')
    control.flow=forbidden
    torch.set_num_threads(1)
    # This seed is an unrelated synthetic fixture, not650/660/670 study entropy.
    fixture=np.random.default_rng(817349)
    observed=fixture.normal(0,.003,(40,257,2)).cumsum(axis=1)
    commands=np.concatenate([np.zeros((40,8,2)),fixture.uniform(-.5,.5,(40,124,2)).repeat(2,axis=1)],axis=1)
    split=np.repeat([0,1,2],[24,8,8]);trajectory=np.r_[np.arange(24),np.arange(8),np.arange(8)]
    seed_paths=[]
    class FakeGenerator:
        def __init__(self,sequence):seed_paths.append(list(sequence.entropy))
        def uniform(self,low,high,size=None):return (low+high)/2 if size is None else np.full(size,(low+high)/2)
        def normal(self,loc,scale,size):return np.full(size,loc)
        def choice(self,values,size=None,replace=True):return 1. if size is None else np.resize(np.asarray(values),size)
    original_rng=np.random.default_rng
    class SyntheticPredictor:
        def __init__(self,config,fail=False):self.config=config;self.calls=0;self.fail=fail
        def paths(self,o,a,plan):
            self.calls+=1
            if self.fail and self.calls==3:raise RuntimeError('intentional synthetic validation failure')
            # Genuine causal dependence; available prefix and proposed commands
            # only. Three coherent distinct fixed synthetic members.
            last=torch.tensor(o[-1]);slope=torch.tensor(o[-1]-o[-2]);outputs=[]
            for member in range(3):
                future=[]
                for k,u in enumerate(plan):
                    for j in [1,2]:future.append(last+(2*k+j)*slope+.001*(member+1)*u)
                outputs.append(torch.stack(future))
            return torch.stack(outputs)
    captures=[];fail_next=[False]
    def local_factory(X,U,Y,bootstrap,scale,config):
        captures.append((X.copy(),U.copy(),Y.copy(),[b.copy() for b in bootstrap],config))
        return SyntheticPredictor(config,fail_next[0])
    unit.local_predictor=local_factory
    with tempfile.TemporaryDirectory(prefix='v17-study-synthetic-preflight-') as directory:
        def corpus(name):
            prefix=Path(directory)/name;p=unit.paths(prefix)
            np.savez_compressed(p['ordinary'],observed=observed,commands=commands,split=split,trajectory_index=trajectory,dt=.1,sensor_sd=.0002)
            # Deliberately NOT an NPZ: hashing is allowed; opening as evaluator
            # arrays would fail and thus expose accidental feature access.
            p['evaluator'].write_bytes(b'synthetic opaque evaluator provenance only')
            meta=dict(replicate=200,status='EXPLORATORY MOVING ACQUISITION; NO LEARNER OR RELEASE',
                source_sha256=sha(ROOT/'cartesian_stream_v17_acquire.py'),plant_source_sha256=sha(ROOT/'coupled_context_v14_pilot.py'),
                design_sha256=sha(ROOT/'cartesian_stream_v17_acquisition_design.md'),counts=[24,8,8],steps=256,dt=.1,
                artifacts={p[k].name:sha(p[k]) for k in ['ordinary','evaluator']},resources={})
            p['acquisition'].write_text(json.dumps(meta));return prefix,p,meta
        prefix,p,meta=corpus('success')
        unit.verify_acquisition(prefix,200)
        meta['replicate']=201;p['acquisition'].write_text(json.dumps(meta))
        rejected=False
        try:unit.verify_acquisition(prefix,200)
        except AssertionError:rejected=True
        assert rejected
        meta['replicate']=200;p['acquisition'].write_text(json.dumps(meta))
        np.random.default_rng=FakeGenerator
        try:
            cases_by_rep={rep:unit.make_cases(rep) for rep in range(200,208)}
            result=unit.fit_local(prefix,200)
        finally:np.random.default_rng=original_rng
        assert result['status'].startswith('COMPLETE') and result['selected_local_candidate']==0
        assert len(result['candidates'])==4 and all(r['validation']['origins']==80 for r in result['candidates'])
        assert result['test_observed_scores']['one_block']['origins']==960 and result['test_observed_scores']['six_block']['origins']==80
        def reference_prediction(i,t,n):
            return np.array([[observed[i,t]+k*(observed[i,t]-observed[i,t-1])+
                .001*(member+1)*commands[i,t+2*((k-1)//2)] for k in range(1,n+1)] for member in range(3)])
        wrapper_error=0.
        for candidate in result['candidates']:
            losses=[]
            for record in candidate['validation']['records']:
                i=record['trajectory'];t=record['time_index'];prediction=reference_prediction(i,t,12)
                expected=float(np.mean((prediction-observed[i,t+1:t+13])**2));losses.append(expected)
                wrapper_error=max(wrapper_error,abs(expected-record['member_coordinate_mse']))
            wrapper_error=max(wrapper_error,abs(np.mean(losses)-candidate['validation']['member_coordinate_mse']))
        assert wrapper_error<1e-14
        assert len(captures)==5 # Four candidate constructors, selected-only test.
        X,U,Y,origins=rows(observed,commands,np.arange(24))
        scale=dict(context_mean=X.mean(0),context_std=np.maximum(X.std(0),.01),target_mean=Y.mean(0),target_std=np.maximum(Y.std(0),.001))
        for xx,uu,yy,bootstrap,config in captures:
            assert np.allclose(xx,(X-scale['context_mean'])/scale['context_std'],rtol=0,atol=1e-14)
            assert np.array_equal(uu,U) and np.allclose(yy,(Y-scale['target_mean'])/scale['target_std'],rtol=0,atol=1e-12)
            assert all(np.array_equal(b,np.arange(2880)) for b in bootstrap)
        scale_error=max(float(np.max(abs(scale[k]-result['scale'][k]))) for k in scale)
        assert scale_error<1e-12
        raw=np.load(p['test_forecasts'],allow_pickle=False)
        for label in ['one_block','six_block']:
            for index,(i,t) in enumerate(raw[label+'_origins']):
                n=2 if label=='one_block' else 12
                assert np.array_equal(raw[label+'_observed_targets'][index],observed[i,t+1:t+n+1])
                assert np.allclose(raw[label+'_member_predictions'][index],reference_prediction(i,t,n),rtol=0,atol=1e-14)
        # A synthetic interrupted validation must retain the active third origin,
        # two prior records and finite/partial fit metadata; no real training.
        failed_prefix,fp,_=corpus('failure');fail_next[0]=True
        np.random.default_rng=FakeGenerator
        try:
            try:unit.fit_local(failed_prefix,200)
            except RuntimeError as error:assert 'intentional synthetic' in str(error)
            else:raise AssertionError('Expected synthetic failure')
        finally:np.random.default_rng=original_rng
        failure=json.loads(fp['fit_failure'].read_text())
        assert failure['status'].startswith('FAILED') and failure['active_validation_origin']==dict(trajectory=24,time_index=64)
        assert len(failure['active_validation_records'])==2 and 'active_validation_prediction' not in failure
    # Validate all replicate slots using fake deterministic generators only.
    for rep,cases in cases_by_rep.items():
        assert len(cases)==8
        for ci,case in enumerate(cases):
            context_id=ci//2;repeat=ci%2
            assert case['initial_seed_path']==[20261020,17,670,rep,context_id,0]
            assert case['warmup_seed_path']==[20261020,17,670,rep,context_id,2]
            assert case['noise_seed_path']==[20261020,17,670,rep,context_id,1,repeat]
            assert case['warmup_commands'].shape==(16,2) and case['sensor_noise'].shape==(197,2)
    assert [20261020,17,660,200,0,0] in seed_paths and [20261020,17,660,200,0,2] in seed_paths
    files=[source,ROOT/'cartesian_stream_v17_study_design.md',ROOT/'cartesian_stream_v17_learn.py',ROOT/'cartesian_stream_v17_local.py',ROOT/'cartesian_stream_v17_control.py',ROOT/'cartesian_stream_v17_retention.py',ROOT/'cartesian_stream_v17_acquire.py']
    audit=dict(status='PASS FRESH-STUDY SYNTHETIC PREFLIGHT; NO FRESH STUDY RNG DRAWS/PHYSICS',artifact_sha256={p.name:sha(p) for p in files},synthetic_scale_max_error=scale_error,
        validation_origins_per_candidate=80,candidates=4,selected_test_origins=dict(one_block=960,six_block=80),mismatched_acquisition_replicate_rejected=rejected,
        opaque_non_npz_evaluator_hashed_without_feature_open=True,synthetic_validation_wrapper_max_score_error=wrapper_error,
        fake_rng_replicate_paths_checked=list(range(200,208)),partial_failure_origin_and_two_prior_records_preserved=True,
        limitations='Synthetic fake predictor/fake RNG exercise wrapper plumbing, not model performance. No actual200..207 corpus, fresh random values, fit, witness or control sampled. Inherited math previously independently audited; actual eight-corpus artifacts require independent replay.')
    (ROOT/'audit_cartesian_stream_v17_study_preflight_results.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
