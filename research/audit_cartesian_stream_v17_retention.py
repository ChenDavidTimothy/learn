#!/usr/bin/env python3
"""Independent frozen-library decisions; scope frozen before ablation outcomes.

No canonical implementation imports or physical state reads. Reconstructs every
fixed-library local fit/plan derivative and arithmetic pairing to audited online
records. Separate physics auditor owns streams/true sampled scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cartesian_stream_v17_fit import ROOT,sha,rows,context,paths
from audit_cartesian_stream_v17_control import fit_local
from audit_cartesian_stream_v17_precontrol import cost_gradient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default=str(ROOT/'cartesian_stream_v17_rep0'));args=parser.parse_args();prefix=Path(args.prefix)
    resultpath=Path(str(prefix)+'_retention_results.json');arraypath=Path(str(prefix)+'_retention_arrays.npz')
    parentpath=Path(str(prefix)+'_control_results.json');parentarray=Path(str(prefix)+'_control_arrays.npz')
    fitpath=Path(str(prefix)+'_fit.json');ordinary=Path(str(prefix)+'_ordinary.npz');weights=Path(str(prefix)+'_weights.npz')
    saved=json.loads(resultpath.read_text());parent=json.loads(parentpath.read_text());fit=json.loads(fitpath.read_text())
    assert saved['status']=='COMPLETE FROZEN ACQUIRED-LIBRARY ABLATION; INDEPENDENT AUDIT PENDING','Refusing incomplete ablation'
    assert saved['source_sha256']==sha(ROOT/'cartesian_stream_v17_retention.py')
    assert saved['design_sha256']==sha(ROOT/'cartesian_stream_v17_retention_design.md') and saved['arrays_sha256']==sha(arraypath)
    assert saved['parent_control_results_sha256']==sha(parentpath) and saved['parent_control_source_sha256']==sha(ROOT/'cartesian_stream_v17_control.py')
    assert parent['ordinary_sha256']==sha(ordinary) and parent['weights_sha256']==sha(weights) and parent['fit_sha256']==sha(fitpath) and parent['arrays_sha256']==sha(parentarray)
    for name,digest in parent['dependency_sha256'].items():assert digest==sha(ROOT/name)
    data=np.load(ordinary,allow_pickle=False);arrays=np.load(arraypath,allow_pickle=False);online_arrays=np.load(parentarray,allow_pickle=False)
    X,U,Y,origins=rows(data['observed'],data['commands'],np.flatnonzero(data['split']==0));scale={k:np.array(v) for k,v in fit['scale'].items()}
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    bs=[np.array(b) for b in fit['bootstrap_rows']];libraries=[(X[b],U[b],Y[b]) for b in bs]
    cfg=fit['local_candidates'][fit['selected']['local']]
    errors=dict(query=0.,coefficient=0.,fit_diagnostic=0.,fit_stationarity=0.,cost=0.,gradient=0.,projected_gradient=0.)
    counts=dict(episodes=8,decisions=0,member_fits=0,modeled_value_gradient_calls=0,optimizer_unsuccessful=0,fallbacks=0,actual_appended_rows=0)
    episodes=[];all_pg=[];worst_pg=None
    assert len(saved['episodes'])==8
    with threadpool_limits(limits=1):
        for ei,record in enumerate(saved['episodes']):
            assert record['case_index']==ei and record['kind']=='local_frozen_library' and len(record['decisions'])==90
            assert record['real_pair_notifications']==90 and record['actual_local_rows_appended']==0
            assert record['local_library_initial_rows']==[2880]*3 and record['local_library_final_rows']==[2880]*3
            assert record['selected_configuration']==dict(neighbors=cfg['neighbors'],ridge=cfg['ridge'])
            paired=next(e for e in parent['ordinary_episodes'] if e['kind']=='local' and e['case_index']==ei)
            keys=record['array_keys'];pk=paired['array_keys']
            observed=arrays[keys['observed']];commands=arrays[keys['commands']]
            assert observed.shape==(197,2) and commands.shape==(196,2) and np.isfinite(observed).all() and np.isfinite(commands).all()
            assert np.max(abs(commands))<=.5
            assert np.array_equal(observed[:17],online_arrays[pk['observed']][:17]) and np.array_equal(commands[:16],online_arrays[pk['commands']][:16])
            assert record['initial_seed_path']==paired['initial_seed_path'] and record['warmup_seed_path']==paired['warmup_seed_path'] and record['noise_seed_path']==paired['noise_seed_path']
            g=.95*observed[16];c=np.cos(.04);s=np.sin(.04)
            goals=np.array([g,[c*g[0]-s*g[1],s*g[0]+c*g[1]],[c*g[0]+s*g[1],-s*g[0]+c*g[1]]])
            assert np.allclose(goals,record['goals'],rtol=0,atol=1e-13) and np.array_equal(record['goals'],paired['goals'])
            coef_saved=arrays[keys['local_fit_coefficients']];query_saved=arrays[keys['local_fit_queries']]
            assert coef_saved.shape==(90,3,69,4) and query_saved.shape==(90,66)
            previous=np.zeros((6,2));max_pg=0.;unsuccessful=0;fallbacks=0
            for di,d in enumerate(record['decisions']):
                t=16+2*di;phase=di//30;assert (d['renewal'],d['phase'],d['time_index'])==(di,phase,t)
                assert np.allclose(d['current_goal'],goals[phase],rtol=0,atol=1e-13)
                obs=observed[:t+1];a=commands[:t];query=(context(obs,a)-scale['context_mean'])/scale['context_std']
                qe=float(np.max(abs(query-query_saved[di])));errors['query']=max(errors['query'],qe);assert qe<1e-12
                models=[]
                for member,(xl,ul,yl) in enumerate(libraries):
                    coefficient,diagnostic,stationarity=fit_local(xl,ul,yl,query,cfg['neighbors'],cfg['ridge']);models.append(coefficient)
                    errors['fit_stationarity']=max(errors['fit_stationarity'],stationarity)
                    ce=float(np.max(abs(coefficient-coef_saved[di,member])));errors['coefficient']=max(errors['coefficient'],ce)
                    assert np.allclose(coefficient,coef_saved[di,member],rtol=2e-8,atol=2e-10),(ei,di,member,'coefficient',ce)
                    sd=d['local_fit_diagnostics'][member];assert sd['n_rows']==2880
                    for key,value in diagnostic.items():
                        err=float(np.max(abs(np.array(value)-sd[key])));errors['fit_diagnostic']=max(errors['fit_diagnostic'],err)
                        assert np.allclose(value,sd[key],rtol=2e-8,atol=2e-10),(ei,di,member,key,err)
                    assert sd['normal_equation_max_residual']<1e-8;counts['member_fits']+=1
                def evaluate(plan):
                    p=np.array(plan);yp,jp=paths(obs,a,p,scale,models,'local',True)
                    return cost_gradient(yp,jp,obs[-1],goals[phase],p)
                shifted=np.r_[previous[1:],np.zeros((1,2))];zero=np.zeros((6,2))
                assert np.array_equal(shifted,d['shifted_plan']) and np.array_equal(zero,d['zero_plan'])
                sc,sg=evaluate(shifted);zc,zg=evaluate(zero)
                for value,gradient,name,flag in [(sc,sg,'shifted_cost','shifted_finite'),(zc,zg,'zero_cost','zero_finite')]:
                    finite=bool(np.isfinite(value) and np.isfinite(gradient).all());assert finite==d[flag]
                    if finite:
                        err=abs(value-d[name]);errors['cost']=max(errors['cost'],err);assert np.isclose(value,d[name],rtol=2e-8,atol=2e-9)
                choose_shift=d['shifted_finite'] and (not d['zero_finite'] or d['shifted_cost']<=d['zero_cost'])
                initial=shifted if choose_shift else zero;initial_cost=sc if choose_shift else zc
                assert d['initial_competitor']==('shifted_previous_plan' if choose_shift else 'zero_plan') and np.array_equal(initial,d['initial_plan'])
                assert np.isclose(initial_cost,d['initial_cost'],rtol=2e-8,atol=2e-9)
                opt=d['optimizer'];candidate=opt['final_plan']
                if candidate is None:
                    assert opt['success'] is False;accept=False
                else:
                    candidate=np.array(candidate);feasible=bool(np.isfinite(candidate).all() and np.max(abs(candidate))<=.5)
                    assert feasible==opt['final_feasible'];cc,cg=evaluate(candidate)
                    if opt['final_finite']:
                        assert np.isfinite(cc) and np.isfinite(cg).all();err=abs(cc-opt['final_cost']);errors['cost']=max(errors['cost'],err)
                        assert np.isclose(cc,opt['final_cost'],rtol=2e-8,atol=2e-9)
                    accept=opt['final_feasible'] and opt['final_finite'] and opt['final_cost']<=d['initial_cost']+1e-12
                    assert d['modeled_value_gradient_calls']==opt['nfev']+3
                selected=np.array(d['selected_plan']);assert np.array_equal(selected,candidate if accept else initial) and d['fallback_to_initial']==(not accept)
                cost,gradient=evaluate(selected);err=abs(cost-d['selected_cost']);errors['cost']=max(errors['cost'],err)
                assert np.isclose(cost,d['selected_cost'],rtol=2e-8,atol=2e-9) and cost<=initial_cost+2e-8*max(1.,abs(initial_cost))
                ge=float(np.max(abs(gradient-d['selected_gradient'])));errors['gradient']=max(errors['gradient'],ge)
                assert np.allclose(gradient,d['selected_gradient'],rtol=5e-7,atol=2e-8)
                pg=selected-np.clip(selected-gradient,-.5,.5);pe=float(np.max(abs(pg-d['selected_projected_gradient'])));errors['projected_gradient']=max(errors['projected_gradient'],pe)
                assert np.allclose(pg,d['selected_projected_gradient'],rtol=5e-7,atol=2e-8) and np.isclose(np.max(abs(pg)),d['selected_projected_gradient_max'],rtol=5e-7,atol=2e-8)
                pgn=float(np.max(abs(pg)));all_pg.append(pgn);max_pg=max(max_pg,pgn)
                if worst_pg is None or pgn>worst_pg['value']:worst_pg=dict(value=pgn,case_index=ei,renewal=di,time_index=t,optimizer_message=opt['message'])
                assert np.array_equal(commands[t],selected[0]) and np.array_equal(commands[t+1],selected[0]) and np.array_equal(d['command'],selected[0])
                if di==0:assert np.array_equal(selected,np.array(paired['decisions'][0]['selected_plan'])),'first decision must match before online additions'
                previous=selected.copy();counts['decisions']+=1;counts['modeled_value_gradient_calls']+=d['modeled_value_gradient_calls'];unsuccessful+=int(not opt['success']);fallbacks+=int(d['fallback_to_initial'])
            # Arithmetic pairing only: independent physical scoring is separate.
            assert record['paired_online_task_cost']==paired['score']['task_cost']
            difference=record['score']['task_cost']-paired['score']['task_cost'];assert difference==record['frozen_minus_online_task_cost']
            counts['optimizer_unsuccessful']+=unsuccessful;counts['fallbacks']+=fallbacks
            episodes.append(dict(case_index=ei,decisions=90,optimizer_unsuccessful=unsuccessful,fallbacks=fallbacks,max_projected_gradient=max_pg,
                frozen_minus_online_recorded_task_cost=difference,recorded_frozen_precision=record['score']['precise_goals'],recorded_online_precision=paired['score']['precise_goals']))
            print('audited frozen library case',ei,'decisions90',flush=True)
    assert counts['decisions']==720 and counts['member_fits']==2160 and counts['modeled_value_gradient_calls']==saved['resources']['modeled_value_gradient_calls']
    files=[ROOT/'cartesian_stream_v17_retention.py',ROOT/'cartesian_stream_v17_retention_design.md',resultpath,arraypath,parentpath,parentarray,fitpath,ordinary,weights,ROOT/'audit_cartesian_stream_v17_control.py',ROOT/'audit_cartesian_stream_v17_fit.py',ROOT/'audit_cartesian_stream_v17_precontrol.py']
    result=dict(status='PASS INDEPENDENT FROZEN-LIBRARY ORDINARY DECISION REPLAY',artifact_sha256={p.name:sha(p) for p in files},maximum_absolute_errors=errors,counts=counts,episodes=episodes,
        selected_projected_gradient_quantiles=dict(zip(map(str,[0,.5,.9,.99,1]),np.quantile(all_pg,[0,.5,.9,.99,1]).tolist())),worst_projected_gradient=worst_pg,
        limitations='No physical state array or canonical implementation imported. Same original libraries reconstructed at all720 decisions, no online additions. Every candidate/selected modeled cost and selected derivative verified; optimizer internal iterates not rerun. Recorded task-cost pairing checked only arithmetically; separate physics auditor verifies scores/streams. Single corpus/four preparations, not independent population evidence.')
    (ROOT/'audit_cartesian_stream_v17_retention_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
