#!/usr/bin/env python3
"""Independent scientific aggregate audit, no canonical analyzer imports.

Recomputes endpoint scores by complex-link kinematics, all corpus means/paired
SEs, profiles/catalogues/failures/resources and provenance. Does not resimulate
physics or rerun fitting/optimization, which have separate independent audits.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('retained','frozen','privileged_reference')
SCALARS=('task_cost_18sec','first_goal_cost','later_goals_total_cost','later_goal_mean_cost',
    'combined_charged_cost','warmup_motion_effort_cost','task_integrated_squared_effort',
    'warmup_integrated_squared_effort','combined_integrated_squared_effort',
    'precision_fraction','first_precision_fraction','later_precision_fraction','phase_costs','phase_precision_fractions')
MAX_ERROR=0.


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(actual,expected,label=''):
    global MAX_ERROR
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(label,'keys',set(actual)^set(expected))
        for key in expected:check(actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected),(label,'length')
        for index,(a,b) in enumerate(zip(actual,expected)):check(a,b,label+f'/{index}')
    elif isinstance(expected,(str,bool)) or expected is None:assert actual==expected,(label,actual,expected)
    elif isinstance(expected,(int,np.integer)):assert actual==int(expected),(label,actual,expected)
    else:
        actual=float(actual);expected=float(expected);assert math.isfinite(actual) and math.isfinite(expected)
        error=abs(actual-expected);MAX_ERROR=max(MAX_ERROR,error)
        assert error<=1e-10+2e-10*abs(expected),(label,actual,expected,error)


def cluster(values):
    v=np.asarray(values,dtype=float);assert v.shape[0]==8
    mean=np.sum(v,axis=0)/8
    se=np.sqrt(np.sum((v-mean)**2,axis=0)/(8*7))
    return dict(mean=mean.tolist(),se_across_eight_corpora=se.tolist(),min=v.min(0).tolist(),max=v.max(0).tolist(),corpus_values=v.tolist())


def kinematics(states):
    # Independent complex rotating-link calculation rather than Jacobian code.
    proximal=np.exp(1j*states[:,0]);distal=np.exp(1j*(states[:,0]+states[:,1]))
    p=proximal+distal;v=1j*(proximal*states[:,2]+distal*(states[:,2]+states[:,3]))
    return np.c_[p.real,p.imag],np.c_[v.real,v.imag]


def episode(record,arrays,method,rep,result_path):
    keys=record['array_keys'];states=arrays[keys['states_evaluator_only']];commands=arrays[keys['commands']]
    position,velocity=kinematics(states);goals=np.array(record['goals']);effort=np.sum(commands**2,axis=1)
    phases=[];costs=[];efforts=[];precision=[]
    for phase in range(3):
        first=16+60*phase;last=first+60;errors=position[first+1:last+1]-goals[phase];vel=velocity[first+1:last+1]
        pos=.1*float(np.sum(errors**2));motion=.025*float(np.sum(vel**2));e=.1*float(np.sum(effort[first:last]));cost=pos+motion+.01*e
        success=bool(np.all(abs(errors[-10:])<=.03) and np.all(abs(vel[-10:])<=.03))
        raw=record['score']['phases'][phase]
        for key,value in [('cost',cost),('integrated_position_error',pos),('integrated_motion_penalty',motion),('integrated_effort_penalty',.01*e),('precision_success',success)]:check(raw[key],value,'raw phase/'+key)
        for key,value in [('position_error_by_sample',errors),('cartesian_velocity_by_sample',vel),('squared_effort_by_interval',effort[first:last])]:
            check(np.asarray(raw[key]).tolist(),value.tolist(),'raw profile/'+key)
        costs.append(cost);efforts.append(e);precision.append(success)
        phases.append(dict(replicate=rep,variant=method,case_index=int(record['case_index']),context_id=int(record['context_id']),noise_repeat=int(record['noise_repeat']),
            phase=phase,cost=cost,precision_success=success,late_max_coordinate_position_error=float(np.max(abs(errors[-10:]))),late_max_coordinate_velocity=float(np.max(abs(vel[-10:]))),
            result_artifact=str(result_path.resolve()),array_keys=keys))
    warm_effort=.1*float(np.sum(effort[:16]));warm_motion=.025*float(np.sum(velocity[1:17]**2));warm=warm_motion+.01*warm_effort
    task=math.fsum(costs)
    check(record['score']['task_cost'],task,'raw task cost');check(record['score']['combined_charged_cost'],task+warm,'raw combined cost')
    check(record['score']['precise_goals'],sum(precision),'raw precision')
    entry=dict(replicate=rep,variant=method,case_index=int(record['case_index']),context_id=int(record['context_id']),noise_repeat=int(record['noise_repeat']),
        task_cost_18sec=task,phase_costs=costs,first_goal_cost=costs[0],later_goals_total_cost=sum(costs[1:]),later_goal_mean_cost=sum(costs[1:])/2,
        combined_charged_cost=task+warm,warmup_motion_effort_cost=warm,task_integrated_squared_effort=sum(efforts),warmup_integrated_squared_effort=warm_effort,
        combined_integrated_squared_effort=sum(efforts)+warm_effort,phase_precision=precision,precise_goals=sum(precision),result_artifact=str(result_path.resolve()),array_keys=keys)
    return entry,phases


def corpus_mean(entries):
    result={}
    for key in SCALARS[:9]:result[key]=math.fsum(e[key] for e in entries)/8
    bits=np.array([e['phase_precision'] for e in entries],dtype=int)
    result.update(phase_costs=np.sum([e['phase_costs'] for e in entries],axis=0).tolist())
    result['phase_costs']=(np.array(result['phase_costs'])/8).tolist()
    result.update(phase_precision_fractions=(bits.sum(0)/8).tolist(),precision_fraction=float(bits.sum()/24),first_precision_fraction=float(bits[:,0].sum()/8),later_precision_fraction=float(bits[:,1:].sum()/16),
        precision_count=int(bits.sum()),first_precision_count=int(bits[:,0].sum()),later_precision_count=int(bits[:,1:].sum()))
    return result


def main():
    aggregate_path=ROOT/'cartesian_stream_v17_study_aggregate_results.json';saved=json.loads(aggregate_path.read_text())
    assert saved['status']=='COMPLETE EIGHT-CORPUS AGGREGATE WITH BOTH INDEPENDENT AUDIT GATES VERIFIED' and saved['publication_eligible'] is True
    assert saved['analysis_source_sha256']==sha(ROOT/'analyze_cartesian_stream_v17_study.py')
    assert saved['replicates']==list(range(200,208))
    for name,digest in saved['artifact_sha256'].items():assert sha(name)==digest,('stale aggregate input',name)
    assert len(saved['corpora'])==8
    all_entries={m:[] for m in METHODS};all_phases=[];decisions=[];means=[];resources=[];forecast_metrics=[]
    selections=[];wall_acq=[];wall_fit=[];wall_study=[];ordinary_audits=[]
    for rep in range(200,208):
        prefix=ROOT/f'cartesian_stream_v17_rep{rep}';rp=Path(str(prefix)+'_study_results.json');ap=Path(str(prefix)+'_study_arrays.npz')
        fp=Path(str(prefix)+'_local_fit.json');cp=Path(str(prefix)+'_acquisition.json');ep=Path(str(prefix)+'_evaluator.npz');tp=Path(str(prefix)+'_local_test_forecasts.npz')
        result=json.loads(rp.read_text());fit=json.loads(fp.read_text());acq=json.loads(cp.read_text());arrays=np.load(ap,allow_pickle=False);ev=np.load(ep,allow_pickle=False);raw=np.load(tp,allow_pickle=False)
        corpus=saved['corpora'][rep-200];assert corpus['replicate']==result['replicate']==fit['replicate']==rep
        assert result['dependency_sha256']==saved['frozen_common_dependency_sha256']
        for name,digest in result['dependency_sha256'].items():assert sha(ROOT/name)==digest
        for key,path in [('results',rp),('arrays',ap),('fit',fp),('acquisition',cp),('evaluator',ep),('forecasts',tp),('ordinary',Path(str(prefix)+'_ordinary.npz'))]:
            assert corpus['artifact_sha256'][key]==sha(path) and Path(corpus['artifact_paths'][key]).resolve()==path.resolve()
        assert len(corpus['audit_gates'])==2 and len({g['path'] for g in corpus['audit_gates']})==2
        for gate in corpus['audit_gates']:
            audit_path=Path(gate['path']);audit=json.loads(audit_path.read_text());assert audit['replicate']==rep and audit['status'].startswith('PASS') and sha(audit_path)==gate['sha256']
            for name,digest in audit['artifact_sha256'].items():
                explicit=audit.get('artifact_paths',{});path=Path(explicit[name]) if name in explicit else Path(name)
                if not path.is_absolute():path=ROOT/path
                assert sha(path)==digest,(rep,'stale per-corpus gate',name)
            if audit_path.name==f'audit_cartesian_stream_v17_study_rep{rep}_results.json':
                assert audit['artifact_sha256']['audit_cartesian_stream_v17_study.py']=='507d3bd427a661a57df07314f10f083bc55d35435225253a250d962b083989d7';ordinary_audits.append(audit)
        entries={m:[] for m in METHODS};phase_entries=[]
        for record in result['ordinary_episodes']+result['witnesses']:
            method=record.get('variant','privileged_reference');entry,phases=episode(record,arrays,method,rep,rp)
            entries[method].append(entry);all_entries[method].append(entry);phase_entries.extend(phases);all_phases.extend(phases)
            if method=='privileged_reference':continue
            for d in record['decisions']:
                plan=np.array(d['selected_plan']);gradient=np.array(d['selected_gradient']);pg=plan-np.clip(plan-gradient,-.5,.5)
                check(d['selected_projected_gradient_max'],float(np.max(abs(pg))),'PG arithmetic')
                decisions.append(dict(replicate=rep,variant=method,case_index=int(record['case_index']),renewal=int(d['renewal']),phase=int(d['phase']),time_index=int(d['time_index']),
                    optimizer_success=bool(d['optimizer']['success']),optimizer_status=d['optimizer']['status'],optimizer_message=d['optimizer']['message'],nfev=d['optimizer']['nfev'],njev=d['optimizer']['njev'],nit=d['optimizer']['nit'],
                    fallback=bool(d['fallback_to_initial']),projected_gradient_max=float(d['selected_projected_gradient_max']),modeled_value_gradient_calls=int(d['modeled_value_gradient_calls']),choose_wall_seconds=float(d['total_choose_wall_seconds']),result_artifact=str(rp.resolve())))
        cmeans={method:corpus_mean(value) for method,value in entries.items()};means.append(cmeans);check(corpus['means'],cmeans,'corpus means')
        selected=int(np.argmin([c['validation']['member_coordinate_mse'] for c in fit['candidates']]))
        assert selected==fit['selected_local_candidate']==corpus['selected_local_candidate'];selections.append(selected)
        check(corpus['validation_member_mse'],[c['validation']['member_coordinate_mse'] for c in fit['candidates']],'val scores')
        check(corpus['selected_configuration'],dict(neighbors=fit['candidates'][selected]['neighbors'],ridge=fit['candidates'][selected]['ridge']),'config')
        fmetrics={}
        for label in ['one_block','six_block']:
            prediction=raw[label+'_member_predictions'];target=raw[label+'_observed_targets'];error=prediction-target[:,None]
            fmetrics[label]=dict(member_coordinate_mse=float(np.mean(error**2)),ensemble_mean_coordinate_mse=float(np.mean((prediction.mean(1)-target)**2)),
                member_mse_by_future_sample=np.mean(error**2,axis=(0,1,3)).tolist(),origins=int(len(target)))
        check(corpus['heldout_forecasts'],fmetrics,'heldout metrics');forecast_metrics.append(fmetrics)
        ordinary=result['ordinary_episodes'];references=result['witnesses'];drows=[d for d in decisions if d['replicate']==rep]
        r=dict(acquired_trajectories=40,acquisition_intervals=int(ev['plant_nfev'].size),acquisition_scalar_camera_draws=int(ev['sensor_noise'].size),acquisition_plant_rhs_evaluations=int(ev['plant_nfev'].sum()),
            ordinary_episodes=len(ordinary),ordinary_goal_windows=sum(len(e['score']['phases']) for e in ordinary),ordinary_decisions=len(drows),ordinary_intervals=sum(len(arrays[e['array_keys']['commands']]) for e in ordinary),
            ordinary_camera_read_opportunities=sum(len(arrays[e['array_keys']['observed']]) for e in ordinary),ordinary_plant_rhs_evaluations=sum(int(arrays[e['array_keys']['plant_nfev']].sum()) for e in ordinary),
            initial_preparations=len({tuple(e['initial_seed_path']) for e in ordinary}),paired_noise_paths=len({tuple(e['noise_seed_path']) for e in ordinary}),distinct_scalar_control_noise_draws=sum(int(arrays[f'case{i}_sensor_noise'].size) for i in range(8)),
            retained_actual_row_events=sum(e['actual_local_row_events'] for e in ordinary if e['variant']=='retained'),retained_member_append_copies=sum(sum(f-i for f,i in zip(e['local_library_final_rows'],e['local_library_initial_rows'])) for e in ordinary if e['variant']=='retained'),
            reference_episodes=len(references),privileged_reference_intervals=sum(len(arrays[e['array_keys']['commands']]) for e in references),reference_plant_rhs_evaluations=sum(int(arrays[e['array_keys']['plant_nfev']].sum()) for e in references),
            modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for d in drows),unsuccessful_optimizer_terminations=sum(not d['optimizer_success'] for d in drows),fallback_decisions=sum(d['fallback'] for d in drows),neural_fits=0)
        r.update(acquisition_simulated_seconds=.1*r['acquisition_intervals'],ordinary_simulated_seconds=.1*r['ordinary_intervals'],ordinary_scalar_noise_opportunities=2*r['ordinary_camera_read_opportunities'],privileged_reference_seconds=.1*r['privileged_reference_intervals'])
        check(corpus['resources'],r,'corpus resources');resources.append(r)
        wall_acq.append(acq['wall_seconds']);wall_fit.append(fit['wall_seconds']);wall_study.append(result['study_invocation_wall_seconds_excluding_separate_acquisition'])
    assert len(ordinary_audits)==8 and len(decisions)==11520
    summary={}
    for method in METHODS:
        summary[method]={key:cluster([m[method][key] for m in means]) for key in SCALARS}
        summary[method].update(episodes=64,precision_denominator=192,first_precision_denominator=64,later_precision_denominator=128,
            precision_count=sum(m[method]['precision_count'] for m in means),first_precision_count=sum(m[method]['first_precision_count'] for m in means),later_precision_count=sum(m[method]['later_precision_count'] for m in means))
    check(saved['summary'],summary,'all summaries')
    paired={key:cluster([np.array(m['frozen'][key])-np.array(m['retained'][key]) for m in means]) for key in SCALARS}
    check(saved['paired_frozen_minus_retained'],paired,'all paired corpus SEs')
    fm={label:{key:cluster([c[label][key] for c in forecast_metrics]) for key in ['member_coordinate_mse','ensemble_mean_coordinate_mse','member_mse_by_future_sample']} for label in ['one_block','six_block']}
    check(saved['ordinary_heldout_forecasts'],fm,'all forecast corpus summaries')
    for method in METHODS:check(saved['episode_catalog'][method],all_entries[method],'complete episode catalogue')
    failures=[phase for phase in all_phases if not phase['precision_success']]
    assert len(failures)==310 and all(f['variant']!='privileged_reference' for f in failures)
    assert len({(f['replicate'],f['variant'],f['case_index'],f['phase']) for f in failures})==310
    check(saved['failures'],dict(precision_failures=failures,unsuccessful_optimizer_terminations=[d for d in decisions if not d['optimizer_success']],fallback_decisions=[d for d in decisions if d['fallback']]),'all failure catalogues')
    for method in METHODS:
        check(saved['worst_episodes'][method],sorted(all_entries[method],key=lambda e:e['task_cost_18sec'],reverse=True)[:20],'worst episodes')
        check(saved['worst_goal_phases'][method],sorted([p for p in all_phases if p['variant']==method],key=lambda p:p['cost'],reverse=True)[:20],'worst phases')
    optimizer={}
    for method in ['retained','frozen']:
        ds=[d for d in decisions if d['variant']==method];values=[d['projected_gradient_max'] for d in ds]
        optimizer[method]=dict(decisions=len(ds),unsuccessful_terminations=sum(not d['optimizer_success'] for d in ds),fallback_decisions=sum(d['fallback'] for d in ds),
            status_counts=dict(Counter(str(d['optimizer_status']) for d in ds)),message_counts=dict(Counter(d['optimizer_message'] for d in ds)),
            projected_gradient_quantiles={str(q):float(np.quantile(values,q)) for q in [0.,.5,.9,.99,1.]},projected_gradient_corpus_maxima=[max(d['projected_gradient_max'] for d in ds if d['replicate']==rep) for rep in range(200,208)],
            worst_projected_gradients=sorted(ds,key=lambda d:d['projected_gradient_max'],reverse=True)[:20],total_modeled_value_gradient_calls=sum(d['modeled_value_gradient_calls'] for d in ds),summed_choose_wall_seconds=math.fsum(d['choose_wall_seconds'] for d in ds))
    check(saved['optimizer'],optimizer,'optimizer summaries')
    totals={key:sum(r[key] for r in resources) for key in resources[0]}
    totals.update(acquired_corpora=8,fixed_physical_body_families=1,total_physical_intervals_acquisition_ordinary_reference=totals['acquisition_intervals']+totals['ordinary_intervals']+totals['privileged_reference_intervals'],distinct_scalar_rng_draws_acquisition_and_control=totals['acquisition_scalar_camera_draws']+totals['distinct_scalar_control_noise_draws'])
    check(saved['resources'],totals,'all resources');check(saved['selected_local_menu_indices'],selections,'selection list');check(saved['selected_local_menu_counts'],dict(Counter(map(str,selections))),'selection counts')
    check(saved['timing']['per_corpus_acquisition_seconds'],wall_acq,'acquisition timing');check(saved['timing']['per_corpus_fit_seconds'],wall_fit,'fit timing');check(saved['timing']['per_corpus_study_invocation_seconds'],wall_study,'invocation timing')
    artifacts=dict(saved['artifact_sha256']);artifacts[str(aggregate_path.resolve())]=sha(aggregate_path);artifacts[str(Path(__file__).resolve())]=sha(Path(__file__))
    result=dict(status='PASS INDEPENDENT SCIENTIFIC AGGREGATE/RESOURCE/FAILURE/HASH AUDIT',artifact_sha256=artifacts,
        final_ordinary_auditor_sha256=sha(ROOT/'audit_cartesian_stream_v17_study.py'),replicates=list(range(200,208)),maximum_numeric_reconstruction_error=MAX_ERROR,
        verified_summary=summary,verified_paired_frozen_minus_retained=paired,verified_resources=totals,
        all_failed_ordinary_windows_verified=310,failed_window_identifiers=[[f['replicate'],f['variant'],f['case_index'],f['phase']] for f in failures],
        original_coefficient_failures_preserved=1,logged_secondary_certificates=sum(len(a['secondary_coefficient_roundoff_certificates']) for a in ordinary_audits),
        limitations='Independent complex-kinematics score/statistics reconstruction, no physical resimulation here. Eight corpus-clustered descriptive SEs; no p-values or universal claim. Physics/RNG, learned fits/decisions and convex-optimizer certificates have separate audited scopes.310failed windows remain retained; numerical replay resolution does not rescue robustness.')
    (ROOT/'audit_cartesian_stream_v17_study_aggregate_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS scientific aggregate, all310failures; maximum numeric error',MAX_ERROR)


if __name__=='__main__':main()
