#!/usr/bin/env python3
"""Audit all fresh continuous-history replicates with separate implementation.

Uses only our independent pilot-audit helper, never canonical research imports.
--replicate N audits one complete file; default audits/reuses all100..107 and
produces the complete aggregate on stdout. Per-replicate audit files are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from audit_continuous_history_v15_pilot import main as audit_one

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a=np.array(values)
    return dict(mean=float(a.mean()),mc_se=float(a.std(ddof=1)/np.sqrt(len(a))),
        median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int);args=parser.parse_args()
    helper_hash=sha(ROOT/'audit_continuous_history_v15_pilot.py')
    reports=[];originals=[]
    for rep in ([args.replicate] if args.replicate is not None else range(100,108)):
        name=f'continuous_history_v15_rep{rep}.json';path=ROOT/name
        original=json.loads(path.read_text());assert 'resources' in original and len(original['controls'])==6
        cache=ROOT/f'audit_continuous_history_v15_rep{rep}_results.json'
        report=json.loads(cache.read_text()) if cache.exists() else None
        if report is None or report.get('independent_helper_sha256')!=helper_hash or report['artifact_sha256'].get(name)!=sha(path) or report['artifact_sha256'].get('continuous_history_v15.py')!=sha(ROOT/'continuous_history_v15.py'):
            report=audit_one(name,production=True);report['independent_helper_sha256']=helper_hash
            cache.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if original['selected_candidate_ids']['rich_delay']==original['selected_candidate_ids']['selected_delay']:
            assert original['controls']['rich_delay']==original['controls']['selected_delay']
        reports.append(report);originals.append(original)
        print(f'independently audited replicate{rep}',file=sys.stderr,flush=True)
    if args.replicate is not None:return reports[0]
    paths=[]
    for r in originals:
        for role in ['training','validation','testing']:
            paths.extend(tuple(e['seed_path']) for e in r[role])
        paths.extend(tuple(e['seed_path']) for e in r['controls']['pose'])
    assert len(paths)==len(set(paths))==608
    assert all(path[3]>=100 and path[:3]==(20261020,15,610) for path in paths)
    # Prior scalar seed streams are expanded from <=43b integer seeds or the
    #50b pilot; these new multi-integer entropy paths are different seeds. Check
    #generated state words, not merely displayed additive integers, for collisions.
    words=[tuple(np.random.SeedSequence(path).generate_state(4).tolist()) for path in paths]
    assert len(set(words))==608
    pilot_paths=[(20261020,15,610,0,role,j) for role,n in [(1,48),(2,12),(3,12),(4,2)] for j in range(n)]
    pilot_words={tuple(np.random.SeedSequence(path).generate_state(4).tolist()) for path in pilot_paths}
    assert not (set(words)&pilot_words)
    legacy_integer_seeds={75071013}
    for plant in range(4):
        for replicate in range(8):
            s=22161020+1000000000*plant+1000000*replicate
            for offset in [0,10000000000]:
                legacy_integer_seeds.add(s+offset)
                legacy_integer_seeds.update(s+offset+100000+100*i+k for i in range(192) for k in range(2))
            legacy_integer_seeds.update(s+500000+j for j in range(2))
            legacy_integer_seeds.update([s+20000000000,s+30000000000])
            legacy_integer_seeds.update(s+40000000000+j for j in range(2))
    for condition in range(2):
        for replicate in range(8):
            s=22461020+1000000000*condition+1000000*replicate
            legacy_integer_seeds.update(s+offset for offset in [100000,500000,600000])
    for base in [22561020,50020261020]:
        legacy_integer_seeds.update(base+1000000*mi+10000*wi+ni for mi in range(8) for wi in range(12) for ni in range(16))
    legacy_words={tuple(np.random.SeedSequence(s).generate_state(4).tolist()) for s in legacy_integer_seeds}
    assert not (set(words)&legacy_words)
    summary={}
    methods=list(originals[0]['controls'])
    resource_totals={};failure_records=[]
    maximum_pg_by_status=dict(converged=0.,not_converged=0.)
    for r in originals:
        expected=dict(acquisition_trajectories=72,acquisition_steps=72*96,acquisition_seconds=72*96*.1,
            acquisition_scalar_sensor_readings=72*97*2,candidates=30,control_episodes=24,control_steps=24*188,
            control_scalar_sensor_readings=24*189*2,
            planning_residual_calls=sum(p['residual_calls'] for episodes in r['controls'].values() for e in episodes for p in e['plans']))
        for key,value in expected.items():
            assert np.isclose(r['resources'][key],value,rtol=1e-14,atol=0.)
            resource_totals[key]=resource_totals.get(key,0)+value
        for method,episodes in r['controls'].items():
            for task,e in enumerate(episodes):
                for t,p in enumerate(e['plans']):
                    assert p['residual_calls']==p['nfev']+2
                    status='converged' if p['success'] else 'not_converged'
                    maximum_pg_by_status[status]=max(maximum_pg_by_status[status],p['projected_gradient_max'])
                    if not p['success']:failure_records.append(dict(replicate=r['meta']['replicate'],method=method,task=task,t=t,**p))
    for method in methods:
        summary[method]={}
        for metric in ['position_mse','velocity_rms','action_energy','late_position_rmse','late_velocity_rms']:
            per_rep=[np.mean([e['metrics'][metric] for e in r['controls'][method]]) for r in originals]
            summary[method][metric]=stats(per_rep)
        summary[method]['precision_goals']=sum(sum(e['metrics']['precision_success_by_goal']) for r in originals for e in r['controls'][method])
        summary[method]['goal_trials']=96
        summary[method]['precision_fraction_by_fit']=stats([sum(sum(e['metrics']['precision_success_by_goal']) for e in r['controls'][method])/12 for r in originals])
        summary[method]['solver_failures']=sum(e['metrics']['solver_failures'] for r in originals for e in r['controls'][method])
        summary[method]['forecast']={key:stats([r['forecasts'][method][key] for r in originals]) for key in ['one_step_true_position_mse','one_step_observation_mse','ten_step_recursive_true_position_mse']}
    pairs={}
    for method in ['pose','one_delay','rich_delay','linear_arx','polynomial']:
        values=[np.mean([a['metrics']['position_mse']-b['metrics']['position_mse'] for a,b in zip(r['controls'][method],r['controls']['selected_delay'])]) for r in originals]
        pairs[method+'_minus_selected_delay_position_mse']=stats(values)
    artifacts={name:sha(ROOT/name) for name in ['continuous_history_v15.py','continuous_history_v15_design.md','audit_continuous_history_v15_pilot.py','audit_continuous_history_v15.py']}
    artifacts.update({f'continuous_history_v15_rep{rep}.json':sha(ROOT/f'continuous_history_v15_rep{rep}.json') for rep in range(100,108)})
    totals={key:sum(r['checked'][key] for r in reports) for key in reports[0]['checked']}
    return dict(status='INDEPENDENTLY VERIFIED EIGHT-REPLICATE CONTINUOUS-HISTORY EXPERIMENT',artifact_sha256=artifacts,
        checked=totals,unique_acquisition_control_noise_roles=608,independent_replicates=8,
        resources_recomputed=dict(**resource_totals,control_simulated_seconds=3609.6,
            planner_learned_prediction_steps=resource_totals['planning_residual_calls']*15,
            forecast_learned_prediction_steps=8*6*(12*88+12*8*10),
            derivative_verification_learned_prediction_steps=8*6*17*4),
        solver_failure_records=failure_records,maximum_projected_gradient_by_status=maximum_pg_by_status,
        seed_state_collision_checks=dict(new_128bit_initialization_states=608,engineering_pilot_role_states=len(pilot_words),
            earlier_v12_v13_v14_and_history_pilot_integer_seed_states=len(legacy_words),collisions=0),
        summary=summary,paired_cluster_differences=pairs,
        selected_histories=[dict(replicate=r['meta']['replicate'],models={method:r['candidates'][index]['h'] for method,index in r['selected_candidate_ids'].items()}) for r in originals],
        replications=reports,
        scope=['All240candidate fits, allfresh trajectories/RNG/forecasts and all selected MPC plans/complex-step gradients checked with separate mathematics.',
            'Means and SEs use eight acquired-data clusters after averaging four fixed tasks per cluster; goal events are dependent within cluster.',
            'Same rich/selected model ID implies byte-identical controllers and trajectories; equality does not show a benefit from adaptive order selection.',
            'Fast failed pose-only trajectories require adaptive local physics checks and explicit global sensitivity diagnostics.',
            'Prior engineering pilot and PD diagnostic remain exploratory development, not fresh comparisons.',
            'No universal learning claim, optimized PD comparison, true-state observer advantage or global nonlinear optimizer certificate.'])


if __name__=='__main__':
    with threadpool_limits(limits=1):report=main()
    print(json.dumps(report,indent=2,allow_nan=False))
