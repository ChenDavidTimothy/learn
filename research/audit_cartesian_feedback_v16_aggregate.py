#!/usr/bin/env python3
"""Independent clustered summary audit from verified per-corpus physical replay."""
from pathlib import Path
import ast,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-9):
    err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def avg(x):return math.fsum(map(float,x))/len(x)
def stats(v):
    assert len(v)==8;m=avg(v)
    return {'mean':m,'se_across_eight_corpora':math.sqrt(math.fsum((x-m)**2 for x in v)/(8*7)),'min':min(v),'max':max(v),'corpus_values':v}
def metrics(rows):
    costs=[x for row in rows for x in row['costs']];success=[x for row in rows for x in row['success']]
    return {'cost':avg(costs),'first_goal_cost':avg([row['costs'][0] for row in rows]),'later_goal_cost':avg([x for row in rows for x in row['costs'][1:]]),
        'phase_costs':[avg([row['costs'][i] for row in rows]) for i in range(6)],'precision_count':sum(success),'precision_fraction':avg(success),
        'first_goal_precision_count':sum(row['success'][0] for row in rows),'later_goal_precision_count':sum(sum(row['success'][1:]) for row in rows),
        'effort_per_goal':avg([x for row in rows for x in row['effort']]),'terminal_position_rmse_per_coordinate':math.sqrt(avg([x for row in rows for x in row['position_squared']])/2),
        'terminal_velocity_rms_per_coordinate':math.sqrt(avg([x for row in rows for x in row['velocity_squared']])/2)}

def main():
    aggregate_path=ROOT/'cartesian_feedback_v16_aggregate_results.json';canonical=json.loads(aggregate_path.read_text());assert sha(ROOT/'analyze_cartesian_feedback_v16.py')==canonical['analysis_source_sha256']
    modules=[ast.parse((ROOT/name).read_text()) for name in ('cartesian_feedback_v16_study_initial.py','cartesian_feedback_v16_study.py')]
    definitions=[{n.name:ast.dump(n,include_attributes=False) for n in module.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for module in modules]
    same_functions=[name for name in definitions[0] if name!='main' and definitions[0][name]==definitions[1][name]]
    assert len(same_functions)==len(definitions[0])-1
    corpora=[];sources=[];paths=set();physical=0;counts={};oracle=0;fit_status=[];mixture_failures=[]
    for rep in range(100,108):
        path=ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_results.json';audit=json.loads(path.read_text());assert audit['replicate']==rep
        assert sha(ROOT/'audit_cartesian_feedback_v16_study.py')==audit['audit_source_sha256']
        for name,h in audit['artifact_sha256'].items():assert sha(ROOT/name)==h,(rep,name)
        acq=ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_acquisition_results.json';acq_audit=json.loads(acq.read_text());assert sha(ROOT/'audit_cartesian_feedback_v16_study.py')==acq_audit['audit_source_sha256']
        for name,h in acq_audit['artifact_sha256'].items():assert sha(ROOT/name)==h,(rep,name)
        raw=json.loads((ROOT/f'cartesian_feedback_v16_rep{rep}_results.json').read_text());stored=[x for x in canonical['corpora'] if x['replicate']==rep];assert len(stored)==1;stored=stored[0]
        if rep>=104:
            assert raw['resources']['new_acquisition_intervals']==768 and not raw['resources']['parent_acquisition_reused']
            assert raw['meta']['acquisition_origin']=='fresh_in_this_invocation'
        else:
            assert raw['resources']['new_acquisition_intervals']==0 and raw['resources']['parent_acquisition_reused']
        for method in audit['individual_episode_metrics']:
            E=[e for e in raw['episodes'] if e['prior']+'_'+e['mode']==method]
            assert len(E)==8 and {(e['sequence'],e['repeat'],e['branch_evaluator_only']) for e in E}=={(s,n,b) for s in range(2) for n in range(2) for b in range(2)}
        for e in raw['episodes']:
            if e['prior']=='mixture' and e['mode']=='dual':
                for phase,p in enumerate(e['phases']):
                    if not p['precision_success']:
                        mixture_failures.append({'replicate':rep,'sequence':e['sequence'],'repeat':e['repeat'],'branch_evaluator_only':e['branch_evaluator_only'],'phase':phase,
                            'cost':p['cost'],'position_error':p['position_error'],'first_posterior_weights':p['belief_after_first']['weights'],
                            'terminal_posterior_weights':p['belief_after_terminal']['weights'],'later_phase_precision':[q['precision_success'] for q in e['phases'][phase+1:]]})
        assert sha(ROOT/stored['result_file'])==stored['result_sha256'];assert stored['heldout_forecasts']==raw['heldout_forecasts'];assert stored['selected_mixture_candidate']==raw['selected_mixture_predictive_candidate']
        assert stored['selected_mixture_components']==len(raw['priors']['mixture']['weights'])
        parentpath=ROOT/raw['meta']['parent_results'];parent=json.loads(parentpath.read_text());assert stored['selected_gaussian_candidate']==parent['selected_gaussian']
        chosen=raw['selected_mixture_predictive_candidate']
        fit_status.append({'replicate':rep,'selected_mixture_components':len(raw['priors']['mixture']['weights']),
            'selected_mixture_EM_converged':parent['mixture_candidates'][chosen]['converged'],
            'nonconverged_EM_candidates':sum(not c['converged'] for c in parent['mixture_candidates']),
            'Gaussian_ML_reported_successes':sum(c['success'] for c in parent['gaussian_candidates'][3:]),
            'selected_Gaussian_candidate':parent['selected_gaussian']})
        methods={k:metrics(v) for k,v in audit['individual_episode_metrics'].items()}
        for method,m in methods.items():
            for name,value in m.items():ck('per_corpus_'+name,value,stored['methods'][method][name])
        corpora.append(methods);sources.extend([path,acq]);physical+=audit['correct_total_new_physical_intervals'];resources=audit['resources_reconstructed_control'];oracle+=resources['nonlinear_witness_plant_flows']
        for key in ('computed_designs','evaluated_action_candidates','tighter_candidate_reevaluations','integration_points'):counts[key]=counts.get(key,0)+resources[key]
        datapath=parentpath.with_name(parentpath.name.replace('_results.json','_data.npz'))
        with np.load(datapath) as data:
            for name in ('training','validation','testing'):
                for key in data[name+'_seed_paths']:
                    t=tuple(key);assert t not in paths;paths.add(t)
        for key in {tuple(e['seed_path']) for e in raw['episodes']}:
            assert key not in paths;paths.add(key)
    summaries={}
    for method in corpora[0]:
        summary={}
        for name in ('cost','first_goal_cost','later_goal_cost','precision_fraction','effort_per_goal','terminal_position_rmse_per_coordinate','terminal_velocity_rms_per_coordinate'):
            s=stats([c[method][name] for c in corpora]);summary[name]=s
            for key,value in s.items():ck('summary_'+name+'_'+key,value,canonical['summary'][method][name][key])
        for name in ('precision_count','first_goal_precision_count','later_goal_precision_count'):
            summary[name]=sum(c[method][name] for c in corpora);ck('summary_'+name,summary[name],canonical['summary'][method][name])
        summary['phase_cost_means']=[avg([c[method]['phase_costs'][i] for c in corpora]) for i in range(6)];ck('summary_phase_costs',summary['phase_cost_means'],canonical['summary'][method]['phase_cost_means']);summaries[method]=summary
    paired={}
    for label,record in canonical['paired_corpus_differences'].items():
        a,b=label.split('_minus_');paired[label]={}
        for name in ('cost','first_goal_cost','later_goal_cost','precision_fraction','effort_per_goal'):
            s=stats([c[a][name]-c[b][name] for c in corpora]);paired[label][name]=s
            for key,value in s.items():ck('paired_'+name+'_'+key,value,record[name][key])
    resources={'acquired_groups':3072,'acquisition_intervals':6144,'acquisition_simulated_seconds':614.4,'acquisition_scalar_reads':18432,'control_episodes':640,'clustered_goal_checks':3840,'control_intervals':11520,'control_simulated_seconds':1152.,'control_scalar_reads':15360,'independent_paired_noise_paths':32,'unique_scalar_control_noise_draws':768,**counts}
    for name,value in resources.items():ck('aggregate_resource_'+name,value,canonical['resources'][name])
    assert physical==17664 and len(paths)==3104
    va=[c['mixture_dual']['cost'] for c in corpora];vb=[c['gaussian_dual']['cost'] for c in corpora];ma=avg(va);mb=avg(vb)
    covariance=math.fsum((x-ma)*(y-mb) for x,y in zip(va,vb))/7
    correlation=covariance/math.sqrt((math.fsum((x-ma)**2 for x in va)/7)*(math.fsum((y-mb)**2 for y in vb)/7))
    se_a=summaries['mixture_dual']['cost']['se_across_eight_corpora'];se_b=summaries['gaussian_dual']['cost']['se_across_eight_corpora'];se_diff=paired['gaussian_dual_minus_mixture_dual']['cost']['se_across_eight_corpora']
    ck('paired_variance_covariance_identity',se_diff**2,se_a**2+se_b**2-2*covariance/8,1e-15)
    out={'status':'INDEPENDENT EIGHT-CORPUS AGGREGATE RECONSTRUCTION PASSED; SE IS ACROSS CORPUS MEANS',
        'artifact_sha256':{p.name:sha(p) for p in [aggregate_path,ROOT/'analyze_cartesian_feedback_v16.py',ROOT/'cartesian_feedback_v16_study_design.md',ROOT/'audit_cartesian_feedback_v16_study.py',ROOT/'audit_cartesian_feedback_v16_pilot.py',ROOT/'audit_cartesian_reuse_v16_pilot.py',ROOT/'audit_cartesian_reuse_v16_adaptive.py',*sources]},'audit_source_sha256':sha(Path(__file__)),
        'maximum_reconstruction_errors':ERROR,'summary':summaries,'paired_corpus_differences':paired,'resources':resources,'nonlinear_nominal_witness_plant_flows_additional':oracle,
        'fit_status_by_corpus':fit_status,
        'mixture_directed_all_precision_failures':mixture_failures,'mixture_gaussian_directed_corpus_cost_correlation':correlation,
        'initial_and_corrected_source_identical_nonmain_definitions':same_functions,
        'total_new_acquisition_plus_control_intervals_excluding_witnesses':physical,'distinct_acquisition_and_control_SeedSequence_keys':len(paths),
        'limitations':['Eight corpus-average clusters, not3840 independent goal samples; errors are descriptive Monte Carlo SE over this fixed prepared study.',
            'Mean per-corpus RMSE is verified as such, not confused with pooled raw squared-error RMSE.','No new-body, unseen-goal-sequence, global active-control or universal learning claim.',
            'Physical/filter/QP replay is independently verified per corpus; actual first-action integrations require separate numerical audit.']}
    (ROOT/'audit_cartesian_feedback_v16_aggregate_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'errors':ERROR,'resources':resources}))

if __name__=='__main__':main()
