#!/usr/bin/env python3
"""Descriptive paired-corpus analysis; refuses incomplete or mismatched artifacts."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('mixture_dual','mixture_task','gaussian_dual','gaussian_task','matched_dual',
         'matched_task','mixture_random','gaussian_random','isotropic_dual','isotropic_task')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_se(values):
    v=np.array(values,dtype=float)
    return dict(mean=float(v.mean()),se_across_eight_corpora=float(v.std(ddof=1)/np.sqrt(len(v))),
        min=float(v.min()),max=float(v.max()),corpus_values=v.tolist())


def metrics(episodes):
    phases=[p for e in episodes for p in e['phases']]
    return dict(cost=float(np.mean([p['cost'] for p in phases])),
        first_goal_cost=float(np.mean([e['phases'][0]['cost'] for e in episodes])),
        later_goal_cost=float(np.mean([p['cost'] for e in episodes for p in e['phases'][1:]])),
        phase_costs=np.mean([[p['cost'] for p in e['phases']] for e in episodes],axis=0).tolist(),
        sequence_phase_costs=[np.mean([[p['cost'] for p in e['phases']] for e in episodes if e['sequence']==seq],axis=0).tolist()
            for seq in range(2)],
        sequence_phase_precision_counts=[np.sum([[p['precision_success'] for p in e['phases']] for e in episodes if e['sequence']==seq],axis=0).tolist()
            for seq in range(2)],
        precision_count=sum(p['precision_success'] for p in phases),
        precision_fraction=float(np.mean([p['precision_success'] for p in phases])),
        first_goal_precision_count=sum(e['phases'][0]['precision_success'] for e in episodes),
        later_goal_precision_count=sum(p['precision_success'] for e in episodes for p in e['phases'][1:]),
        effort_per_goal=float(np.mean([p['effort'] for p in phases])),
        terminal_position_rmse_per_coordinate=float(np.sqrt(np.mean([np.array(p['position_error'])**2 for p in phases]))),
        terminal_velocity_rms_per_coordinate=float(np.sqrt(np.mean([np.array(p['cartesian_velocity'])**2 for p in phases]))))


def main():
    corpora=[];sources={digest(ROOT/n):n for n in ['cartesian_feedback_v16_study.py','cartesian_feedback_v16_study_initial.py']}
    acquisition_intervals=acquisition_reads=control_intervals=control_reads=0
    counts=dict(computed_designs=0,evaluated_action_candidates=0,tighter_candidate_reevaluations=0,integration_points=0)
    for rep in range(100,108):
        path=ROOT/f'cartesian_feedback_v16_rep{rep}_results.json';d=json.loads(path.read_bytes())
        assert d['meta']['replicate']==rep and d['meta']['source_sha256'] in sources
        parentpath=ROOT/d['meta']['parent_results'];parent=json.loads(parentpath.read_bytes())
        assert digest(parentpath)==d['meta']['parent_sha256']
        datapath=parentpath.with_name(parentpath.name.replace('_results.json','_data.npz'))
        assert digest(datapath)==d['meta']['data_sha256']==parent['meta']['data_sha256']
        assert digest(ROOT/'cartesian_reuse_v16_gk21.py')==d['meta']['base_source_sha256']
        assert 'resources' in d and len(d['episodes'])==80,'Incomplete study'
        with np.load(datapath) as raw:
            ai=sum(len(raw[n+'_commands'])*2 for n in ['training','validation','testing'])
            ar=sum(raw[n+'_observed'].size for n in ['training','validation','testing'])
        assert ai==parent['resources']['acquired_intervals']==768
        assert ar==parent['resources']['acquired_scalar_sensor_readings']==2304
        acquisition_intervals+=ai;acquisition_reads+=ar
        ci=sum(len(e['commands']) for e in d['episodes']);cr=sum(np.array(e['observed']).size for e in d['episodes'])
        assert ci==d['resources']['control_intervals']==1440 and cr==d['resources']['scalar_sensor_readings']==1920
        control_intervals+=ci;control_reads+=cr
        for key in counts:counts[key]+=d['resources'][key]
        bymethod={}
        for method in METHODS:
            E=[e for e in d['episodes'] if e['prior']+'_'+e['mode']==method]
            assert len(E)==8 and {(e['sequence'],e['repeat'],e['branch_evaluator_only']) for e in E}=={
                (s,r,b) for s in range(2) for r in range(2) for b in range(2)}
            assert all(len(e['phases'])==6 for e in E)
            bymethod[method]=metrics(E)
        corpora.append(dict(replicate=rep,result_file=path.name,result_sha256=digest(path),
            source_file=sources[d['meta']['source_sha256']],acquisition_file=parentpath.name,
            selected_mixture_candidate=d['selected_mixture_predictive_candidate'],
            selected_mixture_components=len(d['priors']['mixture']['weights']),
            selected_mixture_converged=parent['mixture_candidates'][d['selected_mixture_predictive_candidate']]['converged'],
            nonconverged_em_candidates=sum(not r['converged'] for r in parent['mixture_candidates']),
            selected_gaussian_candidate=parent['selected_gaussian'],heldout_forecasts=d['heldout_forecasts'],methods=bymethod))
    summary={}
    for method in METHODS:
        rows=[c['methods'][method] for c in corpora]
        summary[method]={key:mean_se([r[key] for r in rows]) for key in ['cost','first_goal_cost','later_goal_cost',
            'precision_fraction','effort_per_goal','terminal_position_rmse_per_coordinate','terminal_velocity_rms_per_coordinate']}
        summary[method]['precision_count']=sum(r['precision_count'] for r in rows)
        summary[method]['first_goal_precision_count']=sum(r['first_goal_precision_count'] for r in rows)
        summary[method]['later_goal_precision_count']=sum(r['later_goal_precision_count'] for r in rows)
        summary[method]['phase_cost_means']=np.mean([r['phase_costs'] for r in rows],axis=0).tolist()
        summary[method]['sequence_phase_cost_means']=np.mean([r['sequence_phase_costs'] for r in rows],axis=0).tolist()
        summary[method]['sequence_phase_precision_counts']=np.sum([r['sequence_phase_precision_counts'] for r in rows],axis=0).tolist()
    paired={}
    for a,b in [('gaussian_dual','mixture_dual'),('matched_dual','mixture_dual'),
                ('isotropic_dual','gaussian_dual'),('mixture_task','mixture_dual'),
                ('gaussian_task','gaussian_dual'),('mixture_random','mixture_dual'),
                ('gaussian_random','gaussian_dual')]:
        paired[a+'_minus_'+b]={key:mean_se([c['methods'][a][key]-c['methods'][b][key] for c in corpora])
            for key in ['cost','first_goal_cost','later_goal_cost','precision_fraction','effort_per_goal']}
    out=dict(status='DESCRIPTIVE FRESH STUDY ANALYSIS; independent final audits required',
        analysis_source_sha256=digest(Path(__file__)),
        interpretation='Eight independently acquired/fitted corpora with four paired noise paths each, two fixed physical preparations; clustered checks are not independent bodies. SE is descriptive across eight corpus averages, not a robotics-population guarantee.',
        reporting_correction='First four raw results inherited misleading control-only reuse/new0 acquisition fields. Combined acquisition counts here are reconstructed from separately saved actual acquisition arrays for every corpus. No raw result rewritten.',
        fit_limits=dict(nonconverged_em_candidates=sum(c['nonconverged_em_candidates'] for c in corpora),
            em_candidates_total=136,selected_mixture_converged=[c['selected_mixture_converged'] for c in corpora],
            interpretation='Bounded EM candidates retained even on iteration limit; ML ftol success is not a global or strict-gradient-optimality certificate.'),
        corpora=corpora,summary=summary,paired_corpus_differences=paired,
        resources=dict(acquired_groups=3072,acquisition_intervals=acquisition_intervals,
            acquisition_simulated_seconds=.1*acquisition_intervals,acquisition_scalar_reads=acquisition_reads,
            control_episodes=640,clustered_goal_checks=3840,control_intervals=control_intervals,
            control_simulated_seconds=.1*control_intervals,control_scalar_reads=control_reads,
            independent_paired_noise_paths=32,unique_scalar_control_noise_draws=768,**counts))
    path=ROOT/'cartesian_feedback_v16_aggregate_results.json';path.write_text(json.dumps(out,indent=2)+'\n')
    for method,s in summary.items():
        print(method,'cost',s['cost']['mean'],'SE',s['cost']['se_across_eight_corpora'],
            'first/later',s['first_goal_cost']['mean'],s['later_goal_cost']['mean'],'precision',s['precision_count'],'/384')
    print('saved',path)


if __name__=='__main__':main()
