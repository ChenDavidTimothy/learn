#!/usr/bin/env python3
"""Aggregate predeclared independent acquired-data clusters, never retune.

All inputs are actual complete production artifacts, identified by SHA256.
"""
import hashlib,json
from pathlib import Path
import numpy as np


def stat(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))),values=x.tolist(),n=len(x))


def main():
    root=Path(__file__).resolve().parent;inputs=[];runs=[]
    for rep in range(100,108):
        p=root/f'continuous_history_v15_rep{rep}.json';raw=p.read_bytes();d=json.loads(raw)
        assert d['meta']['replicate']==rep and 'resources' in d and len(d['controls'])==6
        inputs.append(dict(file=p.name,sha256=hashlib.sha256(raw).hexdigest()));runs.append(d)
    methods=list(runs[0]['controls']);summary={}
    for method in methods:
        s={}
        for metric in ('position_mse','velocity_rms','action_energy','late_position_rmse','late_velocity_rms'):
            s[metric]=stat([np.mean([e['metrics'][metric] for e in r['controls'][method]]) for r in runs])
        s['precision_success_fraction']=stat([np.mean([v for e in r['controls'][method] for v in e['metrics']['precision_success_by_goal']]) for r in runs])
        s['precision_success_total']=sum(sum(e['metrics']['precision_success_by_goal']) for r in runs for e in r['controls'][method])
        s['precision_opportunities']=8*4*3
        s['solver_failures']=sum(e['metrics']['solver_failures'] for r in runs for e in r['controls'][method])
        s['saturated_coordinates']=sum(e['metrics']['saturated_coordinates'] for r in runs for e in r['controls'][method])
        s['selected_h']=[r['candidates'][r['selected_candidate_ids'][method]]['h'] for r in runs]
        s['selected_ridge']=[r['candidates'][r['selected_candidate_ids'][method]]['ridge'] for r in runs]
        s['one_step_true_position_mse']=stat([r['forecasts'][method]['one_step_true_position_mse'] for r in runs])
        s['ten_step_recursive_true_position_mse']=stat([r['forecasts'][method]['ten_step_recursive_true_position_mse'] for r in runs])
        summary[method]=s
    paired={}
    for a,b in (('one_delay','selected_delay'),('rich_delay','selected_delay'),('selected_delay','polynomial'),('linear_arx','selected_delay')):
        paired[f'{a}_minus_{b}']={metric:stat(np.array(summary[a][metric]['values'])-summary[b][metric]['values'])
            for metric in ('position_mse','late_position_rmse','late_velocity_rms','precision_success_fraction')}
    coverage=[]
    for r in runs:
        item=dict(replicate=r['meta']['replicate'])
        for role in ('training','validation','testing'):
            states=np.concatenate([np.array(e['states_evaluator_only']) for e in r[role]])
            commands=np.concatenate([np.array(e['commands']) for e in r[role]])
            item[role]=dict(max_abs_velocity=float(np.max(np.abs(states[:,2:]))),
                velocity_rms=float(np.sqrt(np.mean(states[:,2:]**2))),position_min=states[:,:2].min(0).tolist(),
                position_max=states[:,:2].max(0).tolist(),action_min=commands.min(0).tolist(),action_max=commands.max(0).tolist())
        coverage.append(item)
    resources={k:sum(r['resources'][k] for r in runs) for k in runs[0]['resources']}
    resources['control_seconds']=resources['control_steps']*.1
    resources['auxiliary_derivative_residual_calls']=8*6*17
    resources['auxiliary_derivative_predicted_transitions']=8*6*17*4
    resources['heldout_forecast_predicted_transitions']=8*6*(1056+960)
    resources['acquisition_and_control_resets']=resources['acquisition_trajectories']+resources['control_episodes']
    resources['wall_seconds_sum']=sum(r['timing_seconds']['wall'] for r in runs)
    resources['unique_test_conditions_note']='Four starting contexts, not32 independently sampled task geometries; eight independently acquired datasets/noise streams.'
    out=dict(status='PRODUCTION AGGREGATE; CHECK INDEPENDENT AUDITS BEFORE SCIENTIFIC RELEASE',inputs=inputs,summary=summary,
        paired=paired,coverage=coverage,resources=resources,
        limitations=['Conventional supplied model families/history menus, not autonomous invention of state.',
            'One body, finite contexts, conservative ideal torque, no motor memory/contact/gravity.',
            'Certainty-equivalent local finite-horizon planner; no global optimality/stability guarantee.',
            'Prediction and precision are separate metrics; eight independent data fits are the uncertainty clusters.'])
    (root/'continuous_history_v15_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for k,s in summary.items():
        print(k,'h',s['selected_h'],'position',s['position_mse']['mean'],'late',s['late_position_rmse']['mean'],
            'velocity',s['late_velocity_rms']['mean'],'precision',s['precision_success_total'],'/96')


if __name__=='__main__':main()
