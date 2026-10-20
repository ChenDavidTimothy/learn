#!/usr/bin/env python3
"""V11: isolate cutoff knowledge before attributing stopping failures to sensing.

Same frozen v10 controller/filter/plant/fit/goals/noise. Two additional variants
share16 predicted zero-command coast steps. One still plans8 free controls;
the other shrinks to the announced number remaining before time70. No new
physical learning data. Exploration is explicitly a post-v10 diagnostic repair.
"""
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import goal_control_v10 as control
from precision_information_v11 import information_limit


def endpoint_diagnostic(episode,sd):
    A=np.array([[1.,.65,1.],[0.,.65,1.],[0.,0.,.7]])
    f=np.linalg.matrix_power(A,17)[0]
    sensitivity=float((np.linalg.matrix_power(A,16)@np.array([0.,0.,1.2]))[0])
    mean=float(f@episode['estimates'][69]+sensitivity*episode['commands'][69]-episode['goals'][-1])
    variance=information_limit(sd)['terminal_position_conditional_mse_lower_bound']
    realized=episode['states'][-1][0]-episode['goals'][-1]
    return {'conditional_mean_endpoint_error':mean,'conditional_variance':variance,
        'conditional_endpoint_mse':mean*mean+variance,'realized_endpoint_error':realized,
        'mean_centering_last_action':episode['commands'][69]-mean/sensitivity,
        'mean_centering_action_is_feasible':abs(episode['commands'][69]-mean/sensitivity)<=1.}


def run_variant(true_theta,model,goals,seed,aware):
    original=control.choose_action
    calls=0
    def choose(state,goal,theta,warm):
        nonlocal calls
        remaining=control.ACTIVE-calls
        used=min(control.HORIZON,remaining) if aware else control.HORIZON
        u,next_warm,record=original(state,goal,theta,warm[:used])
        record.update(announced_remaining_steps=remaining if aware else None,
                      planned_control_steps=used)
        calls+=1
        return u,next_warm,record
    control.choose_action=choose
    try:
        result=control.run_episode(true_theta,model,'linear_MPC',goals,seed)
    finally:control.choose_action=original
    assert calls==control.ACTIVE
    return result


def main():
    root=Path(__file__).resolve().parent
    prior=json.loads((root/'precision_information_v11_results.json').read_text())
    theta=prior['meta']['plant'];runs=[]
    control.PLAN_COAST=16
    with threadpool_limits(limits=1):
        for row in prior['runs']:
            cases={}
            for sd in (.2,.02):
                control.SENSOR_SD=sd;tasks=[]
                for j,old in enumerate(row['cases'][str(sd)]['tasks']):
                    variants={}
                    for name,model in (('short512',row['short_theta']),('known_parameters',theta)):
                        for label,aware in (('unaware16',False),('aware16',True)):
                            episode=run_variant(theta,model,old['goal_schedule'],row['seed']+2000000+j,aware)
                            if name=='known_parameters':episode['endpoint_diagnostic']=endpoint_diagnostic(episode,sd)
                            variants[name+'_'+label]=episode
                    tasks.append({'goal_schedule':old['goal_schedule'],
                        'unaware12_known_endpoint_diagnostic':endpoint_diagnostic(old['controllers']['known_parameters'],sd),
                        'variants':variants})
                cases[str(sd)]={'tasks':tasks}
            runs.append({'seed':row['seed'],'cases':cases})
    control.SENSOR_SD=.2;control.PLAN_COAST=12
    summary={}
    for sd in ('0.2','0.02'):
        summary[sd]={}
        for name in runs[0]['cases'][sd]['tasks'][0]['variants']:
            episodes=[t['variants'][name] for r in runs for t in r['cases'][sd]['tasks']]
            s={'normalized_tracking_cost':control.stat([np.mean([t['variants'][name]['normalized_tracking_cost']
                for t in r['cases'][sd]['tasks']]) for r in runs]),
                'stop_successes':sum(e['stop_success'] for e in episodes),'task_trials':len(episodes),
                'late_position_rmse':control.stat([np.mean([t['variants'][name]['late_position_rmse']
                    for t in r['cases'][sd]['tasks']]) for r in runs]),
                'optimizer_nonconvergences':sum(not a['success'] for e in episodes for a in e['optimizers'] if a)}
            if name.startswith('known_'):
                s['mean_squared_conditional_endpoint_bias']=float(np.mean([
                    e['endpoint_diagnostic']['conditional_mean_endpoint_error']**2 for e in episodes]))
            summary[sd][name]=s
    print(json.dumps({'meta':{'version':11,'status':'TESTED EXPLORATORY DEADLINE DIAGNOSTIC; independent audits retained',
        'goal_cutoff_time':70,'predicted_coast_for_both_added_variants':16,
        'only_difference_within_added_pair':'known remaining action opportunities versus fixed8',
        'plant_and_information':'same as linear slow-motor v11; no additionaltraining',
        'warning':'known cutoff is task metadata; not autonomous discovery or changed physical sensing'},
        'runs':runs,'summary':summary},indent=2,allow_nan=False))


if __name__=='__main__':main()
