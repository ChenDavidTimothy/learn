#!/usr/bin/env python3
"""Exploratory stronger PD control: tune only on each learned nonlinear model.

Added after the preserved L-BFGS pilot showed coarse-grid PD performed poorly.
This is an explicit baseline repair, not a preregistered comparison. Actual
training data, evaluation goals/noise, action limits and metrics do not change.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from goal_control_v10 import run_episode,PD_GRID,stat


def tune(model,seed,coarse,stop_weight):
    def score(p):
        values=[]
        for j,goals in enumerate(((1.,-1.,.5),(-1.,1.,-.5))):
            r=run_episode(model,model,'model_tuned_PD',goals,seed+j,pd=p,record=False)
            target=np.array([goals[0]]*6+[goals[1]]*22+[goals[2]]*42)
            values.append(r['normalized_tracking_cost']+stop_weight*(
                r['late_position_rmse']**2+.3*r['late_velocity_rms']**2)/(1+np.mean(target**2)))
        return float(np.mean(values))
    records=[]
    starts=(coarse,(.02,1.,.8))
    for p in starts:
        fit=minimize(score,p,method='Powell',bounds=((.002,.5),(0.,4.),(.05,.95)),
                     options={'maxfev':350,'xtol':1e-4,'ftol':1e-5})
        records.append({'gains':fit.x.tolist(),'score':float(fit.fun),
            'success':bool(fit.success),'nfev':int(fit.nfev),'status':int(fit.status)})
    records.append({'gains':list(coarse),'score':score(coarse),
                    'success':True,'nfev':1,'status':-1})
    best=min(records,key=lambda r:r['score'])
    return best['gains'],records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default=str(Path(__file__).with_name('goal_control_v10_results.json')))
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text())
    plants=[]
    with threadpool_limits(limits=1):
        for pi,plant in enumerate(data['plants']):
            runs=[]
            for row in plant['runs']:
                model=row['nonlinear_theta'];seed=row['seed']
                tuned={};optimizers={}
                for label,weight in (('continuous_PD_active',0.),('continuous_PD_stop',5.)):
                    gains,records=tune(model,seed+1000000,row['selected_PD'],weight)
                    tuned[label]=gains;optimizers[label]=records
                tasks=[]
                for j,task in enumerate(row['tasks']):
                    controllers={label:run_episode(plant['true_theta'],model,'model_tuned_PD',
                        task['goal_schedule'],seed+2000000+j,pd=gains)
                        for label,gains in tuned.items()}
                    tasks.append({'goal_schedule':task['goal_schedule'],'controllers':controllers})
                runs.append({'seed':seed,'gains':tuned,'optimization':optimizers,'tasks':tasks})
            summary={}
            for label in tuned:
                summary[label]={key:stat([np.mean([t['controllers'][label][key] for t in r['tasks']])
                    for r in runs]) for key in ('normalized_tracking_cost','late_position_rmse',
                        'late_velocity_rms','mean_action_energy')}
                summary[label]['stop_successes']=sum(t['controllers'][label]['stop_success']
                    for r in runs for t in r['tasks'])
                summary[label]['task_trials']=len(runs)*2
            plants.append({'true_theta':plant['true_theta'],'runs':runs,'summary':summary})
            print('completed PD plant',pi,flush=True,file=__import__('sys').stderr)
    print(json.dumps({'meta':{'version':10,'status':'EXPLORATORY BASELINE REPAIR',
        'bounds':[[.002,.5],[0.,4.],[.05,.95]],'maxfev_per_start':350,
        'virtual_goal_schedules':[[1.,-1.,.5],[-1.,1.,-.5]],
        'tuning_data':'learned nonlinear model only; no extra physical interactions',
        'stop_weight':5.,'selection':'minimum synthetic objective including coarse-grid candidate',
        'training_fits_source':'same frozen512-row fits as canonical v10',
        'warning':'PD gain family remains supplied; no global optimality guarantee'},
        'plants':plants},indent=2,allow_nan=False))


if __name__=='__main__':main()
