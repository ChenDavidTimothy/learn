#!/usr/bin/env python3
"""Active v11 investigation: more training versus current state information.

Frozen follow-up to v10: only linear slow-motor plant [.65,.7,0,1.2].
Reuse all v10 short training rows and test noise seeds. Extend each same physical
trajectory to8192 commands, without resetting it. Compare512/8192 learned linear
OE against known parameters under sensorSD .2/.02; same controller/metrics.
"""
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import goal_control_v10 as control
from nonlinear_identification_v9 import trajectory,stat


def information_limit(sensor_sd,lam=.7,n_active=70,coast=16):
    rho=.65
    A=np.array([[1.,rho,1.],[0.,rho,1.],[0.,0.,lam]])
    Q=np.diag([0.,0.,.03**2]);P=np.zeros((3,3))
    for t in range(n_active):
        variance_observation=P[0,0]+sensor_sd**2
        K=P[:,0]/variance_observation if variance_observation>0 else np.zeros(3)
        P=P-np.outer(K,P[0])
        if t<n_active-1:P=A@P@A.T+Q
    future=np.linalg.matrix_power(A,coast+1)
    impulse=np.linalg.matrix_power(A,coast)
    current=float((future@P@future.T)[0,0])
    fresh=float((impulse@Q@impulse.T)[0,0])
    variance=current+fresh
    return {'sensor_sd':sensor_sd,'lambda':lam,'posterior_covariance_at_last_action':P.tolist(),
        'terminal_variance_current_state':current,'terminal_variance_fresh_innovation':fresh,
        'terminal_position_conditional_mse_lower_bound':variance,
        'terminal_position_conditional_sd':math.sqrt(variance),
        'whole_stop_population_probability_upper_bound':math.erf(.3/math.sqrt(2*variance)),
        'full_current_state_probability_upper_bound':math.erf(.3/math.sqrt(2*fresh)),
        'scope':'linearGaussian knownparameters,allobservationhistory,lastcommandthen16zeros; terminalmarginal bound only'}


def extend_training(row,true_theta):
    u=list(row['training_commands']);observations=list(row['training_positions'])
    rng=np.random.default_rng(row['seed']+4000000)
    while len(u)<8192:
        duration=int(rng.choice((1,3,8)))
        u.extend([float(rng.uniform(-1.,1.))]*duration)
    commands=np.array(u[:8192]);x,_,_=trajectory(true_theta,commands)
    observations.extend((x[513:]+rng.normal(0,.2,len(x)-513)).tolist())
    assert len(observations)==8193
    return commands,np.array(observations)


def main():
    prior=json.loads(Path(__file__).with_name('goal_control_v10_results.json').read_text())
    plant=next(p for p in prior['plants'] if p['true_theta']==[.65,.7,0.,1.2])
    theta=plant['true_theta'];runs=[]
    with threadpool_limits(limits=1):
        for row in plant['runs']:
            commands,observations=extend_training(row,theta)
            long_model,records=control.fit_linear(observations,commands)
            cases={}
            for sd in (.2,.02):
                control.SENSOR_SD=sd  # Explicit sensing intervention in same frozen controller.
                tasks=[]
                for j,task in enumerate(row['tasks']):
                    controllers={}
                    for label,model in (('short512',row['linear_theta']),('long8192',long_model),('known_parameters',theta)):
                        controllers[label]=control.run_episode(theta,model,'linear_MPC',
                            task['goal_schedule'],row['seed']+2000000+j)
                    tasks.append({'goal_schedule':task['goal_schedule'],'controllers':controllers})
                cases[str(sd)]={'sensor_sd':sd,'tasks':tasks}
            runs.append({'seed':row['seed'],'short_theta':row['linear_theta'],
                'long_theta':long_model.tolist(),'long_fit_records':records,
                'long_training_commands':commands.tolist(),'long_training_positions':observations.tolist(),
                'cases':cases})
    control.SENSOR_SD=.2
    summary={}
    for sd in ('0.2','0.02'):
        summary[sd]={}
        for label in ('short512','long8192','known_parameters'):
            values=[np.mean([t['controllers'][label]['normalized_tracking_cost'] for t in r['cases'][sd]['tasks']]) for r in runs]
            summary[sd][label]={'normalized_tracking_cost':stat(values),
                'stop_successes':sum(t['controllers'][label]['stop_success'] for r in runs for t in r['cases'][sd]['tasks']),
                'task_trials':24,
                'late_position_rmse':stat([np.mean([t['controllers'][label]['late_position_rmse'] for t in r['cases'][sd]['tasks']]) for r in runs])}
    limits=[information_limit(sd,lam) for lam in (.2,.7) for sd in (.2,.05,.02,.005,0.)]
    print(json.dumps({'meta':{'version':11,'status':'TESTED; independent audits retained',
        'plant':theta,'replicates':12,'nested_training_n':[512,8192],
        'training_sensor_sd':.2,'test_sensor_sd':[.2,.02],
        'noise_and_goals':'same as v10, motorSD.03active then0coast',
        'costs_of_extra_training_and_sensor_precision_not_equated':True,
        'warning':'knownstructure/noisescale/zeroinitialstate; no generallearningclaim'},
        'information_limits':limits,'runs':runs,'summary':summary},indent=2,allow_nan=False))


if __name__=='__main__':main()
