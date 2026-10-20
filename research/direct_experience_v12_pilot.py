#!/usr/bin/env python3
"""Unreleased v12 pilot: direct behavior search with observation-only fitness.

Preparatory experiment, NOT a completed model-based/model-free comparison.
No physical/predictive model is fitted. A supplied PD family is searched by CEM
using fresh noisy episodes; true physical states are evaluator-only. Continuous
feedback throughout testing avoids the earlier forced-control-cutoff task.
"""
import json
import math
import platform
import numpy as np

SEED=20261020+1800000
THETA=(.65,.7,.2,1.2)
SENSOR_SD=.2
N=86
TRAIN_GOALS=((1.,-1.,.5),(-1.,1.,-.5))
TEST_GOALS=((2.,-2.,1.),(6.,-4.,3.))


def episode(theta,gains,goals,seed,motor_sd,record=True):
    rng=np.random.default_rng(seed)
    sensor=rng.normal(0,SENSOR_SD,N+1)
    motor_noise=rng.normal(0,motor_sd,N)
    rho,lam,c,beta=theta;kp,kd,alpha=gains
    x=v=m=smooth_x=smooth_v=0.
    states=[[x,v,m]];obs=[float(sensor[0])];actions=[]
    target=np.array([goals[0] if t<6 else goals[1] if t<28 else goals[2] for t in range(N)])
    for t in range(N):
        # Controller sees only current observation, goal and its own memory.
        old=smooth_x
        smooth_x+=alpha*(obs[-1]-smooth_x)
        smooth_v+=alpha*((smooth_x-old)-smooth_v)
        u=float(np.clip(kp*(target[t]-smooth_x)-kd*smooth_v,-1.,1.))
        actions.append(u)
        vnext=rho*v-c*math.tanh(v)+m
        m=lam*m+beta*u+motor_noise[t]
        v=vnext;x+=v
        states.append([x,v,m]);obs.append(float(x+sensor[t+1]))
    u=np.array(actions);o=np.array(obs);s=np.array(states)[1:]
    scale=1+np.mean(target**2)
    observed=float(np.mean((o[1:]-target)**2+.3*np.diff(o)**2+.03*u*u)/scale)
    # Everything below is held outside learner fitness/parameter choice.
    actual=float(np.mean((s[:,0]-target)**2+.3*s[:,1]**2+.03*u*u)/scale)
    expected_offset=1.6*SENSOR_SD**2/scale
    result={'seed':seed,'observed_fitness':observed,'actual_evaluator_cost':actual,
        'expected_observation_loss_offset':float(expected_offset),
        'fresh_gap_minus_expected_offset':float(observed-actual-expected_offset),
        'late_position_rmse':float(np.sqrt(np.mean((s[-10:,0]-target[-10:])**2))),
        'late_velocity_rms':float(np.sqrt(np.mean(s[-10:,1]**2))),
        'stationary_success':bool(np.all(np.abs(s[-10:,0]-target[-10:])<=.3)
            and np.all(np.abs(s[-10:,1])<=.15))}
    if record:result.update(states=states,observations=obs,commands=actions,goals=target.tolist())
    return result


def gains_from_coordinates(p):
    return [float(math.exp(p[0])),float(math.exp(p[1])),float(1/(1+math.exp(-p[2])))]


def train(seed):
    rng=np.random.default_rng(seed)
    mean=np.array([math.log(.08),math.log(.6),math.log(.6/.4)])
    sd=np.ones(3)
    lower=np.array([math.log(.005),math.log(.02),math.log(.1/.9)])
    upper=np.array([math.log(.5),math.log(4.),math.log(.98/.02)])
    trials=[];best=None
    for generation in range(4):
        coordinates=np.clip(rng.normal(mean,sd,(12,3)),lower,upper)
        if best is not None:coordinates[0]=best['coordinates']
        elif generation==0:
            coordinates[0]=mean
            coordinates[1]=[math.log(.02),0.,math.log(.8/.2)]
        current=[]
        for j,p in enumerate(coordinates):
            gains=gains_from_coordinates(p)
            index=12*generation+j
            episodes=[episode(THETA,gains,g,seed+100000+100*index+k,0.)
                      for k,g in enumerate(TRAIN_GOALS)]
            # Learner selection reads ONLY observed fitness, never evaluator cost.
            fitness=float(np.mean([e['observed_fitness'] for e in episodes]))
            row={'generation':generation,'candidate':j,'coordinates':p.tolist(),
                 'gains':gains,'observed_fitness':fitness,'episodes':episodes}
            current.append(row);trials.append(row)
            if best is None or fitness<best['observed_fitness']:best=row
        elite=sorted(current,key=lambda r:r['observed_fitness'])[:4]
        values=np.array([r['coordinates'] for r in elite])
        mean=values.mean(axis=0);sd=np.maximum(values.std(axis=0),.15)
    return best['gains'],trials


def motor_first_counterexample():
    # Bounded one-step policy u=-k*tanh(e0/SD), x0=v0=m0=0.
    # Analytic quadrature uses E[e0*tanh(e0/SD)] to expose policy-dependent bias.
    nodes,weights=np.polynomial.hermite.hermgauss(80)
    e=math.sqrt(2)*SENSOR_SD*nodes
    cross=float(np.sum(weights*e*np.tanh(e/SENSOR_SD))/math.sqrt(math.pi))
    rows=[]
    rng=np.random.default_rng(SEED-1)
    e0=rng.normal(0,SENSOR_SD,1000000);e1=rng.normal(0,SENSOR_SD,len(e0))
    for k in (0.,.3,.9):
        u=-k*np.tanh(e0/SENSOR_SD);v=THETA[3]*u
        measured=v+e1-e0
        gap=(v+e1)**2+.3*measured**2-(v*v+.3*v*v)
        target=1.6*SENSOR_SD**2+.6*THETA[3]*k*cross
        rows.append({'k':k,'expected_stage_loss_gap':target,'measured_gap':float(gap.mean()),
                     'mc_se':float(gap.std(ddof=1)/math.sqrt(len(gap)))})
    return {'n':len(e0),'sensor_cross_moment':cross,'cases':rows,
            'warning':'one-step motor-first contrast; not the lagged learning environment'}


def main():
    rows=[]
    for i in range(8):
        seed=SEED+10000*i
        gains,trials=train(seed)
        tests=[]
        for j,goals in enumerate(TEST_GOALS):
            fresh=seed+2000000+j
            tests.append({'goal_schedule':goals,
                'learned':episode(THETA,gains,goals,fresh,.03),
                'fixed_PD':episode(THETA,(.2,.6,.25),goals,fresh,.03)})
        rows.append({'seed':seed,'selected_gains':gains,'training':trials,'tests':tests})
    summary={}
    for label in ('learned','fixed_PD'):
        a=np.array([np.mean([t[label]['actual_evaluator_cost'] for t in r['tests']]) for r in rows])
        b=np.array([np.mean([t[label]['fresh_gap_minus_expected_offset'] for t in r['tests']]) for r in rows])
        summary[label]={'mean_actual_cost':float(a.mean()),'mc_se_actual_cost':float(a.std(ddof=1)/math.sqrt(len(a))),
            'mean_fresh_gap_minus_expected_offset':float(b.mean()),'mc_se_gap':float(b.std(ddof=1)/math.sqrt(len(b))),
            'stationary_successes':sum(t[label]['stationary_success'] for r in rows for t in r['tests']),
            'task_trials':16}
    print(json.dumps({'meta':{'version':12,'status':'UNRELEASED PILOT; artifact audit retained separately; fair model-based comparison pending',
        'python':platform.python_version(),'numpy':np.__version__,'seed':SEED,'true_theta_evaluator_only':THETA,
        'n':N,'replicates':8,'training_generations':4,'candidates_per_generation':12,
        'training_goals_per_candidate':2,'physical_training_steps_per_seed':4*12*2*N,
        'sensor_sd':SENSOR_SD,'training_motor_sd':0.,'test_motor_sd':.03,
        'controller_family_supplied':'PD with two causal smoothing memories; CEM is established policy search',
        'feedback_continues_all86_test_steps':True,
        'selection_bias':'winning finite training fitness is optimistic; only fresh evaluation tests the fixed-policy loss identity',
        'warning':'fixedPD is descriptive, not a strong or matched-data learning baseline; no competitive claim'},
        'motor_first_counterexample':motor_first_counterexample(),'runs':rows,'summary':summary},indent=2,allow_nan=False))


if __name__=='__main__':main()
