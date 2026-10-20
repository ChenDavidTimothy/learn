#!/usr/bin/env python3
"""V10: learned consequences used for goal switching with hidden motor memory.

Frozen design before experiments: four plants, 512 bounded multiscale commands,
12 seeds, two unseen goal schedules; no reset at goal switches. All learned
methods use the same physical data. PD tuning occurs OFFLINE on learned dynamics,
not on the test plant. Oracle variants are explicitly privileged diagnostics.
"""
from __future__ import annotations
import json
import math
import platform
import numpy as np
import scipy
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from nonlinear_identification_v9 import trajectory,fit_output_error,STARTS,stat

SEED=20261020+1700000
PLANTS=[(.65,lam,c,1.2) for lam in (.2,.7) for c in (0.,.2)]
TRAIN_N=512
SENSOR_SD=.2
TEST_MOTOR_SD=.03
REPLICATES=12
ACTIVE=70
TAIL=16
HORIZON=8
PLAN_COAST=12
GOALS=((2.,-2.,1.),(6.,-4.,3.))
PD_GRID=[(kp,kd,alpha) for kp in (.08,.2,.5) for kd in (.2,.6,1.2)
         for alpha in (.25,.6)]


def transition(state,u,theta):
    x,v,m=state;rho,lam,c,beta=theta
    h=math.tanh(v);vn=rho*v-c*h+m
    new=np.array([x+vn,vn,lam*m+beta*u])
    a=rho-c*(1-h*h)
    jac=np.array([[1.,a,1.],[0.,a,1.],[0.,0.,lam]])
    return new,jac


def plan_cost(actions,state,goal,theta):
    # Zero-command continuation makes residual actuation observable in the loss.
    n=len(actions);z=state.copy();dz=np.zeros((3,n));grad=np.zeros(n);cost=0.
    for t in range(n+PLAN_COAST):
        u=actions[t] if t<n else 0.
        z,A=transition(z,u,theta)
        dz=A@dz
        if t<n:dz[2,t]+=theta[3]
        error=z[0]-goal
        cost+=error**2+.3*z[1]**2+.03*u*u
        grad+=2*error*dz[0]+.6*z[1]*dz[1]
        if t<n:grad[t]+=.06*u
    return float(cost),grad


def choose_action(state,goal,theta,warm):
    cache={}
    def calc(actions):
        if 'actions' not in cache or not np.array_equal(actions,cache['actions']):
            n=len(actions);z=state.copy();dz=np.zeros((3,n));residual=[];jac=[]
            for t in range(n+PLAN_COAST):
                u=actions[t] if t<n else 0.
                z,A=transition(z,u,theta);dz=A@dz
                direct=np.zeros(n)
                if t<n:
                    dz[2,t]+=theta[3]
                    direct[t]=math.sqrt(.03)
                residual.extend((z[0]-goal,math.sqrt(.3)*z[1],math.sqrt(.03)*u))
                jac.extend((dz[0].copy(),math.sqrt(.3)*dz[1],direct))
            cache.update(actions=actions.copy(),residual=np.array(residual),jac=np.array(jac))
        return cache
    # Same bounded residual formulation for linear and nonlinear models. The
    # former is convex; the latter still has no global-optimum certificate.
    fit=least_squares(lambda a:calc(a)['residual'],warm,jac=lambda a:calc(a)['jac'],
        bounds=(-1.,1.),method='trf',max_nfev=80,
        ftol=1e-10,xtol=1e-10,gtol=1e-8)
    initial=plan_cost(warm,state,goal,theta)[0]
    actions=fit.x if np.isfinite(fit.cost) and 2*fit.cost<=initial+1e-8 else warm
    selected_cost,selected_gradient=plan_cost(actions,state,goal,theta)
    projected=actions-np.clip(actions-selected_gradient,-1.,1.)
    return float(actions[0]),np.r_[actions[1:],0.],{
        'success':bool(fit.success),'status':int(fit.status),'njev':int(fit.njev),
        'nfev':int(fit.nfev),'initial_cost':initial,'selected_cost':selected_cost,
        'projected_gradient_max':float(np.max(np.abs(projected))),
        'used_optimizer_solution':bool(actions is fit.x),
        'selected_action_plan':actions.tolist()}


def goal_schedule(goals,active=ACTIVE,tail=TAIL):
    # The controller receives only the current goal, never the future schedule.
    return np.array([goals[0] if t<6 else goals[1] if t<28 else goals[2]
                     for t in range(active+tail)])


def metrics(states,commands,goals,active=ACTIVE,tail=TAIL):
    state=np.asarray(states)[1:];u=np.asarray(commands)
    error=state[:,0]-goals
    primary=np.mean(error[:active]**2+.3*state[:active,1]**2+.03*u[:active]**2)
    primary/=1+np.mean(goals[:active]**2)
    stopped=(np.abs(error)<=.3)&(np.abs(state[:,1])<=.15)
    check=stopped[active-5:]
    settle=[i for i in range(28,active) if np.all(stopped[i:])]
    return {'normalized_tracking_cost':float(primary),
        'stop_success':bool(np.all(check)),
        'late_position_rmse':float(np.sqrt(np.mean(error[active-10:]**2))),
        'late_velocity_rms':float(np.sqrt(np.mean(state[active-10:,1]**2))),
        'coast_position_max_error':float(np.max(np.abs(error[active:]))),
        'coast_velocity_max':float(np.max(np.abs(state[active:,1]))),
        'mean_action_energy':float(np.mean(u[:active]**2)),
        'settled_after_final_goal_steps':int(settle[0]-28) if settle else None}


def run_episode(true_theta,model_theta,method,goals,seed,pd=None,record=True):
    rng=np.random.default_rng(seed)
    n=ACTIVE+TAIL
    errors=rng.normal(0,SENSOR_SD,n+1)
    noises=rng.normal(0,TEST_MOTOR_SD,n)
    # Stop stored motion/force, rather than prohibit control while continually
    # adding a random walk disturbance. Sensor noise remains during this coast.
    noises[ACTIVE:]=0.
    targets=goal_schedule(goals)
    z=np.zeros(3);estimate=np.zeros(3);P=np.zeros((3,3));warm=np.zeros(HORIZON)
    Q=np.diag([0.,0.,TEST_MOTOR_SD**2])
    states=[z.copy().tolist()];observations=[float(errors[0])]
    estimates=[];commands=[];optimizers=[]
    smooth_x=0.;smooth_v=0.
    for t in range(n):
        o=z[0]+errors[t]
        if method in ('nominal_MPC','linear_MPC','nonlinear_MPC','oracle_model_EKF'):
            innovation=o-estimate[0]
            K=P[:,0]/(P[0,0]+SENSOR_SD**2)
            estimate=estimate+K*innovation
            correction=np.eye(3);correction[:,0]-=K
            P=correction@P@correction.T+SENSOR_SD**2*np.outer(K,K)
        elif method=='oracle_state_MPC':
            estimate=z.copy()  # Explicit privileged diagnostic.
        elif method in ('model_tuned_PD','fixed_PD'):
            kp,kd,alpha=pd
            previous=smooth_x
            smooth_x+=alpha*(o-smooth_x)
            smooth_v+=alpha*((smooth_x-previous)-smooth_v)
            estimate=np.array([smooth_x,smooth_v,0.])
        else:raise ValueError(method)
        estimates.append(estimate.tolist())
        if t>=ACTIVE:
            u=0.;info=None
        elif method in ('model_tuned_PD','fixed_PD'):
            u=float(np.clip(kp*(targets[t]-smooth_x)-kd*smooth_v,-1.,1.));info=None
        else:
            u,warm,info=choose_action(estimate,targets[t],model_theta,warm)
        commands.append(u);optimizers.append(info)
        z,_=transition(z,u,true_theta);z[2]+=noises[t]
        states.append(z.tolist());observations.append(float(z[0]+errors[t+1]))
        if method in ('nominal_MPC','linear_MPC','nonlinear_MPC','oracle_model_EKF'):
            estimate,A=transition(estimate,u,model_theta)
            P=A@P@A.T+(Q if t<ACTIVE else 0.)
    out=metrics(states,commands,targets)
    if record:
        out.update(states=states,observations=observations,estimates=estimates,
            commands=commands,goals=targets.tolist(),optimizers=optimizers,seed=seed)
    return out


def fit_linear(o,u):
    # Exact same raw position likelihood/initialization as nonlinear OE, c fixed0.
    cache={}
    def calc(p):
        if 'p' not in cache or not np.array_equal(p,cache['p']):
            theta=(p[0],p[1],0.,p[2]);x,_,J=trajectory(theta,u,True)
            cache.update(p=p.copy(),residual=x-o,jac=J[:,[0,1,3]])
        return cache
    records=[]
    for rho,lam,_,beta in STARTS:
        f=least_squares(lambda p:calc(p)['residual'],(rho,lam,beta),
            jac=lambda p:calc(p)['jac'],bounds=((.05,.05,.3),(.95,.95,2.)),
            max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        records.append({'theta':[float(f.x[0]),float(f.x[1]),0.,float(f.x[2])],
            'sse':float(2*f.cost),'success':bool(f.success),'nfev':int(f.nfev)})
    return np.array(min(records,key=lambda r:r['sse'])['theta']),records


def training(seed,theta):
    rng=np.random.default_rng(seed)
    commands=[]
    while len(commands)<TRAIN_N:
        duration=int(rng.choice((1,3,8)))
        commands.extend([float(rng.uniform(-1.,1.))]*duration)
    u=np.array(commands[:TRAIN_N])
    x,_,_=trajectory(theta,u)
    o=x+rng.normal(0,SENSOR_SD,len(x))
    return u,o


def tune_pd(learned_theta,seed):
    # Offline synthetic rollouts use only learned parameters and supplied sensor/
    # disturbance model. They consume NO additional physical plant experience.
    scores=[]
    for pd in PD_GRID:
        losses=[run_episode(learned_theta,learned_theta,'model_tuned_PD',goals,
            seed+j,pd=pd,record=False)['normalized_tracking_cost']
            for j,goals in enumerate(((1.,-1.,.5),(-1.,1.,-.5)))]
        scores.append(float(np.mean(losses)))
    k=int(np.argmin(scores))
    return PD_GRID[k],scores


def predictive_score(theta,true_theta,seed):
    # Separate clean replay used for scoring only, never fit/tuning selection.
    rng=np.random.default_rng(seed);u=rng.uniform(-1.,1.,256)
    x,v,_=trajectory(true_theta,u);xp,vp,_=trajectory(theta,u)
    return {'position_rmse':float(np.sqrt(np.mean((x-xp)**2))),
            'velocity_rmse':float(np.sqrt(np.mean((v-vp)**2)))}


def checks():
    theta=np.array([.65,.7,.2,1.2]);z=np.array([.5,-.4,.2])
    u=np.linspace(-.7,.7,HORIZON);_,g=plan_cost(u,z,2.,theta)
    numeric=[]
    for i in range(HORIZON):
        d=np.zeros(HORIZON);d[i]=1e-6
        numeric.append((plan_cost(u+d,z,2.,theta)[0]-plan_cost(u-d,z,2.,theta)[0])/2e-6)
    e=float(np.max(np.abs(g-numeric)))
    assert e<1e-5
    return {'action_gradient_max_error':e}


def main():
    plants=[]
    methods=('fixed_PD','model_tuned_PD','nominal_MPC','linear_MPC','nonlinear_MPC',
             'oracle_model_EKF','oracle_state_MPC')
    with threadpool_limits(limits=1):
        for pi,true_theta in enumerate(PLANTS):
            runs=[]
            for i in range(REPLICATES):
                seed=SEED+10000*pi+100*i
                u,o=training(seed,true_theta)
                nl,nl_records=fit_output_error(o,u)
                linear,linear_records=fit_linear(o,u)
                pd,pd_scores=tune_pd(nl,seed+1000000)
                tasks=[]
                for j,goals in enumerate(GOALS):
                    controllers={}
                    for name in methods:
                        model=(.6,.4,.1,1.) if name=='nominal_MPC' else linear if name=='linear_MPC' else nl if name in (
                            'nonlinear_MPC','model_tuned_PD','fixed_PD') else true_theta
                        gains=(.2,.6,.25) if name=='fixed_PD' else pd
                        controllers[name]=run_episode(true_theta,model,name,goals,
                            seed+2000000+j,pd=gains)
                    tasks.append({'goal_schedule':goals,'controllers':controllers})
                runs.append({'seed':seed,'training_commands':u.tolist(),
                    'training_positions':o.tolist(),'nonlinear_theta':nl.tolist(),
                    'linear_theta':linear.tolist(),'nonlinear_optimization':nl_records,
                    'linear_optimization':linear_records,'selected_PD':pd,
                    'PD_grid_scores':pd_scores,'tasks':tasks,
                    'clean_prediction':{name:predictive_score(p,true_theta,seed+3000000)
                        for name,p in (('linear',linear),('nonlinear',nl))}})
            summary={}
            for name in methods:
                # Two task outcomes share one fitted-model seed; average per seed
                # before Monte Carlo SE. Success count still reports all24 tasks.
                summaries={}
                for key in ('normalized_tracking_cost','late_position_rmse',
                            'late_velocity_rms','mean_action_energy'):
                    summaries[key]=stat([np.mean([t['controllers'][name][key]
                        for t in r['tasks']]) for r in runs])
                summaries['stop_successes']=sum(t['controllers'][name]['stop_success']
                    for r in runs for t in r['tasks'])
                summaries['task_trials']=len(runs)*len(GOALS)
                summaries['optimizer_nonconvergences']=sum(not o['success']
                    for r in runs for t in r['tasks']
                    for o in t['controllers'][name]['optimizers'] if o is not None)
                summary[name]=summaries
            plants.append({'true_theta':true_theta,'runs':runs,'summary':summary})
            print('completed plant',pi,flush=True,file=__import__('sys').stderr)
    print(json.dumps({'meta':{'version':10,'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,'seed':SEED,
        'plant_grid':PLANTS,'training_n':TRAIN_N,'replicates':REPLICATES,
        'sensor_sd':SENSOR_SD,'training_motor_process_sd':0.,
        'test_motor_process_sd':TEST_MOTOR_SD,'active_steps':ACTIVE,
        'coast_motor_process_sd':0.,'nominal_theta':[.6,.4,.1,1.],
        'zero_command_test_tail':TAIL,'goal_switches':[6,28],
        'MPC_control_horizon':HORIZON,'MPC_zero_command_coast':PLAN_COAST,
        'optimizer':'bounded nonlinear least squares TRF, analytic residual Jacobian, max_nfev80',
        'cost':'((x-goal)^2+.3*v^2+.03*u^2), normalized by1+mean(goal^2)',
        'success':'all final5 active and16 coast steps |x-goal|<=.3 and |v|<=.15',
        'PD_grid':PD_GRID,'PD_tuning':'offline learned nonlinear model only',
        'privileged':['oracle_model_EKF has true parameters','oracle_state_MPC also has true state'],
        'all_noise_scales_initial_state_and_function_family_supplied':True,
        'warning':'conventional identification/filtering/control in supplied family; no general learner claim'},
        'checks':checks(),'plants':plants},indent=2,allow_nan=False))


if __name__=='__main__':main()
