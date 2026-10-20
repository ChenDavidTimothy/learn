#!/usr/bin/env python3
"""V3: action-response footprints from continuous position-only car trajectories.

Reproduce: .venv/bin/python research/car_learning_benchmark_v3.py
All experiments run with project-local NumPy. No direct hidden velocity input to estimators.
"""
from __future__ import annotations
import json
import numpy as np

SEED = 20261020
RHO, BETA = .75, 1.2
N, REPS, BURN = 4400, 48, 400

def simulate(seed, n=N, burn=BURN, sensor_sd=0., process_sd=.06,
             rho=RHO, beta=BETA, gamma=0., nonlinear=0., amplitude=.7,
             feedback=0., return_hidden=False):
    rng = np.random.default_rng(seed)
    steps=n+burn+5
    z=rng.normal(0.,amplitude,steps)
    e=rng.normal(0.,sensor_sd,steps+1)
    w=rng.normal(0.,process_sd,steps)
    x=np.zeros(steps+1)
    v=np.zeros(steps+1)
    a=np.zeros(steps)
    obs=np.zeros(steps+1)
    obs[0]=e[0]
    for t in range(steps):
        d=obs[t]-obs[t-1] if t else 0.
        a[t]=z[t]-feedback*d
        prev=a[t-1] if t else 0.
        v[t+1]=rho*v[t]-nonlinear*v[t]*abs(v[t])+beta*a[t]+gamma*prev+w[t]
        x[t+1]=x[t]+v[t+1]
        obs[t+1]=x[t+1]+e[t+1]
    # Analysis uses ONLY observed positions, actions, and optional known innovations z.
    sl=slice(burn, burn+n)
    output=(obs[burn:burn+n+1],a[sl],z[sl])
    if return_hidden:
        return (*output,float(np.mean(np.abs(v[burn+1:burn+n+1]))))
    return output

def signals(obs,a,z):
    d=np.diff(obs)
    t=np.arange(3,len(a))
    y=d[t]  # observed displacement caused by current a[t]
    cols=np.column_stack([a[t-j] for j in range(4)])
    X=np.column_stack([d[t-1],a[t],a[t-1]])
    Z=np.column_stack([a[t-2],a[t],a[t-1]])
    return d,t,y,cols,X,Z

def least_squares(A,b):
    return np.linalg.lstsq(A,b,rcond=None)[0]

def estimators(obs,a,z):
    d,t,y,cols,X,Z=signals(obs,a,z)
    # No privileged knowledge of v is supplied to any estimator.
    arx=least_squares(X,y)
    moments=Z.T@X
    iv=least_squares(moments,Z.T@y)
    # Regression of outputs on 4 independent current and past motor commands:
    # for white random inputs this estimates Markov/impulse coefficients.
    h=least_squares(np.column_stack([np.ones(len(y)),cols]),y)[1:]
    weak=abs(h[1])<.06 # declared fixed magnitude cutoff for this scale
    if weak:
        theta=[float('nan'),h[0],float('nan')]
    else:
        rho=h[2]/h[1]
        theta=[rho,h[0],h[1]-rho*h[0]]
    # Input moment rank can collapse even for observable physical states.
    return dict(arx=arx,iv=iv,footprint=np.asarray(theta),
                h=h,weak=weak,moment_sv=np.linalg.svd(moments,compute_uv=False),
                direct_action_h0=h[0],exogenous_probe_beta=np.cov(y,z[t],bias=True)[0,1]/np.var(z[t]))

def inspect_case(case,seed):
    kw=case.copy()
    gamma=kw.get('gamma',0.)
    obs,a,z=simulate(seed,**kw)
    estim=estimators(obs,a,z)
    true=np.array([RHO,BETA,gamma])
    return estim,true

def compare_main():
    cases={
      'linear_exact':dict(sensor_sd=0.,gamma=0.),
      'linear_sensor_1_2':dict(sensor_sd=1.2,gamma=0.),
      'linear_sensor_2_4':dict(sensor_sd=2.4,gamma=0.),
      'delayed_sensor_1_2':dict(sensor_sd=1.2,gamma=.34),
      'cancelled_sensor_1_2':dict(sensor_sd=1.2,gamma=-RHO*BETA),
      'nonlinear_sensor_1_2':dict(sensor_sd=1.2,gamma=0.,nonlinear=.12),
    }
    rows={}
    for ci,(label,config) in enumerate(cases.items()):
        individual=[]
        for i in range(REPS):
            est,true=inspect_case(config,SEED+1000*ci+i)
            entry={'weak':est['weak'],
                   'h':est['h'].tolist(),
                   'truth':true.tolist(),
                   'svmin':float(est['moment_sv'][-1])}
            for name in ('arx','iv','footprint'):
                p=est[name]
                entry[name]=p.tolist()
                entry[name+'_error']=float(np.linalg.norm(p-true)) if np.all(np.isfinite(p)) else None
            individual.append(entry)
        summary={'replicates':REPS,'steps_per_run':N,'burn':BURN,
                 'sensor_sd':config.get('sensor_sd'),'gamma':config.get('gamma'),
                 'nonlinear':config.get('nonlinear',0.),
                 'weak_footprint_fraction':sum(q['weak'] for q in individual)/REPS}
        for method in ('arx','iv','footprint'):
            estimates=np.array([q[method] for q in individual],float)
            finite=np.all(np.isfinite(estimates),axis=1)
            errors=[q[method+'_error'] for q in individual if q[method+'_error'] is not None]
            summary[method]={'n_estimable':int(sum(finite)),
                     'mean_parameters':np.nanmean(estimates,axis=0).tolist(),
                     'median_l2_error':float(np.median(errors)) if errors else None,
                     'mean_l2_error':float(np.mean(errors)) if errors else None}
        summary['mean_h']=np.mean([q['h'] for q in individual],axis=0).tolist()
        summary['mean_minimum_moment_singular_value']=float(np.mean([q['svmin'] for q in individual]))
        rows[label]=summary
    return rows

def amplitude_test(seed_offset=16000, reps=REPS):
    # Protocol fixed BEFORE looking at inference outcomes.
    # Compare causal echo h1/h0 under motor amplitude .35 and 1.25.
    # In the linear car its population value is rho; with nonlinear drag
    # the distribution of velocities shifts the average dynamic response.
    out={}
    for law,nonlinear in [('linear',0.),('drag',.12)]:
        differences=[];ratios=[];ols_ratios=[];theory_ratios=[]
        for i in range(reps):
            current=[];current_ols=[];current_theory=[]
            for j,amplitude in enumerate((.35,1.25)):
                obs,a,z,mean_abs_v=simulate(SEED+seed_offset+400*i+2*j,
                         n=5200,sensor_sd=1.,nonlinear=nonlinear,
                         amplitude=amplitude,return_hidden=True)
                est=estimators(obs,a,z)
                current.append(float(est['h'][1]/est['h'][0]))
                current_ols.append(float(est['arx'][0]))
                # Hidden state used solely for an independently checkable
                # ground-truth consequence of Gaussian integration by parts.
                current_theory.append(RHO-2*nonlinear*mean_abs_v)
            ratios.append(current)
            ols_ratios.append(current_ols)
            theory_ratios.append(current_theory)
            differences.append(current[1]-current[0])
        arr=np.asarray(ratios)
        o=np.asarray(ols_ratios)
        th=np.asarray(theory_ratios)
        out[law]={'replicates':reps,'per_amplitude_steps':5200,
                  'amplitudes':[.35,1.25],
                  'mean_first_echo_ratio_small_large':arr.mean(axis=0).tolist(),
                  'mean_ols_rho_small_large':o.mean(axis=0).tolist(),
                  'mean_steins_identity_prediction_small_large':th.mean(axis=0).tolist(),
                  'mean_echo_minus_stein_per_amplitude':(arr-th).mean(axis=0).tolist(),
                  'mean_large_minus_small':float(np.mean(differences)),
                  'mean_ols_large_minus_small':float(np.mean(o[:,1]-o[:,0])),
                  'std_large_minus_small':float(np.std(differences,ddof=1)),
                  'fraction_abs_change_over_0_06':float(np.mean(np.abs(differences)>.06)),
                  'fraction_one_sided_change_below_minus_0_12':float(np.mean(arr[:,1]-arr[:,0]<-.12)),
                  'fraction_change_below_minus_0_06':float(np.mean(arr[:,1]-arr[:,0]<-.06))}
    return out

def feedback_failure():
    # Feedback a_t=z_t-0.35*(o_t-o_{t-1}) is driven by noisy observations.
    # z_t is exogenous intervention injected at each time.
    rows={}
    for cond,k in [('open_loop',0.),('feedback',.35)]:
        values=[]
        for i in range(REPS):
            obs,a,z=simulate(SEED+28000+i,n=5200,sensor_sd=1.2,feedback=k,amplitude=.7)
            est=estimators(obs,a,z)
            values.append([est['direct_action_h0'],est['exogenous_probe_beta'],
                           est['h'][1]])
        q=np.asarray(values)
        rows[cond]={'replicates':REPS,'steps_per_run':5200,
                    'feedback_gain':k,'true_immediate_motor_gain':BETA,
                    'mean_h0_correlated_action':float(q[:,0].mean()),
                    'mean_beta_exogenous_probe':float(q[:,1].mean()),
                    'sd_h0':float(q[:,0].std(ddof=1)),
                    'sd_probe_beta':float(q[:,1].std(ddof=1)),
                    'mean_h1_from_action_regression':float(q[:,2].mean())}
    return rows

def main():
    result={'meta':{'version':3,'seed':SEED,'replicates':REPS,
        'steps_per_main_run':N,'burn':BURN,'process_noise_sd':.06,
        'motor_noise_sd':.7,'rho':RHO,'beta':BETA,
        'observation':'only noisy position, motor actions; no direct velocity sensor',
        'policies':'independent Gaussian motor actions except feedback test'},
        'main':compare_main(),'amplitude_dependence':amplitude_test(),
        'amplitude_independent_confirmation':amplitude_test(seed_offset=50000,reps=64),
        'feedback_instrument':feedback_failure()}
    linear=result['main']['linear_sensor_1_2']
    assert linear['footprint']['median_l2_error']<.11,linear
    assert linear['arx']['median_l2_error']>.2
    assert result['main']['delayed_sensor_1_2']['footprint']['median_l2_error']<.15
    assert result['main']['cancelled_sensor_1_2']['weak_footprint_fraction']>.75
    assert result['amplitude_dependence']['drag']['mean_large_minus_small']<-.06
    assert result['feedback_instrument']['feedback']['mean_beta_exogenous_probe']>1.
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
