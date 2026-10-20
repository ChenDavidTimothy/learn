#!/usr/bin/env python3
"""V8: full input/output pole-label ambiguity and zero-command falsification."""
from __future__ import annotations
import json
import math
import platform
import numpy as np
from threadpoolctl import threadpool_limits
from feedback_virtual_zero_v7 import ATOM_PROB, hermite_weights

SEED=20261020
RHO,LAM,BETA=.65,.4,1.2


def simulate(seed,rho=RHO,lam=LAM,n=65536,sd=.7,sensor_sd=.5,policy='gaussian'):
    rng=np.random.default_rng(seed)
    total=n+604
    normals=rng.normal(size=total)
    coins=rng.random(total)
    motor_noise=rng.normal(0,.06,total)
    e=rng.normal(0,sensor_sd,total+1)
    force_error=rng.normal(0,.2,total)
    v=np.zeros(total+1); m=np.zeros(total+1); d=np.zeros(total+1)
    u=np.zeros(total); z=np.zeros(total); pi=np.zeros(total); atom=np.zeros(total,bool)
    for t in range(total):
        pi[t]=-.5*math.tanh(d[t])
        if policy=='atom':
            p=ATOM_PROB
            variance=sd*sd/(1-p)-p*pi[t]*pi[t]/(1-p)**2
            atom[t]=coins[t]<p
            u[t]=0. if atom[t] else pi[t]/(1-p)+math.sqrt(variance)*normals[t]
            z[t]=u[t]-pi[t]
        else:
            z[t]=sd*normals[t]
            u[t]=pi[t]+z[t]
        m[t+1]=lam*m[t]+BETA*u[t]+motor_noise[t]
        v[t+1]=rho*v[t]+m[t+1]
        d[t+1]=v[t+1]+e[t+1]-e[t]
    t=np.arange(604,total)
    return {'D':d[t],'Dprev':d[t-1],'Y':d[t+1], 'u':u[t], 'z':z[t],
            'zprev':z[t-1],'zprev2':z[t-2],'pi':pi[t],'atom':atom[t],
            'force_measured':m[t+1]+force_error[t],
            'latent_force':m[t+1]}


def fit(data,policy,sd=.7):
    D,Dprev,Y,u,z,z1,z2=[data[k] for k in ('D','Dprev','Y','u','z','zprev','zprev2')]
    X=np.c_[D,u];Z=np.c_[z1,z]
    ar1=np.linalg.lstsq(Z.T@X,Z.T@Y,rcond=None)[0]
    X2=np.c_[D,Dprev,u];Z2=np.c_[z1,z2,z]
    ar2=np.linalg.lstsq(Z2.T@X2,Z2.T@Y,rcond=None)[0]
    roots=np.roots([1.,-ar2[0],-ar2[1]])
    if np.max(np.abs(roots.imag))<1e-8:
        poles=sorted(roots.real.tolist())
        error=float(np.linalg.norm(np.array(poles)-[LAM,RHO]))
    else:
        poles=None;error=None
    weights=hermite_weights(data['pi'],z,sd,1) if policy=='gaussian' else data['atom']/ATOM_PROB
    den=np.mean(z1*D)
    apparent=float(np.mean(z1*weights*Y)/den)
    force_rho=float(np.mean(z1*(Y-data['force_measured']))/den)
    residual=Y-ar1[0]*D-ar1[1]*u
    return {'ar1':ar1.tolist(),'ar2':ar2.tolist(),'poles_unordered':poles,'pole_set_error':error,
            'apparent_passive':apparent,'with_added_force_sensor_rho':force_rho,
            'ar1_unused_lag2_residual_moment':float(np.mean(z2*residual)/sd**2)}


def forecast(theta,commands,order):
    pred=np.zeros(len(commands)+1)
    for t,a in enumerate(commands):
        if order==1:pred[t+1]=theta[0]*pred[t]+theta[1]*a
        else:pred[t+1]=theta[0]*pred[t]+theta[1]*(pred[t-1] if t else 0.)+theta[2]*a
    return pred[1:]


def stat(a,target=None):
    a=np.asarray(a)
    r={'mean':float(a.mean()),'mc_se':float(a.std(ddof=1)/np.sqrt(len(a))),
       'q10':float(np.quantile(a,.1)),'q90':float(np.quantile(a,.9))}
    if target is not None:r['rmse']=float(np.sqrt(np.mean((a-target)**2)))
    return r


def disconnect_check():
    states=[]
    for rho,lam in ((RHO,LAM),(LAM,RHO)):
        v=m=0.
        for _ in range(8):
            m=lam*m+BETA*1.3
            v=rho*v+m
        initial=v
        velocities=[]
        for _ in range(10):
            # New physical intervention: motor force is disconnected, not u=0.
            v=rho*v
            velocities.append(v)
        states.append({'mechanical_rho':rho,'motor_lambda':lam,'initial_velocity':initial,
                       'disconnected_velocities':velocities})
    assert abs(states[0]['initial_velocity']-states[1]['initial_velocity'])<1e-12
    return states


def component_transfer():
    # Same pretraining law, different response after a labeled motor change.
    new_motor=.2
    commands=.7*np.sin(.7*np.arange(20))
    a=forecast([RHO+new_motor,-RHO*new_motor,BETA],commands,2)
    b=forecast([LAM+new_motor,-LAM*new_motor,BETA],commands,2)
    distance=float(np.sqrt(np.mean((a-b)**2)))
    assert distance>0
    return {'new_motor_pole':new_motor,'commands':commands.tolist(),
            'original_world_response':a.tolist(),'swapped_world_response':b.tolist(),
            'response_rmse_distance':distance,
            'two_world_minimax_expected_rmse_lower_bound':distance/2,
            'intervention_is_outside_command_only_training_interface':True}


def main():
    out={}
    with threadpool_limits(limits=1):
        for ci,(policy,noise) in enumerate((('gaussian',.5),('atom',.5),('atom',1.5))):
            rows=[]
            for i in range(48):
                seed=SEED+1300000+1000*ci+i
                a=simulate(seed,policy=policy,sensor_sd=noise)
                b=simulate(seed,rho=LAM,lam=RHO,policy=policy,sensor_sd=noise)
                fa,fb=fit(a,policy),fit(b,policy)
                delta=max(float(np.max(np.abs(a[k]-b[k]))) for k in ('D','Y','u','pi','z'))
                assert delta<1e-10
                rng=np.random.default_rng(seed+2000000)
                actions=rng.normal(0,.7,20)
                truth=forecast([RHO+LAM,-RHO*LAM,BETA],actions,2)
                rows.append({'seed':seed,'original':fa,'swapped':fb,
                    'paired_max_observable_difference':delta,
                    'max_latent_force_difference':float(np.max(np.abs(a['latent_force']-b['latent_force']))),
                    'forecast_ar1_rmse':float(np.sqrt(np.mean((forecast(fa['ar1'],actions,1)-truth)**2))),
                    'forecast_ar2_rmse':float(np.sqrt(np.mean((forecast(fa['ar2'],actions,2)-truth)**2)))})
            finite=[r['original']['pole_set_error'] for r in rows if r['original']['pole_set_error'] is not None]
            out[f'{policy}_sensor{noise}']={'policy':policy,'sensor_sd':noise,'runs':rows,
                'apparent_passive':stat([r['original']['apparent_passive'] for r in rows],RHO+LAM),
                'ar1_rho':stat([r['original']['ar1'][0] for r in rows],RHO+LAM),
                'with_force_original_rho':stat([r['original']['with_added_force_sensor_rho'] for r in rows],RHO),
                'with_force_swapped_rho':stat([r['swapped']['with_added_force_sensor_rho'] for r in rows],LAM),
                'real_pole_replicates':len(finite),'mean_pole_set_error':float(np.mean(finite)) if finite else None,
                'max_paired_observable_difference':max(r['paired_max_observable_difference'] for r in rows),
                'lag2_residual_moment':stat([r['original']['ar1_unused_lag2_residual_moment'] for r in rows],-RHO*LAM*BETA),
                'forecast_ar1_rmse':stat([r['forecast_ar1_rmse'] for r in rows]),
                'forecast_ar2_rmse':stat([r['forecast_ar2_rmse'] for r in rows])}
    print(json.dumps({'meta':{'version':8,'seed':SEED,'python':platform.python_version(),'numpy':np.__version__,
        'rho':RHO,'lambda':LAM,'beta':BETA,'n':65536,'replicates':48,'discarded':604,
        'motor_process_sd':.06,'extra_force_sensor_sd':.2,
        'physical_force_sensor_is_additional_information':True,
        'claim':'full command-position equivalence and actuator-memory falsification; not a new estimator'},
        'physical_disconnect':disconnect_check(),'component_transfer':component_transfer(),
        'experiments':out},indent=2,allow_nan=False))


if __name__=='__main__':main()
