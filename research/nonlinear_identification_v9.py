#!/usr/bin/env python3
"""V9: restricted nonlinear identification versus observation uncertainty.

Lagged timing throughout: vnext=rho*v-c*tanh(v)+m;
mnext=lambda*m+beta*u. Only commanded u and noisy integrated position o
go to the estimators. All initial physical states are known zero. No process
noise in this identification experiment (the separate gauge test includes it).
"""
from __future__ import annotations
import json
import math
import platform
import numpy as np
import scipy
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

SEED=20261020+1500000
RHO,LAM,BETA=.65,.4,1.2
STARTS=((.7,.3,.1,1.),(.3,.7,.02,1.),(.8,.6,.3,1.4))
LOWER=(.05,.05,0.,.3)
UPPER=(.95,.95,.4,2.)


def trajectory(theta,u,jacobian=False):
    """Position and exact sensitivities wrt [rho,lambda,c,beta]."""
    rho,lam,c,beta=theta
    n=len(u)
    x=np.zeros(n+1);v=np.zeros(n+1);m=0.
    dv=np.zeros(4);dm=np.zeros(4);dx=np.zeros(4)
    jac=np.zeros((n+1,4)) if jacobian else None
    for t,a in enumerate(u):
        h=math.tanh(v[t])
        v[t+1]=rho*v[t]-c*h+m
        if jacobian:
            dvnew=(rho-c*(1-h*h))*dv+dm+np.array([v[t],0.,-h,0.])
            dmnew=lam*dm+np.array([0.,m,0.,a])
            dx=dx+dvnew
            jac[t+1]=dx
            dv,dm=dvnew,dmnew
        m=lam*m+beta*a
        x[t+1]=x[t]+v[t+1]
    return x,v,jac


def regression(o,u):
    # D[t]=o[t]-o[t-1], so index 0 here corresponds to physical t=1.
    d=np.diff(o)
    X=np.c_[d[1:-1],d[:-2],np.tanh(d[1:-1]),np.tanh(d[:-2]),u[1:-1]]
    y=d[2:]
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    singular=np.linalg.svd(X,compute_uv=False)
    # For physical t=1,...,n-2, y=v[t+2] and current command is u[t].
    return coef,singular


def observable_forecast(coef,u):
    v=np.zeros(len(u)+1)
    for t in range(len(u)-1):
        v[t+2]=(coef[0]*v[t+1]+coef[1]*v[t]+coef[2]*math.tanh(v[t+1])
                +coef[3]*math.tanh(v[t])+coef[4]*u[t])
    return v


def fit_output_error(o,u):
    cache={}
    def calculate(theta):
        if 'theta' not in cache or not np.array_equal(cache['theta'],theta):
            x,_,jac=trajectory(theta,u,True)
            cache.update(theta=theta.copy(),residual=x-o,jac=jac)
        return cache
    fits=[]
    for start in STARTS:
        fit=least_squares(lambda p:calculate(p)['residual'],start,
            jac=lambda p:calculate(p)['jac'],bounds=(LOWER,UPPER),
            max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        fits.append({'theta':fit.x.tolist(),'sse':float(2*fit.cost),
                     'success':bool(fit.success),'status':int(fit.status),
                     'nfev':int(fit.nfev),'optimality':float(fit.optimality)})
    best=min(fits,key=lambda r:r['sse'])
    return np.array(best['theta']),fits


def fit_structured_regression(o,u):
    """Same physical family, bounds and starts as OE; noisy regressors."""
    d=np.diff(o);v=d[1:-1];p=d[:-2];a=u[1:-1];y=d[2:]
    h=np.tanh(v);hp=np.tanh(p)
    def residual(theta):
        rho,lam,c,beta=theta
        return (rho+lam)*v-rho*lam*p-c*h+lam*c*hp+beta*a-y
    def jacobian(theta):
        rho,lam,c,_=theta
        return np.c_[v-lam*p,v-rho*p+c*hp,-h+lam*hp,a]
    fits=[]
    for start in STARTS:
        fit=least_squares(residual,start,jac=jacobian,bounds=(LOWER,UPPER),
            max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        fits.append({'theta':fit.x.tolist(),'sse':float(2*fit.cost),
                     'success':bool(fit.success),'status':int(fit.status),
                     'nfev':int(fit.nfev),'optimality':float(fit.optimality)})
    return np.array(min(fits,key=lambda r:r['sse'])['theta']),fits


def stat(values):
    a=np.array(values,dtype=float)
    assert np.all(np.isfinite(a))
    return {'mean':float(a.mean()),'mc_se':float(a.std(ddof=1)/np.sqrt(len(a))),
            'median':float(np.median(a)),'q10':float(np.quantile(a,.1)),
            'q90':float(np.quantile(a,.9))}


def clean_checks():
    rows=[]
    for ci,c in enumerate((0.,.01,.2)):
        for ai,amplitude in enumerate((.03,.3,1.2)):
            for i in range(12):
                seed=SEED+10000*ci+1000*ai+i
                u=np.random.default_rng(seed).normal(0,amplitude,4096)
                x,_,_=trajectory((RHO,LAM,c,BETA),u)
                coef,s=regression(x,u)
                target=np.array([RHO+LAM,-RHO*LAM,-c,LAM*c,BETA])
                error=float(np.max(np.abs(coef-target)))
                assert error<1e-6
                ratio=float(-coef[3]/coef[2]) if c else None
                rows.append({'seed':seed,'c':c,'command_sd':amplitude,
                    'coefficients':coef.tolist(),'singular_values':s.tolist(),
                    'coefficient_max_error':error,'lambda_ratio':ratio})
    return rows


def noisy_checks():
    out=[]
    for ci,c in enumerate((0.,.02,.2)):
        for ai,amplitude in enumerate((.08,.7)):
            rows=[]
            for i in range(12):
                seed=SEED+100000+10000*ci+1000*ai+i
                rng=np.random.default_rng(seed)
                u=rng.normal(0,amplitude,512)
                true_theta=np.array([RHO,LAM,c,BETA])
                x,_,_=trajectory(true_theta,u)
                o=x+rng.normal(0,.2,len(x))
                coef,_=regression(o,u)
                theta,starts=fit_output_error(o,u)
                sr_theta,sr_starts=fit_structured_regression(o,u)
                evaluations={}
                for label,test_sd in (('matched',amplitude),('amplitude_shift',.9)):
                    test_u=rng.normal(0,test_sd,256)
                    true_x,true_v,_=trajectory(true_theta,test_u)
                    oe_x,oe_v,_=trajectory(theta,test_u)
                    sr_x,sr_v,_=trajectory(sr_theta,test_u)
                    raw_v=observable_forecast(coef,test_u)
                    raw_x=np.cumsum(raw_v)
                    evaluations[label]={
                        'OE_velocity_rmse':float(np.sqrt(np.mean((oe_v[1:]-true_v[1:])**2))),
                        'OLS_velocity_rmse':float(np.sqrt(np.mean((raw_v[1:]-true_v[1:])**2))),
                        'OE_position_rmse':float(np.sqrt(np.mean((oe_x[1:]-true_x[1:])**2))),
                        'OLS_position_rmse':float(np.sqrt(np.mean((raw_x[1:]-true_x[1:])**2)))}
                    evaluations[label].update(
                        structured_velocity_rmse=float(np.sqrt(np.mean((sr_v[1:]-true_v[1:])**2))),
                        structured_position_rmse=float(np.sqrt(np.mean((sr_x[1:]-true_x[1:])**2))))
                rows.append({'seed':seed,'OLS_coefficients':coef.tolist(),
                    'OLS_lambda_ratio':float(-coef[3]/coef[2]),
                    'OE_theta':theta.tolist(),'all_optimization_starts':starts,
                    'structured_theta':sr_theta.tolist(),'structured_optimization_starts':sr_starts,
                    'OE_abs_lambda_error':float(abs(theta[1]-LAM)),
                    'OE_parameter_L2_error':float(np.linalg.norm(theta-true_theta)),
                    'evaluations':evaluations})
            summaries={}
            for test in ('matched','amplitude_shift'):
                summaries[test]={metric:stat([r['evaluations'][test][metric] for r in rows])
                    for metric in rows[0]['evaluations'][test]}
            out.append({'c':c,'command_sd':amplitude,'runs':rows,
                'OE_abs_lambda_error':stat([r['OE_abs_lambda_error'] for r in rows]),
                'OLS_lambda_ratio':stat([r['OLS_lambda_ratio'] for r in rows]),
                'evaluation_summary':summaries,
                'all_starts_converged':sum(s['success'] for r in rows for s in r['all_optimization_starts']),
                'total_starts':len(rows)*len(STARTS)})
    return out


def derivative_check():
    u=np.random.default_rng(SEED-1).normal(0,.7,80)
    theta=np.array([RHO,LAM,.2,BETA])
    _,_,jac=trajectory(theta,u,True)
    numeric=np.empty_like(jac)
    for k in range(4):
        delta=np.zeros(4);delta[k]=1e-6
        numeric[:,k]=(trajectory(theta+delta,u)[0]-trajectory(theta-delta,u)[0])/2e-6
    error=float(np.max(np.abs(jac-numeric)))
    assert error<1e-6
    return error


def main():
    with threadpool_limits(limits=1):
        out={'meta':{'version':9,'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'seed':SEED,'timing':'lagged motor',
            'rho':RHO,'lambda':LAM,'beta':BETA,'process_sd':0.,'sensor_sd':.2,
            'training_rows':512,'evaluation_steps':256,'replicates':12,
            'known_initial_state':[0.,0.,0.],
            'OE_parameter_order':['rho','lambda','c','beta'],
            'starts':STARTS,'lower':LOWER,'upper':UPPER,
            'warning':'restricted supplied family; no autonomous structure discovery or general robotics claim'},
            'jacobian_max_error':derivative_check(),'clean':clean_checks(),'noisy':noisy_checks()}
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':main()
