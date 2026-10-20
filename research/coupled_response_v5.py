#!/usr/bin/env python3
"""First-order footprint blindness and second-order identification, v5.

Position-only triangular bilinear system. Fixed protocol, all per-seed results.
Models receive noisy position differences and command histories, no latent state.
"""
from __future__ import annotations
import json
import platform
import numpy as np
import scipy
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

SEED = 20261020
R, RHO, BETA, ETA = .5, .45, .9, .7
LAGS = 8


def simulate(seed, n, amplitude=.8, eta=ETA, means=(0., 0.), corr=0.,
             sensor_sd=.5, process_sd=.08, burn=400):
    rng = np.random.default_rng(seed)
    total = n + burn + LAGS
    z = rng.normal(size=(total, 2))
    commands = np.zeros_like(z)
    for t in range(total):
        commands[t] = corr * (commands[t-1] if t else 0) + np.sqrt(1-corr*corr)*z[t]
    commands = amplitude * commands + np.asarray(means)
    u, b = commands.T
    q = np.zeros(total + 1)
    p = np.zeros(total + 1)
    noiseless_p = np.zeros(total + 1)
    w = rng.normal(0, process_sd, total)
    e = rng.normal(0, sensor_sd, total+1)
    for t in range(total):
        q[t+1] = R*q[t] + b[t]
        p[t+1] = RHO*p[t] + BETA*u[t] + eta*q[t]*u[t] + w[t]
        noiseless_p[t+1] = RHO*noiseless_p[t] + BETA*u[t] + eta*q[t]*u[t]
    obs = np.r_[0., np.cumsum(p[1:])] + e
    d = np.diff(obs)
    t = np.arange(burn + LAGS, total)
    X = np.column_stack([commands[t-j, c] for c in range(2) for j in range(LAGS)])
    # No hidden q or p goes to model fitting; truth is separate for evaluation.
    return X, d[t], noiseless_p[t+1]


def raw_moments(X, y, amplitude):
    u0, u1, b1, b2 = X[:, 0], X[:, 1], X[:, LAGS+1], X[:, LAGS+2]
    yc = y - np.mean(y)
    k01 = float(np.mean(yc*u0*b1) / amplitude**4)
    k02 = float(np.mean(yc*u0*b2) / amplitude**4)
    k12 = float(np.mean(yc*u1*b2) / amplitude**4)
    h0 = float(np.mean(yc*u0) / amplitude**2)
    return {'h0': h0, 'k01': k01, 'k02': k02, 'k12': k12,
            'identified': [h0, k01, k02/k01, k12/k01]}


def summarize(values):
    a = np.asarray(values, float)
    return {'mean': float(a.mean()), 'mc_se': float(a.std(ddof=1)/np.sqrt(len(a))),
            'median': float(np.median(a)), 'q10': float(np.quantile(a, .1)),
            'q90': float(np.quantile(a, .9))}


def moments_experiment():
    out = {}
    for j, amplitude in enumerate((.35, 1.25)):
        runs = []
        for i in range(64):
            seed = SEED+500000+1000*j+i
            fits = {}
            for eta in (0., ETA):
                X, y, _ = simulate(seed, 8192, amplitude=amplitude, eta=eta)
                co = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0][1:]
                fits[str(eta)] = {'first_order': co.tolist(), 'moments': raw_moments(X,y,amplitude)}
            delta = np.asarray(fits[str(ETA)]['first_order'])-np.asarray(fits['0.0']['first_order'])
            pars = np.asarray(fits[str(ETA)]['moments']['identified'])
            runs.append({'seed': seed, 'fits': fits,
                         'first_order_delta': delta.tolist(),
                         'parameter_l2_error': float(np.linalg.norm(pars-[BETA,ETA,R,RHO]))})
        out[str(amplitude)] = {'amplitude': amplitude, 'steps': 8192, 'runs': runs,
            'mean_paired_first_order_delta': np.mean([r['first_order_delta'] for r in runs],axis=0).tolist(),
            'paired_delta_mc_se': (np.std([r['first_order_delta'] for r in runs],axis=0,ddof=1)/8).tolist(),
            'parameter_l2_error': summarize([r['parameter_l2_error'] for r in runs]),
            'k01': summarize([r['fits'][str(ETA)]['moments']['k01'] for r in runs]),
            'k02': summarize([r['fits'][str(ETA)]['moments']['k02'] for r in runs]),
            'k12': summarize([r['fits'][str(ETA)]['moments']['k12'] for r in runs])}
    return out


def polynomial(X, degree):
    if degree == 1:
        return X
    i,j=np.triu_indices(X.shape[1])
    return np.c_[X, X[:,i]*X[:,j]]


def fit_ridge(X, y, V, vy):
    # Center all predictors and targets using training data ONLY.
    center, offset = X.mean(axis=0), float(y.mean())
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.
    Z = (X-center)/scale
    ZV = (V-center)/scale
    ev,Q = eigh(Z.T@Z)
    rhs = Q.T@(Z.T@(y-offset))
    best = None
    for lam in (.001,.1,10.,100.):
        coef = Q@(rhs/(np.maximum(ev,0)+lam))
        mse = float(np.mean((ZV@coef+offset-vy)**2))
        if best is None or mse < best[0]:
            best = (mse, lam, coef)
    _,lam,coef = best
    return lambda A: ((A-center)/scale)@coef+offset, {'lambda':lam}


def kernel(A,B,gamma):
    dist = np.maximum(np.sum(A*A,axis=1)[:,None]+np.sum(B*B,axis=1)[None,:]-2*A@B.T,0)
    return np.exp(-gamma*dist)


def fit_krr(X,y,V,vy):
    center, scale = X.mean(axis=0), X.std(axis=0)
    Z,ZV = (X-center)/scale, (V-center)/scale
    offset=float(y.mean())
    best = None
    for factor in (.25,1.,4.):
        gamma=factor/X.shape[1]
        K=kernel(Z,Z,gamma)
        ev,Q=eigh(K)
        rhs=Q.T@(y-offset)
        KV=kernel(ZV,Z,gamma)
        for lam in (.001,.1,10.,100.):
            alpha=Q@(rhs/(np.maximum(ev,0)+lam))
            mse=float(np.mean((KV@alpha+offset-vy)**2))
            if best is None or mse<best[0]: best=(mse,gamma,lam,alpha)
    _,gamma,lam,alpha=best
    return lambda A: kernel((A-center)/scale,Z,gamma)@alpha+offset, {'gamma':gamma,'lambda':lam}


def prediction_experiment():
    # All methods use the exact same data/splits; grids chosen before running.
    conditions = {'iid': {}, 'mean_shift': {'means':(.4,.5)},
                  'correlated': {'corr':.75}, 'amplitude_shift': {'amplitude':1.25}}
    runs=[]
    for i in range(32):
        base=SEED+600000+100*i
        X,y,_=simulate(base,1024)
        V,vy,_=simulate(base+1,512)
        tests={k: simulate(base+10+j,2048,**kw) for j,(k,kw) in enumerate(conditions.items())}
        record={'seed_base':base,'models':{}}
        for name,degree in [('FIR',1),('Volterra2',2),('RBF_KRR',None)]:
            if degree:
                predict,config=fit_ridge(polynomial(X,degree),y,polynomial(V,degree),vy)
            else:
                predict,config=fit_krr(X,y,V,vy)
            scores={}
            for cond,(T,ty,truth) in tests.items():
                pred=predict(polynomial(T,degree) if degree else T)
                scores[cond]={'noiseless_rmse':float(np.sqrt(np.mean((pred-truth)**2))),
                    'observed_rmse':float(np.sqrt(np.mean((pred-ty)**2))),
                    'mean_prediction_error':float(np.mean(pred-truth))}
            record['models'][name]={'config':config,'scores':scores}
        runs.append(record)
    summary={}
    for name in ('FIR','Volterra2','RBF_KRR'):
        summary[name]={cond:{metric:summarize([r['models'][name]['scores'][cond][metric] for r in runs])
            for metric in ('noiseless_rmse','observed_rmse','mean_prediction_error')}
            for cond in conditions}
    return {'train':1024,'validation':512,'test_per_condition':2048,
            'runs':runs,'summary':summary}


def algebra_checks():
    # Check derived finite-history kernel against noiseless simulation.
    X,p,_=simulate(SEED+700000,256,sensor_sd=0,process_sd=0)
    # Closed-form truncated quadratic prediction, independent of model fitting.
    u,b=X[:,:LAGS],X[:,LAGS:]
    pred=np.zeros(len(X))
    for i in range(LAGS):
        pred += BETA*RHO**i*u[:,i]
        for j in range(i+1,LAGS):
            pred += ETA*RHO**i*R**(j-i-1)*u[:,i]*b[:,j]
    error=float(np.sqrt(np.mean((pred-p)**2)))
    assert error < .03 # finite-window truncation, not an exact infinite-history assertion
    return {'truncated_kernel_rmse_seed':error,
            'theoretical_iid_first_order_floor':float(abs(ETA)*.8**2/np.sqrt((1-R*R)*(1-RHO*RHO))),
            'theoretical_mean_shift_FIR_bias':-ETA*.4*.5/((1-R)*(1-RHO)),
            'theoretical_k01_k02_k12':[ETA,ETA*R,ETA*RHO]}


def main():
    with threadpool_limits(limits=1):
        result={'meta':{'version':5,'seed':SEED,'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,'r':R,'rho':RHO,'beta':BETA,'eta':ETA,
            'lags':LAGS,'sensor_sd':.5,'process_sd':.08,'burn':400,
            'same_model_access':'two logged command histories; measured position differences as targets',
            'builtins':'fixed 8-lag history; quadratic dictionary for Volterra2; RBF kernel for KRR',
            'claim':'first-order representation limitation, not superiority over nonlinear learning methods'},
            'algebra':algebra_checks(),'moments':moments_experiment(),
            'prediction':prediction_experiment()}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
