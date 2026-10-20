#!/usr/bin/env python3
"""V7 conditional-intercept instruments and randomized zero-command atoms.

Identifies ONE constant passive coefficient, not a complete nonlinear model.
No latent state/gain or actuator formula is supplied to estimators.
"""
from __future__ import annotations
import json
import math
import platform
import numpy as np
from threadpoolctl import threadpool_limits

SEED=20261020
RHO=.65
ATOM_PROB=.1


def hermite_weights(pi,z,sd,degree):
    a=-pi/sd
    Z=z/sd
    ha0,hz0=np.ones_like(Z),np.ones_like(Z)
    weight=ha0*hz0
    if degree==0:return weight
    ha1,hz1=a,Z
    weight=weight+ha1*hz1
    factorial=1
    for k in range(2,degree+1):
        ha2=a*ha1-(k-1)*ha0
        hz2=Z*hz1-(k-1)*hz0
        factorial*=k
        weight=weight+ha2*hz2/factorial
        ha0,ha1=ha1,ha2
        hz0,hz1=hz1,hz2
    return weight


def simulate(seed,world,n=65536,sd=.7,sensor_sd=.5,policy='gaussian',burn=600):
    rng=np.random.default_rng(seed)
    total=n+burn+4
    normal=rng.normal(size=total)
    coins=rng.random(total)
    b=rng.normal(0,.6,total)
    process=rng.normal(0,.06,total)
    qnoise=rng.normal(0,.1,total)
    e=rng.normal(0,sensor_sd,total+1)
    p=np.zeros(total+1); q=np.zeros(total+1)
    d=np.zeros(total+1); pi=np.zeros(total); u=np.zeros(total); z=np.zeros(total)
    atoms=np.zeros(total,bool)
    d[0]=0.
    for t in range(total):
        pi[t]=-.5*math.tanh(d[t])
        if policy=='atom':
            probability=ATOM_PROB
            variance=sd*sd/(1-probability)-probability*pi[t]*pi[t]/(1-probability)**2
            assert variance>0
            atoms[t]=coins[t]<probability
            u[t]=0. if atoms[t] else pi[t]/(1-probability)+math.sqrt(variance)*normal[t]
            z[t]=u[t]-pi[t]  # conditionally centered with constant variance sd²
        else:
            z[t]=sd*normal[t]
            u[t]=pi[t]+z[t]
        gain=1.2+.7*math.tanh(q[t])
        if world=='affine':force=gain*u[t]
        elif world=='quadratic':force=gain*u[t]+(.2+.15*math.tanh(q[t]))*u[t]**2
        elif world=='saturating':force=gain*math.tanh(u[t])
        else:raise ValueError(world)
        p[t+1]=RHO*p[t]+force+process[t]
        # Semi-implicit coupling makes G_t depend on the preceding probe.
        q[t+1]=.6*q[t]+.3*math.tanh(p[t+1]) + b[t]+qnoise[t]
        d[t+1]=p[t+1]+e[t+1]-e[t]
    t=np.arange(burn+4,total)
    return {'D':d[t], 'Y':d[t+1], 'u':u[t], 'z':z[t], 'zprev':z[t-1],
            'pi':pi[t], 'atoms':atoms[t], 'bprev':b[t-1], 'bprev2':b[t-2],
            'sample_energy':float(np.mean(u[t]**2)),
            'max_abs_velocity':float(np.max(np.abs(p)))}


def estimates(data,sd,n):
    sl=slice(0,n)
    D,Y,u,z,zprev,pi=[data[k][sl] for k in ('D','Y','u','z','zprev','pi')]
    X=np.c_[D,u]
    Z=np.c_[zprev,z]
    ols=np.linalg.lstsq(X,Y,rcond=None)[0]
    iv=np.linalg.lstsq(Z.T@X,Z.T@Y,rcond=None)[0]
    den=float(np.mean(zprev*D))
    out={'OLS':float(ols[0]), 'constant_gain_IV':float(iv[0]),
         'denominator':den,'sample_energy':float(np.mean(u*u))}
    # Flexible conventional series-IV using observed history (no q or G).
    H=np.c_[D,data['bprev'][sl],data['bprev2'][sl]]
    H=(H-H.mean(axis=0))/H.std(axis=0)
    i,j=np.triu_indices(3)
    phi=np.c_[np.ones(n),H,H[:,i]*H[:,j]]
    XX=np.c_[D,u[:,None]*phi]
    ZZ=np.c_[zprev,z[:,None]*phi]
    th=np.linalg.lstsq(ZZ.T@XX,ZZ.T@Y,rcond=None)[0]
    out['series_IV']=float(th[0])
    # Stronger series-IV alternatives cover polynomial nonlinear actuation,
    # using exactly the same Gaussian/Hermite vocabulary as the weights.
    for degree in (2,4):
        control=np.column_stack([(u/sd)**j for j in range(1,degree+1)])
        zz=z/sd
        h0,h1=np.ones(n),zz
        tags=[h1]
        for j in range(2,degree+1):
            h2=zz*h1-(j-1)*h0
            tags.append(h2)
            h0,h1=h1,h2
        XX=np.c_[D,np.column_stack([control[:,j,None]*phi for j in range(degree)])]
        ZZ=np.c_[zprev,np.column_stack([tag[:,None]*phi for tag in tags])]
        th=np.linalg.lstsq(ZZ.T@XX,ZZ.T@Y,rcond=None)[0]
        out[f'action_series_IV{degree}']=float(th[0])
    for m in (1,2,4):
        weights=hermite_weights(pi,z,sd,m)
        out[f'weight_m{m}']=float(np.mean(zprev*weights*Y)/den)
        out[f'weight_m{m}_squared_mean']=float(np.mean(weights*weights))
    atomweight=data['atoms'][sl]/ATOM_PROB
    out['atom']=float(np.mean(zprev*atomweight*Y)/den)
    return out


def stat(values):
    a=np.asarray(values)
    return {'mean':float(a.mean()), 'mc_se':float(a.std(ddof=1)/np.sqrt(len(a))),
            'bias_from_rho':float(a.mean()-RHO),'rmse_from_rho':float(np.sqrt(np.mean((a-RHO)**2))),
            'q10':float(np.quantile(a,.1)),'q90':float(np.quantile(a,.9))}


def moment_checks():
    # Gaussian quadrature checks all polynomial reproduction constraints.
    from numpy.polynomial.hermite_e import hermegauss
    Z,prob=hermegauss(32)
    prob=prob/np.sqrt(2*np.pi)
    checks=[]
    for pi in (-.5,0.,.4):
        for sd in (.2,.7):
            for m in (1,2,4):
                w=hermite_weights(np.full_like(Z,pi),sd*Z,sd,m)
                errors=[float(np.sum(prob*w*(pi+sd*Z)**k)-(1. if k==0 else 0.)) for k in range(m+1)]
                assert max(map(abs,errors))<1e-10
                checks.append({'pi':pi,'sd':sd,'degree':m,'moment_errors':errors,
                               'weight_l2_squared':float(np.sum(prob*w*w))})
    # Mixture preserves conditional mean and second moment, not full action law.
    p=ATOM_PROB
    min_variance=.2**2/(1-p)-p*.5**2/(1-p)**2
    assert min_variance>0
    return {'gaussian_quadrature':checks,'atom_min_nonzero_component_variance':min_variance}


def main():
    cases=[('affine',.7,.5),('quadratic',.7,.5),('saturating',.7,.5),
           ('affine',.2,1.5)]
    result={}
    with threadpool_limits(limits=1):
        for ci,(world,sd,sensor_sd) in enumerate(cases):
            rows=[]
            for i in range(48):
                seed=SEED+1200000+1000*ci+i
                a=simulate(seed,world,sd=sd,sensor_sd=sensor_sd)
                b=simulate(seed,world,sd=sd,sensor_sd=sensor_sd,policy='atom')
                row={'seed':seed,'gaussian':{},'atom_policy':{}}
                for n in (8192,65536):
                    ga=estimates(a,sd,n)
                    at=estimates(b,sd,n)
                    # Hermite weights are NOT valid for non-Gaussian atom policy.
                    row['gaussian'][str(n)]={k:v for k,v in ga.items() if k!='atom'}
                    row['atom_policy'][str(n)]={k:v for k,v in at.items()
                        if not k.startswith(('weight_','action_series_'))}
                rows.append(row)
            summary={}
            for n in ('8192','65536'):
                summary[n]={k:stat([row['gaussian'][n][k] for row in rows])
                    for k in ('OLS','constant_gain_IV','series_IV','action_series_IV2',
                              'action_series_IV4','weight_m1','weight_m2','weight_m4')}
                summary[n]['atom']=stat([row['atom_policy'][n]['atom'] for row in rows])
                summary[n]['mean_energy_gaussian_atom']=[float(np.mean([row[p][n]['sample_energy'] for row in rows]))
                                                        for p in ('gaussian','atom_policy')]
            result[f'{world}_probe{sd}_sensor{sensor_sd}']={
                'world':world,'probe_sd':sd,'sensor_sd':sensor_sd,'runs':rows,'summary':summary}
    print(json.dumps({'meta':{'version':7,'seed':SEED,'python':platform.python_version(),
        'numpy':np.__version__,'rho':RHO,'replicates':48,'burn':600,'process_sd':.06,
        'atom_probability':ATOM_PROB,'controller':'pi=-.5tanh(measured previous displacement)',
        'hidden_gain':'1.2+.7tanh(q); q is coupled to true p and has unobserved process noise',
        'estimators_observe':'D,Y,issued u,probe z,controller pi,secondary b only; no q,p,G or actuator class',
        'comparison':'one passive parameter only; no broad nonlinear-method superiority or control claim'},
        'moment_checks':moment_checks(),'experiments':result},indent=2,allow_nan=False))


if __name__=='__main__':main()
