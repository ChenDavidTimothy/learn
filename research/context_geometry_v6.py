#!/usr/bin/env python3
"""V6: limits of scalar mean/amplitude probes; temporal/covariance repairs.

All trajectories observed through noisy position. Q is supplied for the two
parametric eta estimators; no claim of autonomous nonlinear operator discovery.
"""
from __future__ import annotations
import json
import platform
import numpy as np
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

SEED=20261020
RHO,BETA,ETA=.45,.9,.7
Q=np.array([[1.,0.,-1.],[0.,-1.,1.],[-1.,1.,0.]])/np.sqrt(12)


def simulate(seed,n=8190,sd=.8,mean=0.,phase_height=0.,corr=0.,sensor_sd=.5,
             eta=ETA,burn=400,lags=8):
    rng=np.random.default_rng(seed)
    total=n+burn+lags
    z=lfilter([np.sqrt(1-corr*corr)],[1.,-corr],rng.normal(size=total))
    phase=np.arange(total)%3
    means=mean+np.array([phase_height,-phase_height,0.])[phase]
    a=means+sd*z
    histories=np.c_[a,np.r_[0.,a[:-1]],np.r_[0.,0.,a[:-2]]]
    nonlinear=np.einsum('ij,jk,ik->i',histories,Q,histories)
    w=rng.normal(0,.08,total)
    e=rng.normal(0,sensor_sd,total+1)
    p=lfilter([1.],[1.,-RHO],BETA*a+eta*nonlinear+w)
    truth=lfilter([1.],[1.,-RHO],BETA*a+eta*nonlinear)
    obs=np.r_[0.,np.cumsum(p)]+e
    t=np.arange(burn+lags,total)
    X=np.column_stack([a[t-j]-means[t-j] for j in range(lags)])
    return {'X':X,'y':np.diff(obs)[t],'truth':truth[t], 'phase':phase[t],
            'a':a[t],'current_mean':means[t],'phase_height':phase_height,
            'mean':mean,'sd':sd,'sample_energy':float(np.mean(a[t]**2))}


def first_order(data):
    # Phase-specific intercepts remove deterministic periodic output means.
    X,y,phase=data['X'],data['y'],data['phase']
    hs=[]
    for k in range(3):
        rows=phase==k
        co=np.linalg.lstsq(np.c_[np.ones(rows.sum()),X[rows]],y[rows],rcond=None)[0][1:]
        hs.append(co)
    return np.array(hs)


def mixed_eta(data):
    # E[Y(z_t²-sd²)]=2 eta Q00 sd^4, since past output has no a_t.
    z=data['X'][:,0]
    return float(np.mean((data['y']-data['y'].mean())*(z*z-data['sd']**2)) /
                 (2*Q[0,0]*data['sd']**4))


def phase_eta(h,height):
    return float((h[0,0]-(h[1,0]+h[2,0])/2)/(np.sqrt(3)*height))


def volterra_eta(data):
    X,y=data['X'],data['y']
    i,j=np.triu_indices(X.shape[1])
    quadratic=X[:,i]*X[:,j]
    if data['phase_height']:
        linear=np.column_stack([np.c_[np.ones(len(y)),X]*(data['phase']==k)[:,None]
                                for k in range(3)])
    else:
        linear=np.c_[np.ones(len(y)),X]
    co=np.linalg.lstsq(np.c_[linear,quadratic],y,rcond=None)[0][linear.shape[1]:]
    # Current times lag-two coefficient = 2 eta Q02, no delayed contamination.
    index=np.flatnonzero((i==0)&(j==2))[0]
    return float(co[index]/(2*Q[0,2]))


def structured_eta(data):
    X,y=data['X'],data['y']
    # Identical columns/parameter ties on both designs. Raw Q windows preserve
    # known relation between phase gradients and quadratic coefficient eta.
    phase_means=np.array([data['phase_height'],-data['phase_height'],0.])
    raw=X+data['mean']+phase_means[(data['phase'][:,None]-np.arange(X.shape[1]))%3]
    linear=np.c_[np.eye(3)[data['phase']],X]
    features=np.column_stack([np.einsum('ij,jk,ik->i',raw[:,k:k+3],Q,raw[:,k:k+3])
                               for k in range(X.shape[1]-2)])
    # Equal Q prior as raw/phase inversions, but efficient joint nuisance fit.
    co=np.linalg.lstsq(np.c_[linear,features],y,rcond=None)[0]
    return float(co[linear.shape[1]])


def stats(values):
    a=np.asarray(values)
    return {'mean':float(a.mean()),'mc_se':float(a.std(ddof=1)/np.sqrt(len(a))),
            'rmse_from_eta':float(np.sqrt(np.mean((a-ETA)**2))),
            'median_absolute_error':float(np.median(np.abs(a-ETA)))}


def blindness():
    out={}
    # Fixed before execution; these are theorem checks, not classification tests.
    for j,(mean,sd) in enumerate([(-.8,.35),(0.,.35),(.8,.35),
                                  (-.8,1.25),(0.,1.25),(.8,1.25)]):
        rows=[]
        for i in range(32):
            seed=SEED+800000+j*1000+i
            linear=simulate(seed,n=4095,mean=mean,sd=sd,eta=0.)
            nonlinear=simulate(seed,n=4095,mean=mean,sd=sd)
            h0=first_order(linear).mean(axis=0)
            h1=first_order(nonlinear).mean(axis=0)
            rows.append({'seed':seed,'first_order_delta':(h1-h0).tolist(),
                         'mean_output_delta':float(np.mean(nonlinear['y']-linear['y']))})
        deltas=np.array([r['first_order_delta'] for r in rows])
        out[f'mean{mean}_sd{sd}']={'mean':mean,'sd':sd,'runs':rows,
            'mean_paired_first_order_delta':deltas.mean(axis=0).tolist(),
            'delta_mc_se':(deltas.std(axis=0,ddof=1)/np.sqrt(len(rows))).tolist(),
            'mean_paired_output_delta':float(np.mean([r['mean_output_delta'] for r in rows]))}
    return out


def equal_energy_repairs():
    out={}
    energy=.8**2
    sigma=np.sqrt(energy/2)
    height=np.sqrt(3*energy/4)
    for j,sensor_sd in enumerate((.1,.5,1.5)):
        rows=[]
        for i in range(64):
            seed=SEED+900000+j*1000+i
            centered=simulate(seed,sd=np.sqrt(energy),sensor_sd=sensor_sd)
            scalar=simulate(seed,sd=sigma,mean=np.sqrt(energy/2),sensor_sd=sensor_sd)
            phased=simulate(seed,sd=sigma,phase_height=height,sensor_sd=sensor_sd)
            hs=first_order(scalar)
            hp=first_order(phased)
            rows.append({'seed':seed,'centered_eta2':mixed_eta(centered),
                'phased_eta1':phase_eta(hp,height),
                'centered_volterra2':volterra_eta(centered),
                'phased_volterra2':volterra_eta(phased),
                'centered_structured':structured_eta(centered),
                'phased_structured':structured_eta(phased),
                'scalar_mean_h0':hs[:,0].tolist(),
                'phased_h0':hp[:,0].tolist(),
                'sample_energies':[d['sample_energy'] for d in (centered,scalar,phased)]})
        out[str(sensor_sd)]={'sensor_sd':sensor_sd,'n':8190,'expected_energy':energy,
            'phased_height':height,'shifted_innovation_sd':sigma,'runs':rows,
            'centered_eta2':stats([r['centered_eta2'] for r in rows]),
            'phased_eta1':stats([r['phased_eta1'] for r in rows]),
            'centered_volterra2':stats([r['centered_volterra2'] for r in rows]),
            'phased_volterra2':stats([r['phased_volterra2'] for r in rows]),
            'centered_structured':stats([r['centered_structured'] for r in rows]),
            'phased_structured':stats([r['phased_structured'] for r in rows]),
            'mean_scalar_h0':np.mean([r['scalar_mean_h0'] for r in rows],axis=0).tolist(),
            'mean_phased_h0':np.mean([r['phased_h0'] for r in rows],axis=0).tolist(),
            'mean_sample_energies':np.mean([r['sample_energies'] for r in rows],axis=0).tolist()}
    return out


def covariance_repair():
    out={}
    for j,corr in enumerate((0.,.5,.75)):
        values=[]
        for i in range(64):
            seed=SEED+1000000+j*1000+i
            a=simulate(seed,corr=corr,eta=0.)
            b=simulate(seed,corr=corr)
            values.append({'seed':seed,'paired_mean_difference':float(np.mean(b['y']-a['y']))})
        prediction=2*ETA*.8**2/np.sqrt(12)*(corr-corr**2)/(1-RHO)
        vals=np.array([v['paired_mean_difference'] for v in values])
        out[str(corr)]={'corr':corr,'theoretical_mean_difference':float(prediction),
            'mean_difference':float(vals.mean()),'mc_se':float(vals.std(ddof=1)/8),'runs':values}
    return out


def algebra():
    assert np.max(np.abs(Q@np.ones(3)))<1e-14
    assert abs(np.trace(Q))<1e-14
    assert np.linalg.matrix_rank(Q)==2
    # Q's undetected geometry for any subspace S: QS=0 and trace(Q)=0.
    dimensions={str(d):{str(k):max(0,(d-k)*(d-k+1)//2-1) for k in range(d+1)}
                for d in (3,6,12)}
    sigma=.8
    return {'Q':Q.tolist(),'Q_ones':(Q@np.ones(3)).tolist(),'trace_Q':float(np.trace(Q)),
            'traceless_quadratic_nullspace_dimension_by_d_k':dimensions,
            'iid_first_order_noiseless_floor':float(abs(ETA)*sigma**2*np.sqrt((1-RHO/3)/(1-RHO**2))),
            'F_variance_at_sd_0_8':sigma**4,'F_lag1_covariance':-sigma**4/6,
            'phased_h0_theory':(BETA+ETA*np.array([2.,-1.,-1.])*np.sqrt(3*.8**2/4)/np.sqrt(3)).tolist()}


def main():
    with threadpool_limits(limits=1):
        r={'meta':{'version':6,'seed':SEED,'python':platform.python_version(),'numpy':np.__version__,
            'rho':RHO,'beta':BETA,'eta':ETA,'process_sd':.08,'default_sensor_sd':.5,
            'disclosure':'fixed Q dictionary and family supplied to eta estimators; matched expected energy, not exact realized energy',
            'claim':'structural information limitation and constructive repairs; no novel estimator or broad superiority'},
            'algebra':algebra(),'common_mean_amplitude':blindness(),
            'equal_energy':equal_energy_repairs(),'covariance_change':covariance_repair()}
    print(json.dumps(r,indent=2,allow_nan=False))


if __name__=='__main__':main()
