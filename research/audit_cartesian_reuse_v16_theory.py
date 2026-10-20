#!/usr/bin/env python3
"""Independent Torch likelihood gradients and direct-component terminal-QP audit.

Canonical gaussian_objective is imported only as the implementation under test.
The independent likelihood uses Torch MultivariateNormal; terminal loss sums actual
sample/component trajectories rather than the implementation's moment expansion.
"""
import json
from pathlib import Path
import numpy as np
import torch
from cartesian_reuse_v16_refinement import gaussian_objective

torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parent
R=(.0002/.1**2)**2


def independent_H(exposures):
    H=np.zeros((len(exposures),4,4))
    for n in range(len(exposures)):
        for t in range(2):
            for o in range(2):H[n,2*t+o,2*o:2*o+2]=exposures[n,t]
    return H


def main():
    d=json.loads((ROOT/'cartesian_reuse_v16_refinement_results.json').read_text())
    data=np.load(ROOT/'cartesian_reuse_v16_pilot_data.npz')
    X=data['training_exposures'];Y=(data['training_observed'][:,1:]-np.array([np.sqrt(2),0]))/.1**2
    H=independent_H(X);ht=torch.tensor(H);yt=torch.tensor(Y.reshape(-1,4));rng=np.random.default_rng(86516630)
    checks=[]
    for name,mean,cov in [('random',rng.normal(size=4),np.eye(4)*.2)]+[
        (g['start'],np.array(g['mean']).ravel(),np.array(g['cov'])) for g in d['gaussian_candidates'] if g.get('fit')=='grouped_ML']:
        eigen,V=np.linalg.eigh(cov);F=V*np.sqrt(np.maximum(eigen,1e-14));params=np.r_[mean,F.ravel()]
        x=torch.tensor(params,requires_grad=True);f=x[4:].reshape(4,4)
        C=ht@(f@f.T)@ht.transpose(1,2)+R*torch.eye(4)
        loss=-torch.distributions.MultivariateNormal(ht@x[:4],covariance_matrix=C).log_prob(yt).mean()
        grad=torch.autograd.grad(loss,x)[0].detach().numpy()
        value,canonical=gaussian_objective(params,H,Y)
        error=float(np.max(np.abs(grad-canonical)));value_error=float(abs(value-loss.item()))
        assert error<1e-7 and value_error<1e-10,(name,error,value_error)
        # Direct covariance-space derivative diagnoses PSD-boundary behavior.
        residual=Y.reshape(-1,4)-H@mean;Cs=H@cov@H.transpose(0,2,1)+R*np.eye(4)
        gc=np.zeros((4,4))
        for hi,ci,ri in zip(H,Cs,residual):
            inv=np.linalg.solve(ci,np.eye(4));alpha=inv@ri
            gc+=.5*hi.T@(inv-np.outer(alpha,alpha))@hi/len(Y)
        checks.append(dict(name=name,objective=value,autodiff_max_gradient_error=error,value_error=value_error,
            covariance_gradient_eigenvalues=np.linalg.eigvalsh(gc).tolist()))
    controls=[];maxcost=maxpg=maxposterior=0.
    for method,rows in d['controls'].items():
        prior=d['priors'][method.split('_')[0]]
        for row in [r for r in rows if r['repeat']==0]:
            u=np.array(row['commands']);u0=u[0];y=(np.array(row['first_observation'])-np.array([np.sqrt(2),0]))/.1**2
            h=np.zeros((2,4));h[0,:2]=u0/2;h[1,2:]=u0/2
            if prior['kind']=='mixture':
                Bs=np.array(prior['B']);w=np.array(prior['weights']);r=y-Bs.reshape(-1,4)@h.T
                logs=np.log(w)-np.sum(r*r,axis=1)/(2*R);w=np.exp(logs-logs.max());w/=w.sum()
                maxposterior=max(maxposterior,float(np.max(abs(w-row['posterior']))))
            else:
                cov=np.array(prior['cov']);m=np.array(prior['mean']).ravel()
                K=cov@h.T@np.linalg.solve(h@cov@h.T+R*np.eye(2),np.eye(2))
                postmean=m+K@(y-h@m);postcov=cov-K@h@cov
                maxposterior=max(maxposterior,float(np.max(abs(postcov-row['posterior']))))
                e,V=np.linalg.eigh(postcov);directions=(V*np.sqrt(np.maximum(e,0))).T*2
                Bs=np.concatenate([postmean+directions,postmean-directions]).reshape(8,2,2);w=np.ones(8)/8
            b=torch.tensor(Bs);weights=torch.tensor(w);z=torch.tensor(u[1:].ravel(),requires_grad=True)
            allu=torch.cat([torch.tensor(u0)[None],z.reshape(2,2)])
            position=torch.einsum('koi,i->ko',b,2.5*allu[0]+1.5*allu[1]+.5*allu[2])*.1**2
            velocity=torch.einsum('koi,i->ko',b,allu.sum(0))*.1
            goal=torch.tensor(np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1]])[row['goal_index']]*.001)
            losses=((position-goal)/.001).square().sum(1)+.25*(velocity/.01).square().sum(1)
            loss=weights@losses+.01*allu.square().sum()
            gradient=torch.autograd.grad(loss,z)[0].detach().numpy();flat=z.detach().numpy()
            pg=float(np.max(abs(flat-np.clip(flat-gradient,-.5,.5))))
            delta=float(abs(loss.item()-row['model_expected_cost']));maxcost=max(maxcost,delta);maxpg=max(maxpg,pg)
            assert delta<1e-8 and pg<1e-7,(method,row['goal_index'],delta,pg)
            controls.append(dict(method=method,goal=row['goal_index'],branch=row['branch_evaluator_only'],direct_cost_error=delta,projected_gradient=pg))
    old=np.load(ROOT/'cartesian_reuse_v16_pilot_data.npz');new=np.load(ROOT/'cartesian_reuse_v16_refinement_data.npz')
    assert old.files==new.files and all(np.array_equal(old[k],new[k]) for k in old.files)
    print(json.dumps(dict(scope='Independent likelihood autodiff and direct component/sigma-point physical-normalized terminal loss/KKT; not a full artifact reproduction or global ML claim.',
        likelihood_checks=checks,terminal_checks=len(controls),max_terminal_cost_error=maxcost,max_projected_gradient=maxpg,
        max_posterior_difference=maxposterior,exact_raw_data_reuse=True,all_control_checks=controls),indent=2))


if __name__=='__main__':main()
