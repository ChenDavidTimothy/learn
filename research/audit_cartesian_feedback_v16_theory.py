#!/usr/bin/env python3
"""Independent direct sigma-point/Torch joint-belief and control audit.

The implementation under test is imported only to compare its outputs; reference
costs roll every sigma point through s/w/B dynamics without QP moment formulas.
"""
import json,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import cartesian_feedback_v16 as implementation
import cartesian_reuse_v16_refinement as batch

torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parent


def sigma(mean,cov):
    n=len(mean);e,V=np.linalg.eigh(cov);D=(V*np.sqrt(np.maximum(e,0))).T*np.sqrt(n)
    return np.r_[mean+D,mean-D]


def direct_loss(flat,weights,means,covs,goal,steps,spent):
    points=np.concatenate([sigma(m,P) for m,P in zip(means,covs)])
    probs=np.repeat(weights/16,16);state=torch.tensor(points)
    s=state[:,:2];w=state[:,2:4];B=state[:,4:].reshape(-1,2,2)
    commands=flat.reshape(steps,2)
    for u in commands:
        accel=torch.einsum('noi,i->no',B,u)
        s=s+w+.5*accel;w=w+accel
    cost=((s-torch.tensor(goal)/.01)/.1).square().sum(1)+.25*(w/.1).square().sum(1)
    return torch.tensor(probs)@cost+.01*(commands.square().sum()+spent)


def reference_condition(weights,means,covs,y):
    # Condition via the JOINT PRECISION matrix, independently of Kalman gain code.
    rows=[];Cs=[];logs=[]
    C=np.c_[np.eye(2),np.zeros((2,6))]
    for p,m,P in zip(weights,means,covs):
        S=C@P@C.T+.0004*np.eye(2);cross=P@C.T
        joint=np.block([[P,cross],[cross.T,S]])
        J=np.linalg.inv(joint);post=np.linalg.inv(J[:8,:8])
        rows.append(m-post@J[:8,8:]@(y-C@m));Cs.append(post)
        logs.append(np.log(p)+multivariate_normal.logpdf(y,mean=C@m,cov=S))
    logs=np.array(logs)
    return np.exp(logs-logsumexp(logs)),np.array(rows),np.array(Cs)


def main():
    rng=np.random.default_rng(86516640);weights=np.array([.23,.47,.30])
    means=rng.normal(0,.06,(3,8));means[:,4:]+=np.array([.3,-1.,.4,.4])
    factors=rng.normal(0,.009,(3,8,8));covs=factors@factors.transpose(0,2,1)+1e-4*np.eye(8)
    belief=implementation.Belief(weights,means,covs);reports=[];maxcost=maxpg=0.
    # Full cross covariance and active bounds; both two- and three-pulse objectives.
    for steps in (2,3):
        for goal in (np.array([.001,-.001]),np.array([.005,.004])):
            plan,cost,_=implementation.plan(belief,goal,steps,.03)
            x=torch.tensor(plan.ravel(),requires_grad=True)
            loss=direct_loss(x,weights,means,covs,goal,steps,.03)
            gradient=torch.autograd.grad(loss,x)[0].detach().numpy();flat=x.detach().numpy()
            pg=float(np.max(abs(flat-np.clip(flat-gradient,-.5,.5))));error=abs(cost-loss.item())
            assert error<1e-9 and pg<1e-7,(steps,goal,error,pg)
            gm=belief.gaussian();other,othercost,_=implementation.plan(gm,goal,steps,.03)
            assert np.max(abs(other-plan))<1e-10 and abs(othercost-cost)<1e-9
            maxcost=max(maxcost,error);maxpg=max(maxpg,pg)
            reports.append(dict(steps=steps,goal=goal.tolist(),direct_cost_error=error,projected_gradient=pg,
                full_moment_matched_action_error=float(np.max(abs(other-plan)))))
    u=np.array([.07,-.04]);pred=belief.predict(u)
    # Sigma-point pushforward computes transition moments without A construction.
    transition_errors=[]
    for m,P,mp,Pp in zip(means,covs,pred.means,pred.covs):
        points=sigma(m,P);B=points[:,4:].reshape(-1,2,2);a=B@u
        new=np.c_[points[:,:2]+points[:,2:4]+.5*a,points[:,2:4]+a,points[:,4:]]
        independent=new.mean(0);err=max(np.max(abs(independent-mp)),np.max(abs(np.cov(new,rowvar=False,bias=True)-Pp)))
        transition_errors.append(float(err));assert err<1e-12
    conditioning=[]
    for y in (np.array([.02,-.01]),np.array([-.12,.06])):
        w,m,P=reference_condition(pred.weights,pred.means,pred.covs,y);post=pred.observe(y)
        err=max(np.max(abs(w-post.weights)),np.max(abs(m-post.means)),np.max(abs(P-post.covs)))
        conditioning.append(float(err));assert err<1e-10
        plan,cost,_=implementation.plan(post,np.array([.001,.001]),2,.02)
        x=torch.tensor(plan.ravel(),requires_grad=True);loss=direct_loss(x,w,m,P,np.array([.001,.001]),2,.02)
        error=abs(cost-loss.item());assert error<1e-8;maxcost=max(maxcost,error)
    # Initial exact s/w, uncertain B: reduce to the parameter-only first-batch code.
    old=batch.Prior('gaussian',mean=means[0,4:].reshape(2,2),cov=covs[0,4:,4:])
    C=np.zeros((1,8,8));C[0,4:,4:]=old.cov;m=np.zeros((1,8));m[0,4:]=old.mean.ravel()
    y=np.array([.012,-.007]);goal=np.array([.001,.001])
    post=implementation.Belief([1.],m,C).predict(u).observe(y)
    newactions,newcost,_=implementation.plan(post,goal,2,u@u)
    M,G,_=old.posterior(y,u);actions,cost,_=batch.continuation(M,G,u,goal)
    reduction=max(np.max(abs(newactions-actions[0])),abs(newcost-cost[0]));assert reduction<1e-10
    # Known B: no active bound on the entire truncated domain; expected optimized
    # loss is quadratic in observations and exactly integrated by moment points.
    known=np.zeros((1,8));known[0,4:]=[.353553390593,-1.06066017178,.353553390593,.353553390593]
    P=np.zeros((1,8,8));P[0,:4,:4]=np.eye(4)*1e-5
    knownbelief=implementation.Belief([1.],known,P);first=np.array([.03,-.02]);goal=np.array([.001,0.])
    prediction=knownbelief.predict(first);S=prediction.covs[0,:2,:2]+.0004*np.eye(2);L=np.linalg.cholesky(S)
    corners=np.array(list(itertools.product((-7.,7.),repeat=2)))
    for y in prediction.means[0,:2]+corners@L.T:
        actions,_,_=implementation.plan(prediction.observe(y),goal,2,first@first)
        assert np.max(abs(actions))<.49
    nodes=np.r_[np.sqrt(2)*L.T,-np.sqrt(2)*L.T]+prediction.means[0,:2]
    analytic=np.mean([implementation.plan(prediction.observe(y),goal,2,first@first)[1] for y in nodes])
    integrated,record=implementation.expected_first(knownbelief,goal,first,atol=1e-9,rtol=1e-8)
    integration_error=abs(analytic-integrated)
    assert integration_error<=record['estimated_error']+record['analytic_tail_upper_bound']+1e-9
    print(json.dumps(dict(scope='Independent full8D sigma trajectories/Torch quadratic gradients, precision-matrix conditioning, exact old-batch reduction and known-B analytic integration fixture; no actual controller performance audited.',
        qp_checks=reports,max_direct_cost_error=float(maxcost),max_projected_gradient=maxpg,
        transition_moment_errors=transition_errors,conditioning_errors=conditioning,
        initial_batch_reduction_error=float(reduction),analytic_integration_error=float(integration_error),
        integration_record=record),indent=2))


if __name__=='__main__':main()
