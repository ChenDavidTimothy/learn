#!/usr/bin/env python3
"""Tiny independent relation algebra/gradient fixtures; no campaign outcomes.

Candidate imports are tested units, not an independent result reproduction.
Reference inference uses augmented NumPy least squares and explicit index loops.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from implicit_relation_model import ImplicitRelation


def main():
    torch.set_num_threads(1);torch.manual_seed(64187)
    rng=np.random.default_rng(64187)
    # A fixed table isolates new inference/solve algebra from neural features.
    batch=2;support=4;queries=3;rank=3;ydim=2;entries=support+queries
    f=rng.normal(0,.2,(batch,entries,ydim));b=rng.normal(0,.3,(batch,entries,rank,ydim));A=rng.normal(0,.03,(batch,entries,rank,ydim,ydim))
    observations=rng.normal(0,.3,(batch,support,ydim))
    class Table(ImplicitRelation):
        def __init__(self):
            super().__init__(implicit=True,xdim=1,rank=rank,width=3);self.double()
        def components(self,x):
            index=x[:,:,0].long();task=torch.arange(batch)[:,None]
            return tuple(torch.tensor(value)[task,index] for value in [f,b,A])
    model=Table();xs=torch.arange(support,dtype=torch.float64)[None,:,None].expand(batch,-1,-1)
    xq=torch.arange(support,entries,dtype=torch.float64)[None,:,None].expand(batch,-1,-1);ys=torch.tensor(observations)
    c=model.infer(xs,ys);prediction,diagnostics=model.conditioned(c,xq,diagnostics=True)
    ridge=math.log1p(math.exp(float(model.raw_ridge.detach())))+1e-4
    coefficient_ref=[];prediction_ref=[];residuals=[];minsingular=[]
    for task in range(batch):
        design=np.zeros((support*ydim,rank));residual=np.zeros(support*ydim)
        for i in range(support):
            for output in range(ydim):
                residual[i*ydim+output]=observations[task,i,output]-f[task,i,output]
                for j in range(rank):
                    design[i*ydim+output,j]=b[task,i,j,output]-sum(A[task,i,j,output,k]*observations[task,i,k] for k in range(ydim))
        augmented=np.vstack((design,math.sqrt(ridge)*np.eye(rank)))
        cc=np.linalg.lstsq(augmented,np.r_[residual,np.zeros(rank)],rcond=None)[0];coefficient_ref.append(cc)
        task_predictions=[];task_sv=[]
        for qi in range(support,entries):
            M=np.eye(ydim);target=f[task,qi].copy()
            for j in range(rank):M+=cc[j]*A[task,qi,j];target+=cc[j]*b[task,qi,j]
            yy=np.linalg.solve(M,target);task_predictions.append(yy);task_sv.append(np.linalg.svd(M,compute_uv=False)[-1])
            residuals.append(float(np.max(abs(M@yy-target))))
        prediction_ref.append(task_predictions);minsingular.append(task_sv)
    inference_error=float(np.max(abs(c.detach().numpy()-coefficient_ref)))
    query_error=float(np.max(abs(prediction.detach().numpy()-prediction_ref)))
    singular_error=float(np.max(abs(diagnostics['min_singular'].detach().numpy()-minsingular)))
    assert inference_error<1e-12 and query_error<1e-12 and singular_error<1e-12

    # Exact nested comparison with identical learned base/vector networks and
    # identically zero implicit matrices, not merely approximately small A.
    implicit=ImplicitRelation(implicit=True,width=7,rank=3).double()
    explicit=ImplicitRelation(implicit=False,width=7,rank=3).double()
    explicit.base.load_state_dict(implicit.base.state_dict());explicit.vectors.load_state_dict(implicit.vectors.state_dict())
    with torch.no_grad():
        implicit.matrices[-1].weight.zero_();implicit.matrices[-1].bias.zero_();explicit.raw_ridge.copy_(implicit.raw_ridge)
    sx=torch.randn(2,5,8,dtype=torch.float64);sy=torch.randn(2,5,2,dtype=torch.float64);qx=torch.randn(2,4,8,dtype=torch.float64);qy=torch.randn(2,4,2,dtype=torch.float64)
    nesting_coefficient_error=float(torch.max(abs(implicit.infer(sx,sy)-explicit.infer(sx,sy))).detach())
    nesting_prediction_error=float(torch.max(abs(implicit.predict(sx,sy,qx)-explicit.predict(sx,sy,qx))).detach())
    assert nesting_coefficient_error==0 and nesting_prediction_error==0
    zero_support_error=0.
    for m in [implicit,explicit]:
        cc=m.infer(sx[:,:0],sy[:,:0]);assert torch.equal(cc,torch.zeros_like(cc))
        error=float(torch.max(abs(m.predict(sx[:,:0],sy[:,:0],qx)-m.base(qx))).detach());zero_support_error=max(zero_support_error,error);assert error==0.

    # A fresh nonzero matrix head verifies that both stages and EVERY adaptation
    # head can acquire gradients. Central directional differences include the
    # solve dependence; no network or physical training is performed.
    differentiable=ImplicitRelation(implicit=True,width=7,rank=3).double()
    loss=differentiable.training_loss(sx,sy,qx,qy);loss.backward()
    grad_norms={name:float(parameter.grad.norm()) for name,parameter in differentiable.named_parameters()}
    assert all(math.isfinite(value) and value>0 for value in grad_norms.values())
    checks={}
    for name in ['raw_ridge','vectors.4.weight','matrices.4.weight']:
        parameter=dict(differentiable.named_parameters())[name];direction=torch.randn_like(parameter);gradient=parameter.grad.detach().clone()
        original=parameter.detach().clone();eps=1e-6
        with torch.no_grad():
            parameter.copy_(original+eps*direction);plus=float(differentiable.training_loss(sx,sy,qx,qy))
            parameter.copy_(original-eps*direction);minus=float(differentiable.training_loss(sx,sy,qx,qy));parameter.copy_(original)
        finite_difference=(plus-minus)/(2*eps);analytic=float((gradient*direction).sum());error=abs(analytic-finite_difference)
        assert error<1e-7
        checks[name]=dict(analytic=analytic,finite_difference=finite_difference,error=error)
    # Inputs and support labels are legitimately differentiable. Query targets
    # never enter the inference/prediction method; they only define train loss.
    dsx=sx.clone().requires_grad_();dsy=sy.clone().requires_grad_();dqx=qx.clone().requires_grad_()
    out=differentiable.predict(dsx,dsy,dqx);input_grads=torch.autograd.grad(out.square().sum(),[dsx,dsy,dqx])
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in input_grads)
    # Keep singularity as a failure: no hidden regularization/fallback in query.
    class Singular(ImplicitRelation):
        def components(self,x):
            ff=torch.ones(*x.shape[:-1],2,dtype=x.dtype)
            bb=torch.zeros(*x.shape[:-1],1,2,dtype=x.dtype)
            aa=-torch.eye(2,dtype=x.dtype).expand(*x.shape[:-1],1,2,2)
            return ff,bb,aa
    singular=Singular(rank=1).double();raised=False
    try:singular.conditioned(torch.ones(2,1,dtype=torch.float64),qx)
    except ArithmeticError:raised=True
    assert raised
    root=Path(__file__).resolve().parent
    result=dict(status='PASS TINY RELATION/NESTING/DIFFERENTIABILITY CHECKS; NO LEARNING CLAIM',seed=64187,
        source_sha256=hashlib.sha256((root/'implicit_relation_model.py').read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((root/'implicit_relation_plan.md').read_bytes()).hexdigest(),
        independent_augmented_lstsq_coefficient_max_error=inference_error,independent_query_relation_solve_max_error=query_error,
        diagnostic_min_singular_max_error=singular_error,query_equation_residual_max=max(residuals),ridge_initialization=ridge,
        A_zero_exact_nesting=dict(coefficient_error=nesting_coefficient_error,prediction_error=nesting_prediction_error),
        zero_support_base_prediction_max_error=zero_support_error,all_parameter_gradient_norms=grad_norms,
        central_directional_gradient_checks=checks,input_support_label_query_gradient_norms=[float(g.norm()) for g in input_grads],
        singular_relation_raises_without_fallback=raised,
        limitations='Small author-independent algebra reference and unit fixtures import the candidate as tested adapter. No data sampling, physical integration, optimizer/network training, heldout result reading, OOD invertibility or capability claim. Conditioning-failure threshold1e-4 must be handled by screen diagnostics; solve_ex only catches numerical singularity.')
    (root/'check_implicit_relation_model_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
