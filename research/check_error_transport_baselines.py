#!/usr/bin/env python3
"""Small synthetic baseline-core checks, no campaign sampling or training."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from error_transport_baselines import ALPaCA,CoDA


def main():
    torch.set_num_threads(1);torch.manual_seed(83172)
    xs=torch.randn(3,7,8);ys=torch.randn(3,7,2);xq=torch.randn(3,5,8);yq=torch.randn(3,5,2)
    alp=ALPaCA();mean,var=alp.predictive(xs,ys,xq)
    with torch.no_grad():
        phi=alp.features(xs);prior=alp.prior_precision();P=torch.linalg.inv(prior).expand(3,-1,-1).clone();K=alp.K0.expand(3,-1,-1).clone()
        for t in range(7):
            feature=phi[:,t];v=P@feature[:,:,None];den=1+(feature[:,None,:]@v)
            residual=ys[:,t]-(feature[:,None,:]@K).squeeze(1)
            K=K+(v/den)@residual[:,None,:];P=P-(v@v.transpose(-1,-2))/den
        batch,ch=alp.posterior(xs,ys)
        posterior_error=float(torch.max(abs(K-batch)))
        covariance_error=float(torch.max(abs(P-torch.cholesky_inverse(ch))))
        assert posterior_error<2e-6 and covariance_error<2e-6
        zero=alp.predict(xs[:,:0],ys[:,:0],xq)
        zero_error=float(torch.max(abs(zero-alp.features(xq)@alp.K0)));assert zero_error<2e-6
        direct=.5*torch.mean(torch.sum((yq-mean).square()/var+torch.log(var)+torch.log(torch.tensor(2*torch.pi)),dim=-1))
        assert torch.all(var>0) and torch.allclose(alp.training_loss(xs,ys,xq,yq),direct)
    alp.training_loss(xs,ys,xq,yq).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in alp.parameters())
    coda=CoDA();context=torch.randn(3,8)*.1;theta=coda.parameter_vector(context);functional=coda.functional(xq,theta)
    references=[]
    for task in range(3):
        parts=torch.split(theta[task],coda.sizes)
        w1,b1,w2,b2,w3,b3=[v.reshape(shape) for v,shape in zip(parts,coda.shapes)]
        references.append(F.linear(torch.tanh(F.linear(torch.tanh(F.linear(xq[task],w1,b1)),w2,b2)),w3,b3))
    mapping_error=float(torch.max(abs(functional-torch.stack(references))).detach());assert mapping_error<2e-6
    # Directional derivative checks every theta coordinate through one functional
    # network, and hypernetwork/context jointly, without a numerical ladder.
    double_model=CoDA().double();xd=xq.double();cd=context.double().requires_grad_(True)
    direction=torch.randn_like(double_model.W);weight=torch.randn(3,5,2,dtype=torch.float64)
    output=double_model.functional(xd,double_model.parameter_vector(cd));loss=(output*weight).sum()
    gw,gc=torch.autograd.grad(loss,[double_model.W,cd]);eps=1e-6
    with torch.no_grad():
        baseW=double_model.W.clone();double_model.W.copy_(baseW+eps*direction);plus=(double_model.functional(xd,double_model.parameter_vector(cd))*weight).sum()
        double_model.W.copy_(baseW-eps*direction);minus=(double_model.functional(xd,double_model.parameter_vector(cd))*weight).sum();double_model.W.copy_(baseW)
    derivative_error=float(abs((plus-minus)/(2*eps)-(gw*direction).sum()));assert derivative_error<1e-7
    ids=torch.tensor([0,3,11]);training_loss=coda.training_loss(xs,ys,xq,yq,ids);training_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in coda.parameters())
    before={k:v.detach().clone() for k,v in coda.state_dict().items()};old_grads=[p.grad.clone() for p in coda.parameters()]
    with torch.no_grad():prediction=coda.adapt(xs,ys,xq,steps=3)
    assert prediction.shape==(3,5,2) and torch.isfinite(prediction).all()
    assert all(torch.equal(v,before[k]) for k,v in coda.state_dict().items())
    assert all(torch.equal(p.grad,g) for p,g in zip(coda.parameters(),old_grads))
    zero_pred=coda.adapt(xs[:,:0],ys[:,:0],xq,steps=3)
    base_pred=coda.functional(xq,coda.theta0.detach().expand(3,-1))
    zero_coda_error=float(torch.max(abs(zero_pred-base_pred)).detach());assert zero_coda_error==0.
    # Same support/new context fitting ignores query LABELS by construction and
    # does not depend on the query INPUTS for its inferred context.
    batch_context=coda.adapt_context(xs,ys,steps=3)
    separate=torch.cat([coda.adapt_context(xs[i:i+1],ys[i:i+1],steps=3) for i in range(3)])
    batch_adapt_error=float(torch.max(abs(batch_context-separate)));assert batch_adapt_error<1e-6
    source=Path(__file__).with_name('error_transport_baselines.py')
    result=dict(status='PASS SYNTHETIC BASELINE CORES; NO LEARNING PERFORMANCE CLAIM',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seed=83172,alpaca_recursive_vs_batch_mean_weight_max_error=posterior_error,alpaca_recursive_vs_batch_covariance_max_error=covariance_error,
        alpaca_zero_support_prior_prediction_max_error=zero_error,alpaca_finite_NLL_and_all_parameter_gradients=True,
        coda_whole_parameter_functional_mapping_max_error=mapping_error,coda_hypernetwork_directional_gradient_error=derivative_error,
        coda_zero_support_base_prediction_max_error=zero_coda_error,coda_batch_vs_separate_adaptation_context_max_error=batch_adapt_error,
        coda_adaptation_global_parameters_and_existing_gradients_unchanged=True,
        parameter_counts=dict(alpaca=sum(p.numel() for p in alp.parameters()),coda=sum(p.numel() for p in coda.parameters()),coda_base=coda.theta0.numel(),coda_hypernetwork=coda.W.numel()),
        limitations='Author implementation fixtures, not independent consequential-result reproduction. No campaign physics/data sampled; no meta-training or baseline competitiveness tested.')
    Path(__file__).with_name('check_error_transport_baselines_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
