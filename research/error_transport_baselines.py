"""Scoped ALPaCA and CoDA comparison cores; no data/physics or training driver.

Inputs are standardized xs[B,S,8], ys[B,S,2], xq[B,Q,8] float32.
See error_transport_baselines.md for faithful parts and changed loss/scope.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class ALPaCA(nn.Module):
    """Learned features and matrix-normal Bayesian output layer, predictive NLL.

    Lambda0=L L^T, K0 learned; known diagonal output noise is configurable.
    Posterior Lambda=Lambda0+Phi_s^T Phi_s, K=Lambda^-1(Phi_s^T Y+Lambda0 K0).
    Cov(y_q)=(1+phi_q^T Lambda^-1 phi_q)*Sigma_noise.
    """
    def __init__(self,input_dim=8,hidden=64,feature_dim=32,output_dim=2,noise_variance=.01):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(input_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,feature_dim))
        self.K0=nn.Parameter(torch.randn(feature_dim,output_dim)*.02)
        diagonal=math.log(math.expm1(1.-1e-4))
        self.raw_precision_cholesky=nn.Parameter(torch.eye(feature_dim)*diagonal)
        noise=torch.as_tensor(noise_variance,dtype=torch.float32).expand(output_dim).clone()
        if not bool(torch.all(noise>0)):raise ValueError('noise_variance must be positive')
        self.register_buffer('noise_variance',noise)

    def prior_precision(self):
        raw=self.raw_precision_cholesky
        L=torch.tril(raw,diagonal=-1)+torch.diag_embed(F.softplus(torch.diagonal(raw))+1e-4)
        return L@L.T

    def posterior(self,xs,ys):
        phi=self.features(xs);prior=self.prior_precision()
        precision=prior+phi.transpose(-1,-2)@phi
        rhs=prior@self.K0+phi.transpose(-1,-2)@ys
        cholesky=torch.linalg.cholesky(precision)
        coefficients=torch.cholesky_solve(rhs,cholesky)
        return coefficients,cholesky

    def predictive(self,xs,ys,xq):
        coefficients,cholesky=self.posterior(xs,ys);phi=self.features(xq)
        mean=phi@coefficients
        solved=torch.cholesky_solve(phi.transpose(-1,-2),cholesky).transpose(-1,-2)
        factor=1.+torch.sum(phi*solved,dim=-1)
        variance=factor[...,None]*self.noise_variance
        return mean,variance

    def predict(self,xs,ys,xq):return self.predictive(xs,ys,xq)[0]

    def training_loss(self,xs,ys,xq,yq,task_ids=None):
        mean,variance=self.predictive(xs,ys,xq)
        return .5*torch.mean(torch.sum((yq-mean).square()/variance+torch.log(variance)+math.log(2*math.pi),dim=-1))

    def forward(self,xs,ys,xq):return self.predict(xs,ys,xq)


class CoDA(nn.Module):
    """CoDA-l2 whole-network linear hypernetwork, scoped to measured transitions.

    theta_e=theta0+W xi_e includes EVERY MLP weight and bias. Training contexts
    are train-task IDs only. Test adaptation optimizes a new zero context against
    support labels with theta0/W detached, and never takes query labels.
    """
    def __init__(self,input_dim=8,hidden=64,output_dim=2,context_dim=8,num_tasks=32,regularization=1e-4):
        super().__init__()
        self.shapes=((hidden,input_dim),(hidden,),(hidden,hidden),(hidden,),(output_dim,hidden),(output_dim,))
        self.sizes=tuple(math.prod(shape) for shape in self.shapes)
        layers=[nn.Linear(input_dim,hidden),nn.Linear(hidden,hidden),nn.Linear(hidden,output_dim)]
        self.theta0=nn.Parameter(torch.cat([p.detach().flatten() for layer in layers for p in [layer.weight,layer.bias]]))
        self.W=nn.Parameter(torch.randn(sum(self.sizes),context_dim)*.01)
        self.contexts=nn.Embedding(num_tasks,context_dim);nn.init.normal_(self.contexts.weight,std=.01)
        self.regularization=float(regularization)

    def parameter_vector(self,context):return self.theta0+context@self.W.T

    def functional(self,x,theta):
        """Vectorized complete MLP; theta[B,P], x[B,N,input_dim]."""
        pieces=torch.split(theta,self.sizes,dim=-1)
        values=[p.reshape(theta.shape[0],*shape) for p,shape in zip(pieces,self.shapes)]
        w1,b1,w2,b2,w3,b3=values
        h=torch.tanh(x@w1.transpose(-1,-2)+b1[:,None,:])
        h=torch.tanh(h@w2.transpose(-1,-2)+b2[:,None,:])
        return h@w3.transpose(-1,-2)+b3[:,None,:]

    def training_loss(self,xs,ys,xq,yq,task_ids):
        context=self.contexts(task_ids)
        delta=context@self.W.T;theta=self.theta0+delta
        x=torch.cat((xs,xq),dim=1);y=torch.cat((ys,yq),dim=1)
        return F.mse_loss(self.functional(x,theta),y)+self.regularization*delta.square().sum(-1).mean()

    def adapt_context(self,xs,ys,steps=100,lr=.03):
        """Only local context has an autograd path; global .grad stays untouched."""
        if steps<0 or lr<=0:raise ValueError('steps nonnegative and lr positive required')
        batch=xs.shape[0];base=self.theta0.detach();W=self.W.detach()
        context=torch.zeros(batch,W.shape[1],dtype=base.dtype,device=base.device)
        if xs.shape[1]==0 or steps==0:return context
        with torch.enable_grad():
            context.requires_grad_(True);optimizer=torch.optim.Adam([context],lr=lr)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                delta=context@W.T
                # Sum per-task losses so batch size does not rescale each
                # independent context gradient (Adam epsilon remains comparable).
                errors=(self.functional(xs.detach(),base+delta)-ys.detach()).square().mean(dim=(1,2))
                loss=torch.sum(errors+self.regularization*delta.square().sum(-1))
                gradient,=torch.autograd.grad(loss,context)
                context.grad=gradient;optimizer.step()
        return context.detach()

    def adapt(self,xs,ys,xq,steps=100,lr=.03):
        context=self.adapt_context(xs,ys,steps=steps,lr=lr)
        theta=self.theta0.detach()+context@self.W.detach().T
        return self.functional(xq,theta)

    def predict(self,xs,ys,xq):return self.adapt(xs,ys,xq)

    def forward(self,xs,ys,xq):return self.predict(xs,ys,xq)


ALPaCABaseline=ALPaCA
CoDABaseline=CoDA
