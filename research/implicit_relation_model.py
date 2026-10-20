"""Learned implicit affine relation, contextual ridge conditioning, query solve.

This is a supplied relation family with learned features, not physical parameter
identification or a claimed new general learning principle. Explicit nesting is
available with implicit=False. Query invertibility is tested, not guaranteed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from error_transport_model import mlp


class ImplicitRelation(nn.Module):
    def __init__(self,implicit=True,xdim=8,ydim=2,rank=8,width=64):
        super().__init__();self.implicit=implicit;self.rank=rank;self.ydim=ydim
        self.base=mlp(xdim,ydim,width)
        self.vectors=mlp(xdim,rank*ydim,width)
        self.matrices=mlp(xdim,rank*ydim*ydim,width) if implicit else None
        if implicit:
            nn.init.normal_(self.matrices[-1].weight,std=.001);nn.init.zeros_(self.matrices[-1].bias)
        self.raw_ridge=nn.Parameter(torch.tensor(.54132485))

    def components(self,x):
        shape=(*x.shape[:-1],self.rank,self.ydim)
        base=self.base(x);vectors=self.vectors(x).reshape(shape)
        if self.implicit:
            matrices=.05*torch.tanh(self.matrices(x)).reshape(*shape,self.ydim)
        else:
            matrices=torch.zeros(*shape,self.ydim,device=x.device,dtype=x.dtype)
        return base,vectors,matrices

    def infer(self,xs,ys):
        batch,n,_=xs.shape
        if n==0:return torch.zeros(batch,self.rank,dtype=xs.dtype,device=xs.device)
        base,b,A=self.components(xs)
        # D[B,S,Y,R] is built from OBSERVED support y, never a query label.
        D=(b-(A@ys[:,:,None,:,None]).squeeze(-1)).transpose(-1,-2)
        design=D.reshape(batch,n*self.ydim,self.rank)
        residual=(ys-base).reshape(batch,n*self.ydim,1)
        ridge=F.softplus(self.raw_ridge)+1e-4
        gram=design.transpose(-1,-2)@design+ridge*torch.eye(self.rank,dtype=xs.dtype,device=xs.device)
        rhs=design.transpose(-1,-2)@residual
        return torch.linalg.solve(gram,rhs).squeeze(-1)

    def conditioned(self,c,xq,diagnostics=False):
        base,b,A=self.components(xq)
        M=torch.eye(self.ydim,dtype=xq.dtype,device=xq.device)+torch.einsum('br,bqrjk->bqjk',c,A)
        target=base+torch.einsum('br,bqrj->bqj',c,b)
        prediction,info=torch.linalg.solve_ex(M,target[...,None])
        if bool(torch.any(info!=0)):raise ArithmeticError('Singular learned query relation')
        prediction=prediction.squeeze(-1)
        if not bool(torch.all(torch.isfinite(prediction))):raise ArithmeticError('Nonfinite learned query consequence')
        if diagnostics:
            singular=torch.linalg.svdvals(M)
            return prediction,dict(min_singular=singular[...,-1],condition=singular[...,0]/singular[...,-1],latent=c)
        return prediction

    def predict(self,xs,ys,xq):return self.conditioned(self.infer(xs,ys),xq)
    def training_loss(self,xs,ys,xq,yq,task_ids=None):return (self.predict(xs,ys,xq)-yq).square().mean()
