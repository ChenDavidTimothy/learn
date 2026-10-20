"""Candidate residual-conditioned functional updater, not claimed novel.

Learned representations and iterative function updates overlap MetaFun. This
specific bounded output-matrix update is a testable variant, not a new paradigm.
"""
import torch
from torch import nn


def mlp(din,dout,width=64):
    return nn.Sequential(nn.Linear(din,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,dout))


class ErrorTransport(nn.Module):
    def __init__(self,xdim=8,ydim=2,width=64,embedding=16,updates=3):
        super().__init__();self.base=mlp(xdim,ydim,width);self.encode=mlp(xdim,embedding,width)
        self.operator=nn.Sequential(nn.Linear(embedding*3+ydim,width),nn.Tanh(),nn.Linear(width,ydim*ydim))
        self.gates=nn.Parameter(torch.zeros(updates));self.ydim=ydim;self.updates=updates
        nn.init.zeros_(self.operator[-1].bias)

    def predict(self,xs,ys,xq):
        batch,support,_=xs.shape
        if support==0:return self.base(xq)
        allx=torch.cat([xs,xq],dim=1);embedding=self.encode(allx);se=embedding[:,:support]
        n=allx.shape[1];prediction=self.base(allx)
        qe=embedding[:,:,None,:].expand(-1,-1,support,-1)
        se=se[:,None,:,:].expand(-1,n,-1,-1)
        for i in range(self.updates):
            residual=ys-prediction[:,:support]
            r=residual[:,None,:,:].expand(-1,n,-1,-1)
            inputs=torch.cat([qe,se,qe-se,r],dim=-1)
            matrix=2*torch.tanh(self.operator(inputs)).reshape(batch,n,support,self.ydim,self.ydim)
            correction=(matrix@r[...,None]).squeeze(-1).mean(dim=2)
            prediction=prediction+torch.sigmoid(self.gates[i])*correction
        return prediction[:,support:]

    def training_loss(self,xs,ys,xq,yq,task_ids=None):
        return ((self.predict(xs,ys,xq)-yq)**2).mean()


class SeparableTransport(nn.Module):
    """Matched MetaFun-family ablation: learned scalar similarity/local vector.

    Zero residual produces exactly no update, avoiding a constraint confound.
    This is a matched operation, not reproduction of all published MetaFun heads.
    """
    def __init__(self,xdim=8,ydim=2,width=64,embedding=16,updates=3):
        super().__init__();self.base=mlp(xdim,ydim,width);self.encode=mlp(xdim,embedding,width)
        self.similarity=nn.Sequential(nn.Linear(embedding*3,width),nn.Tanh(),nn.Linear(width,1))
        self.local=nn.Sequential(nn.Linear(embedding+ydim,width),nn.Tanh(),nn.Linear(width,ydim))
        self.gates=nn.Parameter(torch.zeros(updates));self.updates=updates

    def predict(self,xs,ys,xq):
        support=xs.shape[1]
        if support==0:return self.base(xq)
        allx=torch.cat([xs,xq],1);embedding=self.encode(allx);se=embedding[:,:support]
        n=allx.shape[1];prediction=self.base(allx)
        qe=embedding[:,:,None,:].expand(-1,-1,support,-1)
        se_all=se[:,None,:,:].expand(-1,n,-1,-1)
        weights=2*torch.tanh(self.similarity(torch.cat([qe,se_all,qe-se_all],-1)))
        for i in range(self.updates):
            residual=ys-prediction[:,:support]
            local=self.local(torch.cat([se,residual],-1))-self.local(torch.cat([se,torch.zeros_like(residual)],-1))
            correction=(weights*local[:,None,:,:]).mean(2)
            prediction=prediction+torch.sigmoid(self.gates[i])*correction
        return prediction[:,support:]

    def training_loss(self,xs,ys,xq,yq,task_ids=None):
        return ((self.predict(xs,ys,xq)-yq)**2).mean()


class Pooled(nn.Module):
    def __init__(self,xdim=8,ydim=2,width=64):
        super().__init__();self.base=mlp(xdim,ydim,width)
    def predict(self,xs,ys,xq):return self.base(xq)
    def training_loss(self,xs,ys,xq,yq,task_ids=None):
        return ((self.base(torch.cat([xs,xq],1))-torch.cat([ys,yq],1))**2).mean()
