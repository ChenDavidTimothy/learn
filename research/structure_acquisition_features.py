#!/usr/bin/env python3
"""Construct executable features by the value of their future conditioning.

Symbolic feature selection + Bayesian regression, not a new universal principle.
The Gaussian prior is fitted separately; graph search uses sum-ridge .1.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from error_transport_screen import digest,save_json

ROOT=Path(__file__).resolve().parent
UNARY=('sin','cos','abs','recip')

def operation(op,a,b=None):
    if op=='mul':return a*b
    if op=='sin':return torch.sin(a)
    if op=='cos':return torch.cos(a)
    if op=='abs':return torch.abs(a)
    if op=='recip':return 1/(1+a.square())
    raise ValueError(op)

class ProgramFeatures(nn.Module):
    def __init__(self,graph):
        super().__init__();self.graph=copy.deepcopy(graph)
    def forward(self,x):
        values=[]
        for n in self.graph['nodes']:
            if n['op']=='input':v=x[...,n['index']]
            elif n['op']=='constant':v=torch.ones_like(x[...,0])
            else:
                args=[values[j] for j in n['args']];v=operation(n['op'],*args)
                v=(v-n.get('mean',0.))/n.get('std',1.)
            values.append(v)
        return torch.stack([values[j] for j in self.graph['features']],dim=-1)

def values(x,graph):
    vals=[]
    for n in graph['nodes']:
        if n['op']=='input':v=x[...,n['index']]
        elif n['op']=='constant':v=torch.ones_like(x[...,0])
        else:v=(operation(n['op'],*[vals[j] for j in n['args']])-n.get('mean',0.))/n.get('std',1.)
        vals.append(v)
    return vals

def initial_graph():
    nodes=[dict(op='constant')]+[dict(op='input',index=j) for j in range(8)]
    return dict(nodes=nodes,features=list(range(9)))

def complexity(expr,graph):
    if expr[0]=='ref':
        n=graph['nodes'][expr[1]]
        return 0 if n['op'] in ('input','constant') else 1+sum(complexity(('ref',j),graph) for j in n['args'])
    return 1+sum(complexity(e,graph) for e in expr[1:])

def expr_values(expr,vals):
    if expr[0]=='ref':return vals[expr[1]]
    return operation(expr[0],*[expr_values(e,vals) for e in expr[1:]])

def proposals(graph,rng):
    roots=[('ref',j) for j in graph['features'][1:]];atoms=[('ref',j) for j in range(1,9)]
    shallow=set();deep=set()
    mul=lambda a,b:('mul',*sorted((a,b),key=repr))
    for a in roots:
        for op in UNARY:shallow.add((op,a))
        for b in atoms:
            ab=mul(a,b);shallow.add(ab)
            for c in atoms:deep.add(mul(ab,c))
            for op in UNARY:deep.add(mul((op,a),b))
    def subset(pool,n):
        seq=sorted((e for e in pool if complexity(e,graph)<=15),key=repr)
        return seq if len(seq)<=n else [seq[i] for i in sorted(rng.choice(len(seq),n,replace=False))]
    # Both short fragments and nonpredictive intermediates get prospectively fixed coverage.
    return subset(shallow,256)+subset(deep-shallow,256)

def augmented_predictions(phi_s,y,phi_q,z_s,z_q,ridge=.1):
    """All candidate rank-one ridge predictions; z tensors end in candidate axis."""
    k=phi_s.shape[-1];eye=torch.eye(k,dtype=phi_s.dtype,device=phi_s.device)
    G=phi_s.transpose(-1,-2)@phi_s+ridge*eye
    W=torch.linalg.solve(G,phi_s.transpose(-1,-2)@y)
    h=phi_s.transpose(-1,-2)@z_s;v=torch.linalg.solve(G,h)
    denom=(z_s.square().sum(1)+ridge-(h*v).sum(1))
    assert torch.all(denom>ridge*.5),'Schur complement lost positive ridge margin'
    alpha=(z_s.transpose(-1,-2)@(y-phi_s@W))/denom[...,None]
    base=phi_q@W;innovation=z_q-phi_q@v
    return base[:,:,None,:]+innovation[:,:,:,None]*alpha[:,None,:,:],base

def samples(x,y,old,new,rng):
    groups=[]
    for n in (16,64):
        ids=rng.choice(old,32) if new is None else np.r_[rng.choice(old,16),rng.choice(new,16)]
        st=rng.integers(0,8,32);starts=rng.integers(0,97-n,32)
        t=starts[:,None]+np.arange(n)[None];qtraj=(st[:,None]+rng.integers(1,8,(32,32)))%8;qt=rng.integers(0,96,(32,32))
        groups.append((x[ids[:,None],st[:,None],t],y[ids[:,None],st[:,None],t],x[ids[:,None],qtraj,qt],y[ids[:,None],qtraj,qt]))
    return groups

def append_fragment(graph,expr,mean,std):
    def add(e):
        if e[0]=='ref':return e[1]
        args=[add(a) for a in e[1:]];graph['nodes'].append(dict(op=e[0],args=args,mean=0.,std=1.));return len(graph['nodes'])-1
    j=add(expr);graph['nodes'][j].update(mean=float(mean),std=float(std));graph['features'].append(j)

def search(args):
    out=ROOT/f'structure_acquisition_graph_{args.phase}.json';assert not out.exists()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(ROOT/'error_transport_screen_data.npz',allow_pickle=False) as d:
        rawx=d['x'];rawy=d['y'];tr=d['split']==0
    xm=rawx[tr].mean((0,1,2));xs=np.maximum(rawx[tr].std((0,1,2)),.01);ym=rawy[tr].mean((0,1,2));ys=np.maximum(rawy[tr].std((0,1,2)),.01)
    X=rawx[tr];Y=rawy[tr];old=np.arange(32);new=None;graph=initial_graph()
    if args.phase!='old':
        parent=ROOT/'structure_acquisition_graph_old.json';graph=copy.deepcopy(json.loads(parent.read_text()))
        with np.load(ROOT/'structure_acquisition_data.npz',allow_pickle=False) as d:
            keep=d['split']==0;X=np.concatenate([X,d['x'][keep]]);Y=np.concatenate([Y,d['y'][keep]])
        new=np.arange(32,40)
    x=torch.as_tensor((X-xm)/xs,dtype=torch.float64,device=device);y=torch.as_tensor((Y-ym)/ys,dtype=torch.float64,device=device)
    rng=np.random.default_rng(np.random.SeedSequence([20261020,18,42,('old','meta','fit').index(args.phase)]))
    # Fixed training-only behavior probe for scaling and semantic deduplication.
    flat=x.reshape(-1,8);probe=flat[torch.as_tensor(rng.choice(len(flat),2048,replace=False),device=device)]
    records=[];start=time.time();target=32 if args.phase=='old' else len(graph['features'])+16
    status='FEATURE TARGET REACHED'
    while len(graph['features'])<target:
        if time.time()-start>120:status='DECLARED SEARCH TIME BUDGET REACHED';break
        candidates=proposals(graph,rng);pv=values(probe,graph);raw=torch.stack([expr_values(e,pv) for e in candidates],dim=-1)
        mean=raw.mean(0);std=raw.std(0,unbiased=False)
        valid=torch.isfinite(raw).all(0)&(raw.abs().amax(0)<1e6)&(std>.05)
        # A near-linear duplicate also just relieves ridge regularization. Reject it.
        z=(raw-mean)/std.clamp_min(.05);existing=ProgramFeatures(graph)(probe)
        Q=torch.linalg.qr(existing,mode='reduced').Q
        unexplained=(z-Q@(Q.T@z)).square().sum(0)/z.square().sum(0).clamp_min(1e-20)
        valid &= unexplained>1e-5
        ids=torch.nonzero(valid,as_tuple=False).flatten();assert len(ids)>0,'no admissible proposals'
        groups=samples(x,y,old,new,rng);losses=torch.zeros(len(ids),dtype=x.dtype,device=device);basevalue=0.
        for sx,sy,qx,qy in groups:
            if args.phase=='fit':sx=torch.cat([sx,qx],dim=1);sy=torch.cat([sy,qy],dim=1);qx=sx;qy=sy
            sv=values(sx,graph);qv=values(qx,graph)
            ps=torch.stack([sv[j] for j in graph['features']],dim=-1);pq=torch.stack([qv[j] for j in graph['features']],dim=-1)
            zs=torch.stack([expr_values(candidates[i],sv) for i in ids.tolist()],dim=-1);zq=torch.stack([expr_values(candidates[i],qv) for i in ids.tolist()],dim=-1)
            zs=(zs-mean[ids])/std[ids];zq=(zq-mean[ids])/std[ids]
            pred,base=augmented_predictions(ps,sy,pq,zs,zq)
            losses+=(pred-qy[:,:,None,:]).square().mean((0,1,3))/2;basevalue+=float((base-qy).square().mean())/2
        ix=int(torch.argmin(losses));best=float(losses[ix]);j=int(ids[ix]);gain=basevalue-best
        rec=dict(feature_count_before=len(graph['features']),expression=repr(candidates[j]),mean=float(mean[j]),std=float(std[j]),
            baseline_loss=basevalue,selected_loss=best,improvement=gain,candidates=len(candidates),admissible=int(len(ids)),wall_seconds=time.time()-start)
        records.append(rec)
        if gain<=max(1e-9,1e-5*abs(basevalue)):status='NO MEANINGFUL TRAINING-EPISODE IMPROVEMENT';break
        append_fragment(graph,candidates[j],mean[j],std[j]);print(args.phase,len(graph['features']),best,flush=True)
    graph.update(status=status,phase=args.phase,feature_count=len(graph['features']),selection_records=records,
        supplied_grammar=list(UNARY)+['mul'],source_sha256=digest(__file__),plan_sha256=digest(ROOT/'structure_acquisition_plan.md'),
        data_sha256=digest(ROOT/'error_transport_screen_data.npz'),wall_seconds=time.time()-start,search_ridge=.1,
        limitations='Training-episode structure search, no validation/test criterion. Generic operators are supplied priors; semantic dedup is finite on training probe, not a symbolic identity proof.')
    if new is not None:graph.update(acquisition_sha256=digest(ROOT/'structure_acquisition_data.npz'),parent_sha256=digest(parent))
    save_json(out,graph)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('old','meta','fit'));args=parser.parse_args();torch.set_num_threads(1)
    search(args)
