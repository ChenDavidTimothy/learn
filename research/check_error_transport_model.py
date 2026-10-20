#!/usr/bin/env python3
"""Small prefit synthetic checks; no model training or physical simulation."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from error_transport_model import ErrorTransport,SeparableTransport
import error_transport_screen as driver
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-10):
 e=float((a-b).abs().max()) if a.numel() else 0.;assert e<tol,(e,tol);return e

def main():
 start=time.time();torch.set_num_threads(1);torch.manual_seed(1901);results={}
 for cls in(ErrorTransport,SeparableTransport):
  m=cls(xdim=3,ydim=2,width=7,embedding=4,updates=3).double()
  xs=torch.randn(2,5,3,dtype=torch.double);ys=torch.randn(2,5,2,dtype=torch.double);q=torch.randn(2,7,3,dtype=torch.double)
  p=m.predict(xs,ys,q);assert p.shape==(2,7,2)
  check={};check['zero_support_maxerror']=close(m.predict(xs[:,:0],ys[:,:0],q),m.base(q))
  with torch.no_grad():labels=m.base(xs)
  check['zero_residual_maxerror']=close(m.predict(xs,labels,q),m.base(q))
  pi=torch.tensor([3,0,4,1,2]);check['support_permutation_maxerror']=close(m.predict(xs[:,pi],ys[:,pi],q),p)
  qi=torch.tensor([6,0,2,4,1,5,3]);check['query_permutation_maxerror']=close(m.predict(xs,ys,q[:,qi]),p[:,qi])
  check['chunk_maxerror']=close(torch.cat([m.predict(xs,ys,t) for t in q.split(3,1)],1),p)
  # No interaction across task batch or across query points beyond support.
  check['batch_separation_maxerror']=close(torch.cat([m.predict(xs[i:i+1],ys[i:i+1],q[i:i+1]) for i in range(2)]),p)
  probe=m.training_loss(xs,ys,q,torch.randn_like(p));probe.backward()
  grads={name:float(v.grad.norm()) for name,v in m.named_parameters() if v.grad is not None}
  assert all(np.isfinite(v) for v in grads.values()) and all(name in grads for name,_ in m.named_parameters())
  # Independent central differences for inputs; PyTorch numerical gradcheck is a fixture, not independent implementation.
  small=tuple(x[:1,:2].detach().clone().requires_grad_(True) for x in(xs,ys,q))
  check['input_gradcheck']=bool(torch.autograd.gradcheck(lambda a,b,c:m.predict(a,b,c),small,eps=1e-6,atol=2e-5,rtol=2e-4))
  # One parameter direction checks complete update including gates/encoder/operator.
  m.zero_grad(set_to_none=True);loss=m.training_loss(xs,ys,q,torch.zeros_like(p));loss.backward();direction={name:torch.randn_like(v) for name,v in m.named_parameters()};analytic=sum(float((v.grad*direction[name]).sum()) for name,v in m.named_parameters());eps=1e-6
  original={name:v.detach().clone() for name,v in m.named_parameters()}
  with torch.no_grad():
   for name,v in m.named_parameters():v.copy_(original[name]+eps*direction[name])
   plus=float(m.training_loss(xs,ys,q,torch.zeros_like(p)))
   for name,v in m.named_parameters():v.copy_(original[name]-eps*direction[name])
   minus=float(m.training_loss(xs,ys,q,torch.zeros_like(p)))
   for name,v in m.named_parameters():v.copy_(original[name])
  fd=(plus-minus)/(2*eps);check['parameter_direction_error']=abs(fd-analytic);assert abs(fd-analytic)<1e-6*(1+abs(fd))
  check['finite_gradient_norms']=grads
  results[cls.__name__]=check
 # Distinct label-coded records verify support/query indices and wrong-support convention.
 x=torch.arange(6*8*96*8,dtype=torch.float64).reshape(6,8,96,8);y=torch.arange(6*8*96*2,dtype=torch.float64).reshape(6,8,96,2);ids=np.array([2,4,5])
 for n in(0,4,16,64):
  sx,sy,qx,qy=driver.fixed_samples(x,y,ids,n);assert torch.equal(sx,x[ids,0,:n]) and torch.equal(sy,y[ids,0,:n]) and torch.equal(qx,x[ids,1:].flatten(1,2)) and torch.equal(qy,y[ids,1:].flatten(1,2))
  sx,sy,wrongqx,wrongqy=driver.fixed_samples(x,y,ids,n,True);assert torch.equal(sx,x[np.roll(ids,1),0,:n]) and torch.equal(sy,y[np.roll(ids,1),0,:n]) and torch.equal(qx,wrongqx) and torch.equal(qy,wrongqy)
 # Check ordinary data's scale is exactly train-only, and perturb heldout data in a temporary in-memory reader.
 with np.load(driver.DATA,allow_pickle=False) as a:raw={k:a[k].copy() for k in a.files}
 actualx,actualy,split,scale=driver.load('cpu');tr=raw['split']==0
 for key,expected in [('x_mean',raw['x'][tr].mean((0,1,2))),('x_std',np.maximum(raw['x'][tr].std((0,1,2)),.01)),('y_mean',raw['y'][tr].mean((0,1,2))),('y_std',np.maximum(raw['y'][tr].std((0,1,2)),.01))]:assert np.array_equal(scale[key],expected)
 assert list(np.bincount(split))==[32,8,8,8]
 # Driver's sampled query trajectory offset is always nonzero modulo8.
 rr=np.random.default_rng(1902);st=rr.integers(0,8,8);qt=(st[:,None]+rr.integers(1,8,(8,1000)))%8;assert not np.any(qt==st[:,None])
 out=dict(status='PASS SMALL PREFIT MODEL/DATAFLOW FIXTURES; NOT CAPABILITY EVIDENCE',checks=results,driver_checks=dict(fixed_support_queries_disjoint=True,wrong_support_keeps_query_targets=True,train_only_scales_exact=True,body_split_counts=[32,8,8,8],sampled_training_query_trajectory_disjoint=True),source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'error_transport_model.py',ROOT/'error_transport_screen.py',ROOT/'error_transport_baselines.py',driver.DATA]},wall_seconds=time.time()-start,limitations=['Small fixtures only; no training optimization, test forecasts, model capability or independent physics checked.','Autograd gradcheck and finite differences verify local derivatives; not independent reimplementation of model.','Bounded matrix entries do not ensure contraction or prevent iterative amplification.','Candidate and separable comparator are related to existing functional meta-learning; tests do not establish originality.','Driver training objectives differ by algorithm; identical sampled records are not identical optimization.'])
 (ROOT/'check_error_transport_model_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
