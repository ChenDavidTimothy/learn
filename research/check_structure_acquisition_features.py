#!/usr/bin/env python3
"""Tiny independent Schur-complement ridge fixture, no learning or physics."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
import structure_acquisition_features as production
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.time();rng=np.random.default_rng(1918);records=[];ridge=.1
 for name in ('generic','duplicate','zero','nearly_duplicate','rescaled'):
  E,S,K,Q,O=4,17,9,13,2;phi=rng.normal(size=(E,S,K));q=rng.normal(size=(E,Q,K));y=rng.normal(size=(E,S,O));z=rng.normal(size=(E,S));zq=rng.normal(size=(E,Q))
  if name=='duplicate':z=phi[:,:,2].copy();zq=q[:,:,2].copy()
  if name=='zero':z[:]=0;zq[:]=0
  if name=='nearly_duplicate':z=phi[:,:,2]+1e-9*z;zq=q[:,:,2]+1e-9*zq
  if name=='rescaled':z*=30;zq*=30
  G=np.einsum('esk,esj->ekj',phi,phi)+ridge*np.eye(K);h=np.einsum('esk,es->ek',phi,z);W=np.linalg.solve(G,np.einsum('esk,eso->eko',phi,y));v=np.linalg.solve(G,h[...,None])[...,0]
  d=np.einsum('es,es->e',z,z)+ridge-np.einsum('ek,ek->e',h,v)
  residual=z-np.einsum('esk,ek->es',phi,v);stable_d=ridge+np.sum(residual**2,axis=1)+ridge*np.sum(v*v,axis=1)
  a=(np.einsum('es,eso->eo',z,y)-np.einsum('ek,eko->eo',h,W))/d[:,None]
  fast=np.einsum('eqk,eko->eqo',q,W)+(zq-np.einsum('eqk,ek->eq',q,v))[:,:,None]*a[:,None,:]
  # Independent reference: augmented residual design including sqrt(ridge)I;
  # QR/SVD least squares, not an augmented normal-equation solve.
  direct=[];coef=[]
  for i in range(E):
   P=np.c_[phi[i],z[i]];aug=np.r_[P,np.sqrt(ridge)*np.eye(K+1)];target=np.r_[y[i],np.zeros((K+1,O))];c=np.linalg.lstsq(aug,target,rcond=None)[0];coef.append(c);direct.append(np.c_[q[i],zq[i]]@c)
  err=float(abs(fast-np.array(direct)).max());co=np.concatenate([W-v[:,:,None]*a[:,None,:],a[:,None,:]],axis=1);ce=float(abs(co-np.array(coef)).max())
  assert err<1e-9 and ce<1e-9 and float(d.min())>=ridge-1e-9 and np.allclose(d,stable_d,atol=1e-9,rtol=1e-10)
  records.append(dict(case=name,query_prediction_max_error=err,coefficient_max_error=ce,minimum_schur_denominator=float(d.min()),direct_vs_nonnegative_denominator_max_error=float(abs(d-stable_d).max())))
 # Duplicate columns change the effective ridge penalty despite no new span.
 phi=np.array([[1.],[2.],[3.]]);y=np.array([[1.],[2.],[3.]])
 w=np.linalg.solve(phi.T@phi+ridge*np.eye(1),phi.T@y);dup=np.c_[phi,phi];wd=np.linalg.solve(dup.T@dup+ridge*np.eye(2),dup.T@y)
 before=float(w[0,0]);after=float(wd.sum());assert after>before
 # Actual production batched-candidate routine versus independent augmented SVD.
 E,S,K,Q,O,C=3,19,8,11,2,4
 P=rng.normal(size=(E,S,K));PQ=rng.normal(size=(E,Q,K));Y=rng.normal(size=(E,S,O));Z=rng.normal(size=(E,S,C));ZQ=rng.normal(size=(E,Q,C))
 Z[:,:,1]=P[:,:,2];ZQ[:,:,1]=PQ[:,:,2];Z[:,:,2]=0;ZQ[:,:,2]=0
 tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
 fast,base=production.augmented_predictions(*map(tensor,(P,Y,PQ,Z,ZQ)),ridge=ridge);direct=np.zeros((E,Q,C,O))
 for e in range(E):
  for c in range(C):
   pp=np.c_[P[e],Z[e,:,c]];aug=np.r_[pp,np.sqrt(ridge)*np.eye(K+1)];yy=np.r_[Y[e],np.zeros((K+1,O))]
   coefficients=np.linalg.lstsq(aug,yy,rcond=None)[0];direct[e,:,c]=np.c_[PQ[e],ZQ[e,:,c]]@coefficients
 production_error=float(abs(fast.numpy()-direct).max());assert production_error<1e-10
 # Nested retained computations must agree on unseen inputs, not only probe values.
 graph=production.initial_graph();probe=tensor(rng.normal(size=(61,8)));query=tensor(rng.normal(size=(29,8)))
 first=('mul',('sin',('ref',1)),('ref',2));raw=production.expr_values(first,production.values(probe,graph));mean=raw.mean();std=raw.std(unbiased=False)
 expected=(production.expr_values(first,production.values(query,graph))-mean)/std;production.append_fragment(graph,first,mean,std)
 first_error=float(abs(production.ProgramFeatures(graph)(query)[:,-1]-expected).max());assert first_error<1e-12
 second=('mul',('ref',graph['features'][-1]),('recip',('ref',3)));raw=production.expr_values(second,production.values(probe,graph));mean=raw.mean();std=raw.std(unbiased=False)
 expected=(production.expr_values(second,production.values(query,graph))-mean)/std;production.append_fragment(graph,second,mean,std)
 nested_error=float(abs(production.ProgramFeatures(graph)(query)[:,-1]-expected).max());assert nested_error<1e-12
 # The QR screen excludes exact copies of the existing span in a full-rank fixture.
 existing=tensor(P[0]);qq=torch.linalg.qr(existing,mode='reduced').Q;duplicate=existing[:,2];fraction=float(((duplicate-qq@(qq.T@duplicate)).square().sum()/duplicate.square().sum()));assert fraction<1e-20
 production_checks=dict(batched_candidate_prediction_max_error=production_error,first_fragment_unseen_input_error=first_error,nested_fragment_unseen_input_error=nested_error,duplicate_unexplained_fraction=fraction,source_sha256=sha(Path(production.__file__)))
 out=dict(status='PASS TINY INDEPENDENT RIDGE AND PRODUCTION FEATURE-ADDITION FIXTURES; NO SEARCH OR CAPABILITY TEST' ,source_sha256=sha(Path(__file__)),ridge=ridge,records=records,production_checks=production_checks,duplicate_feature_counterexample=dict(single_effective_coefficient=before,duplicate_effective_coefficient=after,meaning='Same function span, changed effective prior penalty; reject as structure novelty.'),conditions=['Same sum-loss ridge convention before and after addition, including new column penalty.','Same fixed normalization and original columns; renormalizing previous columns changes the problem.','Intercept penalty must match: these fixtures penalize every coefficient equally.','Support labels fit coefficients; query labels can select structures only on declared training episodes. Heldout validation selects frozen choices; test labels never select.'],limitations='Small production prediction, nested-program and duplicate fixtures only; not a full search or dataflow execution. No data acquisition, model training, physics, feature search or test scoring.',wall_seconds=time.time()-start)
 (ROOT/'check_structure_acquisition_features_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
