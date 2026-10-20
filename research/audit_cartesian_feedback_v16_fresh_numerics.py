#!/usr/bin/env python3
"""Different-rule/partition audit with independently constructed raw-moment QP.
No canonical learning/controller module is imported. Prior artifact replay is
independent code; numerical estimates are not rigorous integral enclosures.
"""
from pathlib import Path
import argparse,ast,hashlib,json,time
import numpy as np
from scipy.integrate import cubature
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import audit_cartesian_feedback_v16_pilot as independent

ROOT=Path(__file__).resolve().parent
R=.0004;T=.1;DS=.1;WV=.25;WU=.01
D=np.c_[np.eye(2),2*np.eye(2),np.zeros((2,4))]
W=np.c_[np.zeros((2,2)),np.eye(2),np.zeros((2,4))]
P=np.kron(np.array([[1.5,.5]]),np.eye(2));V=np.tile(np.eye(2),(1,2))
A=np.array([independent.B_map(x) for x in P.T]);C=np.array([independent.B_map(x) for x in V.T])

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def core_conditioned_cost(belief,u,goal):
 w,m,cov=independent.predict(belief,u);S=cov[:,:2,:2]+R*np.eye(2);inv=np.linalg.inv(S);K=cov[:,:,:2]@inv
 pc=cov-K@cov[:,:2,:];pc=(pc+pc.transpose(0,2,1))/2
 logw=np.log(np.maximum(w,1e-300))-.5*np.linalg.slogdet(S)[1]-np.log(2*np.pi);g=goal/T**2;spent=WU*(u@u)
 def values(y):
  res=y[:,None]-m[None,:,:2];means=m[None]+np.einsum('kij,nkj->nki',K,res)
  logs=logw[None]-.5*np.einsum('nki,kij,nkj->nk',res,inv,res)
  weight=np.exp(logs-logsumexp(logs,axis=1)[:,None]);mean=np.einsum('nk,nki->ni',weight,means)
  second=np.einsum('nk,kij->nij',weight,pc)+np.einsum('nk,nki,nkj->nij',weight,means,means)
  Q=(np.einsum('ari,nij,brj->nab',A,second,A)+WV*np.einsum('ari,nij,brj->nab',C,second,C))/DS**2+WU*np.eye(4)
  b=(np.einsum('ari,nij,rj->na',A,second,D)-np.einsum('r,ari,ni->na',g,A,mean)+WV*np.einsum('ari,nij,rj->na',C,second,W))/DS**2
  constant=(np.einsum('ri,nij,rj->n',D,second,D)-2*np.einsum('r,ri,ni->n',g,D,mean)+g@g+WV*np.einsum('ri,nij,rj->n',W,second,W))/DS**2+spent
  x=independent.mathcheck.solve_boxes(Q,b);loss=np.einsum('ni,nij,nj->n',x,Q,x)+2*np.sum(b*x,axis=1)+constant
  assert loss.min()>-1e-7,loss.min()
  return np.maximum(loss,0.)
 # Check vectorized reference against the separately derived pointwise helper.
 for y in m[:min(len(m),3),:2]:
  point=independent.observe((w,m,cov),y);reference=independent.plan(point,goal,2,float(u@u))[1]
  assert abs(values(y[None])[0]-reference)<1e-8
 return w,m,S,values

def integral(belief,u,goal):
 w,m,S,values=core_conditioned_cost(belief,u,goal);points=0;parts=[];value=err=0.;start=time.time();edges=[-7.,-2.,2.,7.]
 for j,(weight,center,chol) in enumerate(zip(w,m[:,:2],np.linalg.cholesky(S))):
  if weight<1e-250:continue
  def f(z):
   nonlocal points
   points+=len(z);y=center+z@chol.T
   return values(y)*np.exp(-.5*np.sum(z*z,axis=1))/(2*np.pi)
  for lo0,hi0 in zip(edges[:-1],edges[1:]):
   for lo1,hi1 in zip(edges[:-1],edges[1:]):
    fit=cubature(f,[lo0,lo1],[hi0,hi1],rule='gk15',atol=1e-10/9,rtol=1e-9,max_subdivisions=2000)
    assert fit.status=='converged',(j,fit.status,fit.error,fit.estimate)
    value+=weight*float(fit.estimate);err+=weight*float(fit.error)
    parts.append(dict(component=j,weight=float(weight),bounds=[[lo0,lo1],[hi0,hi1]],estimate=float(fit.estimate),error=float(fit.error),subdivisions=fit.subdivisions,status=fit.status))
 return dict(value=float(value),estimated_error=float(err),points=points,parts=parts,wall_seconds=time.time()-start)

def source_guard(result):
 matches=[p for p in ROOT.glob('cartesian_feedback_v16_study*.py') if sha(p)==result['meta']['source_sha256']]
 assert len(matches)>=1,('missing frozen source',result['meta']['source_sha256'])
 return matches[0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--replicate',type=int,required=True);p.add_argument('--stage',choices=['initial','later','random'],default='initial');a=p.parse_args();rp=ROOT/f'cartesian_feedback_v16_rep{a.replicate}_results.json';raw=rp.read_bytes();r=json.loads(raw);source=source_guard(r);jobs=[]
 if a.stage=='initial':
  for prior in ('mixture','gaussian','matched','isotropic'):
   for seq in (0,1):
    ei=next(i for i,e in enumerate(r['episodes']) if e['prior']==prior and e['mode']=='dual' and e['sequence']==seq)
    jobs.append((ei,0,list(range(34))))
 elif a.stage=='later':
  for prior in ('mixture','gaussian','matched','isotropic'):
   for seq in (0,1):
    branch=(a.replicate+seq)%2;repeat=(a.replicate//2)%2
    ei=next(i for i,e in enumerate(r['episodes']) if e['prior']==prior and e['mode']=='dual' and e['sequence']==seq and e['branch_evaluator_only']==branch and e['repeat']==repeat)
    for phase in (2,5):jobs.append((ei,phase,list(map(int,np.argsort(r['episodes'][ei]['phases'][phase]['design']['values'])[:2]))))
 else:
  for prior in ('mixture','gaussian'):
   seq=a.replicate%2;branch=1-a.replicate%2;repeat=(a.replicate//2)%2
   ei=next(i for i,e in enumerate(r['episodes']) if e['prior']==prior and e['mode']=='random' and e['sequence']==seq and e['branch_evaluator_only']==branch and e['repeat']==repeat)
   jobs.append((ei,3,list(range(33))))
 out=ROOT/f'audit_cartesian_feedback_v16_fresh_numerics_rep{a.replicate}_{a.stage}_results.json';checks=[];menus=[];start=time.time()
 if out.exists():
  previous=json.loads(out.read_text())
  assert previous['input_sha256']==hashlib.sha256(raw).hexdigest() and previous['audit_source_sha256']==sha(Path(__file__))
  checks=previous['checks'];menus=previous['menus']
 done={(c['episode'],c['phase'],c['index']) for c in checks}
 def save():
  out.write_text(json.dumps(dict(replicate=a.replicate,stage=a.stage,input_sha256=hashlib.sha256(raw).hexdigest(),source=source.name,source_sha256=sha(source),audit_source_sha256=sha(Path(__file__)),independent_helper_sha256={p.name:sha(p) for p in [Path(independent.__file__),Path(independent.mathcheck.__file__)]},rule='gk15',partition_edges=[-7,-2,2,7],atol=1e-10,rtol=1e-9,scope='Independent raw8D-moment QP/filter,9-box different-rule quadrature; empirical numerical check, not analytic certification. No new physical interactions.',checks=checks,menus=menus,wall_seconds_current_execution=time.time()-start),indent=2)+'\n')
 with threadpool_limits(limits=1):
  for ei,pi,indices in jobs:
   e=r['episodes'][ei];ph=e['phases'][pi];d=ph['design'];b=independent.to_arrays(ph['belief_before']);goal=np.array(ph['goal'])
   for index in indices:
    if (ei,pi,index) in done:continue
    fit=integral(b,np.array(d['candidates'][index]),goal)
    refined=next((q for q in d['tighter_top_three'] if q['index']==index),None);record=refined['integration'] if refined else d['integration'][index]
    fit.update(episode=ei,phase=pi,index=index,prior=e['prior'],mode=e['mode'],sequence=e['sequence'],branch=e['branch_evaluator_only'],repeat=e['repeat'],original_selected=d['selected'],original_value=d['values'][index],original_estimated_error=record['estimated_error'],analytic_tail_upper_bound=record['analytic_tail_upper_bound'],difference=fit['value']-d['values'][index]);checks.append(fit);save()
   group=[c for c in checks if c['episode']==ei and c['phase']==pi];lookup={c['index']:c['value'] for c in group};best=min(lookup,key=lookup.get)
   menu=dict(episode=ei,phase=pi,prior=e['prior'],mode=e['mode'],sequence=e['sequence'],indices=indices,audited_best=best,original_selected=d['selected'],selected_in_checked_indices=d['selected'] in indices,complete_menu=(len(indices)==34))
   if len(indices)>=33:
    scores=[lookup[0],float(np.mean([lookup[i] for i in range(1,17)])),float(np.mean([lookup[i] for i in range(17,33)]))]
    menu.update(random_radius_scores=scores,original_random_radius_scores=d['random_radius_values'],audited_random_radius=[0.,.05,.1][int(np.argmin(scores))],original_random_radius=d['random_radius'])
   if not any(m['episode']==ei and m['phase']==pi for m in menus):menus.append(menu)
   save();print(a.replicate,a.stage,e['prior'],e['sequence'],pi,'integrals',len(checks),'maxdiff',max(abs(c['difference']) for c in group),'best',best,'oldbest',d['selected'],'seconds',time.time()-start,flush=True)
if __name__=='__main__':main()
