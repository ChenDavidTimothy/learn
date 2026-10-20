#!/usr/bin/env python3
"""Numerical audit: independently selected cubature rule on ACTUAL frozen beliefs.
Uses previously algebra-audited posterior/QP; this is quadrature verification,
not an independently implemented dynamics/control calculation.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
from threadpoolctl import threadpool_limits
import cartesian_feedback_v16 as impl

def integrate(belief,goal,u,rule,atol,rtol,split):
    predicted=belief.predict(u);S=predicted.covs[:,:2,:2]+impl.R*np.eye(2);chols=np.linalg.cholesky(S)
    value=error=0.;points=0;parts=[];start=time.time()
    cuts=[-7.,0.,7.] if split else [-7.,7.]
    for j,(center,chol,weight) in enumerate(zip(predicted.means[:,:2],chols,predicted.weights)):
        if weight<1e-250:continue
        def integrand(z):
            nonlocal points
            points+=len(z);ys=center+z@chol.T;weights,means,covs,_,_=predicted.posterior_stats(ys)
            _,loss,_=impl.future_qp(weights,means,covs,goal,2,float(u@u))
            return np.maximum(loss,0)*np.exp(-.5*np.sum(z*z,axis=1))/(2*np.pi)
        for a,b in zip(cuts[:-1],cuts[1:]):
            for c,d in zip(cuts[:-1],cuts[1:]):
                fit=cubature(integrand,[a,c],[b,d],rule=rule,atol=atol/(len(cuts)-1)**2,rtol=rtol,max_subdivisions=4000)
                value+=weight*float(fit.estimate);error+=weight*float(fit.error)
                parts.append(dict(component=j,weight=float(weight),bounds=[[a,c],[b,d]],estimate=float(fit.estimate),error=float(fit.error),subdivisions=fit.subdivisions,status=fit.status))
    return dict(value=value,estimated_error=error,points=points,parts=parts,wall_seconds=time.time()-start,rule=rule,atol=atol,rtol=rtol,split=split)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',default='mixture');p.add_argument('--sequence',type=int,default=0);p.add_argument('--indices',default='20,19');p.add_argument('--rule',default='gk21');p.add_argument('--atol',type=float,default=1e-8);p.add_argument('--rtol',type=float,default=1e-7);p.add_argument('--split',action='store_true');p.add_argument('--phase',type=int,default=0);p.add_argument('--episode',type=int);p.add_argument('--input',default='research/cartesian_feedback_v16_pilot_results.json');a=p.parse_args()
    path=Path(a.input);raw=path.read_bytes();r=json.loads(raw)
    episode=r['episodes'][a.episode] if a.episode is not None else next(e for e in r['episodes'] if e['prior']==a.prior and e['sequence']==a.sequence and e['mode']=='dual')
    phase=episode['phases'][a.phase];belief=impl.Belief(**{k:np.array(v) for k,v in phase['belief_before'].items()});d=phase['design'];results=[]
    out=Path('research/audit_cartesian_feedback_v16_quadrature_'+a.prior+'_s'+str(a.sequence)+'_e'+str(a.episode)+'_p'+str(a.phase)+'_'+a.rule+('_split' if a.split else '')+'_results.json')
    with threadpool_limits(limits=1):
        for index in map(int,a.indices.split(',')):
            v=integrate(belief,np.array(phase['goal']),np.array(d['candidates'][index]),a.rule,a.atol,a.rtol,a.split)
            v.update(index=index,original_value=d['values'][index],original_record=d['integration'][index],difference=v['value']-d['values'][index]);results.append(v)
            out.write_text(json.dumps(dict(input_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),prior=a.prior,sequence=a.sequence,episode=a.episode,phase=a.phase,original_selected=d['selected'],checks=results),indent=2)+'\n')
            print(a.prior,a.sequence,a.phase,index,v['value'],'diff',v['difference'],'err',v['estimated_error'],'seconds',v['wall_seconds'],flush=True)
if __name__=='__main__':main()
