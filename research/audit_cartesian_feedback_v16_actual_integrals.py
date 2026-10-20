#!/usr/bin/env python3
"""Independent-rule verification of actual recorded adaptive decisions."""
import json,hashlib,time,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_cartesian_feedback_v16_quadrature import integrate,impl

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',default='mixture');p.add_argument('--initial-all',action='store_true');a=p.parse_args()
 raw=Path('research/cartesian_feedback_v16_pilot_results.json').read_bytes();r=json.loads(raw);checks=[];start=time.time();seen=set()
 out=Path('research/audit_cartesian_feedback_v16_actual_integrals_'+a.prior+('_initialall' if a.initial_all else '_toplater')+'_results.json')
 with threadpool_limits(limits=1):
  for ei,e in enumerate(r['episodes']):
   if e['prior']!=a.prior or e['mode']!='dual':continue
   for pi,p in enumerate(e['phases']):
    if a.initial_all and pi!=0:continue
    if not a.initial_all and pi==0:continue
    d=p['design'];key=json.dumps(p['belief_before'],sort_keys=True)+str(e['sequence'])
    if key in seen:continue
    seen.add(key);b=impl.Belief(**{k:np.array(v) for k,v in p['belief_before'].items()})
    indices=list(range(len(d['values']))) if a.initial_all else list(np.argsort(d['values'])[:3])
    for index in indices:
     v=integrate(b,np.array(p['goal']),np.array(d['candidates'][index]),'gk21',1e-8,1e-7,False)
     v.update(episode=ei,phase=pi,sequence=e['sequence'],repeat=e['repeat'],branch=e['branch_evaluator_only'],index=int(index),original_value=d['values'][index],original_error=d['integration'][index]['estimated_error'],original_selected=d['selected'],difference=v['value']-d['values'][index]);checks.append(v)
     out.write_text(json.dumps(dict(input_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),scope='Independent gk21 finite-domain cubature using separately algebra-audited QP/filter, not independent controller implementation or rigorous integration certification.',prior=a.prior,initial_all=a.initial_all,checks=checks,wall_seconds=time.time()-start),indent=2)+'\n')
     print(a.prior,ei,pi,index,'diff',v['difference'],'value',v['value'],'err',v['estimated_error'],'secs',v['wall_seconds'],flush=True)
if __name__=='__main__':main()
