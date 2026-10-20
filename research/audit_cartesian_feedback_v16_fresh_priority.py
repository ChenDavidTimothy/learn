#!/usr/bin/env python3
"""Balanced prioritized ACTUAL fixture calls to independent nine-box gk15 audit.
Can audit completed episodes in a partial append-only result, then verify unchanged
fixtures and update the input hash when the full immutable result becomes available.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_feedback_v16_fresh_numerics as audit
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--replicate',type=int,required=True);p.add_argument('--prior',default='all');a=p.parse_args()
 rp=ROOT/f'cartesian_feedback_v16_rep{a.replicate}_results.json';raw=rp.read_bytes();r=json.loads(raw);source=audit.source_guard(r)
 out=ROOT/f'audit_cartesian_feedback_v16_fresh_priority_rep{a.replicate}_results.json';checks=[]
 if out.exists():
  old=json.loads(out.read_text());assert old['audit_source_sha256']==audit.sha(Path(__file__));assert old['integral_helper_sha256']==audit.sha(Path(audit.__file__));checks=old['checks']
 known={(c['episode'],c['phase'],c['index']):c for c in checks};missing=[];start=time.time()
 def save():
  out.write_text(json.dumps(dict(replicate=a.replicate,input_sha256=hashlib.sha256(raw).hexdigest(),input_complete=len(r['episodes'])==80,input_episodes=len(r['episodes']),source=source.name,source_sha256=audit.sha(source),audit_source_sha256=audit.sha(Path(__file__)),integral_helper_sha256=audit.sha(Path(audit.__file__)),independent_rawmoment_helper_sha256=audit.sha(Path(audit.independent.__file__)),rule='gk15',partition_edges=[-7,-2,2,7],atol=1e-10,rtol=1e-9,scope='Initialtop3 +laterselected/runner phases2/5; fixedbranch=(rep+seq)%2 and repeat=(rep//2)%2; no performance-based fixture selection.',checks=checks,missing_jobs=missing,wall_seconds_current_execution=time.time()-start),indent=2)+'\n')
 with threadpool_limits(limits=1):
  for prior in ('mixture','gaussian','matched','isotropic'):
   if a.prior!='all' and prior!=a.prior:continue
   for seq in (0,1):
    for phase in (0,2,5):
     options=[(i,e) for i,e in enumerate(r['episodes']) if e['prior']==prior and e['mode']=='dual' and e['sequence']==seq and (phase==0 or(e['branch_evaluator_only']==(a.replicate+seq)%2 and e['repeat']==(a.replicate//2)%2))]
     if not options:missing.append(dict(prior=prior,sequence=seq,phase=phase));continue
     ei,e=options[0];ph=e['phases'][phase];d=ph['design'];indices=list(map(int,np.argsort(d['values'])[:3 if phase==0 else 2]));b=audit.independent.to_arrays(ph['belief_before']);goal=np.array(ph['goal'])
     for index in indices:
      refined=next((q for q in d['tighter_top_three'] if q['index']==index),None);original_record=refined['integration'] if refined else d['integration'][index]
      fixture=dict(belief=ph['belief_before'],goal=ph['goal'],action=d['candidates'][index],value=d['values'][index],integration=original_record)
      fingerprint=hashlib.sha256(json.dumps(fixture,sort_keys=True).encode()).hexdigest();key=(ei,phase,index)
      if key in known:assert known[key]['fixture_sha256']==fingerprint;continue
      fit=audit.integral(b,np.array(fixture['action']),goal);fit.update(episode=ei,phase=phase,index=index,prior=prior,sequence=seq,branch=e['branch_evaluator_only'],repeat=e['repeat'],original_selected=d['selected'],original_value=fixture['value'],original_estimated_error=original_record['estimated_error'],analytic_tail_upper_bound=original_record['analytic_tail_upper_bound'],difference=fit['value']-fixture['value'],fixture_sha256=fingerprint,fixture=fixture);checks.append(fit);known[key]=fit;save()
     group=[c for c in checks if c['episode']==ei and c['phase']==phase];print(a.replicate,prior,seq,phase,'n',len(group),'maxdiff',max(abs(c['difference']) for c in group),flush=True)
 save()
if __name__=='__main__':main()
