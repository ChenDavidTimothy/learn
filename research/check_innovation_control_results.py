#!/usr/bin/env python3
"""Bounded independent state/action scoring and selection check. No env helpers,
NN forwards, optimization or physics replay. Preserve earlier fixture results.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
HASH={};ERROR={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path(p):return Path(p) if Path(p).is_absolute() else WORK/p
def record(p):p=Path(p);HASH[str(p.relative_to(WORK))]=sha(p);return p
def read(p):return json.loads(record(p).read_text())
def ck(name,a,b,tol=1e-10):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape);e=float(abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.;ERROR[name]=max(ERROR.get(name,0.),e);assert e<=tol,(name,e,tol)
def sources(s):
 assert set(s)=={'innovation_control_plan.md','innovation_control_screen.py','innovation_control_env.py','error_transport_data.py'}
 for name,h in s.items():assert sha(record(ROOT/name))==h

def score(p,stored=None):
 with np.load(record(p),allow_pickle=False) as a:d={k:a[k] for k in a.files}
 z=d['states'];u=d['action'];goal=d['goals'];assert z.shape==(241,32,4) and u.shape==(240,32,2) and goal.shape==(32,3,2)
 assert np.isfinite(z).all() and np.isfinite(u).all() and abs(u).max()<=1 and np.isfinite(goal).all()
 target=np.repeat(goal.transpose(1,0,2),80,axis=0);delta=z[1:,:,:2]-target;error=np.arctan2(np.sin(delta),np.cos(delta));velocity=z[1:,:,2:]
 tick=np.mean(error*error,axis=2)+.1*np.mean(velocity*velocity,axis=2)+.01*np.mean(u*u,axis=2);ck('all_saved_costs',tick,d['cost'])
 precise=np.empty((32,3),bool)
 for phase in range(3):
  window=slice(phase*80+60,(phase+1)*80);precise[:,phase]=np.all(abs(error[window])<=.15,axis=(0,2))&np.all(abs(velocity[window])<=.15,axis=(0,2))
 ck('all_saved_precision',precise,d['precise'],0);ck('evaluator_goals',goal,d['evaluator_goals'],0);ck('evaluator_tail_counts',d['evaluator_tail_observations'],[20,20,20],0)
 body=tick.mean(0);metrics=dict(mean_cost=float(body.mean()),body_cost=body.tolist(),precise_fraction=float(precise.mean()),precise_goals=precise.astype(int).tolist())
 if stored:
  for name,x in metrics.items():ck('summary_'+name,x,stored[name])
 return metrics,d

def main():
 start=time.time();outp=ROOT/'check_innovation_control_screen_results.json';assert not outp.exists(),'refuse overwrite final result'
 result=read(ROOT/'innovation_control_results.json');selection_path=ROOT/'innovation_control_selection.json';selection=read(selection_path);sources(result['sources']);assert selection['sources']==result['sources'] and result['selection_sha256']==sha(selection_path) and selection['status']=='FROZEN BEFORE FRESH TEST'
 options={'moment':[],'gru':[]};valstreams=[];reports=[];allvalidation=[];trainints=valints=0;seen=set()
 for item in selection['reports']:
  directory=path(item['path']);rp=directory/'report.json';report=read(rp);assert item['report_sha256']==sha(rp) and report['status']=='COMPLETE EXPLORATORY FIT; TEST UNINSPECTED' and report['sources']==result['sources']
  kind=report['kind'];lr=report['lr'];assert kind in options and lr in(.0003,.001) and (kind,lr) not in seen;seen.add((kind,lr));assert report['seed']==20261021
  assert len(report['history'])==128 and [r['update'] for r in report['history']]==list(range(1,129)) and [r['update'] for r in report['candidates']]==[32,64,96,128]
  for row in report['history']:
   assert np.array(row['losses']).shape==(16,4) and np.isfinite(row['losses']).all() and np.isfinite(row['logstd']).all()
   tr=row['training'];assert len(tr['body_cost'])==64 and np.asarray(tr['precise_goals']).shape==(64,3)
   ck('recorded_training_bodymean',np.mean(tr['body_cost']),tr['mean_cost']);ck('recorded_training_precisionmean',np.mean(tr['precise_goals']),tr['precise_fraction'])
  assert report['training_interactions']==128*64*240 and report['validation_interactions']==4*32*240;trainints+=report['training_interactions'];valints+=report['validation_interactions']
  for candidate in report['candidates']:
   update=candidate['update'];wp=directory/candidate['weights'];assert wp.name==f'update_{update:03d}.pt';record(wp)
   state=torch.load(wp,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values()) and sum(v.numel() for v in state.values())==report['parameters']
   metric,d=score(directory/f'update_{update:03d}_validation.npz',candidate['validation']);valstreams.append(d)
   option=dict(path=str(Path(item['path'])/candidate['weights']),sha256=sha(wp),cost=metric['mean_cost'],update=update,lr=lr);options[kind].append(option)
   allvalidation.append(dict(kind=kind,lr=lr,update=update,mean_cost=metric['mean_cost'],precise_goals=int(np.sum(metric['precise_goals']))))
  reports.append(dict(kind=kind,lr=lr,parameters=report['parameters'],updates=128,training_interactions=report['training_interactions'],validation_interactions=report['validation_interactions'],wall_seconds=report['wall_seconds']))
 assert seen=={(k,lr) for k in options for lr in(.0003,.001)}
 for kind in options:
  # Recorded values differ only by scalar summation roundoff; exact provenance
  # fields and actual min location must agree.
  assert len(options[kind])==len(selection['choices'][kind])==8
  for computed,stored in zip(options[kind],selection['choices'][kind]):
   assert {k:v for k,v in computed.items() if k!='cost'}=={k:v for k,v in stored.items() if k!='cost'};ck('validation_option_cost',computed['cost'],stored['cost'])
  selected=min(options[kind],key=lambda x:x['cost']);assert {k:v for k,v in selected.items() if k!='cost'}=={k:v for k,v in selection['selected'][kind].items() if k!='cost'};ck('selected_validation_cost',selected['cost'],selection['selected'][kind]['cost'])
 adaptive=[]
 for row in selection['adaptive_choices']:
  assert row['rho'] in(.98,.995) and sha(path(row['archive']))==row['sha256'];metric,d=score(path(row['archive']),row['metrics']);valstreams.append(d);valints+=32*240;adaptive.append(dict(rho=row['rho'],mean_cost=metric['mean_cost'],precise_goals=int(np.sum(metric['precise_goals']))))
 assert len(adaptive)==2 and selection['selected']['rls']==min(selection['adaptive_choices'],key=lambda r:r['metrics']['mean_cost'])
 def pair(group,label):
  for d in group[1:]:
   for name in('evaluator_parameters','evaluator_input_matrix','goals'):ck(label+'_'+name,d[name],group[0][name],0)
   ck(label+'_initial_states',d['states'][0],group[0]['states'][0],0)
 pair(valstreams,'validation_paired');assert len(valstreams)==18
 final={};gates={};testints=0;failures=[]
 for split,threshold in [('test_id',.75),('test_ood',.5)]:
  final[split]={};group=[]
  for kind in('moment','gru','zero_moment','rls'):
   row=result['results'][split][kind];assert sha(path(row['archive']))==row['sha256'];metric,d=score(path(row['archive']),row);group.append(d);testints+=32*240;final[split][kind]=metric
   for body,phase in np.argwhere(~d['precise']):failures.append(dict(split=split,kind=kind,body=int(body),phase=int(phase)))
  pair(group,split+'_paired');row=final[split];gates[split]=dict(precision=row['moment']['precise_fraction']>=threshold,beats_gru=row['moment']['mean_cost']<=.9*row['gru']['mean_cost'],beats_rls=row['moment']['mean_cost']<=.9*row['rls']['mean_cost'],uses_moment=row['moment']['mean_cost']<=.9*row['zero_moment']['mean_cost'])
 assert result['gates']==gates and result['decision']==('CONTINUE TO CONFIRMATION' if all(all(r.values()) for r in gates.values()) else 'REJECT / RETIRE') and result['test_interactions']==testints
 assert trainints==7864320 and valints==138240 and testints==61440
 compact={split:{kind:dict(mean_cost=r['mean_cost'],precise_goals=int(np.sum(r['precise_goals'])),goal_windows=96) for kind,r in group.items()} for split,group in final.items()}
 output=dict(status='PASS BOUNDED INDEPENDENT SAVED STATE/ACTION SCORES, PAIRING, SELECTION AND RESOURCE CHECK',source_sha256=sha(Path(__file__)),input_sha256=HASH,scope=dict(learned_validation_archives=16,RLS_validation_archives=2,fresh_test_archives=8,all_archives=26,steps_per_archive=240,bodies_per_archive=32,all_scored_interactions=26*32*240,sustained_endpoint_windows=[list(range(61,81)),list(range(141,161)),list(range(221,241))],training_replays=0,neural_forward_passes=0,physical_integrations=0),maximum_absolute_errors=ERROR,fit_summaries=reports,learned_validation=allvalidation,RLS_validation=adaptive,resources=dict(training_interactions=trainints,validation_interactions=valints,test_interactions=testints,total_interactions=trainints+valints+testints,simulated_seconds=(trainints+valints+testints)*.05),fresh_summary=compact,gates=gates,decision=result['decision'],all_failed_test_goal_windows=failures,discrepancies=[],wall_seconds=time.time()-start,limitations=['Independent scoring from saved states/actions; physical dynamics not reintegrated and learned actions not independently regenerated.','Training logs and checkpoint counts/finite weights checked; PPO optimization not reproduced.','Pairing uses saved hidden body/actuator metadata solely for audit, not learner inputs.','Precision is all20 sampled endpoints per goal, not a continuous-time guarantee.','One seed/fixed family and finite PPO budget; failure rejects this formulation, not all correlation-memory or meta-RL methods.'])
 outp.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:output[k] for k in('status','scope','maximum_absolute_errors','resources','fresh_summary','gates','decision','wall_seconds')},indent=2))
if __name__=='__main__':main()
