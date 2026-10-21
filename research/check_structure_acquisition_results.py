#!/usr/bin/env python3
"""Bounded independent final structure-screen arithmetic/provenance checker.
No canonical driver imports, model fitting, NN forwards or physical simulation.
"""
from pathlib import Path
import ast,hashlib,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
METHODS=('meta','fit','frozen','neural','coda');BUDGETS=(0,4,16,64);LRS=(.001,.003)
ERROR={};HASHES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):p=Path(p);HASHES[p.name]=sha(p);return p
def read(p):return json.loads(record(p).read_text())
def ck(name,a,b,rtol=2e-12,atol=2e-13):
 a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape,(name,a.shape,b.shape)
 err=float(abs(a-b).max()) if a.size else 0.;ERROR[name]=max(ERROR.get(name,0.),err);assert np.allclose(a,b,rtol=rtol,atol=atol),(name,err)
def prefix(phase,method,lr):return ROOT/f'structure_acquisition_{phase}_{method}_lr{lr:g}_seed0'
def expected_sources(method,continued):
 p=[ROOT/'structure_acquisition_screen.py',ROOT/'structure_acquisition_plan.md',ROOT/'error_transport_screen_data.npz',ROOT/'error_transport_baselines.py',ROOT/'error_transport_screen.py']
 if continued:p.append(ROOT/'structure_acquisition_data.npz')
 if method in('programold','meta','fit','frozen'):
  phase={'programold':'old','frozen':'old'}.get(method,method);p.extend([ROOT/f'structure_acquisition_graph_{phase}.json',ROOT/'structure_acquisition_features.py'])
 return {x.name:sha(record(x)) for x in p}
def validation_choice(report,original=False):
 records=report['validation_records'];steps=(500,1000,1500,2000) if original or report['phase']=='old' else (2000,4000,6000,8000)
 assert tuple(r['step'] for r in records)==steps
 choices=[]
 for checkpoint in records:
  variants=checkpoint['validation'];expected_lrs=(.01,.03,.1) if report['method']=='coda' else (.03,)
  assert tuple(x['adaptation_lr'] for x in variants)==expected_lrs
  for alternative in variants:
   bs=alternative['body_scores']
   if original:
    assert set(bs)=={'4','16','64'} and all(len(bs[n])==8 for n in bs)
    criterion=float(np.mean([bs[n] for n in('4','16','64')]))
   else:
    expected_groups={'old':8} if report['phase']=='old' else {'old':8,'new':4}
    assert set(bs)==set(expected_groups);gm={}
    for group,size in expected_groups.items():
     assert set(bs[group])=={'4','16','64'} and all(len(bs[group][n])==size for n in bs[group]);assert np.isfinite([bs[group][n] for n in('4','16','64')]).all()
     gm[group]=float(np.mean([bs[group][n] for n in('4','16','64')]))
     ck('validation_group_mean',gm[group],alternative['group_means'][group])
    criterion=float(np.mean(list(gm.values())))
   ck('validation_equal_group_criterion',criterion,alternative['criterion'])
  chosen=min(variants,key=lambda a:a['criterion']);choices.append(dict(chosen,step=checkpoint['step']))
 best=min(choices,key=lambda a:a['criterion']);assert report['best_validation']==best
 return best

def fit_report(phase,method,lr):
 p=prefix(phase,method,lr);rp=Path(str(p)+'.json');wp=Path(str(p)+'.pt');r=read(rp);record(wp)
 assert r['status']=='COMPLETE TRAIN/VALIDATION; TEST NOT SCORED' and (r['phase'],r['method'],r['lr'],r['seed'])==(phase,method,lr,0)
 assert r['artifact_sha256']==expected_sources(method,phase!='old') and r['weights_sha256']==sha(wp)
 steps=2000 if phase=='old' else 8000;assert r['steps']==steps and len(r['all_training_losses'])==steps and np.isfinite(r['all_training_losses']).all()
 expected_seed=int(np.random.SeedSequence([20261020,18,42,int(phase!='old'),(('programold','neural') if phase=='old' else METHODS).index(method)]).generate_state(1)[0]);assert r['initialization_seed']==expected_seed and r['episode_seed_roles']==[20261020,18,41,int(phase!='old')]
 assert sum(r['support_budget_update_counts'].values())==steps and set(r['support_budget_update_counts'])==set(map(str,BUDGETS))
 assert r['training_body_draws']==steps*8 and r['training_query_transitions']==steps*8*32 and r['training_support_transitions']==sum(int(n)*k*8 for n,k in r['support_budget_update_counts'].items())
 choice=validation_choice(r)
 # Metadata only; do not instantiate or evaluate any network.
 state=torch.load(wp,map_location='cpu',weights_only=False)
 assert state['choice']==choice and state['artifact_sha256']==r['artifact_sha256'] and state['parent']==r['parent'] and state['method']==method and state['phase']==phase
 assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
 assert sum(v.numel() for k,v in state['state_dict'].items() if k!='noise_variance')==r['parameter_count']
 return dict(report=str(rp),report_sha256=sha(rp),weights=str(wp),weights_sha256=sha(wp),validation=choice,parameter_count=r['parameter_count']),r,state

def original_coda():
 options=[]
 for lr in LRS:
  rp=ROOT/f'error_transport_screen_coda_lr{lr:g}_seed0.json';wp=rp.with_suffix('.pt');r=read(rp);record(wp)
  assert r['status'].startswith('COMPLETE') and r['data_sha256']==sha(ROOT/'error_transport_screen_data.npz') and r['driver_sha256']==sha(ROOT/'error_transport_screen.py') and r['baseline_source_sha256']==sha(ROOT/'error_transport_baselines.py') and r['weights_sha256']==sha(wp)
  validation_choice(r,True)
  options.append(dict(report=str(rp),report_sha256=sha(rp),weights=str(wp),weights_sha256=sha(wp),validation=r['best_validation'],parameter_count=r['parameter_count']))
 return min(options,key=lambda x:x['validation']['criterion'])
def graph_checks():
 graphs={phase:read(ROOT/f'structure_acquisition_graph_{phase}.json') for phase in('old','meta','fit')};old=graphs['old'];summary={}
 for phase,g in graphs.items():
  plan=record(ROOT/'structure_acquisition_plan.md');current_plan=sha(plan)
  if phase=='old':
   prefix_text=plan.read_text().split('\nPrefit instrument correction:')[0];expected_plan=hashlib.sha256(prefix_text.encode()).hexdigest();assert expected_plan=='d5ddd872265215113bbb7160d21b015f4241a2a892d87ac4bf876cb6e7b0b131'
  else:expected_plan=current_plan
  assert g['phase']==phase and g['source_sha256']==sha(record(ROOT/'structure_acquisition_features.py')) and g['plan_sha256']==expected_plan and g['data_sha256']==sha(record(ROOT/'error_transport_screen_data.npz')) and g['search_ridge']==.1
  assert g['feature_count']==len(g['features']) and len(set(g['features']))==len(g['features']) and g['features'][:9]==list(range(9))
  initial=9 if phase=='old' else len(old['features']);target=32 if phase=='old' else initial+16
  assert initial<=len(g['features'])<=target and len(g['selection_records'])>=len(g['features'])-initial
  for i,n in enumerate(g['nodes']):
   if i==0:assert n==dict(op='constant')
   elif i<9:assert n==dict(op='input',index=i-1)
   else:assert n['op'] in('mul','sin','cos','abs','recip') and all(0<=j<i for j in n['args']) and np.isfinite([n['mean'],n['std']]).all() and n['std']>0
  if phase!='old':
   assert g['parent_sha256']==sha(ROOT/'structure_acquisition_graph_old.json') and g['acquisition_sha256']==sha(record(ROOT/'structure_acquisition_data.npz'))
   assert g['features'][:initial]==old['features'] and g['nodes'][:len(old['nodes'])]==old['nodes']
  reconstructed_nodes=[dict(n) for n in (g['nodes'][:9] if phase=='old' else old['nodes'])];reconstructed_features=list(range(9)) if phase=='old' else list(old['features'])
  for index,r in enumerate(g['selection_records']):
   assert r['feature_count_before']==initial+index and r['candidates']<=512 and 0<r['admissible']<=r['candidates']
   ck('graph_record_gain',r['baseline_loss']-r['selected_loss'],r['improvement']);assert np.isfinite([r['baseline_loss'],r['selected_loss'],r['mean'],r['std']]).all()
   e=ast.literal_eval(r['expression']);assert isinstance(e,tuple)
   if index<len(g['features'])-initial:
    assert r['improvement']>max(1e-9,1e-5*abs(r['baseline_loss']))
    def append(e):
     if e[0]=='ref':assert 0<=e[1]<len(reconstructed_nodes);return e[1]
     assert e[0] in('mul','sin','cos','abs','recip');args=[append(a) for a in e[1:]];reconstructed_nodes.append(dict(op=e[0],args=args,mean=0.,std=1.));return len(reconstructed_nodes)-1
    root=append(e);reconstructed_nodes[root].update(mean=float(r['mean']),std=float(r['std']));reconstructed_features.append(root)
  assert reconstructed_nodes==g['nodes'] and reconstructed_features==g['features']
  summary[phase]=dict(recorded_plan_sha256=g['plan_sha256'],current_plan_sha256=current_plan,old_plan_is_exact_prefix_before_documented_instrument_amendment=(phase=='old'),initial_features=initial,actual_features=len(g['features']),features_added=len(g['features'])-initial,retained_operations=len(g['nodes'])-9,status=g['status'],selection_records=len(g['selection_records']),wall_seconds=g['wall_seconds'])
 return summary

def data():
 with np.load(record(ROOT/'error_transport_screen_data.npz'),allow_pickle=False) as d:ox=d['x'];oy=d['y'];os=d['split']
 with np.load(record(ROOT/'structure_acquisition_data.npz'),allow_pickle=False) as d:nx=d['x'];ny=d['y'];ns=d['split']
 assert ox.shape==(56,8,96,8) and oy.shape==(56,8,96,2) and nx.shape==(36,8,96,8) and ny.shape==(36,8,96,2)
 assert np.array_equal(os,np.repeat(np.arange(4),[32,8,8,8])) and np.array_equal(ns,np.repeat(np.arange(5),[8,4,8,8,8]))
 scale=dict(x_mean=ox[:32].mean((0,1,2)),x_std=np.maximum(ox[:32].std((0,1,2)),.01),y_mean=oy[:32].mean((0,1,2)),y_std=np.maximum(oy[:32].std((0,1,2)),.01))
 report=read(ROOT/'structure_acquisition_data_report.json');assert report['ordinary_sha256']==sha(ROOT/'structure_acquisition_data.npz') and report['source_sha256']==sha(record(ROOT/'structure_acquisition_data.py')) and report['plan_sha256']==sha(ROOT/'structure_acquisition_plan.md')
 y=np.concatenate((oy,ny));return y,scale

def main():
 started=time.time();outpath=ROOT/'check_structure_acquisition_results.json';assert not outpath.exists(),'refuse overwrite'
 graphs=graph_checks();y,scale=data();parents={};fitrecords=[]
 for method in('programold','neural'):
  options=[]
  for lr in LRS:
   choice,r,state=fit_report('old',method,lr);assert r['parent'] is None
   for k in scale:ck('old_saved_train_only_scale_'+k,state['scale'][k],scale[k],rtol=0,atol=0)
   options.append(choice);fitrecords.append(dict(phase='old',method=method,lr=lr,selected_checkpoint=r['best_validation']['step'],criterion=r['best_validation']['criterion']))
  parents[method]=min(options,key=lambda x:x['validation']['criterion'])
 parents['coda']=original_coda();selected={}
 for method in METHODS:
  options=[]
  for lr in LRS:
   choice,r,state=fit_report('continue',method,lr);parent=parents['coda'] if method=='coda' else parents['neural'] if method=='neural' else parents['programold'];assert r['parent']==parent
   for k in scale:ck('continued_saved_train_only_scale_'+k,state['scale'][k],scale[k],rtol=0,atol=0)
   options.append(choice);fitrecords.append(dict(phase='continue',method=method,lr=lr,selected_checkpoint=r['best_validation']['step'],criterion=r['best_validation']['criterion']))
  selected[method]=min(options,key=lambda x:x['validation']['criterion'])
 result=read(ROOT/'structure_acquisition_results.json');selection=read(ROOT/'structure_acquisition_selection.json');assert result['status'].startswith('COMPLETE EXPLORATORY STRUCTURE') and selection['status']=='FROZEN VALIDATION-ONLY SELECTION; TEST NOT SCORED' and selection['choices']==result['choices']==selected and result['selection_sha256']==sha(ROOT/'structure_acquisition_selection.json')
 assert result['artifact_sha256']=={method:expected_sources(method,True) for method in METHODS}
 ap=record(ROOT/'structure_acquisition_predictions.npz');assert result['predictions_sha256']==sha(ap);means={};expectedkeys=set();score_checks=0
 with np.load(ap,allow_pickle=False) as archive:
  for method in METHODS:
   means[method]={}
   for split,label in((2,'new_id'),(3,'new_ood'),(4,'old_id')):
    ids=np.arange(68+8*(split-2),76+8*(split-2));target_raw=y[ids,1:].reshape(8,672,2);target=((target_raw-scale['y_mean'])/scale['y_std']).astype(np.float32)
    for wrong in(False,True):
     group=label+('_wrong_support' if wrong else '');row=result['results'][method][group];assert row['body_ids']==ids.tolist() and row['new_archive_body_ids']==(ids-56).tolist();means[method][group]={}
     for n in BUDGETS:
      key=f'{method}_{group}_support{n}';expectedkeys.add(key);prediction=archive[key];assert prediction.shape==(8,672,2) and np.isfinite(prediction).all()
      delta=prediction-target;standard=np.mean(delta.astype(np.float64)**2,axis=(1,2));raw=np.mean((delta*scale['y_std'])**2,axis=(1,2))
      ck('saved_standardized_prediction_MSE',standard,row['standardized_body_mse'][str(n)],rtol=5e-6,atol=5e-8);ck('saved_raw_prediction_MSE',raw,row['raw_body_mse'][str(n)],rtol=5e-12,atol=1e-10);means[method][group][str(n)]=float(standard.mean());score_checks+=8
      if n==0 and wrong:ck('zero_evidence_wrong_support_invariance',prediction,archive[f'{method}_{label}_support0'],rtol=0,atol=0)
  assert set(archive.files)==expectedkeys and len(expectedkeys)==120
 strongest=lambda s,n:min(means[m][s][str(n)] for m in('neural','coda'))
 gates={}
 for n in(16,64):
  other=64 if n==16 else 16;gates[str(n)]=dict(new_id_improves_10_percent_over_neural_and_frozen=means['meta']['new_id'][str(n)]<=.9*min(strongest('new_id',n),means['frozen']['new_id'][str(n)]),other_new_id_budget_within_5_percent=means['meta']['new_id'][str(other)]<=1.05*strongest('new_id',other),new_ood_within_5_percent_at_both=all(means['meta']['new_ood'][str(k)]<=1.05*strongest('new_ood',k) for k in(16,64)),retention_within_5_percent_at_both=all(means['meta']['old_id'][str(k)]<=1.05*strongest('old_id',k) for k in(16,64)),correct_support_beats_wrong=means['meta']['new_id'][str(n)]<means['meta']['new_id_wrong_support'][str(n)])
 survive=any(all(g.values()) for g in gates.values());criterion={str(n):means['meta']['new_id'][str(n)]<=.9*means['fit']['new_id'][str(n)] for n in(16,64)}
 decision=read(ROOT/'structure_acquisition_decision.json');assert decision['source_sha256']==sha(record(ROOT/'structure_acquisition_decision.py')) and decision['results_sha256']==sha(ROOT/'structure_acquisition_results.json')
 for m in METHODS:
  for group in means[m]:
   for n,v in means[m][group].items():ck('decision_mean',v,decision['means'][m][group][n],rtol=5e-6,atol=5e-8)
 assert decision['continuation_gates']==gates and decision['continuation_pass']==survive and decision['post_support_criterion_beats_fit_by_10_percent']==criterion
 out=dict(status='PASS BOUNDED INDEPENDENT RAW SCORE/VALIDATION SELECTION/GRAPH PROVENANCE CHECK',source_sha256=sha(Path(__file__)),input_sha256=HASHES,scope=dict(old_fits=4,continued_fits=10,original_CoDA_parent_fits=2,prediction_arrays=120,per_body_scores=score_checks,prediction_shape_per_array=[8,672,2],neural_forward_passes=0,model_fits=0,physical_rollouts=0),maximum_absolute_errors=ERROR,graph_checks=graphs,validation_choices=fitrecords,selected_parents=parents,selected_continued_fits=selected,independent_means=means,continuation_gates=gates,continuation_pass=survive,criterion_beats_fit_by_10_percent=criterion,wall_seconds=time.time()-started,limitations=['Saved raw test predictions independently scored, not regenerated through models.','Validation scores checked for arithmetic/equal-group weighting/checkpoint/LR selection and source/dataflow provenance, not independent forward reconstruction.','Weight metadata/scales/finite states inspected; optimizer history and learned global optima not certified.','Graph structure/prefix/records checked; feature search not rerun and semantic equivalence not proved.','Wrong-support body roll inferred from unchanged reviewed driver; predictions are not independently regenerated.','Single exploratory seed and one acquired corpus; fixed physical family and supplied grammar; no control/novelty/framework claim.'])
 outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','scope','maximum_absolute_errors','graph_checks','continuation_gates','continuation_pass','criterion_beats_fit_by_10_percent','wall_seconds')},indent=2))
if __name__=='__main__':main()
