#!/usr/bin/env python3
"""Independent fork physics, NumPy network/SVD-local forecasts and contrasts.
No canonical learner, local predictor, acquisition or mechanics module imported.
Uses frozen fit parameters; does not claim independent training optimization.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as mechanics
ROOT=Path(__file__).resolve().parent
ERROR={};BANK=np.array([[0.,0.],[.4,0.],[-.4,0.],[0.,.4],[0.,-.4]]);TIMES=(16,40,120,240)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-9):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def context(o,a):
 t=len(o)-1;lags=np.arange(1,17);velocity=(o[t]-o[t-lags])/(.1*lags[:,None])
 return np.concatenate((o[t],velocity.ravel(),a[-16:].ravel()))

def main():
 start=time.time();rp=ROOT/'cartesian_stream_v17_rep0_forks.json';fp=ROOT/'cartesian_stream_v17_rep0_fit.json';wp=ROOT/'cartesian_stream_v17_rep0_weights.npz';op=ROOT/'cartesian_stream_v17_rep0_ordinary.npz';ep=ROOT/'cartesian_stream_v17_rep0_evaluator.npz'
 result=json.loads(rp.read_text());fit=json.loads(fp.read_text());acq=json.loads((ROOT/'cartesian_stream_v17_rep0_acquisition.json').read_text());src=ROOT/'cartesian_stream_v17_forks.py';learn=ROOT/'cartesian_stream_v17_learn.py';local=ROOT/'cartesian_stream_v17_local.py';design=ROOT/'cartesian_stream_v17_learning_design.md'
 assert sha(src)==result['source_sha256'] and sha(learn)==fit['source_sha256']==result['learning_source_sha256'] and sha(local)==fit['local_source_sha256']
 assert sha(fp)==result['fit_sha256'] and sha(wp)==fit['weights_sha256'];assert sha(op)==result['ordinary_sha256']==fit['ordinary_sha256'] and sha(ep)==result['evaluator_sha256']
 assert sha(design)==fit['design_sha256']==result['design_sha256'];assert all(sha(ROOT/k)==h for k,h in acq['artifacts'].items())
 with np.load(op) as d:data={k:d[k] for k in d.files}
 with np.load(ep) as d:ev={k:d[k] for k in d.files}
 with np.load(wp) as d:weights={k:d[k] for k in d.files}
 o=data['observed'];a=data['commands'];train=np.flatnonzero(data['split']==0);test=np.flatnonzero(data['split']==2)
 assert len(train)==24 and len(test)==8;X=[];Y=[];origins=[]
 for i in train:
  for t in range(16,256,2):
   X.append(context(o[i,:t+1],a[i,:t]));carry=np.stack((2*o[i,t]-o[i,t-1],3*o[i,t]-2*o[i,t-1]));Y.append((o[i,t+1:t+3]-carry).ravel());origins.append((int(i),t))
 X=np.array(X);Y=np.array(Y);origins=np.array(origins);s=fit['scale'];cx=np.array(s['context_mean']);cs=np.array(s['context_std']);ym=np.array(s['target_mean']);ys=np.array(s['target_std'])
 ck('training_context_mean',X.mean(0),cx,1e-12);ck('training_context_std',np.maximum(X.std(0),.01),cs,1e-12);ck('training_target_mean',Y.mean(0),ym,1e-12);ck('training_target_std',np.maximum(Y.std(0),.001),ys,1e-12);ck('training_origins',origins,fit['training_origins'],1e-15)
 X=(X-cx)/cs;Y=(Y-ym)/ys;U=np.array([a[i,t]/.5 for i,t in origins]);boot=[]
 for member in range(3):
  path=[20261020,17,660,0,0,member];rng=np.random.default_rng(np.random.SeedSequence(path));ids=rng.choice(train,24,replace=True);rows=np.concatenate([np.flatnonzero(origins[:,0]==i) for i in ids]);ck('bootstrap_ids',ids,fit['bootstrap_episode_ids'][member],1e-15);ck('bootstrap_rows',rows,fit['bootstrap_rows'][member],1e-15);boot.append(rows)
 ni=fit['selected']['nonlinear'];li=fit['selected']['local'];assert ni==int(np.argmin([r['validation']['member_coordinate_mse'] for r in fit['nonlinear_candidates']])) and li==int(np.argmin([r['validation']['member_coordinate_mse'] for r in fit['local_candidates']]))
 assert result['selected']==fit['selected'];localrow=fit['local_candidates'][li];k=localrow['neighbors'];ridge=localrow['ridge'];assert k in(128,512) and ridge in(.001,.1)
 def predict_numpy(kind,history,commands):
  query=(context(history,commands)-cx)/cs;carry=np.stack((2*history[-1]-history[-2],3*history[-1]-2*history[-2]));members=[]
  if kind=='nonlinear':
   for member in range(3):
    inp=np.c_[np.repeat(query[None],5,axis=0),BANK/.5]
    for layer in(0,2,4):
     inp=inp@weights[f'nn{ni}_member{member}_layers.{layer}.weight'].T+weights[f'nn{ni}_member{member}_layers.{layer}.bias']
     if layer!=4:inp=np.tanh(inp)
    members.append(carry[None]+(ym+ys*inp).reshape(5,2,2))
  else:
   for rows in boot:
    centered=X[rows]-query;d2=np.sum(centered[:,:2]**2,axis=1)/2+np.sum(centered[:,2:34]**2,axis=1)/32+np.sum(centered[:,34:]**2,axis=1)/32
    indices=np.lexsort((np.arange(len(d2)),d2))[:min(k,len(d2))];weight=np.exp(-d2[indices]/max(d2[indices[-1]],1e-12));weight/=weight.sum()
    design=np.c_[np.ones(len(indices)),centered[indices],U[rows][indices]];penalty=np.sqrt(ridge)*np.eye(69);penalty[0,0]=0
    augmented=np.r_[np.sqrt(weight[:,None])*design,penalty];targets=np.r_[np.sqrt(weight[:,None])*Y[rows][indices],np.zeros((69,4))]
    coef=np.linalg.lstsq(augmented,targets,rcond=None)[0]
    correction=coef[0]+(BANK/.5)@coef[-2:]
    members.append(carry[None]+(ym+ys*correction).reshape(5,2,2))
  return np.stack(members,axis=1)
 rows=result['contexts'];assert len(rows)==32;ck('command_bank',result['command_bank'],BANK,1e-15);assert tuple(result['time_indices'])==TIMES
 expectedkeys=[(int(i),t) for i in test for t in TIMES];assert [(r['trajectory'],r['time_index']) for r in rows]==expectedkeys
 actual=[];storedtruth=[];counter=0;predictions={k:[] for k in ('nonlinear','local')}
 starts=[]
 for row in rows:
  i=row['trajectory'];t=row['time_index'];starts.append(np.repeat(ev['states'][i,t][None],5,axis=0))
  for kind in predictions:
   independent=predict_numpy(kind,o[i,:t+1],a[i,:t]);stored=np.array(row['predicted_positions_by_method'][kind]);ck('NumPy_independent_'+kind+'_forecasts',independent,stored,2e-10);predictions[kind].append(independent)
  assert np.array(row['fork_states_evaluator_only']).shape==(5,2,4) and np.array(row['plant_nfev']).shape==(5,2)
  counter+=int(np.sum(row['plant_nfev']));storedtruth.append(row['actual_positions_evaluator_only'])
 z=np.array(starts).reshape(-1,4);u=np.tile(BANK,(32,1));states=[]
 for sample in range(2):
  z=mechanics.advance(z,u,steps=128);states.append(z.copy())
 physical=np.stack(states,axis=1).reshape(32,5,2,4);truth=mechanics.endpoint(physical[:,:,:,:2]);storedstates=np.array([r['fork_states_evaluator_only'] for r in rows]);ck('all_Hamiltonian_fork_states',physical,storedstates,1e-9);ck('all_Hamiltonian_fork_cartesian_positions',truth,storedtruth,1e-9)
 radau=[]
 for context_index in range(0,32,3):
  for command_index in(0,1,4):
   i,t=expectedkeys[context_index];z=ev['states'][i,t].copy()
   for sample in range(2):
    z=mechanics.radau(z,BANK[command_index]);error=float(np.max(abs(z-storedstates[context_index,command_index,sample])));assert error<1e-9;radau.append(error)
 # Reconstruct every summary from stored arrays (independently replayed above).
 truth=np.array(storedtruth);contrasts=truth[:,1:]-truth[:,:1];signal=float(np.mean(contrasts**2));summaries={}
 for kind in predictions:
  pred=np.array([r['predicted_positions_by_method'][kind] for r in rows]);mean=pred.mean(2);dc=mean[:,1:]-mean[:,:1];error=dc-contrasts
  summary=dict(ensemble_mean_absolute_coordinate_mse=float(np.mean((mean-truth)**2)),member_absolute_coordinate_mse=float(np.mean((pred-truth[:,:,None])**2)),ensemble_mean_contrast_coordinate_mse=float(np.mean(error**2)),contrast_coordinate_mse_by_future_sample=np.mean(error**2,axis=(0,1,3)).tolist(),zero_contrast_coordinate_mse=signal,contrast_mse_over_zero_contrast=float(np.mean(error**2)/signal),mean_true_contrast_norm_by_future_sample=np.mean(np.linalg.norm(contrasts,axis=3),axis=(0,1)).tolist(),mean_predicted_contrast_norm_by_future_sample=np.mean(np.linalg.norm(dc,axis=3),axis=(0,1)).tolist())
  for key,val in summary.items():ck('summary_'+kind+'_'+key,val,result['summary'][kind][key],1e-12)
  summaries[kind]=summary
 resources=dict(contexts=32,counterfactual_preparations=160,plant_flows=320,privileged_physical_seconds=32.,plant_rhs_evaluations=counter,ordinary_new_experience=0,sensor_draws=0);assert resources==result['resources']
 artifacts=[rp,fp,wp,op,ep,src,learn,local,design,ROOT/'cartesian_stream_v17_rep0_acquisition.json',ROOT/'cartesian_stream_v17_acquire.py',ROOT/'coupled_context_v14_pilot.py',Path(mechanics.__file__)]
 out=dict(status='PASS — independent heldoutfork physics/NumPy-SVDforecast/contrast artifact replay; not control or a release',artifact_sha256={p.name:sha(p) for p in artifacts},audit_source_sha256=sha(Path(__file__)),maximum_reconstruction_errors=ERROR,summary=summaries,resources=resources,independent_Radau_interval_checks=len(radau),maximum_Radau_state_error=max(radau),limitations=['Frozen fit weights/selectedhyperparameters reused; this is independent forecast reconstruction, not independent training optimization.','All320fork flows checked with independent Hamiltonian code;66Radau intervals are a separate adaptive solver check.','32contexts derive from8fixedtest trajectories, not32independentbodies.','Fork truth/state used solely for evaluator replay; no ordinary fitting/modelselection/update.','Conditional action contrasts and absolute prediction scores do not establish useful sustained control or universal learning.'],wall_seconds=time.time()-start)
 (ROOT/'audit_cartesian_stream_v17_forks_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','maximum_reconstruction_errors','summary','resources','maximum_Radau_state_error','wall_seconds')}))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
