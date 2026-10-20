#!/usr/bin/env python3
"""Finite independent artifact audit of explicitly POSTHOC ordinary action scores.
No canonical model/local predictor, physical evaluator, fork or control imports.
All NN queries via NumPy; balanced384local fits via augmented SVD; remaining
storedlocal queries are not individually certified. No candidate reselection.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-11):
 err=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),err);assert err<tol,(name,err,tol)
def ctx(observations,commands):
 t=len(observations)-1;lag=np.arange(1,17)
 return np.r_[observations[t],((observations[t]-observations[t-lag])/(lag[:,None]*.1)).ravel(),commands[-16:].ravel()]
def stats(values):
 values=list(map(float,values));m=math.fsum(values)/len(values);se=math.sqrt(math.fsum((v-m)**2 for v in values)/(len(values)*(len(values)-1)))
 return dict(mean=m,trajectory_se=se,trajectory_values=values)

def main():
 start=time.time();rp=ROOT/'cartesian_stream_v17_rep0_action_risk_posthoc.json';ap=ROOT/'cartesian_stream_v17_rep0_action_risk_posthoc_arrays.npz';op=ROOT/'cartesian_stream_v17_rep0_ordinary.npz';fp=ROOT/'cartesian_stream_v17_rep0_fit.json';wp=ROOT/'cartesian_stream_v17_rep0_weights.npz';src=ROOT/'cartesian_stream_v17_action_risk.py';design=ROOT/'cartesian_stream_v17_action_risk_posthoc_design.md';learn=ROOT/'cartesian_stream_v17_learn.py';local=ROOT/'cartesian_stream_v17_local.py'
 r=json.loads(rp.read_text());fit=json.loads(fp.read_text());assert sha(src)==r['source_sha256'] and sha(design)==r['design_sha256'] and sha(ap)==r['arrays_sha256']
 assert sha(op)==r['ordinary_sha256']==fit['ordinary_sha256'] and sha(fp)==r['fit_sha256'] and sha(wp)==r['weights_sha256']==fit['weights_sha256'];assert sha(learn)==fit['source_sha256'] and sha(local)==fit['local_source_sha256']
 with np.load(op) as f:d={k:f[k] for k in f.files}
 with np.load(ap) as f:saved={k:f[k] for k in f.files}
 with np.load(wp) as f:w={k:f[k] for k in f.files}
 train=np.flatnonzero(d['split']==0);val=np.flatnonzero(d['split']==1);assert len(train)==24 and len(val)==8
 def make_rows(ids):
  X=[];U=[];Y=[];origins=[]
  for i in ids:
   for t in range(16,256,2):
    ob=d['observed'][i];ac=d['commands'][i];X.append(ctx(ob[:t+1],ac[:t]));U.append(ac[t]/.5);carry=np.stack((2*ob[t]-ob[t-1],3*ob[t]-2*ob[t-1]));Y.append((ob[t+1:t+3]-carry).ravel());origins.append([int(i),t])
  return np.array(X),np.array(U),np.array(Y),np.array(origins)
 X,U,Y,training_origins=make_rows(train);VX,VU,VY,origins=make_rows(val);assert X.shape==(2880,66) and VX.shape==(960,66)
 ck('training_origins',training_origins,fit['training_origins'],1e-15);ck('validation_origins',origins,saved['origins'],1e-15);ck('validation_target_corrections',VY,saved['target_corrections'],2e-15);ck('actual_validation_commands',VU*.5,saved['actual_commands'],1e-15)
 scale={k:np.array(v) for k,v in fit['scale'].items()};ck('trainingonly_context_mean',X.mean(0),scale['context_mean'],1e-12);ck('trainingonly_context_std',np.maximum(X.std(0),.01),scale['context_std'],1e-12);ck('trainingonly_target_mean',Y.mean(0),scale['target_mean'],1e-12);ck('trainingonly_target_std',np.maximum(Y.std(0),.001),scale['target_std'],1e-12)
 X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std'];VX=(VX-scale['context_mean'])/scale['context_std']
 aux=np.zeros((960,8,2));seedpaths=[]
 for i in val:
  mask=origins[:,0]==i;path=[20261020,17,680,0,1,int(d['trajectory_index'][i])];aux[mask]=np.random.default_rng(np.random.SeedSequence(path)).uniform(-.5,.5,(120,8,2));seedpaths.append(path)
 ck('all_auxiliary_rng',aux,saved['auxiliary_commands'],1e-15);assert seedpaths==r['auxiliary_seed_paths'];assert len(set(map(tuple,seedpaths)))==8
 proposed=np.concatenate((VU[:,None],aux/.5,np.zeros((960,1,2))),axis=1);pred=saved['predicted_corrections'];assert pred.shape==(8,3,960,10,4) and np.isfinite(pred).all()
 labels=[f'nonlinear_{i}' for i in range(4)]+[f'local_{i}' for i in range(4)];assert labels==r['candidate_order']
 input=np.concatenate((np.repeat(VX[:,None],10,axis=1),proposed),axis=2).reshape(-1,68)
 for candidate in range(4):
  for member in range(3):
   answer=input.copy()
   for layer in(0,2,4):
    answer=answer@w[f'nn{candidate}_member{member}_layers.{layer}.weight'].T+w[f'nn{candidate}_member{member}_layers.{layer}.bias']
    if layer!=4:answer=np.tanh(answer)
   expected=scale['target_mean']+scale['target_std']*answer.reshape(960,10,4)
   ck('ALL_independent_NumPy_NN_query_corrections',expected,pred[candidate,member],2e-10)
 boot=[]
 for member in range(3):
  ids=np.random.default_rng(np.random.SeedSequence([20261020,17,660,0,0,member])).choice(train,24,replace=True);rows=np.concatenate([np.flatnonzero(training_origins[:,0]==i) for i in ids]);ck('bootstrap_training_rows',rows,fit['bootstrap_rows'][member],1e-15);boot.append(rows)
 configs=[(128,.001),(128,.1),(512,.001),(512,.1)];subset=np.flatnonzero(np.isin(origins[:,1],[16,40,120,240]));assert len(subset)==32;local_checks=0
 for candidate,(neighbors,ridge) in enumerate(configs):
  for member,rows in enumerate(boot):
   for qi in subset:
    centered=X[rows]-VX[qi];distance=np.sum(centered[:,:2]**2,axis=1)/2+np.sum(centered[:,2:34]**2,axis=1)/32+np.sum(centered[:,34:]**2,axis=1)/32;index=np.lexsort((np.arange(len(distance)),distance))[:neighbors];weight=np.exp(-distance[index]/max(float(distance[index[-1]]),1e-12));weight/=weight.sum();A=np.c_[np.ones(len(index)),centered[index],U[rows][index]];penalty=np.sqrt(ridge)*np.eye(69);penalty[0,0]=0
    B=np.r_[np.sqrt(weight[:,None])*A,penalty];target=np.r_[np.sqrt(weight[:,None])*Y[rows][index],np.zeros((69,4))];coef=np.linalg.lstsq(B,target,rcond=None)[0];expected=scale['target_mean']+scale['target_std']*(coef[0]+proposed[qi]@coef[-2:]);ck('BALANCED384_independent_SVD_local_fits',expected,pred[4+candidate,member,qi],2e-10);local_checks+=1
 assert local_checks==384
 baseline=pred[:4,:,:,9,:].mean(axis=(0,1));ck('common_all12NN_zero_command_baseline',baseline,saved['common_drift_baseline'],1e-15)
 baselines={'zero':np.zeros((960,4)),'all_twelve_nonlinear_zero_command':baseline};summary={};cluster_scores={};ordinary=[]
 def clusters(values):return np.array([math.fsum(map(float,values[origins[:,0]==i]))/120 for i in val])
 for ci,label in enumerate(labels):
  actual=pred[ci,:,:,0,:];delta=actual[:,:,None,:]-pred[ci,:,:,1:9,:]
  mse=np.sum((actual-VY[None])**2,axis=(0,2))/12;ordinary.append(mse);ck('all_ordinary_MSE_rows',mse,saved['ordinary_member_mse_by_row'][ci],1e-13);score_summary=dict(ordinary_one_block_member_coordinate_mse=stats(clusters(mse)),action_scores={})
  for name,b in baselines.items():
   residual=VY-b;square=.5*np.einsum('mrjq,mrjq->mrj',delta,delta)/4;cross=2*np.einsum('rq,mrjq->mrj',residual,delta)/4;perrow=np.sum(square-cross,axis=(0,2))/24
   ens=np.sum(delta,axis=0)/3;ens_score=np.sum(.5*np.einsum('rjq,rjq->rj',ens,ens)/4-2*np.einsum('rq,rjq->rj',residual,ens)/4,axis=1)/8
   ck('all_member_score_rows_'+name,perrow,saved['member_score_'+name][ci],1e-13);ck('all_ensemble_score_rows_'+name,ens_score,saved['ensemble_score_'+name][ci],1e-13)
   cm=clusters(perrow);ce=clusters(ens_score);cluster_scores.setdefault(name,[]).append(cm);score_summary['action_scores'][name]=dict(member_score=stats(cm),ensemble_mean_score=stats(ce))
  for key,value in score_summary['ordinary_one_block_member_coordinate_mse'].items():ck('ordinary_clusterstats_'+key,value,r['summaries'][label]['ordinary_one_block_member_coordinate_mse'][key],1e-13)
  for name in baselines:
   for kind in('member_score','ensemble_mean_score'):
    for key,value in score_summary['action_scores'][name][kind].items():ck('action_clusterstats_'+key,value,r['summaries'][label]['action_scores'][name][kind][key],1e-13)
  summary[label]=score_summary
 paired={}
 for name,values in cluster_scores.items():
  values=np.array(values);means=np.zeros((8,8));se=np.zeros((8,8))
  for i in range(8):
   for j in range(8):
    q=stats(values[i]-values[j]);means[i,j]=q['mean'];se[i,j]=q['trajectory_se']
  ck('paired_matrix_means_'+name,means,r['paired_member_scores'][name]['mean_candidate_row_minus_column'],1e-13);ck('paired_matrix_SE_'+name,se,r['paired_member_scores'][name]['trajectory_se'],1e-13);paired[name]=dict(mean_candidate_row_minus_column=means.tolist(),trajectory_se=se.tolist())
 resources=dict(validation_trajectories=8,validation_rows=960,virtual_auxiliary_commands=7680,candidate_member_block_queries=230400,local_context_member_fits=11520,new_physical_experience=0);assert resources==r['resources']
 artifacts=[rp,ap,op,fp,wp,src,design,learn,local];out=dict(status='PASS — finite independent POSTHOC action-risk artifact audit; no reselection or null-win claim',artifact_sha256={p.name:sha(p) for p in artifacts},audit_source_sha256=sha(Path(__file__)),maximum_reconstruction_errors=ERROR,resources=resources,scope=dict(all_auxiliary_commands=7680,all_NN_member_block_queries=115200,local_independent_SVD_context_fits=384,local_subset_val_trajectories=8,local_subset_time_indices=[16,40,120,240],local_independent_member_block_queries=3840,all_saved_score_rows_verified=True,all_clusterstats_and_paired_matrices_verified=True),summaries=summary,paired_member_scores=paired,limitations=['POSTHOC after negative heldoutfork; no new learner, candidate choice or confirmatory inference.','AllNN predictions independently reconstructed; only balanced384/11520local fits independently reconstructed, not a uniformcertificate of every localquery.','Only ordinarytraining/validation rows contribute; combined NPZ is decompressed before masking, not literal no testbytesread.','Dataaccess wording clarified during initial diagnostic invocation before anyscoresread; formulas/protocol/candidates unchanged according to recorded root provenance.','Eight trajectoryclusters, not960independentobservations or independentbodies; paired descriptive SE is not postselectionvalidity.','No evaluator/fork/control artifact consumed by this audit; no physicalinteractions.','Centeredmean risk does not identify full response law, policy sufficiency or control value.'],wall_seconds=time.time()-start)
 (ROOT/'audit_cartesian_stream_v17_action_risk_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in('status','maximum_reconstruction_errors','scope','wall_seconds')}))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
