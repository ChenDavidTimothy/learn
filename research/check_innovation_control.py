#!/usr/bin/env python3
"""Small conditional-probe math fixtures; no training or physical rollout."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
 start=time.time();rho=.35;mu=.2;logstd=-.7;sigma=np.exp(logstd)
 n,w=np.polynomial.hermite.hermgauss(64);n=np.sqrt(2)*n;w=w/np.sqrt(np.pi);xi,eps=np.meshgrid(n,n,indexing='ij');weight=w[:,None]*w[None,:]
 expectation=lambda value:float(np.sum(value*weight))
 z=mu+rho*xi+sigma*eps;u=np.tanh(z);y=np.sin(u)+.8
 lhs=expectation(y*xi);rhs=rho*expectation(np.cos(u)*(1-u*u));assert abs(lhs-rhs)<1e-9
 # An allowed policy mean using CURRENT probe breaks the reduced identity.
 z_cancel=mu+sigma*eps;u_cancel=np.tanh(z_cancel);actual=expectation((np.sin(u_cancel)+.8)*xi);false=rho*expectation(np.cos(u_cancel)*(1-u_cancel*u_cancel));assert abs(actual)<1e-14 and false>.1
 # Conditional Gaussian latent score estimates true reward gradients. No plant
 # path is differentiated by score estimator; finite differences are a fixture.
 objective=lambda m,l:expectation(np.sin(np.tanh(m+rho*xi+np.exp(l)*eps)))
 score_mu=expectation(np.sin(u)*eps/sigma);score_logstd=expectation(np.sin(u)*(eps*eps-1));h=1e-5
 fd_mu=(objective(mu+h,logstd)-objective(mu-h,logstd))/(2*h);fd_logstd=(objective(mu,logstd+h)-objective(mu,logstd-h))/(2*h)
 assert abs(score_mu-fd_mu)<1e-8 and abs(score_logstd-fd_logstd)<1e-8
 # Fixed sampled latent z must be detached for likelihood-ratio gradients.
 # Resampling z with current parameters would instead make centered mu score0.
 same_ratio=np.exp(-.5*((z-mu-rho*xi)/sigma)**2+ .5*eps**2);assert np.max(abs(same_ratio-1))<1e-12
 # Critical production GRU replay without environment or training.
 import copy
 import torch
 import innovation_control_screen as driver
 torch.manual_seed(1919);torch.set_num_threads(1)
 policy=driver.Policy('gru').double();obs=torch.randn(9,3,22,dtype=torch.double)
 full_mu,full_value,_=policy(obs);hidden=None;single_mu=[];single_value=[]
 for t in range(len(obs)):
  m,v,hidden=policy(obs[t:t+1],hidden);single_mu.append(m);single_value.append(v)
 step_mu=torch.cat(single_mu);step_value=torch.cat(single_value)
 muerr=float(abs(step_mu-full_mu).max().detach());verr=float(abs(step_value-full_value).max().detach());assert muerr<1e-12 and verr<1e-12
 # Both computational paths retain the entire recurrent gradient history.
 direction=torch.randn_like(full_mu);loss=(full_mu*direction).sum()+full_value.square().sum();loss.backward();fullgrad={name:v.grad.clone() for name,v in policy.named_parameters() if v.grad is not None}
 policy.zero_grad(set_to_none=True);hidden=None;sm=[];sv=[]
 for t in range(len(obs)):
  m,v,hidden=policy(obs[t:t+1],hidden);sm.append(m);sv.append(v)
 ((torch.cat(sm)*direction).sum()+torch.cat(sv).square().sum()).backward()
 gradient_error=max(float(abs(v.grad-fullgrad[name]).max()) for name,v in policy.named_parameters() if name in fullgrad);assert gradient_error<1e-10
 # Complete-episode minibatch permutation must not mix independent histories.
 perm=torch.tensor([2,0,1]);pm,pv,_=policy(obs[:,perm]);perr=float(abs(pm-full_mu[:,perm]).max().detach());assert perr<1e-12
 # Reconstruct the same rollout-conditioned Gaussian at unchanged parameters.
 with torch.no_grad():
  xi0=torch.randn_like(full_mu);eps0=torch.randn_like(full_mu);stored_z=full_mu+.35*xi0+policy.logstd.exp()*eps0
  old=torch.distributions.Normal(full_mu+.35*xi0,policy.logstd.exp()).log_prob(stored_z).sum(-1)
  replay_mu,_,_=policy(obs);new=torch.distributions.Normal(replay_mu+.35*xi0,policy.logstd.exp()).log_prob(stored_z).sum(-1)
  ratio_error=float(abs(torch.exp(new-old)-1).max());assert ratio_error<1e-12
 # Synthetic external driver fixture crossing the actual first32 cutoff.
 # Replaces plant entirely; this is not physical integration or a control study.
 class FakeEnv:
  def reset(self,seed,n,split):
   self.n=n;self.t=0;self.state=np.zeros((n,4));self.goals=np.zeros((n,3,2));return self.state.copy()
  def observation(self):return dict(state=self.state.copy(),future_goals=self.goals.copy())
  def step(self,u):
   acc=np.asarray(u)@np.array([[.8,.3],[-.4,.7]]);self.state[:,2:]+=driver.DT*acc;self.state[:,:2]+=driver.DT*self.state[:,2:];self.t+=1
   return self.state.copy(),np.mean(self.state*self.state,axis=1)
  def precision(self):return np.zeros((self.n,3),bool)
  def evaluator_snapshot(self):return {}
 original_env,original_steps=driver.InnovationControlEnv,driver.EPISODE_STEPS
 try:
  driver.InnovationControlEnv=FakeEnv;driver.EPISODE_STEPS=40;moment_policy=driver.Policy('moment')
  data=driver.rollout(moment_policy,2,'validation',1,[1919,1]);assert np.count_nonzero(data['probe'][32:])==0 and np.count_nonzero(data['probe'][:32])>0
  memory=np.zeros((2,2,2));memory_error=0.
  for t in range(40):
   memory_error=max(memory_error,float(abs(data['moment'][t]-memory).max()))
   if t<32:
    acc=(data['states'][t+1,:,2:]-data['states'][t,:,2:])/driver.DT;memory=.98*memory+.02*acc[:,:,None]*data['probe'][t,:,None,:]
  assert memory_error<1e-12 and np.array_equal(data['moment'][32:],np.broadcast_to(data['moment'][32],(8,2,2,2)))
  previous=np.r_[np.zeros((1,2,2)),data['probe'][:-1]];past_probe_error=float(abs(data['obs'][:,:,18:20]-previous).max());assert past_probe_error==0
  with torch.no_grad():
   mm,_,_=moment_policy(torch.from_numpy(data['obs']));expected_z=mm.numpy()+.35*data['probe'];evaluation_error=float(abs(expected_z-data['z']).max());assert evaluation_error<1e-7
  # Actual rollout-to-full-sequence replay under stochastic TRAIN sampling.
  train_policy=driver.Policy('gru');train=driver.rollout(train_policy,2,'train',1,[1919,2])
  with torch.no_grad():
   mm,_,_=train_policy(torch.from_numpy(train['obs']));newlog=torch.distributions.Normal(mm+.35*torch.from_numpy(train['probe']),train_policy.logstd.exp()).log_prob(torch.from_numpy(train['z'])).sum(-1)
   rollout_ratio_error=float(abs(torch.exp(newlog-torch.from_numpy(train['logp']))-1).max());assert rollout_ratio_error<2e-6
 finally:driver.InnovationControlEnv=original_env;driver.EPISODE_STEPS=original_steps
 production=dict(synthetic_driver_probe_zero_after32=True,synthetic_driver_memory_freezes_after32=True,synthetic_memory_recursion_error=memory_error,network_past_probe_input_error=past_probe_error,deterministic_evaluation_latent_action_error=evaluation_error,actual_rollout_to_full_GRU_replay_ratio_error=rollout_ratio_error,driver_source_sha256=driver.digest(Path(driver.__file__)),GRU_step_vs_full_mean_error=muerr,GRU_step_vs_full_value_error=verr,full_BPTT_step_vs_batch_gradient_error=gradient_error,episode_permutation_error=perr,unchanged_conditional_ratio_error=ratio_error,scope='Synthetic 9-step GRU sequences plus40-step mocked-driver traces crossing32-step cutoff; no physical simulator, fit or optimizer update.')
 output=dict(status='PASS SMALL CONDITIONAL STEIN/SCORE AND PRODUCTION GRU FIXTURES; NO CONTROL CAPABILITY TEST',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_GRU=production,conditional_Stein=dict(moment=lhs,sensitivity=rhs,error=abs(lhs-rhs)),current_probe_leak_counterexample=dict(actual_moment=actual,incorrect_reduced_identity_rhs=false,mean_rule='mu_base-rho*current_xi'),conditional_score=dict(mean_score=score_mu,mean_finite_difference=fd_mu,logstd_score=score_logstd,logstd_finite_difference=fd_logstd,same_policy_ratio_max_error=float(abs(same_ratio-1).max())),wall_seconds=time.time()-start,limitations=['Synthetic smooth response and deterministic Gaussian quadrature only.','Production GRU and cutoff timing checked with synthetic sequences/mocked plant only; no physical rollout or fit.','Stein identity requires current mean/std chosen before current independent probe; controlsaturation included bychainrule.','Score validity needs fixed stored latent action/probe during PPO update and correct recurrent history replay.','No convergence, stability, sufficient-statistic or controller competence theorem.'])
 (ROOT/'check_innovation_control_results.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
