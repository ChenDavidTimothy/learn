#!/usr/bin/env python3
"""Tiny official-wrapper autoreset fixture using a mocked plant, not physics."""
from pathlib import Path
import hashlib,json,tempfile,time
import numpy as np
from amago_runtime import activate
activate()
import amago_control_env as adapter
from amago.envs import AMAGOEnv
from amago.envs.amago_env import SequenceWrapper
ROOT=Path(__file__).resolve().parent
class FakePlant:
 def __init__(self):self.generation=0
 def reset(self,seed,n,split):
  self.generation+=1;self.n=n;self.t=0;self.state=np.zeros((n,4));self.state[:,:2]=.1*self.generation;self.goals=np.zeros((n,3,2));return self.state.copy()
 @property
 def done(self):return self.t==3
 def step(self,u):
  self.t+=1;self.state[:,:2]+=.03;self.state[:,2:]=self.t*.02;return self.state.copy(),np.full(self.n,float(self.t))
 def precision(self):assert self.done;return np.zeros((self.n,3),bool)
 def evaluator_snapshot(self):return {'secret_body':np.full(self.n,999.)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();original=adapter.InnovationControlEnv
 try:
  adapter.InnovationControlEnv=FakePlant
  env=adapter.AMAGOControlEnv(n=2,seed=22,split='train');wrapped=AMAGOEnv(env,env_name='mock',batched_envs=2)
  with tempfile.TemporaryDirectory(prefix='amago_interface_') as tmp:
   sequence=SequenceWrapper(wrapped,save_trajs_to=tmp);first,_=sequence.reset();initial=first['observation'].copy()
   for i in range(3):obs,reward,terminated,truncated,info=sequence.step(np.full((2,2),.2,dtype=np.float32))
   assert terminated.all() and not truncated.any() and np.array_equal(reward,[-3.,-3.]);assert len(env.completed)==1 and len(sequence.finished_trajs)==2
   saved=env.completed[0];assert saved['q'].shape==(2,4,2) and saved['u'].shape==(2,3,2) and np.allclose(saved['q'][:,-1],.19)
   reset=obs['observation'];assert np.array_equal(reset,env._observation()) and not np.array_equal(reset,initial)
   assert np.all(reset[:,16:20]==0) and np.all(reset[:,15]==1) and np.all(reset[:,12:15]==[1,0,0])
   # The final terminal sample in replay carries old action/reward/done but
   # reset observation; new active replay/context has reset action/reward/time.
   for _,trajectory in sequence.finished_trajs:
    old=trajectory.freeze();assert len(old.actions)==3 and len(old.obs['observation'])==4 and bool(old.dones[-1,0])
    assert np.all(old.actions[-1]==np.float32(.2)) and old.rews[-1,0]==-3.
    assert np.array_equal(old.obs['observation'][-1],reset[0])
   for trajectory in sequence.active_trajs:
    t=trajectory.timesteps[0];assert np.all(t.prev_action==0) and np.all(t.reward==0) and np.all(t.time_idx==0) and not t.terminal.any()
   ordinary,rl2,time_idx=sequence.current_timestep;assert np.all(rl2==0) and np.all(time_idx==0) and np.array_equal(ordinary['observation'],reset)
   assert all('secret' not in k for k in obs) and all('secret' not in k for k in info)
  out=dict(status='PASS TINY OFFICIAL SEQUENCEWRAPPER AUTO-RESET INTERFACE FIXTURE',source_sha256=sha(Path(__file__)),checked_source_sha256={p.name:sha(p) for p in [ROOT/'amago_control_env.py',ROOT/'amago_control_study.py',ROOT/'amago_runtime.py',ROOT/'third_party/amago/amago/envs/amago_env.py',ROOT/'third_party/amago/amago/experiment.py',ROOT/'third_party/amago/amago/loading.py']},checks=dict(old_terminal_q_private_in_completed=True,terminal_reward_action_done_in_old_replay=True,new_reset_observation_in_terminal_return=True,new_RL2_time_action_reward_cleared=True,no_privileged_secret_in_policy_obs_or_info=True),wall_seconds=time.time()-start,limitations=['Mocked3-step2-memberplant; no physical integration, optimizer update, networkforward or capability result.','Manualepoch and fullsequence masks reviewed in pinned source; not executed in this fixture.','Core wrapper reused unchanged at upstream54c25ab; no new algorithm claim.'])
  (ROOT/'check_amago_control_interface_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
 finally:adapter.InnovationControlEnv=original
if __name__=='__main__':main()
