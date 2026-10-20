#!/usr/bin/env python3
"""Unreleased causal-history diagnostic; not learned history selection or stopping."""
import json,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from coupled_context_v14_pilot import flow,mass
from coupled_learning_v14 import relation
from nonlinear_identification_v9 import stat

SEED=20261020+2300000
T=.2; PAST=np.array([-.2,-.15,-.1,-.05]); SD=1e-4
POSES=((0.,math.pi/2),(1.,.3)); VELOCITIES=((.02,-.01),(.01,.02),(-.015,.005))


def main():
    parent=json.loads(Path('research/coupled_learning_v14_results.json').read_text());models=[r['models']['corrected_inverse'] for r in parent['runs'] if r['condition']==0]
    worlds=[];oracle_calls=0
    for pi,q in enumerate(POSES):
        q=np.array(q)
        for vi,v in enumerate(VELOCITIES):
            for sign in (-1,1):
                initial_v=sign*np.array(v);past=np.array([flow(q,initial_v,np.zeros(2),time)[0] for time in PAST])
                forward,_=flow(past[0,:2],past[0,2:],np.zeros(2),-PAST[0])
                assert np.max(np.abs(forward-np.r_[q,initial_v]))<1e-10
                def residual(u):
                    nonlocal oracle_calls
                    oracle_calls+=1;return flow(q,initial_v,u,T)[0][:2]-q
                start=np.clip(mass(q)@(-2*initial_v/T),-.99,.99)
                opt=least_squares(residual,start,bounds=(-1,1),max_nfev=60,ftol=1e-13,xtol=1e-13,gtol=1e-12)
                assert opt.success and np.linalg.norm(residual(opt.x))<1e-8
                zero,_=flow(q,initial_v,np.zeros(2),T)
                worlds.append(dict(pose_index=pi,velocity_index=vi,sign=sign,current_pose=q.tolist(),initial_velocity_evaluator_only=initial_v.tolist(),
                    exact_past_states_evaluator_only=past.tolist(),back_forward_error=float(np.max(np.abs(forward-np.r_[q,initial_v]))),
                    oracle_command=opt.x.tolist(),oracle_endpoint_error=float(np.linalg.norm(residual(opt.x))),
                    zero_endpoint=zero.tolist(),leading_pose_only_risk_floor=T*T*float(np.sum(initial_v**2))))
    rows=[]
    for mi,model in enumerate(models):
        for wi,w in enumerate(worlds):
            q=np.array(w['current_pose']);v=np.array(w['initial_velocity_evaluator_only']);past=np.array(w['exact_past_states_evaluator_only'])
            B,K=relation(model,q)
            for ni in range(16):
                seed=SEED+1000000*mi+10000*wi+ni;rng=np.random.default_rng(seed)
                observed=past[:,:2]+rng.normal(0,SD,(len(PAST),2))
                slope=np.sum(PAST[:,None]*(observed-q),axis=0)/np.sum(PAST**2)
                desired_acceleration=-2*slope/T
                fit=lsq_linear(B,desired_acceleration,bounds=(-1,1),tol=1e-12,max_iter=100)
                assert fit.success
                commands={'pose_only_zero':np.zeros(2),'learned_relation_history':fit.x,'oracle_state':np.array(w['oracle_command'])}
                outcomes={}
                for label,u in commands.items():
                    end,_=flow(q,v,u,T);outcomes[label]=dict(command=u.tolist(),final_state_evaluator_only=end.tolist(),
                        endpoint_squared_error=float(np.sum((end[:2]-q)**2)),endpoint_velocity_norm=float(np.linalg.norm(end[2:])))
                rows.append(dict(model=mi,world=wi,sensor_seed=seed,observed_past_positions=observed.tolist(),
                    current_pose_and_goal_supplied=q.tolist(),estimated_slope=slope.tolist(),outcomes=outcomes))
    summary={}
    for label in ('pose_only_zero','learned_relation_history','oracle_state'):
        a=[np.mean([r['outcomes'][label]['endpoint_squared_error'] for r in rows if r['model']==mi]) for mi in range(8)]
        summary[label]=dict(endpoint_mse=stat(a),trials=len(rows))
    print(json.dumps(dict(meta=dict(status='UNRELEASED HISTORY DIAGNOSTIC; independent audit required; no learned summary selection',seed=SEED,
        current_pose_exactly_supplied=True,current_velocity_not_given_to_history_controller=True,history_operator_supplied='anchored least-squares slope',
        sensor_sd=SD,past_times=PAST.tolist(),horizon=T,parent_v14_models=8,worlds=12,sensor_repeats=16,
        true_state_oracle_rollouts=oracle_calls,actual_test_pulse_integrations=len(rows)*3,
        warning='Positionreturn only, not stopping; poseonlyzero not certified global nonlinear poseonlyoptimum; no v15 release onpilot'),
        worlds=worlds,runs=rows,summary=summary),indent=2,allow_nan=False))

if __name__=='__main__':main()
