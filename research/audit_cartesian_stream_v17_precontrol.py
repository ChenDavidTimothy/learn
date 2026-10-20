#!/usr/bin/env python3
"""Pre-control adapter checks; no plant flow, witness or control episode run.

Independent saved-weight NumPy paths/cost derivatives are compared with the
candidate planner units. Mock optimizer endpoints test acceptance/fallback only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from audit_cartesian_stream_v17_fit import ROOT, sha, context, rows, paths, local_coefficients


def endpoint(q):
    return np.array([np.cos(q[0])+np.cos(q[0]+q[1]),np.sin(q[0])+np.sin(q[0]+q[1])])


def jac(q):
    return np.array([[-np.sin(q[0])-np.sin(q.sum()),-np.sin(q.sum())],
                     [np.cos(q[0])+np.cos(q.sum()),np.cos(q.sum())]])


def cost_gradient(prediction, derivative, current, goal, plan):
    previous=np.concatenate([np.broadcast_to(current,(3,1,2)),prediction[:,:-1]],axis=1)
    previous_j=np.concatenate([np.zeros((3,1,2,12)),derivative[:,:-1]],axis=1)
    velocity=(prediction-previous)/.1
    vj=(derivative-previous_j)/.1
    error=prediction-goal
    cost=np.mean(np.sum(error**2+.25*velocity**2,axis=(1,2)))+.02*np.sum(plan**2)
    gradient=np.mean(np.sum(2*error[:,:,:,None]*derivative+.5*velocity[:,:,:,None]*vj,axis=(1,2)),axis=0)+.04*plan.ravel()
    return float(cost),gradient.reshape(6,2)


def main():
    torch.set_num_threads(1)
    source=ROOT/'cartesian_stream_v17_control.py'
    spec=importlib.util.spec_from_file_location('planner_under_test',source)
    implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
    # Any accidental physical execution in these checks is a failure.
    def forbidden_flow(*args,**kwargs): raise AssertionError('No physical execution authorized in pre-control check')
    implementation.flow=forbidden_flow
    prefix=ROOT/'cartesian_stream_v17_rep0';ordinary=Path(str(prefix)+'_ordinary.npz');fitpath=Path(str(prefix)+'_fit.json');weightpath=Path(str(prefix)+'_weights.npz')
    fit=json.loads(fitpath.read_text());data=np.load(ordinary,allow_pickle=False);weights=np.load(weightpath,allow_pickle=False)
    selected,_=implementation.load_selected(prefix,ordinary)
    o=data['observed'];a=data['commands'];X,U,Y,origins=rows(o,a,np.flatnonzero(data['split']==0))
    scale={k:np.array(v) for k,v in fit['scale'].items()}
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    i=24;t=40;obs=o[i,:t+1];commands=a[i,:t];plan=a[i,t:t+12:2].copy();goal=.95*obs[-1]
    checks={};prepared_by_kind={}
    with threadpool_limits(limits=1):
        for kind,predictor in selected.items():
            prepared=predictor.prepare(obs,commands);prepared_by_kind[kind]=prepared
            ci=fit['selected'][kind];models=[]
            if kind=='nonlinear':
                for member in range(3):models.append([weights[f'nn{ci}_member{member}_layers.{layer}.{name}'] for layer in [0,2,4] for name in ['weight','bias']])
            else:
                query=(context(obs,commands)-scale['context_mean'])/scale['context_std']
                config=fit['local_candidates'][ci]
                for b in fit['bootstrap_rows']:
                    b=np.array(b);coef,_=local_coefficients(X[b],U[b],Y[b],query,config['neighbors'],config['ridge']);models.append(coef)
            predictions,derivative=paths(obs,commands,plan,scale,models,kind,True)
            cost,gradient=cost_gradient(predictions,derivative,obs[-1],goal,plan)
            variable=torch.tensor(plan,requires_grad=True)
            actual=implementation.modeled_loss(predictor,obs,commands,goal,variable,prepared)
            actual_gradient=torch.autograd.grad(actual,variable)[0].detach().numpy()
            cost_error=abs(float(actual.detach())-cost);gradient_error=float(np.max(abs(actual_gradient-gradient)))
            fd=np.zeros((6,2));eps=1e-6
            for index in np.ndindex(plan.shape):
                pp=plan.copy();pm=plan.copy();pp[index]+=eps;pm[index]-=eps
                yp,jp=paths(obs,commands,pp,scale,models,kind,True);ym,jm=paths(obs,commands,pm,scale,models,kind,True)
                fd[index]=(cost_gradient(yp,jp,obs[-1],goal,pp)[0]-cost_gradient(ym,jm,obs[-1],goal,pm)[0])/(2*eps)
            fd_error=float(np.max(abs(gradient-fd)))
            assert cost_error<1e-10 and gradient_error<1e-9 and fd_error<2e-7
            checks[kind]=dict(cost=cost,torch_vs_numpy_cost_error=cost_error,torch_vs_numpy_gradient_error=gradient_error,numpy_derivative_vs_scalar_finite_difference_error=fd_error,max_absolute_gradient=float(np.max(abs(gradient))))

        # Deliberately artificial optimizer records exercise specified policy,
        # not the quality/convergence of L-BFGS-B or any selected actual action.
        optimizer_checks={};original_minimize=implementation.minimize
        for mode in ['failed_equal_finite','worse_final','infeasible_final','exception']:
            def mock_minimize(fun,x0,mode=mode,**kwargs):
                if mode=='exception':raise RuntimeError('intentional preflight fixture')
                value,gradient=fun(x0)
                x=x0.copy();score=value
                if mode=='worse_final':score+=1.
                if mode=='infeasible_final':x[0]=.6;score-=1.
                return SimpleNamespace(x=x,fun=score,jac=gradient,success=False,status=1,message='intentional preflight fixture',nfev=1,njev=1,nit=0)
            implementation.minimize=mock_minimize
            _,chosen,record,_=implementation.choose_action(selected['nonlinear'],obs,commands,goal,np.zeros((6,2)))
            expected_fallback=mode!='failed_equal_finite'
            assert record['fallback_to_initial']==expected_fallback and record['optimizer']['success'] is False
            assert np.array_equal(chosen,record['initial_plan'])
            g=np.array(record['selected_gradient']);pg=chosen-np.clip(chosen-g,-.5,.5)
            assert np.array_equal(pg,np.array(record['selected_projected_gradient']))
            optimizer_checks[mode]=dict(fallback=record['fallback_to_initial'],reported_unsuccessful=not record['optimizer']['success'],gradient_calls=record['modeled_value_gradient_calls'])
        implementation.minimize=original_minimize

    # Independent RNG preparation/warmup/sensor reconstruction (no physical flow).
    cases=implementation.make_cases();rng_error=0.
    for case in cases:
        k=case['context_id'];repeat=case['noise_repeat']
        generator=np.random.default_rng(np.random.SeedSequence([20261020,17,670,0,k,0]))
        theta=generator.uniform(-np.pi,np.pi);phi=generator.uniform(.4,np.pi-.4);sign=generator.choice([-1.,1.]);speed=generator.uniform(.02,.12);direction=generator.uniform(-np.pi,np.pi)
        q=np.array([theta-sign*phi/2,sign*phi]);state=np.r_[q,np.linalg.solve(jac(q),speed*np.array([np.cos(direction),np.sin(direction)]))]
        u=np.random.default_rng(np.random.SeedSequence([20261020,17,670,0,k,2])).uniform(-.5,.5,(4,2));warm=np.r_[np.zeros((8,2)),np.repeat(u,2,axis=0)]
        noise=np.random.default_rng(np.random.SeedSequence([20261020,17,670,0,k,1,repeat])).normal(0,.0002,(197,2))
        rng_error=max(rng_error,float(np.max(abs(state-case['initial_state']))),float(np.max(abs(warm-case['warmup_commands']))),float(np.max(abs(noise-case['sensor_noise']))))
    assert rng_error==0.
    assert all(np.array_equal(cases[2*k]['initial_state'],cases[2*k+1]['initial_state']) for k in range(4))

    # Scoring fixture deliberately need not be a physical solution. It tests
    # endpoint indices, unannounced-goal warmup, final ten samples and effort.
    rng=np.random.default_rng(21841);states=rng.normal(0,.1,(197,4));commands=rng.uniform(-.5,.5,(196,2));goals=rng.normal(0,.1,(3,2))
    score=implementation.score_episode(states,commands,goals)
    positions=np.array([endpoint(z[:2]) for z in states]);vel=np.array([jac(z[:2])@z[2:] for z in states]);costs=[];score_error=0.;successes=0
    for phase in range(3):
        first=16+60*phase;last=first+60;e=positions[first+1:last+1]-goals[phase];v=vel[first+1:last+1];u=commands[first:last]
        cost=.1*np.sum(e**2+.25*v**2+.01*u**2);costs.append(cost)
        success=bool(np.all(abs(e[-10:])<=.03) and np.all(abs(v[-10:])<=.03));successes+=int(success)
        assert success==score['phases'][phase]['precision_success']
        score_error=max(score_error,abs(cost-score['phases'][phase]['cost']))
    warm=.1*(.25*np.sum(vel[1:17]**2)+.01*np.sum(commands[:16]**2))
    score_error=max(score_error,abs(warm-score['warmup']['cost_without_unannounced_goal']),abs(sum(costs)+warm-score['combined_charged_cost']))
    assert score_error<1e-11 and score['precise_goals']==successes
    files=[source,ROOT/'cartesian_stream_v17_planner_contract.md',ROOT/'cartesian_stream_v17_learn.py',ROOT/'cartesian_stream_v17_local.py',ordinary,fitpath,weightpath]
    result=dict(status='PASS PRE-CONTROL SOURCE/NUMERICAL UNIT CHECK; NO PHYSICAL EPISODE EXECUTED',artifact_sha256={p.name:sha(p) for p in files},selected_actual_model_loss_gradient_checks=checks,
        mock_optimizer_branch_checks=optimizer_checks,paired_case_rng_max_error=rng_error,scoring_fixture_max_error=score_error,
        expected_counts=dict(ordinary_episodes=16,ordinary_phases=48,ordinary_decisions=1440,ordinary_intervals=3136,ordinary_seconds=313.6,ordinary_camera_opportunities=3152,ordinary_scalar_noise_opportunities=6304,distinct_scalar_noise_rng_draws=3152,witness_episodes=8,witness_intervals=1568,witness_seconds=156.8),
        limitations='Canonical control units imported as adapters for loss/scoring/mock optimizer checks; independent audit-fit NumPy equations are the reference. Physical flow replaced by a raising guard. No real L-BFGS convergence, physical feasibility, witness trajectory, closed-loop result or actual append stream certified.')
    (ROOT/'audit_cartesian_stream_v17_precontrol_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
