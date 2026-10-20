#!/usr/bin/env python3
"""Frozen nine-setting causal PD bank; imports only independent audit helpers."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_continuous_history_v15_pilot import integrate, adaptive, metrics

ROOT=Path(__file__).resolve().parent


def main():
    parent_path=ROOT/'continuous_history_v15_pilot.json'
    parent=json.loads(parent_path.read_text())
    cases=[(window,kp,kd,item) for window in [3,5,9] for kp,kd in [(2.,2.),(4.,4.),(8.,8.)] for item in range(2)]
    initial=[parent['controls']['pose'][item]['states_evaluator_only'][0] for _,_,_,item in cases]
    goals=np.array([parent['controls']['pose'][item]['goals'] for _,_,_,item in cases])
    paths=[[20261020,15,610,0,4,item] for _,_,_,item in cases]
    noise=np.array([np.random.default_rng(np.random.SeedSequence(path)).normal(0,.001,(189,2)) for path in paths])
    states=[np.array(initial)];observed=[states[0][:,:2]+noise[:,0]];commands=[]
    weights={}
    for window in [3,5,9]:
        times=np.arange(-window+1,1)*.1
        weights[window]=np.linalg.pinv(np.c_[np.ones(window),times,.5*times**2])[:2]
    local_convergence=0.;worst_local=None
    for t in range(188):
        action=np.zeros((len(cases),2))
        if t>=8:
            for i,(window,kp,kd,item) in enumerate(cases):
                past=np.array([o[i] for o in observed[-window:]])
                position,slope=weights[window]@past
                action[i]=-.5*np.tanh(kp*(position-goals[i,t-8]))-.5*np.tanh(kd*slope)
        next32=integrate(states[-1],action,steps=32);next64=integrate(states[-1],action,steps=64)
        delta=float(np.max(np.abs(next32-next64)))
        if delta>local_convergence:
            local_convergence=delta;where=np.unravel_index(np.argmax(np.abs(next32-next64)),next64.shape)[0]
            worst_local=(states[-1][where].copy(),action[where].copy(),next64[where].copy())
        states.append(next64);commands.append(action);observed.append(next64[:,:2]+noise[:,t+1])
    states=np.stack(states,axis=1);commands=np.stack(commands,axis=1);observed=np.stack(observed,axis=1)
    assert np.max(np.abs(commands))<=1.
    assert local_convergence<1e-7
    radau_error=float(np.max(np.abs(adaptive(worst_local[0],worst_local[1])-worst_local[2])))
    assert radau_error<1e-7
    runs=[]
    for i,(window,kp,kd,item) in enumerate(cases):
        runs.append(dict(window=window,kp=kp,kd=kd,task=item,seed_path=paths[i],states_evaluator_only=states[i],
            observed=observed[i],commands=commands[i],goals=goals[i],metrics=metrics(states[i],commands[i],goals[i])))
    return dict(meta=dict(status='EXPLORATORY FIXED CAUSAL TANH-PD DIAGNOSTIC; ALL NINE SETTINGS RETAINED; NO TEST SELECTION',
        frozen_design='continuous_history_v15_PD_diagnostic_design.md',parent_result_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_helper_sha256=hashlib.sha256((ROOT/'audit_continuous_history_v15_pilot.py').read_bytes()).hexdigest(),
        ordinary_inputs='Current goal plus noisy past/current positions only; causal quadratic polynomial current position/slope.',
        no_true_velocity_or_inertia_in_control=True,no_test_outcome_tuning=True,
        additional_training_samples=0,actual_evaluation_episodes=18,actual_evaluation_intervals=3384,
        actual_evaluation_simulated_seconds=338.4,actual_scalar_sensor_readings=6804,
        distinct_paired_noise_role_paths=2,distinct_noise_scalar_draws=756,
        physics='Separate Hamiltonian RK4 with64substeps; 32substep convergence and worst local Radau checked',
        maximum_local_RK32vs64=local_convergence,worst_local_RK64vsRadau=radau_error),runs=runs)


def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


if __name__=='__main__':
    print(json.dumps(main(),indent=2,default=encode,allow_nan=False))
