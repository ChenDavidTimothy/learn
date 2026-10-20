#!/usr/bin/env python3
"""Frozen-control change diagnostic; not an adaptive learning demonstration."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
from continuous_history_v15 import Model,choose,DT,SD,MAX_H,HORIZON,encode
from coupled_context_v14_pilot import flow as parent_flow


def inertia(q,m2):
    c=math.cos(float(q[1]));return np.array([[1+2*m2+2*m2*c,m2*(1+c)],[m2*(1+c),m2]])


def rhs(z,u,m2):
    v=z[2:];s=math.sin(float(z[1]));v1,v2=v
    h=m2*s*np.array([-(2*v1*v2+v2*v2),v1*v1])
    return np.r_[v,np.linalg.solve(inertia(z[:2],m2),u-h)]


def flow(z,u,m2,duration=DT):
    sol=solve_ivp(lambda t,z:rhs(z,u,m2),(0,duration),z,method='DOP853',rtol=1e-11,atol=1e-13)
    assert sol.success and np.all(np.isfinite(sol.y[:,-1]))
    return sol.y[:,-1],int(sol.nfev)


def energy(z,m2):return float(.5*z[2:]@inertia(z[:2],m2)@z[2:])


def mechanics():
    path=[20261020,16,620,0,9,0];rng=np.random.default_rng(np.random.SeedSequence(path));rows=[]
    for case in range(12):
        z=np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-.5,.5,2)]
        u=rng.uniform(-1,1,2)
        for m2 in (1.,2.):
            end,n=flow(z,u,m2,.3);err=energy(end,m2)-energy(z,m2)-float(u@(end[:2]-z[:2]))
            r=dict(case=case,initial_state=z,command=u,m2=m2,duration=.3,final_state=end,energy_work_error=err,nfev=n)
            assert abs(err)<1e-8
            if m2==1:
                reference,nref=parent_flow(z[:2],z[2:],u,.3)
                r.update(parent_state=reference,parent_nfev=nref,parent_max_abs_difference=float(np.max(np.abs(reference-end))))
                assert r['parent_max_abs_difference']<1e-12
            rows.append(r)
    rest=np.array([.3,-.7,0.,0.]);restchecks=[]
    for m2 in (1.,2.):
        end,n=flow(rest,np.zeros(2),m2,1.)
        assert np.array_equal(end,rest)
        restchecks.append(dict(m2=m2,initial_state=rest,command=[0.,0.],final_state=end,nfev=n))
    return dict(seed_path=path,energy_and_parent_checks=rows,unobservable_rest_checks=restchecks)


def score(states,commands,goals):
    z=np.array(states)[MAX_H+1:];a=np.array(commands)[MAX_H:];g=np.array(goals);e=z[:,:2]-g
    def period(start,end):
        return dict(position_mse=float(np.mean(np.sum(e[start:end]**2,axis=1))),
            velocity_rms=float(np.sqrt(np.mean(z[start:end,2:]**2))),action_energy=float(np.mean(a[start:end]**2)))
    precision=[];latep=[];latev=[]
    for k in range(3):
        segment=slice((k+1)*60-10,(k+1)*60)
        precision.append(bool(np.all(np.abs(e[segment])<=.03) and np.all(np.abs(z[segment,2:])<=.03)))
        latep.append(float(np.sqrt(np.mean(e[segment]**2))));latev.append(float(np.sqrt(np.mean(z[segment,2:]**2))))
    return dict(all=period(0,180),before_change=period(0,60),after_change=period(60,180),
        precision_by_goal=precision,late_position_rmse_by_goal=latep,late_velocity_rms_by_goal=latev)


def episode(model,task,loaded,window=None,gain=None):
    path=[20261020,16,620,0,1,task];rng=np.random.default_rng(np.random.SeedSequence(path))
    noise=rng.normal(0,SD,(MAX_H+181,2));q0=np.array(([-2.,-2.1],[-.5,-.7],[.8,.7],[2.,2.1])[task])
    z=np.r_[q0,[.04,-.02]];states=[z.copy()];observed=[z[:2]+noise[0]];commands=[];nfev=0
    for t in range(MAX_H):
        u=np.zeros(2);z,n=flow(z,u,1.);nfev+=n;commands.append(u);states.append(z.copy());observed.append(z[:2]+noise[t+1])
    warm=np.zeros(HORIZON*2);goals=[];plans=[];masses=[]
    offsets=np.array([[.2,-.15],[-.15,.15],[.1,.1]])
    for t in range(180):
        goal=q0+offsets[t//60];goals.append(goal)
        # No m2, loaded flag, t/change-time label or truth is passed to an action.
        if window is None:
            u,warm,plan=choose(model,np.array(observed),np.array(commands),goal,warm);plans.append(plan)
        else:
            times=np.arange(-window+1,1)*DT
            W=np.linalg.pinv(np.stack([np.ones(window),times,.5*times**2],axis=1))
            coef=W@np.array(observed[-window:]);qhat,vhat=coef[0],coef[1]
            u=-.5*np.tanh(gain*(qhat-goal))-.5*np.tanh(gain*vhat)
        m2=2. if loaded and t>=60 else 1.;masses.append(m2)
        z,n=flow(z,u,m2);nfev+=n;commands.append(u);states.append(z.copy());observed.append(z[:2]+noise[MAX_H+t+1])
    return dict(task=task,loaded_condition_evaluator_only=loaded,seed_path=path,window=window,gain=gain,
        states_evaluator_only=np.array(states),observed=np.array(observed),commands=np.array(commands),goals=np.array(goals),
        m2_evaluator_only=masses,plans=plans,plant_nfev=nfev,metrics=score(states,commands,goals))


def main():
    root=Path(__file__).resolve().parent;parent=root/'continuous_history_v15_rep100.json';raw=parent.read_bytes();data=json.loads(raw)
    row=data['candidates'][data['selected_candidate_ids']['rich_delay']];model=Model(row['h'],row['kind'],row['coef'],row['ridge'])
    start=time.time();runs={}
    with threadpool_limits(limits=1):
        checks=mechanics()
        for key,window,gain in [('frozen_mpc',None,None)]+[(f'pd_w{w}_k{k}',w,k) for w in (3,5,9) for k in (2,4,8)]:
            runs[key]={}
            for loaded in (False,True):
                runs[key]['loaded' if loaded else 'unchanged']=[episode(model,j,loaded,window,gain) for j in range(4)]
            for a,b in zip(runs[key]['unchanged'],runs[key]['loaded']):
                for field,stop in [('states_evaluator_only',MAX_H+61),('observed',MAX_H+61),('commands',MAX_H+60)]:
                    assert np.array_equal(a[field][:stop],b[field][:stop]),(key,field)
            print(key,{c:[sum(e['metrics']['precision_by_goal'][1:]) for e in es] for c,es in runs[key].items()},flush=True)
        out=dict(meta=dict(status='EXPLORATORY PAYLOAD INSTRUMENT; NO ADAPTIVE LEARNER OR PROBE TESTED',
            design='payload_revision_v16_pilot_design.md',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_file=parent.name,parent_sha256=hashlib.sha256(raw).hexdigest(),dt=DT,sensor_sd=SD,
            external_jump='matched-velocity payload attachment, q/v continuous, external energy exchange; never supplied to ordinary controller',
            selected_parent='rep100 richH4 by protocol, not best across8',independent_acquired_parent_fits=1),mechanics=checks,runs=runs,
            resources=dict(control_episodes=80,control_intervals=80*188,control_simulated_seconds=80*188*DT,
                scalar_sensor_readings=80*189*2,distinct_paired_sensor_streams=4,distinct_scalar_noise_draws=4*189*2,
                mechanics_flow_calls=24+12+2,mechanics_simulated_seconds=24*.3+12*.3+2,
                planning_residual_calls=sum(p['residual_calls'] for conditions in runs.values() for es in conditions.values() for e in es for p in e['plans'])),
            wall_seconds=time.time()-start)
        (root/'payload_revision_v16_pilot_results.json').write_text(json.dumps(out,default=encode,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
