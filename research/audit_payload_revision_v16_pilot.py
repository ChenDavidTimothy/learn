#!/usr/bin/env python3
"""Independent variable-mass Hamiltonian, Cartesian-energy and action audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_continuous_history_v15_pilot import residual

ROOT=Path(__file__).resolve().parent


def momenta(states,mass):
    z=np.atleast_2d(states);m=np.broadcast_to(mass,len(z));c=np.cos(z[:,1]);v1,v2=z[:,2],z[:,3]
    return np.c_[(1+2*m+2*m*c)*v1+m*(1+c)*v2,m*(1+c)*v1+m*v2]


def energy(states,mass):
    z=np.atleast_2d(states);m=np.broadcast_to(mass,len(z));q1,q2,v1,v2=z.T
    first=np.c_[-np.sin(q1)*v1,np.cos(q1)*v1]
    tip=first+np.c_[-np.sin(q1+q2)*(v1+v2),np.cos(q1+q2)*(v1+v2)]
    return .5*(np.sum(first*first,axis=1)+m*np.sum(tip*tip,axis=1))


def integrate(states,u,mass,duration=.1,steps=64):
    z,u=np.atleast_2d(states),np.atleast_2d(u);m=np.broadcast_to(mass,len(z))
    y=np.c_[z[:,:2],momenta(z,m)]
    def rhs(s):
        c,sn=np.cos(s[:,1]),np.sin(s[:,1]);den=1+m*sn*sn
        v1=(s[:,2]-(1+c)*s[:,3])/den
        v2=(-(1+c)*s[:,2]+(1/m+2+2*c)*s[:,3])/den
        return np.c_[v1,v2,u[:,0],u[:,1]-m*sn*(v1*v1+v1*v2)]
    dt=duration/steps
    for _ in range(steps):
        a=rhs(y);b=rhs(y+.5*dt*a);c=rhs(y+.5*dt*b);d=rhs(y+dt*c)
        y+=dt*(a+2*b+2*c+d)/6
    c,sn=np.cos(y[:,1]),np.sin(y[:,1]);den=1+m*sn*sn
    return np.c_[y[:,:2],(y[:,2]-(1+c)*y[:,3])/den,
        (-(1+c)*y[:,2]+(1/m+2+2*c)*y[:,3])/den]


def score(states,commands,goals):
    s=np.array(states)[9:];u=np.array(commands)[8:];g=np.array(goals);e=s[:,:2]-g
    def period(a,b):return dict(position_mse=float(np.mean(np.sum(e[a:b]**2,axis=1))),
        velocity_rms=float(np.sqrt(np.mean(s[a:b,2:]**2))),action_energy=float(np.mean(u[a:b]**2)))
    precision=[];latep=[];latev=[]
    for k in range(3):
        segment=slice(k*60+50,k*60+60)
        precision.append(bool(np.all(np.abs(e[segment])<=.03) and np.all(np.abs(s[segment,2:])<=.03)))
        latep.append(float(np.sqrt(np.mean(e[segment]**2))));latev.append(float(np.sqrt(np.mean(s[segment,2:]**2))))
    return dict(all=period(0,180),before_change=period(0,60),after_change=period(60,180),
        precision_by_goal=precision,late_position_rmse_by_goal=latep,late_velocity_rms_by_goal=latev)


def main():
    path=ROOT/'payload_revision_v16_pilot_results.json';r=json.loads(path.read_text())
    parent_path=ROOT/r['meta']['parent_file'];parent=json.loads(parent_path.read_text())
    parent_sha=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert parent_sha==r['meta']['parent_sha256']
    certified=json.loads((ROOT/'audit_continuous_history_v15_results.json').read_text())
    assert parent_sha==certified['artifact_sha256'][parent_path.name]
    helper='audit_continuous_history_v15_pilot.py'
    assert hashlib.sha256((ROOT/helper).read_bytes()).hexdigest()==certified['artifact_sha256'][helper]
    assert r['meta']['source_sha256']==hashlib.sha256((ROOT/'payload_revision_v16_pilot.py').read_bytes()).hexdigest()
    model=parent['candidates'][parent['selected_candidate_ids']['rich_delay']];assert model['h']==4
    errors={}
    def check(key,a,b):errors[key]=max(errors.get(key,0.),float(np.max(np.abs(np.array(a)-np.array(b)))))
    mechanics=r['mechanics'];seed_path=[20261020,16,620,0,9,0];assert mechanics['seed_path']==seed_path
    rng=np.random.default_rng(np.random.SeedSequence(seed_path));cases=mechanics['energy_and_parent_checks'];assert len(cases)==24
    for case in range(12):
        z=np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-.5,.5,2)];u=rng.uniform(-1,1,2)
        for offset,m in enumerate([1.,2.]):
            row=cases[2*case+offset]
            assert row['case']==case and row['m2']==m and row['duration']==.3
            check('mechanics_RNG',z,row['initial_state']);check('mechanics_RNG',u,row['command'])
            end32=integrate(z,u,m,.3,32)[0];end64=integrate(z,u,m,.3,64)[0]
            check('mechanics_RK32vs64',end32,end64);check('mechanics_physics',end64,row['final_state'])
            err=(energy(end64,m)-energy(z,m))[0]-u@(end64[:2]-z[:2])
            check('energy_work',err,0.);check('stored_energy_work',err,row['energy_work_error'])
            check('cyclic_momentum',momenta(end64,m)[0,0]-momenta(z,m)[0,0],.3*u[0])
            if m==1.:
                check('parent_unit_physics',end64,row['parent_state'])
                check('parent_stored_difference',np.max(np.abs(np.array(row['parent_state'])-np.array(row['final_state']))),row['parent_max_abs_difference'])
    for row in mechanics['unobservable_rest_checks']:
        assert np.array_equal(row['initial_state'],row['final_state'])
        assert np.array_equal(integrate(row['initial_state'],row['command'],row['m2'],1.)[0],row['final_state'])
    assert len(mechanics['unobservable_rest_checks'])==2
    episodes=[];masses=[];summary={};plans=0;failures=0;jumps=[]
    for method,conditions in r['runs'].items():
        summary[method]={}
        for condition,es in conditions.items():
            assert len(es)==4
            for task,e in enumerate(es):
                loaded=condition=='loaded';assert e['loaded_condition_evaluator_only']==loaded
                path=[20261020,16,620,0,1,task];assert e['seed_path']==path and e['task']==task
                s=np.array(e['states_evaluator_only']);o=np.array(e['observed']);u=np.array(e['commands']);g=np.array(e['goals'])
                q0=np.array([[-2.,-2.1],[-.5,-.7],[.8,.7],[2.,2.1]][task])
                check('initial_state',s[0],np.r_[q0,[.04,-.02]])
                noise=np.random.default_rng(np.random.SeedSequence(path)).normal(0,.001,(189,2))
                check('sensor_RNG',s[:,:2]+noise,o)
                check('goals',g,np.repeat(q0+np.array([[.2,-.15],[-.15,.15],[.1,.1]]),60,axis=0))
                assert np.all(u[:8]==0.) and np.max(np.abs(u))<=1.+1e-12
                expected_mass=np.r_[np.ones(60),np.full(120,2. if loaded else 1.)]
                check('mass_schedule',e['m2_evaluator_only'],expected_mass)
                if method=='frozen_mpc':
                    assert e['window'] is None and e['gain'] is None and len(e['plans'])==180
                    warm=np.zeros(30)
                    for t,p in enumerate(e['plans']):
                        plan=np.array(p['plan']).ravel();assert np.max(np.abs(plan))<=1.+1e-12
                        check('issued_plan',plan[:2],u[t+8])
                        hist=o[:t+9];past=u[:t+8]
                        rr=residual(plan,model,hist,past,g[t])[0];ri=residual(warm,model,hist,past,g[t])[0]
                        check('plan_cost',rr@rr,p['selected_cost']);check('warm_cost',ri@ri,p['initial_cost'])
                        assert rr@rr<=ri@ri+1e-9 and p['residual_calls']==p['nfev']+2
                        eps=1e-25;perturb=np.tile(plan.astype(complex),(30,1))+1j*eps*np.eye(30)
                        J=residual(perturb,model,hist,past,g[t]).imag.T/eps
                        gradient=2*J.T@rr;pg=plan-np.clip(plan-gradient,-1,1)
                        check('projected_gradient',np.max(np.abs(pg)),p['projected_gradient_max'])
                        warm=np.r_[plan[2:],0.,0.];plans+=1;failures+=not p['success']
                else:
                    window,gain=e['window'],e['gain'];assert method==f'pd_w{window}_k{gain}' and not e['plans']
                    times=np.arange(-window+1,1)*.1;weights=np.linalg.pinv(np.c_[np.ones(window),times,.5*times**2])
                    for t in range(180):
                        position,slope=(weights@o[t+9-window:t+9])[:2]
                        wanted=-.5*np.tanh(gain*(position-g[t]))-.5*np.tanh(gain*slope)
                        check('causal_PD_actions',wanted,u[t+8])
                computed=score(s,u,g)
                for period in ['all','before_change','after_change']:
                    for key,value in computed[period].items():check('metrics',value,e['metrics'][period][key])
                for key in ['late_position_rmse_by_goal','late_velocity_rms_by_goal']:check('metrics',computed[key],e['metrics'][key])
                assert computed['precision_by_goal']==e['metrics']['precision_by_goal']
                if loaded:
                    before=s[68]
                    delta_energy=(energy(before,2.)-energy(before,1.))[0]
                    check('jump_cartesian_energy',delta_energy,energy(before,1.)[0]-.5*np.sum(before[2:3]**2))
                    # Newly added distal mass contributes half its Cartesian tip speed squared.
                    assert delta_energy>=-1e-14
                    jumps.append(dict(method=method,task=task,energy_added=delta_energy,
                        momentum_jump=(momenta(before,2.)-momenta(before,1.))[0].tolist()))
                episodes.append(e);masses.append(np.r_[np.ones(8),expected_mass])
            summary[method][condition]=dict(after_change_position_mse=float(np.mean([e['metrics']['after_change']['position_mse'] for e in es])),
                after_change_velocity_rms=float(np.mean([e['metrics']['after_change']['velocity_rms'] for e in es])),
                post_change_precision=sum(sum(e['metrics']['precision_by_goal'][1:]) for e in es),post_change_goal_windows=8,
                per_task_post_change_precision=[sum(e['metrics']['precision_by_goal'][1:]) for e in es])
        for a,b in zip(conditions['unchanged'],conditions['loaded']):
            for field,stop in [('states_evaluator_only',69),('observed',69),('commands',69)]:
                assert np.array_equal(a[field][:stop],b[field][:stop])
            assert a['plans'][:61]==b['plans'][:61]
    z=np.array([e['states_evaluator_only'][0] for e in episodes]);z64=z.copy();masses=np.array(masses)
    for t in range(188):
        u=np.array([e['commands'][t] for e in episodes]);m=masses[:,t]
        z=integrate(z,u,m,steps=32);z64=integrate(z64,u,m,steps=64)
        check('control_RK32vs64',z,z64);check('control_physics',z64,[e['states_evaluator_only'][t+1] for e in episodes])
    assert len(episodes)==80 and plans==1440
    expected=dict(control_episodes=80,control_intervals=15040,control_simulated_seconds=1504.,scalar_sensor_readings=30240,
        distinct_paired_sensor_streams=4,distinct_scalar_noise_draws=1512,mechanics_flow_calls=38,mechanics_simulated_seconds=12.8,
        planning_residual_calls=sum(p['residual_calls'] for e in episodes for p in e['plans']))
    for key,value in expected.items():assert np.isclose(r['resources'][key],value,rtol=1e-14,atol=0.)
    assert max(errors.values())<1e-7,errors
    paths=[(20261020,16,620,0,1,task) for task in range(4)]+[tuple(seed_path)]
    states={tuple(np.random.SeedSequence(path).generate_state(4).tolist()) for path in paths};assert len(states)==5
    prior_paths=[(20261020,15,610,rep,role,item) for rep in [0,*range(100,108)] for role,n in [(1,48),(2,12),(3,12),(4,2 if rep==0 else 4)] for item in range(n)]
    prior_states={tuple(np.random.SeedSequence(path).generate_state(4).tolist()) for path in prior_paths}
    assert not (states&prior_states)
    names=['payload_revision_v16_pilot.py','payload_revision_v16_pilot_design.md','payload_revision_v16_pilot_results.json',parent_path.name,
           'audit_continuous_history_v15_pilot.py','audit_payload_revision_v16_pilot.py']
    return dict(status='INDEPENDENTLY VERIFIED EXPLORATORY PAYLOAD INSTRUMENT; NO ADAPTIVE LEARNING TESTED',
        artifact_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        maximum_absolute_errors=errors,checked=dict(mechanical_mass_cases=24,parent_unit_recoveries=12,rest_cases=2,
            episodes=80,control_intervals=15040,MPC_plans=1440,full_complex_step_plan_gradients=1440,
            causal_PD_actions=12960,paired_identical_prefixes=40,optimizer_failures=failures),
        resources=expected,summary=summary,loading_events=jumps,
        seed_check=dict(new_role_states=5,compared_v15_pilot_and_production_role_states=len(prior_states),collisions=0),
        limits=['One protocol-fixed acquired model and four fixed tasks, not eight new independent learned bodies; no population SE.',
            'Current q/v are continuous at matched-velocity external attachment; generalized momentum and kinetic energy jump. Energy-work conservation is asserted only between changes.',
            'First loaded command and all observations through the event are bit-identical. The first load-dependent observation arrives after the changed plant executes a command.',
            'All nine fixed PD settings retained; no optimized baseline or post-outcome winner.',
            'Frozen control success can reject necessity of adaptation in this limited instrument; no adaptation, detector or probe has been evaluated.',
            'Rest-reference continuous torque feasibility is not a sampled-input endpoint or arbitrary moving-state recovery certificate.'])


if __name__=='__main__':
    with threadpool_limits(limits=1):report=main()
    print(json.dumps(report,indent=2,allow_nan=False))
