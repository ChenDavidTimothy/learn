#!/usr/bin/env python3
"""Independent vectorized Hamiltonian/feature/complex-step pilot audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DT, SD = .1, .001


def integrate(states, commands, dt=DT, steps=32):
    z, u = np.atleast_2d(states), np.atleast_2d(commands)
    c = np.cos(z[:, 1]); v = z[:, 2:]
    y = np.c_[z[:, :2], (3+2*c)*v[:, 0]+(1+c)*v[:, 1], (1+c)*v[:, 0]+v[:, 1]]
    def rhs(s):
        c, sn = np.cos(s[:, 1]), np.sin(s[:, 1]); d = 2-c*c
        a = (s[:, 2]-(1+c)*s[:, 3])/d
        b = (-(1+c)*s[:, 2]+(3+2*c)*s[:, 3])/d
        return np.c_[a, b, u[:, 0], u[:, 1]-sn*(a*a+a*b)]
    h = dt/steps
    for _ in range(steps):
        a=rhs(y); b=rhs(y+.5*h*a); c=rhs(y+.5*h*b); d=rhs(y+h*c)
        y += h*(a+2*b+2*c+d)/6
    c=np.cos(y[:, 1]); d=2-c*c
    return np.c_[y[:, :2], (y[:, 2]-(1+c)*y[:, 3])/d,
                 (-(1+c)*y[:, 2]+(3+2*c)*y[:, 3])/d]


def adaptive(initial, u):
    q=initial[:2];v=initial[2:];c=np.cos(q[1])
    p=np.array([(3+2*c)*v[0]+(1+c)*v[1],(1+c)*v[0]+v[1]])
    def rhs(t,s):
        c,sn=np.cos(s[1]),np.sin(s[1]);d=2-c*c
        v1=(s[2]-(1+c)*s[3])/d;v2=(-(1+c)*s[2]+(3+2*c)*s[3])/d
        return [v1,v2,u[0],u[1]-sn*(v1*v1+v1*v2)]
    solved=solve_ivp(rhs,[0.,DT],np.r_[q,p],method='Radau',rtol=2e-12,atol=1e-14)
    assert solved.success
    s=solved.y[:,-1];c=np.cos(s[1]);d=2-c*c
    return np.r_[s[:2],(s[2]-(1+c)*s[3])/d,(-(1+c)*s[2]+(3+2*c)*s[3])/d]


def basis(history, actions, current, model):
    h, kind = model['h'], model['kind']
    history, actions, current = np.asarray(history), np.asarray(actions), np.asarray(current)
    if history.ndim == 2:history=history[None];actions=actions[None];current=current[None]
    q=history[:, -1]; n=len(q)
    if kind=='linear':p=np.ones((n, 1),dtype=q.dtype)
    else:
        terms=[np.ones(n,dtype=q.dtype)]
        for j in range(2):
            for k in [1,2]:terms.extend([np.sin(k*q[:,j]),np.cos(k*q[:,j])])
        for a in [np.sin,np.cos]:
            for b in [np.sin,np.cos]:terms.append(a(q[:,0])*b(q[:,1]))
        p=np.stack(terms,axis=1)
    b=[np.ones(n,dtype=q.dtype),current[:,0],current[:,1]]
    if h:
        if kind=='poly':
            t=np.arange(-h,1)*DT
            weights=np.linalg.pinv(np.c_[np.ones(h+1),t,.5*t*t])
            compressed=np.einsum('ij,njk->nik',weights[1:],history[:,-h-1:])
            b.extend(compressed.reshape(n,4).T);motion=compressed[:,0]
        else:
            for k in range(1,h+1):b.extend(((q-history[:,-1-k])/(k*DT)).T)
            motion=(q-history[:,-2])/DT
        for k in range(1,h+1):b.extend(actions[:,-k].T)
        if kind!='linear':b.extend([motion[:,0]**2,motion[:,0]*motion[:,1],motion[:,1]**2])
    return (p[:,:,None]*np.stack(b,axis=1)[:,None,:]).reshape(n,-1)


def predict(history, actions, current, model):
    return np.asarray(history)[:,-1]+DT*basis(history,actions,current,model)@np.asarray(model['coef'])


def rows(episodes, model):
    history=[];actions=[];commands=[];target=[];truth=[]
    for e in episodes:
        o=np.array(e['observed']);u=np.array(e['commands']);s=np.array(e['states_evaluator_only'])
        for t in range(8,len(u)):
            history.append(o[t-8:t+1]);actions.append(u[t-8:t]);commands.append(u[t])
            target.append((o[t+1]-o[t])/DT);truth.append(s[t+1,:2])
    history=np.array(history);actions=np.array(actions);commands=np.array(commands)
    return basis(history,actions,commands,model),np.array(target),history[:,-1],np.array(truth)


def residual(plan, model, history, actions, goal):
    plans=np.atleast_2d(plan);n=len(plans);d=plans.dtype
    hist=np.broadcast_to(history,(n,*np.shape(history))).astype(d).copy()
    past=np.broadcast_to(actions,(n,*np.shape(actions))).astype(d).copy()
    out=[]
    for t in range(plans.shape[1]//2):
        u=plans[:,2*t:2*t+2];q=predict(hist,past,u,model);motion=(q-hist[:,-1])/DT
        out.extend([q-goal,np.sqrt(.3)*motion,np.sqrt(.003)*u])
        hist=np.concatenate([hist,q[:,None]],axis=1);past=np.concatenate([past,u[:,None]],axis=1)
    return np.concatenate(out,axis=1)


def metrics(states, commands, goals):
    s=np.asarray(states)[9:];u=np.asarray(commands)[8:];g=np.asarray(goals);error=s[:,:2]-g
    late=np.r_[50:60,110:120,170:180]
    return dict(position_mse=float(np.mean(np.sum(error*error,axis=1))),velocity_rms=float(np.sqrt(np.mean(s[:,2:]**2))),
        action_energy=float(np.mean(u*u)),late_position_rmse=float(np.sqrt(np.mean(error[late]**2))),
        late_velocity_rms=float(np.sqrt(np.mean(s[late,2:]**2))),
        precision_success_by_goal=[bool(np.all(np.abs(error[k*60+50:k*60+60])<=.03) and np.all(np.abs(s[k*60+50:k*60+60,2:])<=.03)) for k in range(3)],
        saturated_coordinates=int(np.sum(np.abs(u)>=.999999)))


def main(result_name='continuous_history_v15_pilot.json', production=False):
    path=ROOT/result_name;r=json.loads(path.read_text());error={}
    rep=r['meta']['replicate'];nsteps=96 if production else 64
    source_name='continuous_history_v15.py' if production else 'continuous_history_v15_pilot.py'
    poses=[[-2.,-2.1],[-.5,-.7],[.8,.7],[2.,2.1]] if production else [[0.,np.pi/2],[1.,.3]]
    assert 'resources' in r
    def check(key,a,b):error[key]=max(error.get(key,0.),float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    seed_paths=set()
    episodes=[]
    for role,key,count in [(1,'training',48),(2,'validation',12),(3,'testing',12)]:
        assert len(r[key])==count
        for j,e in enumerate(r[key]):
            expected=[20261020,15,610,rep,role,j];assert e['seed_path']==expected
            assert tuple(expected) not in seed_paths;seed_paths.add(tuple(expected))
            rng=np.random.default_rng(np.random.SeedSequence(expected))
            z=np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-.15,.15,2)]
            amplitude=1. if production and j%2 else .35
            u=np.repeat(rng.uniform(-amplitude,amplitude,(nsteps//2,2)),2,axis=0)
            check('initial_states',z,e['states_evaluator_only'][0]);check('exogenous_commands',u,e['commands'])
            noise=rng.normal(0,SD,(nsteps+1,2));check('sensor_RNG',np.array(e['states_evaluator_only'])[:,:2]+noise,e['observed'])
            episodes.append(e)
    z=np.array([e['states_evaluator_only'][0] for e in episodes])
    z64=z.copy()
    for t in range(nsteps):
        u=np.array([e['commands'][t] for e in episodes]);z=integrate(z,u);z64=integrate(z64,u,steps=64)
        check('physics32vs64',z,z64);check('acquisition_physics',z64,[e['states_evaluator_only'][t+1] for e in episodes])
    candidates=r['candidates'];assert len(candidates)==30
    for model in candidates:
        X,Y,_,_=rows(r['training'],model);V,target,q,truth=rows(r['validation'],model)
        scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),.01);A=X/scale
        G=A.T@A/len(A);rhs=A.T@Y/len(A);coef=np.array(model['coef']);W=coef*scale[:,None]
        check('fit_scaling',scale,model['scale'])
        stationarity=np.max(np.abs((G+model['ridge']*np.eye(len(G)))@W-rhs))
        check('fit_stationarity',stationarity,0.)
        check('stored_stationarity',stationarity,model['stationarity_max'])
        predicted=q+DT*V@coef
        check('validation_observation',np.mean(np.sum((DT*(V@coef-target))**2,axis=1)),model['validation_observation_mse'])
        check('validation_true_diagnostic',np.mean(np.sum((predicted-truth)**2,axis=1)),model['validation_true_position_mse_diagnostic'])
    for method,index in r['selected_candidate_ids'].items():
        kind,h={'pose':('raw',[0]),'one_delay':('raw',[1]),'rich_delay':('raw',[4]),'selected_delay':('raw',[0,1,2,4]),'linear_arx':('linear',[1,2,4]),'polynomial':('poly',[2,4,8])}[method]
        eligible=[i for i,m in enumerate(candidates) if m['kind']==kind and m['h'] in h]
        assert index==min(eligible,key=lambda i:candidates[i]['validation_observation_mse'])
        model=candidates[index];V,target,q,truth=rows(r['testing'],model);pred=q+DT*V@np.array(model['coef'])
        record=r['forecasts'][method]
        check('forecast_one_step',np.mean(np.sum((pred-truth)**2,axis=1)),record['one_step_true_position_mse'])
        check('forecast_observation',np.mean(np.sum((DT*(V@np.array(model['coef'])-target))**2,axis=1)),record['one_step_observation_mse'])
        roll=[]
        for e in r['testing']:
            o=np.array(e['observed']);a=np.array(e['commands']);truth=np.array(e['states_evaluator_only'])
            for start in range(8,len(a)-9,10):
                hist=o[:start+1][None];past=a[:start][None]
                for k in range(10):
                    prediction=predict(hist,past,a[start+k][None],model)
                    roll.append(np.sum((prediction[0]-truth[start+k+1,:2])**2))
                    hist=np.concatenate([hist,prediction[:,None]],axis=1);past=np.concatenate([past,a[start+k][None,None]],axis=1)
        check('forecast_recursive',np.mean(roll),record['ten_step_recursive_true_position_mse'])
        assert record['one_step_rows']==12*(nsteps-8) and record['recursive_rows']==120*len(range(8,nsteps-9,10))
    plan_count=gradient_count=0;solver_failures=0
    controls=[]
    for method,es in r['controls'].items():
        model=candidates[r['selected_candidate_ids'][method]]
        for item,e in enumerate(es):
            path=[20261020,15,610,rep,4,item];assert e['seed_path']==path
            seed_paths.add(tuple(path));noise=np.random.default_rng(np.random.SeedSequence(path)).normal(0,SD,(189,2))
            s=np.array(e['states_evaluator_only']);o=np.array(e['observed']);a=np.array(e['commands']);g=np.array(e['goals'])
            check('control_sensor_RNG',s[:,:2]+noise,o)
            q0=np.array(poses[item]);check('control_initial',s[0],np.r_[q0,[.04,-.02]])
            check('goals',g,np.repeat(q0+np.array([[.2,-.15],[-.15,.15],[.1,.1]]),60,axis=0))
            assert np.max(np.abs(a))<=1.+1e-12 and np.all(a[:8]==0.)
            warm=np.zeros(30)
            for t,p in enumerate(e['plans']):
                plan=np.array(p['plan']).ravel();assert len(plan)==30 and np.max(np.abs(plan))<=1.+1e-12
                check('issued_plan',plan[:2],a[t+8])
                hist=o[:t+9];past=a[:t+8]
                rr=residual(plan,model,hist,past,g[t])[0];initial=residual(warm,model,hist,past,g[t])[0]
                check('selected_plan_cost',rr@rr,p['selected_cost']);check('initial_plan_cost',initial@initial,p['initial_cost'])
                assert p['selected_cost']<=p['initial_cost']+1e-9
                # Every stored projected gradient is independently checked via
                # batched complex-step differentiation of the whole rollout.
                eps=1e-25;perturbed=np.tile(plan.astype(complex),(30,1))+1j*eps*np.eye(30)
                J=residual(perturbed,model,hist,past,g[t]).imag.T/eps
                grad=2*J.T@rr;pg=plan-np.clip(plan-grad,-1.,1.)
                check('projected_gradient',np.max(np.abs(pg)),p['projected_gradient_max']);gradient_count+=1
                warm=np.r_[plan[2:],0.,0.];plan_count+=1;solver_failures+=not p['success']
            computed=metrics(s,a,g)
            for key,value in computed.items():
                if isinstance(value,list):assert value==e['metrics'][key]
                else:check('control_metrics',value,e['metrics'][key])
            assert e['metrics']['solver_failures']==sum(not p['success'] for p in e['plans'])
            check('maximum_PG',max(p['projected_gradient_max'] for p in e['plans']),e['metrics']['maximum_projected_gradient'])
            controls.append(e)
    z=np.array([e['states_evaluator_only'][0] for e in controls]);z64=z.copy()
    highspeed=[i for i,e in enumerate(controls) if np.max(np.abs(np.array(e['states_evaluator_only'])[:,2:]))>2.]
    adaptive_states={i:np.array(controls[i]['states_evaluator_only'][0]) for i in highspeed}
    coarse_highspeed_difference=adaptive_global_error=0.
    for t in range(188):
        u=np.array([e['commands'][t] for e in controls]);z=integrate(z,u);z64=integrate(z64,u,steps=64)
        low=[i for i in range(len(controls)) if i not in highspeed]
        check('physics32vs64',z[low],z64[low]);check('control_physics',z64[low],[controls[i]['states_evaluator_only'][t+1] for i in low])
        if highspeed:coarse_highspeed_difference=max(coarse_highspeed_difference,float(np.max(np.abs(z[highspeed]-z64[highspeed]))))
        for i in highspeed:
            source=np.array(controls[i]['states_evaluator_only'])
            check('highspeed_Radau_local',adaptive(source[t],u[i]),source[t+1])
            adaptive_states[i]=adaptive(adaptive_states[i],u[i])
            adaptive_global_error=max(adaptive_global_error,float(np.max(np.abs(adaptive_states[i]-source[t+1]))))
    expected_controls=(24 if production else 10)
    assert plan_count==gradient_count==180*expected_controls and len(seed_paths)==72+len(poses)
    for key,value in error.items():assert value<2e-7,(key,value)
    assert r['resources']['acquisition_steps']==72*nsteps and r['resources']['control_steps']==expected_controls*188
    return dict(status='INDEPENDENTLY VERIFIED CONTINUOUS-HISTORY EXPERIMENT' if production else 'INDEPENDENTLY VERIFIED EXPLORATORY CONTINUOUS-HISTORY PILOT',
        replicate=rep,
        artifact_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [source_name,result_name]},
        maximum_absolute_errors=error,independent_physics='Hamiltonian RK4 at32/64 substeps',
        highspeed_convergence_diagnostic=dict(episodes=len(highspeed),coarse_RK32vs64_global=coarse_highspeed_difference,
            adaptive_Hamiltonian_Radau_global_error=adaptive_global_error,
            scope='Fast failed pose controllers checked by adaptive Radau locally and globally; coarse RK convergence is not asserted there.'),
        checked=dict(acquisition_trajectories=72,acquisition_intervals=72*nsteps,distinct_SeedSequence_paths=len(seed_paths),
            candidate_fits=30,controls=expected_controls,control_intervals=expected_controls*188,plans=plan_count,full_complex_step_plan_gradients=gradient_count,solver_failures=solver_failures),
        selected_candidate_ids=r['selected_candidate_ids'],forecasts=r['forecasts'],
        control_metrics={key:[e['metrics'] for e in es] for key,es in r['controls'].items()},
        limitations=['One engineering replicate, known feature/history menus and periodic position semantics; selected lag is conventional model selection.',
            'Validation selection uses observation risk only; truth diagnostics do not select candidates. Entire role-tagged trajectories are disjoint.',
            'Holding random commands for two samples creates current/past-input dependence at alternate steps. Full command sequence is exogenous; an omitted-history model need not identify interventional dynamics.',
            'No process noise, no global MPC solution or uniform feasibility certificate; forecast mean composition and motion penalties are approximations.',
            'All controls share sensor streams, not physical trajectories. Goal switches do not reset history, warm plan or body state.',
            'Exact optimizer internal iterates/failed proposal are not stored; all selected plans, costs, gradients, warm costs and issued actions are verified.'])


if __name__=='__main__':
    with threadpool_limits(limits=1):report=main()
    print(json.dumps(report,indent=2,allow_nan=False))
