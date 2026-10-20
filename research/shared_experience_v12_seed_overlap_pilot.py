#!/usr/bin/env python3
"""Frozen shared-data dynamic policy CEM versus segmented OE/EKF/MPC.

See shared_experience_v12_design.md. NPZ retains every training segment;
JSON retains CEM draws, selections, fit records and every test trajectory/plan.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from goal_control_v10 import transition
from nonlinear_identification_v9 import STARTS, LOWER, UPPER, stat

SEED=20261020+1900000
PLANTS=[(.65,lam,c,1.2) for lam in (.2,.7) for c in (0.,.2)]
N=86; SD=.2; MOTOR=.03; GENERATIONS=8; CANDIDATES=24; REPLICATES=8
TRAIN_GOALS=((1.,-1.,.5),(-1.,1.,-.5))
TEST_GOALS=((2.,-2.,1.),(6.,-4.,3.))
CHECKPOINTS=(2,8)
LOW=np.array([math.log(.005),math.log(.02),math.log(.1/.9),-2.,-2.,-2.])
HIGH=np.array([math.log(.5),math.log(4.),math.log(.98/.02),2.,2.,2.])
FLOOR=np.array([.15,.15,.15,.03,.03,.03])


def targets(g):return np.array([g[0] if t<6 else g[1] if t<28 else g[2] for t in range(N)])


def decode(p):return [float(np.exp(p[0])),float(np.exp(p[1])),float(1/(1+np.exp(-p[2]))),*map(float,p[3:])]


def score(states,o,u,g):
    scale=1+np.mean(g*g);s=np.asarray(states)[1:];o=np.asarray(o);u=np.asarray(u)
    actual=float(np.mean((s[:,0]-g)**2+.3*s[:,1]**2+.03*u*u)/scale)
    observed=float(np.mean((o[1:]-g)**2+.3*np.diff(o)**2+.03*u*u)/scale)
    return dict(actual_cost=actual,observed_cost=observed,
        gap_minus_offset=float(observed-actual-1.6*SD*SD/scale),
        late_position_rmse=float(np.sqrt(np.mean((s[-10:,0]-g[-10:])**2))),
        late_velocity_rms=float(np.sqrt(np.mean(s[-10:,1]**2))),
        stationary_success=bool(np.all(np.abs(s[-10:,0]-g[-10:])<=.3)&np.all(np.abs(s[-10:,1])<=.15)))


def direct_episode(theta,p,g,seed,motor_sd):
    rng=np.random.default_rng(seed);e=rng.normal(0,SD,N+1);xi=rng.normal(0,motor_sd,N)
    kp,kd,alpha,bv,bu1,bu2=decode(p)
    x=v=m=sx=sv=prevu=olderu=0.;s=[[0.,0.,0.]];o=[float(e[0])];u=[];g=targets(g)
    rho,lam,c,beta=theta
    for t in range(N):
        oldx=sx;oldv=sv;sx+=alpha*(o[-1]-sx);sv+=alpha*(sx-oldx-sv)
        a=float(np.clip(kp*(g[t]-sx)-kd*sv+bv*oldv-bu1*prevu-bu2*olderu,-1.,1.))
        olderu=prevu;prevu=a;u.append(a)
        vn=rho*v-c*math.tanh(v)+m;m=lam*m+beta*a+xi[t];v=vn;x+=v
        s.append([x,v,m]);o.append(float(x+e[t+1]))
    return np.array(s),np.array(o),np.array(u)


def train(theta,seed):
    rng=np.random.default_rng(seed);mean=np.array([math.log(.08),math.log(.6),math.log(.6/.4),0.,0.,0.])
    sd=np.array([1.,1.,1.,.3,.3,.3]);winner=None;records=[];data=[];checkpoints={}
    for generation in range(GENERATIONS):
        ps=np.clip(rng.normal(mean,sd,(CANDIDATES,6)),LOW,HIGH)
        ps[0]=mean if winner is None else winner.copy()
        if generation==0:ps[1]=[math.log(.02),0.,math.log(.8/.2),0.,0.,0.]
        fitness=[]
        for j,p in enumerate(ps):
            values=[]
            for k,g in enumerate(TRAIN_GOALS):
                stream=seed+1000000+100*(CANDIDATES*generation+j)+k
                s,o,u=direct_episode(theta,p,g,stream,0.)
                # Learner reads only o,u,g. True s is not passed to fitness.
                goal=targets(g);values.append(float(np.mean((o[1:]-goal)**2+.3*np.diff(o)**2+.03*u*u)/(1+np.mean(goal**2))))
                data.append((s,o,u))
            fitness.append(float(np.mean(values)))
        elite=np.argsort(fitness,kind='stable')[:6];best=int(np.argmin(fitness));winner=ps[best].copy()
        records.append(dict(generation=generation,coordinates=ps.tolist(),observed_fitness=fitness,
                            elite=elite.tolist(),winner_index=best))
        mean=ps[elite].mean(axis=0);sd=np.maximum(ps[elite].std(axis=0),FLOOR)
        if generation+1 in CHECKPOINTS:checkpoints[generation+1]=winner.copy()
    return records,checkpoints,dict(states=np.array([a[0] for a in data]),
        observations=np.array([a[1] for a in data]),commands=np.array([a[2] for a in data]))


def segmented(theta,u,jac=True):
    """Vectorized independent known-zero-reset episodes. No concatenation."""
    b,n=u.shape;rho,lam,c,beta=theta
    x=np.zeros((b,n+1));v=np.zeros(b);m=np.zeros(b)
    dv=np.zeros((b,4));dm=dv.copy();dx=dv.copy();J=np.zeros((b,n+1,4))
    for t in range(n):
        h=np.tanh(v);vn=rho*v-c*h+m
        if jac:
            dvn=(rho-c*(1-h*h))[:,None]*dv+dm
            dvn[:,0]+=v;dvn[:,2]-=h
            dmn=lam*dm;dmn[:,1]+=m;dmn[:,3]+=u[:,t]
            dx=dx+dvn;J[:,t+1]=dx;dv=dvn;dm=dmn
        m=lam*m+beta*u[:,t];v=vn;x[:,t+1]=x[:,t]+v
    return x,J


def fit_segments(o,u,linear):
    cache={};columns=[0,1,3] if linear else [0,1,2,3]
    def calc(p):
        if 'p' not in cache or not np.array_equal(p,cache['p']):
            theta=[p[0],p[1],0.,p[2]] if linear else p
            x,J=segmented(theta,u)
            cache.update(p=p.copy(),r=(x-o).reshape(-1),J=J[:,:,columns].reshape(-1,len(columns)))
        return cache
    records=[]
    for start in STARTS:
        f=least_squares(lambda p:calc(p)['r'],np.array(start)[columns],jac=lambda p:calc(p)['J'],
            bounds=(np.array(LOWER)[columns],np.array(UPPER)[columns]),max_nfev=160,
            ftol=1e-9,xtol=1e-9,gtol=1e-9)
        theta=[f.x[0],f.x[1],0.,f.x[2]] if linear else f.x.tolist()
        singular=np.linalg.svd(calc(f.x)['J'],compute_uv=False)
        records.append(dict(theta=list(map(float,theta)),sse=float(2*f.cost),success=bool(f.success),
            status=int(f.status),nfev=int(f.nfev),optimality=float(f.optimality),singular_values=singular.tolist()))
    best=min(records,key=lambda r:r['sse']);return np.array(best['theta']),records


def plan(state,goal,theta,warm):
    n=len(warm);cache={}
    def calc(a):
        if 'a' not in cache or not np.array_equal(cache['a'],a):
            z=state.copy();dz=np.zeros((3,n));r=[];J=[]
            for t in range(n):
                z,A=transition(z,a[t],theta);dz=A@dz;dz[2,t]+=theta[3]
                direct=np.zeros(n);direct[t]=math.sqrt(.03)
                r.extend((z[0]-goal,math.sqrt(.3)*z[1],math.sqrt(.03)*a[t]))
                J.extend((dz[0].copy(),math.sqrt(.3)*dz[1],direct))
            cache.update(a=a.copy(),r=np.array(r),J=np.array(J))
        return cache
    initial=calc(warm)['r'].copy();f=least_squares(lambda a:calc(a)['r'],warm,jac=lambda a:calc(a)['J'],
        bounds=(-1.,1.),max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-8)
    a=f.x if np.all(np.isfinite(f.x)) and 2*f.cost<=initial@initial+1e-8 else warm
    values=calc(a);grad=2*values['J'].T@values['r'];project=a-np.clip(a-grad,-1.,1.)
    return float(a[0]),np.r_[a[1:],0.],dict(success=bool(f.success),status=int(f.status),nfev=int(f.nfev),
        initial_cost=float(initial@initial),selected_cost=float(values['r']@values['r']),
        projected_gradient_max=float(np.max(np.abs(project))),actions=a.tolist())


def model_episode(theta,model,g,seed):
    rng=np.random.default_rng(seed);e=rng.normal(0,SD,N+1);xi=rng.normal(0,MOTOR,N);g=targets(g)
    z=np.zeros(3);estimate=np.zeros(3);P=np.zeros((3,3));Q=np.diag([0.,0.,MOTOR**2]);warm=np.zeros(16)
    s=[z.tolist()];o=[float(e[0])];u=[];est=[];plans=[]
    for t in range(N):
        K=P[:,0]/(P[0,0]+SD**2);estimate=estimate+K*(o[-1]-estimate[0])
        corr=np.eye(3);corr[:,0]-=K;P=corr@P@corr.T+SD**2*np.outer(K,K)
        est.append(estimate.tolist());warm=warm[:min(16,N-t)]
        a,warm,info=plan(estimate,g[t],model,warm);plans.append(info);u.append(a)
        z,_=transition(z,a,theta);z[2]+=xi[t];s.append(z.tolist());o.append(float(z[0]+e[t+1]))
        estimate,A=transition(estimate,a,model);P=A@P@A.T+Q
    return dict(**score(s,o,u,g),states=s,observations=o,commands=u,estimates=est,plans=plans,seed=seed)


def checks():
    rng=np.random.default_rng(8317);u=rng.uniform(-1,1,(3,12));p=np.array(PLANTS[-1]);x,J=segmented(p,u)
    errors=[]
    for i in range(4):
        d=np.zeros(4);d[i]=1e-6
        errors.append(float(np.max(np.abs((segmented(p+d,u,False)[0]-segmented(p-d,u,False)[0])/2e-6-J[:,:,i]))))
    assert max(errors)<1e-5
    return dict(segmented_jacobian_max_error=max(errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-prefix',default='research/shared_experience_v12_seed_overlap_pilot');args=parser.parse_args()
    rows=[];arrays={};times=[]
    with threadpool_limits(limits=1):
        for pi,theta in enumerate(PLANTS):
            for i in range(REPLICATES):
                seed=SEED+100000*pi+10000*i;t0=time.perf_counter();records,cp,data=train(theta,seed)
                times.append(dict(plant=pi,replicate=i,stage='physical_training_simulation',seconds=time.perf_counter()-t0))
                key=f'p{pi}_s{i}';arrays.update({key+'_'+k:v for k,v in data.items()});stages=[]
                for generations,p in cp.items():
                    count=generations*CANDIDATES*2;fitted={};fitrecords={}
                    for label in ('linear','nonlinear'):
                        t0=time.perf_counter();fitted[label],fitrecords[label]=fit_segments(data['observations'][:count],data['commands'][:count],label=='linear')
                        times.append(dict(plant=pi,replicate=i,generations=generations,stage='fit_'+label,seconds=time.perf_counter()-t0))
                    tests=[]
                    for j,g in enumerate(TEST_GOALS):
                        stream=seed+20000000+j;s,o,u=direct_episode(theta,p,g,stream,MOTOR)
                        controllers={'direct':dict(**score(s,o,u,targets(g)),states=s.tolist(),observations=o.tolist(),commands=u.tolist(),seed=stream)}
                        for label in ('linear','nonlinear'):
                            t0=time.perf_counter();controllers[label]=model_episode(theta,fitted[label],g,stream)
                            times.append(dict(plant=pi,replicate=i,generations=generations,stage='test_control_'+label,task=j,seconds=time.perf_counter()-t0))
                        tests.append(dict(goal=g,controllers=controllers))
                    stages.append(dict(generations=generations,training_steps=count*N,training_resets=count,
                        selected_coordinates=p.tolist(),models={k:v.tolist() for k,v in fitted.items()},fits=fitrecords,tests=tests))
                rows.append(dict(plant_index=pi,true_theta_evaluator_only=theta,replicate=i,seed=seed,cem=records,checkpoints=stages))
                print(f'completed plant{pi} seed{i}',file=sys.stderr,flush=True)
    summary=[]
    for pi in range(len(PLANTS)):
        for stage,generation in enumerate(CHECKPOINTS):
            current=[r['checkpoints'][stage] for r in rows if r['plant_index']==pi];controllers={}
            for label in ('direct','linear','nonlinear'):
                a=[np.mean([t['controllers'][label]['actual_cost'] for t in c['tests']]) for c in current]
                controllers[label]=dict(cost=stat(a),stationary_successes=sum(t['controllers'][label]['stationary_success'] for c in current for t in c['tests']),trials=16)
            differences={}
            for label in ('linear','nonlinear'):
                d=[np.mean([t['controllers']['direct']['actual_cost']-t['controllers'][label]['actual_cost'] for t in c['tests']]) for c in current]
                differences['direct_minus_'+label]=stat(d)
            summary.append(dict(plant=pi,theta=PLANTS[pi],generations=generation,controllers=controllers,paired_differences=differences))
    prefix=Path(args.output_prefix);np.savez_compressed(str(prefix)+'_data.npz',**arrays)
    digest=hashlib.sha256(Path(str(prefix)+'_data.npz').read_bytes()).hexdigest()
    out=dict(meta=dict(version=12,status='EXPLORATORY SHARED-EXPERIENCE COMPARISON; independent audit pending',seed=SEED,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,plant_grid=PLANTS,
        replicates=REPLICATES,n=N,sensor_sd=SD,train_motor_sd=0.,test_motor_sd=MOTOR,
        generations=GENERATIONS,candidates=CANDIDATES,checkpoints=CHECKPOINTS,train_goals=TRAIN_GOALS,test_goals=TEST_GOALS,
        train_steps_per_seed=GENERATIONS*CANDIDATES*2*N,train_resets_per_seed=GENERATIONS*CANDIDATES*2,
        test_steps_per_seed=2*3*2*N,tests_consume_additional_experience_but_do_not_update_learners=True,
        data_sha256=digest,design='shared_experience_v12_design.md',all_function_families_noise_laws_and_resets_supplied=True,
        limitations='same data; unequal compute/prior vocabulary; CEM not optimal direct learning; toy four-plant family only'),
        checks=checks(),runs=rows,summary=summary)
    Path(str(prefix)+'_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    Path(str(prefix)+'_timing.json').write_text(json.dumps(times,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
