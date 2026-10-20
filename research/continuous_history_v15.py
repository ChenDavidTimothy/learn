#!/usr/bin/env python3
"""Exploratory continuous position-history learning/control; see frozen design.

Ordinary prediction/planning never calls plant RHS or receives hidden q/v.
The plant is imported only for acquisition, external execution and scoring.
"""
from __future__ import annotations
import argparse, json, time, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from coupled_context_v14_pilot import flow

DT=.1
SD=.001
MAX_H=8
HORIZON=15
RIDGES=(1e-8,1e-5,1e-2)
METHODS=('pose','one_delay','rich_delay','selected_delay','linear_arx','polynomial')


def random_role(rep,role,item):
    path=[20261020,15,610,int(rep),int(role),int(item)]
    return np.random.default_rng(np.random.SeedSequence(path)),path


def acquire(rep,role,count,steps=96):
    episodes=[]
    for j in range(count):
        rng,path=random_role(rep,role,j)
        z=np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-.15,.15,2)]
        amplitude=.35 if j%2==0 else 1.
        commands=np.repeat(rng.uniform(-amplitude,amplitude,(steps//2,2)),2,axis=0)
        states=[z.copy()];nfev=0
        for u in commands:
            z,n=flow(z[:2],z[2:],u,DT);states.append(z.copy());nfev+=n
        states=np.array(states);noise=rng.normal(0,SD,(steps+1,2))
        episodes.append(dict(seed_path=path,states_evaluator_only=states,
            observed=states[:,:2]+noise,commands=commands,plant_nfev=nfev))
    return episodes


def context(q,linear=False):
    if linear:return np.ones(1),np.zeros((1,2))
    p=[1.];dp=[np.zeros(2)]
    for j in range(2):
        for k in (1,2):
            d=np.zeros(2);d[j]=k*np.cos(k*q[j]);p.append(np.sin(k*q[j]));dp.append(d)
            d=np.zeros(2);d[j]=-k*np.sin(k*q[j]);p.append(np.cos(k*q[j]));dp.append(d)
    for a in (0,1):
        for b in (0,1):
            f0=np.sin(q[0]) if a==0 else np.cos(q[0])
            f1=np.sin(q[1]) if b==0 else np.cos(q[1])
            d0=np.cos(q[0]) if a==0 else -np.sin(q[0])
            d1=np.cos(q[1]) if b==0 else -np.sin(q[1])
            p.append(f0*f1);dp.append(np.array([d0*f1,f0*d1]))
    return np.array(p),np.array(dp)


def base(history,actions,u,h,kind):
    """Return b and derivatives wrt position history, past u and present u.

    history/actions chronological, position history includes current reading.
    Past differences are an invertible re-expression of raw delays given o_t.
    """
    q=np.asarray(history)[-h-1:] if h else np.asarray(history)[-1:]
    a=np.asarray(actions).reshape(-1,2)[-h:] if h else np.zeros((0,2))
    npos=h+1;nb=3
    b=[1.,*u];Dq=[np.zeros((npos,2)) for _ in range(3)]
    Da=[np.zeros((h,2)) for _ in range(3)]
    Du=[np.zeros(2),np.array([1.,0.]),np.array([0.,1.])]
    motion=None;motion_d=None
    if h:
        if kind=='poly':
            times=np.arange(-h,1)*DT
            # Degree2 causal polynomial, current slope and second derivative.
            weights=np.linalg.pinv(np.stack((np.ones(h+1),times,.5*times**2),axis=1))[1:]
            summaries=weights@q
            for row in range(2):
                for j in range(2):
                    d=np.zeros((npos,2));d[:,j]=weights[row]
                    b.append(summaries[row,j]);Dq.append(d);Da.append(np.zeros((h,2)));Du.append(np.zeros(2))
            motion=summaries[0];motion_d=np.zeros((2,npos,2))
            for j in range(2):motion_d[j,:,j]=weights[0]
        else:
            for k in range(1,h+1):
                for j in range(2):
                    d=np.zeros((npos,2));d[-1,j]=1/(k*DT);d[-1-k,j]=-1/(k*DT)
                    b.append((q[-1,j]-q[-1-k,j])/(k*DT));Dq.append(d)
                    Da.append(np.zeros((h,2)));Du.append(np.zeros(2))
            motion=(q[-1]-q[-2])/DT;motion_d=np.zeros((2,npos,2))
            for j in range(2):motion_d[j,-1,j]=1/DT;motion_d[j,-2,j]=-1/DT
        for k in range(1,h+1):
            for j in range(2):
                b.append(a[-k,j]);d=np.zeros((h,2));d[-k,j]=1
                Da.append(d);Dq.append(np.zeros((npos,2)));Du.append(np.zeros(2))
        if kind!='linear':
            for i,j in ((0,0),(0,1),(1,1)):
                b.append(motion[i]*motion[j]);Dq.append(motion_d[i]*motion[j]+motion_d[j]*motion[i])
                Da.append(np.zeros((h,2)));Du.append(np.zeros(2))
    return np.array(b),np.array(Dq),np.array(Da),np.array(Du)


def features(history,actions,u,h,kind):
    p,_=context(history[-1],kind=='linear');b,*_=base(history,actions,u,h,kind)
    return np.outer(p,b).ravel()


class Model:
    def __init__(self,h,kind,coef,ridge):
        self.h=h;self.kind=kind;self.coef=np.asarray(coef);self.ridge=ridge
        self.tensor=self.coef.reshape(1 if kind=='linear' else 13,-1,2)

    def step(self,history,actions,u,derivatives=False):
        p,dp=context(history[-1],self.kind=='linear')
        b,dq,da,du=base(history,actions,u,self.h,self.kind)
        C=np.einsum('p,pbo->bo',p,self.tensor)
        velocity=b@C;nextq=history[-1]+DT*velocity
        if not derivatives:return nextq
        Jq=DT*np.einsum('bo,bij->oij',C,dq)
        Jq[:,-1,:]+=np.eye(2)+DT*np.einsum('b,pbo,pj->oj',b,self.tensor,dp)
        Ja=DT*np.einsum('bo,bij->oij',C,da)
        Ju=DT*C.T@du
        return nextq,Jq,Ja,Ju

    def record(self):
        return dict(h=self.h,kind=self.kind,ridge=self.ridge,coef=self.coef)


def regression_rows(episodes,h,kind):
    X=[];Y=[];q=[];truth=[]
    for e in episodes:
        o=e['observed'];a=e['commands']
        # All candidates use same row times despite different memory requirements.
        for t in range(MAX_H,len(a)):
            X.append(features(o[:t+1],a[:t],a[t],h,kind))
            Y.append((o[t+1]-o[t])/DT);q.append(o[t]);truth.append(e['states_evaluator_only'][t+1,:2])
    return np.array(X),np.array(Y),np.array(q),np.array(truth)


def fit_models(train,val):
    records=[];models=[]
    for kind,hs in (('raw',(0,1,2,4)),('linear',(1,2,4)),('poly',(2,4,8))):
        for h in hs:
            X,Y,_,_=regression_rows(train,h,kind)
            V,T,q,truth=regression_rows(val,h,kind)
            scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),.01)
            A=X/scale;G=A.T@A/len(A);rhs=A.T@Y/len(A)
            for ridge in RIDGES:
                W=np.linalg.solve(G+ridge*np.eye(G.shape[0]),rhs)
                coef=W/scale[:,None];pred=q+DT*V@coef
                r=dict(kind=kind,h=h,ridge=ridge,dimension=X.shape[1],scale=scale,
                    coef=coef,validation_observation_mse=float(np.mean(np.sum((DT*(V@coef-T))**2,axis=1))),
                    validation_true_position_mse_diagnostic=float(np.mean(np.sum((pred-truth)**2,axis=1))),
                    stationarity_max=float(np.max(np.abs((G+ridge*np.eye(len(G)))@W-rhs))))
                records.append(r);models.append(Model(h,kind,coef,ridge))
    ids={}
    for method in METHODS:
        eligible=[i for i,r in enumerate(records) if
            (method=='pose' and r['kind']=='raw' and r['h']==0) or
            (method=='one_delay' and r['kind']=='raw' and r['h']==1) or
            (method=='rich_delay' and r['kind']=='raw' and r['h']==4) or
            (method=='selected_delay' and r['kind']=='raw') or
            (method=='linear_arx' and r['kind']=='linear') or
            (method=='polynomial' and r['kind']=='poly')]
        ids[method]=min(eligible,key=lambda i:records[i]['validation_observation_mse'])
    return records,models,ids


def forecast_test(model,episodes):
    single=[];roll=[];observation=[]
    for e in episodes:
        o=e['observed'];a=e['commands'];truth=e['states_evaluator_only'][:,:2]
        for t in range(MAX_H,len(a)):
            pred=model.step(o[:t+1],a[:t],a[t]);single.append(np.sum((pred-truth[t+1])**2))
            observation.append(np.sum((pred-o[t+1])**2))
        for t in range(MAX_H,len(a)-9,10):
            hist=list(o[:t+1]);past=list(a[:t])
            for k in range(10):
                pred=model.step(np.asarray(hist),np.asarray(past),a[t+k]);hist.append(pred);past.append(a[t+k])
                roll.append(np.sum((pred-truth[t+k+1])**2))
    return dict(one_step_true_position_mse=float(np.mean(single)),one_step_observation_mse=float(np.mean(observation)),
                ten_step_recursive_true_position_mse=float(np.mean(roll)),one_step_rows=len(single),recursive_rows=len(roll))


def residual_plan(flat,model,history,actions,goal,derivatives=True):
    commands=flat.reshape(-1,2);n=len(commands);ndim=2*n
    hist=[np.asarray(q) for q in history];past=[np.asarray(u) for u in actions]
    Jh=[np.zeros((2,ndim)) for _ in hist];Ja=[np.zeros((2,ndim)) for _ in past]
    residual=[];jac=[]
    for t,u in enumerate(commands):
        pred,Dq,Da,Du=model.step(np.array(hist),np.array(past),u,True)
        Ju=np.zeros((2,ndim));Ju[:,2*t:2*t+2]=np.eye(2)
        J=Du@Ju
        for k in range(model.h+1):J+=Dq[:,k,:]@Jh[-model.h-1+k]
        for k in range(model.h):J+=Da[:,k,:]@Ja[-model.h+k]
        motion=(pred-hist[-1])/DT;Jmotion=(J-Jh[-1])/DT
        residual.extend(pred-goal);residual.extend(np.sqrt(.3)*motion);residual.extend(np.sqrt(.003)*u)
        jac.extend(J);jac.extend(np.sqrt(.3)*Jmotion);jac.extend(np.sqrt(.003)*Ju)
        hist.append(pred);Jh.append(J);past.append(u);Ja.append(Ju)
    return np.array(residual),np.array(jac)


def choose(model,history,actions,goal,warm):
    cache={};calls=0
    def calc(x):
        nonlocal calls
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            r,j=residual_plan(x,model,history,actions,goal);cache.update(x=x.copy(),r=r,j=j);calls+=1
        return cache
    initial=residual_plan(warm,model,history,actions,goal)[0]
    sol=least_squares(lambda x:calc(x)['r'],warm,jac=lambda x:calc(x)['j'],bounds=(-1.,1.),
        max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    selected=sol.x if np.isfinite(sol.cost) and 2*sol.cost<=initial@initial+1e-10 else warm
    r,j=residual_plan(selected,model,history,actions,goal)
    gradient=2*j.T@r;pg=selected-np.clip(selected-gradient,-1,1)
    return selected[:2],np.r_[selected[2:],0.,0.],dict(success=bool(sol.success),status=int(sol.status),nfev=int(sol.nfev),
        residual_calls=calls+2,initial_cost=float(initial@initial),selected_cost=float(r@r),
        projected_gradient_max=float(np.max(np.abs(pg))),plan=selected.reshape(-1,2))


def control(model,rep,item,steps=180):
    q0=np.array(([-2.,-2.1],[-.5,-.7],[.8,.7],[2.,2.1])[item]);z=np.r_[q0,[.04,-.02]]
    rng,path=random_role(rep,4,item);noise=rng.normal(0,SD,(MAX_H+steps+1,2))
    states=[z.copy()];obs=[z[:2]+noise[0]];issued=[];nfev=0
    for t in range(MAX_H):
        u=np.zeros(2);z,n=flow(z[:2],z[2:],u,DT);nfev+=n
        issued.append(u);states.append(z.copy());obs.append(z[:2]+noise[t+1])
    warm=np.zeros(HORIZON*2);plans=[];goals=[]
    offsets=np.array([[.2,-.15],[-.15,.15],[.1,.1]])
    for t in range(steps):
        goal=q0+offsets[min(2,3*t//steps)];goals.append(goal)
        u,warm,info=choose(model,np.array(obs),np.array(issued),goal,warm);plans.append(info)
        z,n=flow(z[:2],z[2:],u,DT);nfev+=n
        issued.append(u);states.append(z.copy());obs.append(z[:2]+noise[MAX_H+t+1])
    states=np.array(states);a=np.array(issued);goals=np.array(goals)
    scoring=states[MAX_H+1:];error=scoring[:,:2]-goals
    late=np.concatenate([np.arange((k+1)*steps//3-10,(k+1)*steps//3) for k in range(3)])
    precision=[bool(np.all(np.abs(error[(k+1)*steps//3-10:(k+1)*steps//3])<=.03) and
        np.all(np.abs(scoring[(k+1)*steps//3-10:(k+1)*steps//3,2:])<=.03)) for k in range(3)]
    metrics=dict(position_mse=float(np.mean(np.sum(error**2,axis=1))),velocity_rms=float(np.sqrt(np.mean(scoring[:,2:]**2))),
        action_energy=float(np.mean(a[MAX_H:]**2)),late_position_rmse=float(np.sqrt(np.mean(error[late]**2))),
        late_velocity_rms=float(np.sqrt(np.mean(scoring[late,2:]**2))),precision_success_by_goal=precision,
        solver_failures=sum(not p['success'] for p in plans),saturated_coordinates=int(np.sum(np.abs(a[MAX_H:])>=.999999)),
        maximum_projected_gradient=max(p['projected_gradient_max'] for p in plans))
    return dict(seed_path=path,states_evaluator_only=states,observed=np.array(obs),commands=a,goals=goals,
        plans=plans,metrics=metrics,plant_nfev=nfev,warmup_steps=MAX_H)


def encode(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)


def check_derivatives(models,val):
    e=val[0];o=e['observed'][:20];a=e['commands'][:19];goal=o[-1]+np.array([.1,-.1])
    rng=np.random.default_rng(np.random.SeedSequence([20261020,15,610,0,9,0]));x=rng.uniform(-.2,.2,8)
    checks=[]
    for key,model in models.items():
        r,J=residual_plan(x,model,o,a,goal);fd=np.zeros_like(J)
        for k in range(len(x)):
            d=np.zeros_like(x);d[k]=1e-6
            fd[:,k]=(residual_plan(x+d,model,o,a,goal)[0]-residual_plan(x-d,model,o,a,goal)[0])/(2e-6)
        err=float(np.max(np.abs(fd-J)));checks.append(dict(method=key,finite_difference_step=1e-6,max_abs_error=err))
        assert err<2e-6,(key,err)
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-prefix',default='research/continuous_history_v15_rep100')
    ap.add_argument('--replicate',type=int,default=100);ap.add_argument('--skip-control',action='store_true');args=ap.parse_args()
    start=time.time();prefix=Path(args.output_prefix);prefix.parent.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(limits=1):
        train=acquire(args.replicate,1,48);val=acquire(args.replicate,2,12);test=acquire(args.replicate,3,12)
        print('acquisition complete',time.time()-start,flush=True)
        candidates,models,ids=fit_models(train,val);selected={key:models[i] for key,i in ids.items()}
        checks=check_derivatives(selected,val)
        forecasts={key:forecast_test(model,test) for key,model in selected.items()}
        print('fits/forecasts',ids,forecasts,flush=True)
        output=dict(meta=dict(status='EXPERIMENT COMPLETE ONLY WHEN RESOURCES PRESENT; INDEPENDENT AUDIT PENDING',
            dt=DT,sensor_sd=SD,horizon=HORIZON,replicate=args.replicate,seed_paths='SeedSequence [20261020,15,610,rep,role,item]',
            python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,design='continuous_history_v15_design.md'),
            training=train,validation=val,testing=test,candidates=candidates,selected_candidate_ids=ids,
            derivative_checks=checks,forecasts=forecasts,controls={},timing_seconds={})
        # Save acquired data and models before expensive controls, including failures.
        prefix.with_suffix('.json').write_text(json.dumps(output,default=encode,indent=2,allow_nan=False)+'\n')
        if not args.skip_control:
            for key,model in selected.items():
                output['controls'][key]=[control(model,args.replicate,j) for j in range(4)]
                print(key,[e['metrics'] for e in output['controls'][key]],'elapsed',time.time()-start,flush=True)
                prefix.with_suffix('.json').write_text(json.dumps(output,default=encode,indent=2,allow_nan=False)+'\n')
        output['timing_seconds']['wall']=time.time()-start
        output['resources']=dict(acquisition_trajectories=72,acquisition_steps=72*96,acquisition_seconds=72*96*DT,
            acquisition_scalar_sensor_readings=72*97*2,candidates=len(candidates),control_episodes=sum(len(es) for es in output['controls'].values()),
            control_steps=sum(len(e['commands']) for es in output['controls'].values() for e in es),
            control_scalar_sensor_readings=sum(2*len(e['observed']) for es in output['controls'].values() for e in es),
            planning_residual_calls=sum(p['residual_calls'] for es in output['controls'].values() for e in es for p in e['plans']))
        prefix.with_suffix('.json').write_text(json.dumps(output,default=encode,indent=2,allow_nan=False)+'\n')
    print('saved',str(prefix),time.time()-start,flush=True)


if __name__=='__main__':main()
