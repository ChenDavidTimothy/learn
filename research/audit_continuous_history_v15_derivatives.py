#!/usr/bin/env python3
"""Separate Torch feature/rollout autodiff vs canonical analytic Jacobians.

Canonical code is imported solely as the implementation under test. The independent
rollout below uses only model numbers, without calling canonical feature routines.
"""
import ast,json
from pathlib import Path
import numpy as np
import torch
from continuous_history_v15_pilot import Model,residual_plan

torch.set_default_dtype(torch.float64)


def independent(flat,coef,h,kind,initial,prior,goal):
    history=[torch.tensor(z) for z in initial];past=[torch.tensor(z) for z in prior]
    W=torch.tensor(coef);res=[];dt=.1
    for u in flat.reshape(-1,2):
        q=history[-1]
        p=[torch.tensor(1.)]
        if kind!='linear':
            p += [f(k*q[j]) for j in range(2) for k in (1,2) for f in (torch.sin,torch.cos)]
            p += [f(q[0])*g(q[1]) for f in (torch.sin,torch.cos) for g in (torch.sin,torch.cos)]
        b=[torch.tensor(1.),u[0],u[1]]
        if h:
            if kind=='poly':
                times=torch.arange(-h,1)*dt
                design=torch.stack([torch.ones(h+1),times,times**2/2],dim=1)
                summaries=torch.linalg.pinv(design)[1:]@torch.stack(history[-h-1:])
                b.extend(summaries.flatten());motion=summaries[0]
            else:
                b.extend([(q[j]-history[-k-1][j])/(k*dt) for k in range(1,h+1) for j in range(2)])
                motion=(q-history[-2])/dt
            b.extend([past[-k][j] for k in range(1,h+1) for j in range(2)])
            if kind!='linear':b.extend([motion[0]**2,motion[0]*motion[1],motion[1]**2])
        feature=torch.outer(torch.stack(p),torch.stack(b)).flatten()
        pred=q+dt*(feature@W)
        res.extend([pred-torch.tensor(goal),np.sqrt(.3)*(pred-q)/dt,np.sqrt(.003)*u])
        history.append(pred);past.append(u)
    return torch.cat(res)


def main():
    root=Path(__file__).resolve().parent
    def nodes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    frozen=nodes(root/'continuous_history_v15_pilot.py');production=nodes(root/'continuous_history_v15.py')
    shared=('context','base','features','Model','residual_plan','choose')
    assert all(frozen[k]==production[k] for k in shared)
    rng=np.random.default_rng(865150013)
    times=np.arange(-19,1)*.1
    initial=np.array([.4,1.2])+times[:,None]*np.array([.03,-.02])+.001*times[:,None]**2
    prior=rng.uniform(-.2,.2,(19,2));goal=np.array([.5,1.1]);reports=[]
    for kind,hs in [('raw',[0,1,2,4]),('linear',[1,2,4]),('poly',[2,4,8])]:
        for h in hs:
            nb=3+(0 if h==0 else ((4 if kind=='poly' else 2*h)+2*h+(0 if kind=='linear' else 3)))
            nc=1 if kind=='linear' else 13
            coef=rng.normal(0,1e-5,(nc*nb,2))
            if h:
                coef[3,0]+=1.;coef[4,1]+=1.
            coef[1,0]+=.05;coef[2,1]+=.05
            model=Model(h,kind,coef,1e-5)
            for horizon in (4,15):
                flat=rng.uniform(-.2,.2,2*horizon)
                canonical,J=residual_plan(flat,model,initial,prior,goal)
                x=torch.tensor(flat,requires_grad=True)
                f=lambda a:independent(a,coef,h,kind,initial,prior,goal)
                actual=f(x).detach().numpy();jac=torch.autograd.functional.jacobian(f,x).detach().numpy()
                err=float(np.max(np.abs(jac-J)));reserr=float(np.max(np.abs(actual-canonical)))
                assert err<1e-9 and reserr<1e-11,(kind,h,horizon,err,reserr)
                reports.append(dict(kind=kind,h=h,horizon=horizon,residual_max_error=reserr,jacobian_max_error=err))
    print(json.dumps(dict(scope='Independent Torch forward feature/rollout differentiation; synthetic stable parameter fixtures, every family and full15step planner.',
        no_parameter_fit_or_control_performance_check=True,checks=reports,
        maximum_jacobian_error=max(r['jacobian_max_error'] for r in reports),
        canonical_under_test='continuous_history_v15_pilot.py',
        fresh_production_identical_feature_model_planner_AST=list(shared)),indent=2))


if __name__=='__main__':main()
