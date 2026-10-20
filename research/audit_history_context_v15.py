#!/usr/bin/env python3
"""Independent causal-history pilot audit; no canonical module imported."""
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def mass(q):
    c=np.cos(q[...,1]);return np.array([[3+2*c,1+c],[1+c,np.ones_like(c)]]).transpose(2,0,1)


def flow(q,v,u,duration,steps):
    q,v,u=np.atleast_2d(q),np.atleast_2d(v),np.atleast_2d(u)
    p=np.einsum('nij,nj->ni',mass(q),v);y=np.c_[q,p]
    def rhs(z):
        c=np.cos(z[:,1]);s=np.sin(z[:,1]);det=2-c*c
        a=(z[:,2]-(1+c)*z[:,3])/det
        b=(-(1+c)*z[:,2]+(3+2*c)*z[:,3])/det
        return np.c_[a,b,u[:,0],u[:,1]-s*(a*a+a*b)]
    dt=duration/steps
    for _ in range(steps):
        a=rhs(y);b=rhs(y+dt*a/2);c=rhs(y+dt*b/2);d=rhs(y+dt*c)
        y+=dt*(a+2*b+2*c+d)/6
    c=np.cos(y[:,1]);det=2-c*c
    v=np.c_[(y[:,2]-(1+c)*y[:,3])/det,(-(1+c)*y[:,2]+(3+2*c)*y[:,3])/det]
    return np.c_[y[:,:2],v]


def main():
    data=json.loads((ROOT/'history_context_v15_pilot_results.json').read_text())
    parent=json.loads((ROOT/'coupled_learning_v14_results.json').read_text())
    models=[r['models']['corrected_inverse'] for r in parent['runs'] if r['condition']==0]
    old=(ROOT/'history_context_v15_seed_collision_pilot.py').read_text()
    current=(ROOT/'history_context_v15_pilot.py').read_text()
    corrected=old.replace('20261020+2300000','20261020+50000000000')
    corrected=corrected.replace('leading_pose_only_risk_floor','ballistic_surrogate_risk_reference_not_nonlinear_certificate')
    assert corrected==current
    past_times=np.array([-.2,-.15,-.1,-.05]);base=50020261020
    errors=dict(past_physics=0.,past_RNG=0.,slope=0.,learned_box_optimality=0.,physics64vs128=0.,endpoint_state=0.)
    inputs=[];recorded=[];seeds=set()
    centers=np.array([(a,b) for a in np.linspace(-math.pi,math.pi,8,endpoint=False) for b in np.linspace(-math.pi,math.pi,8,endpoint=False)])
    for world in data['worlds']:
        q=np.array(world['current_pose']);v=np.array(world['initial_velocity_evaluator_only'])
        states=np.array([flow(q,v,[0.,0.],t,128)[0] for t in past_times])
        errors['past_physics']=max(errors['past_physics'],float(np.max(np.abs(states-world['exact_past_states_evaluator_only']))))
    for row in data['runs']:
        mi,wi=row['model'],row['world'];ni=row['sensor_seed']-(base+1000000*mi+10000*wi)
        assert 0<=ni<16 and row['sensor_seed'] not in seeds;seeds.add(row['sensor_seed'])
        world=data['worlds'][wi];q=np.array(world['current_pose']);v=np.array(world['initial_velocity_evaluator_only'])
        past=np.array(world['exact_past_states_evaluator_only'])[:,:2]
        observed=past+np.random.default_rng(row['sensor_seed']).normal(0,1e-4,(4,2))
        errors['past_RNG']=max(errors['past_RNG'],float(np.max(np.abs(observed-row['observed_past_positions']))))
        slope=past_times@(observed-q)/(past_times@past_times)
        errors['slope']=max(errors['slope'],float(np.max(np.abs(slope-row['estimated_slope']))))
        model=models[mi];phi=np.r_[1.,np.exp(-np.sum(2-2*np.cos(q-centers),axis=1)/(2*model['width']**2))]
        K=np.einsum('f,fio->oi',phi,np.array(model['coefficients']).reshape(65,2,2));B=np.linalg.pinv(K,rcond=1e-8)
        target=-2*slope/.2;u=np.array(row['outcomes']['learned_relation_history']['command'])
        candidates=[np.linalg.lstsq(B,target,rcond=None)[0]]
        for i in (0,1):
            j=1-i
            for edge in (-1.,1.):
                z=np.zeros(2);z[i]=edge;z[j]=np.clip(B[:,j]@(target-B[:,i]*edge)/(B[:,j]@B[:,j]),-1,1);candidates.append(z)
        candidates=[z for z in candidates if np.max(np.abs(z))<=1+1e-12]
        best=min(np.sum((B@z-target)**2) for z in candidates)
        errors['learned_box_optimality']=max(errors['learned_box_optimality'],float(abs(np.sum((B@u-target)**2)-best)))
        for outcome in row['outcomes'].values():
            inputs.append((q,v,np.array(outcome['command'])));recorded.append(outcome['final_state_evaluator_only'])
    q,v,u=(np.array([z[i] for z in inputs]) for i in range(3))
    coarse=flow(q,v,u,.2,64);fine=flow(q,v,u,.2,128)
    errors['physics64vs128']=float(np.max(np.abs(coarse-fine)))
    errors['endpoint_state']=float(np.max(np.abs(fine-recorded)))
    # Production role maxima throughv14, reconstructed from bounded source ranges.
    # v12 4plants/8replicates, v13 fresh tests add40b; all earlier roles are smaller.
    prior_max=(20261020+1900000)+3*1000000000+7*1000000+40000000000+1
    assert min(seeds)>prior_max
    assert max(errors.values())<1e-8
    print(json.dumps(dict(scope='Independent Hamiltonian RK4, causal feature/RNG/box check; not global nonlinear pose-only optimality',
        errors=errors,history_seeds=len(seeds),minimum_history_seed=min(seeds),maximum_history_seed=max(seeds),
        preceding_production_role_maximum=prior_max,
        freshness='Numerically disjoint declared role namespaces throughv14; distinct seeds are not a mathematical proof of independent pseudorandom streams.',
        frozen_source_changes='Seed correction and metadata-only renaming of ballistic reference; no action/physics change',pilot_only=True),indent=2))


if __name__=='__main__':main()
