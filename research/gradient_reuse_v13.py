#!/usr/bin/env python3
"""One bounded capacity/search diagnostic; supplied family, learned models only."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from shared_experience_v12 import LOW,HIGH,SD,N,TRAIN_GOALS,TEST_GOALS,targets,decode,direct_episode,model_episode,score,MOTOR,stat


def bank(seed):
    return np.random.default_rng(seed).normal(0,SD,(16,N+1))


def roll(theta,p,errors):
    """B independently reset virtual episodes; exact pathwise Jacobian a.e."""
    rho,lam,c,beta=theta;kp,kd,alpha,bv,bu1,bu2=decode(p);B=len(errors)
    goal=np.array([targets(TRAIN_GOALS[k%2]) for k in range(B)])
    scale=np.sqrt(B*N*(1+np.mean(goal**2,axis=1)))
    x=v=m=sx=sv=prev=older=np.zeros(B)
    dx=dv=dm=dsx=dsv=dprev=dolder=np.zeros((B,6))
    eye=np.eye(6);dkp=kp*eye[0];dkd=kd*eye[1];dalpha=alpha*(1-alpha)*eye[2]
    states=np.zeros((B,N+1,3));obs=np.zeros((B,N+1));commands=np.zeros((B,N));res=[];jac=[]
    obs[:,0]=errors[:,0]
    for t in range(N):
        oldx=sx;oldv=sv;old_dx=dsx;old_dv=dsv
        sx=oldx+alpha*(x+errors[:,t]-oldx)
        dsx=(1-alpha)*old_dx+alpha*dx+(x+errors[:,t]-oldx)[:,None]*dalpha
        sv=oldv+alpha*(sx-oldx-oldv)
        dsv=(1-alpha)*old_dv+alpha*(dsx-old_dx)+(sx-oldx-oldv)[:,None]*dalpha
        raw=kp*(goal[:,t]-sx)-kd*sv+bv*oldv-bu1*prev-bu2*older
        draw=(goal[:,t]-sx)[:,None]*dkp-kp*dsx-sv[:,None]*dkd-kd*dsv
        draw+=oldv[:,None]*eye[3]+bv*old_dv-prev[:,None]*eye[4]-bu1*dprev-older[:,None]*eye[5]-bu2*dolder
        a=np.clip(raw,-1.,1.);da=draw*(np.abs(raw)<1)[:,None]
        older=prev;prev=a;dolder=dprev;dprev=da
        h=np.tanh(v);vn=rho*v-c*h+m;dvn=(rho-c*(1-h*h))[:,None]*dv+dm
        m=lam*m+beta*a;dm=lam*dm+beta*da;v=vn;dv=dvn;x=x+v;dx=dx+dv
        states[:,t+1,0]=x;states[:,t+1,1]=v;states[:,t+1,2]=m;obs[:,t+1]=x+errors[:,t+1];commands[:,t]=a
        res.extend(((x-goal[:,t])/scale,math.sqrt(.3)*v/scale,math.sqrt(.03)*a/scale))
        jac.extend((dx/scale[:,None],math.sqrt(.3)*dv/scale[:,None],math.sqrt(.03)*da/scale[:,None]))
    return np.array(res).reshape(-1),np.array(jac).reshape(-1,6),states,obs,commands


def optimize(theta,start,errors):
    cache={};calls=0
    def calc(p):
        nonlocal calls
        if 'p' not in cache or not np.array_equal(p,cache['p']):
            r,J,*_=roll(theta,p,errors);calls+=1;cache.update(p=p.copy(),r=r,J=J)
        return cache
    f=least_squares(lambda p:calc(p)['r'],start,jac=lambda p:calc(p)['J'],bounds=(LOW,HIGH),
        max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-8)
    r=calc(f.x)['r'];J=calc(f.x)['J'];g=2*J.T@r;project=f.x-np.clip(f.x-g,LOW,HIGH)
    return dict(coordinates=f.x.tolist(),start=np.asarray(start).tolist(),success=bool(f.success),status=int(f.status),nfev=int(f.nfev),
        njev=int(f.njev),unique_rollout_calls=calls,virtual_steps=calls*16*N,cost=float(r@r),
        projected_gradient_max=float(np.max(np.abs(project))))


def checks():
    theta=(.65,.7,.2,1.2);e=bank(75071013);p=np.array([math.log(.08),math.log(.6),math.log(.6/.4),.1,.2,-.05])
    r,J,_,_,actions=roll(theta,p,e);rows=[]
    for epsilon in (1e-6,1e-7,1e-8):
        errors=[];changes=[]
        for i in range(6):
            d=np.eye(6)[i]*epsilon;rp,_,_,_,up=roll(theta,p+d,e);rm,_,_,_,um=roll(theta,p-d,e)
            errors.append(float(np.max(np.abs(J[:,i]-(rp-rm)/(2*epsilon)))))
            changes.append(int(np.count_nonzero((np.abs(up)<1)!=(np.abs(um)<1))))
        rows.append(dict(epsilon=epsilon,coordinate_errors=errors,clip_active_set_changes=changes))
    # A finite difference crossing a clip kink is not a derivative check there.
    # Two fixed smaller steps must preserve the active set and agree everywhere.
    for row in rows[1:]:
        assert sum(row['clip_active_set_changes'])==0
        assert max(row['coordinate_errors'])<1e-5
    return dict(pathwise_jacobian_max_error=max(max(r['coordinate_errors']) for r in rows[1:]),
                finite_difference_steps=rows,initial_coarser_check_failed_at_clip_crossing=True)



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-prefix',default='research/gradient_reuse_v13');args=parser.parse_args()
    shared_path=Path('research/shared_experience_v12_results.json');reuse_path=Path('research/model_reuse_v12_results.json')
    shared=json.loads(shared_path.read_text());reuse=json.loads(reuse_path.read_text());out=[];arrays={};timing=[]
    with threadpool_limits(limits=1):
        verified=checks()
        for row,virtual in zip(shared['runs'],reuse['runs']):
            seed=row['seed'];model=row['checkpoints'][-1]['models']['nonlinear'];theta=row['true_theta_evaluator_only']
            assert seed==virtual['physical_seed']
            physical=np.array(row['checkpoints'][-1]['selected_coordinates']);virtualp=np.array(virtual['selected_coordinates'])
            generic=np.array([math.log(.08),math.log(.6),math.log(.6/.4),0.,0.,0.])
            errors=bank(seed+20000000000);validation=bank(seed+30000000000);t0=time.perf_counter()
            fits=[optimize(model,p,errors) for p in (physical,virtualp,generic)]
            candidates=[np.array(f['coordinates']) for f in fits]+[physical,virtualp]
            values=[]
            for p in candidates:
                r,*_=roll(model,p,validation);values.append(float(r@r))
            chosen=int(np.argmin(values));p=candidates[chosen]
            timing.append(dict(seed=seed,model_fit_and_validation_seconds=time.perf_counter()-t0))
            tests=[]
            for j,g in enumerate(TEST_GOALS):
                stream=seed+40000000000+j;controllers={}
                for label,q in (('physical_CEM',physical),('virtual_CEM',virtualp),('gradient_selected',p)):
                    s,o,u=direct_episode(theta,q,g,stream,MOTOR)
                    controllers[label]=dict(**score(s,o,u,targets(g)),states=s.tolist(),observations=o.tolist(),commands=u.tolist(),seed=stream)
                controllers['nonlinear_MPC']=model_episode(theta,model,g,stream)
                tests.append(dict(goal=g,controllers=controllers))
            key=f"p{row['plant_index']}_s{row['replicate']}";arrays[key+'_fit_sensor_errors']=errors;arrays[key+'_validation_sensor_errors']=validation
            # Keep all selected virtual trajectories for independent reconstruction.
            for label,noise in (('fit',errors),('validation',validation)):
                _,_,s,o,u=roll(model,p,noise);arrays[key+'_'+label+'_states']=s;arrays[key+'_'+label+'_observations']=o;arrays[key+'_'+label+'_commands']=u
            out.append(dict(plant=row['plant_index'],replicate=row['replicate'],seed=seed,learned_model=model,fits=fits,
                virtual_validation_scores=values,selected_index=chosen,selected_coordinates=p.tolist(),tests=tests))
            print(f"completed gradient plant{row['plant_index']} seed{row['replicate']}",file=sys.stderr,flush=True)
    summary=[]
    for pi in range(4):
        rows=[r for r in out if r['plant']==pi];controllers={}
        for label in ('physical_CEM','virtual_CEM','gradient_selected','nonlinear_MPC'):
            a=[np.mean([t['controllers'][label]['actual_cost'] for t in r['tests']]) for r in rows]
            controllers[label]=dict(cost=stat(a),stationary_successes=sum(t['controllers'][label]['stationary_success'] for r in rows for t in r['tests']),trials=16)
        differences={}
        for label in ('physical_CEM','virtual_CEM','nonlinear_MPC'):
            a=[np.mean([t['controllers'][label]['actual_cost']-t['controllers']['gradient_selected']['actual_cost'] for t in r['tests']]) for r in rows]
            differences[label+'_minus_gradient']=stat(a)
        summary.append(dict(plant=pi,controllers=controllers,paired_differences=differences))
    prefix=args.output_prefix;np.savez_compressed(prefix+'_data.npz',**arrays)
    meta=dict(status='BOUNDED EXPLORATORY GRADIENT DIAGNOSTIC; separate audit required',design='gradient_reuse_v13_design.md',
        shared_result_sha256=hashlib.sha256(shared_path.read_bytes()).hexdigest(),reuse_result_sha256=hashlib.sha256(reuse_path.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(Path(prefix+'_data.npz').read_bytes()).hexdigest(),additional_plant_training_steps=0,
        plant_test_steps=32*2*4*N,plant_test_resets=32*2*4,training_virtual_steps=sum(f['virtual_steps'] for r in out for f in r['fits']),
        validation_virtual_steps=32*5*16*N,retained_trajectory_replay_virtual_steps=32*2*16*N,derivative_verification_virtual_steps=37*16*N,
        actual_clip_derivative=True,no_global_optimality_guarantee=True,selection_uses_virtual_validation_only=True)
    Path(prefix+'_results.json').write_text(json.dumps(dict(meta=meta,checks=verified,runs=out,summary=summary),indent=2,allow_nan=False)+'\n')
    Path(prefix+'_timing.json').write_text(json.dumps(timing,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
