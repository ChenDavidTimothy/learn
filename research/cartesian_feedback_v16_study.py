#!/usr/bin/env python3
"""Fresh same-data Cartesian construction study; adjacent frozen study design."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
from scipy.optimize import least_squares
from scipy.special import logsumexp,ndtr
from threadpoolctl import threadpool_limits
import cartesian_reuse_v16_gk21 as base

T=base.T;SD=base.SD;R=base.R;DS=base.DS;L=base.L;WV=base.WV;WU=base.WU;BOUND=base.BOUND
SEQUENCES=L*np.array([[[1,0],[1,1],[0,1],[-1,1],[-1,0],[0,0]],
                     [[0,1],[1,1],[1,0],[0,0],[-1,0],[-1,1]]],dtype=float)
DOBS=np.hstack([np.eye(2),np.zeros((2,6))])


def transition(u):
    H=np.kron(np.eye(2),u[None]);A=np.eye(8);A[:2,2:4]=np.eye(2)
    A[:2,4:]=.5*H;A[2:4,4:]=H
    return A


class Belief:
    def __init__(self,weights,means,covs):
        self.weights=np.array(weights);self.means=np.array(means);self.covs=np.array(covs)

    def predict(self,u):
        A=transition(u);return Belief(self.weights,self.means@A.T,A[None]@self.covs@A.T)

    def posterior_stats(self,y):
        y=np.atleast_2d(y);innovation=y[:,None]-self.means[None,:,:2]
        S=self.covs[:,:2,:2]+R*np.eye(2)
        K=np.linalg.solve(S,self.covs[:,:2,:]).transpose(0,2,1)
        means=self.means[None]+np.einsum('kij,nkj->nki',K,innovation)
        cov=self.covs-K@self.covs[:,:2,:];cov=.5*(cov+cov.transpose(0,2,1))
        logs=np.log(np.maximum(self.weights,1e-300))[None]-.5*(2*np.log(2*np.pi)+np.linalg.slogdet(S)[1][None]+
            np.einsum('nki,nki->nk',innovation,np.linalg.solve(S[None],innovation[...,None])[...,0]))
        weights=np.exp(logs-logsumexp(logs,axis=1)[:,None])
        return weights,means,cov,K,S

    def observe(self,y):
        weights,means,cov,_,_=self.posterior_stats(y)
        return Belief(weights[0],means[0],cov)

    def record(self):return dict(weights=self.weights,means=self.means,covs=self.covs)

    def gaussian(self):
        mean=np.einsum('k,ki->i',self.weights,self.means);delta=self.means-mean
        cov=np.einsum('k,kij->ij',self.weights,self.covs)+np.einsum('k,ki,kj->ij',self.weights,delta,delta)
        return Belief([1.],mean[None],cov[None])


def predictive_mixture_nll(X,Y,B,weights,coef_covs):
    nobs=2*X.shape[1];H=base.matrix_H(X)
    r=Y.reshape(len(Y),1,nobs)-np.einsum('nri,ki->nkr',H,B.reshape(-1,4))
    C=H[:,None]@coef_covs[None]@H[:,None].transpose(0,1,3,2)+R*np.eye(nobs)
    logs=np.log(weights)[None]-.5*(nobs*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+
        np.einsum('nki,nki->nk',r,np.linalg.solve(C,r[...,None])[...,0]))
    return -float(logsumexp(logs,axis=1).mean())


def learn_priors(parent,data):
    X=data['training_exposures'];Y=(data['training_observed'][:,1:]-base.P0)/T**2
    V=data['validation_exposures'];Z=(data['validation_observed'][:,1:]-base.P0)/T**2
    candidates=[]
    for index,row in enumerate(parent['mixture_candidates']):
        B=np.array(row['B']);weights=np.array(row['weights'])
        parts=base.mixture_log_parts(X,Y,B,weights);resp=np.exp(parts-logsumexp(parts,axis=1)[:,None]);covs=[]
        for k in range(len(weights)):
            gram=np.einsum('n,nti,ntj->ij',resp[:,k],X,X)+1e-10*np.eye(2)
            covs.append(np.kron(np.eye(2),R*np.linalg.inv(gram)))
        covs=np.array(covs)
        candidates.append(dict(parent_candidate=index,B=B,weights=weights,coef_covs=covs,
            effective_groups=resp.sum(0),validation_nll=predictive_mixture_nll(V,Z,B,weights,covs)))
    selected=min(range(len(candidates)),key=lambda i:candidates[i]['validation_nll']);chosen=candidates[selected]
    means=np.zeros((len(chosen['weights']),8));means[:,4:]=chosen['B'].reshape(-1,4)
    covs=np.zeros((len(means),8,8));covs[:,4:,4:]=chosen['coef_covs']
    mixture=Belief(chosen['weights'],means,covs)
    g=parent['gaussian_candidates'][parent['selected_gaussian']];m=np.zeros((1,8));m[0,4:]=np.array(g['mean']).ravel()
    C=np.zeros((1,8,8));C[0,4:,4:]=g['cov']
    priors=dict(mixture=mixture,gaussian=Belief([1.],m,C),matched=mixture.gaussian())
    isotropic=C.copy();isotropic[0,4:,4:]=np.trace(C[0,4:,4:])/4*np.eye(4)
    priors['isotropic']=Belief([1.],m.copy(),isotropic)
    return priors,candidates,selected


def future_qp(weights,means,covs,goal,steps,spent=0.):
    """Batched joint-state/B expectation, including all covariance cross terms."""
    # weights(n,k), means(n,k,8), covs(k,8,8); covariance independent of observation.
    weights=np.atleast_2d(weights);means=np.asarray(means);g=goal/T**2
    B=means[:,:,4:].reshape(len(means),means.shape[1],2,2)
    Gk=np.einsum('nkoi,nkoj->nkij',B,B)+covs[:,4:6,4:6]+covs[:,6:8,6:8]
    d=means[:,:,:2]+steps*means[:,:,2:4]-g;v=means[:,:,2:4]
    hd=np.einsum('nkoi,nko->nki',B,d);hv=np.einsum('nkoi,nko->nki',B,v)
    for i in range(2):
        for o in range(2):
            hd[:,:,i]+=covs[:,4+2*o+i,o]+steps*covs[:,4+2*o+i,2+o]
            hv[:,:,i]+=covs[:,4+2*o+i,2+o]
    D=np.hstack([np.eye(2),steps*np.eye(2),np.zeros((2,4))]);W=np.hstack([np.zeros((2,2)),np.eye(2),np.zeros((2,4))])
    constk=np.sum(d*d,axis=2)+np.trace(D[None]@covs@D.T,axis1=1,axis2=2)[None]
    constk+=WV*(np.sum(v*v,axis=2)+np.trace(W[None]@covs@W.T,axis1=1,axis2=2)[None])
    const=np.sum(weights*constk,axis=1)/DS**2+WU*spent
    G=np.einsum('nk,nkij->nij',weights,Gk);hd=np.einsum('nk,nki->ni',weights,hd);hv=np.einsum('nk,nki->ni',weights,hv)
    P=np.hstack([(steps-j-.5)*np.eye(2) for j in range(steps)]);V=np.tile(np.eye(2),(1,steps))
    Q=(P.T[None]@G@P+WV*V.T[None]@G@V)/DS**2+WU*np.eye(2*steps)
    q=(hd@P+WV*hv@V)/DS**2;x,pg=base.box_qp(Q,q)
    loss=np.einsum('ni,nij,nj->n',x,Q,x)+2*np.sum(q*x,axis=1)+const
    return x.reshape(-1,steps,2),loss,pg


def plan(belief,goal,steps,spent=0.):
    x,loss,pg=future_qp(belief.weights[None],belief.means[None],belief.covs,goal,steps,spent)
    return x[0],float(loss[0]),float(pg[0])


def expected_first(belief,goal,u,atol=1e-8,rtol=1e-7):
    predicted=belief.predict(u);S=predicted.covs[:,:2,:2]+R*np.eye(2);chols=np.linalg.cholesky(S)
    # Component conditional covariances/gains do not depend on observation.
    _,_,C,K,_=predicted.posterior_stats(np.zeros(2))
    D=np.hstack([np.eye(2),2*np.eye(2),np.zeros((2,4))]);W=np.hstack([np.zeros((2,2)),np.eye(2),np.zeros((2,4))])
    p_tail=4*ndtr(-7.);m_tail=4*7*np.exp(-.5*7**2)/np.sqrt(2*np.pi)+8*ndtr(-7.)
    value=0.;error=0.;tail=0.;points=0;subdivisions=0
    for j,(center,chol,weight) in enumerate(zip(predicted.means[:,:2],chols,predicted.weights)):
        if weight<1e-250:continue
        def integrand(z):
            nonlocal points
            points+=len(z);ys=center+z@chol.T;weights,means,covs,_,_=predicted.posterior_stats(ys)
            _,loss,_=future_qp(weights,means,covs,goal,2,float(u@u))
            return np.maximum(loss,0)*np.exp(-.5*np.sum(z*z,axis=1))/(2*np.pi)
        for lo in ((-7.,-7.),(-7.,0.),(0.,-7.),(0.,0.)):
            hi=np.array(lo)+7.
            fit=cubature(integrand,lo,hi,rule='gk21',atol=atol/4,rtol=rtol,max_subdivisions=2000)
            assert fit.status=='converged',(fit.status,fit.error,fit.estimate)
            value+=weight*float(fit.estimate);error+=weight*float(fit.error);subdivisions+=fit.subdivisions
        # Bound optimized nonnegative loss by feasible zero remaining commands.
        aa=[];bb=[]
        for k in range(len(predicted.weights)):
            m=predicted.means[k]+K[k]@(center-predicted.means[k,:2]);J=K[k]@chol
            dd=D@m-goal/T**2;vv=W@m
            a=(2*dd@dd+2*WV*(vv@vv)+np.trace(D@C[k]@D.T)+WV*np.trace(W@C[k]@W.T))/DS**2+WU*(u@u)
            b=2*(np.linalg.norm(D@J,2)**2+WV*np.linalg.norm(W@J,2)**2)/DS**2
            aa.append(a);bb.append(b)
        tail+=weight*(max(aa)*p_tail+max(bb)*m_tail)
    return float(value),dict(estimated_error=float(error),analytic_tail_upper_bound=float(tail),
        integrand_points=points,subdivisions=subdivisions,normal_truncation=7.,atol=atol,rtol=rtol,
        rule='gk21',initial_partition='four quadrants')


def action_design(belief,goal):
    task,_,task_pg=plan(belief,goal,3);angles=np.arange(16)*2*np.pi/16;dirs=np.column_stack([np.cos(angles),np.sin(angles)])
    candidates=[np.zeros(2)];labels=[('zero',0.,0)]
    for radius in (.05,.1):
        for i,v in enumerate(dirs):candidates.append(radius*v);labels.append(('uniform',radius,i))
    matched=belief.gaussian();C=matched.covs[0,4:,4:];var=C[:2,:2]+C[2:,2:];e,V=np.linalg.eigh(var);v=V[:,-1]
    candidates.append(task[0]);labels.append(('cautious',float(np.linalg.norm(task[0])),0));values=[];records=[]
    for u in candidates:
        val,record=expected_first(belief,goal,np.asarray(u));values.append(val);records.append(record)
    best=int(np.argmin(values));verified=[]
    # Re-score the three closest candidates at tighter tolerance before selection.
    # This is an empirical numerical check, not a rigorous bound on other candidates.
    for index in np.argsort(values)[:3]:
        value,record=expected_first(belief,goal,np.asarray(candidates[index]),atol=1e-10,rtol=1e-9)
        assert abs(value-values[index])<=max(2e-7,5*(record['estimated_error']+records[index]['estimated_error'])),(index,values[index],value)
        verified.append(dict(index=int(index),original_value=values[index],value=value,integration=record))
        values[index]=value
    best=int(np.argmin(values));random_values=[values[0],float(np.mean(values[1:17])),float(np.mean(values[17:33]))]
    return dict(candidates=np.array(candidates),labels=labels,values=values,integration=records,selected=best,
        random_radius=(0.,.05,.1)[int(np.argmin(random_values))],random_radius_values=random_values,
        cautious_plan=task,cautious_pg=task_pg,coefficient_variation_eigenvalues=e,
        tighter_top_three=verified)


def nominal_witnesses():
    result=[]
    for branch in (0,1):
        for seq,goals in enumerate(SEQUENCES):
            z=np.r_[base.preparation(branch),[0.,0.]]
            for phase,goal in enumerate(goals):
                initial=z.copy();q=z[:2];B=base.jacobian(q)@np.linalg.inv(base.mass(q));calls=0
                k=np.linalg.solve(B,(base.P0+goal-base.endpoint(q))/T**2)
                def residual(flat):
                    nonlocal calls
                    calls+=1;zz=initial.copy()
                    for u in np.vstack([np.zeros(2),flat.reshape(2,2)]):zz,_=base.flow(zz[:2],zz[2:],u,T)
                    return np.r_[(base.endpoint(zz[:2])-base.P0-goal)/L,np.sqrt(WV)*(base.jacobian(zz[:2])@zz[2:])/(L/T)]
                fit=least_squares(residual,np.r_[k,-k],bounds=(-BOUND,BOUND),max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                assert np.linalg.norm(fit.fun)<1e-7,(branch,seq,phase,fit.fun)
                controls=np.vstack([np.zeros(2),fit.x.reshape(2,2)])
                for u in controls:z,_=base.flow(z[:2],z[2:],u,T)
                result.append(dict(branch_evaluator_only=branch,sequence=seq,phase=phase,initial_state=initial,
                    commands=controls,terminal_state=z.copy(),normalized_residual=fit.fun,residual_calls=calls,
                    plant_flow_calls=3*calls+3))
    return result


def run_episode(prior,name,mode,sequence,branch,repeat,rep,cache):
    path=[20261020,16,640,rep,4,sequence*100+repeat];rng=np.random.default_rng(np.random.SeedSequence(path))
    noise=rng.normal(0,SD,(6,2,2));directions=rng.integers(0,16,6);belief=prior
    z=np.r_[base.preparation(branch),[0.,0.]];states=[z.copy()];commands=[];observed=[];phases=[];nfev=0
    for phase,goal in enumerate(SEQUENCES[sequence]):
        before=belief.record()
        if mode=='task':u0=plan(belief,goal,3)[0][0];design=None
        else:
            key=(name,sequence)
            if phase==0 and key in cache:design=cache[key]
            else:
                design=action_design(belief,goal)
                if phase==0:cache[key]=design
            if mode=='dual':u0=design['candidates'][design['selected']]
            else:
                angle=directions[phase]*2*np.pi/16;u0=design['random_radius']*np.array([np.cos(angle),np.sin(angle)])
        z,k=base.flow(z[:2],z[2:],u0,T);nfev+=k;states.append(z.copy());commands.append(u0)
        o1=base.endpoint(z[:2])+noise[phase,0];observed.append(o1)
        belief=belief.predict(u0).observe((o1-base.P0)/T**2);after_first=belief.record()
        last,predcost,kkt=plan(belief,goal,2,float(u0@u0))
        for u in last:
            z,k=base.flow(z[:2],z[2:],u,T);nfev+=k;states.append(z.copy());commands.append(u);belief=belief.predict(u)
        o3=base.endpoint(z[:2])+noise[phase,1];observed.append(o3);before_terminal_observation=belief.record()
        belief=belief.observe((o3-base.P0)/T**2)
        error=base.endpoint(z[:2])-base.P0-goal;v=base.jacobian(z[:2])@z[2:];actions=np.array(commands[-3:])
        cost=float((error@error+WV*T**2*(v@v))/L**2+WU*np.sum(actions**2))
        phases.append(dict(goal=goal,first_action=u0,design=design,belief_before=before,belief_after_first=after_first,
            belief_before_terminal_observation=before_terminal_observation,belief_after_terminal=belief.record(),
            model_expected_cost=predcost,continuation_pg=kkt,cost=cost,position_error=error,cartesian_velocity=v,
            effort=float(np.sum(actions**2)),precision_success=bool(np.all(np.abs(error)<=.0003) and np.all(np.abs(v)<=.003))))
    return dict(prior=name,mode=mode,sequence=sequence,branch_evaluator_only=branch,repeat=repeat,seed_path=path,
        states_evaluator_only=np.array(states),commands=np.array(commands),observed=np.array(observed),phases=phases,plant_nfev=nfev)


def acquisition(rep,prefix):
    """Same grouped acquisition/fits; omit the redundant short-task deployment."""
    raw={};split={};start=time.time()
    for name,which,count in [('training',1,256),('validation',2,64),('testing',3,64)]:
        physical,X,Y=base.acquire(rep,which,count);split[name]=(X,Y)
        raw.update({name+'_'+k:v for k,v in physical.items()})
    datapath=Path(str(prefix)+'_data.npz');resultpath=Path(str(prefix)+'_results.json')
    assert not datapath.exists() and not resultpath.exists(),prefix
    np.savez_compressed(datapath,**raw)
    X,Y=split['training'];V,Z=split['validation'];records,selection=base.fit_mixtures(X,Y,V,Z,rep)
    gaussians,_=base.fit_gaussian(X,Y,V,Z);row=records[selection]
    mixture=base.Prior('mixture',B=row['B'],weights=row['weights'])
    gselection,checks=base.refine_gaussian(X,Y,V,Z,gaussians,mixture)
    result=dict(meta=dict(status='FRESH GROUPED ACQUISITION; no short-task control rerun',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),replicate=rep,
        base_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        T=T,sensor_sd=SD,action_bound=BOUND,goal_scale=L,velocity_weight=WV,action_weight=WU,
        data_sha256=hashlib.sha256(datapath.read_bytes()).hexdigest()),
        mixture_candidates=records,selected_mixture=selection,gaussian_candidates=gaussians,
        selected_gaussian=gselection,gaussian_gradient_checks=checks,
        resources=dict(acquired_groups=384,acquired_intervals=768,acquired_simulated_seconds=76.8,
            acquired_scalar_sensor_readings=2304,short_task_control_intervals=0),wall_seconds=time.time()-start)
    resultpath.write_text(json.dumps(result,default=base.encode,indent=2,allow_nan=False)+'\n')
    return resultpath


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-results',default=None)
    parser.add_argument('--replicate',type=int,default=None)
    parser.add_argument('--output-prefix',required=True);parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--methods',default='mixture_dual,mixture_task,gaussian_dual,gaussian_task,matched_dual,matched_task,mixture_random,gaussian_random,isotropic_dual,isotropic_task')
    args=parser.parse_args();prefix=Path(args.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    assert (args.replicate is None)!=(args.parent_results is None),'Choose fresh replicate OR frozen parent'
    assert not Path(str(prefix)+'_results.json').exists(),'Refuse to overwrite prior results'
    with threadpool_limits(limits=1):
        parentpath=Path(args.parent_results) if args.parent_results else acquisition(args.replicate,Path(str(prefix)+'_acquisition'))
    parent=json.loads(parentpath.read_bytes())
    datapath=parentpath.with_name(parentpath.name.replace('_results.json','_data.npz'))
    assert hashlib.sha256(datapath.read_bytes()).hexdigest()==parent['meta']['data_sha256']
    for key,value in [('T',T),('sensor_sd',SD),('action_bound',BOUND),('goal_scale',L),
                      ('velocity_weight',WV),('action_weight',WU)]:
        assert parent['meta'][key]==value,(key,parent['meta'][key],value)
    with np.load(datapath) as source:data={k:source[k] for k in source.files}
    rep=parent['meta']['replicate'];start=time.time();cache={}
    with threadpool_limits(limits=1):
        priors,candidates,selection=learn_priors(parent,data);witnesses=nominal_witnesses();episodes=[]
        forecasts={};X=data['testing_exposures'];Y=(data['testing_observed'][:,1:]-base.P0)/T**2
        for name,prior in priors.items():
            B=prior.means[:,4:].reshape(-1,2,2);cov=prior.covs[:,4:,4:]
            joint=predictive_mixture_nll(X,Y,B,prior.weights,cov)
            first=predictive_mixture_nll(X[:,:1],Y[:,:1],B,prior.weights,cov)
            forecasts[name]=dict(joint_group_nll=joint,first_observation_nll=first,second_given_first_nll=joint-first)
        result=dict(meta=dict(status='FRESH FIXED-MENU PERSISTENT CARTESIAN CONSTRUCTION; AUDIT PENDING',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_results=parentpath.name,
            parent_sha256=hashlib.sha256(parentpath.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(datapath.read_bytes()).hexdigest(),
            base_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),replicate=rep,repeats=args.repeats,
            acquisition_origin='fresh_in_this_invocation' if args.replicate is not None else 'reused_frozen_parent',
            acquisition_wall_seconds=parent.get('wall_seconds') if args.replicate is not None else 0.,
            design='cartesian_feedback_v16_study_design.md',observation_schedule='After first and third pulse of every goal; terminal read updates next goal only'),
            mixture_predictive_candidates=candidates,selected_mixture_predictive_candidate=selection,
            priors={k:p.record() for k,p in priors.items()},heldout_forecasts=forecasts,
            nominal_feasibility_witnesses=witnesses,episodes=episodes)
        prefix.parent.mkdir(parents=True,exist_ok=True)
        for method in args.methods.split(','):
            name,mode=method.split('_');count=0
            for sequence in range(2):
                for repeat in range(args.repeats):
                    for branch in (0,1):
                        e=run_episode(priors[name],name,mode,sequence,branch,repeat,rep,cache);episodes.append(e);count+=1
                        print(method,sequence,repeat,branch,'cost',np.mean([p['cost'] for p in e['phases']]),
                            'precision',sum(p['precision_success'] for p in e['phases']),'elapsed',time.time()-start,flush=True)
                        result['wall_seconds']=time.time()-start
                        Path(str(prefix)+'_results.json').write_text(json.dumps(result,default=base.encode,indent=2,allow_nan=False)+'\n')
        designs={id(p['design']):p['design'] for e in episodes for p in e['phases'] if p['design'] is not None}
        result['resources']=dict(parent_acquisition_reused=args.replicate is None,
            new_acquisition_intervals=768 if args.replicate is not None else 0,
            new_acquisition_simulated_seconds=76.8 if args.replicate is not None else 0.,
            new_acquisition_scalar_sensor_readings=2304 if args.replicate is not None else 0,
            control_episodes=len(episodes),
            control_intervals=18*len(episodes),control_simulated_seconds=18*T*len(episodes),
            scalar_sensor_readings=24*len(episodes),unique_paired_noise_paths=2*args.repeats,
            unique_scalar_noise_draws=2*args.repeats*24,nonlinear_witness_plant_flows=sum(w['plant_flow_calls'] for w in witnesses),
            candidate_records_attached_to_episodes=sum(len(p['design']['candidates']) for e in episodes for p in e['phases'] if p['design'] is not None),
            computed_designs=len(designs),evaluated_action_candidates=sum(len(d['candidates']) for d in designs.values()),
            tighter_candidate_reevaluations=sum(len(d['tighter_top_three']) for d in designs.values()),
            integration_points=sum(r['integrand_points'] for d in designs.values() for r in d['integration'])+
                sum(r['integration']['integrand_points'] for d in designs.values() for r in d['tighter_top_three']))
        result['wall_seconds']=time.time()-start
        result['feedback_wall_seconds']=result['wall_seconds']
        result['total_invocation_wall_seconds']=result['wall_seconds']+result['meta']['acquisition_wall_seconds']
        Path(str(prefix)+'_results.json').write_text(json.dumps(result,default=base.encode,indent=2,allow_nan=False)+'\n')
    print('complete',prefix,time.time()-start,flush=True)


if __name__=='__main__':main()
