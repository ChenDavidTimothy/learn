#!/usr/bin/env python3
"""Independent corrected Cartesian batch audit; no canonical implementation imports.

The frozen-pilot checker is an independently written math/physics helper. This
checker uses independent group likelihood/derivatives and a different cubature
rule with tighter tolerances for selected and competing first actions.
"""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from scipy.integrate import cubature
from scipy.optimize import minimize
from scipy.special import ndtr,logsumexp
from scipy.stats import qmc,norm
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as independent

ROOT=Path(__file__).resolve().parent
T=.1;R=.0004;SD=.0002;BOUND=.5;L=.001;WV=.25;WU=.01;SCALE=L/T**2
P0=independent.P0;GOALS=independent.GOALS;ERROR={}
INTEGRAL_CACHE={}

def check(name,a,b,tol=1e-8):
    error=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    ERROR[name]=max(ERROR.get(name,0.),error);assert error<tol,(name,error,tol)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def likelihood_and_gradient(theta,H,y):
    m=theta[:4];F=theta[4:].reshape(4,4);S=F@F.T
    V=H@S@H.transpose(0,2,1)+R*np.eye(4);d=y-H@m;z=np.linalg.solve(V,d[...,None])[...,0]
    val=.5*np.mean(4*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+np.sum(d*z,axis=-1))
    inv=np.linalg.solve(V,np.broadcast_to(np.eye(4),V.shape))
    gm=-np.mean(np.einsum('nri,nr->ni',H,z),axis=0)
    GC=np.mean(H.transpose(0,2,1)@(inv-z[:,:,None]*z[:,None,:])@H,axis=0)/2
    return float(val),np.r_[gm,((GC+GC.T)@F).ravel()]

def fast_boxes(Q,b):
    answer=np.linalg.solve(Q,-b[...,None])[...,0]
    bad=np.flatnonzero((abs(answer)>BOUND+1e-12).any(-1))
    if len(bad):answer[bad]=independent.solve_boxes(Q[bad],b[bad])
    grad=2*(np.einsum('nij,nj->ni',Q,answer)+b)
    assert np.max(abs(answer-np.clip(answer-grad,-BOUND,BOUND)))<1e-7
    return answer

def loss(prior,u,g,y):
    M,G,_=independent.update(prior,y,u);P=independent.SP;V=independent.SV;d=g/T**2;s=2.5*u;t=u
    Q=(P.T[None]@G@P+WV*V.T[None]@G@V)/SCALE**2+WU*np.eye(4)
    b=(np.einsum('ia,nij,j->na',P,G,s)-np.einsum('ia,noi,o->na',P,M,d)+WV*np.einsum('ia,nij,j->na',V,G,t))/SCALE**2
    x=fast_boxes(Q,b);a=s+x@P.T;c=t+x@V.T
    return (np.einsum('ni,nij,nj->n',a,G,a)-2*np.einsum('o,noi,ni->n',d,M,a)+d@d+WV*np.einsum('ni,nij,nj->n',c,G,c))/SCALE**2+WU*(u@u+np.sum(x*x,-1))

def tail_bound(prior,u,g):
    mu,C=independent.prior_moments(prior);d=g/T**2;prob=4*ndtr(-7.)
    if prior['kind']=='mixture':
        Bs=np.array(prior['B']);worst=max((np.linalg.norm(B@(2.5*u)-d)**2+WV*np.linalg.norm(B@u)**2)/SCALE**2+WU*u@u for B in Bs)
        return float(worst*prob)
    H=np.kron(np.eye(2),(.5*u)[None,:]);Obs=H@C@H.T+R*np.eye(2);K=C@H.T@np.linalg.inv(Obs);post=C-K@H@C
    D=K@np.linalg.cholesky(Obs);Ap=np.stack([D[:,i].reshape(2,2)@(2.5*u) for i in range(2)],1);Av=np.stack([D[:,i].reshape(2,2)@u for i in range(2)],1)
    G=post[:2,:2]+post[2:,2:]
    a=(2*np.linalg.norm(mu@(2.5*u)-d)**2+2*WV*np.linalg.norm(mu@u)**2+(2.5*u)@G@(2.5*u)+WV*u@G@u)/SCALE**2+WU*u@u
    b=2*(np.linalg.norm(Ap,2)**2+WV*np.linalg.norm(Av,2)**2)/SCALE**2
    second=4*7*np.exp(-49/2)/np.sqrt(2*np.pi)+8*ndtr(-7.)
    return float(a*prob+b*second)

def integrate(prior,u,g):
    sign=1 if g[np.flatnonzero(g)[0]]>0 else -1
    u=np.asarray(u)*sign;g=np.asarray(g)*sign
    cache_key=(json.dumps(prior,sort_keys=True),tuple(np.round(u,14)),tuple(g))
    if cache_key in INTEGRAL_CACHE:return INTEGRAL_CACHE[cache_key]
    mu,C=independent.prior_moments(prior);H=np.kron(np.eye(2),(.5*u)[None,:])
    if prior['kind']=='mixture':
        centers=np.array(prior['B']).reshape(-1,4)@H.T;chol=np.repeat((np.sqrt(R)*np.eye(2))[None],len(centers),axis=0);weights=np.array(prior['weights'])
    else:
        centers=(H@mu.ravel())[None];chol=np.linalg.cholesky(H@C@H.T+R*np.eye(2))[None];weights=np.ones(1)
    total=0.;error=0.;statuses=[];points=0
    for m,c,w in zip(centers,chol,weights):
        def integrand(z):
            nonlocal points
            points+=len(z)
            return loss(prior,u,g,m+z@c.T)*np.exp(-np.sum(z*z,axis=1)/2)/(2*np.pi)
        ans=cubature(integrand,[-7.,-7.],[7.,7.],rule='gk21',atol=5e-9,rtol=1e-7,max_subdivisions=2000)
        total+=w*float(ans.estimate);error+=w*float(ans.error);statuses.append(ans.status)
    assert all(x=='converged' for x in statuses),statuses
    answer=(float(total),float(error),points);INTEGRAL_CACHE[cache_key]=answer
    return answer

def qmc_verify(prior,u,g):
    mu,C=independent.prior_moments(prior);H=np.kron(np.eye(2),(.5*u)[None,:]);values=[]
    if prior['kind']=='mixture':
        centers=np.array(prior['B']).reshape(-1,4)@H.T;chols=np.repeat((np.sqrt(R)*np.eye(2))[None],len(centers),axis=0);weights=np.array(prior['weights'])
    else:
        centers=(H@mu.ravel())[None];chols=np.linalg.cholesky(H@C@H.T+R*np.eye(2))[None];weights=np.ones(1)
    components=[]
    for seed in range(4):
        z=norm.ppf(qmc.Sobol(2,scramble=True,seed=20261020+seed).random_base2(14));conditional=[]
        for center,chol in zip(centers,chols):conditional.append(float(loss(prior,u,g,center+z@chol.T).mean()))
        components.append(conditional);values.append(float(weights@conditional))
    return {'scrambles':4,'points_per_component_per_scramble':16384,'values':values,'component_values':components,'mean':float(np.mean(values)),'scramble_mean_standard_error':float(np.std(values,ddof=1)/2)}

def main():
    start=time.time();source=ROOT/'cartesian_reuse_v16_adaptive.py';rp=ROOT/'cartesian_reuse_v16_adaptive_results.json';dp=ROOT/'cartesian_reuse_v16_adaptive_data.npz';design=ROOT/'cartesian_reuse_v16_adaptive_design.md'
    r=json.loads(rp.read_text());prior_result=json.loads((ROOT/'cartesian_reuse_v16_pilot_results.json').read_text());prior_audit=json.loads((ROOT/'audit_cartesian_reuse_v16_pilot_results.json').read_text())
    check('source_hash',sha(source)==r['meta']['source_sha256'],True,1e-15);check('data_hash',sha(dp)==r['meta']['data_sha256'],True,1e-15)
    for n,h in prior_audit['artifact_sha256'].items():assert sha(ROOT/n)==h
    assert sha(ROOT/'audit_cartesian_reuse_v16_pilot.py')==prior_audit['audit_source_sha256']
    old=np.load(ROOT/'cartesian_reuse_v16_pilot_data.npz');raw=np.load(dp);assert raw.files==old.files
    for name in raw.files:check('raw_parent_data_identity',raw[name],old[name],1e-15)
    assert r['input_data_reuse']['sha256']==sha(ROOT/'cartesian_reuse_v16_pilot_data.npz')
    assert r['resources']['new_acquisition_intervals']==0
    assert r['mixture_candidates']==prior_result['mixture_candidates'];assert r['selected_mixture']==prior_result['selected_mixture']
    for i in range(3):assert r['gaussian_candidates'][i]==prior_result['gaussian_candidates'][i]
    data={n:(raw[n+'_exposures'],(raw[n+'_observed'][:,1:]-P0)/T**2) for n in ('training','validation','testing')}
    X,Y=data['training'];H=independent.group_H(X);y=Y.reshape(-1,4);VX,VY=data['validation'];VH=independent.group_H(VX);vy=VY.reshape(-1,4)
    derivative_checks=[];ml_records=[]
    for ri,row in enumerate(r['gaussian_candidates'][3:],3):
        m=np.array(row['mean']);C=np.array(row['cov']);e,V=np.linalg.eigh(C);F=V*np.sqrt(np.maximum(e,0));theta=np.r_[m.ravel(),F.ravel()]
        value,gradient=likelihood_and_gradient(theta,H,y);vnll=-independent.gaussian_logs(VH,vy,m,C).mean()
        check('gaussian_ML_train_objective',value,row['training_nll'],1e-8);check('gaussian_ML_validation',vnll,row['validation_nll'],1e-8)
        assert e.min()>-1e-10
        fd=[]
        for j in range(20):
            step=2e-7;delta=np.zeros(20);delta[j]=step
            fd.append((likelihood_and_gradient(theta+delta,H,y)[0]-likelihood_and_gradient(theta-delta,H,y)[0])/(2*step))
        error=float(np.max(abs(np.array(fd)-gradient)));assert error<5e-5,error
        derivative_checks.append({'record':ri,'all20_coordinate_finite_difference_error':error,'step':2e-7,'mean_gradient_max':float(abs(gradient[:4]).max()),'factor_gradient_Frobenius':float(np.linalg.norm(gradient[4:]))})
        ml_records.append({'record':ri,'start':row['start'],'training_nll':value,'validation_nll':float(vnll),'reported_success':row['success'],'reported_message':row['message'],'covariance_eigenvalues':e.tolist()})
    assert int(np.argmin([x['validation_nll'] for x in r['gaussian_candidates']]))==r['selected_gaussian']
    selected=r['gaussian_candidates'][r['selected_gaussian']];check('selected_Gaussian_mean',selected['mean'],r['priors']['gaussian']['mean'],1e-15);check('selected_Gaussian_covariance',selected['cov'],r['priors']['gaussian']['cov'],1e-15)
    forecasts={};A,C=data['testing'];HA=independent.group_H(A);yy=C.reshape(-1,4)
    for name,p in r['priors'].items():
        mu,cov=independent.prior_moments(p)
        if p['kind']=='mixture':joint=logsumexp(independent.mix_logs(HA,yy,np.array(p['B']),np.array(p['weights'])),axis=-1);first=logsumexp(independent.mix_logs(HA[:,:2],yy[:,:2],np.array(p['B']),np.array(p['weights'])),axis=-1)
        else:joint=independent.gaussian_logs(HA,yy,mu,cov);first=independent.gaussian_logs(HA[:,:2],yy[:,:2],mu,cov)
        f={'joint_group_nll':float(-joint.mean()),'first_observation_nll':float(-first.mean()),'second_given_first_nll':float(-(joint-first).mean()),'mean_response_mse':float(np.mean((C-A@mu.T)**2))};forecasts[name]=f
        for key,val in f.items():check('heldout_forecasts_'+key,val,r['forecasts'][name][key],1e-8)
    summaries={};states=[];commands=[];pg_max=0.;noise_paths=set()
    for key,rows in r['controls'].items():
        name,mode=key.split('_');p=r['priors'][name]
        for row in rows:
            gi=row['goal_index'];g=GOALS[gi];u=np.array(row['commands']);z=np.array(row['states_evaluator_only']);branch=row['branch_evaluator_only'];path=[20261020,16,630,r['meta']['replicate'],4,gi*100+row['repeat']]
            rng=np.random.default_rng(np.random.SeedSequence(path));noise=rng.normal(0,SD,2);di=int(rng.integers(16));noise_paths.add(tuple(path));check('control_seed_path',path,row['seed_path'],1e-15)
            check('control_preparation',z[0],np.r_[independent.prepare(branch),[0.,0.]],1e-15);check('control_observation',independent.endpoint(z[1,:2])+noise,row['first_observation'],1e-15)
            if mode=='dual':d=r['first_designs'][name][gi];u0=d['candidates'][d['selected']]
            elif mode=='random':rad=r['first_designs'][name][gi]['random_radius'];u0=rad*np.array([np.cos(di*2*np.pi/16),np.sin(di*2*np.pi/16)])
            else:u0=independent.cautious(p,g)[0]
            check('control_first_command',u[0],u0,1e-8);M,G,post=independent.update(p,(np.array(row['first_observation'])-P0)/T**2,u[0]);x,cost,Q,b,pg=independent.continuation(M,G,u[0],g,True)
            check('posterior_mean',M[0],row['posterior_mean'],1e-8);check('posterior_second',G[0],row['posterior_BtB'],1e-8);check('posterior_distribution',post[0],row['posterior'],1e-8)
            check('control_continuation',x[0].reshape(2,2),u[1:],1e-8);check('model_cost',cost[0],row['model_expected_cost'],1e-7);pg_max=max(pg_max,float(abs(pg).max()))
            endpoint=independent.endpoint(z[-1,:2]);v=independent.jac(z[-1,:2])@z[-1,2:];err=endpoint-P0-g;pe=err@err;ve=v@v;ae=WU*np.sum(u*u);score=pe/L**2+WV*ve/(L/T)**2+ae
            check('cost',score,row['cost'],1e-8);check('position_term',pe,row['position_error_squared'],1e-15);check('velocity_term',ve,row['velocity_squared'],1e-15);check('effort_term',ae,row['action_cost'],1e-12)
            assert bool((abs(err)<=.0003).all() and (abs(v)<=.003).all())==row['precision_success'];assert abs(u).max()<=BOUND+1e-9
            states.append(z);commands.append(u)
        summaries[key]={'episodes':len(rows),'mean_cost':float(np.mean([x['cost'] for x in rows])),'precision_successes':sum(x['precision_success'] for x in rows)}
    states=np.array(states);commands=np.array(commands);zz=states[:,0].copy()
    for t in range(3):zz=independent.advance(zz,commands[:,t]);check('all_Hamiltonian_physical_states',zz,states[:,t+1],3e-10)
    for row in r['candidate_leading_feasibility_evaluator_only']:
        q=independent.prepare(row['branch_evaluator_only']);B=independent.jac(q)@np.linalg.inv(independent.mass(q));z=np.linalg.solve(B,GOALS[row['goal_index']]/T**2);u=np.array(r['first_designs'][row['prior']][row['goal_index']]['candidates'])
        required=np.stack([z-2*u,u-z],1);needed=abs(required).max((1,2));check('complete_candidate_leading_feasibility',needed,row['maximum_required_action'],1e-12);assert (needed<=BOUND).tolist()==row['feasible']
    integration=[];tail_max=0.;overlap=[]
    for name,ds in r['first_designs'].items():
        for gi,d in enumerate(ds):
            expected=np.array(d['expected_costs']);assert int(np.argmin(expected))==d['selected'];u=np.array(d['candidates']);p=r['priors'][name];g=GOALS[gi]
            check('design_cautious_plan',independent.cautious(p,g),d['cautious_plan'],1e-8)
            for ci,(candidate,info) in enumerate(zip(u,d['integration'])):
                bound=tail_bound(p,candidate,g);check('analytic_tail_bound',bound,info['analytic_truncation_upper_bound'],1e-14);tail_max=max(tail_max,bound)
                assert info['normal_truncation']==7. and all(x=='converged' for x in info['statuses']);assert info['estimated_error']>=0
            # Verify the full menu, including every near tie, with an independent
            # tensor Gauss-Kronrod rule versus source's Genz-Malik cubature rule.
            checks=[]
            for ci in np.argsort(expected):
                value,error,points=integrate(p,u[ci],g);difference=value-expected[ci]
                checks.append({'candidate':int(ci),'saved_value':float(expected[ci]),'independent_value':value,'independent_estimated_error':error,'saved_estimated_error':d['integration'][ci]['estimated_error'],'difference':float(difference),'points':points})
                checks[-1]['discrepancy_exceeds_threshold']=bool(abs(difference)>=max(2e-5,5*d['integration'][ci]['estimated_error']))
                if checks[-1]['discrepancy_exceeds_threshold']:
                    checks[-1]['independent_scrambled_Sobol']=qmc_verify(p,u[ci],g)
            best_checked=min(checks,key=lambda a:a['independent_value']);ranking_preserved=best_checked['candidate']==d['selected']
            integration.append({'prior':name,'goal_index':gi,'selected':d['selected'],'independent_best_of_all50':best_checked['candidate'],'ranking_preserved':ranking_preserved,'checks':checks})
            order=np.argsort(expected);a,b=order[:2];gap=expected[b]-expected[a];allow=d['integration'][a]['estimated_error']+d['integration'][b]['estimated_error']
            if gap<=allow:overlap.append({'prior':name,'goal_index':gi,'gap':float(gap),'sum_reported_errors':allow})
            radii=[expected[0]]+[np.mean(expected[1+16*i:1+16*(i+1)]) for i in range(3)];check('random_radius_values',radii,d['random_radius_expected_costs'],1e-10)
            assert [0.,.025,.05,.1][int(np.argmin(radii))]==d['random_radius']
            print('integrated',name,gi,'selected',d['selected'],'independentbest',best_checked['candidate'],flush=True)
    resources=r['resources'];assert len(states)==resources['control_episodes']==1728;assert len(noise_paths)==96
    paths=[source,rp,dp,design,ROOT/'audit_cartesian_reuse_v16_pilot.py',ROOT/'audit_cartesian_reuse_v16_pilot_results.json']
    failures=[{'prior':x['prior'],'goal_index':x['goal_index'],**c} for x in integration for c in x['checks'] if c['discrepancy_exceeds_threshold']]
    out={'status':('FIRST-ACTION INTEGRATION AUDIT FAILED; PHYSICAL AND STATISTICAL RECONSTRUCTION PASSED' if failures else 'INDEPENDENT ADAPTIVE BATCH AUDIT COMPLETED; ESTIMATED INTEGRATION ERRORS ARE NOT RIGOROUS CERTIFICATES'),'artifact_sha256':{x.name:sha(x) for x in paths},'audit_source_sha256':sha(Path(__file__)),
        'reconstruction_errors':ERROR,'gaussian_ML_records':ml_records,'gaussian_all_coordinate_gradient_checks':derivative_checks,'forecasts':forecasts,'control_summaries':summaries,'maximum_control_projected_gradient':pg_max,
        'independent_integration_rule':'gk21 tensor Gauss-Kronrod vs canonical genz-malik; same SciPy adaptive engine, independent response/posterior/QP implementation; tighter atol5e-9 rtol1e-7',
        'complete_candidate_integrals':integration,'saved_estimated_error_intervals_overlap':overlap,'maximum_analytic_truncation_upper_bound_all900_candidates':tail_max,
        'integration_discrepancies':failures,
        'resources':{'new_acquisition_intervals':0,'reused_acquisition_intervals':768,'new_control_episodes':1728,'new_control_intervals':5184,'unique_paired_control_noise_paths':96},
        'limitations':['No rigorous numerical integration error certificate from adaptive estimated errors.','All900 integrals are checked; exact local-model sign symmetry is reused for opposite goals/actions, rounding action cache keys to14decimalplaces.',
            'Both integration schemes use SciPy adaptive cubature engine; rule and model/posterior/QP code are independent.','Gaussian ML is two finite L-BFGS starts with validation selection, no global maximum-likelihood guarantee.',
            'One acquired fit, same six small goals, three pulses and one intermediate adaptation; no sustained feedback or population fit uncertainty.',
            'Current source acquires no additional calibration but repeats physical evaluation on common existing noise paths; this is exploratory reuse, not a fresh confirmatory cohort.'],
        'wall_seconds':time.time()-start}
    (ROOT/'audit_cartesian_reuse_v16_adaptive_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'seconds':out['wall_seconds'],'summaries':summaries,'rank_changes':[x for x in integration if not x['ranking_preserved']]}))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
