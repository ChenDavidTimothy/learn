#!/usr/bin/env python3
"""Independent frozen Cartesian pilot reconstruction; imports no project implementation.

Run .venv/bin/python research/audit_cartesian_reuse_v16_pilot.py
Physics uses vectorized Hamiltonian RK4, checked against independent Radau.
Statistical routines operate on whole-group design matrices, not hidden states.
"""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
T=.1;SD=.0002;R=.0004;BOUND=.5;L=.001;WV=.25;WU=.01;SCALE=L/T**2
P0=np.array([np.sqrt(2.),0.])
GOALS=L*np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1.]])
SP=np.kron(np.array([[1.5,.5]]),np.eye(2));SV=np.tile(np.eye(2),(1,2))
ERRORS={}
CAUTIOUS_CACHE={}

def compare(name,a,b,tol=1e-8):
    err=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    ERRORS[name]=max(ERRORS.get(name,0.),err)
    assert err<tol,(name,err,tol)
    return err

def mass(q):
    c=np.cos(q[...,1]);M=np.empty(q.shape[:-1]+(2,2));M[...,0,0]=3+2*c
    M[...,0,1]=M[...,1,0]=1+c;M[...,1,1]=1
    return M

def endpoint(q):
    return np.stack([np.cos(q[...,0])+np.cos(q.sum(-1)),np.sin(q[...,0])+np.sin(q.sum(-1))],-1)

def jac(q):
    J=np.zeros(q.shape[:-1]+(2,2));J[...,0,0]=-np.sin(q[...,0])-np.sin(q.sum(-1))
    J[...,0,1]=-np.sin(q.sum(-1));J[...,1,0]=np.cos(q[...,0])+np.cos(q.sum(-1));J[...,1,1]=np.cos(q.sum(-1))
    return J

def hamilton_rhs(w,u):
    q=w[...,:2];p=w[...,2:];v=np.linalg.solve(mass(q),p[...,None])[...,0]
    pd=u.copy();pd[...,1]-=np.sin(q[...,1])*v[...,0]*(v[...,0]+v[...,1])
    return np.concatenate([v,pd],-1)

def advance(z,u,steps=128):
    q=z[...,:2];w=np.concatenate([q,(mass(q)@z[...,2:,None])[...,0]],-1);dt=T/steps
    for _ in range(steps):
        k1=hamilton_rhs(w,u);k2=hamilton_rhs(w+.5*dt*k1,u)
        k3=hamilton_rhs(w+.5*dt*k2,u);k4=hamilton_rhs(w+dt*k3,u)
        w+=dt*(k1+2*k2+2*k3+k4)/6
    return np.concatenate([w[...,:2],np.linalg.solve(mass(w[...,:2]),w[...,2:,None])[...,0]],-1)

def radau(z,u):
    w=np.r_[z[:2],mass(z[:2])@z[2:]]
    out=solve_ivp(lambda t,x:hamilton_rhs(x,u),(0,T),w,method='Radau',rtol=2e-12,atol=2e-14)
    assert out.success
    w=out.y[:,-1]
    return np.r_[w[:2],np.linalg.solve(mass(w[:2]),w[2:])]

def prepare(branch):
    return np.array([-np.pi/4,np.pi/2])*(1-2*branch)

def design(X):
    n,t,_=X.shape
    return np.array([np.kron(np.eye(2),x) .reshape(2,4) for x in X.reshape(-1,2)]).reshape(n,t*2,4)

def group_H(X):
    # Construct each output row explicitly to ensure time-major flattening.
    return np.stack([np.array([[*x,0,0],[0,0,*x]]) for x in X.reshape(-1,2)]).reshape(len(X),-1,4)

def mean_fit(X,Y,weights=None):
    if weights is None:weights=np.ones(len(X))
    xx=X.reshape(-1,2);yy=Y.reshape(-1,2);ww=np.repeat(weights,X.shape[1])
    return np.linalg.solve(xx.T@(ww[:,None]*xx)+1e-10*np.eye(2),xx.T@(ww[:,None]*yy)).T

def mix_logs(H,y,B,w):
    residual=y[:,None,:]-np.einsum('nij,kj->nki',H,np.reshape(B,(-1,4)))
    return np.log(w)[None,:]-.5*np.sum(residual**2,axis=-1)/R-.5*y.shape[1]*np.log(2*np.pi*R)

def gaussian_logs(H,y,mu,C):
    delta=y-H@np.ravel(mu);V=H@C@H.transpose(0,2,1)+R*np.eye(y.shape[1])
    return -.5*(y.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+np.sum(delta*np.linalg.solve(V,delta[...,None])[...,0],axis=-1))

def prior_moments(prior):
    if prior['kind']=='mixture':
        B=np.array(prior['B']);w=np.array(prior['weights']);mu=np.sum(w[:,None,None]*B,axis=0)
        D=B.reshape(-1,4)-mu.ravel();C=(D.T*w)@D
    else:mu=np.array(prior['mean']);C=np.array(prior['cov'])
    return mu,C

def second(mu,C):
    return np.einsum('...oi,...oj->...ij',mu,mu)+C[...,:2,:2]+C[...,2:,2:]

def update(prior,observations,u):
    mu,C=prior_moments(prior);H=np.kron(np.eye(2),(.5*np.array(u))[None,:]);y=np.atleast_2d(observations)
    if prior['kind']=='mixture':
        B=np.array(prior['B']);w=np.array(prior['weights']);means=B.reshape(-1,4)@H.T
        l=np.log(w)-np.sum((y[:,None]-means)**2,axis=-1)/(2*R)
        w=np.exp(l-logsumexp(l,axis=-1)[:,None]);m=np.einsum('nk,koi->noi',w,B)
        G=np.einsum('nk,koi,koj->nij',w,B,B)
        return m,G,w
    V=H@C@H.T+R*np.eye(2);K=C@H.T@np.linalg.inv(V);post=C-K@H@C
    m=(mu.ravel()+(y-H@mu.ravel())@K.T).reshape(-1,2,2)
    return m,second(m,post),np.repeat(post[None],len(y),axis=0)

def solve_boxes(Q,b,bound=BOUND):
    # Independent full-face minimization; all faces include their free minimizer.
    Q=np.atleast_3d(Q) if np.ndim(Q)==2 else Q
    n=len(b);dim=b.shape[1];best=np.full(n,np.inf);chosen=np.zeros_like(b)
    for status in itertools.product((0,-1,1),repeat=dim):
        status=np.array(status);free=np.flatnonzero(status==0);fixed=np.flatnonzero(status!=0)
        x=np.broadcast_to(bound*status,(n,dim)).copy()
        if len(free):
            rhs=-b[:,free]
            if len(fixed):rhs-=np.einsum('nij,nj->ni',Q[:,free][:,:,fixed],x[:,fixed])
            x[:,free]=np.linalg.solve(Q[:,free][:,:,free],rhs[...,None])[...,0]
        valid=(np.abs(x)<=bound+1e-10).all(-1)
        val=np.einsum('ni,nij,nj->n',x,Q,x)+2*np.sum(b*x,axis=-1)
        take=valid&(val<best);best[take]=val[take];chosen[take]=x[take]
    assert np.isfinite(best).all()
    return chosen

def continuation(m,G,u,g,return_q=False):
    d=g/T**2;s=2.5*u;t=u
    Q=(np.einsum('ia,nij,jb->nab',SP,G,SP)+WV*np.einsum('ia,nij,jb->nab',SV,G,SV))/SCALE**2+WU*np.eye(4)
    b=(np.einsum('ia,nij,j->na',SP,G,s)-np.einsum('ia,noi,o->na',SP,m,d)+WV*np.einsum('ia,nij,j->na',SV,G,t))/SCALE**2
    x=solve_boxes(Q,b);sv=s+x@SP.T;tv=t+x@SV.T
    val=(np.einsum('ni,nij,nj->n',sv,G,sv)-2*np.einsum('o,noi,ni->n',d,m,sv)+d@d+WV*np.einsum('ni,nij,nj->n',tv,G,tv))/SCALE**2+WU*(u@u+np.sum(x*x,axis=-1))
    pg=x-np.clip(x-2*(np.einsum('nij,nj->ni',Q,x)+b),-BOUND,BOUND)
    assert abs(pg).max()<1e-7
    if return_q:return x,val,Q,b,pg
    return x,val

def quadrature(prior,u,order):
    z,w=hermgauss(order);nodes=np.array(list(itertools.product(z,z)))*np.sqrt(2);ww=np.outer(w,w).ravel()/np.pi
    H=np.kron(np.eye(2),(.5*u)[None,:]);mu,C=prior_moments(prior)
    if prior['kind']=='mixture':
        B=np.array(prior['B']);means=B.reshape(-1,4)@H.T
        return (means[:,None]+np.sqrt(R)*nodes).reshape(-1,2),(np.array(prior['weights'])[:,None]*ww).ravel()
    return H@mu.ravel()+nodes@np.linalg.cholesky(H@C@H.T+R*np.eye(2)).T,ww

def cautious(prior,g):
    key=(json.dumps(prior,sort_keys=True),tuple(g))
    if key in CAUTIOUS_CACHE:return CAUTIOUS_CACHE[key].copy()
    mu,C=prior_moments(prior);G=second(mu,C)
    P=np.kron(np.array([[2.5,1.5,.5]]),np.eye(2));V=np.tile(np.eye(2),(1,3));d=g/T**2
    Q=(P.T@G@P+WV*V.T@G@V)/SCALE**2+WU*np.eye(6);b=-P.T@mu.T@d/SCALE**2
    answer=solve_boxes(Q[None],b[None])[0].reshape(3,2)
    CAUTIOUS_CACHE[key]=answer
    return answer.copy()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started=time.time();respath=ROOT/'cartesian_reuse_v16_pilot_results.json';datapath=ROOT/'cartesian_reuse_v16_pilot_data.npz'
    source=ROOT/'cartesian_reuse_v16_pilot.py';frozen_design=ROOT/'cartesian_reuse_v16_pilot_design.md'
    result=json.loads(respath.read_text());raw=np.load(datapath);rep=result['meta']['replicate'];data={};acqpaths=[]
    compare('source_hash_valid',sha(source)==result['meta']['source_sha256'],True)
    compare('data_hash_valid',sha(datapath)==result['meta']['data_sha256'],True)
    physical_calls=0;energy_errors=[]
    for name,role,count in [('training',1,256),('validation',2,64),('testing',3,64)]:
        initial=[];commands=[];errors=[];paths=[]
        for i in range(count):
            path=[20261020,16,630,rep,role,i];rng=np.random.default_rng(np.random.SeedSequence(path))
            initial.append(np.r_[prepare(int(rng.integers(2))),[0.,0.]])
            commands.append(rng.uniform(-BOUND,BOUND,(2,2)));errors.append(rng.normal(0,SD,(3,2)));paths.append(path)
        initial=np.array(initial);commands=np.array(commands);errors=np.array(errors);states=np.array(raw[name+'_states_evaluator_only'])
        compare('acquisition_commands_rng',commands,raw[name+'_commands'],1e-15)
        compare('acquisition_paths',paths,raw[name+'_seed_paths'],1e-15)
        compare('acquisition_preparation',initial,states[:,0],1e-15)
        observed=endpoint(states[...,:2])+errors
        compare('acquisition_measurement_rng',observed,raw[name+'_observed'],1e-15)
        z=initial.copy()
        for j in range(2):
            znew=advance(z,commands[:,j]);compare('acquisition_hamiltonian_states',znew,states[:,j+1],2e-10)
            E0=.5*np.einsum('ni,nij,nj->n',z[:,2:],mass(z[:,:2]),z[:,2:]);E1=.5*np.einsum('ni,nij,nj->n',znew[:,2:],mass(znew[:,:2]),znew[:,2:])
            energy_errors.extend(abs(E1-E0-np.sum(commands[:,j]*(znew[:,:2]-z[:,:2]),axis=-1)).tolist());z=znew
        X=np.stack([.5*commands[:,0],1.5*commands[:,0]+.5*commands[:,1]],1);Y=(observed[:,1:]-P0)/T**2
        compare('acquisition_exposures',X,raw[name+'_exposures'],1e-15);data[name]=(X,Y);acqpaths+=paths;physical_calls+=count*2
        for i in (0,count-1):compare('independent_radau_physics',radau(states[i,0],commands[i,0]),states[i,1],2e-10)
    assert len(set(map(tuple,acqpaths)))==384
    X,Y=data['training'];V,Z=data['validation'];H=group_H(X);HV=group_H(V);y=Y.reshape(len(Y),-1);z=Z.reshape(len(Z),-1)
    rebuilt=[]
    for row in result['mixture_candidates']:
        k=row['components'];start=row['start'];path=[20261020,16,630,rep,5,k*100+start]
        compare('mixture_init_paths',path,row['seed_path'],1e-15)
        mu=mean_fit(X,Y);scale=np.sqrt(np.mean((Y-X@mu.T)**2)/np.mean(X**2));rng=np.random.default_rng(np.random.SeedSequence(path))
        B=mu[None]+rng.normal(0,max(.1,scale),(k,2,2));w=np.ones(k)/k
        if k==1:B[0]=mu
        history=[];converged=False
        for step in range(150):
            l=mix_logs(H,y,B,w);norm=logsumexp(l,axis=-1);history.append(float(norm.mean()));resp=np.exp(l-norm[:,None])
            w=np.maximum(resp.mean(0),1e-6);w/=w.sum();B=np.array([mean_fit(X,Y,resp[:,j]) for j in range(k)])
            if len(history)>1 and abs(history[-1]-history[-2])<1e-9*(1+abs(history[-2])):converged=True;break
        vnll=-logsumexp(mix_logs(HV,z,B,w),axis=-1).mean();rebuilt.append(float(vnll))
        compare('mixture_B_reconstruction',B,row['B'],2e-7);compare('mixture_weights_reconstruction',w,row['weights'],2e-7)
        compare('mixture_iteration_history',history,row['training_mean_log_likelihood'],2e-6);compare('mixture_validation_nll',vnll,row['validation_nll'],2e-7)
        assert converged==row['converged'] and len(history)==row['iterations']
    assert int(np.argmin(rebuilt))==result['selected_mixture']
    mean=mean_fit(X,Y);res=y-H@mean.ravel();pairs=list(zip(*np.triu_indices(4)));basis=[]
    for i,j in pairs:
        E=np.zeros((4,4));E[i,j]=E[j,i]=1;basis.append(E)
    op=np.stack([(H@E@H.transpose(0,2,1)).ravel() for E in basis],-1);target=(res[:,:,None]*res[:,None,:]-R*np.eye(4)).ravel()
    theta=np.linalg.lstsq(op,target,rcond=None)[0];rawC=np.einsum('k,kij->ij',theta,basis);e,Q=np.linalg.eigh(rawC);gnll=[]
    for row in result['gaussian_candidates']:
        C=(Q*np.maximum(e,row['covariance_floor']))@Q.T;vnll=-gaussian_logs(HV,z,mean,C).mean();gnll.append(float(vnll))
        compare('gaussian_mean',mean,row['mean'],1e-10);compare('gaussian_raw_covariance',rawC,row['raw_covariance'],1e-8)
        compare('gaussian_projected_covariance',C,row['cov'],1e-8);compare('gaussian_validation_nll',vnll,row['validation_nll'],1e-7)
        compare('gaussian_operator_rank',np.linalg.matrix_rank(op),row['moment_operator_rank'],1e-10)
        compare('gaussian_operator_condition',np.linalg.cond(op),row['moment_operator_condition'],1e-8)
    assert int(np.argmin(gnll))==result['selected_gaussian']
    forecasts={};A,C=data['testing'];HA=group_H(A);yy=C.reshape(len(C),-1)
    for name,p in result['priors'].items():
        mu,cov=prior_moments(p);compare('saved_prior_mean',mu,p['mean'],1e-10);compare('saved_prior_covariance',cov,p['cov'],1e-9)
        if p['kind']=='mixture':joint=logsumexp(mix_logs(HA,yy,np.array(p['B']),np.array(p['weights'])),axis=-1);first=logsumexp(mix_logs(HA[:,:2],yy[:,:2],np.array(p['B']),np.array(p['weights'])),axis=-1)
        else:joint=gaussian_logs(HA,yy,mu,cov);first=gaussian_logs(HA[:,:2],yy[:,:2],mu,cov)
        forecasts[name]={'joint_group_nll':float(-joint.mean()),'second_given_first_nll':float(-(joint-first).mean()),'first_nll':float(-first.mean()),'mean_response_mse':float(np.mean((C-A@mu.T)**2))}
        compare('forecast_joint_nll',forecasts[name]['joint_group_nll'],result['forecasts'][name]['joint_group_nll'],1e-8)
        compare('forecast_mean_mse',forecasts[name]['mean_response_mse'],result['forecasts'][name]['mean_response_mse'],1e-12)
    # Independent actual-policy posterior, optimizer, measurement and scoring replay.
    summaries={};qp_examples=[];controls_paths=set();allstates=[];allcommands=[];allrows=[];infeasible={}
    for key,rows in result['controls'].items():
        name,mode=key.split('_');prior=result['priors'][name];group_costs=[]
        for r in rows:
            gi=r['goal_index'];goal=GOALS[gi];u=np.array(r['commands']);states=np.array(r['states_evaluator_only']);branch=r['branch_evaluator_only']
            path=[20261020,16,630,rep,4,gi*100+r['repeat']];rng=np.random.default_rng(np.random.SeedSequence(path));noise=rng.normal(0,SD,2);direction=int(rng.integers(16));controls_paths.add(tuple(path))
            compare('control_seed_paths',path,r['seed_path'],1e-15);compare('control_initial_state',states[0],np.r_[prepare(branch),[0.,0.]],1e-15)
            if mode=='dual':D=result['first_designs'][name][gi];expected=np.array(D['candidates'][D['selected']])
            elif mode=='random':rad=result['first_designs'][name][gi]['random_radius'];expected=rad*np.array([np.cos(direction*2*np.pi/16),np.sin(direction*2*np.pi/16)])
            else:expected=cautious(prior,goal)[0]
            compare('first_actions',u[0],expected,2e-8);assert abs(u).max()<=BOUND+1e-9
            compare('control_measurement_rng',endpoint(states[1,:2])+noise,r['first_observation'],1e-15)
            m,G,post=update(prior,(np.array(r['first_observation'])-P0)/T**2,u[0])
            compare('control_posterior_mean',m[0],r['posterior_mean'],1e-9);compare('control_posterior_second_moment',G[0],r['posterior_BtB'],1e-8);compare('control_posterior',post[0],r['posterior'],1e-9)
            x,val,Q,b,pg=continuation(m,G,u[0],goal,True);compare('continuation_commands',x[0].reshape(2,2),u[1:],2e-8);compare('continuation_cost',val[0],r['model_expected_cost'],2e-7)
            compare('continuation_pg',abs(pg).max(),r['continuation_pg'],1e-7)
            if r['repeat']==0 and branch==0:qp_examples.append((Q[0],b[0],x[0]))
            p=endpoint(states[-1,:2]);v=jac(states[-1,:2])@states[-1,2:];err=p-P0-goal
            pe=err@err;ve=v@v;ae=WU*np.sum(u*u);cost=pe/L**2+WV*ve/(L/T)**2+ae
            compare('physical_endpoint',p,r['endpoint_evaluator_only'],1e-13);compare('physical_velocity',v,r['velocity_evaluator_only'],1e-13)
            compare('record_position_squared',pe,r['position_error_squared'],1e-15);compare('record_velocity_squared',ve,r['velocity_squared'],1e-15);compare('record_effort',ae,r['action_cost'],1e-12);compare('record_cost',cost,r['cost'],1e-10)
            success=bool((abs(err)<=.0003).all() and (abs(v)<=.003).all());assert success==r['precision_success']
            B=jac(states[0,:2])@np.linalg.inv(mass(states[0,:2]));required=np.linalg.solve(B,goal/T**2)
            feasible=bool(abs(np.array([required-2*u[0],u[0]-required])).max()<=BOUND);assert feasible==r['leading_fixed_probe_exact_stop_feasible_evaluator_only']
            infeasible[key]=infeasible.get(key,0)+(not feasible)
            allstates.append(states);allcommands.append(u);allrows.append(r)
        summaries[key]={'episodes':len(rows),'mean_cost':float(np.mean([r['cost'] for r in rows])),'precision_successes':sum(r['precision_success'] for r in rows),'leading_executed_probe_infeasible':infeasible[key],
            'mean_position_term':float(np.mean([r['position_error_squared']/L**2 for r in rows])),'mean_velocity_term':float(np.mean([WV*r['velocity_squared']/(L/T)**2 for r in rows])),'mean_effort_term':float(np.mean([r['action_cost'] for r in rows]))}
    allstates=np.array(allstates);allcommands=np.array(allcommands);zz=allstates[:,0].copy()
    for j in range(3):
        zz=advance(zz,allcommands[:,j]);compare('all_control_hamiltonian_physics',zz,allstates[:,j+1],3e-10)
    for i in range(0,len(allstates),191):compare('control_radau_physics',radau(allstates[i,1],allcommands[i,1]),allstates[i,2],3e-10)
    scipy_qp=[]
    for Q,b,x in qp_examples:
        fit=minimize(lambda a:a@Q@a+2*b@a,np.zeros(4),jac=lambda a:2*(Q@a+b),bounds=[(-BOUND,BOUND)]*4,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000,'maxls':50})
        diff=abs(fit.x-x).max();scipy_qp.append(float(diff));assert (fit.fun-(x@Q@x+2*b@x))<1e-8
    for w in result['feasibility_witnesses']:
        branch=w['branch_evaluator_only'];g=GOALS[w['goal_index']];zz=np.r_[prepare(branch),[0.,0.]];commands=np.array(w['last_two_commands']);assert abs(commands).max()<=BOUND+1e-12
        for u in commands:zz=radau(zz,u)
        residual=np.r_[(endpoint(zz[:2])-P0-g)/L,np.sqrt(WV)*(jac(zz[:2])@zz[2:])/(L/T)]
        compare('oracle_witness_radau_residual',residual,w['normalized_residual'],1e-8);assert np.linalg.norm(residual)<1e-7
    # Recompute every saved first-action score, then a finer independent quadrature.
    designs=[];candidate_feasibility={}
    for name,records in result['first_designs'].items():
        prior=result['priors'][name]
        for gi,d in enumerate(records):
            goal=GOALS[gi];compare('cautious_plan',cautious(prior,goal),d['cautious_plan'],2e-8)
            candidates=np.array(d['candidates']);fine=[];finest=[];original=[]
            assert candidates.shape==(50,2)
            for u in candidates:
                for order,arr in [(9,original),(17,fine),(33,finest)]:
                    obs,w=quadrature(prior,u,order);mu,G,_=update(prior,obs,u);_,loss=continuation(mu,G,u,goal);arr.append(float(w@loss))
            compare('all_quadrature9_scores',original,d['expected_costs'],3e-7)
            assert int(np.argmin(original))==d['selected']
            radii=[original[0]]+[float(np.mean(original[1+16*i:1+16*(i+1)])) for i in range(3)]
            compare('random_radius_scores',radii,d['random_radius_expected_costs'],3e-7);assert [0.,.025,.05,.1][int(np.argmin(radii))]==d['random_radius']
            feasibility=[]
            for branch in (0,1):
                q=prepare(branch);B=jac(q)@np.linalg.inv(mass(q));required=np.linalg.solve(B,goal/T**2)
                necessary=np.stack([required-2*candidates,candidates-required],1);feasibility.append((abs(necessary).max((1,2))<=BOUND).tolist())
            candidate_feasibility[f'{name}_{gi}']={'feasible_count_per_branch':[sum(x) for x in feasibility],'candidates':len(candidates),'selected_feasible_per_branch':[x[d['selected']] for x in feasibility]}
            high_order=[]
            if name=='gaussian':
                for ci in sorted(set([d['selected'],int(np.argmin(fine)),int(np.argmin(finest))])):
                    u=candidates[ci];entry={'candidate':ci,'score33':finest[ci]}
                    for order in (65,129):
                        obs,w=quadrature(prior,u,order);mu,G,_=update(prior,obs,u);_,loss=continuation(mu,G,u,goal);entry[f'score{order}']=float(w@loss)
                    high_order.append(entry)
            designs.append({'prior':name,'goal_index':gi,'saved_selected':d['selected'],'quadrature17_selected':int(np.argmin(fine)),
                'quadrature33_selected':int(np.argmin(finest)),'max_score_difference_17_33':float(np.max(np.abs(np.array(finest)-fine))),
                'max_score_difference_9_17':float(np.max(np.abs(np.array(original)-fine))),'selected_score9':original[d['selected']],'selected_score17':fine[d['selected']],
                'selected_score33':finest[d['selected']],'finest_selected_score_gap':float(finest[d['selected']]-min(finest)),
                'fine_selected_score_gap':float(fine[d['selected']]-min(fine)),'higher_order_selected_candidates':high_order})
    controls_n=sum(len(x) for x in result['controls'].values());resources=result['resources']
    compare('resource_control_episodes',controls_n,resources['control_episodes'],1e-15)
    compare('resource_control_intervals',3*controls_n,resources['control_intervals'],1e-15)
    compare('resource_acquisition_intervals',physical_calls,resources['acquired_intervals'],1e-15)
    assert len(controls_paths)==96 and not (set(map(tuple,acqpaths))&controls_paths)
    out={'status':'INDEPENDENT FROZEN PILOT RECONSTRUCTION PASSED; DESIGN DEVIATIONS RECORDED; NOT A RELEASE AUDIT',
        'artifact_sha256':{p.name:sha(p) for p in [source,frozen_design,respath,datapath]},'audit_source_sha256':sha(Path(__file__)),
        'maximum_absolute_reconstruction_errors':ERRORS,'forecasts_with_missing_conditional_diagnostic_reconstructed':forecasts,'control_summaries':summaries,
        'independent_qp_spot_checks':{'count':len(scipy_qp),'max_action_difference':max(scipy_qp)},'quadrature_refinement':designs,
        'complete_candidate_leading_feasibility_posthoc':candidate_feasibility,'maximum_hamiltonian_energy_work_error':max(energy_errors),
        'resources_reconstructed':{'acquired_episodes':384,'acquired_intervals':768,'control_episodes':controls_n,'control_intervals':3*controls_n,'unique_control_noise_paths':len(controls_paths),
            'independent_radau_acquisition_intervals':6,'independent_radau_control_intervals':len(range(0,len(allstates),191)),'independent_radau_oracle_intervals':24},
        'design_deviations':['Frozen source did not precompute leading stopping feasibility for every first-action candidate before controls; this audit computes it afterward only.',
            'Frozen source saved joint heldout likelihood and mean MSE but omitted prescribed second-observation conditional likelihood; this audit reconstructs it.',
            'Frozen metadata design basename points to mutable canonical design; this audit hashes frozen pilot design explicitly.'],
        'limitations':['One acquired fit, not independent fit-replicate uncertainty.','Prior family, exact Cartesian rest preparation, torque calibration and local constant-B approximation supplied.',
            'Three physical pulses, one intermediate observation, then two commands committed; not sustained feedback.',
            'General Gaussian covariance moment fitting is finite-sample and not maximum likelihood; pilot is not a final competitive learner comparison.',
            'Leading candidate feasibility is not an exact finite-time task certificate; nonlinear witnesses certify only their saved commands.'],
        'wall_seconds':time.time()-started}
    (ROOT/'audit_cartesian_reuse_v16_pilot_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'seconds':out['wall_seconds'],'errors':ERRORS,'summaries':summaries,'quadrature_changed':[x for x in designs if x['saved_selected']!=x['quadrature17_selected']]}))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
