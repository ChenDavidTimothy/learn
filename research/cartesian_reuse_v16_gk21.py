#!/usr/bin/env python3
"""Unreleased learned Cartesian-prior construction; frozen design is adjacent.

Learning uses only Cartesian positions, known rest/Cartesian preparation and
issued actions. Kinematics/mechanics are isolated to acquisition/evaluation.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import least_squares,minimize
from scipy.integrate import cubature
from scipy.special import logsumexp,ndtr
from threadpoolctl import threadpool_limits
from coupled_context_v14_pilot import flow,mass

T=.1;SD=.0002;R=(SD/T**2)**2;BOUND=.5;L=.001;DS=L/T**2
WV=.25;WU=.01
P0=np.array([np.sqrt(2.),0.])
GOALS=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,1.],[-1.,1.]])*L
PS=np.hstack([1.5*np.eye(2),.5*np.eye(2)])
VS=np.hstack([np.eye(2),np.eye(2)])


def role(rep,which,item):
    path=[20261020,16,630,int(rep),int(which),int(item)]
    return np.random.default_rng(np.random.SeedSequence(path)),path


def endpoint(q):return np.array([np.cos(q[0])+np.cos(q.sum()),np.sin(q[0])+np.sin(q.sum())])


def jacobian(q):
    return np.array([[-np.sin(q[0])-np.sin(q.sum()),-np.sin(q.sum())],
                     [np.cos(q[0])+np.cos(q.sum()),np.cos(q.sum())]])


def preparation(branch):return np.array([(-1 if branch==0 else 1)*np.pi/4,(1 if branch==0 else -1)*np.pi/2])


def acquire(rep,which,count):
    states=[];commands=[];observed=[];exposures=[];paths=[];nfev=[]
    for n in range(count):
        rng,path=role(rep,which,n);q=preparation(int(rng.integers(2)));z=np.r_[q,[0.,0.]]
        u=rng.uniform(-BOUND,BOUND,(2,2));zs=[z.copy()];calls=0
        for a in u:
            z,k=flow(z[:2],z[2:],a,T);calls+=k;zs.append(z.copy())
        zs=np.array(zs);p=np.array([endpoint(x[:2]) for x in zs]);o=p+rng.normal(0,SD,p.shape)
        states.append(zs);commands.append(u);observed.append(o);paths.append(path);nfev.append(calls)
        exposures.append([.5*u[0],1.5*u[0]+.5*u[1]])
    raw=dict(states_evaluator_only=np.array(states),commands=np.array(commands),observed=np.array(observed),
        exposures=np.array(exposures),seed_paths=np.array(paths),plant_nfev=np.array(nfev))
    return raw,raw['exposures'],(raw['observed'][:,1:]-P0)/T**2


def matrix_H(X):
    # X shape(n,time,2); outputs flattened time-major, b=row-major B.
    H=np.zeros((len(X),2*X.shape[1],4))
    for t in range(X.shape[1]):
        H[:,2*t,:2]=X[:,t];H[:,2*t+1,2:]=X[:,t]
    return H


def fit_mean(X,Y,w=None):
    if w is None:w=np.ones(len(X))
    G=np.einsum('n,nti,ntj->ij',w,X,X)
    rhs=np.einsum('n,nti,nto->io',w,X,Y)
    return np.linalg.solve(G+1e-10*np.eye(2),rhs).T


def mixture_log_parts(X,Y,B,weights):
    residual=Y[:,None]-np.einsum('nti,koi->nkto',X,B)
    return np.log(weights)[None,:]-np.sum(residual**2,axis=(2,3))/(2*R)-Y.shape[1]*np.log(2*np.pi*R)


def fit_mixtures(X,Y,V,Z,rep):
    records=[];mean=fit_mean(X,Y)
    for count in (1,2,3):
        for start in range(1 if count==1 else 8):
            rng,path=role(rep,5,count*100+start)
            scale=np.sqrt(np.mean((Y-np.einsum('nti,oi->nto',X,mean))**2)/np.mean(X**2))
            B=mean[None]+rng.normal(0,max(.1,scale),(count,2,2));weights=np.ones(count)/count;history=[]
            if count==1:B[0]=mean
            converged=False
            for iteration in range(150):
                logs=mixture_log_parts(X,Y,B,weights);ll=float(logsumexp(logs,axis=1).mean());history.append(ll)
                resp=np.exp(logs-logsumexp(logs,axis=1)[:,None])
                weights=np.maximum(resp.mean(0),1e-6);weights/=weights.sum()
                B=np.array([fit_mean(X,Y,resp[:,k]) for k in range(count)])
                if len(history)>1 and abs(history[-1]-history[-2])<1e-9*(1+abs(history[-2])):
                    converged=True;break
            validation=-float(logsumexp(mixture_log_parts(V,Z,B,weights),axis=1).mean())
            records.append(dict(kind='mixture',components=count,start=start,seed_path=path,B=B,weights=weights,
                training_mean_log_likelihood=history,converged=converged,iterations=len(history),validation_nll=validation))
    selected=min(range(len(records)),key=lambda i:records[i]['validation_nll'])
    return records,selected


def gaussian_nll(X,Y,mean,cov):
    H=matrix_H(X);r=Y.reshape(len(Y),-1)-np.einsum('nij,j->ni',H,mean.ravel())
    C=H@cov@H.transpose(0,2,1)+R*np.eye(r.shape[1])
    return float(np.mean(.5*(r.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+
        np.einsum('ni,ni->n',r,np.linalg.solve(C,r[...,None])[...,0]))))


def fit_gaussian(X,Y,V,Z):
    mean=fit_mean(X,Y);H=matrix_H(X);res=Y.reshape(len(Y),-1)-H@mean.ravel()
    indices=list(zip(*np.triu_indices(4)));basis=[]
    for i,j in indices:
        E=np.zeros((4,4));E[i,j]=1;E[j,i]=1;basis.append(E)
    basis=np.array(basis)
    operator=np.einsum('nai,kij,nbj->nabk',H,basis,H)
    target=res[:,:,None]*res[:,None,:]-R*np.eye(4)
    A=operator.reshape(-1,10);b=target.ravel();theta=np.linalg.lstsq(A,b,rcond=None)[0]
    raw=np.einsum('k,kij->ij',theta,basis);eigen,vectors=np.linalg.eigh(raw);records=[]
    for floor in (0.,1e-4,1e-2):
        cov=(vectors*np.maximum(eigen,floor))@vectors.T
        records.append(dict(kind='gaussian',mean=mean,cov=cov,covariance_floor=floor,
            raw_covariance=raw,raw_eigenvalues=eigen,moment_operator_rank=int(np.linalg.matrix_rank(A)),
            moment_operator_condition=float(np.linalg.cond(A)),moment_residual_rms=float(np.sqrt(np.mean((A@theta-b)**2))),
            validation_nll=gaussian_nll(V,Z,mean,cov)))
    selected=min(range(len(records)),key=lambda i:records[i]['validation_nll'])
    return records,selected


def gaussian_objective(parameters,H,Y):
    """Grouped Gaussian marginal likelihood, exact derivatives in m and F."""
    m=parameters[:4];F=parameters[4:].reshape(4,4);cov=F@F.T
    residual=Y.reshape(len(Y),4)-H@m
    C=H@cov@H.transpose(0,2,1)+R*np.eye(4)
    inv=np.linalg.inv(C);z=np.einsum('nij,nj->ni',inv,residual)
    value=.5*np.mean(4*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+np.sum(residual*z,axis=1))
    gm=-np.einsum('nri,nr->i',H,z)/len(Y)
    gc=.5*np.einsum('nri,nrs,nsj->ij',H,inv-z[:,:,None]*z[:,None,:],H)/len(Y)
    return float(value),np.r_[gm,(2*gc@F).ravel()]


def refine_gaussian(X,Y,V,Z,records,mixture):
    H=matrix_H(X);starts=[('moment',records[0]['mean'],records[0]['cov']),
        ('mixture_moments',mixture.mean,mixture.cov)]
    checks=[]
    for name,m,cov in starts:
        e,Q=np.linalg.eigh(cov);F=Q*np.sqrt(np.maximum(e,1e-8))
        x=np.r_[m.ravel(),F.ravel()];value,gradient=gaussian_objective(x,H,Y)
        rng=np.random.default_rng(np.random.SeedSequence([20261020,16,630,0,8,len(checks)]))
        direction=rng.normal(size=20);direction/=np.linalg.norm(direction);step=1e-6
        fd=(gaussian_objective(x+step*direction,H,Y)[0]-gaussian_objective(x-step*direction,H,Y)[0])/(2*step)
        error=float(abs(fd-gradient@direction));assert error<1e-4,(name,error)
        checks.append(dict(start=name,directional_derivative_error=error,finite_difference_step=step))
        sol=minimize(lambda p:gaussian_objective(p,H,Y),x,jac=True,method='L-BFGS-B',
            options=dict(maxiter=300,ftol=1e-11,gtol=1e-8,maxls=40))
        F=sol.x[4:].reshape(4,4);m=sol.x[:4].reshape(2,2);cov=F@F.T
        records.append(dict(kind='gaussian',fit='grouped_ML',start=name,mean=m,cov=cov,
            training_nll=float(sol.fun),success=bool(sol.success),message=str(sol.message),
            iterations=int(sol.nit),nfev=int(sol.nfev),gradient_max=float(np.max(np.abs(sol.jac))),
            initial_training_nll=value,validation_nll=gaussian_nll(V,Z,m,cov)))
    return min(range(len(records)),key=lambda i:records[i]['validation_nll']),checks


class Prior:
    def __init__(self,kind,B=None,weights=None,mean=None,cov=None):
        self.kind=kind
        if kind=='mixture':
            self.B=np.array(B);self.weights=np.array(weights);self.mean=np.einsum('k,koi->oi',self.weights,self.B)
            delta=self.B.reshape(-1,4)-self.mean.ravel();self.cov=np.einsum('k,ki,kj->ij',self.weights,delta,delta)
        else:self.mean=np.array(mean);self.cov=np.array(cov)

    def moments(self):
        G=self.mean.T@self.mean+self.cov[:2,:2]+self.cov[2:,2:]
        return self.mean[None],G[None]

    def posterior(self,y,u):
        y=np.atleast_2d(y);H=matrix_H(np.array([[.5*u]]))[0]
        if self.kind=='mixture':
            predictions=self.B.reshape(-1,4)@H.T
            logs=np.log(self.weights)[None]-np.sum((y[:,None]-predictions[None])**2,axis=2)/(2*R)
            w=np.exp(logs-logsumexp(logs,axis=1)[:,None])
            M=np.einsum('nk,koi->noi',w,self.B);G=np.einsum('nk,koi,koj->nij',w,self.B,self.B)
            return M,G,w
        S=H@self.cov@H.T+R*np.eye(2);K=np.linalg.solve(S,H@self.cov).T
        means=self.mean.ravel()+(y-self.mean.ravel()@H.T)@K.T
        C=self.cov-K@H@self.cov;M=means.reshape(-1,2,2)
        G=np.einsum('noi,noj->nij',M,M)+C[:2,:2]+C[2:,2:]
        return M,G,np.repeat(C[None],len(y),axis=0)

    def observations(self,u,order=9):
        nodes,weights=hermgauss(order);xy=np.array(list(itertools.product(nodes,nodes)))*np.sqrt(2)
        w=np.outer(weights,weights).ravel()/np.pi;H=matrix_H(np.array([[.5*u]]))[0]
        if self.kind=='mixture':
            means=self.B.reshape(-1,4)@H.T
            return (means[:,None]+np.sqrt(R)*xy[None]).reshape(-1,2),(self.weights[:,None]*w[None]).ravel()
        C=H@self.cov@H.T+R*np.eye(2);chol=np.linalg.cholesky(C)
        return self.mean.ravel()@H.T+xy@chol.T,w

    def record(self):
        d=dict(kind=self.kind,mean=self.mean,cov=self.cov)
        if self.kind=='mixture':d.update(B=self.B,weights=self.weights)
        return d


def box_qp(Q,q,bound=BOUND):
    """Exact small SPD boxQP by all faces for infeasible unconstrained cases."""
    Q=np.asarray(Q);q=np.asarray(q);x=np.linalg.solve(Q,-q[...,None])[...,0]
    bad=np.flatnonzero(np.any(np.abs(x)>bound+1e-12,axis=1));n=q.shape[1]
    if len(bad):
        A=Q[bad];b=q[bad];best=np.full(len(bad),np.inf);chosen=np.zeros_like(b)
        for status in itertools.product((-1,0,1),repeat=n):
            free=np.flatnonzero(np.array(status)==0);fixed=np.flatnonzero(np.array(status)!=0)
            if not len(fixed):continue
            candidate=np.zeros_like(b);candidate[:,fixed]=bound*np.array(status)[fixed]
            if len(free):
                rhs=-b[:,free]-np.einsum('nij,nj->ni',A[:,free][:,:,fixed],candidate[:,fixed])
                candidate[:,free]=np.linalg.solve(A[:,free][:,:,free],rhs[...,None])[...,0]
            feasible=np.all(np.abs(candidate)<=bound+1e-10,axis=1)
            value=np.einsum('ni,nij,nj->n',candidate,A,candidate)+2*np.sum(b*candidate,axis=1)
            improve=feasible&(value<best);chosen[improve]=candidate[improve];best[improve]=value[improve]
        assert np.all(np.isfinite(best));x[bad]=chosen
    gradient=2*(np.einsum('nij,nj->ni',Q,x)+q)
    pg=x-np.clip(x-gradient,-bound,bound)
    assert np.max(np.abs(pg))<1e-7,float(np.max(np.abs(pg)))
    return x,np.max(np.abs(pg),axis=1)


def continuation(M,G,u0,goal):
    d=goal/T**2;s=2.5*u0;t=u0
    Q=(PS.T[None]@G@PS+WV*VS.T[None]@G@VS)/DS**2+WU*np.eye(4)
    q=((np.einsum('ai,nij,j->na',PS.T,G,s)-np.einsum('ai,noi,o->na',PS.T,M,d))+
        WV*np.einsum('ai,nij,j->na',VS.T,G,t))/DS**2
    x,pg=box_qp(Q,q)
    sv=s+x@PS.T;tv=t+x@VS.T
    loss=(np.einsum('ni,nij,nj->n',sv,G,sv)-2*np.einsum('o,noi,ni->n',d,M,sv)+d@d+
        WV*np.einsum('ni,nij,nj->n',tv,G,tv))/DS**2+WU*(u0@u0+np.sum(x*x,axis=1))
    return x.reshape(-1,2,2),loss,pg


def cautious(prior,goal):
    M,G=prior.moments();P=np.hstack([2.5*np.eye(2),1.5*np.eye(2),.5*np.eye(2)]);V=np.tile(np.eye(2),(1,3));d=goal/T**2
    Q=(P.T[None]@G@P+WV*V.T[None]@G@V)/DS**2+WU*np.eye(6)
    q=-np.einsum('ai,noi,o->na',P.T,M,d)/DS**2
    x,pg=box_qp(Q,q)
    return x[0].reshape(3,2),float(pg[0])


def preposterior_cost(prior,goal,u,atol=1e-8,rtol=1e-7):
    """Adaptive2D normal integral; truncate standard normals to[-7,7]."""
    H=matrix_H(np.array([[.5*u]]))[0];count=0;d=goal/T**2
    tail_probability=4*ndtr(-7.)
    tail_norm_second_moment=4*7*np.exp(-.5*7**2)/np.sqrt(2*np.pi)+8*ndtr(-7.)
    if prior.kind=='mixture':
        centers=prior.B.reshape(-1,4)@H.T;chols=np.repeat((np.sqrt(R)*np.eye(2))[None],len(centers),axis=0);weights=prior.weights
        zero_cost=(np.sum((np.einsum('koi,i->ko',prior.B,2.5*u)-d)**2,axis=1)+WV*np.sum((prior.B@u)**2,axis=1))/DS**2+WU*(u@u)
        tail_bound=float(zero_cost.max()*tail_probability)
    else:
        centers=(H@prior.mean.ravel())[None];chols=np.linalg.cholesky(H@prior.cov@H.T+R*np.eye(2))[None];weights=np.ones(1)
        K=np.linalg.solve(H@prior.cov@H.T+R*np.eye(2),H@prior.cov).T
        C=prior.cov-K@H@prior.cov;delta=K@chols[0]
        Ap=np.column_stack([delta[:,j].reshape(2,2)@(2.5*u) for j in range(2)])
        Av=np.column_stack([delta[:,j].reshape(2,2)@u for j in range(2)])
        Gc=C[:2,:2]+C[2:,2:]
        a=(2*np.sum((prior.mean@(2.5*u)-d)**2)+2*WV*np.sum((prior.mean@u)**2)+
            (2.5*u)@Gc@(2.5*u)+WV*u@Gc@u)/DS**2+WU*u@u
        b=2*(np.linalg.norm(Ap,2)**2+WV*np.linalg.norm(Av,2)**2)/DS**2
        tail_bound=float(a*tail_probability+b*tail_norm_second_moment)
    total=0.;error=0.;subdivisions=0;statuses=[]
    for center,chol,weight in zip(centers,chols,weights):
        def integrand(z):
            nonlocal count
            count+=len(z);y=center+z@chol.T;M,G,_=prior.posterior(y,u)
            _,loss,_=continuation(M,G,u,goal)
            return loss*np.exp(-.5*np.sum(z*z,axis=1))/(2*np.pi)
        fit=cubature(integrand,[-7.,-7.],[7.,7.],rule='gk21',atol=atol,rtol=rtol,max_subdivisions=2000)
        total+=weight*float(fit.estimate);error+=weight*float(fit.error);subdivisions+=fit.subdivisions;statuses.append(fit.status)
    assert all(s=='converged' for s in statuses),statuses
    return total,dict(estimated_error=error,subdivisions=subdivisions,integrand_points=count,statuses=statuses,
        normal_truncation=7.,analytic_truncation_upper_bound=tail_bound,atol=atol,rtol=rtol,rule='gk21')


def first_design(prior,goal,order=9):
    task,pg=cautious(prior,goal);angles=np.arange(16)*2*np.pi/16
    directions=np.stack([np.cos(angles),np.sin(angles)],axis=1)
    candidates=[np.zeros(2)];labels=[['grid',0.,0]]
    for radius in (.025,.05,.1):
        for i,direction in enumerate(directions):candidates.append(radius*direction);labels.append(['grid',radius,i])
    candidates.append(task[0]);labels.append(['cautious',float(np.linalg.norm(task[0])),0]);values=[];maxpg=[];integration=[]
    for u in candidates:
        value,info=preposterior_cost(prior,goal,u);values.append(value);integration.append(info)
    best=int(np.argmin(values));radius_scores=[values[0]]+[float(np.mean(values[1+16*i:1+16*(i+1)])) for i in range(3)]
    random_radius=(0.,.025,.05,.1)[int(np.argmin(radius_scores))]
    return dict(cautious_plan=task,cautious_pg=pg,candidates=np.array(candidates),labels=labels,expected_costs=np.array(values),
        selected=best,random_radius=random_radius,random_radius_expected_costs=radius_scores,
        integration=integration,integration_method='adaptive2D normal truncated7',legacy_unused_order_argument=order)


def witnesses():
    rows=[]
    for branch in (0,1):
        q=preparation(branch);B=jacobian(q)@np.linalg.inv(mass(q))
        for gi,goal in enumerate(GOALS):
            target=P0+goal;calls=0
            def residual(flat):
                nonlocal calls
                calls+=1
                z=np.r_[q,[0.,0.]]
                for u in flat.reshape(2,2):z,_=flow(z[:2],z[2:],u,T)
                return np.r_[(endpoint(z[:2])-target)/L,np.sqrt(WV)*(jacobian(z[:2])@z[2:])/(L/T)]
            initial=np.linalg.solve(B,goal/T**2);x0=np.r_[initial,-initial]
            fit=least_squares(residual,x0,bounds=(-BOUND,BOUND),max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-12)
            assert np.linalg.norm(fit.fun)<1e-7
            rows.append(dict(branch_evaluator_only=branch,goal_index=gi,u0=[0.,0.],last_two_commands=fit.x.reshape(2,2),
                normalized_residual=fit.fun,nfev=int(fit.nfev),status=int(fit.status)))
            rows[-1]['residual_calls']=calls
            rows[-1]['plant_flow_calls']=2*calls
    return rows


def candidate_feasibility(designs):
    """External diagnosis only; never used to select or change an ordinary action."""
    rows=[]
    for name,goal_designs in designs.items():
        for gi,design in enumerate(goal_designs):
            for branch in (0,1):
                q=preparation(branch);B=jacobian(q)@np.linalg.inv(mass(q));z=np.linalg.solve(B,GOALS[gi]/T**2)
                u0=np.array(design['candidates']);needed=np.stack([z-2*u0,u0-z],axis=1)
                rows.append(dict(prior=name,goal_index=gi,branch_evaluator_only=branch,
                    maximum_required_action=np.max(np.abs(needed),axis=(1,2)),
                    feasible=np.max(np.abs(needed),axis=(1,2))<=BOUND))
    return rows


def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,default=0)
    parser.add_argument('--output-prefix',default='research/cartesian_reuse_v16_refinement')
    parser.add_argument('--input-data',default=None);args=parser.parse_args()
    prefix=Path(args.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True);start=time.time();raw={};data={}
    with threadpool_limits(limits=1):
        for name,which,count in [('training',1,256),('validation',2,64),('testing',3,64)]:
            if args.input_data:
                with np.load(args.input_data) as source:
                    physical={k[len(name)+1:]:source[k] for k in source.files if k.startswith(name+'_')}
                X=physical['exposures'];Y=(physical['observed'][:,1:]-P0)/T**2
            else:physical,X,Y=acquire(args.replicate,which,count)
            data[name]=(X,Y);raw.update({name+'_'+k:v for k,v in physical.items()})
        np.savez_compressed(str(prefix)+'_data.npz',**raw)
        X,Y=data['training'];V,Z=data['validation'];records,selection=fit_mixtures(X,Y,V,Z,args.replicate)
        gaussians,gselection=fit_gaussian(X,Y,V,Z);mixrow=records[selection]
        mixture=Prior('mixture',B=mixrow['B'],weights=mixrow['weights'])
        gselection,gaussian_gradient_checks=refine_gaussian(X,Y,V,Z,gaussians,mixture);grow=gaussians[gselection]
        priors={'mixture':mixture,
                'gaussian':Prior('gaussian',mean=grow['mean'],cov=grow['cov'])}
        priors['matched']=Prior('gaussian',mean=priors['mixture'].mean,cov=priors['mixture'].cov)
        priors['point']=Prior('gaussian',mean=priors['mixture'].mean,cov=np.zeros((4,4)))
        forecasts={}
        A,C=data['testing']
        for name,prior in priors.items():
            nll=(-float(logsumexp(mixture_log_parts(A,C,prior.B,prior.weights),axis=1).mean()) if prior.kind=='mixture'
                 else gaussian_nll(A,C,prior.mean,prior.cov))
            first=(-float(logsumexp(mixture_log_parts(A[:,:1],C[:,:1],prior.B,prior.weights),axis=1).mean()) if prior.kind=='mixture'
                else gaussian_nll(A[:,:1],C[:,:1],prior.mean,prior.cov))
            forecasts[name]=dict(joint_group_nll=nll,first_observation_nll=first,second_given_first_nll=nll-first,
                mean_response_mse=float(np.mean((C-np.einsum('nti,oi->nto',A,prior.mean))**2)))
        print('models learned',selection,mixrow['components'],gselection,forecasts,'seconds',time.time()-start,flush=True)
        witness=witnesses();designs={};designstart=time.time()
        for name,prior in priors.items():
            if name=='point':continue
            designs[name]=[first_design(prior,g) for g in GOALS]
            print('first designs',name,[d['selected'] for d in designs[name]],'seconds',time.time()-designstart,flush=True)
        # Matched prior moments must give identical fixed open-loop actions.
        for goal in GOALS:assert np.max(np.abs(cautious(priors['mixture'],goal)[0]-cautious(priors['matched'],goal)[0]))<1e-9
        feasibility=candidate_feasibility(designs)
        methods=[(name,mode) for name in ('mixture','gaussian','matched') for mode in ('dual','task')]
        methods +=[('mixture','random'),('gaussian','random'),('point','task')]
        controls={}
        for name,mode in methods:
            prior=priors[name];key=name+'_'+mode;rows=[]
            for gi,goal in enumerate(GOALS):
                design=designs.get(name,[None]*len(GOALS))[gi]
                for repeat in range(16):
                    rng,path=role(args.replicate,4,gi*100+repeat)
                    noise=rng.normal(0,SD,2);direction=int(rng.integers(16))
                    if mode=='dual':u0=design['candidates'][design['selected']]
                    elif mode=='random':u0=design['random_radius']*np.array([np.cos(direction*2*np.pi/16),np.sin(direction*2*np.pi/16)])
                    else:u0=cautious(prior,goal)[0][0]
                    for branch in (0,1):
                        q=preparation(branch);z=np.r_[q,[0.,0.]];zs=[z.copy()];z,n=flow(z[:2],z[2:],u0,T);zs.append(z.copy())
                        observed=endpoint(z[:2])+noise
                        M,G,posterior=prior.posterior((observed-P0)/T**2,u0)
                        last,predcost,kkt=continuation(M,G,u0,goal);commands=np.vstack([u0,last[0]]);nfev=n
                        for u in last[0]:z,k=flow(z[:2],z[2:],u,T);nfev+=k;zs.append(z.copy())
                        p=endpoint(z[:2]);v=jacobian(z[:2])@z[2:];error=p-P0-goal
                        action_cost=WU*np.sum(commands**2);cost=float(np.sum((error/L)**2)+WV*np.sum((v/(L/T))**2)+action_cost)
                        trueB=jacobian(q)@np.linalg.inv(mass(q));required=np.linalg.solve(trueB,goal/T**2)
                        necessary=np.array([required-2*u0,u0-required])
                        rows.append(dict(goal_index=gi,repeat=repeat,branch_evaluator_only=branch,seed_path=path,
                            first_observation=observed,commands=commands,states_evaluator_only=np.array(zs),
                            endpoint_evaluator_only=p,velocity_evaluator_only=v,posterior=posterior[0],posterior_mean=M[0],posterior_BtB=G[0],
                            model_expected_cost=float(predcost[0]),continuation_pg=float(kkt[0]),cost=cost,
                            position_error_squared=float(error@error),velocity_squared=float(v@v),action_cost=float(action_cost),
                            precision_success=bool(np.all(np.abs(error)<=.0003) and np.all(np.abs(v)<=.003)),
                            leading_fixed_probe_exact_stop_feasible_evaluator_only=bool(np.max(np.abs(necessary))<=BOUND),plant_nfev=nfev))
            controls[key]=rows
            print(key,'cost',np.mean([r['cost'] for r in rows]),'precision',sum(r['precision_success'] for r in rows),'/',len(rows),flush=True)
        result=dict(meta=dict(status='EXPLORATORY LEARNED CARTESIAN PRIOR/BATCH CONTROL; NOT SUSTAINED FEEDBACK',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='cartesian_reuse_v16_design.md',replicate=args.replicate,
            T=T,sensor_sd=SD,action_bound=BOUND,goal_scale=L,velocity_weight=WV,action_weight=WU,exact_preparation=P0,
            data_sha256=hashlib.sha256(Path(str(prefix)+'_data.npz').read_bytes()).hexdigest()),
            mixture_candidates=records,selected_mixture=selection,gaussian_candidates=gaussians,selected_gaussian=gselection,
            gaussian_gradient_checks=gaussian_gradient_checks,
            input_data_reuse=None if not args.input_data else dict(file=args.input_data,sha256=hashlib.sha256(Path(args.input_data).read_bytes()).hexdigest()),
            priors={k:p.record() for k,p in priors.items()},forecasts=forecasts,feasibility_witnesses=witness,first_designs=designs,
            candidate_leading_feasibility_evaluator_only=feasibility,
            controls=controls,resources=dict(acquired_episodes=384,acquired_intervals=768,acquired_simulated_seconds=76.8,
                acquired_scalar_sensor_readings=384*3*2,control_episodes=sum(map(len,controls.values())),
                control_intervals=3*sum(map(len,controls.values())),control_simulated_seconds=T*3*sum(map(len,controls.values())),
                control_observed_scalar_readings=2*sum(map(len,controls.values())),
                paired_control_noise_paths=6*16,noise_draws_per_path=2,
                oracle_optimizer_reported_nfev=sum(w['nfev'] for w in witness),
                oracle_actual_residual_calls=sum(w['residual_calls'] for w in witness),
                oracle_plant_flow_calls=sum(w['plant_flow_calls'] for w in witness),
                new_acquisition_intervals=0 if args.input_data else 768),
            wall_seconds=time.time()-start)
        Path(str(prefix)+'_results.json').write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n')
    print('saved',prefix,'seconds',time.time()-start,flush=True)


if __name__=='__main__':main()
