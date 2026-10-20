#!/usr/bin/env python3
"""Independent full feedback artifact replay; no canonical implementation imports.

The QP is constructed by direct raw second moments of an eight-dimensional
random vector and linear terminal maps, independently of component cross-term
formulas in the implementation. Integrals are audited separately by theory.
"""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as mathcheck

ROOT=Path(__file__).resolve().parent
T=.1;R=.0004;SD=.0002;L=.001;BOUND=.5;WV=.25;WU=.01;SCALE=L/T**2
P0=mathcheck.P0
SEQUENCES=L*np.array([[[1,0],[1,1],[0,1],[-1,1],[-1,0],[0,0]],[[0,1],[1,1],[1,0],[0,0],[-1,0],[-1,1]]])
ERRORS={}

def check(name,a,b,tol=1e-8):
    error=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    ERRORS[name]=max(ERRORS.get(name,0.),error);assert error<tol,(name,error,tol)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def to_arrays(record):return tuple(np.array(record[k],dtype=float) for k in ('weights','means','covs'))

def check_belief(name,actual,record):
    for key,a in zip(('weights','means','covs'),actual):check(name+'_'+key,a,record[key],2e-8)

def deterministic_step(z,u):
    impulse=z[...,4:].reshape(z.shape[:-1]+(2,2))@u
    return np.concatenate([z[...,:2]+z[...,2:4]+.5*impulse,z[...,2:4]+impulse,z[...,4:]],axis=-1)

def predict(belief,u):
    weights,means,covs=belief;A=deterministic_step(np.eye(8),u).T
    return weights.copy(),deterministic_step(means,u),A@covs@A.T

def observe(belief,y):
    weights,means,covs=belief;updated=[];posterior_covs=[];logs=[]
    for w,m,C in zip(weights,means,covs):
        S=C[:2,:2]+R*np.eye(2);K=C[:,:2]@np.linalg.inv(S)
        updated.append(m+K@(y-m[:2]));Cp=C-K@C[:2];posterior_covs.append((Cp+Cp.T)/2)
        logs.append(np.log(max(w,1e-300))+multivariate_normal.logpdf(y,mean=m[:2],cov=S))
    logs=np.array(logs);w=np.exp(logs-logsumexp(logs))
    return w,np.array(updated),np.array(posterior_covs)

def moments(belief):
    w,m,C=belief;mean=w@m
    second=np.einsum('k,kij->ij',w,C)+np.einsum('k,ki,kj->ij',w,m,m)
    return mean,second

def B_map(v):
    H=np.zeros((2,8));H[0,4:6]=v;H[1,6:8]=v;return H

def plan(belief,goal,steps,spent=0.):
    mean,second=moments(belief);g=goal/T**2
    D=np.hstack([np.eye(2),steps*np.eye(2),np.zeros((2,4))]);W=np.hstack([np.zeros((2,2)),np.eye(2),np.zeros((2,4))])
    P=np.kron(np.arange(steps-.5,0,-1)[None],np.eye(2));V=np.tile(np.eye(2),(1,steps))
    A=np.array([B_map(v) for v in P.T]);C=np.array([B_map(v) for v in V.T])
    Q=(np.einsum('ari,ij,brj->ab',A,second,A)+WV*np.einsum('ari,ij,brj->ab',C,second,C))/SCALE**2+WU*np.eye(steps*2)
    b=(np.einsum('ari,ij,rj->a',A,second,D)-np.einsum('r,ari,i->a',g,A,mean)+WV*np.einsum('ari,ij,rj->a',C,second,W))/SCALE**2
    constant=(np.trace(D@second@D.T)-2*g@(D@mean)+g@g+WV*np.trace(W@second@W.T))/SCALE**2+WU*spent
    answer=mathcheck.solve_boxes(Q[None],b[None])[0];value=answer@Q@answer+2*b@answer+constant
    pg=answer-np.clip(answer-2*(Q@answer+b),-BOUND,BOUND)
    assert abs(pg).max()<1e-7
    return answer.reshape(steps,2),float(value),float(abs(pg).max())

def mixture_predictive_nll(H,y,B,w,covs):
    parts=[]
    for M,weight,C in zip(B,w,covs):
        pred=H@M.ravel();V=H@C@H.transpose(0,2,1)+R*np.eye(y.shape[1]);res=y-pred
        parts.append(np.log(weight)-.5*(y.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+np.sum(res*np.linalg.solve(V,res[...,None])[...,0],-1)))
    return float(-logsumexp(np.array(parts),axis=0).mean())

def reconstruct_priors(parent,data,candidates):
    X=data['training_exposures'];Y=(data['training_observed'][:,1:]-P0)/T**2;H=mathcheck.group_H(X);y=Y.reshape(-1,4)
    V=data['validation_exposures'];Z=(data['validation_observed'][:,1:]-P0)/T**2;HV=mathcheck.group_H(V);z=Z.reshape(-1,4)
    rebuilt=[]
    for i,row in enumerate(parent['mixture_candidates']):
        B=np.array(row['B']);w=np.array(row['weights']);parts=mathcheck.mix_logs(H,y,B,w);resp=np.exp(parts-logsumexp(parts,axis=1)[:,None]);covs=[]
        for k in range(len(w)):
            xx=X.reshape(-1,2);ww=np.repeat(resp[:,k],2);gram=xx.T@(ww[:,None]*xx)+1e-10*np.eye(2)
            covs.append(np.kron(np.eye(2),R*np.linalg.inv(gram)))
        covs=np.array(covs);vnll=mixture_predictive_nll(HV,z,B,w,covs);r=candidates[i]
        check('predictive_prior_component_means',B,r['B'],1e-15);check('predictive_prior_weights',w,r['weights'],1e-15);check('predictive_curvature_covariance',covs,r['coef_covs'],1e-7)
        check('predictive_effective_group_counts',resp.sum(0),r['effective_groups'],1e-10);check('predictive_validation_nll',vnll,r['validation_nll'],1e-9)
        assert r['parent_candidate']==i;rebuilt.append((vnll,B,w,covs))
    selected=int(np.argmin([x[0] for x in rebuilt]));_,B,w,C=rebuilt[selected];m=np.zeros((len(w),8));m[:,4:]=B.reshape(-1,4);covs=np.zeros((len(w),8,8));covs[:,4:,4:]=C;mix=(w,m,covs)
    row=parent['gaussian_candidates'][parent['selected_gaussian']];m=np.zeros((1,8));m[0,4:]=np.ravel(row['mean']);C=np.zeros((1,8,8));C[0,4:,4:]=row['cov'];gaussian=(np.ones(1),m,C)
    mean,second=moments(mix);matched=(np.ones(1),mean[None],(second-np.outer(mean,mean))[None])
    return {'mixture':mix,'gaussian':gaussian,'matched':matched},selected

def main():
    started=time.time();source=ROOT/'cartesian_feedback_v16.py';design=ROOT/'cartesian_feedback_v16_design.md';rp=ROOT/'cartesian_feedback_v16_pilot_results.json';r=json.loads(rp.read_text())
    parentpath=ROOT/r['meta']['parent_results'];parent=json.loads(parentpath.read_text());dp=parentpath.with_name(parentpath.name.replace('_results.json','_data.npz'));data=np.load(dp)
    assert sha(source)==r['meta']['source_sha256'];assert sha(parentpath)==r['meta']['parent_sha256'];assert sha(dp)==r['meta']['data_sha256'];assert sha(ROOT/'cartesian_reuse_v16_refinement.py')==r['meta']['base_source_sha256']
    prior_audit=json.loads((ROOT/'audit_cartesian_reuse_v16_adaptive_results.json').read_text())
    for name,h in prior_audit['artifact_sha256'].items():assert sha(ROOT/name)==h
    priors,selected=reconstruct_priors(parent,data,r['mixture_predictive_candidates']);assert selected==r['selected_mixture_predictive_candidate']
    for name,b in priors.items():check_belief('initial_'+name,b,r['priors'][name])
    allstates=[];allcommands=[];paths=set();summaries={};goalsummaries={};unique_designs={};design_attached=0;method_keys=set();min_cov_eig=0.;zero_weights=0;future_mean_drift=[]
    for ei,e in enumerate(r['episodes']):
        name=e['prior'];mode=e['mode'];sequence=e['sequence'];branch=e['branch_evaluator_only'];repeat=e['repeat'];method=name+'_'+mode;method_keys.add(method)
        path=[20261020,16,640,r['meta']['replicate'],4,sequence*100+repeat];rng=np.random.default_rng(np.random.SeedSequence(path));noise=rng.normal(0,SD,(6,2,2));directions=rng.integers(0,8,6);paths.add(tuple(path));check('episode_seed_paths',path,e['seed_path'],1e-15)
        states=np.array(e['states_evaluator_only']);commands=np.array(e['commands']);observed=np.array(e['observed']);assert states.shape==(19,4) and commands.shape==(18,2) and observed.shape==(12,2)
        check('initial_physical_preparation',states[0],np.r_[mathcheck.prepare(branch),[0.,0.]],1e-15);assert abs(commands).max()<=BOUND+1e-9
        belief=tuple(x.copy() for x in priors[name]);costs=[];success=[]
        for phase,p in enumerate(e['phases']):
            goal=SEQUENCES[sequence,phase];check('current_goal',goal,p['goal'],1e-15);check_belief('belief_before',belief,p['belief_before'])
            before=belief;design_record=p['design'];u=commands[3*phase:3*phase+3]
            if mode=='task':
                assert design_record is None;u0,_,_=plan(belief,goal,3);u0=u0[0]
            else:
                design_attached+=len(design_record['candidates']);key=('initial',name,sequence) if phase==0 else ('later',ei,phase)
                if key in unique_designs:assert design_record==unique_designs[key]
                else:unique_designs[key]=design_record
                task,_,_=plan(belief,goal,3);check('design_task_plan',task,design_record['cautious_plan'],1e-8)
                candidates=np.array(design_record['candidates']);assert candidates.shape==(22,2)
                uniform=np.array([[np.cos(i*np.pi/4),np.sin(i*np.pi/4)] for i in range(8)])
                check('design_uniform_grid',candidates[1:17],np.vstack([.05*uniform,.1*uniform]),1e-14);check('design_task_candidate',candidates[-1],task[0],1e-8)
                mean,second=moments(belief);C=second-np.outer(mean,mean);variation=C[4:6,4:6]+C[6:8,6:8];ev,V=np.linalg.eigh(variation);v=V[:,-1]
                expected=np.array([-.05*v,.05*v,-.1*v,.1*v]);alt=-expected
                assert min(np.max(abs(expected-candidates[17:21])),np.max(abs(alt-candidates[17:21])))<1e-8
                check('variation_eigenvalues',ev,design_record['coefficient_variation_eigenvalues'],1e-9)
                values=np.array(design_record['values']);assert np.isfinite(values).all() and design_record['selected']==int(np.argmin(values))
                random_values=[values[0],values[1:9].mean(),values[9:17].mean()];check('random_radius_scores',random_values,design_record['random_radius_values'],1e-12)
                assert [0.,.05,.1][int(np.argmin(random_values))]==design_record['random_radius']
                if mode=='dual':u0=candidates[design_record['selected']]
                else:angle=directions[phase]*np.pi/4;u0=design_record['random_radius']*np.array([np.cos(angle),np.sin(angle)])
            check('first_command',u[0],u0,1e-8);check('stored_first_command',u[0],p['first_action'],1e-14)
            o1=mathcheck.endpoint(states[3*phase+1,:2])+noise[phase,0];o3=mathcheck.endpoint(states[3*phase+3,:2])+noise[phase,1]
            check('first_observation_rng',o1,observed[2*phase],1e-15);check('terminal_observation_rng',o3,observed[2*phase+1],1e-15)
            belief=observe(predict(belief,u[0]),(o1-P0)/T**2);check_belief('belief_after_first',belief,p['belief_after_first'])
            continuation,value,pg=plan(belief,goal,2,float(u[0]@u[0]));check('continuation_action',continuation,u[1:],1e-8);check('model_expected_cost',value,p['model_expected_cost'],1e-7);check('projected_gradient',pg,p['continuation_pg'],1e-7)
            for a in u[1:]:belief=predict(belief,a)
            check_belief('belief_before_terminal',belief,p['belief_before_terminal_observation'])
            belief=observe(belief,(o3-P0)/T**2);check_belief('belief_after_terminal',belief,p['belief_after_terminal'])
            zero_weights+=int(np.sum(belief[0]==0));min_cov_eig=min(min_cov_eig,float(np.linalg.eigvalsh(belief[2]).min()))
            if phase<5:check_belief('belief_continuity_to_next_goal',belief,e['phases'][phase+1]['belief_before'])
            endpoint=mathcheck.endpoint(states[3*phase+3,:2]);v=mathcheck.jac(states[3*phase+3,:2])@states[3*phase+3,2:];error=endpoint-P0-goal;effort=np.sum(u*u)
            cost=(error@error+WV*T*T*(v@v))/L**2+WU*effort;ok=bool((abs(error)<=.0003).all() and (abs(v)<=.003).all())
            check('terminal_position_error',error,p['position_error'],1e-13);check('terminal_cartesian_velocity',v,p['cartesian_velocity'],1e-13);check('phase_effort',effort,p['effort'],1e-12);check('phase_cost',cost,p['cost'],1e-8);assert ok==p['precision_success']
            costs.append(cost);success.append(ok);goalsummaries.setdefault((method,phase),[]).append((cost,ok))
        summaries.setdefault(method,[]).append((float(np.mean(costs)),sum(success),costs,success));allstates.append(states);allcommands.append(commands)
    states=np.array(allstates);commands=np.array(allcommands);z=states[:,0].copy()
    for j in range(18):z=mathcheck.advance(z,commands[:,j]);check('all_Hamiltonian_trajectory_states',z,states[:,j+1],3e-10)
    radau=0
    for ei in range(0,len(states),7):
        for t in (0,8,17):check('independent_Radau_intervals',mathcheck.radau(states[ei,t],commands[ei,t]),states[ei,t+1],3e-10);radau+=1
    nominal={};witness_count=0
    for w in r['nominal_feasibility_witnesses']:
        b=w['branch_evaluator_only'];seq=w['sequence'];phase=w['phase'];key=(b,seq);expected=nominal.get(key,np.r_[mathcheck.prepare(b),[0.,0.]])
        check('nominal_witness_continuity',w['initial_state'],expected,1e-13);z=np.array(w['initial_state']);u=np.array(w['commands']);assert abs(u).max()<=BOUND+1e-9
        for action in u:z=mathcheck.radau(z,action);witness_count+=1
        check('nominal_witness_terminal_state',z,w['terminal_state'],3e-10);g=SEQUENCES[seq,phase];res=np.r_[(mathcheck.endpoint(z[:2])-P0-g)/L,np.sqrt(WV)*(mathcheck.jac(z[:2])@z[2:])/(L/T)]
        check('nominal_witness_residual',res,w['normalized_residual'],2e-8);assert np.linalg.norm(res)<1e-7;assert w['plant_flow_calls']==3*w['residual_calls']+3;nominal[key]=np.array(w['terminal_state'])
    n=len(r['episodes']);resources={'parent_acquisition_reused':True,'new_acquisition_intervals':0,'control_episodes':n,'control_intervals':n*18,'control_simulated_seconds':n*18*T,'scalar_sensor_readings':n*24,'unique_paired_noise_paths':len(paths),'unique_scalar_noise_draws':len(paths)*24,
        'nonlinear_witness_plant_flows':sum(w['plant_flow_calls'] for w in r['nominal_feasibility_witnesses']),'candidate_records_attached_to_episodes':design_attached,'computed_designs':len(unique_designs),'evaluated_action_candidates':sum(len(d['candidates']) for d in unique_designs.values()),'integration_points':sum(i['integrand_points'] for d in unique_designs.values() for i in d['integration'])}
    for k,v in resources.items():check('resources_'+k,v,r['resources'][k],1e-9)
    assert n==64 and len(method_keys)==8 and len(paths)==4
    results={}
    for k,records in summaries.items():
        results[k]={'episodes':len(records),'phase_endpoints':6*len(records),'mean_cost':float(np.mean([x[0] for x in records])),'precision_endpoints':sum(x[1] for x in records),
            'episodes_all_six_precision':sum(x[1]==6 for x in records),'phase_mean_costs':[float(np.mean([x[2][i] for x in records])) for i in range(6)],'phase_precision_counts':[sum(x[3][i] for x in records) for i in range(6)]}
    artifacts=[source,design,rp,parentpath,dp,ROOT/'cartesian_reuse_v16_refinement.py',ROOT/'audit_cartesian_reuse_v16_pilot.py',ROOT/'audit_cartesian_reuse_v16_adaptive_results.json']
    out={'status':'INDEPENDENT FULL PHYSICAL/FILTER/POLICY REPLAY PASSED; FIRST-ACTION INTEGRALS AUDITED SEPARATELY','artifact_sha256':{p.name:sha(p) for p in artifacts},'audit_source_sha256':sha(Path(__file__)),
        'maximum_absolute_reconstruction_errors':ERRORS,'selected_predictive_mixture_candidate':selected,'control_summaries':results,'resources_reconstructed':resources,'minimum_component_covariance_eigenvalue':min_cov_eig,'terminal_zero_numeric_component_weights_count':zero_weights,
        'independent_Radau_control_intervals':radau,'independent_Radau_nominal_witness_intervals':witness_count,
        'limitations':['No canonical implementation imported; independent Hamiltonian physics, block conditioning, direct raw-second-moment terminal-map QP.',
            'Stored first-action menu values select the replayed first action; their integration accuracy is a separate theory audit, with preserved Genz-Malik failures.',
            'One parent fit, four paired noise paths, two fixed six-goal sequences; not64 independent trained systems.',
            'No process/drift noise, local constant-B approximation, exact initial Cartesian preparation/rest, known sensing/actuator timing and bounded torques supplied.',
            'Gaussian-component coefficient curvature is plug-in local parameter uncertainty, not an exact empirical-Bayes posterior.',
            'Nominal nonlinear witnesses certify their own trajectories, not every controller recovery state.'],
        'wall_seconds':time.time()-started}
    (ROOT/'audit_cartesian_feedback_v16_pilot_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'seconds':out['wall_seconds'],'errors':ERRORS,'summaries':results,'resources':resources}))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
