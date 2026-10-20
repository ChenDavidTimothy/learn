#!/usr/bin/env python3
"""Independent per-corpus acquisition and persistent-control artifact audit.

No canonical implementation imports. Physics/linear algebra helpers are earlier
independent audit code. Completed per-corpus results can be reused by hash.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as local
import audit_cartesian_feedback_v16_pilot as joint
from audit_cartesian_reuse_v16_adaptive import likelihood_and_gradient

ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-8):
    e=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),e);assert e<tol,(name,e,tol)
def source_for(h):
    found=[p for p in ROOT.glob('cartesian_feedback_v16_study*.py') if sha(p)==h]
    assert found,('missing frozen source',h)
    return found[0]

def acquisition(rep):
    ERROR.clear();rp=ROOT/f'cartesian_feedback_v16_rep{rep}_acquisition_results.json';dp=rp.with_name(rp.name.replace('_results.json','_data.npz'));r=json.loads(rp.read_text());raw=np.load(dp)
    source=source_for(r['meta']['source_sha256']);assert sha(dp)==r['meta']['data_sha256'];assert r['meta']['replicate']==rep
    assert sha(ROOT/'cartesian_reuse_v16_gk21.py')==r['meta']['base_source_sha256'];data={};seedpaths=[]
    for name,role,n in [('training',1,256),('validation',2,64),('testing',3,64)]:
        commands=[];initial=[];noise=[];paths=[]
        for i in range(n):
            path=[20261020,16,630,rep,role,i];rng=np.random.default_rng(np.random.SeedSequence(path));initial.append(np.r_[local.prepare(int(rng.integers(2))),[0.,0.]])
            commands.append(rng.uniform(-.5,.5,(2,2)));noise.append(rng.normal(0,.0002,(3,2)));paths.append(path)
        commands=np.array(commands);states=raw[name+'_states_evaluator_only'];ck('acquisition_commands',commands,raw[name+'_commands'],1e-15);ck('acquisition_paths',paths,raw[name+'_seed_paths'],1e-15);ck('acquisition_initial_states',initial,states[:,0],1e-15)
        observations=local.endpoint(states[...,:2])+np.array(noise);ck('acquisition_camera',observations,raw[name+'_observed'],1e-15)
        z=np.array(initial)
        for t in range(2):z=local.advance(z,commands[:,t]);ck('acquisition_Hamiltonian_physics',z,states[:,t+1],3e-10)
        X=np.stack([.5*commands[:,0],1.5*commands[:,0]+.5*commands[:,1]],1);Y=(observations[:,1:]-local.P0)/.01;ck('acquisition_exposures',X,raw[name+'_exposures'],1e-15);data[name]=(X,Y);seedpaths.extend(paths)
    assert len(set(map(tuple,seedpaths)))==384
    X,Y=data['training'];V,Z=data['validation'];H=local.group_H(X);HV=local.group_H(V);y=Y.reshape(-1,4);z=Z.reshape(-1,4);validation=[]
    for row in r['mixture_candidates']:
        k=row['components'];path=[20261020,16,630,rep,5,k*100+row['start']];ck('EM_init_paths',path,row['seed_path'],1e-15);rng=np.random.default_rng(np.random.SeedSequence(path));mean=local.mean_fit(X,Y)
        scale=np.sqrt(np.mean((Y-X@mean.T)**2)/np.mean(X**2));B=mean[None]+rng.normal(0,max(.1,scale),(k,2,2));w=np.ones(k)/k
        if k==1:B[0]=mean
        history=[];converged=False
        for iteration in range(150):
            parts=local.mix_logs(H,y,B,w);norm=logsumexp(parts,axis=1);history.append(float(norm.mean()));weights=np.exp(parts-norm[:,None]);w=np.maximum(weights.mean(0),1e-6);w/=w.sum();B=np.array([local.mean_fit(X,Y,weights[:,i]) for i in range(k)])
            if len(history)>1 and abs(history[-1]-history[-2])<1e-9*(1+abs(history[-2])):converged=True;break
        val=-logsumexp(local.mix_logs(HV,z,B,w),axis=1).mean();validation.append(float(val));ck('EM_matrices',B,row['B'],1e-6);ck('EM_weights',w,row['weights'],1e-7);ck('EM_history',history,row['training_mean_log_likelihood'],1e-5);ck('EM_validation',val,row['validation_nll'],1e-6)
        assert len(history)==row['iterations'] and converged==row['converged']
    assert int(np.argmin(validation))==r['selected_mixture']
    mean=local.mean_fit(X,Y);res=y-H@mean.ravel();basis=[]
    for i,j in zip(*np.triu_indices(4)):
        E=np.zeros((4,4));E[i,j]=E[j,i]=1;basis.append(E)
    op=np.stack([(H@E@H.transpose(0,2,1)).ravel() for E in basis],1);target=(res[:,:,None]*res[:,None,:]-.0004*np.eye(4)).ravel();theta=np.linalg.lstsq(op,target,rcond=None)[0];rawC=np.einsum('k,kij->ij',theta,basis);e,Q=np.linalg.eigh(rawC)
    for row in r['gaussian_candidates'][:3]:
        C=(Q*np.maximum(e,row['covariance_floor']))@Q.T;ck('Gaussian_moment_mean',mean,row['mean'],1e-10);ck('Gaussian_raw_covariance',rawC,row['raw_covariance'],1e-8);ck('Gaussian_projected_covariance',C,row['cov'],1e-8);val=-local.gaussian_logs(HV,z,mean,C).mean();ck('Gaussian_moment_validation',val,row['validation_nll'],1e-6)
    gradients=[]
    for row in r['gaussian_candidates'][3:]:
        m=np.array(row['mean']);C=np.array(row['cov']);e,Q=np.linalg.eigh(C);F=Q*np.sqrt(np.maximum(e,0));p=np.r_[m.ravel(),F.ravel()];value,gradient=likelihood_and_gradient(p,H,y);ck('Gaussian_ML_training',value,row['training_nll'],1e-7);ck('Gaussian_ML_validation',-local.gaussian_logs(HV,z,m,C).mean(),row['validation_nll'],1e-6)
        fd=[]
        for j in range(20):
            delta=np.zeros(20);delta[j]=2e-7;fd.append((likelihood_and_gradient(p+delta,H,y)[0]-likelihood_and_gradient(p-delta,H,y)[0])/4e-7)
        err=float(np.max(abs(np.array(fd)-gradient)));assert err<5e-5
        gradients.append({'start':row['start'],'max_coordinate_FD_error':err,'reported_success':row['success'],'reported_message':row['message'],'training_nll':value})
    assert int(np.argmin([row['validation_nll'] for row in r['gaussian_candidates']]))==r['selected_gaussian']
    artifacts=[rp,dp,source,ROOT/'cartesian_reuse_v16_gk21.py']
    out={'status':'INDEPENDENT FRESH ACQUISITION/FITS RECONSTRUCTED','replicate':rep,'artifact_sha256':{p.name:sha(p) for p in artifacts},'audit_source_sha256':sha(Path(__file__)),
        'errors':dict(ERROR),'Gaussian_ML_endpoint_checks':gradients,'resources':{'new_groups':384,'new_intervals':768,'scalar_camera_reads':2304},'unique_acquisition_paths':384}
    (ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_acquisition_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');return out

def episode_audit(rep,acq):
    ERROR.clear();joint.ERRORS.clear();rp=ROOT/f'cartesian_feedback_v16_rep{rep}_results.json';r=json.loads(rp.read_text());assert len(r['episodes'])==80 and 'resources' in r,'corpus not complete'
    source=source_for(r['meta']['source_sha256']);ap=ROOT/r['meta']['parent_results'];parent=json.loads(ap.read_text());dp=ap.with_name(ap.name.replace('_results.json','_data.npz'));raw=np.load(dp)
    assert sha(ap)==r['meta']['parent_sha256'] and sha(dp)==r['meta']['data_sha256'];assert sha(ROOT/'cartesian_reuse_v16_gk21.py')==r['meta']['base_source_sha256'];assert parent['meta']['replicate']==rep
    for name,h in acq['artifact_sha256'].items():assert sha(ROOT/name)==h
    priors,selected=joint.reconstruct_priors(parent,raw,r['mixture_predictive_candidates']);assert selected==r['selected_mixture_predictive_candidate'];w,m,C=priors['gaussian'];isotropic=C.copy();isotropic[0,4:,4:]=np.trace(C[0,4:,4:])/4*np.eye(4);priors['isotropic']=(w.copy(),m.copy(),isotropic)
    for name,p in priors.items():joint.check_belief('initial_'+name,p,r['priors'][name])
    X=raw['testing_exposures'];Y=(raw['testing_observed'][:,1:]-local.P0)/.01
    for name,(w,m,C) in priors.items():
        H=local.group_H(X);h=local.group_H(X[:,:1]);jointval=joint.mixture_predictive_nll(H,Y.reshape(-1,4),m[:,4:].reshape(-1,2,2),w,C[:,4:,4:]);first=joint.mixture_predictive_nll(h,Y[:,:1].reshape(-1,2),m[:,4:].reshape(-1,2,2),w,C[:,4:,4:])
        for k,value in [('joint_group_nll',jointval),('first_observation_nll',first),('second_given_first_nll',jointval-first)]:ck('heldout_'+k,value,r['heldout_forecasts'][name][k],1e-7)
    summaries={};allstates=[];allcommands=[];unique={};attached=0;paths=set();strict_records=0
    for ei,e in enumerate(r['episodes']):
        name=e['prior'];mode=e['mode'];sequence=e['sequence'];branch=e['branch_evaluator_only'];path=[20261020,16,640,rep,4,sequence*100+e['repeat']];rng=np.random.default_rng(np.random.SeedSequence(path));noise=rng.normal(0,.0002,(6,2,2));directions=rng.integers(0,16,6);paths.add(tuple(path));ck('feedback_seed',path,e['seed_path'],1e-15)
        z=np.array(e['states_evaluator_only']);u=np.array(e['commands']);o=np.array(e['observed']);assert z.shape==(19,4) and u.shape==(18,2) and o.shape==(12,2);ck('initial_rest_preparation',z[0],np.r_[local.prepare(branch),[0.,0.]],1e-15);assert abs(u).max()<=.5+1e-9
        belief=tuple(a.copy() for a in priors[name]);costs=[];success=[];efforts=[];positions=[];velocities=[]
        for phase,p in enumerate(e['phases']):
            g=joint.SEQUENCES[sequence,phase];ck('goals',g,p['goal'],1e-15);joint.check_belief('before',belief,p['belief_before']);cmd=u[3*phase:3*phase+3];d=p['design']
            if mode=='task':assert d is None;first=joint.plan(belief,g,3)[0][0]
            else:
                attached+=len(d['candidates']);key=('initial',name,sequence) if phase==0 else ('later',ei,phase)
                if key in unique:assert unique[key]==d
                else:unique[key]=d
                candidates=np.array(d['candidates']);assert candidates.shape==(34,2);dirs=np.array([[np.cos(i*np.pi/8),np.sin(i*np.pi/8)] for i in range(16)]);ck('common_uniform_menu',candidates[1:33],np.vstack([.05*dirs,.1*dirs]),1e-14)
                task=joint.plan(belief,g,3)[0];ck('cautious_menu',task,d['cautious_plan'],1e-8);ck('cautious_candidate',candidates[-1],task[0],1e-8)
                values=np.array(d['values']);assert d['selected']==int(np.argmin(values));assert len(d['tighter_top_three'])==3;strict_records+=3
                for record in d['tighter_top_three']:
                    idx=record['index'];ck('strict_value_retained',values[idx],record['value'],1e-15)
                    assert abs(record['original_value']-record['value'])<=max(2e-7,5*(record['integration']['estimated_error']+d['integration'][idx]['estimated_error']))
                    assert record['integration']['rule']=='gk21' and record['integration']['initial_partition']=='four quadrants' and record['integration']['atol']==1e-10
                rv=[values[0],values[1:17].mean(),values[17:33].mean()];ck('random_radius_means',rv,d['random_radius_values'],1e-12);assert [0.,.05,.1][int(np.argmin(rv))]==d['random_radius']
                for info in d['integration']:assert info['rule']=='gk21' and info['initial_partition']=='four quadrants' and info['atol']==1e-8
                if mode=='dual':first=candidates[d['selected']]
                else:a=directions[phase]*np.pi/8;first=d['random_radius']*np.array([np.cos(a),np.sin(a)])
            ck('first_action',cmd[0],first,1e-8);ck('record_first_action',cmd[0],p['first_action'],1e-14)
            y1=local.endpoint(z[3*phase+1,:2])+noise[phase,0];y3=local.endpoint(z[3*phase+3,:2])+noise[phase,1];ck('first_camera',y1,o[2*phase],1e-15);ck('terminal_camera',y3,o[2*phase+1],1e-15)
            belief=joint.observe(joint.predict(belief,cmd[0]),(y1-local.P0)/.01);joint.check_belief('after_first',belief,p['belief_after_first']);last,value,pg=joint.plan(belief,g,2,float(cmd[0]@cmd[0]));ck('continuation',last,cmd[1:],1e-8);ck('expected_cost',value,p['model_expected_cost'],1e-7);ck('PG',pg,p['continuation_pg'],1e-7)
            for action in cmd[1:]:belief=joint.predict(belief,action)
            joint.check_belief('before_terminal',belief,p['belief_before_terminal_observation']);belief=joint.observe(belief,(y3-local.P0)/.01);joint.check_belief('after_terminal',belief,p['belief_after_terminal'])
            if phase<5:joint.check_belief('next_goal_continuity',belief,e['phases'][phase+1]['belief_before'])
            err=local.endpoint(z[3*phase+3,:2])-local.P0-g;v=local.jac(z[3*phase+3,:2])@z[3*phase+3,2:];effort=np.sum(cmd*cmd);cost=(err@err+.25*.01*(v@v))/1e-6+.01*effort;ok=bool((abs(err)<=.0003).all() and (abs(v)<=.003).all())
            ck('phase_error',err,p['position_error'],1e-13);ck('phase_velocity',v,p['cartesian_velocity'],1e-13);ck('phase_effort',effort,p['effort'],1e-12);ck('phase_cost',cost,p['cost'],1e-8);assert ok==p['precision_success'];costs.append(cost);success.append(ok);efforts.append(effort);positions.append(float(err@err));velocities.append(float(v@v))
        summaries.setdefault(name+'_'+mode,[]).append({'costs':costs,'success':success,'effort':efforts,'position_squared':positions,'velocity_squared':velocities,'sequence':sequence,'branch':branch,'repeat':e['repeat']});allstates.append(z);allcommands.append(u)
    z=np.array(allstates)[:,0].copy();states=np.array(allstates);commands=np.array(allcommands)
    for t in range(18):z=local.advance(z,commands[:,t]);ck('all_control_Hamiltonian_physics',z,states[:,t+1],3e-10)
    nominal={}
    for w in r['nominal_feasibility_witnesses']:
        key=(w['branch_evaluator_only'],w['sequence']);initial=nominal.get(key,np.r_[local.prepare(key[0]),[0.,0.]]);ck('nominal_continuity',initial,w['initial_state'],1e-13);z=np.array(initial)
        for a in np.array(w['commands']):z=local.radau(z,a)
        ck('nominal_Radau_terminal',z,w['terminal_state'],3e-10);g=joint.SEQUENCES[key[1],w['phase']];res=np.r_[(local.endpoint(z[:2])-local.P0-g)/.001,.5*(local.jac(z[:2])@z[2:])/.01];ck('nominal_residual',res,w['normalized_residual'],2e-8);assert np.linalg.norm(res)<1e-7;nominal[key]=np.array(w['terminal_state'])
    resources={'control_episodes':80,'control_intervals':1440,'control_simulated_seconds':144.,'scalar_sensor_readings':1920,'unique_paired_noise_paths':len(paths),'unique_scalar_noise_draws':len(paths)*24,'nonlinear_witness_plant_flows':sum(w['plant_flow_calls'] for w in r['nominal_feasibility_witnesses']),
        'candidate_records_attached_to_episodes':attached,'computed_designs':len(unique),'evaluated_action_candidates':sum(len(d['candidates']) for d in unique.values()),'tighter_candidate_reevaluations':sum(len(d['tighter_top_three']) for d in unique.values()),'integration_points':sum(info['integrand_points'] for d in unique.values() for info in d['integration'])+sum(info['integration']['integrand_points'] for d in unique.values() for info in d['tighter_top_three'])}
    for k,v in resources.items():ck('resource_'+k,v,r['resources'][k],1e-8)
    assert len(paths)==4 and len(summaries)==10
    summary={k:{'episodes':len(rows),'mean_cost':float(np.mean([a['costs'] for a in rows])),'precision_endpoints':sum(sum(a['success']) for a in rows),'all_six_precision_episodes':sum(all(a['success']) for a in rows),
        'phase_mean_costs':np.mean([a['costs'] for a in rows],axis=0).tolist(),'phase_precision_counts':np.sum([a['success'] for a in rows],axis=0).tolist(),'mean_effort':float(np.mean([a['effort'] for a in rows]))} for k,rows in summaries.items()}
    artifacts=[rp,source,ROOT/'cartesian_feedback_v16_study_design.md',ap,dp,ROOT/'cartesian_reuse_v16_gk21.py']
    return {'status':'INDEPENDENT COMPLETE CORPUS REPLAY PASSED; NUMERICAL INTEGRALS AUDITED SEPARATELY','replicate':rep,'artifact_sha256':{p.name:sha(p) for p in artifacts},'audit_source_sha256':sha(Path(__file__)),
        'errors':dict(ERROR),'joint_belief_errors':dict(joint.ERRORS),'summaries':summary,'individual_episode_metrics':summaries,'resources_reconstructed_control':resources,
        'correct_total_new_physical_intervals':2208,'new_acquisition_intervals_from_parent':768,'recorded_acquisition_reporting':{k:r['resources'].get(k) for k in ('parent_acquisition_reused','new_acquisition_intervals')},
        'cohort_reporting_erratum':rep in (100,101,102,103),'limitations':['One fixed body/prepared two-branch population and same prespecified goal sequences; corpora replicate fits, not new bodies.',
            'All first actions replayed from saved finite-menu scores; this artifact audit does not independently integrate those scores.',
            'Full source-independent Hamiltonian dynamics, Gaussian block filter and raw-second-moment terminal-map QPs.',
            'Old first4source reports control-phase resource/timing; combined costs corrected using separate acquisition artifacts.']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicates',nargs='+',type=int,default=[100]);parser.add_argument('--acquisition-only',action='store_true');args=parser.parse_args()
    for rep in args.replicates:
        start=time.time();cache=ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_acquisition_results.json';a=None
        if cache.exists():
            candidate=json.loads(cache.read_text())
            if candidate.get('audit_source_sha256')==sha(Path(__file__)) and all(sha(ROOT/n)==h for n,h in candidate['artifact_sha256'].items()):a=candidate
        if a is None:a=acquisition(rep)
        if args.acquisition_only:print('acquisition verified',rep,flush=True);continue
        completed=ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_results.json'
        if completed.exists():
            cached=json.loads(completed.read_text())
            if cached.get('audit_source_sha256')==sha(Path(__file__)) and all(sha(ROOT/n)==h for n,h in cached['artifact_sha256'].items()):print('full cached',rep,flush=True);continue
        result=episode_audit(rep,a);result['wall_seconds']=time.time()-start
        (ROOT/f'audit_cartesian_feedback_v16_study_rep{rep}_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('full verified',rep,'seconds',result['wall_seconds'],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
