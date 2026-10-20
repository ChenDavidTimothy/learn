#!/usr/bin/env python3
"""Compact prescribed relation-learning test; supplied rest/preparation context.
Not continuous feedback control, physical parameter discovery or model necessity.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear,least_squares
from threadpoolctl import threadpool_limits
from coupled_context_v14_pilot import flow,mass
from nonlinear_identification_v9 import stat

SEED=20261020+2200000
DURATION=.1; FACTOR=.5*DURATION**2; SD=1e-4; RESPONSE_VARIANCE=SD**2/FACTOR**2
WIDTHS=(.5,1.,2.,4.); RIDGES=(1e-6,1e-4,.01,.1)
ANGLES=np.linspace(-math.pi,math.pi,8,endpoint=False)
CENTERS=np.array([(a,b) for a in ANGLES for b in ANGLES])
CONTEXTS=np.array([(a,b) for a in (-2.7,-1.1,.4,2.2) for b in (-math.pi,-2.1,-1.2,-.4,0.,.4,1.2,2.1)])
GOALS=np.array(((.1,0.),(0.,.1),(.1,-.1),(-.1,.1),(.05,.1)))
LABELS=('constant_forward','context_forward','naive_inverse','corrected_inverse')


def features(q,width):
    q=np.atleast_2d(q)
    if width is None:return np.ones((len(q),1))
    distance=np.sum(4*np.sin((q[:,None,:]-CENTERS[None,:,:])/2)**2,axis=2)
    return np.c_[np.ones(len(q)),np.exp(-distance/(2*width*width))]


def experience(seed,count,narrow=False):
    rng=np.random.default_rng(seed);q=np.c_[rng.uniform(-math.pi,math.pi,count),rng.uniform(-.4 if narrow else -math.pi,.4 if narrow else math.pi,count)]
    u=rng.uniform(-1,1,(count,2));s=[]
    for qi,ui in zip(q,u):s.append(flow(qi,np.zeros(2),ui,DURATION)[0])
    s=np.array(s);o0=q+rng.normal(0,SD,(count,2));o1=s[:,:2]+rng.normal(0,SD,(count,2))
    return dict(known_preparation=q,commands=u,states_evaluator_only=s,observed_initial=o0,observed_final=o1,
                response=(o1-o0)/FACTOR)


def relation(model,q):
    phi=features(q,model['width'])[0];w=np.array(model['coefficients'])
    matrix=np.einsum('f,fio->oi',phi,w.reshape(len(phi),2,2))
    if 'inverse' in model['label']:
        return np.linalg.pinv(matrix,rcond=1e-8),matrix
    return matrix,None


def fit_all(data):
    train=slice(0,512);validation=slice(512,640);records={};selected={}
    q=data['known_preparation'];u=data['commands'];d=data['response']
    for label in LABELS:
        candidates=[];inverse='inverse' in label
        for width in ((None,) if label=='constant_forward' else WIDTHS):
            phi=features(q[train],width);phiv=features(q[validation],width)
            source=d[train] if inverse else u[train];target=u[train] if inverse else d[train]
            X=np.einsum('nf,ni->nfi',phi,source).reshape(512,-1);raw=X.T@X;G=raw.copy()
            if label=='corrected_inverse':G-=RESPONSE_VARIANCE*np.kron(phi.T@phi,np.eye(2))
            vals,vecs=np.linalg.eigh(G);scale=max(float(np.trace(raw)/len(raw)),1e-12);rhs=X.T@target
            for ridge in RIDGES:
                denominator=np.maximum(vals+ridge*scale,ridge*scale)
                W=vecs@((vecs.T@rhs)/denominator[:,None]);F=len(phi[0])
                tensor=W.reshape(F,2,2);maps=np.einsum('nf,fio->noi',phiv,tensor)
                valinput=d[validation] if inverse else u[validation]
                predicted=np.einsum('noi,ni->no',maps,valinput)
                valtarget=u[validation] if inverse else d[validation]
                valerr=np.sum((predicted-valtarget)**2,axis=1)
                if inverse:valerr-=RESPONSE_VARIANCE*np.sum(maps*maps,axis=(1,2))
                candidates.append(dict(label=label,width=width,ridge=ridge,regularization_scale=scale,
                    raw_gram_min_eigenvalue=float(np.linalg.eigvalsh(raw)[0]),corrected_gram_eigenvalues=vals.tolist(),
                    floored_eigenvalues=int(np.sum(vals<0)),coefficients=W.tolist(),validation_score=float(np.mean(valerr)),
                    training_normal_equation_residual=float(np.linalg.norm(G@W-rhs)),
                    numerical_finite=bool(np.all(np.isfinite(W)))))
        assert all(c['numerical_finite'] for c in candidates)
        best=int(np.argmin([c['validation_score'] for c in candidates]));records[label]=dict(candidates=candidates,selected_index=best);selected[label]=candidates[best]
    return selected,records


def command(B,z):
    fit=lsq_linear(B,z,bounds=(-1.,1.),tol=1e-12,max_iter=100)
    return fit.x,dict(success=bool(fit.success),cost=float(np.sum((B@fit.x-z)**2)),optimality=float(fit.optimality))


def oracle_tasks():
    rows=[];calls=0
    for q in CONTEXTS:
        for z in GOALS:
            def residual(u):
                nonlocal calls
                calls+=1;return (flow(q,np.zeros(2),u,DURATION)[0][:2]-q)/FACTOR-z
            fit=least_squares(residual,mass(q)@z,bounds=(-1.,1.),max_nfev=40,ftol=1e-12,xtol=1e-12,gtol=1e-10)
            error=residual(fit.x)
            assert fit.success and np.linalg.norm(error)<1e-7
            rows.append(dict(context=q.tolist(),goal=z.tolist(),command=fit.x.tolist(),success=bool(fit.success),
                feasibility_error_norm=float(np.linalg.norm(error)),nfev=int(fit.nfev)))
    return rows,calls


def evaluate(theta_models,predictive,seed,oracle):
    rng=np.random.default_rng(seed);tasks=[]
    for baseline in oracle:
        q=np.array(baseline['context']);z=np.array(baseline['goal']);controllers={}
        for label in (*LABELS,'oracle_endpoint'):
            if label=='oracle_endpoint':u=np.array(baseline['command']);info=None;condition=0.
            else:
                B,K=relation(theta_models[label],q);u,info=command(B,z)
                condition=float(np.linalg.cond(B))
            end,_=flow(q,np.zeros(2),u,DURATION);response=(end[:2]-q)/FACTOR
            e0=rng.normal(0,SD,2);e1=rng.normal(0,SD,2)
            controllers[label]=dict(command=u.tolist(),state_evaluator_only=end.tolist(),observed_initial=(q+e0).tolist(),observed_final=(end[:2]+e1).tolist(),
                squared_error=float(np.sum((response-z)**2)),response_evaluator_only=response.tolist(),
                saturated=bool(np.any(np.abs(u)>.999999)),response_condition=condition,inverse_relation_singular_values=(np.linalg.svd(K,compute_uv=False).tolist() if label!='oracle_endpoint' and K is not None else None),bounded_solver=info)
        tasks.append(dict(known_preparation=q.tolist(),desired_normalized_displacement=z.tolist(),controllers=controllers))
    predictions={}
    truth=(predictive['states_evaluator_only'][:,:2]-predictive['known_preparation'])/FACTOR
    for label in LABELS:
        predicted=[];conditions=[]
        for q,u in zip(predictive['known_preparation'],predictive['commands']):
            B,_=relation(theta_models[label],q);predicted.append(B@u);conditions.append(float(np.linalg.cond(B)))
        error=np.sum((np.array(predicted)-truth)**2,axis=1)
        predictions[label]=dict(mean_squared_error=float(np.mean(error)),errors=error.tolist(),predictions=np.array(predicted).tolist(),response_conditions=conditions)
    return tasks,predictions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-prefix',default='research/coupled_learning_v14');a=parser.parse_args()
    rows=[];arrays={};timing=[]
    parent_json=Path('research/coupled_learning_v14_two_reading_pilot_results.json')
    parent_npz=Path('research/coupled_learning_v14_two_reading_pilot_data.npz')
    original=json.loads(parent_json.read_text());archive=np.load(parent_npz)
    def existing(condition,replicate,kind):
        prefix=f'c{condition}_s{replicate}_{kind}_'
        dataset={key[len(prefix):]:archive[key].copy() for key in archive.files if key.startswith(prefix)}
        # Preparation position is EXACT supplied information. Reuse identical raw
        # records, remove the avoidable initial sensor noise from every estimator.
        dataset['response']=(dataset['observed_final']-dataset['known_preparation'])/FACTOR
        return dataset
    with threadpool_limits(limits=1):
        oracle=original['oracle_feasibility'];oracle_calls=0
        for condition in range(2):
            for replicate in range(8):
                seed=SEED+1000000000*condition+1000000*replicate;t0=time.perf_counter()
                data=existing(condition,replicate,'train');models,records=fit_all(data)
                predictive=existing(condition,replicate,'predictive');tasks,predictions=evaluate(models,predictive,seed+500000,oracle)
                key=f'c{condition}_s{replicate}';arrays.update({key+'_train_'+k:v for k,v in data.items()});arrays.update({key+'_predictive_'+k:v for k,v in predictive.items()})
                rows.append(dict(condition=condition,replicate=replicate,seed=seed,models=models,fit_records=records,tasks=tasks,predictive_transfer=predictions))
                timing.append(dict(condition=condition,replicate=replicate,seconds=time.perf_counter()-t0))
                print(f'completed contextcondition{condition} seed{replicate}',file=sys.stderr,flush=True)
    summary=[]
    for c in range(2):
        subset=[r for r in rows if r['condition']==c];models={}
        for label in (*LABELS,'oracle_endpoint'):
            losses=[np.mean([t['controllers'][label]['squared_error'] for t in r['tasks']]) for r in subset]
            models[label]=dict(task_mse=stat(losses),saturations=sum(t['controllers'][label]['saturated'] for r in subset for t in r['tasks']),trials=len(subset)*len(oracle))
            if label in LABELS:models[label]['predictive_mse']=stat([r['predictive_transfer'][label]['mean_squared_error'] for r in subset])
        paired={}
        for label in ('constant_forward','naive_inverse','corrected_inverse'):
            ds=[np.mean([t['controllers'][label]['squared_error']-t['controllers']['context_forward']['squared_error'] for t in r['tasks']]) for r in subset]
            paired[label+'_minus_context_forward']=stat(ds)
        summary.append(dict(condition=c,models=models,paired=paired))
    prefix=a.output_prefix;np.savez_compressed(prefix+'_data.npz',**arrays)
    meta=dict(status='EXPLORATORY AVAILABLE-INFORMATION CORRECTION; audit pending',design='coupled_learning_v14_design.md',information_correction='coupled_learning_v14_information_correction.md',seed=SEED,
        duration=DURATION,sensor_sd=SD,response_noise_variance=RESPONSE_VARIANCE,training_steps_per_seed=512,validation_steps_per_seed=128,
        acquisition_pulse_episodes=16*640,predictive_test_episodes=16*128,task_test_episodes=16*len(oracle)*5,
        oracle_feasibility_solver_rollouts=oracle_calls,reused_original_oracle_rollouts=1456,oracle_targets=len(oracle),
        new_acquisition_pulses=0,new_predictive_pulses=0,
        original_result_sha256=hashlib.sha256(parent_json.read_bytes()).hexdigest(),original_data_sha256=hashlib.sha256(parent_npz.read_bytes()).hexdigest(),maximum_oracle_feasibility_error=max(r['feasibility_error_norm'] for r in oracle),
        exact_preparation_context_is_supplied=True,periodic_topology_and_torque_affinity_are_supplied=True,
        no_continuous_feedback_or_stationkeeping=True,inverse_pseudoinverse_rcond=1e-8,data_sha256=hashlib.sha256(Path(prefix+'_data.npz').read_bytes()).hexdigest())
    Path(prefix+'_results.json').write_text(json.dumps(dict(meta=meta,oracle_feasibility=oracle,runs=rows,summary=summary),indent=2,allow_nan=False)+'\n')
    Path(prefix+'_timing.json').write_text(json.dumps(timing,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
