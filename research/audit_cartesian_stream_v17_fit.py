#!/usr/bin/env python3
"""Independent saved-fit replay. No canonical learning/local/plant imports.

All features, NumPy NN evaluations, coherent paths, local QR fits and analytic
path derivatives are reconstructed here. Torch is used only to reconstruct the
declared independent initialization seeds/first losses, not saved predictions.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def context(o, a):
    slopes = (o[-1] - o[-1-np.arange(1, 17)]) / (.1*np.arange(1, 17)[:, None])
    return np.r_[o[-1], slopes.ravel(), a[-16:].ravel()]


def rows(o, a, indices):
    xx=[]; uu=[]; yy=[]; origins=[]
    for i in indices:
        for t in range(16, 256, 2):
            xx.append(context(o[i,:t+1], a[i,:t])); uu.append(2*a[i,t])
            yy.append(np.concatenate([o[i,t+k]-o[i,t]-k*(o[i,t]-o[i,t-1]) for k in [1,2]]))
            origins.append([i,t])
    return np.array(xx),np.array(uu),np.array(yy),np.array(origins)


def network(x, w, jacobian=False):
    z1=np.tanh(w[0]@x+w[1]); z2=np.tanh(w[2]@z1+w[3]); y=w[4]@z2+w[5]
    if not jacobian:return y
    j=(w[4]*(1-z2*z2)[None,:])@((w[2]*(1-z1*z1)[None,:])@w[0])
    return y,j


def local_coefficients(X,U,Y,query,k,ridge):
    d=X-query
    distances=(np.sum(d[:,:2]**2,axis=1)/2 + np.sum(d[:,2:34]**2,axis=1)/32 + np.sum(d[:,34:]**2,axis=1)/32)
    ids=np.argsort(distances,kind='stable')[:min(k,len(X))]
    weights=np.exp(-distances[ids]/max(distances[ids[-1]],1e-12)); weights/=weights.sum()
    design=np.c_[np.ones(len(ids)),d[ids],U[ids]]
    # Independent residual-augmentation QR/SVD least squares, not normal solve.
    penalty=np.c_[np.zeros(68),np.eye(68)]*np.sqrt(ridge)
    augmented=np.r_[np.sqrt(weights)[:,None]*design,penalty]
    targets=np.r_[np.sqrt(weights)[:,None]*Y[ids],np.zeros((68,4))]
    coefficients=np.linalg.lstsq(augmented,targets,rcond=None)[0]
    stationarity=design.T@(weights[:,None]*(design@coefficients-Y[ids]))
    stationarity[1:]+=ridge*coefficients[1:]
    return coefficients,float(np.max(abs(stationarity)))


def paths(o,a,plan,scale,models,kind,derivative=False):
    q=(context(o,a)-scale['context_mean'])/scale['context_std']; outputs=[]; derivs=[]
    variables=2*len(plan)
    for model in models:
        hist=np.array(o[-17:],copy=True); commands=np.array(a[-16:],copy=True)
        hj=np.zeros((17,2,variables)); aj=np.zeros((16,2,variables)); future=[]; fj=[]
        for step,u in enumerate(plan):
            c=(context(hist,commands)-scale['context_mean'])/scale['context_std']
            cj=np.concatenate([hj[-1],np.concatenate([(hj[-1]-hj[-k-1])/(.1*k) for k in range(1,17)]),aj.reshape(32,variables)]) / scale['context_std'][:,None]
            uj=np.zeros((2,variables));uj[:,2*step:2*step+2]=np.eye(2)
            if kind=='nonlinear':
                y,j=network(np.r_[c,u*2],model,True); yj=j@np.r_[cj,2*uj]
            else:
                y=np.r_[1,c-q,u*2]@model; yj=model[1:67].T@cj+model[67:].T@(2*uj)
            correction=(scale['target_mean']+scale['target_std']*y).reshape(2,2)
            cy=(scale['target_std'][:,None]*yj).reshape(2,2,variables)
            nxt=np.array([hist[-1]+k*(hist[-1]-hist[-2]) for k in [1,2]])+correction
            nj=np.array([hj[-1]+k*(hj[-1]-hj[-2]) for k in [1,2]])+cy
            future.extend(nxt);fj.extend(nj)
            hist=np.r_[hist,nxt][-17:];hj=np.r_[hj,nj][-17:]
            commands=np.r_[commands,np.repeat(u[None],2,axis=0)][-16:]
            aj=np.r_[aj,np.repeat(uj[None],2,axis=0)][-16:]
        outputs.append(future);derivs.append(fj)
    return (np.array(outputs),np.array(derivs)) if derivative else np.array(outputs)


def main():
    torch.set_num_threads(1)
    prefix=ROOT/'cartesian_stream_v17_rep0'
    fitpath=Path(str(prefix)+'_fit.json');weightpath=Path(str(prefix)+'_weights.npz');ordinarypath=Path(str(prefix)+'_ordinary.npz')
    saved=json.loads(fitpath.read_text());data=np.load(ordinarypath,allow_pickle=False);weights=np.load(weightpath,allow_pickle=False)
    for key,p in [('source_sha256',ROOT/'cartesian_stream_v17_learn.py'),('local_source_sha256',ROOT/'cartesian_stream_v17_local.py'),('design_sha256',ROOT/'cartesian_stream_v17_learning_design.md'),('ordinary_sha256',ordinarypath),('weights_sha256',weightpath)]:assert saved[key]==sha(p),(key,'hash mismatch')
    assert set(data.files)=={'observed','commands','split','trajectory_index','dt','sensor_sd'}
    o=data['observed'];a=data['commands'];train=np.flatnonzero(data['split']==0);val=np.flatnonzero(data['split']==1);test=np.flatnonzero(data['split']==2)
    X,U,Y,origins=rows(o,a,train);assert np.array_equal(origins,saved['training_origins'])
    scale=dict(context_mean=X.mean(0),context_std=np.maximum(X.std(0),.01),target_mean=Y.mean(0),target_std=np.maximum(Y.std(0),.001))
    scale_errors={k:float(np.max(abs(v-np.array(saved['scale'][k])))) for k,v in scale.items()};assert max(scale_errors.values())<1e-13
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    bootstrap=[]
    for member in range(3):
        path=[20261020,17,660,0,0,member];ids=np.random.default_rng(np.random.SeedSequence(path)).choice(train,len(train),replace=True)
        b=np.concatenate([np.flatnonzero(origins[:,0]==i) for i in ids]);bootstrap.append(b)
        assert np.array_equal(ids,saved['bootstrap_episode_ids'][member])
        assert np.array_equal(b,saved['bootstrap_rows'][member]) and path==saved['bootstrap_seed_paths'][member]
    nns=[];training=[]
    for ci,(width,decay) in enumerate([(w,d) for w in [64,128] for d in [1e-4,.01]]):
        record=saved['nonlinear_candidates'][ci];assert (record['width'],record['weight_decay'])==(width,decay)
        members=[]
        for member,b in enumerate(bootstrap):
            keys=[f'nn{ci}_member{member}_layers.{layer}.{kind}' for layer in [0,2,4] for kind in ['weight','bias']]
            state=[weights[k] for k in keys];assert all(np.isfinite(v).all() for v in state);members.append(state)
            path=[20261020,17,660,0,1,ci,member];seed=int(np.random.default_rng(np.random.SeedSequence(path)).integers(0,2**31))
            member_record=record['members'][member];assert path==member_record['initialization_seed_path'] and seed==member_record['torch_seed']
            losses=np.array(member_record['normalized_training_loss_before_each_update']);assert losses.shape==(300,) and np.isfinite(losses).all()
            # Independent initialization constructor; no canonical Network import.
            torch.manual_seed(seed)
            initial=torch.nn.Sequential(torch.nn.Linear(68,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,4)).double()
            with torch.no_grad(): first=float(torch.mean((initial(torch.tensor(np.c_[X[b],U[b]]))-torch.tensor(Y[b]))**2))
            first_error=abs(first-losses[0]);assert first_error<1e-12
            predictions=np.array([network(np.r_[x,u],state) for x,u in zip(X[b],U[b])])
            final_loss=float(np.mean((predictions-Y[b])**2))
            training.append(dict(candidate=ci,member=member,initial_loss_error=first_error,saved_before_last_update=float(losses[-1]),reconstructed_final_weight_loss=final_loss,updates_logged=len(losses)))
        nns.append(members)
    assert len(weights.files)==72
    comparisons=[];max_stationarity=0.;local_models={}

    def get_models(kind,ci,i,t):
        nonlocal max_stationarity
        if kind=='nonlinear':return nns[ci]
        key=(ci,i,t)
        if key not in local_models:
            q=(context(o[i,:t+1],a[i,:t])-scale['context_mean'])/scale['context_std']
            k,ridge=[(k,r) for k in [128,512] for r in [1e-3,.1]][ci]
            models=[]
            for b in bootstrap:
                coef,res=local_coefficients(X[b],U[b],Y[b],q,k,ridge);max_stationarity=max(max_stationarity,res);models.append(coef)
            local_models[key]=models
        return local_models[key]

    def compare(kind,ci,ids,blocks,stride,reference,label):
        errors=[];emeans=[];max_abs=0.;max_relative=0.;expected=[];reconstructed_records=[]
        for i in ids:
            for t in range(16,256-2*blocks+1,stride):
                models=get_models(kind,ci,int(i),t)
                pred=paths(o[i,:t+1],a[i,:t],a[i,t:t+2*blocks:2],scale,models,kind)
                target=o[i,t+1:t+2*blocks+1]
                errors.append(float(np.mean((pred-target)**2)));emeans.append(float(np.mean((pred.mean(0)-target)**2)))
                reconstructed_records.append(np.r_[errors[-1],emeans[-1],np.mean((pred-target)**2,axis=(0,2))]);expected.append((int(i),t))
        assert len(expected)==reference['origins']
        for index,(r,pair,new) in enumerate(zip(reference['records'],expected,reconstructed_records)):
            assert pair==(r['trajectory'],r['time_index'])
            old=np.r_[r['member_coordinate_mse'],r['ensemble_mean_coordinate_mse'],r['member_mse_by_future_sample']]
            max_abs=max(max_abs,float(np.max(abs(new-old))));max_relative=max(max_relative,float(np.max(abs(new-old)/np.maximum(abs(old),1e-12))))
            assert np.allclose(new,old,rtol=2e-8,atol=2e-11),(kind,ci,label,index,new,old)
        score=float(np.mean(errors));mean_score=float(np.mean(emeans))
        assert np.allclose([score,mean_score],[reference['member_coordinate_mse'],reference['ensemble_mean_coordinate_mse']],rtol=2e-8,atol=2e-11)
        comparisons.append(dict(family=kind,candidate=ci,label=label,origins=len(expected),member_coordinate_mse=score,ensemble_mean_coordinate_mse=mean_score,max_record_absolute_error=max_abs,max_record_relative_error=max_relative))
        print(kind,ci,label,score,'max error',max_abs,flush=True)
        return score

    with threadpool_limits(limits=1):
        scores={}
        for kind,key in [('nonlinear','nonlinear_candidates'),('local','local_candidates')]:
            scores[kind]=[compare(kind,ci,val,6,24,r['validation'],'validation_six_block') for ci,r in enumerate(saved[key])]
            selected=int(np.argmin(scores[kind]));assert selected==saved['selected'][kind]
            compare(kind,selected,test,6,24,saved['test_observed_scores'][kind]['six_block'],'test_six_block')
            compare(kind,selected,test,1,2,saved['test_observed_scores'][kind]['one_block'],'test_one_block')
        gradient_checks={}
        # Actual selected weights and local QR fits, with a heldout real prefix.
        i=int(val[0]);t=40;plan=a[i,t:t+12:2].copy()
        for kind in ['nonlinear','local']:
            models=get_models(kind,saved['selected'][kind],i,t)
            prediction,jac=paths(o[i,:t+1],a[i,:t],plan,scale,models,kind,True)
            per_eps=[]
            for eps in [1e-5,1e-6]:
                fd=np.zeros_like(jac)
                for d in range(12):
                    plus=plan.copy();minus=plan.copy();plus.flat[d]+=eps;minus.flat[d]-=eps
                    fd[:,:,:,d]=(paths(o[i,:t+1],a[i,:t],plus,scale,models,kind)-paths(o[i,:t+1],a[i,:t],minus,scale,models,kind))/(2*eps)
                error=float(np.max(abs(fd-jac)));relative=error/max(1.,float(np.max(abs(jac))))
                assert relative<2e-7,(kind,eps,error)
                per_eps.append(dict(epsilon=eps,max_path_jacobian_error=error,scaled_max_error=relative))
            gradient_checks[kind]=dict(trajectory=i,time_index=t,max_absolute_path_jacobian=float(np.max(abs(jac))),checks=per_eps)
    result=dict(status='PASS INDEPENDENT ACTUAL SAVED-FIT REPLAY; NO CONTROL CLAIM',
        artifact_sha256={p.name:sha(p) for p in [ROOT/'cartesian_stream_v17_learn.py',ROOT/'cartesian_stream_v17_local.py',ROOT/'cartesian_stream_v17_learning_design.md',ordinarypath,fitpath,weightpath]},
        scale_max_errors=scale_errors,training_rows=len(X),shared_complete_trajectory_bootstraps_reconstructed=True,
        training_seed_first_losses_and_final_weights=training,comparisons=comparisons,selected=saved['selected'],
        independent_local_augmented_lstsq_stationarity_max=max_stationarity,unique_three_member_local_fits=len(local_models),actual_selected_path_derivatives=gradient_checks,
        limitations='Middle Adam histories are retained and finite but not independently replayed epoch by epoch; first seeded losses and final saved-weight losses are reconstructed. Selection/scores all replayed from saved weights and independent local QR/SVD equations. No plant/evaluator import or data; no fork/control outcome audited here.')
    (ROOT/'audit_cartesian_stream_v17_fit_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
