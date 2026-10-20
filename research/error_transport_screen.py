#!/usr/bin/env python3
"""Exploratory mechanism screen, train/validation selection before test scoring.

No physical model or hidden parameter is imported. Separate fresh-body validation
chooses global learning rates/checkpoints and ordinary adaptation LR. Raw test
predictions, wrong-system-support ablation and per-body errors are preserved.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from error_transport_model import ErrorTransport,SeparableTransport,Pooled
from error_transport_baselines import ALPaCA,CoDA

ROOT=Path(__file__).resolve().parent
METHODS=('joint','separable','alpaca','coda','pooled')
BUDGETS=(0,4,16,64)
DATA=ROOT/'error_transport_screen_data.npz'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make(method):
    return dict(joint=ErrorTransport,separable=SeparableTransport,alpaca=ALPaCA,coda=CoDA,pooled=Pooled)[method]()
def synchronize():
    if torch.cuda.is_available():torch.cuda.synchronize()
def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(device):
    with np.load(DATA,allow_pickle=False) as d:x=d['x'];y=d['y'];split=d['split']
    tr=split==0
    xm=x[tr].mean((0,1,2));xs=np.maximum(x[tr].std((0,1,2)),.01)
    ym=y[tr].mean((0,1,2));ys=np.maximum(y[tr].std((0,1,2)),.01)
    scale=dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
    x=torch.as_tensor((x-xm)/xs,dtype=torch.float32,device=device)
    y=torch.as_tensor((y-ym)/ys,dtype=torch.float32,device=device)
    return x,y,split,scale


def fixed_samples(x,y,ids,n,shuffle=False):
    ids=torch.as_tensor(ids,device=x.device)
    supports=ids.roll(1) if shuffle else ids
    return x[supports,0,:n],y[supports,0,:n],x[ids,1:].flatten(1,2),y[ids,1:].flatten(1,2)


def ordinary_adapt(model,xs,ys,xq,lr,steps=100):
    if xs.shape[1]==0:
        with torch.no_grad():return model.predict(xs,ys,xq)
    results=[]
    # Each task has its own copy; no adaptation labels cross physical bodies.
    for i in range(xs.shape[0]):
        local=copy.deepcopy(model.base);optimizer=torch.optim.Adam(local.parameters(),lr=lr)
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True);loss=((local(xs[i])-ys[i])**2).mean()
                loss.backward();optimizer.step()
        with torch.no_grad():results.append(local(xq[i]))
    return torch.stack(results)


def prediction(model,method,xs,ys,xq,adapt_lr):
    if method=='coda':
        # Adapt context once, then chunk query evaluation with frozen globals.
        context=model.adapt_context(xs,ys,steps=100,lr=adapt_lr)
        theta=model.theta0.detach()+context@model.W.detach().T
        with torch.no_grad():return model.functional(xq,theta)
    if method=='pooled':return ordinary_adapt(model,xs,ys,xq,adapt_lr)
    with torch.no_grad():
        return torch.cat([model.predict(xs,ys,q) for q in xq.split(128,dim=1)],dim=1)


def evaluate(model,method,x,y,ids,adapt_lr=.03,budgets=BUDGETS,shuffle=False):
    predictions={};scores={};start=time.time()
    for n in budgets:
        sx,sy,qx,qy=fixed_samples(x,y,ids,n,shuffle)
        pred=prediction(model,method,sx,sy,qx,adapt_lr)
        assert pred.shape==qy.shape and torch.isfinite(pred).all(),('nonfinite forecast',method,n)
        error=((pred-qy)**2).mean((1,2)).detach().cpu().numpy()
        predictions[str(n)]=pred.detach().cpu().numpy();scores[str(n)]=error.tolist()
    synchronize()
    return predictions,scores,time.time()-start


def validation(model,method,x,y,ids):
    lrs=(.003,.01,.03) if method=='pooled' else ((.01,.03,.1) if method=='coda' else (.03,))
    records=[]
    for lr in lrs:
        _,scores,seconds=evaluate(model,method,x,y,ids,lr,budgets=(4,16,64))
        value=float(np.mean(list(scores.values())))
        records.append(dict(adaptation_lr=lr,criterion=value,body_scores=scores,wall_seconds=seconds))
    return min(records,key=lambda d:d['criterion']),records


def fit(args):
    prefix=ROOT/f'error_transport_screen_{args.method}_lr{args.lr:g}_seed{args.seed}'
    jp=Path(str(prefix)+'.json');wp=Path(str(prefix)+'.pt')
    assert not jp.exists() and not wp.exists(),'refuse overwrite'
    device='cuda' if torch.cuda.is_available() else 'cpu';x,y,split,scale=load(device)
    tr=np.flatnonzero(split==0);val=np.flatnonzero(split==1)
    initseed=int(np.random.SeedSequence([20261020,18,20,args.seed,METHODS.index(args.method)]).generate_state(1)[0])
    torch.manual_seed(initseed);np.random.seed(initseed);model=make(args.method).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(np.random.SeedSequence([20261020,18,21,args.seed]))
    best=float('inf');best_state=None;best_choice=None;records=[];losses=[];start=time.time()
    for step in range(args.steps):
        ids=rng.choice(tr,8);st=rng.integers(0,8,8);n=BUDGETS[int(rng.integers(0,len(BUDGETS)))];starts=rng.integers(0,97-n,8)
        body=torch.as_tensor(ids,device=device);supporttr=torch.as_tensor(st,device=device)
        times=torch.as_tensor(starts[:,None]+np.arange(n)[None],device=device)
        sx=x[body[:,None],supporttr[:,None],times];sy=y[body[:,None],supporttr[:,None],times]
        qt=(st[:,None]+rng.integers(1,8,(8,32)))%8;qi=rng.integers(0,96,(8,32))
        qx=x[body[:,None],torch.as_tensor(qt,device=device),torch.as_tensor(qi,device=device)]
        qy=y[body[:,None],torch.as_tensor(qt,device=device),torch.as_tensor(qi,device=device)]
        optimizer.zero_grad(set_to_none=True);loss=model.training_loss(sx,sy,qx,qy,body)
        assert torch.isfinite(loss),('nonfinite training',args.method,step)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1)%500==0 or step+1==args.steps:
            model.eval();choice,allchoices=validation(model,args.method,x,y,val);model.train()
            records.append(dict(step=step+1,mean_recent_training_loss=float(np.mean(losses[-100:])),validation=allchoices))
            if choice['criterion']<best:
                best=choice['criterion'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_choice=dict(choice,step=step+1)
                torch.save(dict(method=args.method,state_dict=best_state,scale=scale,choice=best_choice),wp)
            partial=dict(status='RUNNING EXPLORATORY FIT',method=args.method,lr=args.lr,seed=args.seed,steps_completed=step+1,best_validation=best_choice,records=records)
            save_json(jp,partial)
            print(args.method,args.lr,step+1,'validation',choice['criterion'],'best',best,flush=True)
    synchronize()
    out=dict(status='COMPLETE TRAIN/VALIDATION FIT; TEST NOT SCORED',method=args.method,lr=args.lr,seed=args.seed,steps=args.steps,
        parameter_count=sum(p.numel() for p in model.parameters()),initialization_seed=initseed,data_sha256=digest(DATA),driver_sha256=digest(Path(__file__)),
        model_source_sha256=digest(ROOT/'error_transport_model.py'),baseline_source_sha256=digest(ROOT/'error_transport_baselines.py'),
        best_validation=best_choice,validation_records=records,all_training_losses=losses,weights_sha256=digest(wp),wall_seconds=time.time()-start,
        device=device,limitations='Exploratory single-seed finite training; exact observed q/v. Validation reuses checkpoint selection. No test outcome selects candidates. Original baseline scopes documented separately.')
    save_json(jp,out)


def score(args):
    output=ROOT/f'error_transport_screen_seed{args.seed}_results.json';arraypath=ROOT/f'error_transport_screen_seed{args.seed}_predictions.npz'
    assert not output.exists() and not arraypath.exists()
    device='cuda' if torch.cuda.is_available() else 'cpu';x,y,split,scale=load(device)
    arrays={};results={};choices={};start=time.time()
    for method in METHODS:
        candidates=[]
        for lr in (.001,.003):
            prefix=ROOT/f'error_transport_screen_{method}_lr{lr:g}_seed{args.seed}'
            report=json.loads(Path(str(prefix)+'.json').read_text())
            assert report['status'].startswith('COMPLETE') and report['data_sha256']==digest(DATA)
            assert report['driver_sha256']==digest(Path(__file__)) and report['model_source_sha256']==digest(ROOT/'error_transport_model.py') and report['baseline_source_sha256']==digest(ROOT/'error_transport_baselines.py')
            candidates.append((report,prefix))
        report,prefix=min(candidates,key=lambda pair:pair[0]['best_validation']['criterion'])
        weights=Path(str(prefix)+'.pt');assert digest(weights)==report['weights_sha256']
        state=torch.load(weights,map_location=device,weights_only=False);model=make(method).to(device);model.load_state_dict(state['state_dict']);model.eval()
        choices[method]=dict(report=str(prefix)+'.json',report_sha256=digest(Path(str(prefix)+'.json')),weights_sha256=digest(weights),validation=report['best_validation'],parameter_count=report['parameter_count'])
        methodscores={}
        for splitid,label in ((2,'test_id'),(3,'test_ood')):
            ids=np.flatnonzero(split==splitid)
            for wrong in (False,True):
                pred,scores,seconds=evaluate(model,method,x,y,ids,report['best_validation']['adaptation_lr'],shuffle=wrong)
                key=label+('_wrong_support' if wrong else '')
                methodscores[key]=dict(body_ids=ids.tolist(),standardized_body_mse=scores,wall_seconds=seconds,
                    raw_body_mse={n:np.mean(((p-y[torch.as_tensor(ids,device=device),1:].flatten(1,2).cpu().numpy())*scale['y_std'])**2,axis=(1,2)).tolist() for n,p in pred.items()})
                for n,p in pred.items():arrays[f'{method}_{key}_support{n}']=p
        results[method]=methodscores
        print('scored',method,flush=True)
    np.savez_compressed(arraypath,**arrays)
    out=dict(status='COMPLETE EXPLORATORY CROSS-BODY SCREEN; INDEPENDENT REVIEW PENDING',seed=args.seed,choices=choices,results=results,
        source_sha256=digest(Path(__file__)),data_sha256=digest(DATA),predictions_sha256=digest(arraypath),wall_seconds=time.time()-start,
        statistics_unit='Independent physical body within this one family; one trained seed, no confirmatory significance.',
        limitations='Exact observed q/v, measured delta-v targets, no controller capability result. A small adaptation screen cannot establish a competitive framework. Wrong support rolls bodies within same heldout split.')
    save_json(output,out);print('saved',output,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('fit','score'));parser.add_argument('--method',choices=METHODS);parser.add_argument('--lr',type=float,default=.001);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='fit':assert args.method is not None;fit(args)
    else:score(args)


if __name__=='__main__':main()
