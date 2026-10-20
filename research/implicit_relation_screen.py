#!/usr/bin/env python3
"""Frozen second mechanism screen. Original train/validation, fresh test bodies.

No evaluator archive, physical parameter or query label enters inference.
Test scoring runs only after candidate selection is saved.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from implicit_relation_model import ImplicitRelation
from error_transport_screen import load,fixed_samples,digest,save_json,synchronize,make as old_make

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'error_transport_screen_data.npz'
HOLDOUT=ROOT/'implicit_relation_holdout.npz'
BUDGETS=(0,4,16,64)

def make(method):return ImplicitRelation(implicit=method=='implicit')

def predict(model,method,sx,sy,qx,lr=.03):
    diagnostics={}
    if method in ('implicit','explicit'):
        with torch.no_grad():
            c=model.infer(sx,sy);pred=[];mins=[];conds=[]
            for q in qx.split(128,dim=1):
                p,d=model.conditioned(c,q,diagnostics=True);pred.append(p);mins.append(d['min_singular']);conds.append(d['condition'])
            predictions=torch.cat(pred,dim=1);minimum=torch.cat(mins,dim=1).amin(1);maximum=torch.cat(conds,dim=1).amax(1)
            diagnostics=dict(min_singular=minimum.cpu().tolist(),max_condition=maximum.cpu().tolist(),latent_norm=c.norm(dim=1).cpu().tolist(),
                conditioning_failure=(minimum<1e-4).cpu().tolist())
            return predictions,diagnostics
    if method=='alpaca':
        with torch.no_grad():
            coefficients,_=model.posterior(sx,sy)
            return model.features(qx)@coefficients,diagnostics
    if method=='coda':
        c=model.adapt_context(sx,sy,steps=100,lr=lr)
        with torch.no_grad():return model.functional(qx,model.theta0.detach()+c@model.W.detach().T),diagnostics
    raise ValueError(method)

def evaluate(model,method,x,y,ids,budgets=BUDGETS,wrong=False,lr=.03):
    arrays={};scores={};diagnostics={}
    for n in budgets:
        sx,sy,qx,qy=fixed_samples(x,y,ids,n,shuffle=wrong)
        p,d=predict(model,method,sx,sy,qx,lr)
        assert torch.isfinite(p).all()
        arrays[str(n)]=p.detach().cpu().numpy();scores[str(n)]=((p-qy)**2).mean((1,2)).cpu().tolist();diagnostics[str(n)]=d
    return arrays,scores,diagnostics

def fit(args):
    prefix=ROOT/f'implicit_relation_{args.method}_lr{args.lr:g}_seed0'
    jp=Path(str(prefix)+'.json');wp=Path(str(prefix)+'.pt');failure=Path(str(prefix)+'_failure.pt')
    assert not any(p.exists() for p in (jp,wp,failure))
    device='cuda' if torch.cuda.is_available() else 'cpu';x,y,split,scale=load(device)
    tr=np.flatnonzero(split==0);val=np.flatnonzero(split==1)
    seed=int(np.random.SeedSequence([20261020,18,31,0,('implicit','explicit').index(args.method)]).generate_state(1)[0])
    torch.manual_seed(seed);model=make(args.method).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(np.random.SeedSequence([20261020,18,21,0]))
    best=float('inf');choice=None;records=[];losses=[];start=time.time()
    try:
        for step in range(2000):
            ids=rng.choice(tr,8);st=rng.integers(0,8,8);n=BUDGETS[int(rng.integers(0,4))];starts=rng.integers(0,97-n,8)
            body=torch.as_tensor(ids,device=device);supporttr=torch.as_tensor(st,device=device)
            times=torch.as_tensor(starts[:,None]+np.arange(n)[None],device=device)
            sx=x[body[:,None],supporttr[:,None],times];sy=y[body[:,None],supporttr[:,None],times]
            qt=(st[:,None]+rng.integers(1,8,(8,32)))%8;qi=rng.integers(0,96,(8,32))
            qx=x[body[:,None],torch.as_tensor(qt,device=device),torch.as_tensor(qi,device=device)]
            qy=y[body[:,None],torch.as_tensor(qt,device=device),torch.as_tensor(qi,device=device)]
            optimizer.zero_grad(set_to_none=True);loss=model.training_loss(sx,sy,qx,qy)
            assert torch.isfinite(loss),'nonfinite training loss'
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            assert torch.isfinite(norm),'nonfinite gradient'
            optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%500==0:
                model.eval();_,scores,diag=evaluate(model,args.method,x,y,val,budgets=(4,16,64));model.train()
                conditioning_failure=any(any(d['conditioning_failure']) for d in diag.values())
                value=float(np.mean(list(scores.values())))
                rec=dict(step=step+1,criterion=value,body_scores=scores,diagnostics=diag,conditioning_failure=conditioning_failure)
                records.append(rec)
                # A failed checkpoint is recorded, never silently averaged or eligible.
                if not conditioning_failure and value<best:
                    best=value;choice=rec
                    torch.save(dict(method=args.method,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},scale=scale,choice=choice),wp)
                save_json(jp,dict(status='RUNNING TRAIN/VALIDATION',records=records,all_training_losses=losses,best_validation=choice))
                print(args.method,args.lr,step+1,value,'condition failure',conditioning_failure,flush=True)
        assert choice is not None,'no eligible validation checkpoint'
    except Exception as exc:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng_state=rng.bit_generator.state,
            sx=sx,sy=sy,qx=qx,qy=qy),failure)
        save_json(jp,dict(status='FAILED; NO RETRY',exception=repr(exc),step=step,records=records,all_training_losses=losses))
        raise
    synchronize()
    save_json(jp,dict(status='COMPLETE TRAIN/VALIDATION; TEST NOT SCORED',method=args.method,lr=args.lr,seed=0,initialization_seed=seed,
        parameter_count=sum(p.numel() for p in model.parameters()),best_validation=choice,validation_records=records,all_training_losses=losses,
        data_sha256=digest(DATA),holdout_sha256=digest(HOLDOUT),driver_sha256=digest(__file__),model_sha256=digest(ROOT/'implicit_relation_model.py'),
        plan_sha256=digest(ROOT/'implicit_relation_plan.md'),weights_sha256=digest(wp),wall_seconds=time.time()-start,device=device))

def select():
    choices={}
    for method in ('implicit','explicit','alpaca','coda'):
        options=[]
        for lr in (.001,.003):
            stem=f'implicit_relation_{method}' if method in ('implicit','explicit') else f'error_transport_screen_{method}'
            prefix=ROOT/f'{stem}_lr{lr:g}_seed0'
            path=Path(str(prefix)+'.json');r=json.loads(path.read_text());assert r['status'].startswith('COMPLETE')
            wp=Path(str(prefix)+'.pt');assert digest(wp)==r['weights_sha256'] and digest(DATA)==r['data_sha256']
            if method in ('implicit','explicit'):
                assert r['driver_sha256']==digest(__file__) and r['model_sha256']==digest(ROOT/'implicit_relation_model.py')
            options.append((r,prefix,path,wp))
        r,prefix,path,wp=min(options,key=lambda t:t[0]['best_validation']['criterion'])
        choices[method]=dict(report=str(path),report_sha256=digest(path),weights=str(wp),weights_sha256=digest(wp),
            validation=r['best_validation'],parameter_count=r['parameter_count'])
    return choices

def score():
    output=ROOT/'implicit_relation_results.json';arraypath=ROOT/'implicit_relation_predictions.npz';selection=ROOT/'implicit_relation_selection.json'
    assert not any(p.exists() for p in (output,arraypath,selection))
    choices=select();save_json(selection,dict(status='FROZEN VALIDATION-ONLY SELECTION; TEST NOT SCORED',choices=choices))
    device='cuda' if torch.cuda.is_available() else 'cpu';_,_,_,scale=load(device)
    with np.load(HOLDOUT,allow_pickle=False) as d:
        x=torch.as_tensor((d['x']-scale['x_mean'])/scale['x_std'],dtype=torch.float32,device=device)
        y=torch.as_tensor((d['y']-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device);split=d['split']
    results={};arrays={};start=time.time()
    for method,ch in choices.items():
        state=torch.load(ch['weights'],map_location=device,weights_only=False)
        model=(make(method) if method in ('implicit','explicit') else old_make(method)).to(device)
        model.load_state_dict(state['state_dict']);model.eval();results[method]={}
        for si,label in ((2,'test_id'),(3,'test_ood')):
            ids=np.flatnonzero(split==si)
            for wrong in (False,True):
                key=label+('_wrong_support' if wrong else '')
                pred,scores,diag=evaluate(model,method,x,y,ids,wrong=wrong,lr=ch['validation'].get('adaptation_lr',.03))
                results[method][key]=dict(body_ids=ids.tolist(),standardized_body_mse=scores,diagnostics=diag)
                for n,p in pred.items():arrays[f'{method}_{key}_support{n}']=p
        print('scored',method,flush=True)
    np.savez_compressed(arraypath,**arrays)
    save_json(output,dict(status='COMPLETE EXPLORATORY SECOND SCREEN; INDEPENDENT CHECK PENDING',choices=choices,results=results,
        data_sha256=digest(DATA),holdout_sha256=digest(HOLDOUT),selection_sha256=digest(selection),predictions_sha256=digest(arraypath),
        source_sha256=digest(__file__),wall_seconds=time.time()-start,
        scope='One trained seed per method, 8 new ID/8 new OOD bodies; no inferential ranking, no controller or framework claim. Ordinary inputs only.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit','score'));p.add_argument('--method',choices=('implicit','explicit'));p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args();torch.set_num_threads(1)
    if args.mode=='fit':assert args.method;fit(args)
    else:score()
