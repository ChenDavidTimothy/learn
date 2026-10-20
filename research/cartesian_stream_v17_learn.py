#!/usr/bin/env python3
"""Exploratory ordinary-only causal learning. No plant/evaluator imports."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

H=16;DT=.1;CONTEXT_DIM=66
NN_CANDIDATES=[(w,d) for w in (64,128) for d in (1e-4,.01)]
LOCAL_CANDIDATES=[(n,r) for n in (128,512) for r in (1e-3,.1)]
FAILURE_STATE={}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def encode(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)

def context(observations,commands):
    o=np.asarray(observations);a=np.asarray(commands).reshape(-1,2)
    assert len(o)>=H+1 and len(a)>=H
    return np.r_[o[-1],np.concatenate([(o[-1]-o[-k-1])/(k*DT) for k in range(1,H+1)]),a[-H:].ravel()]

def tensor_context(observations,commands):
    p=observations[-1]
    return torch.cat((p,torch.cat([(p-observations[-k-1])/(k*DT) for k in range(1,H+1)]),commands[-H:].reshape(-1)))

def rows(o,a,ids):
    X=[];U=[];Y=[];origins=[]
    for i in ids:
        for t in range(H,len(a[i]),2):
            X.append(context(o[i,:t+1],a[i,:t]));U.append(a[i,t]/.5)
            baseline=o[i,t]+np.arange(1,3)[:,None]*(o[i,t]-o[i,t-1])
            Y.append((o[i,t+1:t+3]-baseline).ravel());origins.append([int(i),t])
    return np.array(X),np.array(U),np.array(Y),np.array(origins)

class Network(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(68,width),torch.nn.Tanh(),
            torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,4));self.double()
    def forward(self,x):return self.layers(x)

class Predictor:
    def __init__(self,kind,members,scale,config):
        self.kind=kind;self.members=members;self.scale=scale;self.config=config
        self.cx=torch.tensor(scale['context_mean'],dtype=torch.float64)
        self.cs=torch.tensor(scale['context_std'],dtype=torch.float64)
        self.ym=torch.tensor(scale['target_mean'],dtype=torch.float64)
        self.ys=torch.tensor(scale['target_std'],dtype=torch.float64)

    def prepare(self,observations,commands):
        if self.kind=='nonlinear':return self.members,None,[]
        query=(context(observations,commands)-self.cx.numpy())/self.cs.numpy()
        fits=[m.fit(query) for m in self.members]
        return [torch.tensor(coef,dtype=torch.float64) for coef,_ in fits],torch.tensor(query),[d for _,d in fits]

    def paths(self,observations,commands,plan,prepared=None):
        if prepared is None:prepared=self.prepare(observations,commands)
        members,query,_=prepared;outputs=[]
        for member in members:
            o=torch.tensor(np.asarray(observations)[-H-1:],dtype=torch.float64)
            a=torch.tensor(np.asarray(commands).reshape(-1,2)[-H:],dtype=torch.float64)
            predictions=[]
            for u in plan:
                c=(tensor_context(o,a)-self.cx)/self.cs
                if self.kind=='nonlinear':y=member(torch.cat((c,u/.5)))
                else:y=torch.cat((torch.ones(1,dtype=torch.float64),c-query,u/.5))@member
                correction=(self.ym+self.ys*y).reshape(2,2)
                baseline=o[-1]+torch.arange(1,3,dtype=torch.float64)[:,None]*(o[-1]-o[-2])
                nxt=baseline+correction;predictions.append(nxt)
                o=torch.cat((o,nxt),dim=0)[-H-1:]
                a=torch.cat((a,u.repeat(2,1)),dim=0)[-H:]
            outputs.append(torch.cat(predictions,dim=0))
        return torch.stack(outputs)

    def append_completed(self,observations,commands):
        if self.kind!='local':return
        o=np.asarray(observations);a=np.asarray(commands).reshape(-1,2);t=len(a)-2
        if t<H:return
        x=(context(o[:t+1],a[:t])-self.cx.numpy())/self.cs.numpy()
        baseline=o[t]+np.arange(1,3)[:,None]*(o[t]-o[t-1])
        y=((o[t+1:t+3]-baseline).ravel()-self.ym.numpy())/self.ys.numpy()
        for m in self.members:m.append(x,a[t]/.5,y)

def forecast_score(predictor,o,a,ids,blocks=6,stride=24):
    records=[]
    with torch.no_grad():
        for i in ids:
            for t in range(H,a.shape[1]-2*blocks+1,stride):
                plan=torch.tensor(a[i,t:t+2*blocks:2],dtype=torch.float64)
                pred=predictor.paths(o[i,:t+1],a[i,:t],plan).numpy()
                target=o[i,t+1:t+2*blocks+1]
                if not np.all(np.isfinite(pred)) or not np.isfinite(np.mean((pred-target)**2)):
                    raise FloatingPointError('Nonfinite validation/test rollout; failed candidate preserved')
                records.append(dict(trajectory=int(i),time_index=t,
                    member_coordinate_mse=float(np.mean((pred-target)**2)),
                    ensemble_mean_coordinate_mse=float(np.mean((pred.mean(0)-target)**2)),
                    member_mse_by_future_sample=np.mean((pred-target)**2,axis=(0,2))))
    return dict(member_coordinate_mse=float(np.mean([r['member_coordinate_mse'] for r in records])),
        ensemble_mean_coordinate_mse=float(np.mean([r['ensemble_mean_coordinate_mse'] for r in records])),
        origins=len(records),records=records)

def local_predictor(X,U,Y,bootstrap_rows,scale,config):
    from cartesian_stream_v17_local import LocalPredictor
    n,r=config
    return Predictor('local',[LocalPredictor(X[b],U[b],Y[b],n,r) for b in bootstrap_rows],scale,dict(neighbors=n,ridge=r))

def load_selected(prefix,ordinary):
    p=Path(str(prefix)+'_fit.json');meta=json.loads(p.read_text());weights=np.load(str(prefix)+'_weights.npz',allow_pickle=False)
    assert digest(ordinary)==meta['ordinary_sha256']
    data=np.load(ordinary,allow_pickle=False);o=data['observed'];a=data['commands'];ids=np.flatnonzero(data['split']==0)
    X,U,Y,_=rows(o,a,ids);s=meta['scale'];X=(X-s['context_mean'])/s['context_std'];Y=(Y-s['target_mean'])/s['target_std']
    bootstrap=[np.array(b) for b in meta['bootstrap_rows']];chosen={}
    ni=meta['selected']['nonlinear'];record=meta['nonlinear_candidates'][ni];nets=[]
    for member in range(3):
        net=Network(record['width']);state={k:torch.tensor(weights[f'nn{ni}_member{member}_{k}']) for k in net.state_dict()}
        net.load_state_dict(state);net.eval();net.requires_grad_(False);nets.append(net)
    chosen['nonlinear']=Predictor('nonlinear',nets,s,dict(width=record['width'],weight_decay=record['weight_decay']))
    li=meta['selected']['local'];chosen['local']=local_predictor(X,U,Y,bootstrap,s,LOCAL_CANDIDATES[li])
    return chosen,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--ordinary',default='research/cartesian_stream_v17_rep0_ordinary.npz')
    p.add_argument('--output-prefix',default='research/cartesian_stream_v17_rep0');args=p.parse_args()
    output=Path(str(args.output_prefix)+'_fit.json');weight_path=Path(str(args.output_prefix)+'_weights.npz')
    failure=Path(str(args.output_prefix)+'_fit_failure.json')
    if output.exists() or weight_path.exists() or failure.exists():raise FileExistsError('Refusing to overwrite learned artifacts')
    FAILURE_STATE.update(prefix=args.output_prefix,ordinary=args.ordinary,active='load ordinary corpus')
    start=time.time();torch.set_num_threads(1);data=np.load(args.ordinary,allow_pickle=False)
    assert set(data.files)=={'observed','commands','split','trajectory_index','dt','sensor_sd'}
    assert float(data['dt'])==DT
    o=data['observed'];a=data['commands'];splits=data['split'];train=np.flatnonzero(splits==0);val=np.flatnonzero(splits==1);test=np.flatnonzero(splits==2)
    X,U,Y,origins=rows(o,a,train)
    scale=dict(context_mean=X.mean(0),context_std=np.maximum(X.std(0),.01),target_mean=Y.mean(0),target_std=np.maximum(Y.std(0),.001))
    X=(X-scale['context_mean'])/scale['context_std'];Y=(Y-scale['target_mean'])/scale['target_std']
    bootstrap=[];bootstrap_ids=[];bootstrap_paths=[]
    for member in range(3):
        path=[20261020,17,660,0,0,member];rng=np.random.default_rng(np.random.SeedSequence(path));bs=rng.choice(train,len(train),replace=True)
        bootstrap.append(np.concatenate([np.flatnonzero(origins[:,0]==i) for i in bs]));bootstrap_ids.append(bs);bootstrap_paths.append(path)
    nn_records=[];local_records=[];all_weights={};nn_predictors=[];local_predictors=[]
    FAILURE_STATE.update(nonlinear_candidates=nn_records,local_candidates=local_records,weights=all_weights)
    with threadpool_limits(limits=1):
        for ci,(width,decay) in enumerate(NN_CANDIDATES):
            nets=[];members=[]
            for member,bs in enumerate(bootstrap):
                path=[20261020,17,660,0,1,ci,member]
                seed=int(np.random.default_rng(np.random.SeedSequence(path)).integers(0,2**31))
                torch.manual_seed(seed);net=Network(width);optim=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=decay)
                tx=torch.tensor(np.c_[X[bs],U[bs]],dtype=torch.float64);ty=torch.tensor(Y[bs],dtype=torch.float64);losses=[]
                FAILURE_STATE.update(active=dict(kind='nonlinear',candidate=ci,member=member,seed=seed),current_net=net,current_losses=losses)
                for epoch in range(300):
                    optim.zero_grad();loss=torch.mean((net(tx)-ty)**2);losses.append(float(loss.detach()))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; candidate not selected')
                    loss.backward();optim.step()
                for k,v in net.state_dict().items():all_weights[f'nn{ci}_member{member}_{k}']=v.numpy().copy()
                net.eval();net.requires_grad_(False);nets.append(net)
                members.append(dict(initialization_seed_path=path,torch_seed=seed,normalized_training_loss_before_each_update=losses))
            predictor=Predictor('nonlinear',nets,scale,dict(width=width,weight_decay=decay));nn_predictors.append(predictor)
            score=forecast_score(predictor,o,a,val);nn_records.append(dict(width=width,weight_decay=decay,members=members,validation=score))
            print('nonlinear',ci,width,decay,'validation member MSE',score['member_coordinate_mse'],flush=True)
        for ci,config in enumerate(LOCAL_CANDIDATES):
            FAILURE_STATE.update(active=dict(kind='local',candidate=ci))
            predictor=local_predictor(X,U,Y,bootstrap,scale,config);local_predictors.append(predictor)
            score=forecast_score(predictor,o,a,val);local_records.append(dict(neighbors=config[0],ridge=config[1],validation=score))
            print('local',ci,config,'validation member MSE',score['member_coordinate_mse'],flush=True)
        selected=dict(nonlinear=int(np.argmin([r['validation']['member_coordinate_mse'] for r in nn_records])),
            local=int(np.argmin([r['validation']['member_coordinate_mse'] for r in local_records])))
        test_scores={}
        for kind,predictor in [('nonlinear',nn_predictors[selected['nonlinear']]),('local',local_predictors[selected['local']])]:
            test_scores[kind]=dict(six_block=forecast_score(predictor,o,a,test),
                one_block=forecast_score(predictor,o,a,test,blocks=1,stride=2))
    np.savez_compressed(weight_path,**all_weights)
    result=dict(status='EXPLORATORY ORDINARY MOVING-HISTORY FIT; NOT A LEARNING RELEASE',source_sha256=digest(__file__),
        local_source_sha256=digest(Path(__file__).with_name('cartesian_stream_v17_local.py')),
        design_sha256=digest(Path(__file__).with_name('cartesian_stream_v17_learning_design.md')),ordinary_sha256=digest(args.ordinary),
        weights_sha256=digest(weight_path),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        scale=scale,training_origins=origins,bootstrap_episode_ids=bootstrap_ids,bootstrap_rows=bootstrap,bootstrap_seed_paths=bootstrap_paths,
        nonlinear_candidates=nn_records,local_candidates=local_records,selected=selected,test_observed_scores=test_scores,
        resources=dict(training_rows=len(X),validation_trajectories=len(val),test_trajectories=len(test),nonlinear_fits=12,
            nonlinear_full_batch_updates=3600,new_physical_experience=0),wall_seconds=time.time()-start,
        limitations='Finite16-delay interface; mean path rollouts; episode bootstrap fit variation is not complete context uncertainty. Local receives same-data warmstart and online rows in control; nonlinear weights frozen. Test diagnostics never select candidates.')
    output.write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n');print('selected',selected,'seconds',time.time()-start)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if FAILURE_STATE:
            state=FAILURE_STATE.copy();prefix=state.pop('prefix');weights=state.pop('weights',{}).copy();net=state.pop('current_net',None)
            if net is not None:
                weights.update({'failed_current_'+k:v.numpy().copy() for k,v in net.state_dict().items()})
            state.update(status='FAILED/INTERRUPTED FIT; NOT SELECTED',error_type=type(error).__name__,error=str(error),source_sha256=digest(__file__))
            # Nonfinite arrays remain exact in NPZ; stringify nonfinite scalar losses.
            state['current_losses']=[v if np.isfinite(v) else str(v) for v in state.get('current_losses',[])]
            Path(str(prefix)+'_fit_failure.json').write_text(json.dumps(state,default=encode,indent=2,allow_nan=False)+'\n')
            np.savez_compressed(str(prefix)+'_fit_failure_weights.npz',**weights)
        raise
