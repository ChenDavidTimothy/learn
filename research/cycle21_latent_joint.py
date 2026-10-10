"""Cycle 21: test whether an ordinary learned two-factor controlled latent model
closes Cycle 20's decision-regret gap. New seeds; frozen protocol; no true
nuisance index, emission probabilities, or transition parameters during fit.

This is a CONTROL, not a proposed novel algorithm. Its explicitly supplied
structural family assumes one interacting nuisance and two binary latent bits.
"""
import argparse, json, time, statistics
import numpy as np
import torch
from scipy.optimize import minimize
from cycle12_partial_obs import make_task,sample,selected_channel,log_loss,N_PRE
from cycle15_decision import train,predict
from cycle13_oracle import prob_oracle

torch.set_num_threads(1)
S=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.long)
U=S[:,0];Z=S[:,1]
BOUNDS=[(.035,.44),(.035,.44),(.01,.49)]+[(.01,.49)]*4
STARTS=[
 np.array([.18,.18,.16,.09,.19,.19,.09],float),
 np.array([.24,.24,.26,.20,.20,.20,.20],float)
]

def prepare(x,y,w,k,t,j):
    obs=np.asarray(x[:,[t,j,k+t,k+j,2*k,2*k+1]],dtype=np.int64)
    tx=torch.tensor(obs,dtype=torch.long)
    return tx,torch.tensor(np.asarray(y,dtype=float),dtype=torch.double),torch.tensor(np.asarray(w,dtype=float),dtype=torch.double)

def forecast(tx,theta):
    """Condition on two observation frames and previous action.
    Four target/nuisance latent states; all parameters estimated from labels.
    """
    eu,ez,pz=theta[:3];rate=theta[3:].reshape(2,2)
    u0,z0,u1,z1,ap,a=[tx[:,i] for i in range(6)]
    n=len(tx)
    # Independent, unbiased initial latent bits are a structural assumption.
    e0u=torch.where(u0[:,None]==U[None,:],1-eu,eu)
    e0z=torch.where(z0[:,None]==Z[None,:],1-ez,ez)
    e1u=torch.where(u1[:,None]==U[None,:],1-eu,eu)
    e1z=torch.where(z1[:,None]==Z[None,:],1-ez,ez)
    start=.25*e0u*e0z
    end=e1u*e1z
    pu=rate[ap[:,None,None],Z[None,:,None]]
    tu=torch.where(U[None,:,None]==U[None,None,:],1-pu,pu)
    tz=torch.where(Z[None,:,None]==Z[None,None,:],1-pz,pz)
    unnorm=(start[:,:,None]*tu*tz*end[:,None,:]).sum(dim=1)
    pf=rate[a[:,None],Z[None,:]]
    p2=torch.where(U[None,:]==0,pf,1-pf)
    py=eu+(1-2*eu)*p2
    return torch.clamp((unnorm*py).sum(dim=1)/unnorm.sum(dim=1),1e-7,1-1e-7)

def fit(x,y,w,k,t,j,maxiter=42):
    tx,ty,tw=prepare(x,y,w,k,t,j)
    def f(theta_np):
        theta=torch.tensor(theta_np,dtype=torch.double,requires_grad=True)
        p=forecast(tx,theta)
        loss=(-(ty*torch.log(p)+(1-ty)*torch.log1p(-p))*tw).sum()/tw.sum()
        grad=torch.autograd.grad(loss,theta)[0]
        return loss.item(),grad.detach().numpy()
    fits=[minimize(f,start,jac=True,bounds=BOUNDS,method='L-BFGS-B',
                   options={'maxiter':maxiter,'ftol':5e-8}) for start in STARTS]
    best=min(fits,key=lambda v:v.fun)
    return best.x,float(best.fun),bool(best.success)

def pred(model,x,k,t,j):
    q=np.asarray(x[:,[t,j,k+t,k+j,2*k,2*k+1]],dtype=np.int64)
    with torch.no_grad():
        return forecast(torch.tensor(q,dtype=torch.long),
                        torch.tensor(model,dtype=torch.double)).numpy()

def evaluate_policy(predicted,test,task,regime):
    x0=test.copy();x0[:,-1]=0
    x1=test.copy();x1[:,-1]=1
    a=(predicted(x1)>predicted(x0)).astype(np.uint8)
    p0=prob_oracle(task,x0,regime)
    p1=prob_oracle(task,x1,regime)
    return float(np.mean(np.maximum(p0,p1)-np.where(a==1,p1,p0)))

def trial(seed,regime,budget):
    task=make_task(seed,7);k=task['k']
    pre,py=sample(task,seed+1,N_PRE,'stationary')
    target=selected_channel(pre,py,k)
    post,qy=sample(task,seed+883,budget,regime)
    test,yy=sample(task,seed+991,3500,regime)
    val_n=max(96,budget//5)
    train_x=np.r_[pre,post[:-val_n]]
    train_y=np.r_[py,qy[:-val_n]]
    train_w=np.r_[np.full(N_PRE,.1),np.ones(budget-val_n)]
    all_x=np.r_[pre,post];all_y=np.r_[py,qy]
    all_w=np.r_[np.full(N_PRE,.1),np.ones(budget)]
    started=time.perf_counter()
    scored=[]
    for j in range(k):
        if j==target:continue
        params,loss,_=fit(train_x,train_y,train_w,k,target,j)
        p=pred(params,post[-val_n:],k,target,j)
        scored.append((log_loss(p,qy[-val_n:]),j))
    score,chosen=min(scored)
    params,_,ok=fit(all_x,all_y,all_w,k,target,chosen)
    joint=lambda z: pred(params,z,k,target,chosen)
    dt=time.perf_counter()-started
    result={'latent_joint':{'loss':log_loss(joint(test),yy),
              'regret':evaluate_policy(joint,test,task,regime),
              'selection_loss':score,'fit_seconds':dt,
              'correct_nuisance_selected':int(chosen==int(np.where(task['perm']==1)[0][0])),
              'converged':ok}}
    for method in ['two_state_hmm','target_action_features']:
        tic=time.perf_counter()
        model=train(method,all_x,all_y,all_w,k,target)
        predictor=lambda z,m=method,mdl=model:predict(m,mdl,z,k,target)
        result[method]={'loss':log_loss(predictor(test),yy),
                         'regret':evaluate_policy(predictor,test,task,regime),
                         'fit_seconds':time.perf_counter()-tic}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=91921026)
    parser.add_argument('--budget',type=int,default=8192)
    parser.add_argument('--conditions',nargs='+',default=['stationary','nuisance_shift','contextual_action','target_rate_shift'])
    parser.add_argument('--output',default='research/cycle21_latent_joint_results.json')
    a=parser.parse_args();t0=time.perf_counter()
    out={'protocol':{'novel_seeds':a.seeds,'seed_start':a.seed_start,'budget':a.budget,
        'conditions':a.conditions,'nuisance_factors':7,'pre':N_PRE,
        'heldout_selection_fraction':.2,'same_labeled_data':True,
        'joint_model_structural_prior':'two latent binary bits; one unknown partner; uniform initial state',
        'no_oracle_training':True},'results':{}}
    for regime in a.conditions:
        rows=[]
        for i in range(a.seeds):
            rows.append(trial(a.seed_start+917*i,regime,a.budget))
            if (i+1)%5==0:
                print(regime,i+1,'/',a.seeds,'elapsed',round(time.perf_counter()-t0,1),flush=True)
        by={}
        for method in ['two_state_hmm','target_action_features','latent_joint']:
            by[method]={key:{'mean':statistics.mean(row[method][key] for row in rows),
                   'SE':statistics.stdev(row[method][key] for row in rows)/len(rows)**.5 if len(rows)>1 else None}
                   for key in rows[0][method]}
        by['paired_regret_latent_minus_target']={
           'mean':statistics.mean(row['latent_joint']['regret']-row['target_action_features']['regret'] for row in rows),
           'SE':statistics.stdev(row['latent_joint']['regret']-row['target_action_features']['regret'] for row in rows)/len(rows)**.5 if len(rows)>1 else None}
        out['results'][regime]=by
        print('SUMMARY',regime,[(m,round(by[m]['regret']['mean'],6)) for m in ('two_state_hmm','target_action_features','latent_joint')],flush=True)
    out['seconds']=time.perf_counter()-t0
    with open(a.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',a.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':
    main()
