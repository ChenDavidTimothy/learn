"""Cycle 41: in-context reward-rule inference on external POPGym observation traces.

A composite latent-rule process uses independent observation trajectories
from POPGym RepeatFirstEasy, but REPLACES its reward with a synthetic
per-step first, lag4, or withheld lag7 memory rule. All reward mechanisms
score every step at +/-1/T to avoid trivial phase/onset cues. No rule ID is
given to neural learners; training uses uniformly randomized chosen actions
and their observed reward signs. At evaluation models use real policy actions,
observe those chosen-action rewards, and update history within each episode.

Contrasts standard GRU, causal Transformer with/without reward context,
and exact Bayesian policies endowed with both an in-class/exhaustive
candidate rule menu. This is established meta-/in-context system identification,
not a new representation mechanism. Overlaid rewards are NOT native POPGym.
"""
import argparse,json,time,copy
import numpy as np
import torch
from cycle40_raw_memory_baselines import GRU,CausalAttention,mask_bce,DEVICE
torch.set_num_threads(2)
RULES=('first','lag4','lag7')
TRAIN_RULES=('first','lag4')
L=51
def base_cards(seed,n):
    # External POPGym card observations; only observed stimuli retained.
    import popgym
    e=popgym.envs.RepeatFirstEasy()
    out=np.zeros((n,L),np.int64)
    for i in range(n):
        ob,_=e.reset(seed=seed+101*i)
        for t in range(L):
            out[i,t]=int(ob)
            ob,r,end,trunc,_=e.step(0)
    e.close()
    return out
def target(cards,rule,t):
    if rule=='first':return cards[:,0]
    k=int(rule.replace('lag',''))
    ix=max(0,t-k)
    return cards[:,ix]
def observations(cards,rules,seed,feedback=True):
    n,T=cards.shape
    rng=np.random.default_rng(seed)
    action=rng.integers(0,4,size=(n,T))
    yy=np.zeros((n,T),np.float32)
    X=np.zeros((n,T,12),np.float32)
    for t in range(T):
        goal=np.array([target(cards[j:j+1],rules[j],t)[0] for j in range(n)])
        yy[:,t]=(action[:,t]==goal).astype(np.float32)
        X[np.arange(n),t,cards[:,t]]=1.
        X[:,t,10]=t/51.
        X[:,t,11]=1.
        if t>0 and feedback:
            X[np.arange(n),t,4+action[:,t-1]]=1.
            X[:,t,8]=2*yy[:,t-1]-1
            X[:,t,9]=1.
    return X,action,yy
def build_model(name):
    if name=='GRU':return GRU(d=12)
    return CausalAttention(d=12)
def fit(name,train,val,seed,max_epochs):
    rng=np.random.default_rng(seed)
    model=build_model(name).to(DEVICE)
    torch.manual_seed(seed+117)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.001)
    tensors=[tuple(torch.from_numpy(v) for v in seq) for seq in (train,val)]
    best=1e9;weights=None;bad=0;used=0
    for ep in range(max_epochs):
        model.train()
        perm=rng.permutation(len(train[0]))
        for j in range(0,len(perm),32):
            ids=perm[j:j+32]
            x,a,y=(v[ids].to(DEVICE) for v in tensors[0])
            optim_mask=torch.ones_like(y)
            optimizer.zero_grad(set_to_none=True)
            loss=mask_bce(model(x),a,y,optim_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x,a,y=(v.to(DEVICE) for v in tensors[1])
            dev=float(mask_bce(model(x),a,y,torch.ones_like(y)).item())
        used=ep+1
        if dev<best-.0003:
            best=dev;weights=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=10:break
    model.load_state_dict(weights);model.eval()
    return model,{'epochs':used,'reward_bce':best}
def evaluate_neural(model,name,cards,rule,has_feedback):
    n,T=cards.shape
    x=np.zeros((n,T,12),np.float32)
    for t in range(T):
        x[np.arange(n),t,cards[:,t]]=1.
        x[:,t,10]=t/51.;x[:,t,11]=1.
    reward=np.zeros((n,T),np.float32)
    acts=np.zeros((n,T),np.int64)
    # Policy's own previous action and reward, never counterfactual labels.
    model.eval()
    with torch.no_grad():
        state=None
        for t in range(T):
            if name=='GRU':
                xi=torch.from_numpy(x[:,t:t+1,:]).to(DEVICE)
                z= torch.relu(model.enc(xi))
                output,state=model.gru(z,state)
                logits=model.head(output[:,0])
            else:
                logits=model(torch.from_numpy(x[:,:t+1]).to(DEVICE))[:,-1]
            a=logits.argmax(-1).cpu().numpy()
            acts[:,t]=a
            true=target(cards,rule,t)
            reward[:,t]=(a==true).astype(np.float32)
            if t+1<T and has_feedback:
                x[np.arange(n),t+1,4+a]=1.
                x[:,t+1,8]=2*reward[:,t]-1
                x[:,t+1,9]=1.
    return {'accuracy':float(reward.mean()),
            'return':float(2*reward.mean()-1),
            'early_accuracy':float(reward[:,:8].mean()),
            'late_accuracy':float(reward[:,8:].mean())}
def evaluate_experts(cards,rule,candidates):
    n,T=cards.shape
    beliefs=np.full((n,len(candidates)),1/len(candidates))
    reward=np.zeros((n,T),float)
    for t in range(T):
        goals=np.stack([target(cards,c,t) for c in candidates],axis=1)
        q=np.zeros((n,4))
        for a in range(4):q[:,a]=np.sum(beliefs*(goals==a),axis=1)
        acts=q.argmax(1)
        truth=target(cards,rule,t)
        y=(acts==truth)
        reward[:,t]=y
        predicted=(goals==acts[:,None])
        # Small model mismatch allowance prevents numerical collapse OOD.
        likelihood=np.where(predicted==y[:,None],.98,.02)
        beliefs*=likelihood
        beliefs/=beliefs.sum(1,keepdims=True)
    return {'accuracy':float(reward.mean()),'return':float(2*reward.mean()-1),
            'early_accuracy':float(reward[:,:8].mean()),'late_accuracy':float(reward[:,8:].mean())}
def run(seed,ntrain,neval,epochs):
    rng=np.random.default_rng(seed)
    cards=base_cards(seed,ntrain)
    rules=rng.choice(TRAIN_RULES,size=ntrain)
    ix=rng.permutation(ntrain)
    nval=max(80,ntrain//5)
    train_ix,val_ix=ix[nval:],ix[:nval]
    result={'methods':{},'evaluation':{}}
    fitted={}
    for name,feedback in [('GRU',True),('Transformer',True),('Transformer_no_feedback',False)]:
        x,a,y=observations(cards,rules,seed+10000,feedback)
        tic=time.perf_counter()
        mdl,stats=fit('GRU' if name=='GRU' else 'Transformer',
                      (x[train_ix],a[train_ix],y[train_ix]),
                      (x[val_ix],a[val_ix],y[val_ix]),seed+len(name)*419,epochs)
        result['methods'][name]={**stats,'train_seconds':time.perf_counter()-tic}
        fitted[name]=mdl
    for j,rule in enumerate(RULES):
        held=base_cards(seed+800000+j*27119,neval)
        per={}
        for name,feedback in [('GRU',True),('Transformer',True),('Transformer_no_feedback',False)]:
            per[name]=evaluate_neural(fitted[name],
                'GRU' if name=='GRU' else 'Transformer',held,rule,feedback)
        per['Bayes_in_class']=evaluate_experts(held,rule,TRAIN_RULES)
        per['Bayes_full_library']=evaluate_experts(held,rule,RULES)
        per['Oracle_rule']={'return':1.,'accuracy':1.}
        result['evaluation'][rule]=per
        print('EVAL',seed,rule,[(k,round(v['return'],3)) for k,v in per.items()],flush=True)
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--train',type=int,default=1200)
    p.add_argument('--eval',type=int,default=100)
    p.add_argument('--epochs',type=int,default=70)
    p.add_argument('--output',default='research/cycle41_hidden_reward_results.json')
    args=p.parse_args();start=time.perf_counter()
    out={'protocol':{'base_observations':'POPGym RepeatFirstEasy 1.0.6 card sequences',
       'reward':'experimenter overlay: fully scored +/-1/T, correct action given first/lag4/lag7 hidden rule',
       'training_rule_types':TRAIN_RULES,'heldout_rule_types':RULES,
       'out_of_distribution_rule':'lag7',
       'train_reward':'randomized chosen-action binary reward only',
       'model_inputs':'current and past cards, prior CHOSEN actions and received rewards, timestep, NO hidden rule ID',
       'online_test':'models update context with actual chosen actions and rewards within each episode; no gradient adaptation',
       'train_episodes':args.train,'eval_episodes':args.eval,
       'trials':args.trials,'max_epochs':args.epochs,
       'Bayes_in_class_prior':'explicit true first and lag4 candidates, known deterministic reward likelihood .98/.02',
       'Bayes_full_library_prior':'includes HELDOUT lag7 candidate (privileged)',
       'novelty':'standard in-context/meta-learning and Bayesian model selection; no novel mechanism',
       'device':str(DEVICE)},'trials':[],'summary':{}}
    for i in range(args.trials):
        trial=run(41111026+73171*i,args.train,args.eval,args.epochs)
        out['trials'].append(trial)
        for rule in RULES:
            out['summary'].setdefault(rule,{})
            for method in trial['evaluation'][rule]:
                scores=[r['evaluation'][rule][method]['return'] for r in out['trials']]
                out['summary'][rule][method]={'mean':float(np.mean(scores)),
                    'SE':float(np.std(scores,ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else None}
        with open(args.output,'w') as f:json.dump(out,f,indent=2)
        print('TRIAL',i+1,'/',args.trials,'elapsed',round(time.perf_counter()-start,1),flush=True)
    out['seconds']=time.perf_counter()-start
    with open(args.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',args.output,round(out['seconds'],1),flush=True)
if __name__=='__main__':main()
