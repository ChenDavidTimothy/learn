"""Cycle 42: test-time gradient adaptation to an unseen reward rule.

The reward mechanism is a synthetic overlay on action-invariant external
POPGym card observation sequences, as in Cycle41. Source random-action
feedback uses first or lag4; target is withheld lag7.

Compare *standard* fine-tuning of a GRU/causal Transformer pretrained
on seen tasks, training the same model from scratch on target-only
data, a frozen source model, and an exact online Bayes rule-selector
privileged with lag7 in its hypothesis library (Cycle41 comparator).
No learner receives the target name or target action labels.
No novel adaptation algorithm is proposed.
"""
import argparse,copy,json,time
import numpy as np,torch,torch.nn.functional as F
from cycle41_hidden_reward_context import (
    base_cards, observations,fit,build_model,evaluate_neural,
    evaluate_experts, TRAIN_RULES)
from cycle40_raw_memory_baselines import mask_bce,DEVICE
torch.set_num_threads(2)
BUDGETS=(0,4,16,64,256)
def adapt(model,train,val,seed,epochs=64):
    random=np.random.default_rng(seed)
    m=copy.deepcopy(model).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=.0012,weight_decay=.001)
    train=[torch.from_numpy(v) for v in train]
    val=[torch.from_numpy(v) for v in val]
    best=1e9;best_state=copy.deepcopy(m.state_dict());stale=0;used=0
    for epoch in range(epochs):
        idx=random.permutation(len(train[0]))
        m.train()
        for j in range(0,len(idx),24):
            sl=idx[j:j+24]
            x,a,y=(v[sl].to(DEVICE) for v in train)
            opt.zero_grad(set_to_none=True)
            loss=mask_bce(m(x),a,y,torch.ones_like(y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1)
            opt.step()
        m.eval()
        with torch.no_grad():
            x,a,y=(v.to(DEVICE) for v in val)
            metric=float(mask_bce(m(x),a,y,torch.ones_like(y)))
        used=epoch+1
        if metric<best-.0005:
            best=metric;stale=0;best_state=copy.deepcopy(m.state_dict())
        else:stale+=1
        if stale>=9:break
    m.load_state_dict(best_state)
    return m,{'epochs':used,'validation_bce':best}
def train_target(base,post_data,budget,seed,epochs):
    x,a,y=(v[:budget] for v in post_data)
    shuffle=np.random.default_rng(seed+311).permutation(budget)
    nval=max(1,budget//5)
    ti,vi=shuffle[nval:],shuffle[:nval]
    return adapt(base,(x[ti],a[ti],y[ti]),(x[vi],a[vi],y[vi]),seed+121,epochs)
def trial(seed,source_n,eval_n,epochs):
    rg=np.random.default_rng(seed)
    cards=base_cards(seed,source_n)
    modes=rg.choice(TRAIN_RULES,size=source_n)
    x,a,y=observations(cards,modes,seed+118,True)
    ix=rg.permutation(source_n)
    nval=max(60,source_n//5)
    pre={}
    for name in ('GRU','Transformer'):
        tic=time.perf_counter()
        m,fit_stats=fit(name,(x[ix[nval:]],a[ix[nval:]],y[ix[nval:]]),
                       (x[ix[:nval]],a[ix[:nval]],y[ix[:nval]]),seed+len(name)*337,epochs)
        pre[name]=(m,fit_stats,time.perf_counter()-tic)
    target=base_cards(seed+700000,256)
    post_data=observations(target,np.array(['lag7']*len(target)),seed+12345,True)
    eval_cards=base_cards(seed+900000,eval_n)
    result={'frozen':{},'finetune':{},'scratch':{},'reference':{}}
    for name,(m,stats,seconds) in pre.items():
        result['frozen'][name]={'return':evaluate_neural(
            m,name,eval_cards,'lag7',True)['return'],
            'source_fit_seconds':seconds}
    for budget in BUDGETS:
        k=str(budget)
        result['finetune'][k]={};result['scratch'][k]={}
        if budget==0:continue
        for name in pre:
            tic=time.perf_counter()
            ft,stats=train_target(pre[name][0],post_data,budget,seed+budget+len(name),epochs)
            train_time=time.perf_counter()-tic
            perf=evaluate_neural(ft,name,eval_cards,'lag7',True)
            result['finetune'][k][name]={
                'return':perf['return'],'update_seconds':train_time,**stats}
            start=time.perf_counter()
            # Same post samples and algorithm budget, but scratch init.
            torch.manual_seed(seed+budget+987+len(name))
            scratch=build_model(name).to(DEVICE)
            fresh,stats2=train_target(scratch,post_data,budget,
                                 seed+budget+len(name),epochs)
            result['scratch'][k][name]={
               'return':evaluate_neural(fresh,name,eval_cards,'lag7',True)['return'],
               'update_seconds':time.perf_counter()-start,**stats2}
        print('TARGET',seed,budget,
              {name:(round(result['finetune'][k][name]['return'],3),
                     round(result['scratch'][k][name]['return'],3))
               for name in pre},flush=True)
    result['reference']['Bayes_in_class']=evaluate_experts(
        eval_cards,'lag7',TRAIN_RULES)['return']
    result['reference']['Bayes_explicit_lag7']=evaluate_experts(
        eval_cards,'lag7',('first','lag4','lag7'))['return']
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=4)
    p.add_argument('--source',type=int,default=1200)
    p.add_argument('--eval',type=int,default=96)
    p.add_argument('--epochs',type=int,default=70)
    p.add_argument('--output',default='research/cycle42_ooc_adaptation_results.json')
    a=p.parse_args();started=time.perf_counter()
    results={'protocol':{
       'source':'POPGym card observation trajectories, synthetic hidden reward overlay from Cycle41',
       'source_rules':TRAIN_RULES,'unseen_target_rule':'lag7',
       'source_episodes':a.source,'post_budgets':BUDGETS,'test_episodes':a.eval,
       'trials':a.trials,'max_training_epochs':a.epochs,
       'training_actions':'uniform random; same logged target actions and reward data for all methods and budgets',
       'fine_tune_vs_scratch':'same model class and target examples, early stopping on 20% post logs',
       'method_models':['GRU','causal Transformer'],
       'evaluation':'unseen independent on-policy episode feedback, no target rule labels',
       'privileged_comparator':'Bayes with true lag7 explicitly in supplied rule family',
       'no_novel_algorithm':True},'trials':[],'summary':{}}
    for i in range(a.trials):
        data=trial(42111026+77131*i,a.source,a.eval,a.epochs)
        results['trials'].append(data)
        for key in BUDGETS:
            if not key:continue
            for mode in ('finetune','scratch'):
                for arch in ('GRU','Transformer'):
                    y=np.array([row[mode][str(key)][arch]['return'] for row in results['trials']])
                    results['summary'][f'{mode}_{arch}_{key}']={
                        'mean':float(y.mean()),
                        'SE':float(y.std(ddof=1)/np.sqrt(len(y))) if len(y)>1 else None}
        with open(a.output,'w') as f:json.dump(results,f,indent=2)
        print('TRIAL',i+1,'/',a.trials,'elapsed',round(time.perf_counter()-started,1),flush=True)
    results['seconds']=time.perf_counter()-started
    with open(a.output,'w') as f:json.dump(results,f,indent=2)
    print('SAVED',a.output,round(results['seconds'],1),flush=True)
if __name__=='__main__':main()
