"""Cycle 22: diagnose why predictive log loss and action regret rank
learners differently. Fresh independent seeds, oracle ONLY during evaluation.

A standard plug-in policy inequality under randomized binary actions is
R <= sqrt(2 * E_{H,A} KL(Ber(p_A)||Ber(q_A))) for P(A=0|H)=P(A=1|H)=1/2.
Proof: greedy-policy regret <= |p0-q0|+|p1-q1|;
Pinsker, Cauchy-Schwarz and Jensen yield the final bound.
This is known decision-theoretic reasoning, NOT a new theorem.
"""
import json,statistics,math,argparse,time
import numpy as np
from scipy.stats import t as student_t
from cycle12_partial_obs import N_PRE,make_task,sample,selected_channel,log_loss
from cycle15_decision import train,predict
from cycle13_oracle import prob_oracle

METHODS=('two_state_hmm','target_action_features')
MODES=('stationary','nuisance_shift','contextual_action','target_rate_shift')
def kl(p,q):
    q=np.clip(q,1e-8,1-1e-8)
    p=np.clip(p,1e-8,1-1e-8)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def one(seed,regime):
    task=make_task(seed,7);k=task['k']
    pre,py=sample(task,seed+1,N_PRE,'stationary')
    index=selected_channel(pre,py,k)
    post,qy=sample(task,seed+883,8192,regime)
    test,ty=sample(task,seed+991,3500,regime)
    x=np.r_[pre,post];y=np.r_[py,qy]
    w=np.r_[np.full(N_PRE,.1),np.ones(len(post))]
    x0=test.copy();x0[:,-1]=0
    x1=test.copy();x1[:,-1]=1
    p0=prob_oracle(task,x0,regime);p1=prob_oracle(task,x1,regime)
    tau=p1-p0
    gold=np.maximum(p0,p1)
    result={}
    for name in METHODS:
        model=train(name,x,y,w,k,index)
        q0=predict(name,model,x0,k,index)
        q1=predict(name,model,x1,k,index)
        pred=predict(name,model,test,k,index)
        action=(q1>q0).astype(np.uint8)
        risk=float(np.mean(gold-np.where(action==1,p1,p0)))
        ce=float((np.mean(kl(p0,q0))+np.mean(kl(p1,q1)))/2)
        result[name]=dict(
            regret=risk,
            realized_logloss=log_loss(pred,ty),
            oracle_excess_logloss=ce,
            regret_bound=math.sqrt(2*ce),
            wrong_action_rate=float(np.mean((action!=(tau>0)) & (np.abs(tau)>1e-9))),
            contrast_mae=float(np.mean(np.abs((q1-q0)-tau))),
            action_one_rate=float(action.mean()))
        assert risk <= math.sqrt(2*ce)+1e-9, (seed,regime,name,risk,ce)
    return result

def summarize(vals):
    n=len(vals);avg=float(np.mean(vals))
    se=float(np.std(vals,ddof=1)/np.sqrt(n))
    half=float(student_t.ppf(.975,n-1)*se)
    return dict(mean=avg,SE=se,CI95=[avg-half,avg+half])

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--seeds',type=int,default=120)
    a.add_argument('--seed-start',type=int,default=94921026)
    a.add_argument('--output',default='research/cycle22_policy_loss_results.json')
    args=a.parse_args()
    t0=time.perf_counter();out={'config':{'seeds':args.seeds,'seed_start':args.seed_start,'pre':N_PRE,'post':8192,'test':3500,'regimes':MODES,
        'true_oracle_used_only_for_scoring':True,'stronger_joint_baseline_separate_cycle21':True},'conditions':{}}
    for regime in MODES:
        trials=[]
        for i in range(args.seeds):
            trials.append(one(args.seed_start+1031*i,regime))
            if (i+1)%30==0: print(regime,i+1,'/',args.seeds,flush=True)
        sums={}
        for name in METHODS:
            sums[name]={k:summarize([r[name][k] for r in trials]) for k in trials[0][name]}
        for k in ('regret','realized_logloss','oracle_excess_logloss','contrast_mae'):
            sums['paired_target_minus_hmm_'+k]=summarize(
                 [r['target_action_features'][k]-r['two_state_hmm'][k] for r in trials])
        out['conditions'][regime]=sums
        print('RESULT',regime,'regret',sums['paired_target_minus_hmm_regret'],
              'loss',sums['paired_target_minus_hmm_realized_logloss'],flush=True)
    out['runtime_seconds']=time.perf_counter()-t0
    with open(args.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',args.output,'seconds',round(out['runtime_seconds'],1))
if __name__=='__main__': main()
