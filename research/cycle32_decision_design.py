"""Cycle 32: decision-identifying experimental design under an explicit finite model.

Standard exact Bayesian model selection, not autonomous model discovery.
Compare greedy policy-value information, entropy, policy information,
two-query planned value of information, and random exploration.
All strategies receive identical *privileged* candidate likelihood/reward
tables and the same probe budget. True model used only in simulator.
Includes complementarity (XOR) to falsify purely myopic value exploration.
"""
import argparse,json,time,math
import numpy as np

BUDGET=2
METHODS=('dp_optimal','evi_greedy','evi_lookahead','model_mi','policy_mi','random')
def entropy(p):
    p=np.clip(p,1e-12,1.-1e-12)
    return -(p*np.log(p)+(1-p)*np.log1p(-p))
def model_value(post,mu):
    return float(np.max(post@mu))
def updated(post,q):
    prob=float(post@q)
    return prob,post*q/prob,post*(1-q)/(1-prob)
def expected_next(post,lik,mu):
    result=np.zeros(lik.shape[1])
    for j in range(lik.shape[1]):
        pr,p1,p0=updated(post,lik[:,j])
        result[j]=pr*model_value(p1,mu)+(1-pr)*model_value(p0,mu)
    return result
def design_scores(post,lik,mu,kind):
    E=lik.shape[1]
    if kind in ('evi_greedy','evi_lookahead'):
        next_val=expected_next(post,lik,mu)
        if kind=='evi_greedy':
            return next_val-model_value(post,mu)
        score=np.empty(E)
        for j in range(E):
            pr,p1,p0=updated(post,lik[:,j])
            # Two-stage dynamic programming: *optimal next* experiment
            # after each possible first-step response.
            v1=np.max(expected_next(p1,lik,mu))
            v0=np.max(expected_next(p0,lik,mu))
            score[j]=pr*v1+(1-pr)*v0-model_value(post,mu)
        return score
    if kind=='model_mi':
        marg=post@lik
        return entropy(marg)-post@entropy(lik)
    labels=(mu[:,1]>mu[:,0])
    yprob=post@labels.astype(float)
    if kind=='policy_mi':
        base=float(entropy(yprob))
        scores=np.empty(E)
        for j in range(E):
            pr,p1,p0=updated(post,lik[:,j])
            scores[j]=base-pr*entropy(float(p1@labels.astype(float)))-(1-pr)*entropy(float(p0@labels.astype(float)))
        return scores
    raise ValueError(kind)
def sample_world(seed,family):
    rg=np.random.default_rng(seed)
    H=64
    bits=((np.arange(H)[:,None]>>np.arange(6))&1).astype(float)
    E=8
    if family=='nuisance':
        relevant=int(rg.integers(6))
        # Each probe learns one of 6 bits; 2 redundant probes.
        probe_bits=rg.permutation(np.r_[np.arange(6),rg.integers(6,size=2)])
        eps=rg.uniform(.08,.22,size=E)
        lik=eps[None,:]+(1-2*eps[None,:])*bits[:,probe_bits]
        gap=rg.uniform(.30,.48)
        contrast=(2*bits[:,relevant]-1)*gap
    elif family=='xor':
        b0,b1=rg.choice(6,size=2,replace=False)
        probe_bits=rg.permutation(np.r_[np.arange(6),rg.integers(6,size=2)])
        eps=rg.uniform(.07,.14,size=E)
        lik=eps[None,:]+(1-2*eps[None,:])*bits[:,probe_bits]
        gap=rg.uniform(.34,.48)
        contrast=(2*(bits[:,b0]!=bits[:,b1]).astype(float)-1)*gap
    elif family=='random':
        # Full-rank random experiment families, randomized rewards,
        # no engineered bit alignment.
        x=rg.standard_normal((H,E))
        lik=.10+.8/(1+np.exp(-1.8*x))
        contrast=np.tanh(rg.standard_normal(H))*rg.uniform(.22,.5)
    elif family=='blind':
        # Information-theoretic negative control: ALL probes are
        # independent of the unknown reward-optimal action.
        relevant=int(rg.integers(6))
        allowed=[j for j in range(6) if j!=relevant]
        probe_bits=rg.choice(allowed,size=E,replace=True)
        eps=rg.uniform(.1,.25,size=E)
        lik=eps[None,:]+(1-2*eps[None,:])*bits[:,probe_bits]
        contrast=(2*bits[:,relevant]-1)*rg.uniform(.3,.48)
    else:raise ValueError(family)
    mu=np.column_stack((.5-contrast/2,.5+contrast/2))
    return mu,lik,int(rg.integers(H))
def optimal_probe(post,lik,mu,remaining,cost):
    """Finite-horizon exact Bayesian experimental design with stop option.

    Returns (Bayes expected value NET of future experiment costs, query index)
    without consulting the unknown true model. This is the strongest coherent
    finite-model benchmark, NOT a proposed new method.
    """
    stop=model_value(post,mu)
    if remaining<=0:return stop,None
    best=stop;best_j=None
    for j in range(lik.shape[1]):
        pr,p1,p0=updated(post,lik[:,j])
        y1,_=optimal_probe(p1,lik,mu,remaining-1,cost)
        y0,_=optimal_probe(p0,lik,mu,remaining-1,cost)
        value=pr*y1+(1-pr)*y0-cost
        if value>best+1e-12:best=value;best_j=j
    return best,best_j

def run(seed,family,probe_cost):
    mu,lik,truth=sample_world(seed,family)
    H,E=lik.shape
    # Common potential probe outcomes (counterfactual uniform noise)
    rng=np.random.default_rng(seed+812777)
    uniforms=rng.random((BUDGET,E))
    result={}
    for m in METHODS:
        post=np.full(H,1/H)
        picked=[]
        for step in range(BUDGET):
            if m=='dp_optimal':
                _,query=optimal_probe(post,lik,mu,BUDGET-step,probe_cost)
                if query is None:break
                j=int(query)
            elif m=='random':
                j=int(np.random.default_rng(seed+9377+step).integers(E))
            else:
                choice=m
                if m=='evi_lookahead' and step>0:choice='evi_greedy'
                scores=design_scores(post,lik,mu,choice)
                j=int(np.argmax(scores))
                if m.startswith('evi_'):
                    threshold=probe_cost*(BUDGET-step if m=='evi_lookahead' and step==0 else 1)
                    if scores[j]<=threshold+1e-12:break
            outcome=int(uniforms[step,j]<lik[truth,j])
            prob,p1,p0=updated(post,lik[:,j])
            post=p1 if outcome else p0
            picked.append(j)
        action=int(np.argmax(post@mu))
        regret=float(np.max(mu[truth])-mu[truth,action])
        result[m]={'regret':regret,'probe_count':len(picked),
                   'net_cost':regret+probe_cost*len(picked),
                   'correct_action':int(action==np.argmax(mu[truth]))}
    return result
def summarize(xs):
    x=np.array(xs,float);n=len(x)
    return {'mean':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(n))}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=600)
    parser.add_argument('--probe-cost',type=float,default=.012)
    parser.add_argument('--seed-start',type=int,default=32291026)
    parser.add_argument('--output',default='research/cycle32_decision_design_results.json')
    a=parser.parse_args();now=time.perf_counter()
    families=('nuisance','xor','random','blind')
    out={'config':{'source':'synthetic finite hypothesis Bayesian experimental design',
         'hypotheses':64,'candidate_experiments':8,'budget':BUDGET,'seeds_each':a.seeds,
         'probe_cost':a.probe_cost,'seed_start':a.seed_start,'families':families,
         'prior':'uniform; FULL true candidate likelihood/reward tables known to ALL methods',
         'true_index':'used in simulator and regret metric ONLY',
         'policy':'Bayesian action maximizing expected reward under posterior',
         'methods':METHODS,'predeclared_failure':'greedy EVI can fail on XOR'},
         'results':{}}
    for fam_index,family in enumerate(families):
        records=[]
        for i in range(a.seeds):
            records.append(run(a.seed_start+400000*fam_index+211*i,family,a.probe_cost))
            if (i+1)%200==0:
                print(family,i+1,'/',a.seeds,'elapsed',round(time.perf_counter()-now,1),flush=True)
        summary={}
        for method in METHODS:
            summary[method]={key:summarize([r[method][key] for r in records])
                             for key in records[0][method]}
        for m in METHODS:
            if m=='dp_optimal':continue
            summary['paired_dp_minus_'+m+'_net']=summarize(
               [r['dp_optimal']['net_cost']-r[m]['net_cost'] for r in records])
        out['results'][family]=summary
        print('RESULT',family,[(m,round(summary[m]['net_cost']['mean'],5),
                      round(summary[m]['regret']['mean'],5),
                      round(summary[m]['probe_count']['mean'],2)) for m in METHODS],flush=True)
    out['runtime_seconds']=time.perf_counter()-now
    with open(a.output,'w') as f:json.dump(out,f,indent=2)
    print('SAVED',a.output,'SECONDS',round(out['runtime_seconds'],1),flush=True)
if __name__=='__main__':main()
