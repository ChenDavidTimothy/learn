"""Cycle 37: real randomized-treatment policy-learning audit on Hillstrom RCT.

Three treatment arms (No, Mens, Womens E-Mail), binary post-campaign
visit, only pre-treatment covariates. Known randomized assignment each arm
~1/3. Evaluate policy value on strictly held-out customers with doubly
robust scores and paired bootstrap. Compare constant policies against
linear, boosted, forest, and DR pseudo-outcome heterogeneous learners.
No person-level counterfactual ground truth is available.

Data downloaded on PC from a public third-party mirror of the original
Kevin Hillstrom MineThatData Email Challenge 2008. Do NOT commit data,
which have upstream terms distinct from our code. See source below.
https://github.com/AdityaDabrase/ab-testing-email-marketing
"""
import argparse,hashlib,json,time
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier

DATA='research/local_datasets/hillstrom.csv'
OUTCOME='visit'
ARMS=('No E-Mail','Mens E-Mail','Womens E-Mail')
FEATURES=('recency','history','mens','womens','zip_code','newbie','channel')
COSTS=(0.,.006,.012,.025)
PRIMARY_COST=.012
def load():
    df=pd.read_csv(DATA)
    assert len(df)==64000
    assert set(df.segment)==set(ARMS)
    assert df[list(FEATURES)+['visit','segment']].notna().all().all()
    x=pd.get_dummies(df[list(FEATURES)],columns=['zip_code','channel'],dtype=float).astype(float)
    y=df[OUTCOME].to_numpy(int)
    a=df.segment.map({name:i for i,name in enumerate(ARMS)}).to_numpy(int)
    assert np.array_equal(np.unique(a),[0,1,2])
    return x.to_numpy(float),y,a,list(x.columns),df
def features_s(x,a):
    one=np.eye(3)[a]
    return np.column_stack((x,one,(x[:,:,None]*one[:,None,:]).reshape(len(x),-1)))
def s_logit(x,y,a,queries,seed):
    scale=StandardScaler().fit(x)
    sx=scale.transform(x)
    all_train=features_s(sx,a)
    model=LogisticRegression(C=.2,solver='lbfgs',max_iter=350,random_state=seed)
    model.fit(all_train,y)
    return [np.column_stack([model.predict_proba(features_s(scale.transform(q),np.full(len(q),arm,int)))[:,1] for arm in range(3)]) for q in queries]
def t_logit(x,y,a,queries,seed):
    scaler=StandardScaler().fit(x)
    model=[]
    for arm in range(3):
        clf=LogisticRegression(C=.25,solver='lbfgs',max_iter=350,random_state=seed+arm)
        clf.fit(scaler.transform(x[a==arm]),y[a==arm]);model.append(clf)
    return [np.column_stack([model[arm].predict_proba(scaler.transform(q))[:,1] for arm in range(3)]) for q in queries]
def t_hgb(x,y,a,queries,seed):
    models=[]
    for arm in range(3):
        model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=11,
            max_depth=4,learning_rate=.05,l2_regularization=15.,
            min_samples_leaf=80,early_stopping=False,random_state=seed+arm)
        model.fit(x[a==arm],y[a==arm]);models.append(model)
    return [np.column_stack([m.predict_proba(q)[:,1] for m in models]) for q in queries]
def s_hgb(x,y,a,queries,seed):
    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,
        max_depth=5,learning_rate=.05,l2_regularization=15.,
        min_samples_leaf=80,early_stopping=False,random_state=seed)
    model.fit(np.column_stack((x,np.eye(3)[a])),y)
    return [np.column_stack([model.predict_proba(
        np.column_stack((q,np.eye(3)[np.full(len(q),arm,int)])))[:,1]
        for arm in range(3)]) for q in queries]
def t_forest(x,y,a,queries,seed):
    models=[]
    for arm in range(3):
        model=RandomForestClassifier(n_estimators=120,min_samples_leaf=70,
            max_features=.8,n_jobs=2,random_state=seed+arm)
        model.fit(x[a==arm],y[a==arm]);models.append(model)
    return [np.column_stack([m.predict_proba(q)[:,1] for m in models]) for q in queries]
def dr_pseudoutcome(x,y,a,queries,seed):
    # T-logit nuisance estimated on 50% split only, pseudo-outcome trees
    # fit on a disjoint 50%; prevents in-sample residual bias.
    tfit,tpseudo=train_test_split(np.arange(len(x)),test_size=.5,
                                 stratify=a,random_state=seed+333)
    nuisance=t_logit(x[tfit],y[tfit],a[tfit],[x[tpseudo]],seed)[0]
    ma=nuisance[np.arange(len(tpseudo)),a[tpseudo]]
    fits=[]
    for arm in range(3):
        z=nuisance[:,arm]+3*(a[tpseudo]==arm)*(y[tpseudo]-ma)
        learner=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=13,
              max_depth=4,learning_rate=.035,min_samples_leaf=120,
              l2_regularization=25.,early_stopping=False,random_state=seed+arm)
        learner.fit(x[tpseudo],z);fits.append(learner)
    return [np.column_stack([np.clip(m.predict(q),.005,.995) for m in fits])
            for q in queries]
MODEL_FUNCTIONS={
    'S_logistic_interactions':s_logit,
    'T_logistic':t_logit,
    'S_boosted':s_hgb,
    'T_boosted':t_hgb,
    'T_random_forest':t_forest,
    'DR_pseudo_boosted':dr_pseudoutcome
}
def policies(mu,cost):
    pay=mu.copy()
    pay[:,1:]-=cost
    return np.argmax(pay,axis=1)
def dr_unit_values(a,y,pi,m_evaluation,cost,prop=1/3):
    chosen=m_evaluation[np.arange(len(a)),pi]
    observed=m_evaluation[np.arange(len(a)),a]
    score=chosen + (a==pi)/prop*(y-observed)-cost*(pi!=0)
    return np.asarray(score,float)
def bootstrap_diff(scores,base,seed,reps=800):
    vals=np.asarray(scores)-np.asarray(base);n=len(vals)
    rg=np.random.default_rng(seed)
    # sample rows with replacement preserving paired policy scores
    draws=np.array([np.mean(vals[rg.integers(0,n,size=n)]) for _ in range(reps)])
    return {'mean':float(vals.mean()),'SE':float(vals.std(ddof=1)/np.sqrt(n)),
        'CI95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]}
def split_run(x,y,a,seed,bootstrap=400):
    indices=np.arange(len(a))
    train_ix,tmp_ix=train_test_split(indices,test_size=.4,stratify=a,
                                    random_state=seed)
    val_ix,test_ix=train_test_split(tmp_ix,test_size=.5,stratify=a[tmp_ix],
                                   random_state=seed+13)
    xt,at,yt=x[train_ix],a[train_ix],y[train_ix]
    xv,av,yv=x[val_ix],a[val_ix],y[val_ix]
    xh,ah,yh=x[test_ix],a[test_ix],y[test_ix]
    # independent holdout data; nuisance training restricted to train.
    eval_m_val,eval_m_test=t_hgb(xt,yt,at,[xv,xh],seed+11919)
    estimates={}
    for label,fit in MODEL_FUNCTIONS.items():
        estimates[label]=fit(xt,yt,at,[xv,xh],seed)
    # Fixed policies supplied without fitting, strongest trivial controls.
    candidate={}
    for arm,name in enumerate(ARMS):
        candidate['always_'+name]=(
            np.full(len(xv),arm,int),np.full(len(xh),arm,int))
    for label,(mv,mh) in estimates.items():
        for cost in COSTS:
            candidate[label+f'_cost{cost}']=(policies(mv,cost),
                                            policies(mh,cost))
    # EVALUATION policy choice uses fixed PRIMARY_COST; different costs
    # are separately disclosed sensitivity policies, not tuned on test.
    selected_candidates={k:v for k,v in candidate.items()
        if (k.startswith('always_') or k.endswith(f'_cost{PRIMARY_COST}'))}
    validation={k:float(dr_unit_values(av,yv,v[0],eval_m_val,PRIMARY_COST).mean())
                for k,v in selected_candidates.items()}
    selected=max(validation,key=validation.get)
    selected_pi=selected_candidates[selected][1]
    test_scores={k:dr_unit_values(ah,yh,v[1],eval_m_test,PRIMARY_COST)
                 for k,v in selected_candidates.items()}
    chosen_score=test_scores[selected]
    # Compare to predeclared sensible constant reference Mens Email.
    baseline=test_scores['always_Mens E-Mail']
    test_summary={name:float(s.mean()) for name,s in test_scores.items()}
    result={'seed':seed,'split_sizes':[len(train_ix),len(val_ix),len(test_ix)],
            'val_choice':selected,'validation_policy_values':validation,
            'test_policy_values':test_summary,
            'test_selected_minus_mens':bootstrap_diff(chosen_score,baseline,seed+44,bootstrap),
            'test_test_arm_counts':np.bincount(ah,minlength=3).tolist(),
            'test_send_fraction':float(np.mean(selected_pi!=0))}
    if bootstrap:
        result['test_paired_vs_mens']={
          name:bootstrap_diff(v,baseline,seed+113+ii,bootstrap)
          for ii,(name,v) in enumerate(test_scores.items())}
    return result
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--splits',type=int,default=5)
    parser.add_argument('--bootstrap',type=int,default=800)
    parser.add_argument('--output',default='research/cycle37_hillstrom_policy_results.json')
    args=parser.parse_args();timer=time.perf_counter()
    x,y,a,names,data=load()
    arm_rates={ARM:{'n':int((a==i).sum()),
                    'visit_rate':float(y[a==i].mean())}
               for i,ARM in enumerate(ARMS)}
    result={'protocol':{'dataset':'Hillstrom MineThatData 2008 randomized 3-arm',
       'source_url':'https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv',
       'source_sha256':hashlib.sha256(open(DATA,'rb').read()).hexdigest(),
       'n_total':len(a),'features':names,'outcome':'visit, binary',
       'arms':ARMS,'known_randomization_propensity':[1/3]*3,
       'train_val_test':'60/20/20, independent split by customer',
       'selection':'validation DR value at cost 0.012; final test once',
       'reported_secondary_costs':COSTS,
       'cost_interpretation':'hypothetical probability-scale penalty per message, not measured dollar cost',
       'bootstrap':'paired evaluation-customer resampling; training uncertainty not included',
       'split_repetitions':'nonindependent reuses of same finite RCT; sensitivity only',
       'counterfactual_observation':'only one realized arm per customer; no unit-level truth',
       'strong_baselines':list(MODEL_FUNCTIONS)+['constant policies'],
       'no_new_algorithm_claim':True},'arm_rates':arm_rates,'runs':[]}
    for i in range(args.splits):
        seed=37011026+1711*i
        run=split_run(x,y,a,seed,args.bootstrap)
        result['runs'].append(run)
        print('RUN',i+1,'/',args.splits,'selected',run['val_choice'],
              'test_minus_mens',run['test_selected_minus_mens'],
              'elapsed',round(time.perf_counter()-timer,1),flush=True)
    result['elapsed_seconds']=time.perf_counter()-timer
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',args.output,round(result['elapsed_seconds'],1),flush=True)
if __name__=='__main__':main()
