"""Cycle 39: independent audit of randomized policy-value estimator.

Verify the AIPW/DR estimator against a synthetic RCT with known true
counterfactual probabilities, validate linear mixture identities, and
check original Hillstrom three-arm sample ratio. Not a new method.
"""
import json,time
import numpy as np
from scipy.stats import chi2
from cycle37_hillstrom_policy import load,dr_unit_values,ARMS,FEATURES
from cycle38_budgeted_uplift import dr_random_scores,dr_deterministic

def check_simulation():
    rng=np.random.default_rng(39111026)
    trials=120;n=12000
    errors=[];errors_ips=[]
    for trial in range(trials):
        x=rng.uniform(size=n)
        a=rng.integers(0,3,size=n)
        mu=np.column_stack((.04+.16*x,.08+.23*x,.06+.09*x))
        y=(rng.random(n)<mu[np.arange(n),a]).astype(float)
        policy=np.where(x>.5,1,0)
        true=float(np.mean(mu[np.arange(n),policy]))
        # deliberately misspecified outcome regression: no x dependence
        miss=np.tile(np.array([.12,.16,.11]),(n,1))
        dr=float(dr_unit_values(a,y,policy,miss,0).mean())
        ips=float((3*(a==policy)*y).mean())
        errors.append(dr-true);errors_ips.append(ips-true)
        # stochastic q/send budget must be convex mixture at unit level
        q=.2
        actual=dr_random_scores(a,y,miss,q)
        mix=(q*dr_deterministic(a,y,miss,np.ones(n,int))+
             (1-q)*dr_deterministic(a,y,miss,np.zeros(n,int)))
        assert np.allclose(actual,mix,atol=1e-12)
    return {'trials':trials,'n_per_trial':n,
        'misspecified_nuisance_values':[.12,.16,.11],
        'mean_dr_error':float(np.mean(errors)),
        'SE_mean_dr_error':float(np.std(errors,ddof=1)/np.sqrt(trials)),
        'mean_ips_error':float(np.mean(errors_ips)),
        'SE_mean_ips_error':float(np.std(errors_ips,ddof=1)/np.sqrt(trials))}

def check_hillstrom():
    x,y,a,names,_=load()
    count=np.bincount(a,minlength=3)
    chisq=float(np.sum((count-len(a)/3)**2/(len(a)/3)))
    result={'arm_counts':count.tolist(),
      'chi_squared_vs_equal_three_arm_randomization':chisq,
      'p_sample_ratio':float(chi2.sf(chisq,2)),
      'feature_names':names,
      'post_treatment_excluded':all(n not in names for n in
                                   ['visit','conversion','spend','segment'])}
    assert result['p_sample_ratio']>.05
    assert result['post_treatment_excluded']
    return result
def main():
    start=time.perf_counter()
    out={'simulated_RCT_estimator_audit':check_simulation(),
       'real_Hillstrom_design_check':check_hillstrom(),
       'runtime_seconds':time.perf_counter()-start,
       'not_novel':True}
    with open('research/cycle39_evaluator_sanity.json','w') as f:json.dump(out,f,indent=2)
    print('AUDIT',out,flush=True)
if __name__=='__main__':main()
