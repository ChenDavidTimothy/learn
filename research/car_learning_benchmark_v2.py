#!/usr/bin/env python3
"""V2: finite-grammar growth, nonidentifiability, errors-in-variables, interventions.

All computation uses the existing project-local .venv. No external datasets.
Run: .venv/bin/python research/car_learning_benchmark_v2.py > research/car_learning_benchmark_v2_results.json
"""
from __future__ import annotations
import json
import numpy as np

BASE_SEED = 20261020

def rmse(y, yh):
    return float(np.sqrt(np.mean((y-yh)**2)))

def make_data(seed, n, case, shift=0., wide=False, confound=False):
    rng = np.random.default_rng(seed)
    v = rng.uniform(-2.4 if wide else -1.3, 2.4 if wide else 1.3, n)
    x_prev = rng.uniform(-5., 5., n) + shift
    x = x_prev + v
    a = rng.uniform(-1.4 if wide else -1., 1.4 if wide else 1., n)
    p = 2*v if confound else rng.uniform(-1.4 if wide else -1., 1.4 if wide else 1., n)
    displacement = .85*v + 1.1*a
    if case in ('nonlinear','mixed'):
        displacement -= .14*v*np.abs(v)
    if case in ('lag','mixed','confounded'):
        displacement += .28*p
    if case == 'noise':
        displacement += rng.normal(0., .12, n)
    return dict(x=x, xp=x_prev, a=a, p=p, y=displacement)

def grammar(data):
    # Operator vocabulary (sum, difference, abs, product); not a given drag expression.
    x, xp, p = data['x'], data['xp'], data['p']
    primitives = {'x':x, 'x_prev':xp, 'a_prev':p}
    result={}
    cost={}
    def add(name, array, complexity):
        if name not in result:
            result[name]=array
            cost[name]=complexity
    for name, arr in primitives.items():
        add(name, arr, 1)
    pair_sources=[('x',x),('x_prev',xp)]
    for i in range(len(pair_sources)):
        for j in range(i+1,len(pair_sources)):
            n1, y1=pair_sources[i]
            n2, y2=pair_sources[j]
            add('('+n1+'-'+n2+')',y1-y2,2)
            add('('+n1+'+'+n2+')',y1+y2,2)
    level1=list(result.items())
    for name, q in level1:
        add('abs('+name+')',np.abs(q),cost[name]+1)
        add('('+name+'*'+name+')',q*q,cost[name]+1)
        add('('+name+'*abs('+name+'))',q*np.abs(q),cost[name]+2)
        add('('+name+'*'+name+'*'+name+')',q*q*q,cost[name]+2)
        add('('+name+'*a_prev)',q*p,cost[name]+1)
        add('('+name+'*a)',q*data['a'],cost[name]+1)
    return result,cost

def regress(data, feature_list, names):
    X=np.column_stack([np.ones(len(data['y'])),data['a']]+[feature_list[k] for k in names])
    return X

def fit_coef(X,y):
    return np.linalg.lstsq(X,y,rcond=None)[0]

def adaptive(train,val,max_terms=3):
    ft,complexity=grammar(train)
    fv,_=grammar(val)
    selected=[]
    trace=[]
    remaining=list(ft)
    for _ in range(max_terms):
        coef=fit_coef(regress(train,ft,selected),train['y'])
        base=rmse(val['y'],regress(val,fv,selected)@coef)
        scores=[]
        for name in remaining:
            names=selected+[name]
            coef=fit_coef(regress(train,ft,names),train['y'])
            score=rmse(val['y'],regress(val,fv,names)@coef)
            scores.append((score+complexity[name]*1e-5, score,complexity[name],name))
        scores.sort()
        _,best,_,name=scores[0]
        improvement=base-best
        if improvement < max(.008,.04*base):
            break
        selected.append(name)
        remaining.remove(name)
        trace.append({'added':name,'validation_before':base,
                      'validation_after':best,'improvement':improvement})
    coef=fit_coef(regress(train,ft,selected),train['y'])
    return selected,trace,coef

def representation_cases():
    cases={}
    for i,case in enumerate(('linear','nonlinear','lag','mixed','noise','confounded')):
        isconf=(case=='confounded')
        train=make_data(BASE_SEED+100*i,500,case,confound=isconf)
        val=make_data(BASE_SEED+100*i+1,500,case,shift=20,confound=isconf)
        test=make_data(BASE_SEED+100*i+2,1600,case,shift=100,wide=True)
        selected,trace,coef=adaptive(train,val)
        ft,_=grammar(train)
        ftest,_=grammar(test)
        test_adaptive=rmse(test['y'],regress(test,ftest,selected)@coef)
        def baseline(keys):
            features={'x':lambda d:d['x'],
                      'd':lambda d:d['x']-d['xp'],
                      'a_prev':lambda d:d['p'],
                      'd_abs_d':lambda d:(d['x']-d['xp'])*np.abs(d['x']-d['xp'])}
            def mat(d):
                return np.column_stack([np.ones(len(d['y'])),d['a']]+[features[k](d) for k in keys])
            c=fit_coef(mat(train),train['y'])
            return {'test_rmse':rmse(test['y'],mat(test)@c),
                    'parameters':c.tolist()}
        cases[case]={
            'data':{'train':500,'validation':500,'test':1600,'seed_train':BASE_SEED+100*i,
                    'seed_validation':BASE_SEED+100*i+1,'seed_test':BASE_SEED+100*i+2,
                    'train_velocity_interval':[-1.3,1.3],'test_velocity_interval':[-2.4,2.4],
                    'val_shift_x':20,'test_shift_x':100, 'train_confound_p_equals_2v':isconf},
            'adaptive_features':selected,'trace':trace,
            'adaptive_coefficients':coef.tolist(),
            'adaptive_test_rmse':test_adaptive,
            'position_only':baseline(['x'])['test_rmse'],
            'fixed_velocity':baseline(['d'])['test_rmse'],
            'fixed_linear_lag':baseline(['d','a_prev'])['test_rmse'],
            'maximal_library':baseline(['d','a_prev','d_abs_d'])['test_rmse'],
        }
        if isconf:
            c=fit_coef(np.column_stack([train['x']-train['xp'],train['p']]),train['y']-1.1*train['a'])
            cases[case]['full_design_rank']=int(np.linalg.matrix_rank(
                np.column_stack([train['x']-train['xp'],train['p']])))
            cases[case]['linear_parameter_fit_v_and_p']=c.tolist()
    assert cases['linear']['adaptive_test_rmse'] < 1e-10
    assert cases['nonlinear']['adaptive_test_rmse'] < 1e-10
    assert cases['lag']['adaptive_test_rmse'] < 1e-10
    assert cases['mixed']['adaptive_test_rmse'] < 1e-10
    assert cases['noise']['adaptive_features'] == ['(x-x_prev)'],cases['noise']['adaptive_features']
    assert cases['confounded']['full_design_rank']==1
    assert cases['confounded']['adaptive_test_rmse']>.1
    return cases

def sensor_noise_trial(seed,n,sensor_sd):
    rng=np.random.default_rng(seed)
    a=rng.normal(0.,.65,n+1)
    rho,beta=.8,1.3
    v=np.zeros(n+2)
    x=np.zeros(n+2)
    for i in range(n+1):
        v[i+1]=rho*v[i]+beta*a[i]
        x[i+1]=x[i]+v[i+1]
    observed=x+rng.normal(0.,sensor_sd,n+2)
    d=np.diff(observed)
    # response d_(t+1), regressors d_t and a_t, instrument a_(t-1).
    X=np.column_stack((d[1:-1],a[2:]))
    Z=np.column_stack((a[1:-1],a[2:]))
    y=d[2:]
    ols=fit_coef(X,y)
    iv=np.linalg.solve(Z.T@X,Z.T@y)
    true_var=float(np.var(v[1:-1]))
    expected_ols_rho=(rho*true_var-sensor_sd**2)/(true_var+2*sensor_sd**2)
    return ols,iv,expected_ols_rho

def noisy_observation_cases():
    results={}
    for sd in [0.,.5,1.]:
        estimates=[]
        for rep in range(35):
            ols,iv,formula=sensor_noise_trial(BASE_SEED+300+rep,6000,sd)
            estimates.append([*ols,*iv,formula])
        q=np.array(estimates)
        results[str(sd)]={
            'n_trajectories':35,'transitions_per_trajectory':6000,
            'ols_mean_rho_beta':q[:,0:2].mean(axis=0).tolist(),
            'iv_mean_rho_beta':q[:,2:4].mean(axis=0).tolist(),
            'ols_sd_rho_beta':q[:,0:2].std(axis=0,ddof=1).tolist(),
            'iv_sd_rho_beta':q[:,2:4].std(axis=0,ddof=1).tolist(),
            'expected_ols_rho_approx_mean':float(q[:,4].mean()),
        }
    assert abs(results['0.5']['ols_mean_rho_beta'][0]-.8)>.1
    assert abs(results['0.5']['iv_mean_rho_beta'][0]-.8)<.03
    return results

def exploration_one(seed,policy,n=12):
    rng=np.random.default_rng(seed)
    v=0.
    gram=np.zeros((2,2))
    X=[];y=[]
    actions=np.array([-1.,0.,1.])
    for _ in range(n):
        if policy=='zero':
            a=0.
        elif policy=='random':
            a=float(rng.choice(actions))
        else:
            # D-opt: maximize log det of regressor information Gram.
            shuffled=rng.permutation(actions)
            scores=[np.linalg.slogdet(gram+np.outer([v,a],[v,a])+
                          np.eye(2)*.01)[1] for a in shuffled]
            a=float(shuffled[int(np.argmax(scores))])
        next_v=.8*v+1.3*a+rng.normal(0.,.12)
        f=np.array([v,a])
        X.append(f);y.append(next_v)
        gram+=np.outer(f,f)
        v=next_v
    coef=fit_coef(np.array(X),np.array(y))
    return float(np.linalg.norm(coef-[.8,1.3])),float(np.linalg.eigvalsh(gram)[0]),coef

def intervention_cases():
    out={}
    for policy in ('zero','random','d_opt'):
        runs=np.array([exploration_one(BASE_SEED+1000+k,policy)[:2]
                       for k in range(200)])
        out[policy]={'episodes':200,'steps_per_episode':12,
                     'parameter_l2_error_median':float(np.median(runs[:,0])),
                     'parameter_l2_error_mean':float(np.mean(runs[:,0])),
                     'information_min_eigenvalue_median':float(np.median(runs[:,1]))}
    assert out['zero']['information_min_eigenvalue_median']<1e-12
    return out

def main():
    d={'metadata':{'version':2,'seed':BASE_SEED,'position_observation_only':True,
                  'regression':'ordinary least squares unless instrumented',
                  'train_test_data':'independent randomized intervention samples'},
       'representation_growth':representation_cases(),
       'observation_noise_identification':noisy_observation_cases(),
       'active_exploration':intervention_cases()}
    print(json.dumps(d,indent=2))

if __name__ == '__main__':
    main()
