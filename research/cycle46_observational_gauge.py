"""Cycle 46: exact action-conditioned observational equivalence certificate.

World A: binary controlled Markov hidden X with original T_a,
          sensor output O=X XOR 1 XOR iid Bernoulli(eps).
World B: hidden X'=1-X, transition T'_a=C T_a C and sensor
          O=X' XOR iid Bernoulli(eps).
With uniform initial hidden prior, ALL finite action/observation trace
distributions coincide under A and B. The terminal correct actions are
opposites because the reward is 1[action==hidden state].

This is an explicit classical gauge/bisimulation counterexample for
nonstationary model criticism, NOT a new theorem. Learners given only
action/observation history cannot identify A versus B. Corrective
reward observations or restrictions on transition drift are required.
"""
import json, itertools, time
import numpy as np
from cycle45_sensor_self_calibration import DYNAMICS,NOISE
BASE=DYNAMICS['asymmetric']
def transition_matrix(dyn,action):
    p01,p10=dyn[action]
    return np.array([[1-p01,p01],[p10,1-p10]])
def obs_likelihood(o,invert):
    return np.array([(1-NOISE) if o==(x^invert) else NOISE
                     for x in (0,1)])
def probability(actions,observations,dyn,invert):
    n=len(observations)
    assert len(actions)==n-1
    p=np.full(2,.5)
    for t in range(n):
        p*=obs_likelihood(observations[t],invert)
        if t<n-1:p=p@transition_matrix(dyn,actions[t])
    return float(p.sum())
def verify():
    switched=BASE[::-1]
    C=np.array([[0,1],[1,0]])
    for a in (0,1):
        lhs=C@transition_matrix(BASE,a)@C
        rhs=transition_matrix(switched,a)
        assert np.allclose(lhs,rhs,atol=1e-16),(a,lhs,rhs)
    max_error=0;tested=0
    for T in range(1,8):
        for actions in itertools.product((0,1),repeat=T-1):
            for obs in itertools.product((0,1),repeat=T):
                pa=probability(actions,obs,BASE,1)
                pb=probability(actions,obs,switched,0)
                max_error=max(max_error,abs(pa-pb))
                assert abs(pa-pb)<1e-14,(T,actions,obs,pa,pb)
                tested+=1
    # For every coupled realization X' = 1-X, reward signs are opposite
    # for any single fixed terminal action. Sum exactly one.
    for x in (0,1):
        for action in (0,1):
            assert int(action==x)+int(action==(1-x))==1
    return {'verified_action_observation_strings':tested,
            'max_absolute_likelihood_difference':max_error,
            'max_checked_history_length':7,
            'transition_conjugacy':True,
            'opposite_optimal_actions':True,
            'worst_case_terminal_reward_upper_bound':.5,
            'assumptions':'uniform complement-invariant prior; two binary state labels; observable action history; symmetric flip emission noise; no direct reward feedback during adaptation'}
def main():
    start=time.perf_counter()
    result=verify()
    result['runtime_seconds']=time.perf_counter()-start
    result['original_transition']=BASE.tolist()
    result['swapped_transition']=BASE[::-1].tolist()
    result['novelty']='explicit counterexample using established observational equivalence, not novel theorem'
    with open('research/cycle46_action_observation_equivalence.json','w') as f:
        json.dump(result,f,indent=2)
    print('PASS',result,flush=True)
if __name__=='__main__':main()
