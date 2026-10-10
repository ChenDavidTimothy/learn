"""Exact finite-instance tests of the complementary-information obstruction.

For uniformly distributed independent bits and XOR decision target,
observing any one noisy bit retains an exactly uniform optimal-action
posterior, so all one-step Bayes value improvements are zero. Two
appropriately chosen observations can improve decisions. This is
elementary parity symmetry, not a novel theorem.
"""
import json
import numpy as np
from cycle32_decision_design import sample_world,expected_next,model_value,optimal_probe
def main():
    results=[]
    for i in range(100):
        mu,lik,truth=sample_world(36511026+877*i,'xor')
        prior=np.full(len(mu),1/len(mu))
        immediate=expected_next(prior,lik,mu)-model_value(prior,mu)
        improvement,j=optimal_probe(prior,lik,mu,2,.012)
        net=improvement-model_value(prior,mu)
        assert np.max(np.abs(immediate))<1e-12,(i,immediate)
        assert j is not None and net>0.0,(i,j,net)
        results.append({'max_immediate_EVI':float(immediate.max()),
                        'two_step_net_gain':float(net)})
    out={'instances':len(results),'max_abs_immediate_value':float(max(
          abs(r['max_immediate_EVI']) for r in results)),
          'minimum_positive_two_step_net_gain':float(min(
          r['two_step_net_gain'] for r in results)),
          'theory':'uniform XOR: each individual sensor independent of optimal action; a jointly informative two-probe sequence exists.',
          'not_novel':True}
    with open('research/cycle32_xor_validation.json','w') as f:json.dump(out,f,indent=2)
    print('PASS',out,flush=True)
if __name__=='__main__':main()
