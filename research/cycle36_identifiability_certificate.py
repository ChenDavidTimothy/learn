"""Cycle 36: generic finite-model decision-identifiability certificate.

Given observational response kernels Q_e(m), action rewards U_a(m), group
models indistinguishable under *all* accessible experiments. An ambiguity
class is unsafe if its members have different optimal actions. An additional
experiment breaks a class only if response kernels separate those members.
This is elementary statistical identification and automata/causal equivalence,
not new theory or invention of novel probes.
"""
import json,time
import numpy as np
from cycle34_reward_anchor import generate
REFERENCE_ERROR=.08

def classes(q,decimals=10):
    groups={}
    for i,row in enumerate(q):
        groups.setdefault(tuple(np.round(row,decimals)),[]).append(i)
    return list(groups.values())
def ambiguous_pairs(q,mu):
    acts=np.argmax(mu,axis=1)
    pairs=[]
    for group in classes(q):
        for z,i in enumerate(group):
            for j in group[z+1:]:
                if acts[i]!=acts[j]:pairs.append((i,j))
    return pairs
def coverage(qbase,extras,mu):
    original=set(ambiguous_pairs(qbase,mu))
    report={'hypothesis_count':int(len(mu)),
        'sensor_count':int(qbase.shape[1]),
        'equivalence_classes':int(len(classes(qbase))),
        'decision_ambiguous_pairs':len(original),
        'extra_tests':{}}
    for name,extra in extras.items():
        combined=np.column_stack((qbase,extra))
        new=set(ambiguous_pairs(combined,mu))
        report['extra_tests'][name]={'ambiguous_pairs_remaining':len(new),
            'pairs_separated':len(original-new)}
    return report
def world(seed):
    mu,base,_=generate(seed)
    qm=np.vstack((base,1-base))
    um=np.vstack((mu,mu))
    ref=np.r_[np.full(len(mu),REFERENCE_ERROR),
              np.full(len(mu),1-REFERENCE_ERROR)][:,None]
    dummy=np.full((len(um),1),.5)
    return um,qm,{'reference':ref,'reward_action0':um[:,[0]],
                  'reward_action1':um[:,[1]],'unrelated_dummy':dummy}
def validate():
    results=[]
    for i in range(100):
        reward,probe,extra=world(36111026+971*i)
        r=coverage(probe,extra,reward)
        assert r['hypothesis_count']==128
        assert r['equivalence_classes']==64,r
        assert r['decision_ambiguous_pairs']==64,r
        assert r['extra_tests']['reference']['ambiguous_pairs_remaining']==0,r
        assert r['extra_tests']['unrelated_dummy']['ambiguous_pairs_remaining']==64,r
        assert r['extra_tests']['reward_action0']['ambiguous_pairs_remaining']==0,r
        assert r['extra_tests']['reward_action1']['ambiguous_pairs_remaining']==0,r
        results.append(r)
    return results

def main():
    now=time.perf_counter()
    results=validate()
    outcome={'scope':'finite known candidate classes and known observation response tables',
      'theory':'if models have identical distributions for every allowed experiment, adaptive sequences cannot distinguish them; a difference in optimal action implies irreducible decision ambiguity',
      'not_novel':True,'trials':100,'first_instance':results[0],
      'all_instances_passed':len(results),
      'known_limitation':'rounding tolerance, finite discrete models, structural not finite-sample identification; assumes exact kernels',
      'runtime_seconds':time.perf_counter()-now}
    with open('research/cycle36_identifiability_certificate.json','w') as f:
        json.dump(outcome,f,indent=2)
    print('PASS',outcome['all_instances_passed'],'instances',
          outcome['first_instance'],flush=True)
if __name__=='__main__':main()
