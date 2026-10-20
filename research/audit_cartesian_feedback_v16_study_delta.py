#!/usr/bin/env python3
"""Pre-outcome source delta / actual-belief numerical crosscheck; no plant run."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import cartesian_feedback_v16_study as study
import cartesian_feedback_v16 as pilot
import cartesian_reuse_v16_refinement as oldbase
from audit_cartesian_feedback_v16_quadrature import integrate

def defs(path):
 t=ast.parse(Path(path).read_text());return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def main():
 s=defs(study.__file__);p=defs(pilot.__file__);equivalence={n:s[n]==p[n] for n in ['transition','Belief','future_qp','plan','nominal_witnesses']};assert all(equivalence.values())
 assert defs(study.base.__file__)['box_qp']==defs(oldbase.__file__)['box_qp']
 r=json.loads(Path('research/cartesian_feedback_v16_pilot_results.json').read_text());checks=[]
 # First-rest selected and near-selected integrals, plus actual previous false estimates.
 fixtures=[(0,0,20),(0,0,19),(4,0,21),(20,0,3),(20,0,10),(6,5,9),(39,1,15)]
 with threadpool_limits(limits=1):
  for ei,pi,ci in fixtures:
   e=r['episodes'][ei];p=e['phases'][pi];b=study.Belief(**p['belief_before']);u=np.array(p['design']['candidates'][ci]);goal=np.array(p['goal'])
   value,record=study.expected_first(b,goal,u)
   # Independently orchestrated tighter quadrant integration, old audited core.
   old=pilot.Belief(**p['belief_before']);reference=integrate(old,goal,u,'gk21',1e-10,1e-9,True)
   diff=abs(value-reference['value']);assert diff<2e-8,(ei,pi,ci,diff)
   checks.append(dict(episode=ei,phase=pi,candidate=ci,production_value=value,reference_value=reference['value'],absolute_difference=diff,production_record=record,reference_record=reference))
  # Heldout one/two-read NLL and isotropic transformation use exactly same pilot fit.
  parent=json.loads(Path('research/cartesian_reuse_v16_gk21_results.json').read_text())
  with np.load('research/cartesian_reuse_v16_gk21_data.npz') as z:data={k:z[k] for k in z.files}
  priors,_,_=study.learn_priors(parent,data);g=priors['gaussian'];iso=priors['isotropic'];C=g.covs[0,4:,4:]
  assert np.array_equal(g.means,iso.means) and abs(np.trace(C)-np.trace(iso.covs[0,4:,4:]))<1e-12
  assert np.max(abs(iso.covs[0,4:,4:]-np.trace(C)/4*np.eye(4)))<1e-15
  X=data['testing_exposures'];Y=(data['testing_observed'][:,1:]-study.base.P0)/study.T**2
  m=g.means[:,4:].reshape(-1,2,2);cg=g.covs[:,4:,4:]
  nll=study.predictive_mixture_nll(X,Y,m,g.weights,cg);reference=study.base.gaussian_nll(X,Y,m[0],cg[0]);assert abs(nll-reference)<1e-12
  oldnll=pilot.predictive_mixture_nll(X,Y,m,g.weights,cg);assert abs(nll-oldnll)<1e-12
 result=dict(scope='PRE-OUTCOME delta audit; no fresh physical acquisition/control outcomes generated or inspected',study_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),core_ast_equivalence=equivalence,base_box_qp_ast_unchanged=True,actual_integral_checks=checks,isotropic_mean_unchanged=True,isotropic_trace_preserved=True,general_gaussian_nll_crosscheck_error=abs(nll-reference))
 Path('research/audit_cartesian_feedback_v16_study_delta_results.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PASS',len(checks),'actual integrals; maxdifference',max(c['absolute_difference'] for c in checks))
if __name__=='__main__':main()
