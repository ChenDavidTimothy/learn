#!/usr/bin/env python3
"""Independent polar-Lagrangian derivation check; not a learning experiment."""
from pathlib import Path
import json,hashlib
import numpy as np
import coupled_context_v14_pilot as implementation

def rotation(theta):
 return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def matched_state(r,theta,vr,vt,sign):
 phi=2*np.arccos(r/2);d=np.sqrt(4-r*r);omega=vt/r
 q=np.array([theta-sign*phi/2,sign*phi]);v=np.array([omega+sign*vr/d,-2*sign*vr/d])
 return q,v

def endpoint(q):return np.array([np.cos(q[0])+np.cos(q.sum()),np.sin(q[0])+np.sin(q.sum())])
def jacobian(q):
 return np.array([[-np.sin(q[0])-np.sin(q.sum()),-np.sin(q.sum())],[np.cos(q[0])+np.cos(q.sum()),np.cos(q.sum())]])
def actual_acceleration(q,v,u):
 acc=implementation.rhs(0,np.r_[q,v],u)[2:]
 kinematic=-np.array([np.cos(q[0])*v[0]**2+np.cos(q.sum())*v.sum()**2,np.sin(q[0])*v[0]**2+np.sin(q.sum())*v.sum()**2])
 return jacobian(q)@acc+kinematic

def polar_acceleration(r,theta,vr,vt,u,sign):
 d=np.sqrt(4-r*r);omega=vt/r
 metric=np.array([[1+1/d**2,sign/d],[sign/d,1+r*r]])
 generalized=np.array([sign*(u[0]-2*u[1])/d,u[0]])
 drift=np.array([r*omega**2-r*vr**2/d**4,-2*r*vr*omega-sign*r*vr**2/d**3])
 rdd,tdd=np.linalg.solve(metric,generalized+drift)
 return rotation(theta)@np.array([rdd-r*omega**2,r*tdd+2*vr*omega])

def branch_difference_BminusA(r,theta,vr,vt,u):
 d=np.sqrt(4-r*r);N=4+4*r*r-r**4
 return rotation(theta)@(2*d/N*np.array([-r*r*u[0]+2*(1+r*r)*u[1]-2*vr*vt,vt*vt+r*r/d**2*vr*vr]))

def main():
 rng=np.random.default_rng(202610211701);errors=[];kinetic_errors=[];velocity_errors=[];minimum_tangent=1e10;records=[]
 for _ in range(128):
  r=rng.uniform(.4,1.8);theta=rng.uniform(-np.pi,np.pi);vr,vt=rng.uniform(-.4,.4,2);u=rng.uniform(-.5,.5,2);actual=[]
  for sign in (1,-1):
   q,v=matched_state(r,theta,vr,vt,sign);a=actual_acceleration(q,v,u);prediction=polar_acceleration(r,theta,vr,vt,u,sign)
   errors.append(float(np.max(abs(a-prediction))));actual.append(a);velocity_errors.append(float(np.max(abs(jacobian(q)@v-rotation(theta)@np.array([vr,vt])))))
   d=np.sqrt(4-r*r);omega=vt/r
   kinetic=.5*((1+r*r)*omega**2+2*sign/d*vr*omega+(1+1/d**2)*vr**2)
   kinetic_errors.append(abs(kinetic-.5*v@implementation.mass(q)@v))
  error=float(np.max(abs(actual[1]-actual[0]-branch_difference_BminusA(r,theta,vr,vt,u))));errors.append(error)
  tangent=(rotation(theta).T@(actual[1]-actual[0]))[1];minimum_tangent=min(minimum_tangent,float(tangent));assert tangent>0
 assert max(errors)<1e-11 and max(kinetic_errors)<1e-12 and max(velocity_errors)<1e-12
 # Actual zero-input short-past/future paths check quadratic separation; no noise/learner.
 for velocity in ([.1,0.],[0.,.1],[.1,.1],[0.,0.]):
  r=np.sqrt(2.);theta=.4;vr,vt=velocity;states=[matched_state(r,theta,vr,vt,s) for s in(1,-1)];rows=[]
  diff=branch_difference_BminusA(r,theta,vr,vt,np.zeros(2))
  for h in(.02,.01,.005):
   futures=[];pasts=[]
   for q,v in states:
    f,_=implementation.flow(q,v,np.zeros(2),h,tol=1e-13);past,_=implementation.flow(q,-v,np.zeros(2),h,tol=1e-13)
    futures.append(endpoint(f[:2]));pasts.append(endpoint(past[:2]))
   sep=futures[1]-futures[0];backsep=pasts[1]-pasts[0]
   rows.append(dict(h=h,actual_future_separation=sep.tolist(),actual_past_separation=backsep.tolist(),leading_separation=(.5*h*h*diff).tolist(),leading_error=float(np.linalg.norm(sep-.5*h*h*diff))))
  records.append(dict(r=r,theta=theta,cartesian_velocity_radial_tangential=velocity,zero_command_acceleration_difference=diff.tolist(),shortpath_checks=rows))
 out=dict(status='PASS: independent polar kinetic/EL derivation agrees with joint mechanics; no learner or milestone',audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),implementation_source_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),random_configs=128,maximum_acceleration_or_difference_error=max(errors),maximum_kinetic_error=max(kinetic_errors),maximum_matched_velocity_error=max(velocity_errors),minimum_tangent_difference_sampled=minimum_tangent,zero_command_path_checks=records,limitations=['Nonsingular interior0<r<2 only; no finite-noise identifiability or unknown-model learning guarantee.','Mechanics functions imported as implementation under test; polar kinetic/EL/difference formulas independently constructed.','Short paths are a small mechanics falsification instrument, no learning/control comparison.'])
 Path('research/audit_cartesian_moving_v17_theory_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','maximum_acceleration_or_difference_error','maximum_kinetic_error','maximum_matched_velocity_error')}))
if __name__=='__main__':main()
