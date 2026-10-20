#!/usr/bin/env python3
"""Independent horizontal point-mass two-link rest-pulse mechanics check.
No learner or canonical simulator imported; no learning claim.
"""
import json
import math
import numpy as np
from scipy.integrate import solve_ivp


def inertia(angle):
    c=math.cos(angle)
    return np.array([[3+2*c,1+c],[1+c,1.]])


def derivative(_,state,torque):
    q1,q2,v1,v2=state
    h=math.sin(q2)*np.array([-(2*v1*v2+v2*v2),v1*v1])
    acceleration=np.linalg.solve(inertia(q2),torque-h)
    return np.r_[v1,v2,acceleration]


def fourth_derivative(angle,torque):
    M=inertia(angle);a=np.linalg.solve(M,torque);s=math.sin(angle)
    Mprime=-s*np.array([[2.,1.],[1.,0.]])
    coriolis_on_a=s*np.array([-(2*a[0]*a[1]+a[1]**2),a[0]**2])
    return -np.linalg.solve(M,a[1]*Mprime@a+2*coriolis_on_a)


def main():
    torque=np.array([0.,1.]);rows=[]
    for angle in (0.,.7,math.pi/2,math.pi):
        M=inertia(angle);a=np.linalg.solve(M,torque);b=fourth_derivative(angle,torque)
        points=[]
        for duration in (.4,.2,.1,.05,.025):
            initial=np.array([.37,angle,0.,0.])
            end=solve_ivp(lambda t,z:derivative(t,z,torque),(0,duration),initial,
                method='DOP853',rtol=2e-13,atol=1e-15)
            assert end.success
            state=end.y[:,-1];delta=state[:2]-initial[:2]
            velocity=state[2:];finalM=inertia(state[1]);kinetic=.5*velocity@finalM@velocity
            momentum=float((finalM@velocity)[0])
            approximation=.5*duration**2*a
            remainder=delta-approximation
            points.append({'duration':duration,'displacement':delta.tolist(),
                'quadratic_remainder_norm':float(np.linalg.norm(remainder)),
                'quartic_coefficient_error_norm':float(np.linalg.norm(remainder/duration**4-b/24)),
                'energy_minus_work':float(kinetic-torque@delta),
                'unforced_cyclic_momentum':momentum,'nfev':end.nfev})
        for i in range(len(points)-1):
            points[i]['quadratic_remainder_halving_ratio']=points[i]['quadratic_remainder_norm']/points[i+1]['quadratic_remainder_norm']
        rows.append({'q2':angle,'inertia':M.tolist(),'initial_acceleration':a.tolist(),
                     'fourth_time_derivative':b.tolist(),'pulses':points})
    assert np.allclose(fourth_derivative(math.pi/2,torque),[1.75,-3.],atol=1e-14)
    assert max(abs(r['energy_minus_work']) for c in rows for r in c['pulses'])<1e-10
    assert max(abs(r['unforced_cyclic_momentum']) for c in rows for r in c['pulses'])<1e-10
    print(json.dumps({'status':'independent mechanics/convergence check; no learning experiment',
        'geometry':'unit endpoint masses on unit massless links, horizontal, relative elbow angle',
        'equations':'M(q)qdd+h(q,v)=constant torque; no gravity/friction/actuator lag',
        'method':'DOP853 rtol2e-13 atol1e-15','cases':rows},indent=2,allow_nan=False))


if __name__=='__main__':main()
