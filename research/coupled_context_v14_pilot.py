#!/usr/bin/env python3
"""Unreleased mechanics/measurement pilot, NOT a learning comparison.

Continuous fully-actuated horizontal two-link point-mass arm, masses/lengths1.
Only noisy endpoint positions/commands would be ordinary learner observations.
Reference physics/state/response/energy are evaluator diagnostics here.
"""
import json,math,platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp

SEED=20261020+2100000
SENSOR_SD=1e-5


def mass(q):
    c=math.cos(float(q[1]));return np.array([[3+2*c,1+c],[1+c,1.]])


def rhs(t,z,u):
    q=z[:2];v=z[2:];s=math.sin(float(q[1]));v1,v2=v
    coriolis=np.array([-s*(2*v1*v2+v2*v2),s*v1*v1])
    return np.r_[v,np.linalg.solve(mass(q),u-coriolis)]


def flow(q,v,u,duration,tol=1e-11):
    z=np.r_[q,v];sol=solve_ivp(lambda t,z:rhs(t,z,u),(0,duration),z,method='DOP853',rtol=tol,atol=tol*.01)
    assert sol.success and np.all(np.isfinite(sol.y[:,-1]))
    return sol.y[:,-1],int(sol.nfev)


def kinetic(z):return float(.5*z[2:]@mass(z[:2])@z[2:])


def main():
    rng=np.random.default_rng(SEED);rows=[]
    for q2 in np.linspace(-math.pi,math.pi,17):
        for u in ((.6,0.),(0.,.6),(.6,-.6)):
            errors=[]
            for dt in (.1,.05,.025):
                q=np.array([.4,q2]);command=np.array(u);end,nfev=flow(q,np.zeros(2),command,dt)
                prediction=.5*dt*dt*np.linalg.solve(mass(q),command)
                displacement=end[:2]-q
                errors.append(float(np.linalg.norm(displacement-prediction)))
                o0=q+rng.normal(0,SENSOR_SD,2);o1=end[:2]+rng.normal(0,SENSOR_SD,2)
                rows.append(dict(q0_evaluator_only=q.tolist(),v0_prepared_rest=[0.,0.],command=list(u),duration=dt,
                    final_state_evaluator_only=end.tolist(),observed_initial_position=o0.tolist(),observed_final_position=o1.tolist(),
                    displacement_evaluator_only=displacement.tolist(),leading_response_evaluator_only=prediction.tolist(),
                    leading_error_norm=errors[-1],energy_minus_command_work=kinetic(end)-float(command@displacement),nfev=nfev))
            # Generic q2 gives quartic position remainder; symmetry poses can be
            # higher order, and zero response cancellation is not a rate failure.
            rows[-1]['three_duration_error_norms']=errors
    energy=[]
    for i in range(24):
        q=rng.uniform(-math.pi,math.pi,2);v=rng.uniform(-.6,.6,2);u=rng.uniform(-1,1,2);z=np.r_[q,v]
        end,nfev=flow(q,v,u,.5);tighter,_=flow(q,v,u,.5,1e-13)
        error=kinetic(end)-kinetic(z)-float(u@(end[:2]-q));energy.append(dict(q0=q.tolist(),v0=v.tolist(),command=u.tolist(),
            final_state=end.tolist(),tighter_state=tighter.tolist(),energy_work_error=error,tolerance_state_difference=float(np.max(np.abs(end-tighter))),nfev=nfev))
    analytic=[]
    for q2 in (0.,math.pi):
        q=np.array([0.,q2]);u=np.array([0.,1.]);end,_=flow(q,np.zeros(2),u,.05)
        analytic.append(dict(q2=q2,inverse_inertia=np.linalg.inv(mass(q)).tolist(),command=u.tolist(),
            displacement=end[:2]-q,short_pulse_response=(end[:2]-q)/(.5*.05**2)))
    for a in analytic:
        a['displacement']=a['displacement'].tolist();a['short_pulse_response']=a['short_pulse_response'].tolist()
    eigen=[np.linalg.eigvalsh(mass([0.,q2])) for q2 in np.linspace(-math.pi,math.pi,1025)]
    targets=((.1,0.),(0.,.1),(.1,-.1),(-.1,.1),(.05,.1))
    max_required=max(np.max(np.abs(mass([0,q2])@z)) for q2 in np.linspace(-math.pi,math.pi,1025) for z in targets)
    summary=dict(rest_pulse_episodes=len(rows),rest_environment_integration_time=sum(r['duration'] for r in rows),
        general_energy_episodes=len(energy),general_environment_integration_time=24*.5,
        maximum_energy_work_error=float(max(abs(e['energy_work_error']) for e in energy)),
        maximum_tighter_solver_state_difference=float(max(e['tolerance_state_difference'] for e in energy)),
        minimum_inertia_eigenvalue=float(np.min(eigen)),
        maximum_linearized_target_torque=float(max_required),target_acceleration_vectors=targets,
        generic_pose_error_ratios=[dict(q2=r['q0_evaluator_only'][1],command=r['command'],errors=r['three_duration_error_norms'],
            coarse_over_middle=r['three_duration_error_norms'][0]/r['three_duration_error_norms'][1],
            middle_over_fine=r['three_duration_error_norms'][1]/r['three_duration_error_norms'][2]) for r in rows if 'three_duration_error_norms' in r])
    assert summary['maximum_energy_work_error']<1e-8
    assert summary['maximum_tighter_solver_state_difference']<1e-8
    assert summary['minimum_inertia_eigenvalue']>0
    assert max_required<1.
    out=dict(meta=dict(status='UNRELEASED MECHANICS AND MEASUREMENT PILOT; NO LEARNING CLAIM',seed=SEED,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        masses_and_lengths_evaluator_only=[1.,1.,1.,1.],horizontal_no_gravity_or_friction=True,sensor_sd=SENSOR_SD,
        continuous_solver='DOP853 rtol1e-11 atol1e-13; comparisonrtol1e-13',
        known_preparation_vs_sensor='Rest preparation supplied; ordinary contextual learner should read noisy initial positions, not hidden q0.',
        warning='No learner trained, no continuous feedback control, no general robotics result; linearized target feasibility not exact endpoint guarantee.'),
        rest_pulses=rows,energy_checks=energy,coupling_examples=analytic,summary=summary)
    print(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
