#!/usr/bin/env python3
"""Matched p/V passive-information instrument, not a learned mechanism."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from coupled_context_v14_pilot import flow,mass,rhs

DT=.1;STEPS=10;SD=.0002


def endpoint(q):
    return np.array([np.cos(q[0])+np.cos(q.sum()),np.sin(q[0])+np.sin(q.sum())])


def jacobian(q):
    return np.array([[-np.sin(q[0])-np.sin(q.sum()),-np.sin(q.sum())],
        [np.cos(q[0])+np.cos(q.sum()),np.cos(q.sum())]])


def acceleration(z,u):
    q=z[:2];v=z[2:];e1=np.array([np.cos(q[0]),np.sin(q[0])]);e2=np.array([np.cos(q.sum()),np.sin(q.sum())])
    return jacobian(q)@rhs(0,z,u)[2:]-e1*v[0]**2-e2*v.sum()**2


def main():
    start=time.time();rows=[];nfev=flows=0;maxp=maxv=maxa=0.
    with threadpool_limits(limits=1):
        for phi in (np.pi/3,np.pi/2,2*np.pi/3):
            for theta in (0.,.7):
                er=np.array([np.cos(theta),np.sin(theta)]);et=np.array([-np.sin(theta),np.cos(theta)])
                for speed in (0.,.03,.1):
                    for angle in ((0.,) if speed==0 else (0.,np.pi/4,np.pi/2,3*np.pi/4)):
                        vr=speed*np.cos(angle);vt=speed*np.sin(angle);V=vr*er+vt*et
                        qA=np.array([theta-phi/2,phi]);qB=np.array([theta+phi/2,-phi])
                        initial=[np.r_[q,np.linalg.solve(jacobian(q),V)] for q in (qA,qB)]
                        pe=float(np.max(abs(endpoint(qA)-endpoint(qB))))
                        ve=float(np.max(abs(jacobian(qA)@initial[0][2:]-jacobian(qB)@initial[1][2:])))
                        maxp=max(maxp,pe);maxv=max(maxv,ve)
                        s=np.sin(phi/2);c=np.cos(phi/2);D=2-np.cos(phi)**2
                        predicted=(-2*s*vr*vt/D)*er+((s*vt**2+c*c*vr**2/s)/D)*et
                        actual=acceleration(initial[1],np.zeros(2))-acceleration(initial[0],np.zeros(2))
                        maxa=max(maxa,float(np.max(abs(predicted-actual))))
                        # Common inputs cannot cancel the tangential discrepancy.
                        for u in (np.zeros(2),np.array([.2,-.1])):
                            delta=acceleration(initial[1],u)-acceleration(initial[0],u)
                            assert abs(delta@et-predicted@et)<1e-12
                        trajectories=[];positions=[]
                        for z0 in initial:
                            z=z0.copy();states=[z.copy()]
                            for step in range(STEPS):
                                z,k=flow(z[:2],z[2:],np.zeros(2),DT);nfev+=k;flows+=1;states.append(z.copy())
                            states=np.array(states);trajectories.append(states)
                            positions.append(np.array([endpoint(z[:2]) for z in states]))
                        positions=np.array(positions);differences=positions[1,1:]-positions[0,1:]
                        squared=np.cumsum(np.sum(differences**2,axis=1))/SD**2
                        errors=ndtr(-np.sqrt(squared)/2)
                        if speed==0:assert np.max(abs(differences))<1e-14
                        else:assert predicted@et>0
                        taylor=[]
                        for duration in (.01,.005,.0025):
                            ends=[]
                            for z0 in initial:
                                zz,k=flow(z0[:2],z0[2:],np.zeros(2),duration);ends.append(endpoint(zz[:2]));nfev+=k;flows+=1
                            separation=ends[1]-ends[0];leading=.5*duration**2*predicted
                            taylor.append(dict(duration=duration,separation=separation,leading=leading,
                                remainder_norm=float(np.linalg.norm(separation-leading))))
                        rows.append(dict(phi=phi,theta=theta,cartesian_speed=speed,direction=angle,
                            shared_cartesian_velocity=V,initial_states_evaluator_only=np.array(initial),
                            initial_endpoint_difference=pe,initial_velocity_difference=ve,
                            acceleration_delta_B_minus_A=actual,analytic_delta=predicted,
                            states_evaluator_only=np.array(trajectories),cartesian_means_evaluator_only=positions,
                            times=DT*np.arange(1,STEPS+1),cumulative_squared_mahalanobis=squared,
                            cumulative_KL_B_to_A=.5*squared,equal_prior_known_pair_testing_error=errors,
                            finite_time_taylor_checks=taylor))
    assert len(rows)==54 and maxp<1e-12 and maxv<1e-12 and maxa<1e-12
    summaries=[]
    for speed in (0.,.03,.1):
        subset=[r for r in rows if r['cartesian_speed']==speed]
        error=np.array([r['equal_prior_known_pair_testing_error'] for r in subset])
        summaries.append(dict(speed=speed,pairs=len(subset),times=DT*np.arange(1,STEPS+1),
            min_known_pair_error=error.min(0),mean_known_pair_error=error.mean(0),max_known_pair_error=error.max(0)))
    result=dict(meta=dict(status='PASSIVE MOVING INFORMATION INSTRUMENT; NO LEARNER OR RELEASE',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dt=DT,sensor_sd_assumed=SD,
        design='cartesian_moving_v17_question.md',subtraction='branchB minus branchA; identical currentp and CartesianV',
        physical_model='Same horizontal conservative unit point-mass two-link arm as v14..v16',
        limitations='Known-pair Gaussian testing law only; exact initial twins and physical derivatives evaluator-only. No acquired priors, policy, sampled sensor noises or task-success claim.'),
        max_initial_endpoint_difference=maxp,max_initial_cartesian_velocity_difference=maxv,
        max_analytic_acceleration_difference=maxa,summary=summaries,paired_cases=rows,
        resources=dict(paired_contexts=54,passive_episodes=108,passive_intervals=1080,
            passive_simulated_seconds=108.,taylor_flows=54*6,taylor_simulated_seconds=54*2*(.01+.005+.0025),
            total_plant_flows=flows,plant_rhs_evaluations=nfev,sampled_sensor_readings=0,
            hypothetical_scalar_sensor_opportunities=54*2*10*2),wall_seconds=time.time()-start)
    path=Path(__file__).with_name('cartesian_moving_v17_instrument_results.json')
    path.write_text(json.dumps(result,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item(),indent=2,allow_nan=False)+'\n')
    print('acceleration check',maxa,'contexts',len(rows),'seconds',time.time()-start)
    for s in summaries:print('speed',s['speed'],'known-pair errors at .1/.5/1sec',s['mean_known_pair_error'][[0,4,9]])


if __name__=='__main__':main()
