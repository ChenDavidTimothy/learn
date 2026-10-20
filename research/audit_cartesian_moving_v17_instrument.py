#!/usr/bin/env python3
"""Independent matched-moving Cartesian instrument audit; no canonical imports.

Vectorized Hamiltonian RK4 and Radau independently check physics. Gaussian
information is recomputed from saved deterministic means, without sensor draws.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
import audit_cartesian_reuse_v16_pilot as independent

ROOT=Path(__file__).resolve().parent
ERROR={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,a,b,tol=1e-10):
    e=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));ERROR[name]=max(ERROR.get(name,0.),e);assert e<tol,(name,e,tol)

def evolve(z,duration,steps=128):
    q=z[...,:2];w=np.concatenate([q,(independent.mass(q)@z[...,2:,None])[...,0]],axis=-1);h=duration/steps;u=np.zeros(q.shape)
    for _ in range(steps):
        k1=independent.hamilton_rhs(w,u);k2=independent.hamilton_rhs(w+.5*h*k1,u);k3=independent.hamilton_rhs(w+.5*h*k2,u);k4=independent.hamilton_rhs(w+h*k3,u);w+=h*(k1+2*k2+2*k3+k4)/6
    return np.concatenate([w[...,:2],np.linalg.solve(independent.mass(w[...,:2]),w[...,2:,None])[...,0]],axis=-1)

def qv_rhs(z,u):
    q=z[:2];v=z[2:];M=independent.mass(q);p=M@v;pd=independent.hamilton_rhs(np.r_[q,p],u)[2:]
    s=np.sin(q[1]);Mq=np.array([[-2*s,-s],[-s,0.]])
    return np.r_[v,np.linalg.solve(M,pd-v[1]*(Mq@v))]

def cartesian_acceleration(z,u):
    q=z[:2];v=z[2:];a=qv_rhs(z,u)[2:]
    e1=np.array([np.cos(q[0]),np.sin(q[0])]);e2=np.array([np.cos(q.sum()),np.sin(q.sum())])
    return independent.jac(q)@a-e1*v[0]**2-e2*v.sum()**2

def radau(z,duration):
    M=independent.mass(z[:2]);w=np.r_[z[:2],M@z[2:]]
    sol=solve_ivp(lambda t,w:independent.hamilton_rhs(w,np.zeros(2)),(0,duration),w,method='Radau',rtol=2e-12,atol=2e-14);assert sol.success
    w=sol.y[:,-1];return np.r_[w[:2],np.linalg.solve(independent.mass(w[:2]),w[2:])]

def main():
    started=time.time();source=ROOT/'cartesian_moving_v17_instrument.py';rp=ROOT/'cartesian_moving_v17_instrument_results.json';question=ROOT/'cartesian_moving_v17_question.md';r=json.loads(rp.read_text());assert sha(source)==r['meta']['source_sha256']
    assert r['meta']['dt']==.1 and r['meta']['sensor_sd_assumed']==.0002
    cases=r['paired_cases'];assert len(cases)==54
    initial=[];recorded=[];keys=set();accelerations=[];rotation={};taylor=[];nfev=0;mass_counterexample=0.;common_input_tangent_error=0.
    for row in cases:
        phi=row['phi'];theta=row['theta'];speed=row['cartesian_speed'];angle=row['direction'];keys.add((phi,theta,speed,angle))
        c=np.cos(phi/2);s=np.sin(phi/2);vr=speed*np.cos(angle);vt=speed*np.sin(angle);a=vt/(2*c);b=vr/(2*s)
        expected=np.array([[theta-phi/2,phi,a+b,-2*b],[theta+phi/2,-phi,a-b,2*b]])
        ck('matched_polar_preparations',expected,row['initial_states_evaluator_only'],1e-14);q=expected[:,:2];v=expected[:,2:]
        positions=independent.endpoint(q);velocity=(independent.jac(q)@v[...,None])[...,0];target=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])@np.array([vr,vt])
        ck('matched_current_positions',positions[0],positions[1],1e-14);ck('matched_current_velocities',velocity,np.repeat(target[None],2,axis=0),1e-14)
        ck('stored_common_velocity',target,row['shared_cartesian_velocity'],1e-14)
        rotation_matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        delta=cartesian_acceleration(expected[1],np.zeros(2))-cartesian_acceleration(expected[0],np.zeros(2));accelerations.append(delta)
        ck('independent_Hamiltonian_cartesian_acceleration',delta,row['acceleration_delta_B_minus_A'],1e-13);ck('analytic_acceleration_formula',delta,row['analytic_delta'],1e-13)
        local=rotation_matrix.T@delta;D=2-np.cos(phi)**2
        ck('local_radial_acceleration',local[0],-2*s*vr*vt/D,1e-13);ck('local_tangent_acceleration',local[1],(s*vt*vt+c*c*vr*vr/s)/D,1e-13)
        if speed:assert local[1]>0
        else:ck('rest_acceleration',delta,[0.,0.],1e-14)
        for u in (np.array([.2,-.1]),np.array([-1.,1.]),np.array([.5,.5])):
            d=cartesian_acceleration(expected[1],u)-cartesian_acceleration(expected[0],u);common_input_tangent_error=max(common_input_tangent_error,abs((rotation_matrix.T@d)[1]-local[1]))
            # Scale both masses: M_k=kM, h_k=kh. Zero-input q/v RHS is
            # unchanged; known nonzero torque gain is inversely proportional.
            M=independent.mass(expected[0,:2]);f=qv_rhs(expected[0],np.zeros(2))[2:]
            h=-M@f
            for k in (.3,2.,7.):mass_counterexample=max(mass_counterexample,float(np.max(abs(np.linalg.solve(k*M,-k*h)-f))))
        states=np.array(row['states_evaluator_only']);means=np.array(row['cartesian_means_evaluator_only']);assert states.shape==(2,11,4) and means.shape==(2,11,2)
        ck('stored_initial_state',states[:,0],expected,1e-14);ck('Cartesian_mean_conversion',independent.endpoint(states[...,:2]),means,1e-14);ck('times',row['times'],.1*np.arange(1,11),1e-15)
        diffs=means[1,1:]-means[0,1:];d2=[];running=0.
        for x in diffs:running+=math.fsum(float(t*t) for t in x)/(.0002**2);d2.append(running)
        kl=[x/2 for x in d2];error=[.5*math.erfc(math.sqrt(x)/(2*math.sqrt(2))) for x in d2]
        ck('squared_Mahalanobis',d2,row['cumulative_squared_mahalanobis'],1e-9);ck('Gaussian_KL',kl,row['cumulative_KL_B_to_A'],1e-9);ck('known_pair_Bayes_error',error,row['equal_prior_known_pair_testing_error'],2e-15)
        assert np.all(np.diff(d2)>=0) and np.all(np.diff(error)<=1e-15)
        rotation[(phi,speed,angle,theta)]={'states':states,'positions':means,'d2':d2,'error':error}
        initial.append(expected);recorded.append(states)
        for item in row['finite_time_taylor_checks']:
            duration=item['duration'];leading=.5*duration**2*delta
            ck('Taylor_leading_term',leading,item['leading'],1e-15);taylor.append((len(initial)-1,duration,np.array(item['separation']),float(item['remainder_norm'])))
    assert len(keys)==54;initial=np.array(initial);recorded=np.array(recorded);z=initial.copy();energy0=.5*np.einsum('nbi,nbij,nbj->nb',z[...,2:],independent.mass(z[...,:2]),z[...,2:]);momentum0=(independent.mass(z[...,:2])@z[...,2:,None])[...,0][...,0];energy_error=momentum_error=0.
    for step in range(10):
        z=evolve(z,.1);ck('all_passive_Hamiltonian_states',z,recorded[:,:,step+1],3e-11)
        E=.5*np.einsum('nbi,nbij,nbj->nb',z[...,2:],independent.mass(z[...,:2]),z[...,2:]);P=(independent.mass(z[...,:2])@z[...,2:,None])[...,0][...,0]
        energy_error=max(energy_error,float(abs(E-energy0).max()));momentum_error=max(momentum_error,float(abs(P-momentum0).max()))
    for duration in (.01,.005,.0025):
        end=evolve(initial,duration);separation=independent.endpoint(end[:,1,:2])-independent.endpoint(end[:,0,:2])
        for i,dt,observed,norm in taylor:
            if dt==duration:
                ck('all_Taylor_Hamiltonian_separations',separation[i],observed,2e-14);ck('Taylor_remainder_norm',np.linalg.norm(observed-.5*duration**2*accelerations[i]),norm,1e-15)
    radau_checks=0
    for i in range(0,54,5):
        for b in range(2):
            for step in (0,4,9):ck('Radau_passive_state',radau(recorded[i,b,step],.1),recorded[i,b,step+1],3e-11);radau_checks+=1
    for i in range(54):
        for dt in (.01,.005,.0025):
            end=np.array([radau(initial[i,b],dt) for b in range(2)]);sep=independent.endpoint(end[1,:2])-independent.endpoint(end[0,:2]);stored=next(item[2] for item in taylor if item[0]==i and item[1]==dt);ck('Radau_Taylor_separation',sep,stored,2e-14);radau_checks+=2
    for phi in (np.pi/3,np.pi/2,2*np.pi/3):
        Q=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
        for speed in (0.,.03,.1):
            for angle in ((0.,) if speed==0 else (0.,np.pi/4,np.pi/2,3*np.pi/4)):
                A=rotation[(phi,speed,angle,0.)];B=rotation[(phi,speed,angle,.7)]
                ck('rotation_equivariance_positions',A['positions']@Q.T,B['positions'],2e-14);ck('rotation_invariant_information',A['d2'],B['d2'],1e-7)
    summaries=[]
    for speed in (0.,.03,.1):
        rows=[row for row in cases if row['cartesian_speed']==speed];errs=np.array([row['equal_prior_known_pair_testing_error'] for row in rows]);stored=next(s for s in r['summary'] if s['speed']==speed)
        for name,value in [('min_known_pair_error',errs.min(0)),('mean_known_pair_error',errs.mean(0)),('max_known_pair_error',errs.max(0))]:ck('summary_'+name,value,stored[name],1e-15)
        assert len(rows)==stored['pairs'];summaries.append({'speed':speed,'pairs':len(rows),'mean_error_at_.1_.5_1_seconds':errs.mean(0)[[0,4,9]].tolist(),'minimum_error_at_1_second':float(errs[:,-1].min()),'maximum_error_at_1_second':float(errs[:,-1].max())})
    # Counts independent of adaptive solver internals; also repeat DOP853 with
    # an independently reconstructed q/v RHS solely to check its reported nfev.
    for item,z0 in enumerate(initial.reshape(-1,4)):
        zz=z0.copy()
        for step in range(10):
            fit=solve_ivp(lambda t,z:qv_rhs(z,np.zeros(2)),(0,.1),zz,method='DOP853',rtol=1e-11,atol=1e-13);assert fit.success;nfev+=fit.nfev;zz=fit.y[:,-1]
        for dt in (.01,.005,.0025):
            fit=solve_ivp(lambda t,z:qv_rhs(z,np.zeros(2)),(0,dt),z0,method='DOP853',rtol=1e-11,atol=1e-13);assert fit.success;nfev+=fit.nfev
    resources={'paired_contexts':54,'passive_episodes':108,'passive_intervals':1080,'passive_simulated_seconds':108.,'taylor_flows':324,'taylor_simulated_seconds':1.89,'total_plant_flows':1404,'sampled_sensor_readings':0,'hypothetical_scalar_sensor_opportunities':2160}
    for k,v in resources.items():ck('resources_'+k,v,r['resources'][k],1e-12)
    # Different floating operation order can change an adaptive step decision;
    # preserve rather than mislabel a solver-internal count discrepancy.
    out={'status':'INDEPENDENT MOVING INSTRUMENT PHYSICS/INFORMATION AUDIT PASSED; NO LEARNER CLAIM',
        'artifact_sha256':{p.name:sha(p) for p in [source,rp,question,ROOT/'audit_cartesian_reuse_v16_pilot.py']},'audit_source_sha256':sha(Path(__file__)),
        'reconstruction_errors':ERROR,'summary':summaries,'maximum_energy_drift':energy_error,'maximum_cyclic_momentum_drift':momentum_error,'Radau_interval_checks':radau_checks,
        'common_torque_tangential_discrepancy_error':common_input_tangent_error,'passive_global_mass_scale_counterexample_rhs_error':mass_counterexample,
        'Taylor_remainders_below_2e_minus14_meters':sum(t[3]<2e-14 for t in taylor),'minimum_saved_Taylor_remainder_norm':min(t[3] for t in taylor),
        'resources':resources,'reported_DOP853_rhs_evaluations':r['resources']['plant_rhs_evaluations'],'independent_qv_DOP853_rhs_evaluations':nfev,
        'limitations':['Known fixed-pair Gaussian means with iid noise, equal priors and known timing; no sensor realizations or learned classifier.',
            'Matched current Cartesian velocity and physical derivatives are evaluator preparations, not ordinary inputs or learned estimates.',
            'These are forward passive observation windows, not pre-existing causal histories; no task cost of waiting/moving is evaluated.',
            'Different passive paths establish a distinction for these specified alternatives, not general nonlinear observability, learnability or reliable finite-window diagnosis.',
            'Passive motion cannot identify common scaling of all masses: zero-torque dynamics unchanged while known-torque response scales inversely.',
            'Approximation starts at equal p/V; generic moving separation has quadratic leading term and generally cubic remainder, unlike the resting quartic position expansion.'],
        'wall_seconds':time.time()-started}
    assert common_input_tangent_error<1e-12 and mass_counterexample<1e-12
    (ROOT/'audit_cartesian_moving_v17_instrument_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'errors':ERROR,'summary':summaries,'nfev':nfev,'recorded_nfev':r['resources']['plant_rhs_evaluations'],'seconds':out['wall_seconds']}))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
