#!/usr/bin/env python3
"""V9: observable equivalence and explicitly different physical drag locations.

World A drag belongs to body; hidden delivered motor force m.
World B drag belongs to removable motor; total delivered force q=m-c*tanh(v).
The two worlds have the same connected command/position interface. Removing
the entire motor assembly is deliberately a new, physically labeled interface.
The coordinate transform alone does not define this new intervention.
"""
import json
import math
import platform
import numpy as np

SEED=20261020+1600000
RHO,LAM,C,BETA=.65,.4,.2,1.2


def simulate(seed,policy,n=8192):
    rng=np.random.default_rng(seed)
    z=rng.normal(0,.7,n);xi=rng.normal(0,.06,n)
    e=rng.normal(0,.5,n+1);coins=rng.random(n)
    v=w=m=q=x=y=0.
    maxima=np.zeros(5)
    square_force_difference=0.
    for t in range(n):
        d=v+e[t]-(e[t-1] if t else 0.)
        dp=w+e[t]-(e[t-1] if t else 0.)
        base=0. if policy=='open_loop' else -.5*math.tanh(d)
        basep=0. if policy=='open_loop' else -.5*math.tanh(dp)
        a,ap=base+z[t],basep+z[t]
        if policy=='zero_atom' and coins[t]<.2:a=ap=0.
        vn=RHO*v-C*math.tanh(v)+m
        wn=RHO*w+q
        mn=LAM*m+BETA*a+xi[t]
        qn=LAM*q+LAM*C*math.tanh(w)+BETA*ap-C*math.tanh(wn)+xi[t]
        x+=vn;y+=wn
        maxima=np.maximum(maxima,[abs(vn-wn),abs(x-y),abs(a-ap),
            abs(qn-(mn-C*math.tanh(vn))),abs(mn-qn)])
        square_force_difference+=(mn-qn)**2
        v,w,m,q=vn,wn,mn,qn
    assert max(maxima[:4])<1e-9
    return {'seed':seed,'max_velocity_difference':float(maxima[0]),
        'max_position_observation_difference':float(maxima[1]),
        'max_command_difference':float(maxima[2]),
        'max_force_gauge_error':float(maxima[3]),
        'max_total_delivered_force_difference':float(maxima[4]),
        'rms_total_delivered_force_difference':math.sqrt(square_force_difference/n)}


def component_removal():
    # Same connected preparation, then physically remove the entire motor.
    commands=[1.3]*8
    v=m=0.
    for a in commands:
        v,m=RHO*v-C*math.tanh(v)+m,LAM*m+BETA*a
    initial_v=v
    a=[];b=[];x=y=0.;w=v
    for _ in range(12):
        v=RHO*v-C*math.tanh(v)  # Body drag remains in world A.
        w=RHO*w                 # Motor-mounted drag is removed in B.
        x+=v;y+=w
        a.append(v);b.append(w)
    distance=float(np.sqrt(np.mean((np.array(a)-b)**2)))
    assert distance>0
    return {'preparation_commands':commands,'initial_common_velocity':initial_v,
        'world_A_body_drag_response':a,'world_B_motor_drag_response':b,
        'response_velocity_rmse_distance':distance,
        'two_world_minimax_expected_velocity_rmse_lower_bound':distance/2,
        'two_world_equal_prior_velocity_mse_lower_bound':distance**2/4,
        'additional_assumption':'different physical drag location and motor-removal semantics',
        'same_physical_intervention_under_coordinate_change':
            'If only coordinates change, m=0 maps to q=-c*tanh(v), and outputs remain equal.'}


def main():
    cases={}
    for j,policy in enumerate(('open_loop','feedback','zero_atom')):
        cases[policy]=[simulate(SEED+1000*j+i,policy) for i in range(16)]
    print(json.dumps({'meta':{'version':9,'seed':SEED,'n':8192,'replicates':16,
        'python':platform.python_version(),'numpy':np.__version__,'rho':RHO,
        'lambda':LAM,'c':C,'beta':BETA,'probe_sd':.7,'process_sd':.06,
        'position_sensor_sd':.5,'zero_branch_probability':.2,
        'initial_states':'v=m=w=q=x=y=0; phi(0)=0',
        'timing':'vnext=f(v)+m; mnext=lambda*m+beta*u+xi',
        'observations':'issued commands and noisy integrated position only',
        'warning':'calibrated total force is not the same as motor current'},
        'connected':cases,'component_removal':component_removal()},indent=2,allow_nan=False))


if __name__=='__main__':main()
