#!/usr/bin/env python3
"""Unreleased v9 pilot: nonlinear passive force versus coupled actuator gauge.

Numerical illustration only; independent artifact audit and full cycle pending.
Use lagged motor timing vnext=f(v)+m, mnext=h(m,v,u)+xi.
"""
import json
import math
import numpy as np

SEED=20261020+1400000
RHO,LAM,C,BETA=.65,.4,.2,1.2
rng=np.random.default_rng(SEED)
n=8192
z=rng.normal(0,.7,n)
xi=rng.normal(0,.06,n)
e=rng.normal(0,.5,n+1)
v=np.zeros(n+1);m=np.zeros(n+1);vp=np.zeros(n+1);mp=np.zeros(n+1)
u=np.zeros(n);up=np.zeros(n)
for t in range(n):
    D=v[t]+e[t]-(e[t-1] if t else 0.)
    Dp=vp[t]+e[t]-(e[t-1] if t else 0.)
    u[t]=-.5*math.tanh(D)+z[t]
    up[t]=-.5*math.tanh(Dp)+z[t]
    v[t+1]=RHO*v[t]-C*math.tanh(v[t])+m[t]
    m[t+1]=LAM*m[t]+BETA*u[t]+xi[t]
    vp[t+1]=RHO*vp[t]+mp[t]
    mp[t+1]=LAM*mp[t]+LAM*C*math.tanh(vp[t])+BETA*up[t]-C*math.tanh(vp[t+1])+xi[t]
velocity_error=float(np.max(np.abs(v-vp)))
gauge_error=float(np.max(np.abs(mp-(m-C*np.tanh(v)))))
command_error=float(np.max(np.abs(u-up)))
assert max(velocity_error,gauge_error,command_error)<1e-10
print(json.dumps({'status':'UNRELEASED PILOT; independent artifact audit pending',
    'seed':SEED,'n':n,'numpy':np.__version__,'rho':RHO,'lambda':LAM,'c':C,'beta':BETA,
    'original_passive':'rho*v-c*tanh(v)', 'original_actuator':'lambda*m+beta*u+xi',
    'alternative_passive':'rho*v',
    'alternative_actuator':'lambda*mp+lambda*c*tanh(v)+beta*u-c*tanh(vnext)+xi',
    'max_velocity_difference':velocity_error,'max_command_difference':command_error,
    'max_force_gauge_error':gauge_error},indent=2,allow_nan=False))
