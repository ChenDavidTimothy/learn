#!/usr/bin/env python3
"""Independent PyTorch-autodiff reconstruction of v13 policy sensitivities.

Canonical code is imported ONLY as an array comparison target after the separate
state/controller implementation; no canonical derivative is used by autodiff.
"""
import json
import math
import numpy as np
import torch
import gradient_reuse_v13 as reference

torch.set_num_threads(1)
DT=torch.float64


def independent_residual(theta,coordinates,noise,trace=False):
    rho,lam,c,beta=map(float,theta)
    batch,observations=noise.shape
    steps=observations-1
    plans=((1.,-1.,.5),(-1.,1.,-.5))
    goals=torch.tensor([[plans[b%2][0] if t<6 else plans[b%2][1] if t<28 else plans[b%2][2]
                         for t in range(steps)] for b in range(batch)],dtype=DT)
    scale=torch.sqrt(batch*steps*(1+torch.mean(goals.square(),dim=1)))
    kp,kd=torch.exp(coordinates[0]),torch.exp(coordinates[1])
    alpha=torch.sigmoid(coordinates[2])
    bv,bu1,bu2=coordinates[3:]
    state=torch.zeros((batch,3),dtype=DT)
    memory=torch.zeros((batch,4),dtype=DT) # sx,sv,last issuedu,previous issuedu
    residuals=[];raws=[];states=[state];commands=[]
    for t in range(steps):
        position_est=(1-alpha)*memory[:,0]+alpha*(state[:,0]+noise[:,t])
        velocity_est=(1-alpha)*memory[:,1]+alpha*(position_est-memory[:,0])
        raw=kp*(goals[:,t]-position_est)-kd*velocity_est+bv*memory[:,1]-bu1*memory[:,2]-bu2*memory[:,3]
        command=torch.clamp(raw,min=-1.,max=1.)
        velocity=rho*state[:,1]-c*torch.tanh(state[:,1])+state[:,2]
        motor=lam*state[:,2]+beta*command
        state=torch.stack((state[:,0]+velocity,velocity,motor),dim=1)
        memory=torch.stack((position_est,velocity_est,command,memory[:,2]),dim=1)
        residuals.append(torch.stack((state[:,0]-goals[:,t],math.sqrt(.3)*velocity,math.sqrt(.03)*command))/scale[None,:])
        raws.append(raw);states.append(state);commands.append(command)
    values=torch.stack(residuals).reshape(-1)
    if trace:return values,torch.stack(raws,dim=1),torch.stack(states,dim=1),torch.stack(commands,dim=1)
    return values


def case(theta,p,seed,batch=4):
    errors=np.random.default_rng(seed).normal(0,.2,(batch,87))
    noise=torch.tensor(errors,dtype=DT);coords=torch.tensor(p,dtype=DT)
    residual,raw,states,commands=independent_residual(theta,coords,noise,True)
    automatic=torch.autograd.functional.jacobian(lambda q:independent_residual(theta,q,noise),coords,
                                                vectorize=True,strategy='forward-mode')
    actual,J,s,o,u=reference.roll(theta,np.asarray(p,dtype=float),errors)
    automatic=automatic.detach().numpy();raw=raw.detach().numpy()
    abs_error=float(np.max(np.abs(automatic-J)))
    scaled_error=float(np.max(np.abs(automatic-J)/(1+np.abs(automatic))))
    assert scaled_error<1e-6
    residual_error=float(np.max(np.abs(residual.detach().numpy()-actual)))
    assert residual_error<1e-7
    return {'theta':theta,'coordinates':p,'seed':seed,'batch':batch,
        'residual_max_error':residual_error,'autodiff_jacobian_max_error':abs_error,
        'strict_scaled_1e8_agreement':bool(scaled_error<1e-8),
        'maximum_absolute_autodiff_entry':float(np.max(np.abs(automatic))),
        'autodiff_jacobian_max_scaled_error':scaled_error,
        'state_max_error':float(np.max(np.abs(states.detach().numpy()-s))),
        'command_max_error':float(np.max(np.abs(commands.detach().numpy()-u))),
        'saturated_command_count':int(np.count_nonzero(np.abs(raw)>1)),
        'minimum_distance_to_clip_kink':float(np.min(np.abs(np.abs(raw)-1)))}


def independent_kink_check():
    theta=(.65,.7,.2,1.2)
    errors=np.random.default_rng(75071013).normal(0,.2,(16,87))
    noise=torch.tensor(errors,dtype=DT)
    p=torch.tensor([math.log(.08),math.log(.6),math.log(.6/.4),.1,.2,-.05],dtype=DT)
    J=torch.autograd.functional.jacobian(lambda q:independent_residual(theta,q,noise),p,
                                        vectorize=True,strategy="forward-mode").detach().numpy()
    rows=[]
    for epsilon in (1e-6,1e-7,1e-8):
        differences=[];switches=[]
        for coordinate in range(6):
            d=torch.zeros(6,dtype=DT);d[coordinate]=epsilon
            plus,rp,*_=independent_residual(theta,p+d,noise,True)
            minus,rm,*_=independent_residual(theta,p-d,noise,True)
            numeric=((plus-minus)/(2*epsilon)).detach().numpy()
            differences.append(float(np.max(np.abs(J[:,coordinate]-numeric))))
            switches.append(int(torch.count_nonzero((torch.abs(rp)<1)!=(torch.abs(rm)<1))))
        rows.append({"epsilon":epsilon,"coordinate_max_errors":differences,"independent_raw_clip_active_set_changes":switches})
    assert rows[0]["independent_raw_clip_active_set_changes"][3]==1
    assert all(sum(r["independent_raw_clip_active_set_changes"])==0 for r in rows[1:])
    return rows


def main():
    rows=[]
    generic=[math.log(.08),math.log(.6),math.log(.6/.4),0.,0.,0.]
    memory=[math.log(.08),math.log(.6),math.log(.6/.4),.1,.2,-.05]
    saturated=[math.log(.4),math.log(3.),math.log(.9/.1),1.3,-.5,.7]
    for pi,theta in enumerate([(.65,lam,c,1.2) for lam in (.2,.7) for c in (0.,.2)]):
        for j,p in enumerate((generic,memory,saturated)):
            rows.append(case(theta,p,20261020+991000+100*pi+j))
    # Reconstruct the exact canonical failed-coarse-FD setup independently.
    rows.append(case((.65,.7,.2,1.2),memory,75071013,16))
    print(json.dumps({'status':'independent autodiff audit completed; conditioning limits retained',
        'correction':'Initial strict1e-8 scaled tolerance failed on one extreme saturated long-horizon stresscase; all errors are retained and strict-agreement flags distinguish it. Broad1e-6 stress tolerance is not machine-precision agreement.',
        'torch':torch.__version__,'dtype':'float64','cases':rows,
        'independent_kink_check':independent_kink_check(),
        'max_scaled_jacobian_error':max(r['autodiff_jacobian_max_scaled_error'] for r in rows),
        'scope':'independent PyTorch dynamics/controller/autodiff, canonical arrays only for final comparison; no optimizer or plant-performance claim'},indent=2))


if __name__=='__main__':main()
