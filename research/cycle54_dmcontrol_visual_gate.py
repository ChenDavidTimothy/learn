"""Cycle 54: independent high-entropy visual partial-observability gate.

Use OFFICIAL DeepMind Control Suite continuous-action MuJoCo dynamics and
EGL-rendered RGB observations, not POPGym Arcade sprites.

Before proposing richer sensor representation recovery, measure RGB support,
how well missing channels can actually be inferred from other two channel
values, and action sensitivity, independently across physics tasks.

NO POLICY IS TRAINED. No new algorithm. Source/test random-action image
streams are independent and aggregated only; no raw frames committed.
"""
import argparse,json,time,os
import numpy as np
from dm_control import suite
TASKS=(('cartpole','balance'),('cartpole','swingup'),
       ('finger','spin'),('reacher','easy'),('walker','walk'),
       ('cheetah','run'))
def make(domain,task,seed):
    return suite.load(domain_name=domain,task_name=task,
        task_kwargs={'random':np.random.RandomState(seed)},
        visualize_reward=False)
def collect(domain,task,seed,episodes,steps,side):
    env=make(domain,task,seed)
    rg=np.random.default_rng(seed+1127)
    low=np.asarray(env.action_spec().minimum,dtype=np.float32)
    high=np.asarray(env.action_spec().maximum,dtype=np.float32)
    images=[]
    returns=[]
    for ep in range(episodes):
        ts=env.reset();total=0.
        for t in range(steps):
            images.append(env.physics.render(height=side,width=side))
            a=rg.uniform(low,high).astype(env.action_spec().dtype)
            ts=env.step(a)
            total+=float(ts.reward or 0)
            if ts.last():break
        returns.append(total)
    return np.array(images,dtype=np.uint8),returns
def per_channel_errors(source,test):
    # Source RGB color lookup conditional on the two surviving values;
    # established histogram estimator, not privileged ground truth.
    rgb=source.reshape(-1,3).astype(np.int32)
    truth=test.reshape(-1,3).astype(np.int32)
    evaluations={}
    for c in range(3):
        pair=[j for j in range(3) if j!=c]
        source_ids=rgb[:,pair[0]]*256+rgb[:,pair[1]]
        test_ids=truth[:,pair[0]]*256+truth[:,pair[1]]
        ct=np.bincount(source_ids,minlength=65536).astype(float)
        weighted=np.bincount(source_ids,weights=rgb[:,c],minlength=65536)
        fallback=float(np.mean(rgb[:,c]))
        est=(weighted+2*fallback)/(ct+2)
        predicted=est[test_ids]
        actual=truth[:,c]
        rmse=float(np.sqrt(np.mean((actual-predicted)**2)))
        sd=float(np.std(actual))
        support=float(np.mean(ct[test_ids]>0))
        evaluations[str(c)]={'source_distinct_surviving_pairs':int((ct>0).sum()),
             'heldout_RMSE':rmse,'heldout_channel_SD':sd,
             'normalized_RMSE':rmse/max(sd,1e-9),
             'heldout_pair_observed_in_source_fraction':support}
    return evaluations
def influence(domain,task,seed,steps,side):
    pairs=[]
    for direction in (-1,1):
        env=make(domain,task,seed)
        ts=env.reset()
        spec=env.action_spec()
        scale=spec.minimum if direction<0 else spec.maximum
        start=env.physics.render(height=side,width=side)
        for _ in range(steps):
            ts=env.step(np.asarray(scale,dtype=spec.dtype))
            if ts.last():break
        pairs.append((start,env.physics.render(height=side,width=side)))
    # Same initial condition, perturb only action sequence, then compare
    # visual trajectory endpoint. Not an experiment proving strict POMDP.
    return {'same_start_pixel_difference':float(np.mean(np.abs(
       pairs[0][0].astype(float)-pairs[1][0].astype(float)))),
       'opposite_action_endpoint_pixel_difference':float(np.mean(np.abs(
       pairs[0][1].astype(float)-pairs[1][1].astype(float))))}
