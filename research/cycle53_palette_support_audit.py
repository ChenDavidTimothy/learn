"""Cycle 53: audit conditional recoverability of RGB channels across POPGym Arcade.

Tests whether Cycle50's surprisingly complete missing-channel repair is a
special property of a tiny, redundant renderer palette. On independent
random-action observations from multiple task families, measure the
number of distinct RGB colors, number of distinct surviving channel
pairs, and minimum empirical conditional variance H(channel|other two)
using a source-trained nonparametric LUT tested on *different* episodes.

No RL reward labels or hidden state. This is a dataset support/entropy
audit, not a new learning method or a control benchmark.
"""
import json,time,argparse
import numpy as np
import popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from cycle50_missing_sensor_imputation import get_frames,fit_source
TASKS=['NoisyCartPoleEasy','NavigatorEasy','BattleShipEasy',
       'CountRecallEasy','MineSweeperEasy','SkittlesEasy',
       'TetrisEasy','BreakoutEasy']
def audit(train,test):
    source=train.reshape(-1,3)
    heldout=test.reshape(-1,3)
    unique_colors=np.unique(source,axis=0).shape[0]
    source_mu,model=fit_source(train)
    result={'training_RGB_pixel_samples':len(source),
            'holdout_RGB_pixel_samples':len(heldout),
            'unique_source_colors':int(unique_colors),
            'per_missing_channel':{}}
    for c in range(3):
        other=model[c]['other']
        # Evaluate on independent observations, no fitting to holdout.
        s=test[...,other[0]].astype(np.int32)*256+test[...,other[1]].astype(np.int32)
        lookup=np.asarray(model[c]['lookup'])
        prediction=lookup[s]
        target=test[...,c].astype(np.float32)
        err=target-prediction
        # Fallback smoothing means even train-observed pure palette
        # combinations have tiny nonzero error, which is disclosed.
        sd=float(np.std(target))
        result['per_missing_channel'][str(c)]={
            'source_distinct_survivor_pairs':model[c]['nonzero_palette_bins'],
            'holdout_RMSE':float(np.sqrt(np.mean(err**2))),
            'holdout_channel_SD':sd,
            'relative_RMSE':float(np.sqrt(np.mean(err**2))/max(sd,1e-9)),
            'within_1_intensity_accuracy':float(np.mean(np.abs(err)<1.))}
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--actors',type=int,default=4)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--output',default='research/cycle53_palette_support_audit.json')
    a=p.parse_args();now=time.perf_counter()
    result={'protocol':{'suite':'native POPGym Arcade 0.0.6 pixel renderers',
        'tasks':TASKS,'train_actors':a.actors,'train_steps':a.steps,
        'test_actors':a.actors,'test_steps':a.steps,
        'selection':'random-action source/test histories, independent seeds',
        'measurement':'conditional pixel color reconstruction; not task return',
        'source_model':'simple discrete conditional source color LUT',
        'not_general':'observed random-action support only; no rare control states guarantee',
        'no_novel_algorithm':True},'tasks':{}}
    for i,task in enumerate(TASKS):
        env,params=popgym_arcade.make(task,partial_obs=True,obs_size=128)
        env=LogWrapper(env)
        source=get_frames(env,params,53111026+5000*i,a.actors,a.steps)
        test=get_frames(env,params,53131026+5000*i,a.actors,a.steps)
        metrics=audit(source,test)
        result['tasks'][task]=metrics
        print('TASK',task,'colors',metrics['unique_source_colors'],
           'relative_MSE',[round(metrics['per_missing_channel'][str(c)]['relative_RMSE'],4) for c in range(3)],
           'elapsed',round(time.perf_counter()-now,1),flush=True)
        with open(a.output,'w') as f:json.dump(result,f,indent=2)
    result['seconds']=time.perf_counter()-now
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print('SAVED',a.output,round(result['seconds'],1),flush=True)
if __name__=='__main__':main()
