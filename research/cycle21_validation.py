"""Sanity checks for Cycle 21: correct-family model reproduces exact oracle.
Oracle parameters are used only in these isolated tests, never in training.
"""
import numpy as np
from scipy.optimize import check_grad
import torch
from cycle12_partial_obs import make_task,sample,NOISE
from cycle13_oracle import prob_oracle
from cycle21_latent_joint import pred,prepare,forecast

def validate():
    task=make_task(91921026,7)
    t=task['target_channel']
    j=int(np.where(task['perm']==1)[0][0])
    for regime in ['stationary','nuisance_shift','contextual_action','target_rate_shift']:
        x,y=sample(task,12346,1000,regime)
        p0=.32 if regime=='target_rate_shift' else .07
        rates=[p0,p0,min(.49,p0+.08),min(.49,p0+.08)]
        if regime=='contextual_action':rates=[.08,.32,.32,.08]
        pz=.45 if regime=='nuisance_shift' else task['flip'][1]
        theta=np.array([NOISE,NOISE,pz,*rates])
        true=prob_oracle(task,x,regime)
        got=pred(theta,x,8,t,j)
        err=float(np.max(np.abs(true-got)))
        print(regime,'oracle_max_abs_error',err,flush=True)
        assert err < 1e-12
    print('ALL ORACLE AGREEMENT CHECKS PASS')
if __name__=='__main__':validate()
