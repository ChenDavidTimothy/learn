"""Noise-robust passive parity learner using classical RANSAC over GF(2).

Draw D+1 labeled random inputs; if noise-free and full rank,
the parity coefficients and intercept are exactly determined.
Score each candidate on the entire labeled set. With fixed noise,
repeat enough times to have high recovery probability.
This is deliberately an established baseline, not a new method.
"""
import math, time
import numpy as np
from cycle05_interventions import D,EPS,bit_parity

TRIALS=1500
def solve_gf2(x,y):
    basis={}
    for i in range(len(x)):
        row=int(x[i]) | (1<<D)
        target=int(y[i])
        while row:
            pivot=row.bit_length()-1
            if pivot in basis:
                b,r=basis[pivot]
                row^=b
                target^=r
            else:
                basis[pivot]=(row,target)
                break
    if len(basis)!=D+1:
        return None
    sol=0
    for pivot in sorted(basis):
        b,r=basis[pivot]
        if r ^ ((b&sol).bit_count()&1):
            sol |= (1<<pivot)
    return np.uint64(sol&((1<<D)-1)),(sol>>D)&1

def solve_passive(x,y,seed,max_trials=TRIALS):
    rng=np.random.default_rng(seed)
    best=(None,None,2*len(y))
    best_trial=0
    bound=EPS*len(y)+3.5*math.sqrt(len(y)*EPS*(1-EPS))
    for step in range(max_trials):
        ix=rng.choice(len(x),D+1,replace=False)
        pair=solve_gf2(x[ix],y[ix])
        if pair is None:continue
        mask,intercept=pair
        guess=bit_parity(x&mask)^intercept
        mistakes=int(np.count_nonzero(guess!=y))
        if mistakes<best[2]:
            best=(mask,intercept,mistakes)
            best_trial=step+1
            if mistakes<=bound:break
    return best[0],best[1],best[2],best_trial

def smoke_test():
    from cycle05_interventions import world,draw_noise
    acc=[]
    elapsed=[]
    for n in range(12):
        seed=100010+n*73
        rng=np.random.default_rng(seed)
        true,mask,intercept=world(rng,"dense_parity12")
        x=np.random.default_rng(seed+1).integers(
            0,1<<D,size=824,dtype=np.uint64)
        y=draw_noise(x,true,np.random.default_rng(seed+2))
        t=time.perf_counter()
        m,b,loss,attempts=solve_passive(x,y,seed+3)
        elapsed.append(time.perf_counter()-t)
        ok=m is not None and m==mask and b==intercept
        acc.append(ok)
        print(n,"exact",ok,"mistakes",loss,
              "attempts",attempts,"seconds",round(elapsed[-1],3),
              flush=True)
    print("recovered",sum(acc),"/",len(acc),
          "avg_sec",sum(elapsed)/len(elapsed))
if __name__=="__main__":
    smoke_test()
