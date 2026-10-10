"""Cycle 03: exact obstruction to low-order mechanisms representing parity-4.

For uniform x in {0,1}^10, a predictor with Fourier degree <=3
cannot correlate with 4-variable parity. Jensen's inequality yields
irreducible average excess log loss log(2)-binary_entropy(epsilon),
even for an optimally chosen low-degree prediction function.
"""
import itertools, json, math, random
import numpy as np
from cycle02_unknown_structure import (
    D, GRAMMAR, NOISE, XALL, PRIOR
)
N = len(XALL)
def h(p):
    return -p*math.log(p)-(1-p)*math.log1p(-p)
def coeffs(f):
    arr = f.astype(float).copy()
    width=1
    while width<N:
        arr=arr.reshape(-1,2,width)
        upper=arr[:,0,:].copy()
        lower=arr[:,1,:].copy()
        arr[:,0,:]=upper+lower
        arr[:,1,:]=upper-lower
        arr=arr.reshape(N)
        width*=2
    return arr/N

def test():
    r=random.Random(97011)
    bits=[0,2,4,9]
    truth=np.bitwise_xor.reduce(
        XALL[:,bits].astype(np.uint8),axis=1).astype(float)
    Fourier=coeffs(truth)
    high=sum(float(Fourier[s]**2)
             for s in range(N) if s.bit_count()>3)
    mask=sum(1<<i for i in bits)
    other=max(abs(float(Fourier[s]))
              for s in range(N) if s not in (0,mask))
    assert np.isclose(high,.25,atol=1e-12)
    assert np.isclose(other,0,atol=1e-12)
    floor=math.log(2)-h(NOISE)
    pinsker=2*(1-2*NOISE)**2*high
    print("Fourier high-degree squared mass:",high)
    print("Pinsker bound (nats):",round(pinsker,9))
    print("Tight Jensen/orthogonality bound:",round(floor,9))
    print("Each fixed-grammar rule must have Fourier degree <=3")
    losses=[]
    q=NOISE+(1-2*NOISE)*truth
    for _ in range(100):
        w=np.zeros(GRAMMAR.shape[1])
        ids=r.sample(range(GRAMMAR.shape[1]),30)
        weights=np.array([r.expovariate(1) for _ in ids])
        weights/=weights.sum()
        w[ids]=weights
        p=NOISE+(1-2*NOISE)*(GRAMMAR@w)
        excess=float(np.mean(-(q*np.log(p)+(1-q)*np.log1p(-p)))-h(NOISE))
        assert excess+1e-10>=floor,(excess,floor)
        losses.append(excess)
    column_zero=next(i for i in range(GRAMMAR.shape[1])
                     if np.array_equal(GRAMMAR[:,i],XALL[:,0]))
    column_one=next(i for i in range(GRAMMAR.shape[1])
                    if np.array_equal(GRAMMAR[:,i],1-XALL[:,0]))
    constant=(GRAMMAR[:,column_zero]+GRAMMAR[:,column_one])/2.0
    assert np.allclose(constant,0.5)
    print("Achievable constant model excess loss:",
          round(math.log(2)-h(NOISE),9))
    print("min of 100 random mixture losses:",round(min(losses),9))
    path="/home/dtdav/dev/learn/research/cycle03_parity_bound.json"
    with open(path,"w") as f:
        json.dump({"D":D,"r":4,"max_degree":3,"noise":NOISE,
                   "high_degree_energy":high,
                   "pinsker_bound_nats":pinsker,
                   "tight_loss_floor_nats":floor,
                   "verified_mixtures":len(losses),
                   "min_random_mixture_loss_nats":min(losses)},
                  f,indent=2)
    print("saved:",path)
if __name__=="__main__":
    test()
