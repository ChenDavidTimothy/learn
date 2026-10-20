#!/usr/bin/env python3
"""Independently executable finite-dimensional constraint-rank checks for v6."""
import json
import numpy as np

SEED=20261020+1100000
rng=np.random.default_rng(SEED)
rows=[]
for d in (3,6,12):
    frame=np.linalg.qr(rng.normal(size=(d,d)))[0]
    basis=[]
    for i in range(d):
        for j in range(i,d):
            e=np.zeros((d,d))
            e[i,j]=1.
            e[j,i]=1.
            if i!=j:e/=np.sqrt(2)
            basis.append(e)
    for k in range(d+1):
        S=frame[:,:k]
        constraints=np.column_stack([np.r_[(q@S).ravel(),np.trace(q)] for q in basis])
        rank=int(np.linalg.matrix_rank(constraints,tol=1e-10))
        nullity=len(basis)-rank
        predicted=max(0,(d-k)*(d-k+1)//2-1)
        assert nullity==predicted
        rows.append({'dimension':d,'context_dimension':k,'constraints_rank':rank,
                     'actual_nullity':nullity,'predicted_nullity':predicted})
print(json.dumps({'seed':SEED,'numpy':np.__version__,'rank_tolerance':1e-10,
                  'checks':rows},indent=2,allow_nan=False))
