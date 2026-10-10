"""Algebraic identifiability check for the Phase IV Autoencode example."""
import numpy as np
from cycle27_transfer_identifiability import COEF
W1,W2=52,104
equiv=[(-1,0,102),(-1,1,50),(-1,2,-2)]
for t in range(W1-1,2*W1-1):
    indices=[a*t+b*W1+c for a,b,c in equiv]
    assert len(set(indices))==1,(t,indices)
assert any(len({a*t+b*W2+c for a,b,c in equiv})>1
           for t in range(W2-1,2*W2-1))
design=np.array([[W1,1],[W2,1]])
assert np.isclose(abs(np.linalg.det(design)),52,atol=1e-10)
assert np.linalg.matrix_rank(design)==2
assert np.linalg.matrix_rank(np.array([[W1,1],[W1,1]]))==1
print('PASS: confounding at fixed length; full-rank variation across lengths')
