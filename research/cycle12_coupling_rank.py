"""Cycle 12c. Conditional predictive rank and task-relevance under coupling.

Rank of matrix P((O_target[t+1],O_target[t+2]) | O[t]=(ot,oz))
for four current observation histories and four possible two-step future
task observations, with fixed actions a0 then a1. Under independent
latent factors rank <=2; under nuisance-dependent target transition
rank can rise to four. Side observation is useful iff nuisance matters.
"""
import json
import numpy as np

NOISE=.14
STATES=[(u,z) for u in (0,1) for z in (0,1)]
ROWS=[(a,b) for a in (0,1) for b in (0,1)]
def transition(action, coupled):
    t=np.zeros((4,4))
    for i,(u,z) in enumerate(STATES):
        flipz=.08
        base=.08 if action==0 else .22
        flipu=(base + (.06 if z else -.06)) if coupled else base
        # Match unconditional mean for coupling regime.
        for j,(nu,nz) in enumerate(STATES):
            pu=(1-flipu if nu==u else flipu)
            pz=(1-flipz if nz==z else flipz)
            t[i,j]=pu*pz
    return t

def emission_joint():
    e=np.zeros((4,4))
    for s,(u,z) in enumerate(STATES):
        for o,(a,b) in enumerate(ROWS):
            e[s,o]=((1-NOISE) if u==a else NOISE)*(
                     (1-NOISE) if z==b else NOISE)
    return e

def emission_target():
    e=np.zeros((4,2))
    for i,(u,z) in enumerate(STATES):
        for y in range(2):
            e[i,y]=(1-NOISE if y==u else NOISE)
    return e

def conditional(coupled):
    t0=transition(0,coupled);t1=transition(1,coupled)
    e=emission_joint();f=emission_target()
    pi=np.full(4,.25)
    posterior=(e*pi[:,None]).T
    posterior/=posterior.sum(axis=1,keepdims=True)
    cols=[]
    for a in range(2):
        for b in range(2):
            vector=t0@(f[:,a]*(t1@f[:,b]))
            cols.append(posterior@vector)
    mat=np.stack(cols,axis=1)
    rank=int(np.linalg.matrix_rank(mat,tol=1e-10))
    ss=np.linalg.svd(mat,compute_uv=False)
    # I(future target two-bit test; nuisance current obs | target obs)
    conditional_mi=0.0
    for ot in range(2):
        row=[ROWS.index((ot,z)) for z in range(2)]
        prior=(mat[row[0]]+mat[row[1]])/2
        for index in row:
            conditional_mi+=.25*np.sum(
                mat[index]*np.log2(mat[index]/prior))
    return mat,rank,ss,conditional_mi

def run():
    record={}
    for coupled in (False,True):
        matrix,rank,s,mi=conditional(coupled)
        print("coupled",coupled,"rank",rank,
              "singular_values",s.round(10).tolist(),
              "conditional_future_MI_bits",round(mi,9))
        record[str(coupled)]={
            "rank":rank,"singular_values":s.tolist(),
            "conditional_mutual_information_bits":mi,
            "rows":ROWS,"columns":ROWS,
            "test_matrix":matrix.tolist()
        }
    path="/home/dtdav/dev/learn/research/cycle12_coupling_rank.json"
    with open(path,"w") as f:json.dump(record,f,indent=2)
    print("saved",path)
if __name__=="__main__":run()
