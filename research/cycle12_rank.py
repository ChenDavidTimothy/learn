"""Cycle 12a. Rank separation for task-relevant vs full predictive state.

For independent binary latent Markov factors with distinguishable noisy
observations, the full one-step observable Hankel matrix factors as a
Kronecker product of nondegenerate 2x2 observable joint matrices, so
its algebraic rank is 2^(n+1). Task-only future observations ignore
nuisance factors and have rank two. This is classical PSR theory.
"""
import json
import numpy as np
from numpy.linalg import matrix_rank, svd
from scipy.linalg import block_diag

def joint_one(p_flip, emission_error):
    t=np.array([[1-p_flip,p_flip],[p_flip,1-p_flip]])
    e=np.array([[1-emission_error,emission_error],
                [emission_error,1-emission_error]])
    return e.T@np.diag([.5,.5])@t@e

def run():
    cases={}
    for nuisance_count in range(0,9):
        single=[joint_one(.13,.17)]
        single += [joint_one(.03+.02*(j%5),.10+.01*(j%4))
                   for j in range(nuisance_count)]
        full=single[0]
        for item in single[1:]: full=np.kron(full,item)
        svals=np.linalg.svd(full,compute_uv=False)
        # Numerical rank unreliable for large dimensions and tiny values,
        # so also use the exact Kronecker rank product theorem.
        exact_theoretical=2**len(single)
        empirical_rank=int(np.sum(svals > 1e-14*svals[0]))
        cases[str(nuisance_count)]={
          "full_hankel_size":len(full),
          "full_hankel_rank_theoretical":exact_theoretical,
          "rank_numerical_threshold":empirical_rank,
          "task_only_hankel_rank":2,
          "min_singular_value":float(svals[-1]),
          "max_singular_value":float(svals[0])
        }
        print("nuisance",nuisance_count,"full rank",exact_theoretical,
              "task rank",2,"numeric",empirical_rank,flush=True)
    path="/home/dtdav/dev/learn/research/cycle12_rank.json"
    with open(path,"w") as f:json.dump({
      "assumptions":"independent two-state Markov factors, separate noisy observations, stationary initial state",
      "theory":"H_full is Kronecker product of H_i, H_task involves target only",
      "cases":cases},f,indent=2)
    print("saved",path)
if __name__=="__main__":run()
