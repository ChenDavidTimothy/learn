# Closest existing mechanisms: early source reconciliation

2026-10-20. Read during candidate selection, before dataset collection or fitting.
This note limits novelty claims; no full published benchmark reproduction implied.

## ALPaCA — existing learned adaptation

Harrison et al., *Meta-Learning Priors for Efficient Online Bayesian Regression*:
https://arxiv.org/html/1807.08912 . Root read problem formulation and sections3--4,
including equations5--9 and joint learned feature/prior objective. The method
learns nonlinear features plus a matrix-normal output-layer prior and conditions
it analytically on new samples. Its posterior mean is learned-feature ridge
regression; predictive covariance and likelihood are part of the original method.
A scalar/separable linear error-transport kernel can reproduce this conventional
update. Our candidate must beat competent learned-prior adaptation, not merely
pooled regression, and a mean-only squared-loss adaptation is a scoped variant,
not a faithful full ALPaCA implementation. Original repository:
https://github.com/StanfordASL/ALPaCA . Age does not establish current maintenance.

## CoDA and more recent low-rank adaptation

Kirchmeyer et al., *Generalizing to New Physical Systems via Context-Informed
Dynamics Model*: https://arxiv.org/html/2202.01889 . Root inspected sections2--3.3,
including whole-network theta_e=theta_shared+W*context_e, loss plus locality
regularization, trainable per-environment contexts and new-context optimization.
It directly challenges claims that a learned transferable adaptation operation is
new. Its original training integrates trajectories; finite observed-transition
regression would adapt its parameterization to this instrument, with that change
disclosed. Original source: https://github.com/yuan-yin/CoDA . GEPS2024 is a newer
low-rank dynamics adaptation reference:
https://papers.neurips.cc/paper_files/paper/2024/file/82844e428d9163a9f94830dc03af4f9c-Paper-Conference.pdf .
Root inspected the abstract/discussion and comparison scope so far; complete
method reading is required before claiming its reproduction or superiority.

## MetaFun — direct prior art against novelty

Xu et al., *MetaFun: Meta-Learning with Iterative Functional Updates*:
https://proceedings.mlr.press/v119/xu20i/xu20i.pdf . Root inspected introduction
and sections2.1--2.2/3 of the functional encoder/update construction (independent
critic also read these sections). It explicitly learns
iterative function updates and context-target interactions; thus the broad
proposal 'learn how an error changes predictions elsewhere' is already established.
Residual-conditioned output-matrix transport is at most a candidate variant,
not an invented paradigm. Original code:
https://github.com/jinxu06/metafun-tensorflow . A useful result must establish an
actual data/compute/capability advantage and expose where any advantage comes from.
