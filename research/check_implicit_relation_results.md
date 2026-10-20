# Independent second-screen result check

**TESTED: the saved evidence supports retiring this candidate.** Run
`.venv/bin/python research/check_implicit_relation_results.py` to reproduce
`check_implicit_relation_results.json`. This is a proportional result and
selection check, not certification of a learning framework.

The checker independently reconstructs normalization from the original 32 training
bodies, then recomputes normalized and physical-unit MSEs for all 64 saved
prediction arrays and 512 body/budget scores against the new holdout labels. It
checks all eight fit reports, all four checkpoints per training learning rate,
validation eligibility and selection, selected finite weights and stored scales,
and source/data/report/weight provenance. ALPaCA and CoDA use exactly their frozen
first-screen selections. Static source review confirms support trajectory 0,
query trajectories 1–7, and no evaluator arrays as model inputs. The largest
saved-versus-independent MSE reduction difference is 1.30e-6, consistent with the
declared float32 versus binary64 reduction tolerance.

At support 16/64 the implicit candidate's mean OOD normalized MSE is
0.710211/0.578818, versus CoDA's 0.609899/0.306228 and the nested explicit model's
0.707521/0.564881. The prespecified 10% advantage and efficiency-branch accuracy
prerequisite both fail. Correct support improves predictions over wrong-body
support, but this does not establish competitive performance. No saved query
conditioning failure occurs; the smallest saved singular value is 0.668801.
The saved rejection decision agrees with independently recomputed gate arithmetic.

No network forward replay, singular-value recomputation, retraining, trajectory
physics replay or timing benchmark was performed. Conditioning checks validate
stored diagnostics and their eligibility flags, rather than independently
recovering the matrices. The screen has one trained seed, supplied exact full
configuration/velocity features, noiseless finite-transition targets and one
dynamics family. The ALPaCA fixed-noise and CoDA transition-loss adaptations,
parameter counts and compute disparities remain part of its scope. This result
rejects this implemented second candidate under the declared screen; it does not
rule out implicit relations generally or establish control competence.
