# Independent v13 bounded-gradient artifact audit

Auditor: `v9_artifact_audit`. No canonical experiment module is imported by the
audit, and no canonical source/result was edited.

```bash
.venv/bin/python research/audit_gradient_reuse_v13.py
```

Actual numerical evidence is retained in `audit_gradient_reuse_v13_results.json`.

## Independent pathwise derivative verification

The audit implements the six-coordinate stationary controller through a separate
motor-eliminated velocity recurrence. Derivatives include log/logit gains,
current observation-dependent position smoothing, the second smoothing memory,
previous velocity memory, both **actually issued** past commands, and all future
physical effects. The clip derivative is zero outside the interval and one
inside; no straight-through surrogate is used. Exact boundary points are
nondifferentiable, so the claim is a pathwise derivative almost everywhere.

Independent finite differences reproduce the original diagnostic problem.
At epsilon1e-6, the velocity-memory coordinate changes one clip active set and
gives discrepancy.331221. This is not a branch-preserving derivative test.
At the two fixed smaller steps1e-7/1e-8, **no** plus/minus/base mask differs;
maximum independent discrepancies are1.20e-8 and1.29e-7. The unchanged derivative
is verified. The initial failed check and its correction are preserved rather
than erased; no controller or derivative formula was changed to obtain a pass.
The successful check uses37 virtual rollouts,50912 transitions, not the earlier
13-rollout17888 count. Final source/result metadata matches this correction.

## Learning data, banks, candidate selection and budgets

All32 physical learned models and original physical/virtual CEM winners match
their v12 parent records and exact hashes. Fit and validation banks are regenerated
exactly from disjoint+20b/+30b domains, and fresh physical tests use+40b streams,
disjoint from all previous physical/virtual CEM roles. Sixteen virtual episodes
per bank alternate the two prescribed goals, eight draws per goal. Memories and
physical model states reset to zero per episode.

All **96** optimizer endpoint objectives/projected gradients and prescribed
starts are independently recomputed. The three candidates are optimized only on
predicted learned-model states. True physical states/parameters are not used in
optimization or validation. All **160** validation candidate scores are checked,
including the two unmodified original policies, and every selected coordinate
is the actual minimum virtual-validation candidate. Selection uses no fresh test
outcome and remains susceptible to finite-validation-bank optimism.

All32 selections are optimized candidates:8 from the physical-CEM start,11 from
the virtual-CEM start,13 from the generic start. One of96 attempts is nonconverged:
plant3/replicate7/generic start, status0,160 evaluations, projected gradient
6.28e-6. It is retained and **not selected**. Local optimizer convergence does
not establish a global optimum, especially with clipping and nonconvex feedback.

Budget arithmetic is verified:4,793 cached rollout evaluations produce6,595,168
fit-optimization virtual transitions; validation220160, retained trajectory
replays88064, derivative verification50912. These are recorded successful-run
counts, with the initial aborted derivative check separately documented as
development work. Cache counters are checked arithmetically; complete optimizer
paths were not independently rerun. Physical training added is zero. Fresh
physical evaluation adds22016 transitions/256 resets. The same learned models
are reused, so new test streams are not new independently fitted models.

## Complete trajectory, solver and metric reconstruction

The audit reconstructs all **1,024 retained virtual episodes** and **256 fresh
physical tests**, with continuing active feedback and motor noise throughout
testing. Every direct action, observation, physical state, model filter belief
and task metric is checked. All **5,504 MPC plans** are independently
differentiated, including remaining-horizon lengths, shifted warm costs,
selected costs and projected gradients. No goal-switch state/memory reset or
future-goal revelation is introduced.

Maximum discrepancies: virtual objective5.56e-17, policy projected gradient
2.23e-16, validation score5.56e-17, virtual states1.56e-15, physical states and
observations1.60e-14, filter beliefs8.89e-16, metrics1.17e-14, MPC projected
gradients1.68e-12. All direct/MPC issued commands agree exactly. Every summary
and paired MC SE agrees exactly, using eight independent physical-training seed
clusters per plant and two tasks averaged within each cluster. Sixteen task
trials are not sixteen independent learned policies.

## Mean cost and late precision are different capabilities

The mean-cost improvement is a constructive demonstration within the same
supplied stationary policy family. It does not resolve precise station-keeping
capacity. Across64 fresh tasks, late-window successes are physical-CEM14,
virtual-CEM16, gradient-selected1, nonlinear-MPC25. For the gradient selection,
one failure is position-only,27 velocity-only,35 both: **62 of64 tasks violate
the velocity threshold**. Every outcome is retained.

As a descriptive equal-task pooled decomposition, physical-CEM to gradient
changes normalized position cost.339797 to.270589, velocity cost.009869 to.013201,
and effort cost.000174 to.000549. Late position RMSE averages.199636 versus
.200485, while late velocity RMS rises.110705 to.160634. Nonlinear MPC's pooled
late velocity RMS is.097406. These decompositions are evaluator-only diagnostics,
not new fitting objectives or substitutes for the frozen primary metric.

Thus improvement in the selected quadratic mean score is not broader capability
dominance. Early tracking, noisy velocity/action variability, absolute late
thresholds and simultaneous ten-step requirements trade off. The diagnostic
does not optimize that whole late-window success event, and it does not justify
claiming reliable stopping or infer the event from a low scalar cost.

## Scope and continuation

This is a model-using hybrid with strong supplied dynamics/observation vocabulary,
known noise/reset facts and substantial extra virtual computation. The physical
model is learned; exact true plant states are not supplied as rewards. Fixed-bank
optimization and candidate validation are legitimate computation but do not
provide fresh-data or global optimality guarantees. Better heldout mean behavior
can refute an absolute claim that this family cannot attain that mean behavior;
failure would not prove incapacity. Neither result establishes architectural
necessity, universal model superiority or competitive robotics capability.

No unresolved artifact discrepancy was found. This completes the one prescribed
scalar search diagnostic; its negative precision evidence should be retained
when moving to a more consequential coupled or missing-vocabulary learning test.
