# Independent v11 information and nested-data audit

Auditor: `v9_artifact_audit`. Canonical source/results were not edited. Execute:

```bash
.venv/bin/python research/audit_precision_information_v11.py
```

Actual numerical evidence is retained in
`audit_precision_information_v11_results.json`.

## Independent information calculation

Instead of repeating the canonical Riccati loop, the audit constructs the joint
Gaussian distribution of s69 and all position observations o0,...,o69 from the
69 preceding motor innovations. One batch Gaussian conditioning calculation
gives the last-action posterior covariance. For R=0 it uses the Moore-Penrose
nullspace projection from an SVD of the observation loading matrix, avoiding
singular inversion and cancellation. All ten analytic cases agree: maximum
posterior-covariance discrepancy 2.15e-15, scalar variance/SD/probability
discrepancy 2.19e-13. The lambda=.2 table rows are analytic comparisons, not
additional v11 control experiments.

Timing is consequential. The last usable observation is o69. Last active u69 is
chosen before xi69; the system then advances to s70, followed by sixteen
zero-command transitions to s86. Current-state uncertainty therefore propagates
through A17, while fresh xi69 propagates through A16. The two variances add
because that innovation is independent of the usable history. Conditioning on
known observation-adaptive issued commands does not create privileged state
information: conditional Gaussian Kalman covariance remains deterministic under
the specified linear model, known parameters/noise law and initial state.

The terminal conditional variance lower-bounds squared error to the target; a
control-dependent nonzero conditional mean adds error. The centered Gaussian
interval probability only upper-bounds final-position success. Whole-stop
success imposes further constraints, and neither bound is an achievable-control
certificate. Future coast observations could reduce later estimation error but
cannot affect the forced-zero physical commands. The bound conditions on history
available for the final action, not on every measurement subsequently collected.

At perfect position sensing, x69 and v69 are known but xi68 in m69 is not yet
visible in position: P69=diag(0,0,.0009). Both that innovation and fresh xi69
affect terminal position. For the slow plant, perfect-position terminal SD is
.3997888421 with whole-stop upper bound .5469838391, whereas perfect **current
state** leaves only the fresh shock and gives the weaker upper bound .7122642919.
Perfect position history must not be identified with perfect current-state access.

## Nested physical experience and fair observation channels

All twelve 512-command prefixes and their 513 actual noisy positions are retained
literally. New commands append to each same physical trajectory. Independent
motor-eliminated simulation confirms that samples t>=513 continue its original
velocity/motor history without reset or a duplicate boundary sample. A deliberate
reset-at512 countercheck disagrees with every actual continuation. Maximum
long-trajectory numerical discrepancy is 2.17e-12. All long fit objectives and
minimum-SSE model choices are independently checked (SSE discrepancy <=3.27e-11).
The estimator receives only commands/positions and the same supplied linear
output-error family; true states are used for simulation and auditing.

The precision intervention changes both actual sensor noise and calibrated filter
R, retaining the same standard-normal sensor streams and motor innovations. It
therefore represents a genuinely different observation channel, not posthoc
filtering of the same coarse measurements. Training still has sensor SD .2 and
zero process noise in both data conditions. The perfect-position table is
analytic only; actual control compares sensor SD .2 and .02.

All 144 physical episodes are independently replayed from logged actions using
the eliminated motor recurrence. Sensor streams, filter estimates, metrics,
action bounds and zero tail are reconstructed. All 10,080 saved plans are
independently differentiated to check shifted-warm cost, selected cost,
projected-gradient norm and first executed command. State/observation discrepancy
is <=2.36e-14, filter estimate <=8.89e-16, metric <=2.23e-14, plan cost <=9.10e-13,
projected gradient <=2.68e-12. Every first action agrees exactly. The 48 SD-.2
short-data/known-parameter episodes reproduce their corresponding v10 episodes
as exactly equal dictionaries, confirming alignment.

Summary means, quantiles, Monte Carlo SEs and success counts are independently
recomputed. SEs use twelve fitted-model seed clusters, each averaging two tasks.
The 24 task trials within a condition are not 24 independent fitted models.

## Interpretation

Extra physical training and better sensing have different, unmatched costs.
This is not a resource-optimal allocation experiment or a universal claim that
more data is unhelpful. Known parameters are a privileged information diagnostic;
the finite training sets already perform similarly on this one fixed plant.
Supplied dynamics/observation families, noise scales, zero initialization,
controller, task timing and tolerances remain substantial assumptions.

Observed success is below the population upper bounds (0/24 versus about .391
at SD .2 and 3/24 versus about .505 at SD .02). The bounds establish unavoidable
conditional uncertainty under the declared protocol; they do not explain all
observed failure or eliminate planner/filter design as further limitations.

No unresolved implementation discrepancy was found in the final audited source
and results. No novelty, broad robotics generality or completion of the persistent
learning objective follows from this class-specific information diagnosis.

## Subsequent within-cycle correction: planner/executor deadline mismatch

After this initial data-versus-sensing audit, the independent theory auditor
identified that the planner continued to propose eight future free controls even
when the executor would shut down sooner. The known-parameter conditional
endpoint mean was substantially biased. The added deadline experiment and its
independent audit in `audit_deadline_control_v11.md` show that this avoidable
planning/task mismatch dominated the original endpoint conditional risk. The
covariance bounds and numerical/data checks above remain valid, but they must
not be used to attribute the original failure chiefly to missing state
information. Preserve this correction when interpreting the earlier protocol.
