# Second post-v17 candidate: implicit relation conditioning rejected

2026-10-20. **TESTED NEGATIVE, exploratory.** The frozen continuation criterion
fails. The learned implicit relation uses new-system evidence, but earns no
advantage over the matched explicit model or strong ALPaCA/CoDA adaptation cores.
This ends the second candidate and the current two-candidate attempt. No rescue
training, ridge/matrix tuning, timing benchmark, control rollout or new ledger is
justified by these outcomes. The ultimate learning-mechanism objective is ACTIVE
and unachieved. Rejection of two constructions is not a general impossibility.

## Actual result

Standardized forecast MSE, mean across eight independent fresh bodies in each
split; one trained seed per method. Support is a prefix of trajectory 0, queries
are all 672 intervals of trajectories 1–7. Physical groups, observation features
and noiseless finite velocity-difference labels are supplied. No physical
parameter is available to the learner.

| Method | ID, 16 | ID, 64 | OOD, 16 | OOD, 64 |
| --- | ---: | ---: | ---: | ---: |
| implicit | 0.019851 | 0.017576 | 0.710211 | 0.578818 |
| explicit | 0.020466 | 0.021797 | 0.707521 | 0.564881 |
| alpaca | 0.020799 | 0.017168 | 0.636735 | 0.464097 |
| coda | 0.020420 | 0.018103 | 0.609899 | 0.306228 |

Implicit OOD error is about 16% / 89% worse than CoDA at 16 / 64 support points.
It also slightly loses to the explicit model in both OOD columns. Correct support
improves on zero support (OOD 3.027605) and wrong-body support (3.996884 / 4.230778).
Thus this operation uses useful evidence, but its additional structure gives no
practical gain here. ID parity does not rescue OOD failure. Neither the 10%
accuracy gate nor the within-5% prerequisite for the efficiency route passes.
No speed claim is made from training logs.

No recorded implicit query has min singular value below the frozen 1e-4 threshold,
including wrong-body-support evaluations. The minimum is 0.668801; maximum
condition number is 1.497226. All four fits and all validation checkpoints are
finite and pass the declared conditioning check. There is no numerical-failure
basis for a repair branch. This is finite observed stability, not an OOD theorem.

## Mechanism and fair comparison scope

Learn f0, vector basis b and matrix basis A, infer context c from the ordinary
support design b(x)-A(x)y by ridge, then solve
[I+sum c_j A_j(x)]y=f0(x)+sum c_j b_j(x). Query prediction loss trains the features.
A=0 gives a nested explicit model. Implicit identification and rational families
are established; this is a candidate learned conditioning rule, not a newly
established general principle. Read implicit_relation_plan.md and the independent
premise audit for derivation, errors-in-variables/gauge hazards and prior art.

Both new models received exactly the frozen two learning rates and 2,000 steps.
Original train/validation and the two original fits of each ALPaCA/CoDA core are
reused. Original validation selects all checkpoints and learning rates. Fresh
holdout bodies were generated from the declared roles before fitting; original
test bodies are not rescored. The stored selection precedes fresh test scoring.
All four selected models use LR .003, step 2,000; CoDA adaptation LR remains .01.
No choice is tuned against new test values. This is another exploratory attempt,
not blind confirmation after choosing a formulation from earlier findings.

Parameter counts: implicit 17,459, nested explicit 10,643, ALPaCA 7,904,
CoDA 44,050. The extra matrix network means the nesting comparison is not a
capacity-controlled attribution. The existing cores have explicit reproduction
limits in error_transport_baselines.md; no full published robotics benchmark is
reproduced. Shared exact q/v and body identity simplify the problem substantially.
No claims about noisy sensing, unidentified task boundaries, robot control,
humanoids, universal learning, or statistically established method rankings follow.

## Reproducibility, costs and independent scope

- Data: implicit_relation_data.py and implicit_relation_holdout*.npz/report.
  New sampling adds 16 bodies, 128 trajectories, 12,288 physical intervals /
  614.4 simulated seconds. Earlier training/validation/test acquisition cost
  43,008 intervals / 2,150.4 seconds. Total unique acquisition across both
  candidates is 55,296 intervals / 2,764.8 seconds; shared data is not free or
  counted twice. The owner PC and existing .venv were used, without installation.
- Fitting/scoring: implicit_relation_screen.py. Four JSON reports retain every
  training loss, checkpoint validation body scores/conditioning, chosen weights,
  seeds, parameter counts, resources and source/data guards. Selection JSON,
  raw predictions and per-body results remain. Fit artifacts refuse overwrites;
  reproduce into a copied checkout or fresh artifact paths.
- Frozen gate: implicit_relation_decision.py/.json. Its conservative ID guard
  evaluates both support budgets; the failed OOD comparisons decide rejection
  regardless of that interpretation. Gate output is a screening decision,
  not a significance test.
- Independent model fixture: check_implicit_relation_model.py/_results.json.
  Augmented NumPy least squares and separate query solves match to 2.3e-16;
  zero-support and A=0 nesting are exact, adaptation gradients nonzero and
  checked against directional finite differences. This is not network training
  replication, a physics replay, or a universal invertibility certificate.
- Independent result check: check_implicit_relation_results.py/.json/.md PASS.
  It recomputes all 64 raw forecast arrays / 512 body-budget errors using original
  train-only scales, checks all eight fit reports and saved validation selection,
  exact reuse of baseline selection, hashes and gate arithmetic. Maximum float32
  reduction discrepancy is 1.30e-6, immaterial to the failed comparisons. Stored
  conditioning flags are checked against saved diagnostics; matrices/SVDs,
  network forwards, optimizers and physics are not independently replayed.
  The direction critic independently recommends ending this two-candidate attempt.

Both-LR summed fit wall seconds (concurrent processes, so not campaign elapsed):
- implicit: 21.752 seconds.
- explicit: 19.727 seconds.
- alpaca: 15.517 seconds (reused earlier fits).
- coda: 16.737 seconds (reused earlier fits).

All ten original fits cost 109.990 summed fit seconds; all four new fits
cost 41.480. Neither is an inference latency benchmark.
Raw query/test values and validation reuse are fully disclosed; no independent
training-seed confirmation is claimed or owed after failure.

## Decision and next direction

Retire this construction and close the two-candidate campaign. Both proposed
extra inductive biases lost to established learned-prior/context adaptation under
these conditions. Do not generalize that observation into failure of functional
updates, implicit modeling or first-principles research as a whole.

Before a new campaign, identify a concrete capability limitation of the strongest
existing mechanism that matters for unfamiliar physical decision making. These
screens do not establish such a limitation merely because their errors are
nonzero. Do not automatically invent a third feature head, add an environment,
or revive exhaustive v17 physics diagnostics. The next direction decision needs
a consequential problem and a derived operation with a serious competing account.
