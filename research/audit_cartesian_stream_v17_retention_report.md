# Independent actual-decision audit: frozen acquired-library ablation

2026-10-20. **PASS** for all eight ablation episodes, 720 ordinary decisions and
2,160 independent local member fits. The auditor source/scope was prepared before
reading the ablation outcomes. Canonical source/results were not edited. Physics,
paired sensor streams and true task-score reconstruction have a separate auditor.

Reproduce with `.venv/bin/python research/audit_cartesian_stream_v17_retention.py`.
Executable equations and actual results are preserved in that source and
`audit_cartesian_stream_v17_retention_results.json`. Its hashes guard the immutable
retention source/design, result/arrays, paired parent result/arrays, original
ordinary/fit/weights and independent audit helpers. Actual source remains the
preflight-checked `94e3e5d7ac4e415aa6caecbe1c8042d1ef6a14d63eb5e6fa4a4eb46b4ca0f598`.

## Reconstruction and intervention validity

No canonical learning/local/control implementation is imported; no evaluator-state
array is read. The audit reconstructs the same three original 2,880-row bootstrap
libraries independently from the ordinary acquisition. At EVERY actual renewal it
fits the full affine local correction using independent weighted residual-
augmentation least squares. It compares all saved coefficients, queries and fit
diagnostics. The libraries never acquire an online row. All 90 completed-real-pair
notifications per episode, zero additions and final2,880/member counts match.

For every decision, it independently checks coherent member paths and every
shifted/zero/optimizer-final/selected modeled cost, initial-competitor and final
acceptance/fallback policies, selected gradient, projected gradient and actual
first command held twice. Command/time/phase/current-goal indices and previous
plan continuity survive all goal switches. Coefficients are indexed directly by
renewal because the wrapper saves them independently of the parent's exact-tag
capture branch. Every local fit is conditioned only on the current actual noisy
history and the unchanged acquired corpus.

Paired episode preparation/warmup/noise seed paths match the immutable online
local episode. Warmup commands, the first17 noisy readings and all three goals
match exactly. Importantly, the entire FIRST selected plan equals the online
plan exactly in all eight cases, before the online mechanism can append its
first new example. Later divergence therefore does not arise from a changed
initial model, planner or prepared prefix. True paired-noise verification after
trajectories diverge belongs to the physical auditor.

All 8,943 modeled calls reconcile with recorded SciPy counts. There are zero
recorded optimizer failures and zero fallbacks. Independently reconstructed
projected-gradient quantiles are median5.78e-7,90th3.20e-6,99th1.74e-5,
maximum3.11e-5. The maximum is case5/renewal41/time98, ending on relative function
reduction. No optimizer path was rerun and no global physical-control claim is
made. The frozen local planning subproblem remains a strictly convex quadratic
under its fixed coefficients, as established in the parent decision report.

## Numerical agreement

| Quantity | Maximum absolute discrepancy |
|---|---:|
| actual query | 0 |
| saved local coefficient vs independent augmented least squares | 4.34e-11 |
| local fit diagnostics | 2.30e-13 |
| local regularized stationarity residual | 1.01e-13 |
| every reconstructed modeled cost | 1.06e-11 |
| selected gradient | 1.95e-12 |
| selected projected gradient | 2.24e-13 |

## Pairing arithmetic and interpretation boundary

The following reported task-cost differences are checked arithmetically against
the paired immutable parent records. The true physical scores/windows themselves
are NOT independently reconstructed by this ordinary-decision auditor.

| paired case | frozen-minus-online recorded task cost | recorded frozen/online precise goals |
|---|---:|---:|
| 0 | .001234851321 | 3/3 |
| 1 | .000469103382 | 3/3 |
| 2 | .000136723995 | 3/3 |
| 3 | .000166131027 | 3/3 |
| 4 | 11.113761005127 | 0/2 |
| 5 | 1.877081762521 | 1/2 |
| 6 | .019681120044 | 0/3 |
| 7 | .019180086619 | 0/3 |

Thus the preserved records contain13/24 frozen versus22/24 online windows; four
physical preparations have two paired noise repeats each. This is not eight
independent bodies or independent-corpus evidence. The frozen mechanism still
uses acquired experience, current history, local refitting and feedback. It is
not a no-learning/no-adaptation baseline. Any verified total closed-loop effect
of appending includes its subsequent influence on visited histories and actions;
it does not isolate a particular row's value, identify a physical parameter or
establish universal online-learning necessity. Both the successful frozen cases
and the two retained online failures must remain visible. Independent physics
and overall direction review are needed alongside this decision audit before
interpreting the ablation scientifically.
