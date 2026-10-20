# Independent heldout command-fork audit

2026-10-20. **PASS** for the prepared audit executed unchanged after the artifact
became available. No learner, model selection, goal, fork context or audit scope
changed in response to outcomes. No canonical source/artifact was edited.

Source: `audit_cartesian_stream_v17_forks.py`. Actual results:
`audit_cartesian_stream_v17_forks_results.json`. The hash chain binds source,
ordinary/evaluator data, frozen weights, fit/selection JSON, learning/local code,
acquisition and design. It additionally verifies actual weight/source hashes that
the canonical load function itself did not assert.

## Independent verification

The evaluator has32contexts: eight strictly heldout test trajectories at indices
16,40,120,240. Each uses the five predeclared commands for two intervals, producing
160counterfactual preparations/320flows/32privileged simsec. No new sensor noise or
ordinary training experience is generated. Both mechanisms see only the unchanged
causal noisy camera/action prefix. Local coefficients are fitted once per context
and frozen across the command bank; there is no fork-driven append or selection.

The audit imports no canonical predictor/acquisition/mechanics implementation.
Independent NumPy tanh forward passes reconstruct all nonlinear member forecasts
within4.44e-16. Augmented weighted least-squares SVD, rather than the canonical
normal-equation solve, reconstructs all retained-local forecasts within2.38e-14.
Training-only scales, whole-episode bootstrap rows and validation-based candidate
indices independently check. Frozen network weights are reused: this is independent
prediction reconstruction, not a second training optimization claim.

Independent Hamiltonian RK4 replays ALL320fork flows: maximum state discrepancy
6.29e-13 and Cartesian discrepancy1.44e-13. A separate adaptive Radau check covers
66intervals, with maximum state discrepancy3.41e-13. Every summary field and
resource counter reconstructs exactly. Stored RHS evaluations sum to8,824.

## Verified negative finding and its limit

| Frozen selected mechanism | Contrast coordinate MSE | MSE / zero-contrast MSE |
|---|---:|---:|
| Nonlinear history predictor | 1.1528399265e-5 | .9696996625 |
| Retained local predictor | 1.1369977870e-5 | .9563742069 |

Zero-contrast coordinate MSE is1.1888628727e-5. Thus the learned contrasts provide
only small mean improvement over predicting no command contrast in this fixed
instrument. The mean true contrast norms at the two future camera times are
.001512m and.006056m. Nonlinear predicted norms are.000340m/.001000m; local norms
.000636m/.002517m. Smaller norms do not alone establish a pure gain-calibration
error: directions and context dependence can also be wrong.

Absolute ensemble-mean coordinate MSE is2.62055e-5 nonlinear and1.86426e-5 local.
It must not be described as successful action learning merely because endpoint
trajectories are smooth or absolute positions are large. The fork check exposes
precisely that distinction. Conversely these results do not prove zero useful
learning, global incapacity of either family, insufficient history in principle,
or necessity of a different architecture. They are one learned corpus and32
contexts clustered within eight test trajectories, not independent-body statistics.

Potential explanations include finite-data/locality, regularization, loss allocation
between moving drift and small command effects, representation or selection, and
remaining sensory ambiguity. This audit does not choose among them. The original
frozen charged-control pilot can determine whether those imperfect forecasts are
still useful for its actual goals; no control outcome is implied here. Preserve
this negative evidence even if later online revision or ordinary feedback works.
