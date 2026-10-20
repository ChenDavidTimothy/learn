# All-decision posthoc modeled optimization bound

2026-10-20. **PASS:** all 11,520 actual fresh ordinary decisions from corpora
200–207 are included. This is a posthoc diagnostic after physical outcomes were
known. No plan was changed and no model was fitted, optimizer rerun or physical
interaction added. The design was saved before evaluating these bound values.

## Why the actual modeled objective is convex

At a particular decision, all local coefficients and the query are held fixed
throughout the hypothetical horizon. Causal histories and the current goal are
fixed inputs. Every future context is linear in preceding predicted positions
and planned actions; normalization, the learned correction and linear carry are
affine. Induction therefore gives each member's entire 12-sample predicted path
as an affine function of the 12 planned action coordinates. Predicted finite
difference velocities are affine as well. The planner loss is a mean of squared
position errors plus one quarter squared velocities, plus 0.02 times action
norm squared. Its Hessian is positive definite, at least 0.04 times the identity.
There is no refitting, neighbor selection, clipping or other nonlinear operation
inside this fixed learned planning problem. This argument does not apply to a
neural predictor or a query refitted at hypothetical states.

For any differentiable convex f, feasible x and exact g=gradient f(x), the
supporting hyperplane gives f(y) >= f(x)+g dot (y-x). Minimizing its linear part
over y in [-0.5,0.5]^12 gives

    0 <= f(x) - min_box f <= g dot x + 0.5 ||g||_1
                            = sum_i |g_i| (0.5 + sign(g_i) x_i).

This is the standard first-order convex box bound, not a new learning principle.
The final expression is evaluated with an accurate sum of nonnegative terms to
avoid cancellation. A large raw gradient can correctly point outward at a box
optimum and contribute zero to this bound. The objective is nonnegative, so
max(0,f(x) minus bound) is also a lower bound on its modeled optimum.

## Exact artifact and numerical scope

The executable first requires the final all-eight numerical fitting/decision
audit and physical audit to pass. It checks both aggregates, all per-corpus
numerical cache hashes, all physical cache/source hashes and all final raw
artifacts against current files, and repeats the guards after computation.
The final numerical source maps include the documented coefficient roundoff
repair. This diagnostic does not independently refit those coefficients: their
provenance is established by that separate numerical audit.

The canonical chooser recomputes `selected_gradient` at `selected_plan` after
acceptance or fallback. This diagnostic independently reconstructs every actual
selected cost and gradient with the saved exact coefficients, ordinary causal
history, training scales and fixed query through earlier independent NumPy paths
and analytic derivatives. It checks every first planned action against the
issued physical command, the decision clock and current goal. Query and projected
gradient records match exactly; maximum cost discrepancy is 6.82e-13 and gradient
discrepancy 2.84e-14. No reconstruction check fails.

Raw dot-plus-norm and stable bounds differ by at most 1.18e-15. Replaying the
gradient changes the stable bound by at most 1.19e-14. The recorded discrepancy
allowance compares two floating-point evaluations, not either one's unknown
exact-real error. The inequality is an exact-arithmetic theorem; these computed
values are numerically checked evaluations, not outward-rounded interval
certificates. No theorem about exact real gradients is inferred solely from an
optimizer's termination status.

## Actual bounds

All 11,520 selected objectives are positive, so every reported relative ratio is
defined. Relative values divide the bound by the **selected modeled objective**,
not the unknown optimum, and are not physical task-cost percentages. The planner
loss is its unscaled sampled surrogate; the scored physical cost includes the
0.1-second integration factor and true velocities, among other differences.

The absolute bound has mean 2.59946e-6, median 1.02960e-6, 99th percentile
2.62335e-5 and maximum **1.6664783e-4**. The relative bound has median
1.10666e-6 (0.000110666%), 99th percentile 0.00365984 (0.365984%) and maximum
**0.0227143 (2.27143%)**. The maximum absolute and relative bounds occur at
different decisions.

The maximum absolute bound is at corpus 204, frozen variant, case 6, phase 0,
renewal 25, camera index 66: selected objective 121.9735725963, bound 0.0001666478303,
ratio 1.3662618e-6. The maximum relative bound is at corpus 202, retained variant,
case 7, phase 2, renewal 80, camera index 176: selected objective 0.0003838777335,
bound 0.000008719495314, ratio 0.02271425132. Small absolute loss need not mean an
arbitrarily small relative loss near a nearly zero objective.

Each corpus contains 720 decisions per variant. Maxima within a row need not
occur at the same decision:

| Corpus | Retained max absolute | Retained max relative | Frozen max absolute | Frozen max relative |
|---|---:|---:|---:|---:|
| 200 | 0.000109089 | 0.35336% | 6.03535e-05 | 0.485334% |
| 201 | 5.36419e-05 | 0.468835% | 2.84627e-05 | 0.108811% |
| 202 | 8.01213e-05 | 2.27143% | 0.000102316 | 2.10631% |
| 203 | 5.65884e-05 | 0.0840551% | 8.63438e-05 | 0.0231432% |
| 204 | 3.56461e-05 | 1.27364% | 0.000166648 | 0.159265% |
| 205 | 8.18338e-05 | 0.583943% | 4.53922e-05 | 0.137775% |
| 206 | 7.77397e-05 | 0.57152% | 0.000161751 | 0.114622% |
| 207 | 2.06407e-05 | 0.667762% | 1.24815e-05 | 0.592975% |

Each variant/phase below contains 1,920 decisions across eight corpora. Phase
indices are zero-based. All 48 corpus-by-variant-by-phase groups of 240 decisions,
including their precise worst locations, are stored in the JSON artifact.

| Variant | Phase | Max absolute bound | Max relative bound |
|---|---:|---:|---:|
| retained | 0 | 8.01213e-05 | 0.270556% |
| retained | 1 | 0.000109089 | 0.764791% |
| retained | 2 | 5.09834e-05 | 2.27143% |
| frozen | 0 | 0.000166648 | 0.519524% |
| frozen | 1 | 0.000130763 | 2.10631% |
| frozen | 2 | 0.000161751 | 1.14492% |

## What the bound does and does not establish

The result bounds incomplete minimization of each actual fixed learned quadratic
by a small absolute modeled amount. It excludes a large single-decision objective
gap hidden behind the recorded successful solver status, within the numerical
checks. It does not prove that the selected action is physically good, that an
alternative optimizer cannot alter a sensitive closed-loop trajectory, or that
optimization error is causally irrelevant to every failed goal. Horizons,
objective choice, current-goal information, learned consequences and physical
closed-loop robustness remain distinct issues. No new optimizer ladder follows
from this diagnostic.

Convexity and exact selected-point provenance are essential. On [-0.5,0.5], the
nonconvex function -x^2 has zero gradient at zero but suboptimality 0.25. Reusing a
zero gradient from the optimum of (x-0.5)^2 at x=-0.5 would report zero instead of
the true gap 1. These analytic counterexamples are preserved as executable
fixtures. Even a perfectly minimized learned objective (u-0.5)^2 can select a
command with physical cost 1 under the true objective (u+0.5)^2. A modeled bound
therefore cannot turn the fresh physical failures into success or establish model
correctness. The physical report retains all 310 failed ordinary goal windows.

Reproduce in the existing environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/cartesian_stream_v17_optimizer_certificate.py
```

The executable refuses to overwrite existing results. Source and design hashes,
final input/audit hashes, all 11,520 decision records, the 48 grouped summaries,
all numerical discrepancies and the NPZ hash are in
`cartesian_stream_v17_optimizer_certificate_results.json`. Plans, both saved and
independently replayed gradients, identifiers and bound arrays are preserved in
`cartesian_stream_v17_optimizer_certificate_arrays.npz`. Reusing earlier
independent derivative helpers is stated openly; this is not another independent
training or physical-model derivation. No outcome enters model selection.
