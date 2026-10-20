# Independent audit of actual ordinary decisions and online local learning

2026-10-20. **PASS** for complete ordinary-decision artifact reconstruction.
The audit scope/source was prepared before reading control outcomes. All sixteen
ordinary episodes, 1,440 decisions and 2,160 online local bootstrap-member fits
were verified independently. No canonical source/result was edited. This report
does not certify actual physics/RNG/task scores, which have a separate auditor.

Reproduce with `.venv/bin/python research/audit_cartesian_stream_v17_control.py`.
Executable source and actual results are retained as
`audit_cartesian_stream_v17_control.py` and `_results.json`; the JSON contains
exact control/result/arrays/fit/ordinary/weights/contract/reference-helper hashes.
The actual control source is unchanged from preflight:
`220f12b194e31e32cea916cb0d65628d5ee0ab15ea3f3de1e6ccfd40b23814ca`.

## Independent construction and complete checks

The auditor imports no canonical learning/local/control implementation and reads
no evaluator-state array. It uses the original ordinary acquisition and saved
selected network parameters, its own previously audited NumPy nonlinear forward
and history-update equations, and independent weighted residual-augmentation
least squares for the local model. Every hypothetical rollout uses its own
member's synthetic history and the actual decision's frozen local coefficients.

At the start of EACH local episode, the auditor reconstructs each original
complete-episode bootstrap library. At each actual renewal it independently
computes the standardized query, stable context-neighbor set, bandwidth,
weights, full affine context/action regression and diagnostics. Its augmented
least-squares solve is distinct from the canonical regularized normal solve.
Only after checking that decision and its two actual commands does it append
the single newly completed ordinary row, using the real recorded observations.
No hypothetical reading or evaluator target is appended. All saved local
queries, coefficients, fit summaries and library counts match. Each library
starts with 2,880 rows and ends with 2,970; the same library survives goal changes.

For EVERY decision the audit reconstructs the shifted previous plan and zero
competitor, their modeled costs, initial choice, optimizer final-plan cost,
nonworsening finite-feasible acceptance/fallback and selected plan. It separately
reconstructs the selected full gradient and box projected-gradient mapping.
All current-goal/phase/time indices, actually issued first commands and repeated
second commands match. Selected plans and libraries continue across the two goal
switches; each new method episode starts independently from its prescribed
acquired predictor. Goals independently recompute from the ordinary warmup
reading, without true preparation/state inputs. The source-level preflight
establishes current-goal-only dataflow; artifact agreement alone would not prove
absence of every possible hidden software access.

All optimizer-call counts reconcile with recorded SciPy counts and the two
initial/one selected reevaluations. There are 21,265 total modeled value/gradient
calls, zero recorded unsuccessful terminations and zero fallbacks. These are
recorded solver outcomes, not a theorem of optimality.

## Actual numerical agreement

| Quantity | Maximum absolute discrepancy |
|---|---:|
| local query | 0 |
| local saved coefficient vs independent augmented least squares | 1.60e-11 |
| local fit diagnostics | 7.23e-14 |
| local regularized stationarity residual | 4.19e-14 |
| every reconstructed modeled plan cost | 6.68e-13 |
| selected plan gradient | 2.50e-13 |
| selected projected gradient | 2.50e-13 |
| goal recomputation | 2.23e-16 |

The exact recorded acceptance policy is checked using recorded cost values;
those values are independently verified to the tolerances above. No optimizer
path is rerun or substituted. Its unsaved internal iterates and original final
Jacobian are not independently reconstructed. Selected derivatives and all
retained final-plan costs are reconstructed explicitly.

## Necessary optimization qualification

All 1,440 runs returned `success=True`, but that is NOT uniform stationarity.
Independently reconstructed selected projected-gradient maxima across the 720
decisions per family are approximately:

| Family | median | 90th percentile | 99th percentile | maximum |
|---|---:|---:|---:|---:|
| nonlinear | 1.76e-6 | 3.00e-5 | 2.10e-4 | .147447 |
| local | 5.90e-7 | 2.83e-6 | 7.76e-6 | 6.03e-5 |

Five nonlinear decisions exceed .001 and two exceed .01. The largest is ordinary
episode14/case7, renewal63, phase2, time142. Its selected cost is30.2449567 versus
initial30.8945893; the solver reports relative-function-reduction convergence
after14 iterations/23 evaluations despite projected gradient .147447. This is
a retained numerical limitation of the frozen planner, not an artifact replay
failure. The largest local residual is episode11/case5, renewal9, time34,
6.03e-5, likewise ending on relative function reduction.

With local coefficients/query frozen, every synthetic feature and position is
affine in planned commands by induction. The declared squared-position,
squared-difference and effort objective is consequently a convex quadratic;
positive effort makes it strictly convex. This describes the local subproblem,
not an exact nonlinear physical controller. The nonlinear network planner has
no such convexity guarantee. Neither successful termination nor correct
derivatives establishes a global optimum. Any physical performance contrast
must retain the observed nonlinear stationarity caveat rather than be attributed
solely to learned representation.

One acquired corpus and four preparations do not establish population superiority.
Local online revision versus frozen neural weights remains an explicit complete-
mechanism difference. Independent physical replay, paired streams, sampled true
scores, nominal witness authority and all failures belong with this ordinary
decision audit before a scientific claim is released.
