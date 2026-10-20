# Independent prospective artifact/design review: minimal moving-experience comparison

2026-10-20. Independent artifact auditor, before any learner/control outcome. Read
the latest verified v16 prefix/current continuation, the audited moving instrument,
`cartesian_stream_v17_acquisition_design.md`, acquisition source, and the other
independent prospective note. This note does not edit canonical designs or select
a result. The acquisition arrays exist; no learner outcome or fork outcome was
used in this review. No v17 milestone is established.

## Judgment and smallest useful pair

The proposed pair is scientifically defensible: (1) a small regularized nonlinear
camera/action-history predictor, with three trajectory-bootstrap fits; (2) a
retained local drift/action estimator with a same-data contextual warm start.
Both should drive the same bounded .2-second-renewal planner. There is no present
need for an episodic third method, a new mixture architecture, or another
observability theorem. Accept the local method if it suffices.

The consequential uncertainty is whether ordinary moving records acquire useful
dependence of consequences on context AND a new command, beyond passive
extrapolation and local feedback estimation. A good forecast score alone does
not resolve this. The experiment should end with continuous charged goal control,
but the heldout command-contrast diagnostic must precede interpretation of that
control. A single corpus remains engineering evidence, not independent-corpus
statistical evidence or a competitive robotics benchmark.

## Concrete fair information/timing contract

At a renewal time t, both methods receive only observations through o[t], commands
through u[t-1], and a candidate new command u held over the next two intervals.
The target can be the four-vector

    Y[t] = (o[t+1]-o[t], o[t+2]-o[t]).

Predicting two increments supports a common rollout interface without supplied
q, v, branch, Jacobian or constant response law. A candidate command must be an
input, not a target accidentally inferred from future camera samples. Restrict
new-command training/evaluation to t=8,10,...,254; the second half of a held block
does not contain an independently randomized new action. The frozen collector
has 256 TOTAL intervals including its eight passive intervals, so its stated
10,240 physical intervals and 10,280 camera reads are consistent.

Current position plus causal relative history/past commands is legitimate.
Train-only scaling and causal slopes are legitimate too, but slopes, smoothing
and linear-extrapolation residual targets are supplied inductive choices. Do not
describe them as discovered velocity or proof that the chosen history is a
sufficient state. Give the local model the same causal readings and command
history. Do not use an acausal smoother or evaluator velocity as either target
or input. A target relative to a current camera reading shares that reading's
noise with its input; ordinary squared-error prediction is permissible, but an
iid output-error likelihood or unbiased physical-coefficient claim would need
additional justification.

For autonomous multi-block validation/planning, append predicted camera outputs,
not the heldout future observations, to the synthetic history. Preserve the
issued candidate commands. Do not count a teacher-forced forecast as an
autonomous rollout. Use the same horizon, goal information, action bounds,
optimization budget and observed-output cost for both methods; record optimizer
failure and candidate costs. Instantaneous stopping velocity can be scored
externally, but a camera-difference penalty is a different decision quantity.
There must be no reset of physical motion or retained estimator memory at a goal
change. Future goals remain hidden until announced.

## Avoiding weak rivals and mean-alias traps

The retained estimator needs BOTH moving drift and a full two-input response,
not a zero-drift Jacobian controller or frozen v16 response. A local affine
relation for the four-vector Y, with an intercept and full action matrix, is one
possible implementation. A more restricted second-order form is also legitimate
if its supplied approximation is disclosed. Population/contextual initialization,
ridge/forgetting/window settings and any causal filter should use the same
training/validation corpus. Log local command-design conditioning: one repeated
command cannot separately identify drift and both action columns. Updating from
ordinary observations is permitted; never initialize or repair it with truth.
If the nonlinear model is frozen but the local estimator updates, this is a
comparison of two complete mechanisms, not a pure representation ablation.

Three bootstrap fits are a reasonable small baseline, not a calibrated posterior
over hidden physical contexts. In a simple counterexample, two hidden contexts
give Y=+u and Y=-u with equal probability at the same available history. Every
well-fit squared-error model can converge to the same zero conditional mean;
bootstrap disagreement then vanishes while response ambiguity remains. Longer
informative history can remove this in particular contexts, but the ensemble
does not prove it has done so. Nor does a Gaussian output variance by itself
preserve a persistent multimodal context. This does not require adding another
architecture now: label the limitation and test it rather than claiming a belief
mechanism. If planning uses ensemble members, freeze whether a member persists
through a rollout; averaging model outputs each step changes the mechanism.

Generic MLPs are an adequate first conventional construction if optimization,
scale and regularization are checked and history/width/decay are selected only
on whole heldout validation trajectories. Count all starts and failed fits. A
small validation menu is preferable to rescuing a failed control outcome with a
growing feature menu. No evidence here licenses superiority to full PETS or to
modern learning-control benchmarks.

## Before-control checks that distinguish explanations

1. **Numerics and actual coverage.** Retain all trajectories. Report step/speed,
   energy-work and tighter-solver checks including extremes, along with observable
   camera-history coverage. A conservative random-torque arm can become fast;
   broad slow interior starts do not imply slow or nonsingular later histories.
   Bounded camera coordinates do not bound angular speed or prevent sampled
   rotation aliasing. Truth-based coverage diagnostics stay evaluator-only.
2. **Passive versus action-dependent prediction.** Score at command renewals and
   report persistence/linear-extrapolation and ignore-new-command diagnostic
   errors. Low total next-position error can conceal a wrong small O(dt²) action
   effect behind larger O(dt) moving drift. These are diagnostic references,
   not two more main controller competitors.
3. **Frozen fork contrasts.** At prespecified heldout prefixes, give both models
   the identical available noisy history and a fixed command bank. Compare
   predicted Y(u)-Y(v), direction/sign and finite command consequences, as well
   as absolute forecasts. The evaluator may fork the same hidden state to form
   this score; those states/continuations never train or select the learner.
   Include zero, opposite axes/diagonals and interior commands within the frozen
   acquisition bounds, and count all extra privileged flows. Independent sensor
   draws add contrast noise; common noise or true-position scoring is a declared
   evaluator diagnostic, not an observed training target. If fork results select
   settings they belong to validation, with a separate untouched test bank.
4. **Multi-block forecasts.** Score autonomous rollouts over the proposed planner
   horizon, not only two-step teacher-forced fits. Separate error growth from
   new-command response error. Preserve high-error tails and failed rollouts.
5. **Support versus model failure.** Report train-only scaled history-neighbor
   distances, local action-design conditioning/effective sample counts where
   applicable, and errors by observable displacement/command-switch regime.
   Such diagnostics are heuristics, not an identifiability certificate. Near
   support, disagreement between a successful local fit and failed MLP points to
   that construction/optimization. Both failing only far from support points to
   acquisition. Both accurately predicting feasible consequences but failing
   goals points toward planner/horizon/objective or control-distribution shift.

Split and bootstrap by complete trajectories, never overlapping windows. Keep
trajectory boundaries out of windows and report trajectory-clustered uncertainty.
Forty trajectories are not thousands of independent physical preparations. One
acquisition replicate also cannot support a corpus-level standard error by
relabeling its windows. The passive prelude, goal switches, waiting, torque and
all failed actual control are charged. Feasibility witnesses may diagnose
authority externally without becoming an inverse available to the controller.

## Primary literature reconciliation

[Dalla Libera et al. (2020), Sections II–III and V-D](https://arxiv.org/html/2002.10621)
give a serious precedent for learning transitions from raw position histories
and testing rollout separately from one-step prediction. Their physical kernels
and measured generalized-coordinate interface are additional structure; the
paper does not establish that our noninjective endpoint history is sufficient.
A generic history MLP uses a different approximation, so do not label it their
reproduced algorithm.

[Atkeson, Moore and Schaal (1997), Sections 2–3](https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1997_2/moore_andrew_1997_2.pdf)
provide the conventional forward-local-learning/control perspective and discuss
coverage failures and multivalued inverse mappings. This strengthens the case
for a serious retained local forward rival rather than direct action averaging;
it does not supply hidden state or solve this camera alias automatically.

[Chua et al. (2018), Sections 4–5](https://arxiv.org/html/1805.12114)
explicitly distinguish deterministic ensembles from probabilistic ensembles and
compare ways of propagating uncertainty. This is a useful warning against
equating three deterministic bootstrap predictions with the full PETS method or
a complete hidden-context distribution. Borrow the distinction, not a claim of
implementing or surpassing that system.

The strongest useful conclusion may be that this ordinary local method already
suffices. If the generic history relation improves supported command contrasts
and charged continuous control, it establishes a useful acquired relationship
under the fixed-body/camera/timing assumptions. Neither outcome alone establishes
a universal representation or general intelligent learning.
