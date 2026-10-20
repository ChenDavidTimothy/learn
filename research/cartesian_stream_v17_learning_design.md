# First moving-history learning construction — frozen exploratory protocol

2026-10-20. Written after acquisition coverage but BEFORE learner fitting, heldout
command forks or control outcomes. Rep0 is exploratory, not a release. The
acquisition/design/source/raw arrays are immutable. Prospective independent
reviews: cartesian_moving_v17_learning_design_independent.md and the separate
artifact/theory notes. The general-learning Goal remains active.

## Question and sufficient conventional explanations

Can ordinary camera/action records acquire changing action consequences and use
them for new decisions, without exact initial preparation, joint observations or
the prepared constant-response law? Compare two conventional constructions: a
generic nonlinear delay predictor and a retained local affine predictor. Neither
must win. A local method sufficing favors the simpler account. This extends v15's
history/control construction to noninjective sensing and unprepared motion; it
is not a new universal principle or a modern benchmark claim.

The retained model must learn drift AND the full two-input response from the same
acquired corpus. A global zero-gain initialization or frozen v16 coefficients
would be an unfair rival. An ensemble is fit uncertainty, not a sufficient
multimodal physical belief. A finite delay cannot retain every remote quiet alias.

## Shared ordinary supervised interface

Only cartesian_stream_v17_rep0_ordinary.npz is opened for fitting and selection.
Use h=16 past .1sec intervals, current noisy p, and past issued commands. Features:
current p(2), (p_t-p_{t-k})/(k*.1) for k=1..16 (32), and chronological past16
two-coordinate commands (32). Candidate input is one two-coordinate command held
for the next TWO intervals. Rows t=16,18,...,254 only; 120 rows/trajectory,
2,880 training /960 validation /960 test rows. Split whole trajectories.

Targets are the TWO future observed positions minus linear extrapolation
p_t+k*(p_t-p_{t-1}), k=1,2. The linear extrapolator is a disclosed generic motion
bias, not an exact physical law. Networks/local fits acquire its correction.
Scale features and target corrections using training data only, floor stds at
.01(feature) and .001meter(target). Candidate torques are scaled by .5. No true
derivative, evaluator noise, branch, body coefficient or polar feature is used.

Three shared episode-bootstrap members draw24 complete training trajectory IDs
with replacement. SeedSequence [20261020,17,660,0,0,member]. Nonlinear network
initialization uses independent role1 and candidate/member IDs; save actual IDs
and seeds. All candidates/failed fits/weights/selection scores remain available.

## Nonlinear predictor

Two tanh hidden layers, widths64 or128, four output corrections. Four candidates
width x Adam weight_decay(1e-4,.01), 300 full-batch epochs, lr=.003, independent
three-member initialization, squared normalized-target loss. No early stopping
or task-outcome tuning. Save loss histories and terminal weights. Full-batch
training order is fixed. No claim of global optimum or full posterior inference.

## Retained local predictor

Use the SAME three bootstrapped corpora and feature/target scales. Select128 or512
nearest acquired context rows (exclude candidate action from the distance).
Distance is sum of mean squared standardized differences over the current-p,
motion-history and past-command blocks. Gaussian weights exp(-d²/d_k²) where d_k
is the kth distance (positive numerical floor). Regress four target corrections
on [1, standardized_context-query_context, command/.5]. Full cross-coordinate
linear response and drift are fitted jointly; ridge1e-3 or.1 on coefficients,
unpenalized intercept, weighted mean squared loss. Four candidates k x ridge.
At an actual decision, fit each local member once and freeze its coefficients
through the hypothetical plan. Refit after each actual two-interval observation.
Append newly completed causal rows to each member's library online; preserve
them across goals. This is ordinary local regression, not an acquired physical
constant-B model. Its linear extrapolation and local validity are supplied bias.
Online library revision versus frozen nonlinear weights is an explicit complete-
mechanism difference; don't attribute a result solely to representation.

## Selection and heldout diagnostics

Each family selects its candidate solely by mean per-coordinate validation error
over six-block/12-interval open-loop predictions under ACTUAL heldout commands.
Origins t=16,40,...,232 in each of8 validation trajectories (80 origins). Members
roll out their own mean observations coherently; combine squared errors averaged
over members, not only the error of the ensemble mean. This discourages unstable
uncertain rollouts. No evaluator truth or test target is used. The score includes
the limits of mean rollouts; it does not anticipate future measurements. Report
all candidate scores and finite/optimizer failures.

After selection, score actual test observed one-/two-step and six-block forecasts.
External evaluator-only truth can additionally score mean positions. At32 fixed
test contexts (8 test trajectories x t=16,40,120,240), independently fork the same
hidden state under commands (0,0),(.4,0),(-.4,0),(0,.4),(0,-.4), each heldtwo
intervals. Models receive only the unchanged available ordinary prefix and each
proposed command. Score all predicted means and contrasts against the zero
command, at both future camera times. The160 continuations/320 flows/32seconds
are privileged evaluation, zero training data; retain all evaluator outcomes and
reset accounting. Do not select candidates or change methods from fork outcomes.
No sensor noises are sampled for these true-mean contrast diagnostics; separate
realized noisy test scores remain. Prediction alone is not the milestone.

## First charged decision pilot (implement after frozen predictive checks)

Same .1 camera/.2 command renewals and |u_j|<=.5. Four fresh preparations, two
paired noise/action-warmup replicates, both selected mechanisms:16 method episodes.
Initialization uses acquisition's distributions and independent roles. Warmup16
intervals: eightzero then four exogenous random heldtwo commands, matching the
acquisition pattern. Charge all1.6seconds/17initial camera reads. No exact motion
is supplied. Neither body nor ordinary memory resets at goal changes.

At warmup end let g0=.95*the actual noisy endpoint reading, and disclose goals
g0,R(.04)g0,R(-.04)g0 successively, sixseconds each. Only currentgoal enters a
decision. These modest inward/rotational goals depend on an ordinary reading,
not evaluator pose. Singular/failure/recovery cases are retained. Freeze nominal
bounded exact-state witness protocol before testing authority; a witness is
privileged diagnostic, not an equally informed learned baseline or a guarantee
of recovery from arbitrary states.

Both mechanisms use the same six-block bounded planner, expected mean-path loss
averaged over three fit members. Per predicted camera sample use squared endpoint
error +.25*squared finite-difference velocity; effort .01*sum commanded u² per
physical interval. Lastsample receives the same cost (no hidden terminal oracle).
This supplies a task/motion penalty, not actual Cartesian-velocity inference.
Warm start from shifted previous plan, retain zero-plan competitor, use the same
bounded gradient optimizer/budget; freeze precise implementation before outcomes.
Save initial/final plan costs, optimization failures and gradients. Planner
computation is reported separately; do not claim physical real-time feasibility.

External actual task score uses integrated endpointerror +.25*trueCartesian speed²
+.01*torque² over every .1sec interval; report warmup separately and combined.
Precision means per-coordinate endpointerror<=.03m AND trueCartesian velocity
<=.03m/s at EVERYsample in the final1second of each6secgoal. This is a sustained
window, unlike v16's endpointchecks. Also retain phase profiles, effort, early/
late performance and all failed trajectories. One pilot cannot establish
population superiority; fresh independently acquired corpora and independent
mathematical/numerical/dataflow audits are required for a meaningful release.

## Falsifiers and limits

Correct passive forecasts with wrong command contrasts do not establish learning
action consequences. Better contrasts without useful decisions do not establish
a practical mechanism benefit. Local parity favors conventional sufficient local
estimation. Unsupported history/action combinations, singular authority or poor
numerical planning cannot establish a missing representation. Do not increase
difficulty or tune test recovery until one method wins. No mandatory probe,
discrete branch, full world model or novel architecture follows from this test.
