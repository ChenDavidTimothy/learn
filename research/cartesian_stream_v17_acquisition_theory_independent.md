# Moving acquisition: independent design review and learning falsifier

2026-10-20. This preserves the pre-data review communicated before collection,
then records resolved source issues and explicitly marked post-collection support
facts. It is prospective about learners/controllers: no learner, fork forecast or
closed-loop outcome was inspected, and no canonical acquisition artifact is edited.
Latest released ledger remains v16. This is not another observability milestone.

## What the stream can identify

A useful question is whether continuous noisy history can acquire/reuse action
consequences without exact rest/p0, supplied joint state or a constantB propagation
law. A hidden physical state is one description, not a mandatory learner object.
Let H_t be the full ordinary camera/action history at a command-renewal boundary,
and U_t the newly drawn command, committed for the next two camera intervals.
The source makes U_t independent of H_t AND the hidden physical state through its
separate exogenous command stream. Therefore, under the stated randomized
acquisition law,

    P(future camera path | H_t,do(U_t=u))
      =P(future camera path | H_t,U_t=u)

on the supported renewal-time experiment. This identifies a conditional
INTERVENTIONAL consequence kernel in population, without identifying mass, joint
coordinates, branch labels or a deterministic state representation. Continuous
commands supply full box support at renewals; exact pointwise and out-of-support
claims still require regularity assumptions. Finite40 trajectories do not uniquely
identify an arbitrary function on all possible histories/actions. Smoothness,
finite memory, parametric or other generalization choices remain real assumptions.

A conditional-mean predictor may average consequences when its history leaves
latent uncertainty. That can be statistically correct yet insufficient for some
decisions. Prediction success, uncertainty calibration and actual control are
separate claims. Passive moving history can reveal configuration in this known
body but not all possible action response: scaling both point masses preserves
all passive dynamics while scaling command gain. Cross-command acquisition is
still meaningful; no special probing architecture follows.

## Resolved action-clock and numerical concerns

The original held-two design would not justify arbitrary .1sec command switches.
In a block's second half, current command equals the previous command
deterministically conditional on observed history. A planner querying a new
command there would ask for an unsupported conditional intervention. The frozen
source/design now correctly use .2sec renewal/decision boundaries, with camera
sampling .1sec. Pooling more correlated rows cannot repair missing conditional
action support. Within a block ordinary forecasts know the committed command.

The conservative body has no damping to bound later speed under prolonged random
torque. Initial phi in[.4,pi-.4] and speed[.02,.12] are initial distributions, NOT
trajectory guarantees. Positive-definite joint inertia remains regular at endpoint
Jacobian singularities, but the earlier interior Cartesian-inverse argument is not
uniform there. Pre-data review therefore requested retained full motion, phase/
speed/singularity coverage and tighter numerical spots, with no truth-based
selection/filtering/resampling.

Energy/work consistency alone is not a position-accuracy certificate: conservative
integrators can have phase error despite good energy. Independent trajectory or
tighter-solver checks should include extreme motion and both joint phases q1 and
q1+q2. Small sampled phase increments are useful diagnostics, not a formal Nyquist
or global nonlinear observability theorem.

The first source had final numerical assertions before raw artifact writes. This
would have discarded failed/outlier arrays despite the promised preservation.
Before acquisition, root corrected this to save arrays/status and then raise on a
failed consistency check, and added distal phase-step diagnostics. Executed source
hash54f5b71020a163aa76326d94fefc7db8f5723947b6efceed4139ec5a5910f6f2 is the corrected
version. No failure occurred. The ordinary NPZ contains only noisy positions,
issued commands, split/trajectory IDs and calibrated dt/noise scale. Separate
files alone do not prove future learner code avoids evaluator leakage.

## First meaningful falsification before closed-loop design

One-step absolute position error can be dominated by inertial continuation and
smooth kinematics. A predictor can look excellent while learning little about
what a different command would do. The next separately frozen diagnostic should
hold an AVAILABLE noisy history fixed, fork the evaluator's physical state under
a small prespecified command set at a renewal, and compare predicted versus actual
PAIRED CONSEQUENCE CONTRASTS. Do not give the fork's state or outcomes to the learner.
Do not select histories/commands after seeing model successes. Count fork resets,
rollouts and simulated time as external evaluation work, not acquired experience.

The same predictor should produce both forecasts F(H,u) and F(H,v); score their
difference against p_future(u)-p_future(v). Common inertial carry cancels. Include
zero-contrast/kinematic continuation as a descriptive floor and a serious
conventional history/local-response alternative before interpreting any advantage
as more than fitted capacity. Absolute forecast error and fresh whole-trajectory
validation remain necessary; a contrast-only score can miss shared drift errors.
Some irreducible ambiguity can remain in an individual history, so zero contrast
MSE is not automatically attainable or an appropriate universal requirement.

For this EXOGENOUS dataset, fresh future camera noise is independent of a causal
forecast and the physical path. Thus terminal raw-camera squared-position error
has additive expectation offset2*SD² in2D (4*SD² for a sum over both future2D
reads). This permits ordinary raw-position forecast validation. It does not give
an unbiased finite-difference velocity reward for arbitrary feedback policies,
where current sensor error can influence the issued action and physical increment.
No previous lagged-actuator noise identity should be transported automatically.

Offline consequence contrasts are a gate, not a final learning milestone. A
substantive next conclusion still needs actual charged decisions with honest
feasibility/support and strong conventional alternatives. Do not stop at another
local-rank theorem or turn fork outcomes into hidden hyperparameter training data.

## When history length becomes an artificial handicap

A fixed short window is an inductive/computational choice, not a universal
information limit. Earlier passive diagnostics show evidence changes materially
between subsecond windows; choosing a shorter window while claiming all available
history is insufficient would be unfair. At rest, useful context can lie in older
motion. A retained conventional estimator or another mechanism may preserve it
without explicit named modes. Conversely very long raw windows increase finite-
sample estimation difficulty; longer is not automatically better.

Every method should receive the same causal acquired history and clock. If a
history-family grid is used, validation should select from an adequate predeclared
range; no zero-padding that pretends unavailable pre-start observations were
stationary or exact. Use a shared scored burn-in/missingness convention, and charge
waiting or preparation if it becomes part of an actual control contract. The eight
initial zero intervals are a supplied acquisition procedure, not exact velocity
knowledge. Split trajectories before making overlapping windows; evaluate
trajectory-level clusters rather than calling thousands of rows independent.

Randomized current actions identify conditional effects given FULL history. A
finite history summary need not be sufficient across later policies: truncation
can change the hidden-state mixture when the action/history distribution changes.
Good heldout one-step fit under the acquisition law alone cannot establish
closed-loop transportability. No generic delay-embedding slogan resolves the
controlled, sampled, noisy noninjective sensor problem automatically.

## Post-collection support fact, separate from the prospective argument

Root reports preserved full trajectories reaching speed1.906m/s rather than only
initial.02–.12m/s, with67sampled near-singular points. My first evaluator-only array
check finds182sign changes of sin(q2) bracketed by consecutive samples, across39 of
40 trajectories. This diagnoses actual changing endpoint chirality. It is NOT an
ordinary context feature or a reason to filter data. A single elbow label or fixed
response coefficient persisting throughout an episode would be a false physical
assumption here. The subsequent independent RNG/physics/artifact audit verifies
these facts and all counters; its source, actual results and report are saved as
`audit_cartesian_stream_v17_acquisition.py`, `_results.json` and `_report.md`. This
prospective note does not substitute for that separate artifact audit.

The new stream can therefore test something consequential: acquiring relations
that remain useful through changing motion/context rather than merely fitting a
certified constant coefficient. An ordinary well-validated history learner being
sufficient is a positive scientific outcome, not a failure to invent an architecture.
