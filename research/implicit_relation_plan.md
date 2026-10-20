# Bounded second candidate: learn an implicit relation and condition it

2026-10-20. Selected after retiring joint error transport; not a rescue or a new
ledger. The ultimate Goal stays active. First-principles question: can a simple
relationship be more reusable than a simple forward consequence function, while
still learning its features from ordinary experience?

## Derivation and limitations before code or fitting

Learn f0(x), vectors b_j(x), matrices A_j(x), j=1..8. Supplied family:

    [I + sum_j c_j A_j(x)] y = f0(x) + sum_j c_j b_j(x).

A measured support pair (x_i,y_i) supplies linear coefficient constraints:
D_i[:,j]=b_j(x_i)-A_j(x_i)y_i, r_i=y_i-f0(x_i). Infer
c=(sum_i D_i^T D_i + lambda I)^-1 sum_i D_i^T r_i,
then solve the learned 2x2 relation at a query x. Meta-learn feature functions
through the actual query prediction loss AFTER inference, not by minimizing a
homogeneous relation residual (which admits degenerate solutions). c=0 gives
f0; A=0 is a nested explicit learned-function adaptation comparator.

This changes the learned object and evidence-conditioning operation, not the
similarity metric or the number of neural layers. It acquires feature relations
and infers a reusable new-system coefficient from experience; it does not receive
true physical parameters, inertia equations or physical-basis labels. The affine
output relation, low-dimensional persistence and grouping remain supplied priors.
No parameter recovery or universal architecture is claimed.

Motivation: nonlinear forward functions may have simpler implicit equations.
But y=g*u/m is NOT proof of advantage: an explicit learner can infer k=g/m with
one feature u, while separate m/g are unidentifiable from those observations.
The meaningful hypothesis concerns x-dependent rational variation that is harder
for a finite explicit shared feature family; evidence must decide. Finite observed
delta-v/dt is not exact acceleration and does not obey an exact mass equation.

Independent critic hazards: ridge regularizes coefficient inference, not query
invertibility. A small support residual is M(c) times prediction residual and
can reward small M. Query meta-loss mitigates that but proves no OOD guarantee.
Noisy y enters both D and r (errors in variables), so initial scope is noiseless.
No broad novel-principle claim: implicit identification/rational dynamics precede
this construction (SINDy-PI2020), and learning basis functions also has prior art.
The particular learned-relation meta-conditioning rule needs practical evidence.

## Fixed implementation and screening scope

Base/basis MLPs use two64-wide tanh layers. Latent rank8; initialize matrix-head
weights/biases at small scale, A entries=.05*tanh(raw), learn positive ridge
initialized1. Same standardized input8/output2 interface and ordinary data as
first screen. Matrix bounds do NOT guarantee invertibility or stable adaptation.
No condition correction or inference fallback will be silently introduced.

Matched explicit model sets A identically0 and learns vector b under the same
support/query loss/rank/base/batch/steps. ALPaCA and CoDA are serious independent
competitors, not just this nested ablation. Use their two frozen first-screen
training runs with validation selection, avoiding another baseline training loop.
The new candidate/explicit model receive the same2000steps, twoLRs .001/.003,
same training tasks, support0/4/16/64 and original validation body set. Select
checkpoint/global LR by validation only, exactly as before. Original test bodies
are NOT rescored for this new claim.

Before fitting or new-body sampling, freeze a NEW holdout: replicate1 role prefix
[20261020,18,30,1,split,role], 8new ID and8new OOD physical bodies, same acquisition
law/ranges/8trajectories/96steps/dt.05. Reuse original training data/scales and
validation for all methods. No range chosen from candidate outcomes. Physical
parameters remain evaluator-only. No extra sensor noise in this bounded screen.

Record all per-body MSEs, raw forecasts, correct versus rolled wrong-body support,
latent norms, minimum query singular values/condition numbers and failures.
Declare min singular value<1e-4 a conditioning failure; no test body is dropped.
Any heldout singular/nonfinite failure prevents a reliability/continuation claim.
Training numerical failures are preserved before considering a separately scoped
repair; no silent retry or tolerance relaxation.

Continue only if, at16or64 support, new-body OOD mean MSE improves >=10% over
BOTH strong conventional competitors AND matched explicit conditioning, without
>5% worse mean ID error or >5% opposite-budget OOD loss versus the strongest
competitor, and correct support improves over wrong support. Or if accuracy is
within5% of the strongest atBOTH budgets and adaptation is >=5times faster than
that accuracy-matched strongest competitor, with comparable-or-smaller capacity,
under repeated synchronized warm benchmarks that cache support statistics for all
methods. Single noisy walltimes do not establish efficiency. This is exploratory
screening, not confirmatory inference. No post-test architecture/ridge/threshold
search. If it fails, retire this second candidate and reassess rather than tune.

If it survives, the next requirement is fresh-seed/body confirmation, decision
capability with common control access and a genuinely different nonlinear family;
not a framework claim from prediction. Existing physics/shared baseline fixtures
are reused; check only the new ridge relation, nesting, gradients and no leakage.
No exhaustive replay, no new environment/packages/services, ownerPC/project.venv.

Prefit implementation disclosure: initialization seeds use
[20261020,18,31,0,method_index], implicit index 0, explicit index 1; the shared
training-episode RNG is unchanged from the first screen. Ridge initializes at
1.0001 including its declared floor. Implicit has an extra matrix MLP, so a win
over the nested explicit model alone cannot isolate representational advantage
from extra capacity. Record all parameter counts and both learning-rate costs.
A complete-method win can earn a capacity-controlled follow-up. Select only
validation checkpoints without any declared conditioning failures; record all
ineligible checkpoints. Fresh holdout conditioning failures forbid continuation.
