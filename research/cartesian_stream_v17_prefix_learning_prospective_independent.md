# Prospective challenge: how much population experience is needed?

2026-10-20. Independent first-principles note, written before a fresh-study
outcome. **No implementation or new protocol is prescribed here.** The frozen
rep200–207 retained-versus-frozen study is unchanged. This question is useful
because a successful acquired prior does not establish that its acquisition
cost was necessary for the rather local regulation tasks being tested.

## The actual comparison to question

The population corpus contains 614.4 seconds of training motion, 204.8 seconds
of validation motion and 204.8 seconds of ordinary heldout evaluation motion:
1,024 seconds total. Training plus validation is 819.2 seconds, not 614.4.
Separate heldout diagnostic acquisition from experience available to the learner
when reporting experimental efficiency. The test prefix costs 1.6 seconds,
consisting of 0.8 seconds with zero torque followed by four independently
randomized 0.2-second held commands (0.8 actuated seconds). It supplies only noisy
Cartesian positions and issued actions; initial position, rest, velocity,
joint state, branch and response matrices remain unavailable.

Could a conventional learner infer a useful local relation from that prefix and
subsequent task feedback, without population records? Possibly. If it succeeds,
that would weaken a claim that a population prior is needed here, and might
identify a substantially cheaper sufficient mechanism. If it fails, that does
not prove that population learning is necessary: failure could result from an
unhelpful local assumption, weak excitation, estimator/optimizer choices or
poorly charged initialization. The present retained/frozen experiment cannot
answer this different comparison, because both variants start with the same
acquired population library.

## What a short prefix can identify under an explicit approximation

Let camera spacing be h=0.1 seconds, and hold u constant for two camera intervals.
The ordinary statistic

    Y_k = (o[t+2] - 2 o[t+1] + o[t]) / h^2

eliminates unknown initial position and constant velocity exactly. It uses no
privileged velocity or prepared-rest information. For a locally smooth path it
is midpoint acceleration plus an O(h^2) term involving fourth derivatives,
within the held-command interval. Smoothness across a command discontinuity is
not needed when the midpoint is inside that interval. It is not an exact
finite-horizon action law for the nonlinear plant.

If one additionally approximates all these local accelerations by

    Y_k = d + B u_k + error_k,

then each output coordinate has three unknown coefficients: one drift and two
action coefficients. Four independently randomized two-dimensional commands
make [1,u1,u2] full column rank with probability one under the continuous
randomizer. The preceding four zero-command blocks can help estimate drift.
Thus unknown motion alone does not make local estimation algebraically
impossible, and a scratch baseline need not start with zero gains or known B.
There are very few residual degrees of freedom; rank is not statistical
precision, useful conditioning or validity of the approximation.

For independent zero-mean camera errors of variance R=(0.0002)^2 per coordinate,
the acceleration-statistic noise variance is 6R/h^4=0.0024, standard deviation
approximately 0.04899 m/s^2. Adjacent held-pair statistics share a boundary camera
error: their noise covariance is R/h^4=0.0004, rather than zero. A correct fit or
uncertainty calculation must account for this correlation or acknowledge the
approximation. Using more camera points can trade variance against local bias.

The simple affine fit presupposes that drift and gain remain sufficiently
constant during the chosen window. Actual acceleration depends on position,
motion and hidden configuration; the midpoint state also changes with the
current command. The two-interval displacement relation is therefore only
approximately affine in that command. State-dependent drift can mimic an
apparent input gain in a single short record. Independent randomized commands
help separate effects across repeated contexts; they do not by themselves
supply smoothness, constant coefficients or enough repetitions at one context.

A second-order output model is itself a supplied assumption. Away from Cartesian
singularities, a tracked local branch supports smooth Cartesian mechanics, but
(p,pdot) is not a globally unique state: two branches can share these values.
Branch changes, near singularity and rapid motion can invalidate a frozen
local approximation. Do not quietly give the scratch method the true branch,
the true second-order law, a global inertia structure or a safe gain matrix.
Conversely, regularity and local affine parameterization already enter the
population learner too; differences in those priors should be stated explicitly
rather than interpreted solely as different amounts of data.

## What cannot be established from the prefix without assumptions

The existing local regressor has 69 coefficients per output after fixing a query,
using a 66-dimensional history context plus action. Four randomized action blocks
and four passive blocks cannot determine that unrestricted family from scratch.
A compact scratch method must borrow structure, regularize, or accept substantial
uncertainty. A population method borrows relations from other histories instead.
These are competing information/assumption allocations, not learning versus no
learning.

More generally, any finite observed record is compatible with smooth extensions
that agree on every observed history/action and disagree on a new one. This
applies to finite population corpora too. No assumption-free argument establishes
that either a local prefix or a large corpus determines all future consequences.
Full-rank two-dimensional excitation is necessary for recovering an arbitrary
local two-input gain, but an actual task may need only a particular combination
of actions. Complete physical identification is not automatically the relevant
goal. Safe robust feedback, local trial correction, or a small task-relevant
relation might suffice even when B is poorly estimated.

Later closed-loop data are useful causal experience, but actions then depend on
past noisy observations. They are not independent randomized instruments with
the warmup covariance. Naively reusing the warmup moment identity, or regressing
noisy finite differences while treating all regressors as independent errors,
can introduce bias. Sequential likelihood or appropriately justified local
estimation is needed; added probing is a possible remedy, not a logical
requirement for every task.

## The consequential falsifier, if a later protocol is warranted

A strong bounded comparison would ask whether a defensible ordinary local online
learner, initialized only from the exact same noisy prefix and allowed all later
causal task feedback, reaches comparable sustained performance with much less
prior interaction. It should receive useful conventional assumptions openly,
nonzero data-derived gains when supported, the same action/observation bounds,
and honest fallback/regularization. A zero-response model with arbitrary zero
commands would be a straw baseline. No privileged state, response matrix,
prepared rest, tuned true-plant sign or uncharged failed initialization is
allowed. Any extra exploratory commands or longer prefix must be charged in
physical time, effort and task performance. Selecting scratch settings using
many true-plant validation runs also constitutes experience and must be counted.

The important uncertainty is whether population experience reduces transient
risk, required online excitation, computation, or total interactions for these
tasks—not whether one can construct a formally different controller. If both
approaches work comparably, prefer the simpler information contract for this
instrument and redirect toward tasks where reuse or transfer has an actual
consequence. If population reuse helps, isolate its scope across fresh contexts
before proposing a more elaborate architecture. No result here alone resolves
general intelligent learning, and this prospective note changes no frozen study.
