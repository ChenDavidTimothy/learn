# Next investigation: learn a change in what an action means

Written while the one bounded v13 diagnostic is independently audited. This is a
research question/design, not a verified result or a mandated architecture.

The scalar branch supplied a fixed correct function family and control direction.
A more consequential question is whether acquired knowledge can express a changing
relation between commands and consequences when a body's configuration couples its
actuators. No further scalar optimizer escalation is justified.

Start from intervention consequences, without requiring named hidden physical
parameters. Consider a horizontal two-link articulated body, rest at a supplied
pose q=(q1,q2), no gravity. A short bounded torque command produces a measured joint
position displacement. Its direction depends on configuration through joint inertia.
A controller calibrated separately at each joint or using a constant command matrix
can predict the wrong coupled effect even at rest. This states a family limitation,
not a learned-method comparison. Can a simple general learner
acquire this context-dependent map from noisy position/action episodes and use it
for NEW desired displacements and held-out poses?

Derive the simplest discriminating observation/action experiment before learning
machinery. The initial discrete-instrument derivation used one semi-implicit Euler step at rest:

    v_next=dt*M(q)^-1*u, q_next=q+dt*v_next,
    displacement=dt^2*M(q)^-1*u.

Before executing a pilot, revise the instrument to continuous dynamics integrated
by DOP853. At rest with no friction/gravity, the short-pulse displacement is
.5*dt^2*M(q)^-1*u+O(dt^4), rather than the Euler factor1. Verify the convergence
and energy/work relation, and treat actual continuous endpoints as experiment
outcomes. This correction is prescribed before pilot execution. The earlier Euler
formula remains here as design history, not the executed physical ground truth. For two point masses/lengths1,

    M(q2)=[[3+2*cos(q2),1+cos(q2)],[1+cos(q2),1]].

It is SPD (det=2-cos(q2)^2>=1). At q2=0, inverse=[[1,-2],[-2,5]];
at q2=pi, inverse=[[1,0],[0,1]]. Thus the SAME joint2 torque has a nonzero joint1 LEADING response at
one pose and zero leading coefficient at the other; independent scalar models
cannot represent both leading responses. At finite pulse duration, coupling can
return even at the second pose (joint1 motion starts at higher order).
This derives a context requirement within this instrument; it is not a universal
proof that state/world-model variables are mandatory. A direct context-to-action
map, episodic retrieval, or a learned consequence map are all legitimate candidates.

Primary mechanics reference checked: Tedrake's multibody notes
https://underactuated.mit.edu/multibody.html, point-mass two-link Lagrangian/inertia.
Our choice removes gravity/velocity terms only by using the simplest rest pulse.
The longer-term test must include continuous coupled motion/Coriolis/noisy history,
not stop indefinitely at static force calibration.

First pilot should verify the physical map, measurement timing, bounds, context
support and decision feasibility. Then freeze a same-data comparison: constant
coupled linear response, context-conditioned generic kernel/RBF response, and a
genuinely different direct inverse/retrieval learner. Correct-mechanics predictor
may be a clearly privileged diagnostic. Do not give fitted mass/cosine functions
to the general learner. Do not claim superiority from beating the deliberately
misspecified independent baseline; use strong joint/context-aware alternatives.

Vary context coverage as a scientific discriminator, not just sample count: broad
poses versus a narrow calibration region under matched command/measurement budgets.
Hold out actual desired displacement tasks and context regions; count resets and
all acquisition/tuning/test interactions. Observed positions/actions only for
ordinary learners. Sensor error on initial pose must be specified; avoid secretly
providing exact context while claiming full noisy state inference. A supplied
known preparation pose is acceptable only if labeled as such.

The primary capability is choosing a bounded command that achieves a desired small
joint displacement, with held-out context/action consequences scored separately.
If task targets require infeasible torque, distinguish unreachable task from learning
failure. If a simple context-aware established method succeeds, accept it and move
toward continuous articulated control; do not manufacture novelty. This first test
can expose a missing dependency in the present vocabulary, not complete robotics
or the general-learning objective.


## Independent pre-learning criticisms, retained before any method comparison

The mechanics pilot is NOT a learning comparison and has q1 fixed. Its noisy
initial observation would correlate context error with observed displacement if
used as both context and response baseline. Do not treat that conditional regression
noise as independent. A future dataset should use an independent context read and
baseline read, explicitly count all readings, or fit the joint measurement model.
Known preparation pose is also a possible supplied fact, but then claim that fact
openly rather than pretending to infer noiseless context from a noisy sensor.

Direct inverse regression has noisy displacement regressors; naive least squares
suffers EIV shrinkage and is an unfair sole direct baseline. Use calibrated EIV
correction/replicated observations or another serious inverse/episodic method. No
victory over that known error would establish model necessity.

Horizontal global rotation makes q1 irrelevant to the leading response; handing
learners q2 alone supplies this relevance fact. Give both encoders to generic
learners and test new q1 as well as held-out q2, labeling any q2-only diagnostic as
privileged. Use the SAME small desired displacements across poses with a common
feasible set. Targets generated from unknown context-specific random torques can
encode the very relation being tested. Pulse displacement is not reach-and-stop.

Future mechanics audits should check the cyclic momentum M11*v1+M12*v2 when tau1=0,
energy change equals constant torque work, generic-pose fourth-order remainder,
solver refinement, and the special-pose higher-order correction. These checks must
precede learning claims. Do not publish v14 just because this instrument works.
