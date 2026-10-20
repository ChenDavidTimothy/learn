# V16 alternative: what can new experience distinguish that passive memory cannot?

OPEN, not a milestone or prescribed architecture. The payload instrument did not
create a new precision failure: frozen learned control passed the same7/8post-change
windows with/without doubled distal mass. Cost changed, but this does not justify
building a detector/probing system or increasing the change until a baseline fails.

Another framing changes available information rather than only dynamics/basis fit.
Remove joint encoders; observe ONLY Cartesian end-effector position of the same
two-link body. All ordinary learning/control must then use those readings and
issued actions. True q/v and the kinematic formula are evaluator tools, never
ordinary features or branch labels. This is not merely an invertible relabeling
of the same information: elbow branches can have identical present/past outputs.

## Independently challenge this derivation before experiments

For unit lengths p(q)=[cos(q1)+cos(q1+q2),sin(q1)+sin(q1+q2)]. Let
qA=(-phi/2,phi), qB=(phi/2,-phi), with0<phi<pi. They have the same p=(2c,0),
where c=cos(phi/2),s=sin(phi/2), and the SAME joint inertia M (cos(q2) agrees).
At rest, end-effector acceleration is B(q)u=J(q)M(q2)^-1u, with

    J_A=[[0,-s],[2c,c]], J_B=[[0,s],[2c,c]].

At phi=pi/2,

    B_A=[[ 1/(2sqrt2), -3/(2sqrt2)],
         [ 1/(2sqrt2),  1/(2sqrt2)]]
    B_B=[[-1/(2sqrt2),  3/(2sqrt2)],
         [ 1/(2sqrt2),  1/(2sqrt2)]].

Zero torques from zero motion keep BOTH worlds exactly stationary indefinitely.
Their entire Cartesian observation histories have identical distributions under
identically distributed independent sensing noise. No longer passive lag or more
offline population data identifies the current elbow branch from those observations.
This does not rule out ordinary history prediction after informative motion.

For a common initial action, leading radial consequences have opposite signs.
For either branch individually, radial acceleration a with zero tangential
acceleration is feasible with opposite commands uA=(a/sqrt2,-a/sqrt2),uB=-uA.
Finite time p(T)-p0=.5T²B*u+O(T4) at rest; do NOT convert the leading statement
into an exact finite-T global lower bound without controlling the remainder.
Their mean B has radial row0. A certainty-equivalent planner using that mean may
see no route to a radial goal; that is a property of a specified planner/model,
not a theorem that every history learner needs an explicit discrete belief.

The ambiguity is task-dependent. Vertical leading acceleration can be achieved
with the SAME command in both worlds by choosing u in ker(row_x B_A), here
u proportional to(3,1). This is a direction blind to the branch at leading order,
yet useful for a vertical goal. A minimum-effort controller using only meanB need
not choose that direction; a controller retaining learned response variance can
favor it without first classifying the branch. Leading blindness is not exact
finite-time indistinguishability under that nonzero action.

## The consequential learning question

Can population experience preserve useful relationships BETWEEN possible action
consequences, so that a small amount of new experience selects/revises useful
behavior more efficiently than relearning a local relation? Do not give true B,
true branch labels, velocity or a hand-coded kinematic inverse to a learner.
Do not label a perfect-state oracle or a supplied slope an acquired representation.

Genuinely competing formulations include conventional local online Jacobian/ARX
adaptation, learned conditional mixtures, and a continuous distribution/covariance
over response relations. A Gaussian coefficient prior may exploit correlations
between possible responses even when its shape is not the true two-point prior;
explicit physical/discrete-state identification might be unnecessary. Mean-only
population regression is a useful descriptive ablation, not a strong sole baseline.
Generic equal-cost random excitation and ordinary task-induced updating must be
serious alternatives to directed probes. A probe must earn total time/effort/task
cost, and must not receive extra calibration resets or branch-specific resources.

One independently suggested simplification is worth deriving before any large
architecture: in a local short-pulse family B=Bbar+zeta*D, population residual
second moments can expose correlations between alternative response coefficients.
If initial rest and a locally constant Cartesian B are explicitly assumed, three
equal-duration pulses give

    p3-p0=T²*B*(2.5u0+1.5u1+.5u2),
    v3=T*B*(u0+u1+u2).

After observing the first probe, expected quadratic endpoint/stopping/effort loss
for the remaining deterministic bounded commands depends only on posterior first
and second coefficient moments. That subproblem is a convex quadratic decision,
not a requirement to classify a named physical state. Gaussian and discrete-mixture
priors can then compete fairly using the SAME acquired response data. A first
action can change what is known before those decisions; its benefit must be charged
against task cost and equal-budget random/task-directed alternatives. The exact
nonlinear body is not a locally constant double integrator; this is a candidate
instrument/derivation requiring remainder and actual-trajectory checks, not an
established full-body control theorem or the final experiment protocol.

Crucial limit: those posterior moments suffice ONLY for the fixed remaining
quadratic subproblem. They need not determine the value of the first probe or an
observation-conditioned future policy. Equal prior first/two moments can yield
different posterior updates under Gaussian versus discrete alternatives; the
full prior can therefore change the expected benefit of acquiring evidence.
Do not claim that an entire learning/control problem reduces to two moments.

Before choosing an algorithm, derive which interventions distinguish the worlds,
what remains ambiguous, feasible goals under command/time bounds, and what ordinary
measurements can score. Preserve full nonlinear continuous mechanics/kinematics;
test consequences and useful decisions separately. If starting Cartesian rest/pose
preparation is supplied for a clean information instrument, disclose it to ALL and
use it efficiently (avoid v14's needless noisy baseline). A rest-aliasing pilot
alone is NOT a new release; the follow-up must actually learn from experience and
use what was learned in new decisions, preferably sustained control without goal
resets. Escalate only to resolve a consequential uncertainty.

Underactuated gravity swing-up is another serious route: temporally coordinated
actions can require moving away from an immediate goal. However, short-horizon local
MPC failure would mostly expose a planner choice unless strong energy-shaping/global
trajectory and modern learning competitors are included fairly. Do not equate a
harder environment with deeper understanding. Unknown invertible sensor remapping
also risks a predictable wrong-basis test; isotropic kernels already have exact
orthogonal-coordinate invariance. These alternatives remain open to evidence.

Relevant modern baselines to STUDY before claims: PETS (Chua et al.2018,
https://arxiv.org/abs/1805.12114), SAC algorithms/applications (Haarnoja et al.2018/19,
https://arxiv.org/abs/1812.05905), TD-MPC2 (Hansen et al.2024,
https://arxiv.org/abs/2310.16828). Only their primary abstracts were read by the
coordinator so far; no implementation, complete-paper review or superiority claim.
The installed environment has torch/gymnasium/sklearn, noMuJoCo/SB3/PyBullet;
preserve it and do not install infrastructure merely to force a benchmark.
