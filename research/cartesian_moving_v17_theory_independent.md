# Prospective independent audit: moving Cartesian experience

2026-10-20. Latest verified ledger is v16, SSH publication
1f1af0cef125d35a9e143bc7fbdef614a60d22a1. This is a prospective mechanics and
framing note, **not v17, not a learner, and not a new milestone**. Canonical
experiments were not edited. The question is whether v16's stationary ambiguity
survives ordinary moving experience, before constructing more probing machinery.

## First-principles answer

**PROVEN UNDER ASSUMPTIONS:** For this horizontal two-link unit point-mass body,
at every nonsingular endpoint radius0<r<2, the two joint branches sharing the SAME
Cartesian position and SAME NONZERO Cartesian velocity have different zero-input
Cartesian accelerations. There is NO nonzero velocity direction for which the
complete2D acceleration remains aliased. In fact no common joint command can
cancel their full instantaneous acceleration difference while they are moving.

Stationary rest is the exceptional case underlying v16. This does not imply
finite noisy histories identify the branch reliably, that every relevant decision
requires branch identification, or that an explicit branch/mode representation
is a necessary learning architecture.

## Independent Lagrangian derivation

Write p=r*e_r(theta), d=sqrt(4-r²), phi=2arccos(r/2), branch sign sigma=+/-1,

    q1=theta-sigma*phi/2, q2=sigma*phi,
    vr=rdot, vt=r*thetadot, omega=vt/r,
    q1dot=omega+sigma*vr/d, q2dot=-2sigma*vr/d.

Each branch therefore realizes the same actual endpoint velocity. The first point
mass moves with speed q1dot and the distal point mass with squared speed
vr²+r²omega². Hence, without using the joint Coriolis formula as a foundation,

    kinetic=.5[(1+r²)omega²+2sigma*vr*omega/d+(1+1/d²)vr²].

The polar inertia in coordinates(r,theta) and virtual-work forces are

    G_sigma=[[1+1/d²,sigma/d],[sigma/d,1+r²]],
    Qr=sigma*(u1-2u2)/d, Qtheta=u1.

This kinetic cross term changes chirality. Its presence means that the two
preparations are not equivalent moving mechanisms merely because their endpoint
positions and joint inertia cos(q2) agree. Euler–Lagrange gives

    G_sigma*[rdd,thetadd]
      = [Qr+r*omega²-r*vr²/d⁴,
         Qtheta-2r*vr*omega-sigma*r*vr²/d³].

Convert to actual Cartesian acceleration:

    a=(rdd-r*omega²)*e_r+(r*thetadd+2vr*omega)*e_t.

Let A denote sigma=+1 and B denote sigma=-1, and N=4+4r²-r⁴>0. Direct solution gives

    aB-aA = (2d/N)*{
        [-r²u1+2(1+r²)u2-2vr*vt]*e_r
        +[vt²+(r²/d²)vr²]*e_t }.

Thus Delta B*u is purely radial. The tangential passive term is strictly positive
for any(vr,vt)!=(0,0), independently of the common command. No common action masks
that tangential difference. This is a statement about equal CURRENT p/pdot states;
subsequent physical trajectories generally cease to share them.

For a direct joint-coordinate check, put c=cos(phi/2),s=sin(phi/2), and
D=2-cos²(phi)=det M. The same difference at theta=0 is

    passive(aB-aA)=[-2s*vr*vt/D, (s*vt²+c²*vr²/s)/D].

This agrees exactly with the root's independently derived formula. At phi=pi/2,

    passive(aB-aA)=[-vr*vt/sqrt(2),(vr²+vt²)/(2sqrt(2))].

The branch sign convention matters: aA-aB is the negative of every expression
above. Joint-coordinate Jdot*qdot is identical for the matched pair; the passive
separation comes from the different J*M^-1*h terms, not from incorrectly giving
the branches different endpoint velocities.

## Counterexamples and boundaries

- Pure radial motion or pure tangential motion makes the RADIAL passive difference
  zero. The TANGENTIAL difference remains nonzero. A scalar radial sensor or a
  radial-only acceleration diagnostic would miss this; the full Cartesian camera
  does not have that limitation.
- At rest, drift difference is zero. Commands satisfying
  u2=r²u1/[2(1+r²)] also have identical LEADING instantaneous endpoint acceleration
  on both branches. This does not make their whole finite moving trajectories
  identical. A nonzero common tangential acceleration creates motion, after which
  passive tangential differences occur. Rest with zero input remains exactly
  indistinguishable indefinitely, as established in v16.
- The statement excludes r=0 and r=2, where the endpoint Jacobian is singular or
  the polar direction/branch parametrization degenerates. Near those boundaries,
  matched endpoint velocities can imply large joint velocities and weak numerical
  conditioning. Do not claim uniform global observability from an interior result.
- Observable physical distinctions need not matter for a current goal. A robust
  controller may meet the specification without accurately naming the branch.
  Conversely earlier moving evidence can remain useful after stopping even when
  new passive stationary readings carry no branch information. A finite raw
  window that forgets the motion is a limitation of that window, not a universal
  impossibility or proof that explicit mixture modes are required.

## Finite camera histories are the actual information contract

For zero torque, matched current p/pdot and smooth deterministic trajectories,

    pB(t)-pA(t)=.5*(aB-aA)*t²+O(t³).

The same quadratic separation occurs for short past paths. A conservative body
with zero torque is time reversible: past configurations can be independently
checked by integrating forward from(q,-qdot). This is evaluator mechanics, not
permission to supply true velocities or noiseless p0 to an ordinary learner.

The leading acceleration distinction is O(V²), so endpoint separation over a
fixed short window is O(V²*h²). Lower speed rapidly hides the distinction under
fixed camera noise. For example, a three-reading second-difference acceleration
estimate has per-coordinate noise variance6*sensorVariance/h⁴ for independent
camera errors. A backward three-point estimate centers naturally at the previous
sample; treating it as the current acceleration introduces additional time bias.
There is no free exact current derivative in sampled noisy positions.

Zero-torque equations are homogeneous of degree two in velocity: scaling the
initial velocity byV scales the same configuration path in time, q_V(t)=q_1(Vt).
Consequently the weakness at lowV concerns a FIXED observation-time contract.
With more waiting or more independent readings one can collect additional passive
evidence; no intrinsic nonzero speed threshold or mandatory probe follows. Charge
that delay in the actual task instead of treating waiting as either free or banned.

Under the ORACLE diagnostic in which both candidate deterministic mean paths and
their matched current p/pdot are known, independent Gaussian camera noise gives

    KL(A||B)=sum_j ||meanA(t_j)-meanB(t_j)||²/(2*SD²)
            = ||Delta a||²/(8*SD²)*sum_j t_j⁴ + higher order.

For a fixed short sampling pattern this scales as V⁴*h⁴/SD². It usefully measures
how weak the signal can become. It is NOT a trained learner or a general bound
for unknown initial position/velocity/model. Nuisance initial states and correlated
history estimation must be handled under the actual available information rather
than silently giving this oracle diagnostic to an ordinary algorithm.

## Executed independent checks

`audit_cartesian_moving_v17_theory.py` independently constructs polar kinetic
energy, virtual work, Euler–Lagrange accelerations and the branch-difference formula.
The existing joint mechanics is imported only as an implementation under test.
Its actual results are saved in `audit_cartesian_moving_v17_theory_results.json`.

Across128 random nonsingular radii/orientations, matched velocities and nonzero
commands, maximum acceleration/difference discrepancy is4.44e-16; kinetic-energy
discrepancy1.11e-16; matched Cartesian velocity discrepancy2.36e-16. Every sampled
moving tangential difference is positive. Tiny passive future/past paths at three
durations independently check the quadratic separation for radial, tangential,
mixed and zero velocities. This is a small falsification instrument, not a new
learning/control experiment or evidence of finite noisy inference capability.

## Consequence for the next learning question

The correct next uncertainty is not automatically 'which probe reveals the elbow?'
Ordinary motion already creates informative history. A learner should be allowed
to acquire and reuse the relationship between motion history and consequences,
without being forced into modes, prepared rest or a globally constantB vocabulary.
Passive evidence may be sufficient; active evidence is a cost/precision question,
not a logically mandatory component of this moving instrument.

A defensible moving acquisition contract would provide ONLY noisy Cartesian
positions, issued commands and timestamps, including a causal camera history
before scored decisions. Joint pose/velocity may generate and externally audit
support, but are not training labels or ordinary context. Initial velocities must
not be quietly supplied; exact initial rest/p0 must not reappear as a hidden
assumption. Whole continuous trajectories, not overlapping rows, should separate
training/validation/test. Passive windows and bounded exogenous commands can occur
within those same streams without hidden state resets between them.

A passive history identifies local drift only under learned or supplied relations;
it does not, by itself, establish the response to every new command in an unknown
mechanism. Reusable cross-command consequences still need population experience
or other assumptions. This separates 'camera motion reveals context' from 'the
learner knows how that context changes action effects.'

The simplest serious alternatives are a conventional nonlinear delay predictor
using causal position/action history; a kinematic derivative/filter plus learned
local action response; and, only if justified, a persistent latent/context learner.
The conventional history alternative must be strong enough to use this passive
signal, not intentionally restricted to current endpoint position. A fixedB fit
that omits current motion would confound missing drift/history with failure to
learn modes. No one-step prediction advantage alone establishes sustained control;
new-pose/moving-goal decisions must eventually test whether acquired relations
remain useful with the same sensing, bounds and interaction budget.

Choose support and objectives before outcomes, include weak and stronger natural
motion, and permit an ordinary established method to be sufficient. If the
mechanics/info instrument simply shows passive histories are adequate, preserve
that correction and simplify the construction. Do not publish another rank theorem
or increase difficulty until active probes become necessary. The larger objective
is learning useful relations from continuous experience, not maintaining a favored
interpretation of the stationary v16 example.
