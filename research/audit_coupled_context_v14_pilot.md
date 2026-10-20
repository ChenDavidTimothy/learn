# Independent audit of unreleased v14 coupled-mechanics pilot

**Mechanics/measurement preparation only; no learner or learning comparison.**
Auditor: `v9_artifact_audit`. No canonical source/result was edited or imported.

```bash
.venv/bin/python research/audit_coupled_context_v14_pilot.py
```

Actual evidence is retained in `audit_coupled_context_v14_pilot_results.json`.

## Derivation from Cartesian mechanics

For unit mass at each tip of two unit massless links, q1 is first-link angle and
q2 relative elbow angle. Cartesian positions are

```text
r1=(cos q1,sin q1), r2=r1+(cos(q1+q2),sin(q1+q2)).
T=.5*(|r1dot|²+|r2dot|²)=.5*v' M(q2) v,
M=[[3+2cos(q2),1+cos(q2)],[1+cos(q2),1]].
```

Euler-Lagrange gives the canonical Coriolis vector
`C=sin(q2)*[-2*v1*v2-v2², v1²]`. This is a horizontal system with no gravity,
friction or actuator dynamics. Constant generalized torque does work u·dq.
The independent energy calculation uses Cartesian velocities, not the
canonical matrix expression. It confirms `dT/dt=u·v`.

q1 is cyclic, so p1=M11*v1+M12*v2 obeys dp1/dt=u1. This provides a second
mechanics check independent of the work-energy identity. det(M)=2-cos²(q2)>=1;
eigenvalues are2+cos(q2) +/- sqrt(2)*(1+cos(q2)), so the **global** minimum
eigenvalue is3-2sqrt(2)=.1715728752538097. Positive definiteness is established
analytically, not inferred merely from a sampled pose grid.

## Different state representation and different numerical solver

The audit integrates Hamiltonian coordinates(q,p):

```text
qdot=M^-1*p,
p1dot=u1,
p2dot=u2-sin(q2)*(v1²+v1*v2).
```

It uses implicit Radau, rtol2e-12/atol1e-14/max_step=duration/40, rather than the
canonical explicit DOP853 evolution of(q,v). All153 rest pulses,24 general
energy cases and2 coupling illustrations are reconstructed. Maximum state
differences: rest2.66e-14, general2.62e-12, versus canonical tighter solution
5.62e-14; coupling displacement4.45e-16. Canonical Cartesian work-energy error
is<=1.66e-12 and cyclic-momentum error<=8.27e-13. All measurement/random-preparation
draws reproduce exactly. These support this instrument and tested tolerances,
not numerical validity for every future controller/trajectory.

## Short-pulse expansion and symmetry

At rest with constant torque, time reversal makes q(t) even. If a=M^-1*u,

```text
q(t)=q0+.5*a*t²+(q4/24)*t⁴+O(t⁶),
q4=-M^-1*(a2*M'(q2)*a+2*C(q2,a)).
```

Equivalently q4=M^-1*sin(q2)*[6*a1*a2+3*a2²,a1*a2-2*a1²]. For nonzero torque
and sin(q2)!=0, the vector cannot vanish, so the leading position remainder is
quartic. The42 generic pose/command groups have coarse/middle norm ratios
15.936..16.064 and middle/fine15.984..16.016, consistent with16.

The9 groups at q2=0,+/-pi have q4=0. Their remainder can be sixth order or
exactly quadratic; floating-point subtraction may dominate their tiny errors.
For example at q2=pi with u=(.6,0), q2 stays fixed and q1 is exactly quadratic
in the mathematical system. Ratios there need not be16 or64 at numerical
precision. The JSON field named generic_pose_error_ratios actually retains all
51 groups, including those special cases; do not interpret it as generic-only.

## Coupling and feasibility claims

Initial inverse inertia at q2=0 is[[1,-2],[-2,5]], whereas at pi it is the identity.
A torque only on joint2 therefore initially accelerates both joints at0 but has
zero leading cross response at pi. That does **not** imply exact finite-motion
decoupling at pi: elbow movement changes the inertia and higher-order coupling.

For the five prescribed leading acceleration targets, M is affine in cos(q2).
The global maximum component magnitude of M*target occurs at cos(q2)=+/-1 and
is.5, below the torque limit1. This proves feasibility of those initial local
accelerations. It is not an exact finite-pulse displacement guarantee, an inverse
controller or an identified general-velocity model. Rest pulses alone do not
identify arbitrary velocity-dependent forces without additional assumptions.

## Joint sensing, shared-noise correlation and inverse EIV

The simulated measurements are **joint-angle coordinates** at the initial and
final times, not Cartesian end-effector positions. Their covariance is R I with
R=1e-10. The same initial reading enters both pose context and displacement:

```text
context=q0+e0,
measured_displacement=true_displacement+e1-e0,
Cov(context_error,displacement_error)=-R I.
```

The exact RNG check confirms this shared construction. Thus unconditional
zero-mean displacement error at a fixed true pose does not establish an unbiased
response conditioned on **measured** context. In fact, conditioning on q0+e0
reveals part of e0. There is no learner in this pilot, so no claimed estimator
has already failed; the issue constrains the future dataset/claim.

A separately randomized context measurement removes that direct shared-error
covariance but still leaves latent-pose errors-in-variables. A joint observation
likelihood or a separately warranted known-pose preparation fact may be needed;
independence alone does not turn noisy context into exact true pose.

Naive inverse regression has a second trap even at fixed context. In scalar
Y=B*u+eta with independent input/noise, regressing u on measured Y gives
`B*Var(u)/(B²*Var(u)+Var(eta))`, generally not1/B. Prediction of a command given
a noisy realized response is not the same problem as causing a desired true
response. No inverse method has been trained in the preserved pilot.

## Complete episode accounting and claim boundary

The canonical summary accurately gives153 rest pulses(time8.925) and24 general
energy cases(time12), but omits the2 additional coupling illustrations(time.1)
from an overall total. Complete distinct prescribed cases are therefore
**179 episodes/time21.025**. The24 tighter general-case integrations are numerical
reintegrations, not new independent learning episodes; canonical solver calls
total203. Independent audit replay is verification compute, not extra independent
scientific samples. This accounting correction is preserved here.

No trained mechanism, context-response identification, continuous-feedback
controller, unknown-function generalization or competitive robotics result is
established. Known rest preparation, calibrated instantaneous torque, joint
coordinates, tiny sensor noise and this idealized mechanical family remain
substantial supplied facts. No unresolved pilot artifact discrepancy was found.
