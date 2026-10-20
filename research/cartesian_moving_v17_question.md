# After v16: when can moving experiences substitute for one another?

2026-10-20. V16 is published and SSH-remote verified at
1f1af0cef125d35a9e143bc7fbdef614a60d22a1. The general-learning Goal remains active.
This begins a new investigation; there is no v17 ledger or learning claim yet.
Read v16 and the independent prospective question before choosing an architecture.

V16 supplied known rest, one Cartesian preparation, a constant torque-response
vocabulary and context-persistence boundaries. Correct Bayesian inference inside
that vocabulary is an adequate explanation of its results. The next uncertainty
is which conditional relationships can be acquired without those certifications.
Retained local estimation, whole-episode conditional completion, generic causal
history learning and direct decision reuse are serious competing formulations.

First challenge the premise: Cartesian branch ambiguity at stationary rest does
not imply that moving histories need a special probe or discrete identification.
Choose two interior arm configurations with the SAME endpoint p and the SAME
Cartesian velocity V, setting each joint velocity through its own Jacobian inverse.
These are evaluator-controlled twins, not ordinary exact-state inputs. Their next
consequences can differ even with zero torque. If natural motion supplies enough
information, a conventional history learner is a stronger explanation than a
forced new active mechanism.

## Independent root derivation, before finite-trajectory outcomes

At polar angle theta=0, write qA=(-phi/2,phi),qB=-qA,0<phi<pi,
s=sin(phi/2),c=cos(phi/2),D=2-cos(phi)^2. Let equal Cartesian velocity have radial/
tangential components(vr,vt), a=vt/(2c),b=vr/(2s). Then

    joint_velocity_A=(a+b,-2b), joint_velocity_B=(a-b,2b).

The Cartesian geometric acceleration Jdot*v is identical in the two worlds:
[-2c(a²+b²),4sab]. Their Coriolis terms are
hA=sin(phi)[4ab,(a+b)²],hB=sin(phi)[4ab,-(a-b)²]. Direct substitution into
pddot=J*M^-1*(u-h)+Jdot*v gives, with delta=aB-aA and u=0,

    delta_radial = -2s*vr*vt/D,
    delta_tangent = (s*vt²+c²*vr²/s)/D.

The tangential difference is strictly positive for ANY nonzero Cartesian velocity
at an interior nonsingular pose. The common-input response difference is purely
radial, so no common torque cancels the complete acceleration difference when
moving. Rotation transports both components to arbitrary theta. At phi=pi/2,
delta=[-vr*vt/sqrt(2),(vr²+vt²)/(2sqrt(2))]. A separate auditor derives the same
identity through the polar kinetic metric with the opposite subtraction convention.

This distinguishes noiseless structural information from finite experience.
Passive acceleration separation is quadratic in speed; with small speed and noisy
positions it can be far below a short observation window's resolution. At exact
rest both passive histories remain identical indefinitely. Endpoints r=0 or2 are
excluded: inverse-Jacobian/polar reasoning is singular there. Exact derivatives,
known velocity and branch matrices are NOT granted to an ordinary learner.

## Frozen first instrument; not a learner or a milestone

Before any model family or controller menu, evaluate actual passive nonlinear
trajectories for phi=pi/3,pi/2,2pi/3, theta=0,.7, shared Cartesian speeds0,.03,.1m/s;
nonzero directions0,pi/4,pi/2,3pi/4. Rest has one direction only. This gives54paired
contexts, two branches each. Independent nominal time sequences have ten0.1sec
zero-torque intervals; no physical reset within a sequence. Keep all states and
Cartesian means, and check matched initial p/V, analytic acceleration, finite-time
leading approximation and rotation. Sampling SD=.0002m matches v16 for comparison.

Compute cumulative squared separation d²=sum_t||pA_t-pB_t||²/SD², KL=d²/2 and equal-
prior known-pair Gaussian testing error Phi(-d/2). The latter is exact ONLY for two
specified trajectory mean laws, independent Gaussian sensor noise and known time/
world alternatives; it is not a trained learner, actual task loss, universal Bayes
lower bound, noise realization or demonstration of successful online diagnosis.
No sensor draws are generated; all mean sequences are evaluator diagnostics.
Retain resource counts and actual finite ODE values, not only Taylor predictions.

The instrument can falsify the framing that Cartesian motion always needs extra
probing. It cannot settle the substantive learning question. If it passes, use its
finite-noise/time scales to choose an honest moving acquisition/support contract,
without exact preparation, future goals, true-state features or a hardcoded
constant-B propagation law. Do not publish another version for an observability
identity alone, enlarge the task merely until a baseline fails, or optimize a
new covariance menu. A useful next milestone must acquire/reuse a less supplied
relationship and test charged decisions against a serious conventional alternative.
