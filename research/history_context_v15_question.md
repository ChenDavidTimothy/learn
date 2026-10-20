# V15 next question: when must experience revise a summary?

V14 learns only rest-pulse relations. Remove rest. If two histories reach the SAME
current pose with different motion, the same goal can require different actions.
Fitting the pose-to-response function more accurately cannot remove that aliasing.
What additional distinction is necessary, and can a learning mechanism detect/use
it from available experience rather than accept a scientist-supplied physical state?

Do not assert that named velocity/state/world models are universal foundations.
Possible mechanisms include finite position/action history, learned compression,
direct history-to-action feedback, or conventional observer/physical dynamics.
The discrimination concerns information retained by a summary and useful action.

First pilot, not a milestone: horizontal two-link body, current pose fixed, equal
prior opposite initial velocities, same return-to-current-pose goal and onebounded
constant torque opportunity. Past positions differ while current input/goal do not.
For an explicitly simplified ballistic/no-Coriolis response model,

    displacement=T*v + .5*T²*M(q)^-1*u,

any current-pose-only action independent of velocity sign has expected squared
endpoint error T²||v||² + .25*T⁴||M^-1*u||², minimized byu0. History that resolves
motion can choose different commands. This is an EXACT identity in the declared
leading approximation, not a finite-time global control certificate for nonlinear
ODEs. For exact continuous endpoints fplus(u),fminus(u), the midpoint/half-separation
identity holds, but a grid/local minimizer cannot prove a uniform action lowerbound.
Time reversibility gives separation2T*v+O(T³) uniformly on suitable compact sets;
state assumptions and avoid unjustified quantitative finite-T claims.

Verify individually feasible bounded endpoint commands for BOTH histories, otherwise
failure may be unreachable task. Use actual continuous mechanics and independent
checks. Return-to-position ends moving and is NOT stopping. Currentpose/goal being
supplied exactly must be explicit (goal is the same q0, so already reveals q0);
no current true velocity or future sensor noise may enter ordinary commands.

A limited construction can reuse learned v14 inverse relation plus a fixed causal
past-position slope. Such a slope is supplied arithmetic, not autonomously discovered
state. It may show the learned rest relation can be used with history, but cannot
complete the research question. A serious follow-up must select/revise history from
continuous experience, compare strong applicable derivative-free/history methods,
and evaluate useful new decisions rather than only hidden velocity estimates.

Candidate fixed pilot settings before outcomes: two poses(0,pi/2),(1,.3), velocities
plus/minus(.02,-.01),(.01,.02),(-.015,.005), horizonT=.2, four past samples at
-.2,-.15,-.1,-.05, supplied current q, sensorSD1e-4. Construct legitimate zero-torque
prehistories by backward/forward integration and verify they reach q/v0. Use all8
broad-data v14 calibrated inverse fits,16 fresh sensor seeds per pair. Fit a slope
anchored at the SUPPLIED current q; do not add avoidable currentreadingnoise.
History actions use learned K, no true M or v. True-state endpoint oracle is a
privileged feasibility diagnostic. Preserve every realization; no v15 release for
this pilot alone or an isolated aliasing identity.


Pre-outcome source correction: initial pilot seedbase20261020+2300000 equaled
v14's first acquisition seed (v14base+100000). It was caught before reviewing
pilot outcomes. Full firstsource/results retained as history_context_v15_seed_collision_pilot*,
excluded from any claim of fresh noise. Canonical seed namespace20261020+50000000000
is disjoint from earlier v12/v13/v14 role ranges; model/world/repeat offsets remain.
No controller, task or physical setting changed. Audit freshness globally.


Independent theory correction before publication: the physical non-rest acceleration
is M^-1(u-h(q,v)), with h even in v. The displayed surrogate omits that common
Coriolis term. For the actual reversible dynamics, paired endpoint midpoint minus
current q is .5T²M^-1(u-h)+O(T4), and half-separation=Tv+O(T3). Therefore leading
paired risk is T²||v||²+O(T4) uniformly over bounded torque on suitable compact sets.
The bare T²||v||² value is a ballistic-surrogate reference, not an exact finite-T
lower bound or a global certificate that zero command is optimal in the nonlinear
plant. Rename the stored reference field accordingly; all numerical trajectories
and methods remain unchanged. A history learner need not name velocity or h.
