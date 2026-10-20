# Cartesian experience reuse: frozen construction pilot

2026-10-20. No outcomes yet. This is a learning/control engineering pilot, not a
new milestone by itself or a claim of novelty. Latest verified ledger isv15.

Question: can acquired correlations between possible action consequences reduce
new interaction needed for useful decisions when current observations genuinely
alias physical configurations? Explicit physical state/modes are not prescribed.
Compare conventional alternatives fairly; accepting Gaussian/local adaptation
sufficiency or rejecting directed probing is a valid outcome.

Plant: unchanged conservative horizontal two-link unit-point-mass arm, calibrated
direct torques in[-.5,.5]^2. Only Cartesian endpoint readings and issued actions
reach ordinary learners. Hidden preparations randomly choose qA=(-pi/4,pi/2) or
qB=(pi/4,-pi/2), v=0, same endpoint p0=(sqrt2,0). Exact Cartesian p0 AND rest are
supplied to ALL as preparation resources; no joint coordinates, kinematic formula,
trueB, physical branch labels or relevant output axis reach fitting/planning.
Resetting a hidden branch is an external episode preparation, never an instantaneous
unannounced branch switch during physical control. Goal coordinates are common.

Sampling/pulse durationT=.1sec, independent Cartesian sensorSD=.0002meters.
Acquisition:256training+64validation+64test GROUPED episodes, each two consecutive
commands independently uniform[-.5,.5]^2, NO reset between them. Independent hidden
branch per episode; boundaries reveal persistence only, not the label. All branches,
commands and sensor streams use documented SeedSequence paths
[20261020,16,630,replicate,role,item]. Pilotreplicate0, role1/2/3=train/val/test,
role4=control, role5=fit initialization. Truth retained solely for evaluation.

Supplied local model (approximation, not discovered physics): for j=1,2,

    (p_j-p0)/T² = B * sum_{i<j}(j-i-.5)*u_i + noise/T².

B persists within one episode. Actual nonlinear mechanics/Cartesian transformation
generate ALL observations; no linear surrogate generates claimed physical outcomes.
Groupwise fresh normalized output noise covarianceR=(SD/T²)^2*I. Exact supplied p0
is used efficiently instead of adding a noisy baseline. Current preparation p0
never resolves the hidden branch. Two noncollinear actions make the grouped design
invertible almost surely. This repairs a first-principles caveat found independently
by both auditors: independent one-pulse marginal covariance cannot identify the
antisymmetric cross-output block of a general covariance overvec(B).

All ordinary models share identical acquired groups:

- Coherent mixture of1/2/3 general2x2 response matrices, episode-level EM, known
  sensor likelihood, eight generic initializations per nontrivial component count,
  at most150iterations. Select whole-group validation negative log likelihood.
  Component count is a supplied menu; no physical branch supervision or radial axis.
- General Gaussian random-coefficient prior: ordinary mean regression, full10-entry
  symmetric covariance by grouped residual-second-moment least squares, PSD
  projection and floor0/1e-4/1e-2 selected by whole-group validation likelihood.
  Preserve moment operator rank/conditioning and finite-estimation limitations.
- Gaussian prior matching the learned mixture's first/two coefficient moments,
  isolating prior shape from learned covariance/mean differences.
- Mixture-mean point predictor as a descriptive variance-removal ablation.

These are standard conditional-mixture/empirical-Bayes/random-coefficient tools,
not a new architecture or proof such objects are universal. Fit/validation/test
groups are independent; likelihood selection itself is optimistic. Test joint
AND second-observation-conditional likelihood, not just one-step meanMSE. Repeated
physical groups are the independent acquisition units, not their overlapping rows.

Deployment pilot is explicitly a two-stage command-batch contract: three pulses,
observe after the first, then commit the remaining two commands without another
revision. The terminal physical outcome is scored externally. This removes a planning-optimizer
confound but is not sustained feedback; do not call it general articulated control.
There is NO reset after the informative action, which leaves real motion. All
methods share this contract. Future sustained feedback remains required if the
construction is consequential enough to continue.

Goals are p0+L*d, L=.001meter, d=(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1), the SAME
for both hidden branches.16fresh sensor repeats per branch/goal, one acquired fit.
True Cartesian endpoint velocity is external scoring, not an input. Cost is
||endpoint_error/L||²+.25*||velocity/(L/T)||²+.01*sum_i||u_i||². Position, velocity,
effort, probe motion, success and all bounded solves are reported separately.
Precision thresholds .0003meters and .003meters/sec per coordinate, final endpoint
only, not a sustained station-keeping claim. Noise repeats do not create new fits.

For the supplied constant-B approximation, after first actionu0,

    displacement/T²=B*(2.5u0+1.5u1+.5u2),
    velocity/T=B*(u0+u1+u2).

Condition the learned prior on first observed position exactly once; do not also
initialize an independent state estimate from that same measurement. Expected
quadratic loss for deterministic remainingu1,u2 depends on posterior meanB and
E[B'B], giving a convex four-variable boxQP. Solve it accurately, record KKT/active
bounds. Those moments do NOT generally determine information value before a probe;
different full priors imply different observation-conditioned posterior moments.

First-action rules, all charged the same three-pulse time/effort budget:

- For learned mixture/generalGaussian/matchedGaussian, use a grid of torque
  directions16 and radii0,.025,.05,.1, plus each prior's ordinary three-pulse
  cautious action, and minimize expected posterior-optimal continuation cost.
  Integrate2D observation laws with9x9Gauss-Hermite; record numerical approximation.
- Same priors choose the cautious open-loop three-pulse optimum first action,
  then update from the real first observation and replan the last two. This is
  a serious ordinary task-directed learned-prior/update competitor.
- For mixture and generalGaussian, generic random direction from the same16grid
  with radius chosen to minimize direction-averaged expected cost. All settings
  use the same learned prior; no prior-data discard or hidden extra calibration.
- Point-mean ablation uses the same ordinary cautious/continuation solver.

Quadrature/model-based first-action selection is offline computation, not extra
plant data or a global dual-optimality certificate. Grid includes cautious action
so directed design need not fail simply because it cannot choose that competitor.
Random probing may choose a different radius; total actual objective includes
effort, so do not claim strictly matched energy unless explicitly computed.

Before control results, privileged nonlinear endpoint/velocity optimizers witness
each branch/goal's feasibility withu0=0 and boundedu1/u2. Derive leading feasibility
after every candidate actual probe too: u1=B^-1(delta/T²)-2u0,u2=u0-B^-1(delta/T²).
Large probes can consume stopping authority. Record candidate infeasibility rather
than label a poor continuation lack of knowledge. TrueB belongs only to these
external diagnostics, never model fitting or first-action selection.

Keep all rawq/v,Cartesianobservations,commands,roles, fits, probabilities/covariances,
validation scores, candidate action values, actions and actual endpoints. Audit
independently. No new ledger solely for this pilot, a branch-classification score
or a covariance identity. Literature includes established dual-control turn-off
behavior(Klenske/Hennig2016) and learned adaptation priors(Nagabandi etal2018/19);
no novelty/superiority over their full algorithms is claimed. Strong generic
history/sustained-control alternatives must be addressed before a broader release.

## Preserved first construction and bounded baseline/audit repair

The complete first source/design/data/results are frozen as
cartesian_reuse_v16_pilot*. It learns real unlabeled response relations and uses
them for actual three-pulse decisions, but remains one-fit engineering evidence.
It selected3components(the third weight.0058), with mixture-directed192/192endpoint
precision, matchedGaussian165/192, momentGaussian160/192; cautious mixture64/192
(the vertical targets), random-mixture148/192. No broad superiority is established.

Independent audit identified two protocol omissions: conditional heldout second-
observation likelihood was not saved; full candidate leading feasibility was not
checked BEFORE control (only each executed probe's check was saved afterward).
All executed nonpoint probes were leading-feasible; point ablation128were not.
The revised source computes both missing diagnostics before new control. It does
not filter ordinary candidate actions with privileged feasibility information.

The momentGaussian raw covariance has a spurious secondary positive eigenvalue
.0565(main1.1677), and its forecast likelihood is worse than mixture-matchedGaussian.
To avoid a weak sole Gaussian fit, add ONE bounded conventional Gaussian marginal-
likelihood refinement, retaining all old candidates. Parameterize covarianceFF',
initialize from moment and mixture-matched estimates with covariance eigenfloor1e-8
(full rank), optimize mean4+factor16 using exact likelihood derivatives, L-BFGS-B
max300, maxline40, ftol1e-11,gtol1e-8. No global-MLE claim; retain termination and
gradient records. Validate over oldthree and newtwo Gaussian candidates. Gradient
directional checks precede optimization. MatchedGaussian remains to isolate shape.

This repair follows pilot outcomes and is explicitly exploratory. Reuse EXACT
pilot raw arrays with --input-data; no additional calibration data. New controls
are actual simulator evaluations charged separately, same seeded noise/tasks,
not fresh independent evidence. Preserve the original source/results byte-for-byte.
Do not continue an optimizer ladder: after this bounded baseline diagnostic and
independent audit, either freeze a fresh study or move to a consequential actual
feedback construction. Three-pulse success alone remains insufficient for release.

Independent pilot audit also falsified numerical adequacy of9x9Gauss-Hermite:
generalGaussian candidate winners differ at17/33, with some values still shifting
at65/129. Mixture/matched action rankings were stable, but full-menu values also
were not uniformly converged. Frozen pilot/refinement retain original results.
Replace this integration in a separately recorded corrected batch run with adaptive
2Dnormal cubature(Genz-Malik), atol1e-6,rtol1e-5,max2000subdivisions, standardnormal
domain[-7,7]^2. Requireconverged status, record error/point/subdivision diagnostics
and an analytic tail upper bound via the feasible zero-continuation loss. This
is a numerical verification repair, not evidence for new learning capability.
Independent higher-accuracy checks remain required before a fresh comparison.
