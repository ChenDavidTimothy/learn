# Prescribed v14 compact coupled-context learning test

Written after mechanics pilot/independent critique, BEFORE any learning outcomes.
One compact test then move to continuous coupled motion or missing-history revision;
rest-pose calibration alone is not a general learning mechanism or destination.

Question: can equally acquired experience support NEW coupled displacement decisions
through either learned consequences or a direct inverse relation, without giving
learners the inertia/cosine formula? No universal model-necessity comparison.

Continuous horizontal two-link point-mass arm from the verified pilot. Known rest
PREPARATION POSE q0 is supplied to every ordinary learner, BOTHjoint encoders,
control bounds [-1,1]^2, durationT=.1 and independent sensorSD1e-4. This explicitly
supplies exact preparation context; it is not inference from a noisy current robot
state. No hidden masses/velocity/force/geometry formula enters ordinary learners.
Positions are joint angles at beginning/end in time, not end-effector coordinates.
Measured response d=(oT-o0)/(.5T^2), noise covariance R=2*SD^2/(.5T^2)^2 I.
Since context is the separately supplied preparation fact, shared o0 is not used as
noisy context. Count resets/preparation capability; no claim of practical robot cost.

Conditions: broad q1/q2 uniform[-pi,pi], or same q1 but q2 uniform[-.4,.4]. Eight
replicate datasets each,512 training+128 separate validation rest-pulse episodes,
commands uniform[-1,1]^2 independent of context. Same complete data to all learners.
Known physical rest preparation is supplied at EVERY episode; no continuum control.
Seed domains conditionstride1e9, replicatestride1e6, dataoffset1e5, testoffset5e5.

Four ordinary conventional alternatives:
1. Constant FULLcoupled forward response matrix, ridge fit d≈B*u.
2. Context-conditioned forward matrix B(q), periodic Gaussian RBFfeatures over BOTH
   encoders, centers8x8uniformanglegrid plusconstant; d≈B(q)u.
3. Same feature family for direct inverse K(q), u≈K(q)d, naive ridge.
4. Calibrated moment-corrected inverse, same family. Subtract from X'X the known
   sum phi(q)phi(q)' tensor R, then regularize/eigenfloor. This corrects ordinary
   errors-in-variables bias under the stated linear-in-d approximation; finitepulse
   nonlinearity and regularization remain. Not a claimed exact universal likelihood.

For context learners choose width in{.5,1,2,4}, ridge in{1e-6,1e-4,.01,.1} by their
separate validationdata. Same candidate grid/experience for both inverse variants.
Inverse validation uses corrected squared error ||K*dobs-u||²-tr(K R K'), which has
unbiased expectation for a FIXED candidate versus the noiseless response under
independent output measurement noise. Forward validation uses rawpredictionSSE
(the outputnoise constant does not change ranking). Validationselection is optimistic;
fresh tasks decide. Eigenfloor and all rejected/failed candidates must be retained.
The same known R is supplied to all; naive inverse is a descriptive bias baseline,
NOT the sole direct competitor. Model families/control-affinity are supplied priors.

Response-feature design X=phi⊗issuedcommand; inversefeatures X=phi⊗measureddisplacement.
All fits retain2outputcoefficients and training/validation scores, eigenvalues,
regularization scale and selected settings. Constantmatrix fits ridge grid too.
Global ridge solutions are numeric linearalgebra; EIVmomentregularization not a
global nonlinear optimum or arbitraryfunction discovery. No proper physicalnames fit.

Heldout tasks BEFORE outcomes: q1[-2.7,-1.1,.4,2.2] crossed withq2[-pi,-2.1,-1.2,
-.4,0,.4,1.2,2.1], common desired normalized displacements z in{(.1,0),(0,.1),
(.1,-.1),(-.1,.1),(.05,.1)}. Desired actual displacement=.5T²z. Allleadingorder
true torques are<=.5 bypilot; the privileged continuous endpoint oracle must retain
a numerical bounded-command feasibility witness/residual for EVERY task, or report
task infeasibility. Its optimization rollout cost is counted separately and never
supplied to ordinarylearners. No globalfeasibility theorem follows. Every controller gets sameqprep andz.
Forward methods solve boundedleast-squares min||B(q)u-z||²; inverse methods use the implied pinv(K(q)) in the SAME bounded displacement
least-squares decision rule, reporting conditioning. For a nonsingular map and
unconstrained feasible solution this equals K(q)z. This avoids a command-space
clipping versus displacement-space projection confound when extrapolated fits
request infeasible commands. The clearly privileged diagnostic starts with M(q)z
and numerically optimizes the exact continuous endpoint under the same bounds;
it is not a learner/baseline entitled to physicaldatafairness. All get actualODE
endpoint scoring. Retain every command/truthstate/noisyendpoint, no task updates.

Primary score: squared actualdisplacement error normalized by(.5T²)^2, averaged
across160tasks then across8independent acquisitionseeds withpairedSE. Also command
saturation, max/quantile errors, taskgoalerror and contextdependence/q1-invariance
ofchosencommands. Pulse ends moving; this is NOT stationkeeping/reach-and-stop.
Compare broad versus narrowcontext acquisition at matchedbudgets, not just samples.
Privilegedpose/family/timing/noisemetadata and counterexamplelimits stayexplicit.

No hypothesized winner. EIV-corrected inverse and forwardconditioned are serious
competing forms of usableknowledge; beating a constantmodel or naiveinverse cannot
settle architecture. A success ofsimpleestablishedmethods is acceptable. Any v14
ledger requires independent physics/statistics/numeric audit and actual heldout
results, completev13history andSSHpublication. No v14 release solelyforpilot.

Before any learning execution, also prescribe128 fresh broad-context random-command
response episodes per acquisitionseed (new RNGoffset600000), independent of fit/
validation/task commands. Score true normalized displacement prediction for forward
B and implied pinv(K) for inverse models, with inverse conditioning reported. This
is a secondary predictive-transfer diagnostic, not extra training; adds2048pulse
resets total. Ordinary heldout-noise observations are retained but not used to fit.
No singular case is dropped. Together with decisions this tests acquired knowledge
rather than merely fitting a training-response table. Main task streams offset500000,
all below nextreplicatestride1e6. Timings, solver failures and exact budgets retained.


Independent review corrected the initial testgrid duplicate -pi/+pi BEFORE execution:
replace +pi by0, retaining160tasks and adding genuine narrow-region interpolation.
Angular wraparound topology is explicitly supplied to the RBF family. The common
bounded consequence-metric action rule is also a pre-execution fairness correction;
this comparison concerns learned relation parameterization, not pure model-free
necessity. Validation may favor a noisy highnorm inverse because noise-corrected
scores can be negative; retain ALL such scores and conditioning/eigenfloor details.
Fresh tasks, not a lucky corrected validation minimum, determine evidence.
