# Active continuation after v10

The persistent Goal remains ACTIVE. Latest scientific ledger v10 preserves full
v9 and all earlier versions verbatim. V9 was published at426524a38a041b7e566fece00831b073cfdddab6.
Check Git/SSH origin/main to establish v10 publication status; do not infer it
from this file. Read v10's new section and both independent audits before extending.

## Verified v10 findings

Conventional output-error learning from512 noisy-position commands, EKF and bounded
MPC controls new goal sequences with hidden motor memory on four supplied plants.
Nonlinear replay prediction improves greatly on nonlinear plants but does not
improve the selected tracking objective over learned linear MPC. True-parameter
EKF and learned models behave similarly; full-state diagnostics improve stopping
but still fail some trials. No general-learning/modern-superiority claim.

Full L-BFGS-B pilot source/results preserved: it frequently exhausted40 iterations
and violated numerical pole-swap action equality. Final bounded least-squares
planner has0 convergence failures in33600 solves; independent auditors reconstructed
all672 main episodes plus192 stronger PD episodes, all plan costs/gradients and
clustered summaries. Stronger PD tuning is explicitly post-pilot baseline repair.

Known final motor innovation limits even full-state controller stopping in linear
case: for lambda.7, final-position probability within.3 is at most.712264 at the
population level before extra trajectory constraints. Future noise is not learned
by adding offline data. This does not certify optimal control or nonlinear limits.

## Investigation already begun: v11 (UNRELEASED)

`precision_information_v11.py` follows v10's linear slow-motor c0 plant. It reuses
the exact512-row training prefix, extends the SAME physical trajectory to8192
commands/8193 noisy positions without resetting hidden state, fits the same linear
OE model, and compares short/long/known parameters with test sensorSD .2/.02.
It uses the same test goals/noise seeds/controller/stop definition. Extra training
and improved sensor cost are NOT equated; this is a causal ablation of information
source, not an economic efficiency ranking. Source/results pending independent
artifact audit and mathematical verification, not a released conclusion.

Exact next work:

1. Independently derive/check conditional covariance from ALL history at last
   allowed action t69. Known-parameter linearGaussian KF posterior covariance P69
   propagates to terminal x86 with variance
   (A^17 P69 A^17T)[0,0]+(A^16 Q A^16T)[0,0].
   Last action changes only the conditional mean. Gaussian interval probability
   bounds terminal accuracy; whole-stop success is a subset. Verify conditioning
   under adaptive observation-based actions, indices, zero initialization and
   motor-noise placement. Do not turn an upper bound into an achievable full-stop
   guarantee or apply it to nonlinear c.2 without a fresh argument.
2. Audit/reproduce v11 data prefix, absence of reset, true-state access boundaries,
   sensing intervention, fixednoise/teststreams, fits, every replay/metric/summary.
   Check current sensor precision changes BOTH simulated e and supplied filter R;
   training remains sensorSD .2.12 paired fitted-model seed clusters.
3. Determine whether more offline data or current sensing actually changes useful
   control. Keep all failures. Do not claim exact information-limit attribution
   merely because one estimator/control recipe fails.
4. If meaningful verified evidence emerges, build self-contained v11 embedding
   full v10, retain source/results/audits, commit/push SSH and verify; immediately
   continue. Reassess drift: the larger target is learning mechanisms that acquire
   useful generalizable consequences, not accumulating isolated Riccati formulas.

Owner PC, canonical repo, existing `.venv` only. No external blocker or approval
required. Do not mark Goal complete from release count or finite toy success.
