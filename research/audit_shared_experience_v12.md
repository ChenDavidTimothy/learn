# Independent audit of corrected v12 shared experience and model reuse

Auditor: `v9_artifact_audit`. Canonical sources/results were not edited. This
audit imports **no canonical experiment/controller implementation**.

```bash
.venv/bin/python research/audit_shared_experience_v12.py
```

Actual checks are retained in `audit_shared_experience_v12_results.json`.

## Excluded seed-overlap execution and corrected lineage

The first shared-experience execution used a replicate stride smaller than the
candidate-noise seed span. Source inspection caught overlapping streams before
the corrected outcomes were inspected. The entire excluded execution remains as
`shared_experience_v12_seed_overlap_pilot*`, including compressed JSON, NPZ and
timings. It is not used for inferential summaries or for the numerical conclusions
of this audit. The frozen design records the error and correction explicitly.

Corrected seeds use plant stride1e9, replicate stride1e6, CEM at the base,
candidate episodes at base+100000+100*candidate_index+task, and heldout streams at
base+500000+task. Virtual-model CEM uses physical base+1e10 with identical role
offsets. The audit enumerates every role globally: **24,704 distinct CEM,
training and heldout-role seeds**, with no cross-role/replicate overlap. Heldout
noise is intentionally shared across methods/checkpoints and model reuse; that
is a common evaluation role, not additional independent data. NPZ archive hashes
and model-reuse's exact parent-result hash all match.

## Physical experience, CEM and no fitness leakage

Independent eliminated-motor simulation reconstructs all **12,288 physical
training episodes**, each beginning at independently supplied x=v=m=0. The two
checkpoints use literal prefixes of96 and384 segments, respectively8256 and33024
transitions. The latter includes the former; their physical budgets are not
added. Shadow model fitting adds no physical training. Known resets and supplied
function/noise/measurement vocabularies are substantive experimental assumptions.

The direct six-coordinate controller is reconstructed from observations, current
goals, two smoothing memories, one prior-velocity memory and two issued-command
lags. It contains the supplied PD policy exactly when the final three parameters
are zero. The audit reconstructs every CEM population draw, clipping, six-member
elite set, mean/SD floor, previous-generation incumbent reevaluation and current
generation winner at checkpoints. No stale global minimum is used. Every physical
and virtual candidate episode has fresh noise.

Candidate fitness is computed only from observations/actions/goals. True states,
plant parameters and evaluator costs do not enter candidate ranking, elite
updates or checkpoint choice. Full NPZ trajectories, including poor policies,
are preserved. The model learner receives exactly those same complete commands
and noisy positions; it does not receive hidden physical state. CEM's scalar
episode-loss use and OE's use of the full temporal data are distinct supplied
mechanisms, not a universal model-free/model-based dichotomy.

Training state/observation discrepancies are <=3.98e-13; CEM coordinates and
issued direct actions agree **exactly**; observed fitness discrepancies are
<=2.78e-17. All **512 physical/virtual CEM generations** are checked.

## Segmented likelihood and parameter derivatives

Each model fit independently resets its simulation and sensitivity at every
logged zero-state segment; episodes are never concatenated as if physically
continuous. The audit differentiates the eliminated recurrence directly and
checks its analytic sensitivities against finite differences. It independently
recomputes all **384** saved fit objectives, Jacobian singular spectra, bounded
optimality residuals and minimum-SSE model selections, using only the appropriate
checkpoint prefix. Maximum SSE discrepancy1.30e-10; singular value discrepancy
1.64e-11; bounded-optimality discrepancy1.49e-8. All recorded fits converge.

Training has no process noise, so true positions given parameters and issued
inputs are deterministic. Under the known observation-adaptive acquisition
policy, the sequential/joint likelihood retains iid observation innovations;
the policy factor is determined by observed history and its random coins, not
unknown plant parameters. This justifies the raw-position SSE here. It is not
the claim that sensor errors remain independent after conditioning on the entire
future command sequence. Supplied families and noise/reset assumptions remain
essential.

## Heldout control and metrics

All **384 main heldout episodes** and **64 reused-policy episodes** are independently
replayed from logged actions, including active motor noise on every step. There
is no artificial coast or hidden goal-switch reset. Sensor streams, direct
memory recurrences, observation-only EKF beliefs and every score are checked.
All **22,016 MPC plans** are independently differentiated to verify warm cost,
selected cost, projected gradient, action bounds and first issued command.
The free horizon is exactly min(16,86-t), with no terminal zero continuation.

Test state/observation discrepancies are <=2.05e-14; EKF beliefs8.89e-16; task
metrics7.39e-15; plan selected costs1.43e-13; projected gradients1.62e-12.
Every test command agrees exactly. All MPC solves report convergence, with
maximum recorded projected-gradient norm3.57e-5; this is not a certificate of
global nonlinear optimality. Every summary and paired difference agrees exactly
with eight independent training-seed clusters, each averaging two tasks. The
sixteen task trials per cell are not sixteen independently fitted learners.

Physical training costs33024 transitions/384 resets per seed at the final
checkpoint. Main heldout evaluation additionally uses1032 transitions per seed
and does not update the learner. Unequal model-fitting/planning runtime, prior
vocabulary and hyperparameter-search costs must remain disclosed. The saved
timings are compute records, not matched computational budgets.

## Prescribed same-family virtual-reuse diagnostic

All **12,288 virtual training episodes** are independently regenerated using the
final fitted nonlinear model as the simulator. No true plant parameters enter
that search. The identical stationary six-parameter family, gain bounds and CEM
procedure are used. Its final winner is evaluated on the original true plant
with the exact shared heldout streams, after selection. This adds33024 virtual
steps per seed, zero physical training steps, and5504 total physical test
transitions/64 resets. The model family is supplied; an exact physical model is
not handed to the learner.

All prefix70 diagnostics are recomputed exactly. MPC first shortens its horizon
at t71, so this prefix cannot be caused by shortened end-horizon actions. These
are secondary prescribed diagnostics, not a retrospectively substituted primary
score. The comparison still includes EKF/predictive structure and other supplied
controller-class differences.

Same-family reuse gives mixed effects: original-direct minus reused-direct mean
cost is approximately-.00784,+.01963,+.00758,-.04965 over the four plants, with
paired SEs.01407,.01259,.01703,.02720. This does not support a robust claim that
this extra virtual search improves direct policy learning. It controls family
and stationary time features while changing predictive prior use, simulator
bias, optimization randomness and offline computation.

## Scientific claim limits

The shared-data MPC cost comparison tests these complete supplied mechanisms.
Direct search is a bounded stationary rational-controller family with diagonal
CEM, not every direct learner; model MPC has a richer state estimator and known
remaining-time horizon. Equal physical experience does not equalize those
capacities or supplied priors. The final action can change only unscored future
motor state and current effort; zeroing direct u85 removes at most
.03/(86*(1+mean(goal²))) from its cost, but broader horizon effects remain a
controller-class distinction.

Lower average cost does not imply better late-window success on every plant;
all outcomes are retained. Late-window stationarity is measured under continuing
feedback, not after physical disconnection or indefinitely after the episode.
Neither these results nor convergence establishes competitive modern robotics,
architectural necessity, universal model-based superiority or a solution to
general intelligent learning. No unresolved corrected-artifact implementation
discrepancy was found.
