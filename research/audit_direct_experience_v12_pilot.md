# Independent audit of unreleased v12 direct-experience pilot

**Status: preparatory artifact audit only.** No v12 milestone, fair learning-method
comparison, competitive robotics result or general-learning advance is declared.
Auditor: `v9_artifact_audit`; no canonical pilot/source/result was edited.

```bash
.venv/bin/python research/audit_direct_experience_v12_pilot.py
```

Actual checks are in `audit_direct_experience_v12_pilot_results.json`.

## Physics, causal feedback and selection access

Independent motor-eliminated recurrence reproduces all768 training episodes and
32 fresh test episodes. Separate observation-only PD replay reconstructs every
issued action from current observation/goal and two supplied smoothing memories.
Maximum state/observation discrepancy is1.07e-13, action4.11e-15, observed
fitness1.14e-13 and evaluator metric4.55e-13. Targets are exogenous and change at
the declared times. Actions and motor process noise continue throughout all86
test steps; there is no forced-control-cutoff coast in this pilot. Training motor
noise is zero; test motor SD is.03. Sensor SD is.2 throughout.

The learner's CEM generation, candidate clipping, supplied coordinate-to-gain
transform, incumbent reevaluation, top-four elite update and final global-best
choice were independently reconstructed across all eight seeds and384 candidate
policies. Reconstruction uses only observed positions/actions/goals/fitness for
ranking, never true state, plant parameters or evaluator loss. Coordinate and
gain checks agree exactly. Physical parameters are simulator inputs; true state
and actual loss are evaluator-only logged fields. Their presence in a result
record does not imply that the actual selection code reads them; source and
reconstruction confirm it does not.

Each candidate receives two fresh sensor-noise episodes, with unique seeds across
all training candidates. Reusing incumbent coordinates does not reuse its noise.
All training noise seeds are disjoint from test seeds; the learned policy is fixed
before its fresh test noise. Learned/fixed test policies share exogenous streams
within a task. Four generations times twelve candidates times two episodes
times86 steps gives8256 physical training steps per seed, including96 separate
known-zero-state episode preparations. These resets and the supplied controller
family/bounds remain experimental capabilities, not learned facts.

Training winner fitness is selected from noisy trials and cannot be treated as
an independent generalization estimate. A minimum training value can be
optimistic; fresh fixed-policy evaluations are the appropriate loss-identity
check. The two new test goal schedules also shift amplitudes, so training/test
cost differences cannot be attributed to selection bias alone.

## Why the observed fitness has a constant expectation offset here

For a fixed policy evaluated on fresh iid sensor errors, lagged actuator timing
makes v[t+1] and x[t+1] independent of current e[t] and next e[t+1]. Although u[t]
depends on e[t], it first affects the following physical velocity. Thus

```text
o[t+1] - o[t] = v[t+1] + e[t+1] - e[t],
E[observed stage loss - physical stage loss] = R + .3*(2R) = 1.6R.
```

The command-energy terms cancel. Deterministic exogenous goals and normalization
give the saved offset1.6R/(1+mean(goal²)). The derivation requires the stated
timing/noise/exogeneity conditions and a policy fixed before its fresh sensor
stream; it is not a universal noisy-feedback reward correction.

Fresh learned-policy residual gap is .001678 +/- .004617 across eight seed
clusters; fixed-PD gap is -.040450 +/- .026896. These finite samples are
consistent with the derived identity but do not prove it. Large true trajectory
errors can increase sample cross-term variability. Summary SEs correctly average
two task outcomes within each of eight independently trained-policy seeds.

## Bounded motor-first counterexample

The separate contrast uses zero initial physical state and bounded
u=-k*tanh(e0/SD), k=0,.3,.9, with immediate motor-first v1=beta*u. Now v1 depends
on current sensor error e0, and

```text
E[observed stage loss - physical stage loss]
    = 1.6R + .6*beta*k*E[e0*tanh(e0/SD)].
```

The correction is policy dependent. Adaptive one-dimensional quadrature,
independent of the canonical80-node Hermite rule, yields cross moment
.12114110192043179; its difference from the saved quadrature is2.99e-12. The
million-sample experiment is independently reconstructed using the expanded
loss-difference polynomial, not the original subtraction of squared losses.
Means/SEs agree within2.78e-17; sample means lie-.715,-.689,-.575 standard errors
from the respective theoretical expectations. The contrast changes timing and
does not contradict the lagged-environment identity.

## Required continuation before a scientific comparison is complete

The fixed PD gains are a descriptive weak baseline. A large improvement over
them is not a fair learning-method comparison. The direct search uses a supplied
three-parameter PD family, two smoothing memories, gain bounds, CEM design and
known-zero initial preparations. It does not discover those objects or learn a
general representation. Fair same-data output-error/model-based competitors
and a sufficiently capable dynamic control family remain outstanding, together
with explicit accounting for physical data, resets and offline computation.

No unresolved pilot implementation discrepancy was found. This audit supports
retaining the pilot as reproducible preparatory evidence while the persistent
research Goal remains active; it does not authorize releasing an incomplete
competitive-learning claim.
