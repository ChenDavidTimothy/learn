# Independent actual-control physics, witness, RNG and scoring audit

2026-10-20. **PASS** for the complete frozen pilot. Audit source and scope were
prepared before reading control outcomes, then executed unchanged. No canonical
source, goal, model, data or witness was changed. This audit treats ordinary
commands as recorded inputs; ordinary optimizer/gradient/local-library replay is
separately audited and must not be implied by this physics/scoring pass.

Executable source: `audit_cartesian_stream_v17_control_physics.py`. Actual results:
`audit_cartesian_stream_v17_control_physics_results.json`. Source220f12b... and all
actual arrays/fit/dependency/contract hashes match. The audit imports no canonical
control, learner, acquisition or mechanics module.

## Actual execution and independent mechanics

All24episodes are preserved:8privileged witnesses plus16ordinary model episodes.
Each has16warmup and180goal intervals. Independent Hamiltonian RK4 uses256substeps
per.1sec interval. ALL4,704actual intervals replay with maximum state difference
1.36e-11. ALL24complete trajectories also replay from their original starts with
NO resetting to intermediate stored states: maximum state difference1.28e-10 and
Cartesian position difference8.55e-11. A separate adaptive Radau check covers124
fixed phase-boundary/last-command and speed/joint/singularity extreme intervals;
maximum state difference5.13e-12. No extra numerical repair was required.

All720witness decisions are independently reconstructed. Unit-link IK is computed
from polar geometry with both signs and an enumerated nearby2pi lattice, rather
than copying the source's inverse formula. Target and endpoint errors are at most
2.22e-16. Coriolis torque is assembled from metric derivatives/Christoffel terms;
computed and clipped torques agree within1.39e-17. There are2saturated witness
decisions and NO goal-radius clipping. These are actual bounded reference paths,
not a proof of global feasibility/recovery or an equally informed learned rival.

## Randomness, information clock and sustained score

Four initial preparations, four shared random warmups and eight paired noise paths
reconstruct exactly from the prescribed namespace. Every physical observation
matches independent endpoint geometry plus its saved camera error. All three
methods sharing a case have identical physical warmups and ordinary sensor-derived
goals; two noise repeats share preparation/warmup but have distinct camera paths.
No exact p/rest/v is supplied to ordinary methods. Goals are verified as.95times
the noisy warmup endpoint and its specified rotations.

Every recorded control decision occurs at t=16+2*renewal, with90renewals per
episode, uses its current phase goal, and issues the same command for both physical
intervals. The separate ordinary policy audit must verify how that command was
chosen, not merely its timing/bounds. This audit verifies no physical or scoring
reset at goal switches.

All score profiles, costs, warmup motion/effort, resource counts and precision flags
reconstruct; maximum sum discrepancy7.11e-15. The final-second precision windows
are the TEN endpoint indices67..76,127..136,187..196 (inclusive). Position error
and TRUE Cartesian velocity are checked percoordinate at EVERY endpoint. This is
the declared sampled window, not a continuous between-sample guarantee. Goal cost
starts after the first issued task interval; warmup has no secretly announced
position goal and is charged separately for motion/effort.

## Verified outcome, with limits

| Complete mechanism | Precise goals | Phase1 / Phase2 / Phase3 |
|---|---:|---:|
| Privileged bounded witness |24/24|8/8,8/8,8/8|
| Frozen nonlinear history predictor/planner |0/24|0/8,0/8,0/8|
| Retained local regression/planner |22/24|6/8,8/8,8/8|

The local first-phase failures are retained; both later phases succeed in every
case without physical/library resets. These differences are real outcomes of the
frozen pilot, not physical simulation, goal clipping or score-index errors.
They do not isolate representation from online local-library revision, locality,
regularization, training/selection or model-error accumulation. Both mechanisms
had weak initial heldout command contrasts; that score alone did not predict this
large difference in complete closed-loop behavior. Do not infer that every future
learned nonlinear predictor fails or that one local implementation is universally
superior. Eight paired cases derive from four preparations and one acquired corpus.

Actual ordinary cost/precision still require integration with the separate
optimizer/gradient/library audit before a combined conclusion. There are no
recorded unsuccessful optimizer terminations or fallback decisions; that is not a
proof that the learned physical loss/gradients are correct or that plans are global
optima. Source/physics witnesses establish nominal authority on their OWN paths,
not recoverability from every failed ordinary trajectory.

## Resources and continuation

Ordinary16episodes:3,136intervals/313.6simsec,1,440decisions,3,152camera-read
opportunities/6,304scalar-noise opportunities. There are8pairednoise paths containing
3,152distinct scalar RNG draws, shared across methods and witnesses. Privileged
witnesses add1,568intervals/156.8simsec, no ordinary training experience. Stored
ordinary/witness RHS totals are92,696/40,768; modeled value/gradient calls21,265.
These reconstruct stored counters, not independently observed historical walltime.

This is meaningful evidence that a conventional retained local mechanism can use
ordinary moving Cartesian records/online consequences for the specified goals,
while the frozen global nonlinear construction can fail badly despite model-level
optimizer success. It is not yet a replicated general-learning or modern-method
superiority result. Preserve both positive local recovery and negative nonlinear
cases; do not alter their goals or refit them from the heldout forks. Any action-risk
or loss-allocation explanation remains a separate explicitly posthoc question.
