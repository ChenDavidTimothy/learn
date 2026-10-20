# Moving experience acquisition contract — exploratory, before sampling

2026-10-20. After v16 and audited moving-information progress commit
7f1775cc1a1865a1680c0a4ff87399e30252c81b. This freezes a first corpus, NOT a v17
milestone or a controller comparison. No learning architecture or success claim
is selected. See cartesian_moving_v17_question.md and independent prospective
critiques. The general-learning Goal remains active.

## Scientific purpose and ordinary interface

Acquire consequences of exogenous interventions during actual continuous motion,
without supplied exact rest, initial position, velocity, joint coordinates, branch,
response matrix or constant-response propagation. A learner may use noisy
Cartesian endpoint readings, issued torques, their clock, trajectory boundaries
and split identities. Timing, calibrated torque units/bounds, noise scale,
independent-trajectory acquisition and a fixed body remain supplied assumptions.
This is not learning without prior knowledge or arbitrary body generalization.

The same fixed conservative horizontal two-link unit point-mass body as v14–v16
generates observations. Plant code is imported ONLY by acquisition/evaluation.
No true-state derivative target or physical feature is put in the ordinary file.
Evaluator states/counters/preparations/noise arrays are in a separate file; merely
separating files is not proof against leakage, so learner source must be audited.

## Frozen first corpus

Replicate0, 24 training / 8 validation / 8 test trajectories, 256 physical intervals
per trajectory, dt=.1 seconds. Camera at every boundary including initial time,
independent N(0,.0002² I2) noise. Initial evaluator preparation chooses theta uniform
[-pi,pi], phi uniform[.4,pi-.4], independent elbow sign; Cartesian speed uniform
[.02,.12]m/s and direction uniform[-pi,pi]. Set q=(theta-sign*phi/2,sign*phi),
v=J(q)^-1 V. These preparation parameters are not ordinary inputs.

Each trajectory starts with eight zero-torque intervals. Then 124 independent
uniform[-.5,.5]^2 commands, each held for exactly two .1sec intervals. Actions
depend only on their independent random stream, never on hidden state or noisy
observations. There are no physical resets within a trajectory. No joint limits,
friction or gravity are silently added. No hidden-state selection, rejection or
resampling occurs. Initial interior/slow support does NOT constrain later motion:
the random torque sequence can reach fast motion and Cartesian singularities.

Independent SeedSequence paths are
[20261020,17,650,replicate,split_id,trajectory_index,role], with split train=0,
validation=1,test=2 and role preparation=0,commands=1,sensor=2. Each role is used
once per trajectory. Log paths, arrays and software/source hashes. Existing local
Python environment, no dependency/environment changes. Never overwrite output.

Expected ordinary resources:40 trajectories,10,240 plant intervals/1,024 simulated
seconds,10,280 Cartesian reads/20,560 independent scalar noise draws. Passive
prefixes cost320 intervals/32 seconds; subsequent interventions9,920 intervals/
992 seconds. Solver evaluations and acquisition walltime are additional counts.

## Conditional support and before-control falsification

The command is deterministic given previous command in a block's second half.
Thus claims about *new* action consequences and ordinary control decisions use
renewal times t=8,10,...,254, each with a two-interval command commitment. Camera
sampling remains .1sec. No arbitrary .1sec-switch planner may claim equal action
support from these records. Passive-prefix zero actions are not an independently
randomized command class.

Before closed-loop comparison, inspect actual numerical consistency and support:
energy/work residuals, camera/joint step sizes, Cartesian speeds, inertia and
Jacobian diagnostics, and tighter-solver spots including extreme motion. Save all
outliers; do not filter by unavailable true state. Singularity affects the sensor
map, not positive-definiteness of the physical mass matrix. Report observed
coverage; don't reinterpret initial sampling bounds as global guarantees.

Heldout next-position error alone is insufficient: inertia and known held actions
can dominate it. A subsequent, separately frozen evaluator-only fork diagnostic
must hold an available noisy history fixed and compare actual continuations for
a prespecified command bank at renewal times. Report paired consequence contrasts,
not only absolute prediction error. Fork states/outcomes cannot train/select the
learner; their extra resets, flows, physical seconds and computations must be
counted as privileged validation, not ordinary acquired experience. Freeze this
diagnostic and learner selection protocol before inspecting its outcomes.

## What this cannot establish

One corpus is an engineering pilot, not independent-corpus statistical evidence.
Boundary-aware memory/smoothness/representation choices remain inductive priors.
A finite lag is not universally sufficient for a noninjective sensor; retained
conventional estimation and history/episodic alternatives require a fair test.
No architecture must win. A serious retained estimator sufficing is a valid
scientific result. Missing action support or optimizer failure cannot refute a
representation. A useful milestone still needs a less supplied acquired relation,
charged feasible decisions, independent audit and fresh replication as justified.
