# Independent frozen-library retention physics and pairing audit

2026-10-20. **PASS.** This audit fixes its scope from the frozen retention
protocol before inspecting outcomes. The audit executable was completed after
the parent notified completion, without examining result values first. It
imports the earlier independent Hamiltonian/RK4/Radau implementation and score
reconstruction, not the canonical plant, learner or controller. Reusing that
independent derivation is not a new independent derivation. Ordinary local
regression, append content, gradients and optimization have a separate auditor.

Reproduce on the owner's existing environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_cartesian_stream_v17_retention_physics.py
```

The JSON binds source `94e3e5d7ac4e415aa6caecbe1c8042d1ef6a14d63eb5e6fa4a4eb46b4ca0f598`,
final results/arrays, original control artifacts and all recorded dependencies.
It reconstructs all initial preparations, held-pair warmup actions and camera
noise from their original RNG roles. Original online and frozen runs have
identical initial conditions, warmups, observations, sensor-derived goals and
first actual command pair before any append could affect a decision.

All **1,568** held physical intervals independently reproduce under Hamiltonian
RK4 with 256 substeps, maximum state discrepancy **1.87e-12**. All **eight whole
196-interval trajectories**, rolled from their initial state without replacing
intermediate states, agree to **4.12e-11** in state and **1.89e-11** in Cartesian
position. **43** Radau comparisons cover phase boundaries and extrema of
Cartesian speed, joint speed, Jacobian singularity and joint step; maximum
error **1.76e-12**.

All sustained score components, effort profiles, current-goal/command indices,
paired differences and resource counts reconstruct. Precision requires both
Cartesian coordinates and velocities to satisfy the thresholds at all ten
saved endpoints **67–76, 127–136, 187–196**. This is a sampled condition, not a
continuous-time guarantee. The frozen-library variant has 720 decisions,
1,568 physical intervals (156.8 seconds), 1,576 camera opportunities and 3,152
scalar noise opportunities. It introduces no independent noise paths relative
to its paired original runs, and appends no rows. Actual measured counters are
41,608 plant RHS evaluations and 8,943 modeled value/gradient calls; audit
reconstruction of counters does not independently establish historical timing.

The frozen acquired library attains **13/24** precise goal windows (4/4/5 by
phase), versus **22/24** for retained experience (6/8/8). Mean frozen-minus-retained
task cost is **1.628964**, but the four preparation-level paired mean differences
are **0.000852, 0.000151, 6.495421, 0.019431**. The large mean is concentrated in
one preparation; two paired camera streams per preparation do not constitute
eight independent bodies. These are exploratory known pilot cases, not fresh
population confirmation or proof that online append is universally necessary.

The frozen variant still uses an acquired library, causal current history,
current contextual refitting, planning and feedback. It is not a no-learning or
no-adaptation baseline. Once append causes policies to diverge, subsequent
histories differ: the paired difference is the total closed-loop effect of
append enablement, not an isolated per-row estimate at identical histories.

Machine-readable scope, hashes, all eight case profiles and every comparison
are preserved in `audit_cartesian_stream_v17_retention_physics_results.json`.
