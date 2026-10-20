# Independent pre-control audit of the common planner and physical driver

2026-10-20. **PASS** for the inspected source versus the frozen
`cartesian_stream_v17_planner_contract.md`; no blocking indexing, information or
loss/gradient defect found. No actual witness/control episode was executed in
this audit. No canonical source was edited.

Reproduce numerical checks with
`.venv/bin/python research/audit_cartesian_stream_v17_precontrol.py`.
Actual checks and source/artifact hashes are recorded in
`audit_cartesian_stream_v17_precontrol_results.json`. Inspected control source:
`220f12b194e31e32cea916cb0d65628d5ee0ab15ea3f3de1e6ccfd40b23814ca`;
planner contract:
`35d35de822a5d33767b616a2f15dd71a552989fceffb03696b412b150cccec91`.

## Dataflow and frozen protocol

`modeled_loss`/`choose_action` take only the ordinary selected predictor, actual
noisy position history, issued commands, current goal and previous selected
plan. They receive no physical state, Jacobian, inverse kinematics, future noise
or future goal. The external driver computes the three goals from the ordinary
warmup endpoint, then passes only the current one. Physics/reference functions
are imported in the driver module, but their outputs are absent from this
ordinary decision dataflow; this is a source audit rather than a software access
control guarantee.

The six-block loss averages member-specific coherent paths, charges every
predicted camera position and its finite-difference motion, and adds effort
twice per held command. The first difference anchors to the actual current
noisy reading. Local preparation happens once per real decision; the frozen
coefficients/query are reused by every hypothetical evaluation. A single real
command executes twice, both actual camera readings are appended, and exactly
one completed causal training row is appended afterward. The driver asserts
one-row growth for every member. It preserves the predictor library and previous
selected plan across goal switches and reloads initial selected predictors for
each separate method episode. Actual execution remains to be verified from its
saved stream.

Initial competitor selection, shifted-plan/zero-plan comparison, specified
L-BFGS-B options/bounds, finite-feasible nonworsening acceptance and fallback
match the contract. Failed termination is recorded even when its lower finite
plan is accepted. Selected cost/gradient are reevaluated and the stated box
projected-gradient residual is recorded. Solver status alone is not convergence
evidence; actual projected gradients, fallback rates and cost reductions need
post-run review. `maxfun` is the declared SciPy option, not a proof that every
line search obeys a strict independent call cap; actual calls are recorded.

The four preparations, two paired noise repetitions, shared warmups and separate
SeedSequence role paths match the contract. Witnesses run before ordinary
episodes and supply no learner data. Their inverse target selects among two
branches/nearest 2pi windings, clips radius only for the external inverse, and
scores against the original goal. The fixed computed-torque/Coriolis formula
agrees with the unit-body plant convention. This particular bounded reference
is a diagnostic of nominal authority, not a proof of task infeasibility if it
fails or of recovery from arbitrary failed states if it succeeds.

Scoring uses true endpoint positions and Cartesian velocity only externally.
It charges endpoint-sampled integrated task costs, warmup motion/effort without
an unannounced goal, and all ten final-second sample checks per phase. These
sampled sums are the declared metric, not exact continuous-time quadrature.
All ordinary/witness counts in source match the frozen contract: 16 ordinary
episodes, 48 phases, 1,440 decisions, 3,136 intervals/313.6 seconds and eight
witness episodes adding 1,568 intervals/156.8 seconds. Ordinary 6,304 scalar
noise opportunities reuse eight paired paths containing 3,152 distinct draws.
These are four preparations and one acquired corpus, not sixteen independent
bodies. Partial failure checkpoints preserve current actual streams and plans.

## Independent numerical fixtures and limits

The audit imports candidate control units as tested adapters and replaces their
physical `flow` with a raising guard. Separate saved-weight NumPy history/path
equations, independent local augmented least-squares fits, and recursive analytic
path Jacobians provide the reference. For an actual validation prefix and both
selected models, planner cost agrees within 1.31e-14 and Torch gradient within
8.0e-16; independent scalar central differences agree within 3.14e-9.

Artificial optimizer endpoints exercise four policy branches: failed termination
with equal finite cost is accepted and still labeled unsuccessful; worse final
cost, infeasible command and raised exception all retain the initial competitor.
The recorded projected gradient reconstructs exactly. This tests branch behavior,
not real L-BFGS-B quality or a physical selected action.

All eight generated case preparation/warmup/noise arrays independently reconstruct
exactly, including pairing. A deliberately nonphysical scoring fixture independently
reconstructs phase slices, warmup accounting, final ten-sample precision and total
cost within 7.11e-15. No nominal witness trajectory, real optimizer convergence,
physical goal authority, closed-loop motion, actual online append stream or
control superiority follows from these preflight checks. Those require the
subsequent preserved actual artifacts and independent replay.
