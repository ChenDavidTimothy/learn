# Active research continuation after v9

The persistent Goal is ACTIVE. Latest verified scientific release is v9,
`first_principles_robot_learning_ledger_v9.md`, preserving complete v8 and all
earlier versions verbatim. Read the new v9 section, source/results,
`v9_theory_independent.md` and `audit_v9_artifacts.md` before extending.
Publication status is established by Git and SSH origin/main, not this file alone.

## Verified findings

Nonlinear curvature does not universally locate passive versus actuator force.
The lagged transform q=m+phi(v) preserves all ordinary connected command/position
behavior; preserving physical interventions also requires mapping their semantics.
Explicit distinct physical drag locations (body versus removable motor assembly)
match connected behavior but differ after physical removal. Saved clean12-step
response distance .27338567; pre-change expected-RMSE lower bound .13669283.

Within the supplied velocity-independent linear-actuator/tanh-passive family,
nonzero curvature labels the motor pole under full-rank exact observations.
Standard raw-position output-error learning handles measurement noise, whereas
same-family bounded one-step regression is biased. At c=.2/trainSD.7, mean shifted
velocity RMSE .009084 vs .288947; at trainSD.08 OE transfer error .23414. Known
zero initial state, no process noise and correct supplied family are major limits.

## Exact next investigation (begin immediately)

Move beyond component-label theorems to a complete observe → fit → plan → act
cycle with new reach-and-stop goals. The fundamental uncertainty is whether
useful learned command consequences suffice under noisy partial observation and
energized actuator memory, and whether nonlinear learning is needed at all.

1. Freeze a small plant grid (motor memory/curvature) and task/score definitions
   before results. Switch goals with no hidden-state reset. Include new target
   signs/distances, stop tolerance and terminal zero-command tail.
2. All learned models get the same command/noisy-position training. Compare
   conventional linear delay/OE dynamics with nonlinear OE; optionally a nonlinear
   observable delay predictor. Use common bounded receding-horizon action search.
3. Essential practical baseline: observation-only PD tuned on separate training
   tasks under equal access. A true-state/true-dynamics oracle is privileged and
   only diagnoses feasibility/search limitations. No true state to learned control.
4. Score actual motion, residual velocity, effort and predictive replay separately.
   Save trajectories/actions and all failures. Do not infer success from raw
   parameter labels. Equivalent force gauges give identical ordinary planning.
5. If a consequential uncertainty remains, compare equally budgeted random and
   coverage exploration against prospective observable prediction disagreement.
   Save forecasts BEFORE executing/refitting; gauge-label variance is not useful
   command uncertainty and numerical explosions are not automatically information.
6. Audit independently, consult primary literature, publish substantive v10 with
   full v9 and SSH push/verification, then continue immediately. Negative results
   favoring simple established methods count as evidence, not failure to progress.

Use owner's PC, canonical repository and existing `.venv`. No external blocker.
