# Active continuation after v12

The general-learning Goal is ACTIVE and unresolved. Latest verified living ledger:
`first_principles_robot_learning_ledger_v12.md`, incorporating complete v11 and all
preceding ledgers verbatim. See `git log -1`/SSH `origin/main` for the current release
commit (a document cannot contain its own commit hash). Prior verified v11 commit:
`90b7e9b8198a1854982588a2ed3553c51681129c`; preparatory pilot checkpoint:
`6dc0c5037b5f23ee62ed04e07914ac595a7fc6bb`. Publish over existing SSH origin and verify
its exact main commit. No release is a stopping condition.

## What v12 actually establishes

- With the declared lagged timing, fresh iid zero-mean position noise and exogenous
  fixed goals/horizon/normalization, observation-only quadratic position/displacement/
  command fitness differs from hidden position/velocity cost by expectation offset
  1.6R/D. Biased/colored sensors or motor-first feedback can break policy ranking.
- Direct CEM improves a supplied six-parameter dynamic stationary controller.
  Shadow segmented raw-position OE followed by EKF/MPC gets lower mean tracking cost
  at both data checkpoints on all four supplied scalar plants from EXACTLY the
  same complete environment experience. This is a comparison of specified mechanisms,
  not equal priors/capacity/compute or universal model necessity.
- Feedback training OE is valid joint/sequential likelihood at zero process noise,
  known zero resets and all episodes retained. Do not claim full-command exogeneity
  or concatenate separate reset episodes. Do not fit only elites.
- Identical CEM inside the learned nonlinear model gives no consistent benefit to
  the SAME controller family. Capacity versus finite search remains OPEN. Learned
  linear and nonlinear MPC remain close; station keeping is unreliable.
- MPC uses remaining86-t; the direct class lacks a clock. Prescribed first70-step
  scores retain the MPC advantage before any horizon shortening, but class/compute
  differences remain. Feedback continues through86; this is not v10/v11 forcedcoast.

## Reproducibility and independent evidence

`shared_experience_v12.py`, NPZ, JSON, design, timing:32 training fits, four plants,
eight seeds each, all384 episodes/33024 training steps per seed. Checkpoints2/8 CEM
generations8256/33024 steps. Final direct means.314099/.331481/.401574/.357066;
nonlinear MPC means.267673/.274189/.266358/.274551. Main test384 episodes/22016plans.
`model_reuse_v12.py`, NPZ, JSON, design, timing:33024 VIRTUAL steps per seed, no new
plant training,64 extra test episodes; same-family costs.321943/.311855/.393994/.406712.

`audit_shared_experience_v12.py`/JSON/report independently reconstruct ALL24576
corrected physical+virtual training episodes,448 tests,22016 plans,384 fit records,
CEM draws/selections, seed domains, fits/Jacobians/EKF/metrics and clustered SEs.
No main implementation imported; eliminated dynamics independently implemented.
Audit passes: states<4e-13, tests<2.1e-14, CEMcoordinates/actions exact, summaries
exact. Main/reuse reexecution yielded byte-identical NPZ+JSON, independent audit
reexecution identical too. Raw metadata says auditpending because it records
pre-audit generation; the separate completed audit certifies it. Do NOT edit that
metadata casually: reuse JSON hashes the exact shared JSON bytes; audits verify it.

Independent theory/design: `audit_observation_loss_v12.md` and
`v12_comparison_design_independent.md`; older PD pilot and audit preserved.
First large run reused some adjacent-replicate noise seeds; caught before viewing
outcomes. All excluded source/data/compressedJSON/timing are retained as
`shared_experience_v12_seed_overlap_pilot*`. Never infer independent SEs from it.
Corrected roles globally disjoint; virtualdomain+1e10; test streams shared only for
intentional paired comparisons. Exact budgets/limitations are in ledger/design.

## Immediately next: one bounded optimizer diagnostic, then change problem

`gradient_reuse_v13_design.md` is prescribed after v12 outcomes, not yet a verified
milestone. Implement/execute it using existing PC/.venv, no new environment:

1. Optimize SAME6coordinate clipped direct family on each final learned nonlinear
   model. Exact pathwise sensitivities through smoothing, true clip derivative and
   lagged plant. Verify numerically and independently, never straight-through.
2. Fixed16-episode training-noise bank, same two training goals/sensor.2/process0;
   three fixed starts (physical winner, virtual winner, generic). Bounded TRF max160
   each; keep failures. Separate16-episode virtual validation chooses among optimized
   candidates and both original winners. No plant test in model/policy selection.
3. NEW plant test streams with all four methods originaldirect, virtualdirect,
   gradientselected and nonlinear MPC, same two newgoal amplitude schedules. Seed
   namespaces and exact counts specified in design; no new plant TRAINING.
4. Good transferred behavior constructively shows family capability and finite-search
   limitation; local failure cannot prove incapacity. Extra modeled computation and
   structural priors stay explicit. Audit independently, preserve code/results.
5. This is ONE diagnostic. Do not escalate through endless optimizers. Then choose
   a consequential coupled/unknown-function-vocabulary/context-learning question,
   derive competing formulations and use strong applicable baselines. Return to
   general-learning relevance, rather than polishing scalar scores indefinitely.

Complete/audit any meaningful new milestone, next sequential ledger embedding v12
verbatim, commit/push SSH/verify and immediately continue. Preserve honest negative
findings. The ultimate objective remains much broader than these simulations.
