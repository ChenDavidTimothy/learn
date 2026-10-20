# Independent complete persistent Cartesian pilot artifact audit

2026-10-20. **TESTED:** full physical, learning-prior, filter, continuation-policy, timing, scoring and resource reconstruction passes. First-action expectation accuracy is a separate audit and the original Genz-Malik errors remain retained. This is an engineering-pilot audit, not a release recommendation or modern-method superiority claim. No canonical source/results were edited.

```sh
.venv/bin/python research/audit_cartesian_feedback_v16_pilot.py
```

Executable results and exact artifact/source hashes are in `audit_cartesian_feedback_v16_pilot_results.json`. The source imports only an independently written audit helper, never a canonical project implementation. The audited pilot source hash is `18dff96906de372929322e716c3ad377cd5a96b1bfac3239dcd64107d651d5f8`. Parent result and frozen base-module hashes are explicitly checked. The complete parent acquisition was independently audited earlier and is reused without additional calibration.

## What was reconstructed independently

All 17 fitted point-mixture candidates are re-evaluated under their actual deployed Gaussian-component predictive densities. Soft responsibilities, effective group counts, weighted input Grams, coefficient-curvature covariances and validation likelihoods reconstruct. Candidate 11 is selected. The general Gaussian and matched-Gaussian initial beliefs also reconstruct. The latter includes both between-component and within-component uncertainty. These are conventional plug-in uncertainties and validation selection, not exact integration over unknown assignments or learned physical labels.

Nearly empty discarded candidates have large curvature covariances because their Grams approach the declared ridge. Their largest absolute reconstruction difference is 1.53e-10, negligible relative to their matrix scale; selected-mixture initial covariance differs by less than 4e-18. No candidate was silently removed. Maximum validation likelihood difference is 2.3e-13.

The checker independently propagates the normalized random vector [Cartesian displacement/T², Cartesian velocity/T, row-major B]. It conditions complete Gaussian components using block conditioning and independent Gaussian log densities. The continuation QP is derived by applying terminal linear maps directly to the full raw second moment E[zz'], rather than reproducing the implementation's expanded state/coefficient cross-term formula. This retains the same correlations through a different derivation.

Every stored belief before/after both observations in all 384 goal phases reconstructs. Maximum posterior-mean difference is 3.8e-13, covariance difference 7.3e-14 and component-weight difference 4.2e-15. The minimum component covariance eigenvalue is -7.1e-16, consistent with numerical roundoff. No terminal component weight becomes exactly zero. Every ordinary task action, cautious candidate, learned variation direction, random radius, realized first action, bounded continuation and expected continuation loss reconstructs. Maximum continuation-command difference is 8.9e-14; maximum expected-cost difference is 9.1e-14.

The first action for directed/random methods is checked against the recorded candidate scores and RNG; this does not independently reintegrate those scores. The separate `cartesian_feedback_v16_theory_independent.md` and actual-integral audit artifacts establish their algebraic and numerical findings, including Genz-Malik false-convergence examples and the limited set of checked action orderings. This audit does not replace those limitations with a blanket numerical pass.

## Physics, causal information and goal continuity

All 1,152 issued physical intervals independently replay using Hamiltonian RK4 with maximum saved-state difference 8.5e-15. Thirty separately selected Radau intervals agree within 8.9e-15. All 24 nominal nonlinear witnesses are replayed with Radau (72 intervals); normalized endpoint/stopping residual differences are below 1.2e-12, and their saved phase-to-phase continuity is exact. They certify their own nominal trajectories, not arbitrary recovery states reached by another controller.

All camera draws and issued random directions reconstruct from the four declared SeedSequence paths. The first camera read follows the first actual pulse. The final read follows both committed continuation pulses and updates only the next goal's belief. Later phases begin with exactly the preceding terminal belief and physical state. The fixed Cartesian origin is retained. There is no goal-change rest reset, covariance reset, hidden branch switch, true-state filter input, or future-goal argument in ordinary decisions. Future noise is generated in advance by the environment but only its scheduled current observations reach the filter. Initial cache reuse is legitimate because its prior/current goal is identical; subsequent designs use each trajectory's current belief.

## Verified descriptive results

Each method has eight episodes, 48 goal endpoints, two fixed goal sequences, both hidden preparations and two paired sensor repeats. There is one learned corpus and only four independent paired noise paths.

| Method | Mean phase cost | Precision endpoints / 48 | All six precise episodes / 8 |
|---|---:|---:|---:|
| Mixture directed | 0.000376472 | 48 | 8 |
| Mixture ordinary task | 0.417843872 | 27 | 0 |
| Mixture random | 0.084058946 | 44 | 5 |
| General Gaussian directed | 0.034535144 | 41 | 2 |
| General Gaussian ordinary task | 0.566171917 | 19 | 0 |
| General Gaussian random | 0.074462837 | 41 | 4 |
| Matched Gaussian directed | 0.033992108 | 42 | 2 |
| Matched Gaussian ordinary task | 0.475099162 | 21 | 0 |

Every endpoint position error, Cartesian velocity, effort, cost and success flag recomputes. The JSON retains every individual phase and phase-specific summaries. These endpoint thresholds are not within-phase stationkeeping. Ordinary methods often improve after early interactions; their complete time profiles must remain visible. These data do not show that every Gaussian, conditional-mean, history-based or conventional controller must fail.

## Resource accounting

All source counters reconstruct exactly: 64 episodes, 1,152 control intervals, 115.2 simulated seconds, 1,536 scalar camera readings, four paired paths and 96 distinct scalar sensor draws. There are 5,280 candidate records attached to serialized phases but only 206 actually computed designs / 4,532 actual candidate integrations, totaling 40,160,748 integrand points. The checker reconstructs phase-zero cache reuse explicitly rather than deduplicating unrelated equal-valued records. Privileged nominal witnesses account for 1,386 plant-flow calls, separate from learner experience. Audit reintegrations are validation computation, not new ordinary calibration.

## Scope and fresh-study review

This is persistent repeated two-stage-per-goal feedback under an explicitly supplied local constant-B approximation. It is not arbitrary per-pulse feedback, general articulated robotics, discovery of physical state, or a general architecture. Exact initial Cartesian preparation/rest, known noise and actuator semantics, a small coefficient vocabulary and very small goals remain strong supplied resources. The nonlinear body need not satisfy the local filter model indefinitely; no drift/process correction is inserted.

The proposed fresh study keeps physical/scoring/filter mechanics, adds a trace-matched isotropic diagnostic, uses a common uniform direction grid plus each model's cautious action, and replaces the failed integrator with quadrant-partitioned gk21 and tighter nearby checks. Its stated diagnostic interpretation is appropriate: isotropy changes both directions and marginal variances/spectrum, so it is not a pure cross-action-coherence ablation. Fresh corpora provide fit replication on the same prepared task, not new bodies or unseen goal sequences. One reporting issue was sent to the coordinator during the first four study runs, before publication: a fresh-acquisition invocation must not report its combined calibration cost as zero/reused, and control-phase timing must be distinguished from acquisition time. Their exact old source/results were preserved, and later-run reporting was corrected. No fresh-study result is validated here.
