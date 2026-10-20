# Independent fresh Cartesian feedback study artifact and aggregate audit

2026-10-20. **TESTED: all eight complete corpora, physical/filter/continuation-policy records, and aggregate statistics independently reconstruct.** Actual first-action numerical integrations have a separate theory/numerical audit. This report does not turn numerical estimates into rigorous integration certificates or claim general robotics capability. No canonical source/results were edited.

Reproduce from the repository root:

```sh
.venv/bin/python research/audit_cartesian_feedback_v16_study.py --replicates 100 101 102 103 104 105 106 107
.venv/bin/python research/audit_cartesian_feedback_v16_aggregate.py
```

Per-corpus acquisition and full-control audit JSON files have prefix `audit_cartesian_feedback_v16_study_repNNN`. The aggregate is `audit_cartesian_feedback_v16_aggregate_results.json`. All contain source/artifact hash guards; the aggregate additionally guards independent helper implementations. Full-record caches are reused only if source and input hashes agree. The checker imports only independently written audit helpers, never canonical research implementations.

## Independent reconstruction

All 3,072 acquired groups / 6,144 intervals were reconstructed from declared SeedSequence roles: hidden initial preparation, both uniform commands, all three noisy Cartesian readings, known-preparation response normalization and time-indexed group exposures. Hamiltonian integration independently reproduces the continuous nonlinear motion. The exact known Cartesian preparation is used, not the unused noisy baseline. No true joint coordinates, branch labels or evaluator response matrices enter fitting.

All 136 mixture EM candidates, generic initialization streams, iteration histories, weights, matrices and validation selections reproduce. The 24 Gaussian moment-fit matrices/floors/validation scores independently reconstruct. For the 16 Gaussian likelihood fits, the saved endpoints' training/validation likelihoods and all selections are independently checked; their optimizer trajectories are not replayed. Finite differences of all 20 derivative coordinates at those endpoints have maximum error 6.25e-9. These finite optimizations do not certify global likelihood maxima: all likelihood fits report relative-objective stopping success, rather than meeting a demonstrated strict gradient threshold.

Eighteen EM candidates reach the fixed 150-iteration cap without convergence and remain retained. The deployed predictive mixture in corpus 107 selects an unconverged K=3 candidate. The other seven deploy converged K=2 fits. This is a preserved finite-computation result, not a reason to tune, drop or relabel corpus 107. All 17 candidate predictive curvature laws are independently reconstructed and re-selected using validation, with no heldout/control selection.

The full 640 episodes / 3,840 goal phases reconstruct. Independent augmented position/velocity/coefficient block conditioning matches every recorded belief before and after both scheduled observations. The continuation QP is constructed from the complete raw second moment E[zz'] and terminal linear maps, independently of the implementation's componentwise cross-term expansion. Every ordinary task action, uniform menu, cautious candidate, selected/random first action, bounded continuation, expected continuation loss, effort and precision flag reconstructs. Maximum posterior-mean difference is below 4.3e-13, continuation-command difference below 7.8e-14 and expected-loss difference below 2.4e-13.

All 11,520 physical control intervals replay independently with maximum state difference 1.18e-14. Every nominal witness trajectory is replayed with Radau; normalized witness residual differences are below 1.2e-12. Those witnesses establish their own nominal trajectories' feasibility, not arbitrary failed-state recoverability. There is no physical/controller reset at goal changes. Final observations update the next goal, and only the current goal enters ordinary planning. The fixed Cartesian origin is preserved. All sensor streams and directions reconstruct; 3,104 distinct acquired/control SeedSequence keys comprise 3,072 acquired groups and 32 paired control paths.

Both coarse and tighter `original_value`/`value` integration records are present in all eight corpora. Strict nearby checks, finite-menu argmins, random-radius averages and actual issued actions are checked against the saved numerical records. This artifact audit does not independently reintegrate all 67,456 candidate expectations. Separate actual-integral audits, not a convergence flag alone, determine their numerical verification scope.

## Verified outcomes and uncertainty unit

Costs below are means across eight corpus averages; ± denotes descriptive standard error across those eight values. Each method has 64 episodes and 384 endpoint checks. Endpoints, branches and repeated sensor draws inside a corpus are not independent fit replicates.

| Method | Mean cost ± corpus SE | First-goal cost | Later-goal cost | Precision / 384 |
|---|---:|---:|---:|---:|
| Mixture directed | 0.008429 ± 0.008012 | 0.048574 | 0.000399 | 383 |
| Gaussian directed | 0.024777 ± 0.009488 | 0.093811 | 0.010971 | 363 |
| Matched Gaussian directed | 0.025426 ± 0.009992 | 0.095373 | 0.011436 | 363 |
| Isotropic Gaussian directed | 0.177347 ± 0.010309 | 0.417589 | 0.129299 | 250 |
| Mixture ordinary task | 0.397200 ± 0.035650 | 0.493206 | 0.377999 | 236 |
| Gaussian ordinary task | 0.429290 ± 0.029260 | 0.493149 | 0.416519 | 204 |
| Matched Gaussian ordinary task | 0.406458 ± 0.034520 | 0.493398 | 0.389070 | 216 |
| Isotropic Gaussian ordinary task | 0.698768 ± 0.024783 | 0.820660 | 0.674389 | 106 |
| Mixture random | 0.029663 ± 0.012160 | 0.140234 | 0.007549 | 370 |
| Gaussian random | 0.052300 ± 0.016583 | 0.184325 | 0.025896 | 346 |

All per-corpus metrics, phase costs, first/later counts, effort, mean per-corpus coordinate RMSEs, and seven paired comparisons match the canonical analyzer to floating-point roundoff. A mean of per-corpus RMSEs is not silently reinterpreted as pooled RMSE. Uncertainty is computed after forming corpus averages and corpus-paired differences.

For example Gaussian-directed minus mixture-directed cost is **0.016348709 ± 0.001956210**, with all eight corpus differences positive (range 0.007876926–0.025964958). Marginal mean SEs should not be combined as if these methods were independent. Their observed corpus-cost correlation is 0.989160389; the paired variance identity including covariance reconstructs. This explains the smaller paired SE. It does not create a population-of-robots confidence guarantee from eight prepared corpora.

## Preserved rare failure

The directed mixture is not perfect. Corpus 105, sequence 0, repeat 0, hidden branch 1, phase 0 has actual cost 3.071873411 and x error -0.001752353 m. Its first posterior component weights are approximately (0.876008,0.123992). The bad first observation is not removed, resampled or tuned away. Random probing performs better on this particular case; expected-cost design is not pointwise dominance over every noise realization.

The terminal observation revises weights to approximately (3.83e-7,0.999999617), and all five subsequent goals meet precision without a reset. These complete belief/action records independently reproduce. The same paired case also produces large directed-Gaussian costs, contributing to positive corpus covariance. The mixture's raw mean and large marginal SE must be reported alongside paired comparisons and first/later behavior. One rare failure is not a tail-risk guarantee, and 383/384 precision is not universal reliability.

## Resources, source lineage and reporting correction

Actual acquisition totals are 3,072 groups, 6,144 intervals, 614.4 simulated seconds and 18,432 scalar camera reads. Control totals are 640 episodes, 11,520 intervals, 1,152 seconds and 15,360 reads. There are 32 paired noise paths / 768 distinct scalar control noise draws. Independent reconstruction verifies 1,984 actually computed designs, 67,456 candidate integrations, 5,952 tighter reevaluations and **1,419,473,144 integrand points**. The 11,088 nominal-witness plant-flow calls are additional privileged evaluation work. No offline computational-efficiency advantage is established by matching physical data.

The first four runs had already begun when their misleading control-only acquisition reuse/new-zero labels were discovered. Their exact source and raw metadata are preserved in `cartesian_feedback_v16_study_initial.py` and the original results. Later metadata/timing labels were corrected. Aggregate costs are independently rebuilt from the saved acquisition arrays for every corpus, not from the incorrect first-four labels. AST comparison verifies all non-main definitions—including acquisition, fitting-prior construction, filtering, planning, integration and episode control—are identical between the two source versions. No mathematical/controller change accompanies this reporting correction.

## Scientific boundary

This supports acquisition and reuse of local response structure without joint encoders or physical mode labels, followed by actual repeated goals without goal-change resets. Conventional Gaussian inference is a serious and often successful comparator. Matched moments test prior-shape consequences within this declared policy family; the isotropic diagnostic changes marginal variances and spectrum as well as directional/coupling structure, so it does not isolate cross-action coherence alone.

Each episode lasts 1.8 seconds on the same conservative unit arm, starting from one of two prepared resting configurations. Millimeter goals, exact initial Cartesian preparation, known sensor/actuator timing/noise, fixed coefficient language, fixed goal sequences and approximate constant-B filtering remain strong supplied assumptions. Corpus replication is not transfer to unfamiliar bodies, configurations, goal sequences or sustained stationkeeping. A finite candidate heuristic is not globally optimal dual control. These results do not establish novelty, universal need for modes or Bayesian objects, or superiority to full modern robot-learning systems. Close this prepared instrument's answer without another optimizer ladder, and keep the broader research objective open.
