# Independent corrected Cartesian batch audit

2026-10-20. **PHYSICAL / STATISTICAL RECONSTRUCTION PASSED; FIRST-ACTION INTEGRATION ACCURACY FAILED.** This report concerns the immutable `cartesian_reuse_v16_adaptive.py` and adjacent adaptive design/data/results. Failed GH pilot and refinement artifacts remain separate. No canonical source or results were edited.

Reproduce:

```sh
.venv/bin/python research/audit_cartesian_reuse_v16_adaptive.py
```

`audit_cartesian_reuse_v16_adaptive_results.json` records exact artifact/source hashes, all 900 independently checked candidate expectations, all discrepancy diagnostics and actual summaries. The checker imports only the independent frozen-pilot audit helper, whose hash is verified; it imports no canonical project implementation. Frozen adaptive source SHA256 is `9a067f6b7f3e540641eb5a5276bf2894f6801cd761d0bea828a3b13290839e36`.

## What reconstructs

The complete raw NPZ equals the independently audited original acquisition byte for byte (SHA256 `014c0350acbb828e1300afb709773a81ee6416f6a8bf3377b748d1087624c6b4`). No new calibration was acquired. All 17 mixture candidate records and first three Gaussian moment-fit records are identical to that parent. Exact known Cartesian preparation, rest, action calibration, known noise and local constant-B vocabulary remain supplied structure. There is no branch supervision, true-coordinate feature, or test-group model selection.

The two additional Gaussian fits use grouped marginal likelihood with covariance F F'. Independent train/validation likelihoods match within 3e-15. All 20 derivative coordinates at both endpoints agree with finite differences within 7.1e-9 (step 2e-7). The selected record is the mixture-moment initialization, with training NLL -7.093593728 and validation NLL -7.264527821. Covariance is positive semidefinite to numerical precision. Both finite L-BFGS runs report success from their stopping rules; this is not a global likelihood certificate or evidence that a strict gradient tolerance was met. Independent factor-gradient Frobenius norms are approximately 1.92e-4 and 8.46e-5; mean-gradient maxima are 7.45e-5 and 2.46e-5. The original factor rotation is not saved, so its entrywise gradient maximum is not an invariant endpoint reconstruction target.

All heldout joint/first/conditional likelihoods and mean MSEs reconstruct. Every saved posterior, second moment, first action, continuation QP and actual score in all 1,728 control episodes reconstructs. Maximum continuation action difference is 6.4e-15; maximum projected-gradient residual is below 1e-7. All physical trajectories independently replay with Hamiltonian RK4 to 4.0e-15. The complete external candidate feasibility arrays reconstruct exactly; the repaired source computes them before controls without using them to select actions. The prescribed conditional likelihood is now saved. These repair the prior pilot's two implementation omissions.

Actual mean costs and success counts are verified:

| Method | Mean actual cost | Precision / 192 |
|---|---:|---:|
| Mixture directed | 0.000947949 | 192 |
| Mixture task | 0.666607659 | 64 |
| Mixture random | 0.178850321 | 148 |
| Refined Gaussian directed | 0.036128083 | 168 |
| Refined Gaussian task | 0.667183072 | 64 |
| Refined Gaussian random | 0.203853125 | 125 |
| Matched Gaussian directed | 0.036181712 | 165 |
| Matched Gaussian task | 0.666607155 | 64 |
| Point task | 591.414627911 | 64 |

There is one acquired fit, with shared sensor paths across methods/branches. These outcomes are exploratory reuse of the same evaluation cohort, not a fresh confirmatory population comparison. The point ablation retains the known stopping-authority confound and does not establish universal conditional-mean failure.

## Tail bound is valid; adaptive error estimate is insufficient

The integral is over standard normal z in [-7,7]^2. Its nonnegative optimized continuation loss is upper bounded by the feasible continuation with both remaining commands zero. For a finite point mixture, the largest component zero-continuation loss bounds every posterior average. The discarded probability is at most 4 Phi(-7).

For a Gaussian coefficient prior, the posterior mean is affine in z and the posterior covariance is constant. Applying ||a+b||^2 <= 2||a||^2+2||b||^2 gives a feasible-loss envelope a+b||z||^2. The complement-of-square second moment is bounded by

    E[||z||^2 1{max|z_i|>7}] <= 4*7*phi(7) + 8*Phi(-7).

Independent reconstruction verifies every saved analytic truncation bound; their maximum over all 900 candidates is 3.91e-9. This controls discarded tails under the declared probabilistic model. It does not control the cubature error inside the square.

The canonical Genz-Malik rule reports convergence, but independent tensor Gauss-Kronrod (`gk21`) integration with atol 5e-9 / rtol 1e-7 finds material errors. This audit evaluates the complete 50-candidate menu for each of 18 prior/goal cases, reusing the exact local-model simultaneous sign symmetry for opposite actions/goals. It independently constructs likelihoods, posteriors, response moments and box QPs. Both rules use SciPy's adaptive engine, so this is a different-rule/codepath check, not two unrelated integration libraries.

| Prior, zero-based goal, candidate | Saved Genz-Malik | Independent gk21 | Difference |
|---|---:|---:|---:|
| Mixture, 4, 46 (selected) | 0.008256303 | 0.010201587 | +0.001945284 |
| Mixture, 4, 44 | 0.057844 approximately | 0.056022 approximately | -0.001821756 |
| Gaussian, 2, 21 | 0.003401832 approximately | 0.003299027 approximately | -0.000102805 |
| Gaussian, 3, 29 | 0.003401832 approximately | 0.003299027 approximately | -0.000102805 |

The selected mixture discrepancy is about 1,848 times its saved estimated error; the analytic tail bound is far too small to explain it. Every listed discrepancy is independently corroborated by four scrambled Sobol estimates with 16,384 standard-normal points per component per scramble. The selected mixture value averages 0.01020457 with scramble-mean standard error 3.11e-6, agreeing with gk21 rather than the saved result. Mixture candidate 44 has QMC mean 0.05602261 and standard error 2.35e-7. The two Gaussian values have QMC means near 0.003299027 and standard errors near 1e-10. QMC scramble errors are empirical diagnostics, not deterministic integration bounds.

The full gk21 menus retain all 18 directed winners and every direction-averaged random radius. Thus these detected value errors do not change the actions or actual outcomes in this saved batch. This fact does not make the original integration accurate. In particular, refined-Gaussian vertical winners have only a roughly 1.52e-6 gap to their next saved candidate, smaller than the sum of their original reported error estimates. The tighter independent checks preserve their ordering, resolving that observed near tie numerically.

Maximum estimated error in the independent complete-menu integrations is 8.42e-8. No rigorous global numerical certificate follows from that estimate either. A repaired production integrator should retain cross-rule/QMC discrepancy evidence and a disclosed accuracy criterion rather than interpreting `converged` as a proof. A finite candidate menu also remains a heuristic restriction; accurately integrating every menu value does not establish globally optimal dual control.

## Scope and next audit

This batch repeats 1,728 physical control episodes / 5,184 intervals while reusing 768 acquisition intervals; it uses 96 paired sensor-noise paths. The source's nonlinear witnesses remain privileged evaluator costs. No sustained-feedback claim follows from three pulses with one intermediate observation and then two committed actions.

The next persistent-filter source has been inspected structurally: its joint position/velocity/coefficient transition and QP include the needed cross covariances, and terminal observations update the next goal without resets. That source also uses Genz-Malik and therefore inherits an unresolved numerical accuracy concern. Its saved episodes require their own independent filter/trajectory/expectation audit. This report does not validate that extension or recommend a new ledger release.
