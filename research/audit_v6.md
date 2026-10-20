# Independent v6 audits, 2026-10-20

Auditors: independent_theory and v5_adversarial_audit. No canonical writes by either auditor.

## Mathematics

- Verified Gaussian-score/Hermite identities under declared exogeneity and moment conditions.
- Independently derived the exact quadratic-context nullspace: symmetric traceless block on S-perp, dimension max(0,h(h+1)/2-1). Known varied isotropic variance is necessary for the stated trace information.
- For arbitrary non-scalable covariances and unknown intercept, use covariance differences, not absolute covariance span; correction integrated before release.
- Two separate random-orthogonal-subspace matrix-rank calculations checked all 24 cases d3,6,12 and every context dimension. All match theory.
- Q annihilates ones and has trace zero; common mean/common iid amplitude blindness survives overlapping windows and stable output filtering.
- Derived F variance sigma4, lag-one covariance -sigma4/6, all later covariances zero; corrected floor differs from v5's white forcing formula.
- Phase gains beta+eta*(2h,-h,-h)/sqrt3 and covariance mean shift verified.

## Numerical artifacts

- Final context_geometry_v6.py rerun is byte-identical to canonical JSON (/tmp/context_geometry_v6_audit.json).
- Independent loop implementation without scipy.signal.lfilter reconstructs phased AR(.75) command simulation: measured targets within 1.31e-13 and noiseless truth within 8.89e-16.
- Phase/noise/command indices checked. Exactly8190 fit rows,2730/phase; equal expected energy .64.
- All six estimators' means, MC SE, eta RMSE and median error summaries independently recomputed for all noise levels.
- Matched structured baseline correctly reconstructs raw action windows and uses identical17column forms on both designs: 3phase intercepts,8linear innovation lags,6Q windows.
- Covariance-repair theoretical values and all saved summaries verified.

## Consequential skeptical correction

The early phase-versus-raw-moment comparison conflated design with estimator efficiency. Auditor independently constructed a centered supplied-Q FIR baseline that reversed the apparent ranking. Final matched baseline on both designs gives centered/phased RMSE:

| Sensor SD | Centered | Phased |
| --- | --- | --- |
| .1 | .003318 | .003102 |
| .5 | .012580 | .013860 |
| 1.5 | .041138 | .042669 |

No consistent design advantage. This negative finding is retained, not hidden by choosing the weaker baseline.

## Limits

- Q supplied to parametric estimators; full Volterra receives a quadratic vocabulary. No autonomous operator learning.
- Static finite-history quadratic theorem separated from the filtered physical counterexample; static sufficiency does not identify arbitrary hidden dynamics.
- Six Q-FIR terms use finite memory and do not tie all coefficients by rho; negligible truncation remains.
- Paired simulation differences are theorem audits, not oracle information available to a learner in one world.
- Established Gaussian-score/Wiener/BLA mathematics; no novelty or broad superiority claim supported.

No unresolved blocking mathematical or implementation disagreement remains.
