# Independent v7 audits, 2026-10-20

Auditors: independent_theory and v5_adversarial_audit. Neither edited canonical artifacts.

## Independent mathematical audit

- Main agent and independent_theory separately derived conditional intercept cancellation; q need not be independent of earlier probes. Current gain must be pre-probe measurable.
- Hermite reproduction, zero-at-zero polynomial assumption and minimum conditional L2 norm verified. Minimum weight norm is not minimum full rho-estimator variance.
- Independently verified no integrable Gaussian input/context weight can universally reproduce point evaluation over all bounded continuous zero-at-zero actuator functions. This does not exclude full-data nonparametric identification/localization.
- Randomized zero atom gives exact conditional cancellation for arbitrary zero-at-zero actuation; relevance remains essential.
- Atom policy has conditional mean pi and variance sigma²; minimum branch variance .0135802469 under specified bounds. Sequential mean-zero innovation suffices for prior-sensor-error orthogonality.
- Sample rho ratios are not finite-sample unbiased; state-varying rho, nonlinear passive drift, additive zero-command bias and endogenous randomization remain outside conclusions.

## Independent implementation and numerical audit

- Frozen final source rerun saved at /tmp/feedback_virtual_zero_v7_audit_final.json. Main agent verified final canonical JSON is BYTE IDENTICAL after removing only unused obsolete action-series diagnostics under the atom policy (which frozen source excludes). No numerical results were altered in that schema cleanup.
- All72 final summary dictionaries (4conditions×2sample sizes×9methods) independently recomputed, including means,SE,RMSE,quantiles.
- Separate scalar simulation of quadratic world for both policies reproduces D,Y,u,z,zprev,pi to approximately 1e-15.
- Independent scipy.special Hermite construction and80node quadrature verify reproduction; at pi=.4,sigma=.7,G1=1.37,G2=.21, affine weight's quadratic residual is .0693=G2*(sigma²-pi²), while m2/m4 reproduce zero.
- Saturating conditional force residuals .01837/.07346/.02257 for m1/m2/m4 demonstrate nonmonotonic misspecification.
- Independent1million adaptive-atom samples give E[z*prior sensor noise]=.0000655+/- .0002001SE; innovation variance .04005 against target .04.
- Added action-series IV2/4 use21/41coefficients; independent feature/Hermite implementation reproduces estimates within2.3e-16 on a separate quadratic dataset. Gaussian instrument exogeneity survives noisy observed-history features; actuator/history approximation can still bias drift.
- Simulation semi-implicit qnext(pnext) makes gain depend on previous probe; old unnecessary gain/probe independence is truly violated.

## Skeptical conclusions retained

- Series IV often outperforms weighted moments in RMSE: no estimator-superiority claim supported.
- At65536 samples degree2seriesIV RMSE .005015/.005798 in affine/quadratic cases, versus m2 .007270/.006987.
- Under saturation degree2seriesIV and m2 share strong model-class bias: RMSE .048235/.048516. Actual zero-atom RMSE .019532 with higher sampling variance and mean estimate .649700.
- Under weak probes/noisy sensors, conventional seriesIV RMSE .045148 versus m1 .106701. Higher order is costly.
- Gaussian/atom policies match conditional action mean/variance, not exact trajectories or global energy.
- Classical polynomial reproducing-kernel/IV precedence is explicit. Modern neural/LPV methods have not been benchmarked; no claim to beat them.

No unresolved blocking mathematics, indexing, implementation or final-result disagreement remains within the stated scope.
