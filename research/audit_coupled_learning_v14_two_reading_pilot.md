# Independent audit of the superseded two-reading coupled-learning pilot

**TESTED / independently verified, 2026-10-20.** This audit imports no canonical research implementation. Run from the project root:

```sh
.venv/bin/python research/audit_coupled_learning_v14_two_reading_pilot.py > research/audit_coupled_learning_v14_two_reading_pilot_results.json
```

The JSON records hashes of the prescribed design, source, full result JSON and NPZ. The canonical result's `audit pending` metadata describes its generation stage; this later report records independent verification without changing the original numerical artifact. No methods, settings, seeds or cases were changed following the results.

## Independent mechanics and available information

The dynamics were reconstructed in canonical momentum coordinates rather than the source's joint-acceleration solver. With `c=cos(q2)`, the inverse inertia is

\[
M^{-1}=\frac1{2-c^2}\begin{pmatrix}1&-(1+c)\\-(1+c)&3+2c\end{pmatrix}.
\]

Hamilton's equations give `qdot=M^-1 p`, `pdot1=u1`, and `pdot2=u2-sin(q2)(v1^2+v1*v2)`. An independently implemented vectorized fourth-order Runge–Kutta integrator at both 32 and 64 substeps checked all 10,240 acquisition pulses, 2,048 predictive pulses, 12,800 task pulses and 160 distinct privileged feasibility witnesses. The largest final-state difference between 32 and 64 steps was `1.441e-11`; between 64 steps and the saved DOP853 result it was `9.434e-13`. The 160 witness commands all satisfied their bounds and independently met their endpoint targets within `1e-7` in normalized displacement. Witness existence is a feasibility statement, not a learner.

Each pulse starts at rest and lasts 0.1 time units. It generally ends with nonzero velocity. The observations and responses concern joint angles, not Cartesian endpoint coordinates. Exact preparation angles for both joints, zero initial velocity, periodic topology and a relation affine in torque are supplied to ordinary methods. This is not noisy-context inference, continuous feedback or stationkeeping.

All 48 acquisition, predictive and task-measurement RNG role seeds were distinct. Every acquisition/predictive context, command and measurement was reconstructed from the specified RNG order. Measurement errors matched exactly. The same 512 fitting and 128 independent validation rows were supplied to all ordinary methods. Predictive and task endpoints were not used by fitting or selection. Source inspection confirms that saved evaluator-only states do not enter `fit_all`, and true inertia enters only the privileged oracle. Ordinary task decisions use the fitted relation, exact preparation and requested displacement. Their saved noisy endpoint readings do not affect their decisions or primary scores.

Exact supplied preparation `q` also makes the initial reading error `o0-q` available. The specified estimators deliberately use `(o1-o0)/0.005`; an estimator using `(o1-q)/0.005` could halve its response noise covariance from `0.0008 I` to `0.0004 I`. This is a limitation of information use, not a discrepancy in the shared-response comparison or an irreducible sensor-noise floor.

## Fits, validation and bounded decisions

All 832 candidates were checked independently using the equivalent cosine expression for periodic chord distances. Their layout, raw and noise-corrected Gram matrices, spectra, regularization scales, finite coefficients, validation losses and selected indices matched. Candidate selection used validation only. Both inverse methods use the same known-noise correction in their validation scores; the calibrated method additionally corrects its training Gram matrix.

The solver is a PSD-projected, ridge-regularized moment estimator. Its correct stationarity equation is

\[
\bigl(Q\max(\Lambda,0)Q^T+\alpha s I\bigr)W=X^T Y.
\]

The largest normalized stationarity residual was `3.797e-13`. The stored `training_normal_equation_residual` is instead the unregularized `||G W-X^T Y||`; it is not a solver-failure measure. Tiny negative eigenvalues of otherwise PSD raw Grams are consistent with floating-point roundoff. The audit checks signs with scale-aware tolerance and does not label all such negatives substantive indefiniteness. All floored counts are retained. Thirteen naive-inverse and 31 calibrated-inverse candidate validation scores were negative. Corrected validation is unbiased for a fixed candidate with independent validation noise; the selected minimum can be optimistic or negative.

All 10,240 ordinary task decisions used the same bounded displacement error objective. For inverse fits, this uses the prescribed pseudoinverse of the fitted inverse relation with `rcond=1e-8`. A separate exact two-dimensional box enumeration checked every unconstrained interior, edge and corner candidate. All saved solver-success flags were true. The maximum difference from the globally minimal convex quadratic objective was `1.963e-12`. All commands respected `[-1,1]^2`. Saved condition numbers and inverse singular values were checked; no ill-conditioned or saturated outcomes were removed.

## Results and uncertainty

The independent audit reproduced all primary task scores, predictive outputs/errors, saturation counts and summary numbers. Statistics average the 160 fixed tasks within each acquisition replicate and use the eight independent acquisition replicates for MCSE. The repeated oracle tasks do not create additional independent witnesses or imply zero uncertainty outside this deterministic instrument. Primary error is squared **normalized joint displacement** error; squared physical joint displacement error is `0.005^2` times this quantity.

| Training coverage | Constant forward | Context forward | Naive inverse | Calibrated inverse |
|---|---:|---:|---:|---:|
| Broad: mean task error | 0.00898141 | 0.0000702875 | 0.0000179176 | 0.00000941635 |
| Narrow: mean task error | 0.0196274 | 0.0455312 | 0.0231309 | 0.0254856 |

Under broad coverage the calibrated inverse is a serious successful same-data competitor to the tested forward parameterization. Under narrow coverage it is worse than the naive inverse on mean task error. These results cannot establish forward-model necessity, a universal inverse advantage, or the superiority of arbitrary models over direct policies. Matching feature grids does not make forward and inverse approximation geometries equal. The finite-duration physical response is also not exactly affine in torque.

The retained post hoc coverage diagnostic separates 60 task points with `|q2|<=0.4` from 100 outside. For narrow training, mean inside/outside errors are `2.01621e-5 / 0.0728379` for context forward and `2.28696e-5 / 0.0407632` for calibrated inverse. The constant relation gives `1.00391e-4 / 0.0313436`. Flexible fits work within sampled support and do not reliably determine unsupported responses. No cases or settings were removed to improve these summaries.

The post hoc `q1_command_RMS` in the audit JSON is defined **per replicate** as the root mean squared Euclidean command deviation from the four-q1 command mean at fixed q2 and goal, followed by mean/MCSE across eight replicates. Broad mean RMS is `0.0210009`, `0.00423893`, `0.00410251` for context forward, naive inverse and calibrated inverse. Narrow means are `0.0826621`, `0.0183252`, `0.0198261`. These measure unnecessary dependence on an irrelevant coordinate in this instrument. They are not pooled-across-seed roots or pairwise RMS values. The oracle deviation is about `2.82e-13`; the constant relation has exact zero deviation, despite its poor broad task accuracy.

## Budget and scope

The ordinary comparison contains 10,240 acquisition pulses, 2,048 predictive pulses and 12,800 task pulses. Each pulse has four scalar coordinate sensor readings (two coordinates at each endpoint): 49,152 acquisition-plus-predictive readings and 51,200 task readings. The 1,456 privileged continuous solver rollouts used to establish 160 feasible goals are separate simulator computation, not ordinary learner experience. The oracle's repeated task evaluations are included in the 12,800 task pulses. The corrected candidate solver adds no physical observations.

This is a verified compact test of context-dependent consequence relations and noisy inverse regression under explicit supplied priors. It establishes neither competitive robotics performance nor a general learning architecture. Coverage, representation and known calibration remain consequential assumptions, and complex fits are unsafe outside the demonstrated support. Those negative findings are part of the result.

This fully verified comparison was subsequently superseded because exact supplied preparation makes the noisy initial reading redundant. Its source/data/results remain preserved under the `coupled_learning_v14_two_reading_pilot*` names. This report describes the original two-reading responses only.
