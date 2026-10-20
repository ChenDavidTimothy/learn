# Independent audit of actual moving-history fit artifacts

2026-10-20. **PASS** for the actual saved ordinary-data fits, candidate selection
and all reported ordinary validation/test forecast scores. This is stronger than
the preceding synthetic pre-fit fixture, and does not certify control or the
external fork physics. No canonical source/result was edited.

Reproduce with `.venv/bin/python research/audit_cartesian_stream_v17_fit.py`.
Executable independent equations and actual numerical results are preserved in
that source and `audit_cartesian_stream_v17_fit_results.json`. The JSON records
exact learning/local/design/ordinary-data/fit/weights hashes. The audited learning
source is `56dc34b38a269d482d843a4207259f316c7329064aa49660dddb37592fcce5f0`.

## Independent implementation and scope

The audit imports neither canonical learning/local code nor a plant/evaluator.
It reconstructs the causal 66-component contexts, residual targets, train-only
normalization and complete-trajectory row identities directly from ordinary
positions/commands. It implements NumPy tanh-network evaluation from all saved
weights and coherent member-specific autonomous history updates. For local
regression it independently selects neighbors and solves the weighted residual
augmentation by `numpy.linalg.lstsq`, instead of the canonical regularized normal
equations. It keeps the origin's local fit/query fixed throughout each forecast.

All source/data/weights hashes match the saved fit metadata. Context scales match
exactly and target scales within 3.47e-18. All 2,880 training origins match
exactly. All three bootstrap trajectory
draws, full row blocks and SeedSequence paths reconstruct exactly. Both families
use these same bootstrapped corpora and normalization. All twelve networks' seed
paths, declared widths/weight decays and stored finite parameter arrays check;
all 72 saved parameter arrays are accounted for.

Independent Torch layer construction from the declared seed reproduces each
initial normalized training loss within 3.34e-16. All 300 logged pre-update losses
per member are finite, and the audit separately calculates the training loss
of every final saved network from NumPy predictions. The final saved-weight loss
is after the last update, whereas the last logged training loss is before it;
these are deliberately distinct. The audit does not independently reconstruct
all intermediate Adam states/updates. The twelve finite histories are retained
evidence, not a global optimization certificate.

## All actual forecast scores and selection

Every four-candidate validation score was reconstructed at all 80 actual origins.
Every per-origin member MSE, ensemble-mean MSE and future-sample MSE agrees. For
each selected family, the audit additionally reconstructs all 80 six-block and
960 one-block test origins, without modifying selection.

| Family/candidate | Reconstructed validation member coordinate MSE |
|---|---:|
| nonlinear 0: width64, decay.0001 | .014294514728990218 |
| nonlinear 1: width64, decay.01 | .008322576513327162 |
| nonlinear 2: width128, decay.0001 | .012729773528530709 |
| nonlinear 3: width128, decay.01 | .007323092407556192 |
| local 0: k128, ridge.001 | .009902840179968585 |
| local 1: k128, ridge.1 | .010143567937433248 |
| local 2: k512, ridge.001 | .007021591774915012 |
| local 3: k512, ridge.1 | .008081215372028785 |

Thus validation selects nonlinear index3 and local index2 exactly as saved.
Selected test member coordinate MSE is .006765875288015641/.006814440909052856
for nonlinear/local six-block forecasts, and .000047607496914668864/
.00003469579226031475 for their one-block forecasts. These are forecast metrics,
not intervention success, control results or a statistical ranking of methods.

Maximum nonlinear per-record numerical discrepancy is 1.17e-15. Maximum local
per-record discrepancy is 1.96e-13 across 1,280 distinct origin/candidate local
preparations (3,840 bootstrap-member augmented least-squares fits). The largest
local regularized stationarity residual is 1.52e-13. Cached fits are reused only
when the same candidate and actual origin recur, preserving frozen-fit semantics.

## Actual selected derivative check and remaining limits

For a real validation prefix (trajectory24,t40), independently derived recursive
NumPy path Jacobians were checked against central differences of the complete
six-block path, for all twelve planned command coordinates and all three
members. Largest absolute error at epsilon1e-5 is 4.09e-10 (nonlinear) and
3.80e-10 (local); at epsilon1e-6 both remain below 4.18e-9. Maximum actual path
derivatives are .130773 and .074355, respectively. These checks concern learned
forecast derivatives with the local coefficients frozen; no optimizer or actual
physical action sensitivity is thereby certified.

The previous source audit establishes the causal/selection dataflow; actual
artifact reconstruction confirms it produced the saved records. No true joint
state, velocity, branch, physical coefficient or evaluator target is needed by
this audit's fit reconstruction. Its conclusions remain conditional on the
declared fixed body/camera/timing and mean-path approximation. Finite history and
bootstrap disagreement are not proven sufficient context uncertainty. Fork
consequences, closed-loop support, physical goal feasibility, planner convergence,
online append timing and sustained control need their separate actual audits.
