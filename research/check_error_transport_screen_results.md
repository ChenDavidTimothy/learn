# Independent decisive-result check: reject current joint transport candidate

2026-10-20. **TESTED / REJECTED in this exploratory screen.** The saved evidence
supports retirement of the current residual-conditioned joint operator, without
another controller rollout, refit or precision campaign.

Run `.venv/bin/python research/check_error_transport_screen_results.py`.
Actual results are `check_error_transport_screen_results.json`. Independent code
opens ordinary transition targets/splits, saved predictions and ten fit reports/
checkpoints. It imports no candidate/baseline/physics implementation, performs no
network forward pass or retraining, and does not re-integrate trajectories.

Training-only normalization independently reconstructs exactly from32 training
bodies. All80 saved prediction arrays, comprising640 per-body/support-budget
comparisons, reproduce standardized and raw MSEs. It additionally scores against
the original binary64 measured targets; target float32 quantization changes raw
body MSE by at most5.77e-8. Maximum discrepancy versus recorded GPU standardized
reduction is1.28e-6 and raw reduction1.63e-7, within declared float32 reduction
precision and immaterial to the decision. No tolerance was changed after a failure.

All ten complete2000-step fit logs have finite losses and four validation
checkpoints. The checker reconstructs each adaptation-rate/checkpoint argmin,
then training-rate argmin from validation body scores only. Saved checkpoint
choices/scales, weights/parameter counts and current source/data/report hashes
match. It does not independently recompute validation network predictions or
optimizer histories; it verifies the stored validation-selection arithmetic and
reviews the driver's split/dataflow. Test outcomes are scored after selection.

| OOD support | joint candidate MSE | strongest comparator (CoDA core) | relative candidate improvement |
|---|---:|---:|---:|
| 16 | .750616358 | .359610775 | −108.73% |
| 64 | .640076822 | .306447769 | −108.87% |

The prespecified necessary10% improvement condition fails at BOTH budgets.
This conclusion does not depend on interpreting the opposite-budget-loss clause:
there is no winning budget. ALPaCA and the separable ablation also outperform
the joint operator at both budgets. Correct same-body support is useful relative
to rolled wrong-body support (wrong joint MSE3.51205/3.48956), but that establishes
use of relevant evidence, not an advantage of the proposed update.

The selected settings are joint/separable lr.001 atstep2000, ALPaCA lr.003 at2000,
CoDA lr.003 at2000 with context-adaptation lr.01, pooled lr.003 at1000 with
fine-tuning lr.003. Candidate/baseline capacity differs: joint14,169,
separable15,192, ALPaCA7,904, CoDA44,050 and pooled4,866 parameters. Training and
adaptation compute/tuning costs differ and remain recorded; no universal or
inferential ranking follows. The baselines are scoped implementations: ALPaCA
retains fixed-noise predictive NLL, CoDA retains whole-network low-rank context
adaptation with a disclosed measured-transition loss replacing original
integrated trajectories.

This is one trained seed on eight heldout ID/eight OOD bodies in one nonlinear
family, exact observed q/v, noiseless measured finite velocity increments. It
provides no control capability or framework claim and does not establish that
all possible functional-update mechanisms fail. It decisively fails the planned
continuation gate for THIS candidate. Accept the stronger conventional explanation
and change the mechanism decision; do not rescue the candidate through test tuning.
