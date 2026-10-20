# Independent pre-fit audit: causal moving-history pipeline

2026-10-20. PASS for the inspected frozen learning/common implementation and
heldout fork source; no blocking scientific indexing/dataflow defect found.
This is a pre-fit source/unit audit, not independent reproduction of trained
artifacts, command forks or control performance. No canonical file was edited.
Exact inspected learning/local/design hashes and actual numerical checks are in
`audit_cartesian_stream_v17_prefit_results.json`.

## Information, selection and rollout review

The 66-context ordering is current camera position, 16 backward causal slopes,
then the chronological 16 issued commands. Candidate torques are additional
inputs, divided by .5. A row at even t uses observations through t and commands
strictly before t; only its target uses observed t+1/t+2. The disclosed linear
extrapolator is computed from t and t-1. There are 120 rows per trajectory and
2,880 training rows; no window crosses a trajectory boundary.

Scales use only the training split, with the prescribed numerical floors.
Bootstrap draws select complete training trajectory IDs with replacement and
expand them to complete corresponding row blocks. Those same row arrays feed
both families. Network initialization has separate candidate/member seed roles.
The main fit code opens only the ordinary NPZ and imports no plant/evaluator.
Candidate selection uses only the 80 six-block validation origins. Test scoring
occurs after both selected indices are fixed and never changes them.

Every member rolls forward its own predicted observations. It repeats each
candidate torque for two intervals, appends both predicted positions, shifts the
causal history and preserves the same member across all six blocks. It does not
substitute future heldout camera readings or average members before advancing
history. Local coefficients and the query center are prepared once at the real
origin and remain frozen within that hypothetical plan. The score intentionally
averages squared errors over members; it is not merely the squared error of the
ensemble mean. Its ten origins per validation trajectory are precisely
16,40,64,88,112,136,160,184,208,232.

`append_completed` reconstructs the row whose two future intervals have just
completed and appends the same real row to all local libraries. The future
control caller must call it exactly once after each completed held pair, with
aligned real observations/commands, and must preserve libraries across goals.
Those caller obligations cannot be verified before that source exists.

The local implementation matches the protocol's three-block standardized
distance, command-independent neighbor selection, stable duplicate handling,
Gaussian weights, full affine context/action regression and coefficient ridge
with an unpenalized intercept. Its own independent augmented-least-squares
fixture is preserved separately by its author. This review also verifies the
integration's scaling, fixed query and frozen rollout use.

The external fork source keeps truth separate: predictors receive only the
ordinary prefix/proposed command; plant states generate evaluator-only targets.
It uses the frozen 32 origins and five-command bank, without local appends or
selection. Its axes correctly average the three members and subtract the zero
command's two future camera means. The counts are 160 counterfactual
preparations, 320 .1-second flows, 32 privileged seconds and zero sensor draws.
Actual fork numerics remain unverified until artifacts exist.

## Independent numerical checks and correction

`audit_cartesian_stream_v17_prefit.py` imports the audited units only as tested
adapters. Separate NumPy equations reconstruct features, targets and autonomous
three-member/three-block rollouts. Synthetic future corruption leaves the
forecast-origin feature/candidate input unchanged. Feature/tensor-context error
is zero; target correction error is 2.23e-16; completed-row append error is zero.
Both local and nonlinear path arrays match the independent reference exactly on
these fixtures. Autograd of a scalar path objective agrees with central
differences of the independent NumPy rollout within 3.17e-10 (nonlinear) and
2.62e-10 (local). These are small implementation fixtures, not global numerical
accuracy or optimizer-convergence proofs.

An initial source concern was that a failed/nonfinite fit could abort without
serializing its history. It was reported before fitting. The owner added a
top-level failure checkpoint before execution: active candidate/member/losses,
previous records, previous weights and current network are preserved and the
error re-raised; nonfinite forecast checks precede candidate selection. The
numerical JSON hashes this corrected source. No model, candidate menu, selection
score or mathematical formula changed for this correction.

No prediction/calibration claim follows from these checks. Camera-input and
residual-target noise are correlated, a deterministic bootstrap ensemble does
not resolve intrinsic hidden-context mixtures, and finite H=16 is not proven
sufficient. Actual artifact audit must reconstruct saved weights/scores/shared
bootstrap membership and heldout interventions. A precise common control
optimizer, nominal witness and failure-accounting contract are still pending
and must be frozen before control outcomes, as the protocol already states.
