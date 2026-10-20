# Precalculation review of the exploratory action-risk diagnostic

2026-10-20. Reviewed `cartesian_stream_v17_action_risk.py` and its frozen posthoc
design before reading any computed action-risk values. **No blocking math,
dataflow, scaling or axis defect found.** No source edited or diagnostic executed
by this reviewer. The already frozen control pilot remains unchanged.

The score correctly uses the independently derived factor1/2 on the auxiliary
pairwise squared difference. Actual commands and eight independent auxiliary
commands are scaled consistently by.5. Predictions/targets/baselines use physical
meter CORRECTION units. A zero correction baseline means the raw linear-carry
baseline, not a zero physical position. Subtracting the common past-measurable
carry from model/target leaves action contrasts and the risk identity unchanged.

Prediction axes are candidate8,member3,row960,command10,coordinate4. Command0 is
the actual normalized action;1..8 are the common auxiliaries;9 is zero command.
The common drift baseline averages exactly the four candidates' twelve nonlinear
zero-command predictions and uses no validation target. Member and ensemble-mean
scores use the correct distinct order of averaging/quadratic evaluation. The
ordinary one-block MSE is evaluated on exactly the same960rows, so its comparison
does not silently confound the original six-block validation horizon with centering.

Each local member freezes its fitted coefficients at the actual query across all
ten commands; no validation append occurs. Actual validation targets enter scores
only. Independent auxiliary streams use the specified new namespace and are shared
across all candidates/members. Eight trajectory means, descriptive cluster SE and
paired row-minus-column score matrices have the correct axes and denominators.

The combined ordinary NPZ is loaded then test rows are masked. The precise
information claim is NO TEST ROW CONTRIBUTES to any calculation; literal zero
test bytes read is not possible with this combined-array loading implementation.
No evaluator, heldout-fork or control artifact is opened. Weights/learning/local
source hashes are checked against frozen fits. No candidate selection or deployed
model update is performed.

The method was proposed after the negative fork result, so this remains explicitly
POSTHOC even though its exact calculation was frozen before seeing its values.
Validation trajectories already helped choose the original candidate. Neither an
unbiased fixed-model score identity nor predeclaring two drift baselines makes
later model selection on these same values independent confirmation. Candidate
reselection, retuning or a new method claim needs a fresh/frozen protocol. Scores
can have substantial finite-sample variance; a drift baseline can help or harm it.
The score concerns conditional mean action dependence, not the full response law,
state sufficiency, actual control value or universal information availability.
