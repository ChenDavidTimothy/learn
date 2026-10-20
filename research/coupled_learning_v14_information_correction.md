# Available-information correction before v14 publication

Independent audit found that supplying exact preparation q0 makes the initial
sensor noise o0-q0 observable. The original all-method response (o1-o0)/F therefore
wastes available information. It has the stated covariance .0008I, so the original
statistics/comparison are internally valid, but that noise is not irreducible.

Preserve FULL original source/raw data/results/design/timing and independent audit
under coupled_learning_v14_two_reading_pilot* and corresponding audit names. The
archived source is byte-identical; for reproduction pass --output-prefix explicitly.
Its original generated metadata refers to filenames in use when generated; archived
reports distinguish that provenance from the corrected current files.

Correct ALL four methods to d=(o1-q0)/F, with independent output noise .0004I.
Reuse the EXACT raw acquisition/validation/predictive records from the original
NPZ; no new acquisition or predictive experiments. Width/ridge grids, features,
conditions/seeds, validation selection logic, goals/bounds and all other settings
remain. Only the response statistic and its known covariance change. Reuse all160
original privileged endpoint-feasibility witnesses, with no new oracle search.
Corrected actions require12800 additional plant task pulses, counted as evaluation.

This repair is exploratory after original outcomes and an information audit, not a
new blind task benchmark. No hyperparameter is selected from those task outcomes.
Both full comparisons remain. It is an information-use correction, not another
calibration-optimizer ladder or a claim that a model direction is universally best.
Corrected noise-aware inverse moment/validation identities use the new covariance;
finite regularization/function/coverage limitations remain. Audit raw-data identity
and both budgets before publication. Then leave rest calibration for continuous
coupled motion/missing-history revision as prescribed.
