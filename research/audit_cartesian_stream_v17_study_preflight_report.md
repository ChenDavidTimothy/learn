# Independent pre-fresh-study audit

2026-10-20. **PASS** for the final frozen source against the fixed eight-corpus
design. No blocking protocol/dataflow defect found. No actual fresh acquisition,
fresh study random values, model fit, reference or control was sampled by this
audit. Canonical source/design were not edited.

Exact final source SHA256:
`a213d5eb27ab1736fe0a855787f8d61293a5a15f52c7d773ba4d3634454a1754`.
Design-file SHA256:
`68ff9394753ed08eccc34162f70cb47b5dbad592d10d0e20db36cdac9ddd3321`.
The design's Git publication commit is a different identifier from its file hash.
Reproduce with `.venv/bin/python research/audit_cartesian_stream_v17_study_preflight.py`.
Source/dependency hashes and actual checks are in its `_results.json`.

## Source and scope review

The fixed scope is replicates200..207, with unchanged650 acquisition law/provenance,
660 replicate-specific complete-trajectory bootstrap draws and670 replicate-specific
preparation/warmup/sensor paths. Acquisition metadata must match the requested
replicate, source/design/body hashes, shapes/counts and ordinary/evaluator hashes.
Evaluator bytes are hashed for provenance; evaluator arrays are never opened by
fitting. Ordinary feature construction, scales, targets and local regression are
the previously independently audited common implementation.

Only the four frozen local candidates are fit/validated, each with the same three
whole-trajectory bootstrap libraries. Scales use training rows only. The selected
index is written after validation and before any ordinary test forecast. There
are no neural fits, test-selected settings or control-selected repairs.
`ValidationProgress` calls the underlying predictor on its available prefix and
proposed commands BEFORE consulting validation targets for recording; it returns
that same prediction unchanged. Selected test forecasts preserve raw predictions
and observed targets, independently of selection.

All eight privileged references complete before the sixteen ordinary episodes.
Each case/variant loads a new library from the same selected acquired configuration,
scales and bootstrap rows. Retained/frozen library initialization is identical;
the audited parent episode runner/chooser/loss remains unchanged. Current history,
body and previous plan persist across goals. The frozen wrapper's distinct tag,
coefficient/query captures,90 no-op notifications and constant row counts remain
correct; retained counts grow by90. Reference state/actions/outcomes never enter
ordinary fitting/decisions. Failures/checkpoints preserve partial actual streams,
current fit metadata and correct active-origin provenance; previous active model/
prediction entries are cleared before advancing.

Per-replicate and eight-replicate resource arithmetic matches the design, including
ordinary acquisition, paired noise opportunities versus distinct draws,720 real
retained row events per corpus versus2,160 member copies, privileged references,
and separate planner computation. The maximum four concurrent single-threaded
invocations is a launcher obligation; the wrapper sets Torch and BLAS limits
inside its fitting/control execution. This is still one fixed body and a narrow
local-retention question, not a repeated neural benchmark or universal learning
claim.

## Synthetic-only numerical/dataflow fixtures

Temporary synthetic ordinary corpora have exactly the declared shapes/splits.
Their evaluator file is deliberately NOT an NPZ. Provenance verification succeeds
by hashing it; accidental evaluator-array parsing would fail. A mismatched
replicate is rejected. A deterministic synthetic causal predictor exercises all
four candidate validations and selected-only test forecasts. Independent NumPy
equations reproduce the recorded validation scores exactly and every saved test
prediction/target within roundoff. There are80 validation origins/candidate and
960/80 selected one-/six-block test origins. Train-only scales agree within
8.68e-19 with independent feature/target equations.

A fake deterministic RNG checks the660 bootstrap and all200..207 670 replicate
slots without drawing a single actual fresh study random value. It is a seed-
plumbing fixture, not independent reproduction of future real RNG outputs.
An injected failure at the third synthetic validation origin preserves that
active origin, two completed records and failure metadata, without stale previous
prediction. Physical flow is replaced with a raising guard. Temporary fixture
artifacts are removed; no real fresh acquisition/result is manufactured.

Inherited model/controller mathematics were independently audited on actual pilot
artifacts; these tests check the new wrapper, not population performance. The
actual eight fresh corpora still require independent seed/data/scale/bootstrap/
fit/selection/forecast and full ordinary decision replay, plus separate physical
RNG/stream/reference/score verification. Report all failures, corpus-clustered
paired differences and optimizer residuals. Appending changes subsequent visited
experience as well as the library; any total closed-loop effect does not isolate
per-row value or establish universal online-learning necessity.
