# Acquired executable features: useful expansion, competitive proposal rejected

2026-10-21. Following the frozen 2026-10-20 plan, all four old-only and ten
continued fits finished during the interruption. Their completed states were
checked before scoring; no fit was restarted. Validation choices were frozen,
then all later bodies were scored once. **TESTED NEGATIVE, exploratory:** the
proposed support-to-query feature acquisition fails its competitive continuation
criterion. No grammar, physics, learning-rate, feature-count or search-budget
change will be selected from these test values. No v18 or breakthrough is claimed.
The broad scientific objective remains active and unachieved.

## Actual later-learning result

Mean standardized forecast MSE over eight fresh physical bodies per column group.
Each body's support is trajectory 0 prefix; all 672 intervals of trajectories
1–7 are queries. One trained seed and one acquired corpus are used.

| Method | New-law ID, 16 | New-law ID, 64 | New-law OOD, 16 | New-law OOD, 64 | Old-law retention, 16 | Old-law retention, 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Support-to-query grown program | 0.177233 | 0.118422 | 0.873052 | 0.317656 | 0.038553 | 0.038632 |
| Ordinary-fit grown program | 0.190874 | 0.094799 | 0.701064 | 0.351544 | 0.035610 | 0.025119 |
| Old program, continued prior | 0.329424 | 0.332147 | 1.568541 | 0.728036 | 0.052862 | 0.049040 |
| Neural ALPaCA64, full continuation | 0.214774 | 0.072591 | 0.792237 | 0.365560 | 0.027846 | 0.025151 |
| CoDA, full continuation | 0.126821 | 0.102658 | 1.922390 | 0.925092 | 0.013868 | 0.012931 |

The proposed selector loses the primary new-law comparison at both budgets:
CoDA is strongest at 16, neural ALPaCA64 at 64. Its old-law retention error also
exceeds both neural competitors. Thus all declared accuracy/retention guards fail;
correct support being useful does not override them. Proposed new-law ID error
falls from .693293 with zero support to .177233/.118422 at 16/64, versus
1.351449/1.350000 with wrong-body support. The method learns from evidence, but
that is insufficient for the desired advantage.

**TESTED, narrower positive:** both growing feature pipelines improve the recorded
new-law means over the old program with its prior still allowed to continue
learning. Feature expansion therefore helps within this supplied grammar and
screen; readable expressions or acquisition-data fit are not the evidence for
that conclusion. The entire feature-plus-prior change is compared, not an
isolated effect of one mathematical fragment.

The neural rankings are mixed. CoDA wins new-law ID at 16 and retention at both
budgets, but transfers poorly to the new-law OOD group. Neural ALPaCA64 wins
new-law ID at 64. Meta-grown features have the best OOD-64 mean among these five;
fit-grown features have the best OOD-16 mean. These local wins remain recorded.
They cannot replace the frozen primary criterion with an easier post hoc target.

Support-to-query selection earns no separate claim: about 7.1% better than fit
selection on new ID at 16 (below the declared 10%), worse at 64. The proposal
streams were separate fixed seeds; this single-run contrast would need paired
replication even if promising. Do not infer a general ordering of symbolic,
neural, modular or meta-learning mechanisms from these means.

## Learning operation and explicit assumptions

The candidate persistently acquired a DAG of executable feature fragments.
Initial atoms are constant 1 and eight old-train-standardized observed features;
operators are multiplication, sin, cos, abs and 1/(1+z*z). Twenty-three additions
produce the old 32-feature library. Each new growth rule adds 16 features, giving
48. The old library's computations and normalization remain fixed. Candidate
fragments are evaluated on ordinary training-body observations, not true physical
parameters, formulas or test labels. Two-operation lookahead permits useful
fragments with an intermediate that alone need not predict the target.

The proposed selector scores disjoint query error AFTER short-support ridge
inference; its matched comparator scores ordinary fit error on the same
support/query union. All searched episodes are training data. Exact block-ridge
updates make additions cheap; they are standard linear algebra, not a learning
principle. After acquisition all programs learn their Gaussian coefficient prior
and precision through ALPaCA predictive NLL. New-system observations then update
coefficients. Read structure_acquisition_derivation.md for the derivation and
supplied/acquired account, including why duplicate-feature regularization relief
is not discovery of a new computation.

Strong conventional competitors continue learning their entire shared neural
models with acquisition data and old experience replay. They are not frozen.
The fixed-old-program ablation also continues its prior; it is not an unlearned
baseline. Their loss/reproduction limits remain those of the existing ALPaCA and
CoDA cores; this is not a reproduction of a complete robotics benchmark or the
entire modern literature.

The new physical law uses cubic actuation and quadratic velocity drag. Those
align particularly well with multiplication/abs, a substantial supplied prior.
Exact q/v, periodic angle encoding, grouped bodies, grammar, search budget,
Bayesian coefficient form and replay split are also supplied. A finite velocity
difference is observed, not exact acceleration. Nothing establishes identification
of physical parameters, unknown task boundaries, noisy perception, decision
competence, variable body dimension or humanoid capability.

## Actual resources and preserved failure

- New ordinary/evaluator archives contain 36 bodies, 288 trajectories,
  27,648 physical intervals / 1,382.4 simulated seconds: eight acquisition,
  four validation, eight later new-law ID, eight new-law OOD and eight old-law ID.
  Hidden coefficients and states remain evaluator-only; all learner input
  is ordinary observations/actions and measured finite consequences.
- Original data acquisition cost 43,008 intervals / 2,150.4 simulated seconds.
  It is reused, not free. The new campaign does not reuse the earlier test bodies
  as new evidence. Across all three post-v17 screens, total unique acquisition
  is 82,944 intervals / 4,147.2 seconds, including the second screen's 16 bodies.
- Old graph search took .997 seconds; meta/fit growth 1.512/1.509 seconds.
  Counts: old graph 36 retained operation nodes beyond nine atoms; final meta/fit
  graphs 62/63 operation nodes. This is a finite supplied grammar/search, not
  open-ended discovery of arbitrary programs.
- Four old-only fits use 2,000 updates each, ten continued fits 8,000 each:
  88,000 new gradient updates. Both learning rates and every recorded checkpoint
  are preserved. The reused CoDA parent had its own two old-only 2,000-step fits;
  their costs remain in the preceding campaign history.
- Before sampling, the eight-substep RK4 preflight failed two of six new-law
  cases (maximum 4.95e-7 against 1e-7). Original source, RNG and failure remain.
  Prospective refinement to 16 substeps passes the SAME six spots at maximum
  1.25e-8, with no tolerance widening. No completed trajectory/test case was
  removed and no learner was rescued by this correction.
- Independent old-only pretraining ran concurrently with new sampling. No
  acquisition-dependent fitting preceded generation. The data report's initially
  overbroad timing wording was corrected with its original status/source hash
  retained; numerical data, law and RNG bytes were not changed.

Both-rate summed fit wall seconds (processes ran concurrently; these sums are
neither campaign elapsed time nor inference latency):

| Phase/method | Summed fit seconds | Learned parameter count |
| --- | ---: | ---: |
| old / programold | 22.337 | 1,088 |
| old / neural | 22.698 | 13,120 |
| continue / meta | 82.944 | 2,400 |
| continue / fit | 154.755 | 2,400 |
| continue / frozen | 85.583 | 1,088 |
| continue / neural | 182.953 | 13,120 |
| continue / coda | 68.454 | 44,114 |

Parameter counts exclude graph selection, operations and normalization constants;
counts alone cannot establish comparable capacity or efficiency. The owner PC
and existing project environment were used; no dependency/service was installed.

## Reproducibility and verification scope

Executable source: structure_acquisition_data.py, features.py, screen.py,
decision.py with the common prefix structure_acquisition_. Graphs, all 14 fit
reports/weights, every training loss and saved validation checkpoint, frozen
selection, raw forecasts, body-level results and decision remain. Fit/score paths
refuse overwrites; reproduce in fresh artifact paths or a copied checkout.

- Independent feature algebra checks match actual production ridge predictions
  against augmented SVD within 7.6e-15 and verify nested feature normalization.
- Driver fixtures check body mapping, support/query trajectory separation,
  old/new replay balance, Gaussian prior expansion, CoDA context extension and
  equal weighting of the eight-old/four-new validation groups.
- The independently implemented result checker was run unchanged by the root
  after the interruption: PASS. It checks 120 saved forecast arrays / 960 body
  scores, train-only scales, all 14 new fit reports, the earlier CoDA parent,
  recorded validation selection, graph reconstruction/prefix and provenance.
  Maximum standardized-MSE reduction difference is 2.69e-6; raw-MSE arithmetic
  agrees exactly. The failed comparisons are not close to that discrepancy.
- Validation forward passes, optimizer trajectories, full neural forecasts,
  exhaustive graph search and physical trajectories are NOT independently
  regenerated. Saved graph metadata and actual numerical checks are bounded
  evidence, not a global correctness certificate. The old graph's historical
  plan hash matches exactly the preserved prefix before the documented prefit
  integration amendment; no model/source mismatch was silently accepted.
- An independent scientific critic recommends rejecting the formulation while
  retaining its expansion benefit and mixed rankings. See the direction audit
  and check_structure_acquisition_results_scope.md for exact scope.

## Scientific decision

Retire this feature-acquisition formulation and close this bounded attempt.
No extra seed, control test, grammar change, precision ladder or new benchmark
is justified as a rescue. This does not prove that acquired executable knowledge
cannot improve future learning. It establishes that the tested construction and
selection rule have not earned a competitive claim despite a favorable grammar.

Three post-v17 constructions have now been screened against serious alternatives.
Each changed an actual learning operation; none demonstrated the desired overall
advantage. The evidence favors taking established learning mechanisms seriously.
It does not justify declaring the ultimate objective complete, lowering its
standard, or turning the next investigation into an architecture lottery.
