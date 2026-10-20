# Independent review of the v17 ledger draft and release builder

2026-10-20. Read-only review of `ledger_v17_prefix.md` and `build_ledger_v17.py`, checked against the stored pilot, acquisition, fork, retention, fresh physics and roundoff artifacts, current independent audit sources, and literature notes. No builder, learner, controller, diagnostic or physics was executed; no canonical/frozen history was edited. Intentionally pending aggregate/certificate/optimization placeholders are not counted as errors.

## Findings and corrections

Two F61 qualifications were raised and are now corrected in the draft: the general conditional projection uses nonsingular Cov(U|H), and the post hoc scores explicitly acknowledge reuse of validation trajectories that already informed original candidate selection. The pairwise centered-risk identity has the correct factor 1/2 under the stated independent conditional randomization. It does not validate individual-world contrasts or latent-state inference, and the draft properly limits its interpretation.

One builder hash-binding issue was raised and is now corrected on reinspection. The optimizer diagnostic uses input_and_final_audit_sha256, rather than the generic artifact_sha256 map. The builder now verifies that map plus its source/design/NPZ hashes, all eight replicate IDs, scope of 11,520 decisions, and empty reconstruction_failures. This closes the stale-diagnostic admission path; PASS status alone would not have bound those inputs.

One wording improvement remains advisable: the opening status calls this a “failed generalization claim.” Archived pilot/protocol text already qualified the pilot as exploratory and did not establish robust population sufficiency. “Failed extrapolation from the exploratory pilot” or “negative fresh replication” would distinguish the failed expectation from an overstated prior published conclusion. This is an attribution clarification, not a change to the numerical finding.

## Checks without consequential contradictions

The moving acceleration formula agrees with the independent polar/joint derivations, including B-minus-A sign, tangential positivity, common-command qualification and interior-domain exclusions. The local observation-map determinant statement does not incorrectly claim global observability. Finite noisy known-pair errors, fixed-pattern information scaling and the positive common-mass scaling counterexample remain properly scoped.

Counts align with artifacts: pilot acquisition uses 10,240 intervals and 20,560 scalar noise draws; heldout forks use 320 intervals; combined pilot ordinary/reference episodes use 4,704 intervals, with 1,440 ordinary decisions and 2,160 local member fits. The ablation uses 720 decisions, 2,160 fits and 1,568 intervals. Fresh scope is 119,552 intervals, 512 full rolling trajectories, 1,416 Radau checks, 11,520 ordinary decisions and 34,560 control member fits. Acquisition/control/reference simulated seconds sum to 11,955.2. The 55/192 and 19/192 ordinary successes imply 310 failed windows; all remain. Preparation/noise/corpus/body counts and the distinction between copied rows and independent experiences are sound.

The pilot-to-fresh correction and retained-versus-frozen comparison are fair as comparisons of complete closed-loop mechanisms. The draft does not credit all behavior to online retention, separate feedback benefit, physical-model discovery, or neural representation superiority. It preserves the weak action contrasts, neural optimization qualification, original roundoff assertion, high-precision diagnosis and unchanged downstream criteria. Pending secondary completion must remain an actual gate rather than be inferred from the diagnosis alone.

The conventional-learning attribution is appropriate. The CMU-hosted manuscript dates itself June 13, 1996, and the university publication record lists 1997. Its stored-experience/query-local forward-control content supports the limited precedent asserted here: [manuscript](https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1997_2/moore_andrew_1997_2.pdf), [CMU record](https://publications.ri.cmu.edu/locally-weighted-learning-for-control). No reproduction or superiority over the complete method is claimed.

The next-direction question has a defensible connection to general learning: when population experience earns reuse over learning a useful relationship from the current encounter. The prefix estimator is a serious comparator with disclosed extra regularity assumptions, not a prescribed architecture. The draft properly keeps support, history, approximation and uncertainty explanations open rather than declaring an information-theoretic failure.

## Release plan

The builder reads v16 as bytes, appends it verbatim, rejects unresolved placeholders, refuses replacing an existing ledger/manifest, binds release inputs and offers exact reconstruction under --check. The revised positive audit/aggregate/optimizer gates are adequate for the stated finite scopes, conditional on their actual completion and current hashes. This review is not itself a substitute for running those gates. No additional scientific blocker was found in the reviewed claims once the remaining real audit/certificate results are completed and accurately inserted.
