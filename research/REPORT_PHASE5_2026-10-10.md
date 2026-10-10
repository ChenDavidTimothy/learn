# Phase V: decision-identifying experiments, unknown sensor mechanisms, and real randomized policy value

**Date:** 2026-10-10. **Directory:** `/home/dtdav/dev/learn/research`.
**Status:** Eight further investigation cycles (32 through 39) completed. All code and aggregate results are reproducible. The controlled negative results materially constrain the original ambition. **No new general learning algorithm, general superiority, or breakthrough established.**

**Governing standard:** `STANDING_RESEARCH_STANDARD.md`. Read with `REPORT_PHASE4_2026-10-10.md` and `REPORT_PHASE3_2026-10-10.md`.

## Scientific questions and competing hypotheses

The prior phases showed that reward-relevant representations depend on structure, that stronger Bayesian controls often erase surprising results, and that task success need not identify a transferable model. Phase IV's simple Autoencode program grammar is **closed** to further optimization; continued wins there would be researcher-induced.

Phase V tests four substantially different, falsifiable questions:

1. **Information complementarity:** Does selecting experiments by *immediate* decision improvement fail when evidence is jointly but not separately useful? Are full-model entropy and optimal-action entropy reliable substitutes? Strong control: exact finite-horizon Bayes optimal experimental design, not a weak straw man.
2. **Misspecified observation models:** Is optimal data acquisition under an assumed model safe when the environment changes its sensors unannounced? Include no-probe abstention.
3. **Identifiability versus information availability:** If distinct latent models have exactly the same distribution for every available observation probe but imply different decisions, is autonomous revision even possible without new *intervention rights*? Compare reward observations and known-reference calibrations; corrupt the calibration as an adversarial control.
4. **Real intervention data:** Do flexible prediction/representation families produce *decision-relevant policy value* beyond competent constant policies on an independent human randomized experiment? Include finite-sample uncertainty, cross-fitted/heldout nuisance models, treatment budgets, covariate heuristics, and evaluation-estimator validation.

This phase mixes controlled finite-model mathematical tests with an externally collected, public 2008 randomized field experiment. **The finite-model programs have privileged likelihood and reward tables. They do NOT test invention of unknown model classes.** The real RCT has randomized treatment labels but no individual counterfactual ground truth.

## Specific predecessor audit (not exhaustive)

- Baisero & Amato (2021), *Reconciling Rewards with Predictive State Representations*, https://www.ijcai.org/proceedings/2021/299 : reward-oriented state predictions and control are already an important established direction.
- Toro Icarte et al. (2023), *Learning reward machines: A study in partially observable reinforcement learning*, https://doi.org/10.1016/j.artint.2023.103989 : reward machines can be constructed from experience to improve partially observable RL; a closest predecessor for any claim of automatically building decision-relevant memory.
- Zhang, Wu & Lin (2017), *Counterexample-guided Abstraction Refinement for POMDPs*, https://arxiv.org/abs/1701.06209 : existing formal POMDP abstraction refinement and model checking.
- Xie et al. (2024), *Learning to Explore in POMDPs with Informational Rewards*, https://proceedings.mlr.press/v235/xie24a.html : task-relevant information gathering in a POMDP is not novel in itself.
- Ehrlichman et al. (2026), *Causal Discovery for Efficient Offline RL with Factored Action Spaces*, https://proceedings.mlr.press/v323/ehrlichman26a.html : unknown causal structure can be discovered and used for efficient unbiased policy optimization, already with theory and real application classes. Its assumptions and action-factorization target differ from ours, but it substantially raises the novelty bar.
- *Detecting changes and anomalies in nonstationary contextual bandits* (2025), https://doi.org/10.1016/j.ins.2025.122270 : existing nonstationary contextual bandit anomaly/shift detection, against which future unknown-onset claims need comparisons.
- Bingham & Arussy (2026 preprint), *Cost-Aware Active Feature Acquisition for Differential Diagnosis under Realistic Clinical Availability Constraints*, https://doi.org/10.64898/2026.08.30.26361745 : even cost-aware diagnosis under missing/unavailable test panels is already empirically studied. **Preprint evidence must not be treated as peer-reviewed final truth**.
- Hillstrom's MineThatData Email Analytics Challenge (2008) is a public randomized three-arm field experiment; originally authored by Kevin Hillstrom. We used a public third-party mirror only because the origin's HTTPS certificate check failed. Dataset source details and SHA256 are stored in Cycle 37 results. Upstream data are **not committed**.
- UpliftBench dataset documentation https://github.com/binshuangli/uplift-bench/blob/main/docs/DATASETS.md highlights Hillstrom's lack of individual ground-truth treatment effects and upstream redistribution/license caveats. Comparisons use observable randomized policy value, NOT PEHE or invented individual effects.

These references document **actual close mechanisms**, not proof that every possible future joint framework is equivalent. A definitive novelty claim would require comprehensive comparison and a distinct demonstrated consequence.

## Cycle 32: a decisive counterexample to myopic decision-directed exploration

Files: `cycle32_decision_design.py`, `cycle32_decision_design_results.json`, `cycle32_xor_validation.py` and JSON.

- Unknown model is one of **64 explicitly enumerated** hypotheses. The learner is supplied **all 64 candidate reward tables and all eight experiment likelihood columns**, uniform prior, and the true family. The simulator hides only the selected hypothesis.
- Two adaptive binary observation queries maximum. Each costs 0.012 in expected reward units. Utility is **final action regret plus query costs**; all methods use the same model and budget.
- Methods: exact 2-stage Bayesian dynamic programming with stop option (DP; strong normative oracle *given supplied model class*); greedy immediate expected value of sample information; a two-step non-optimal lookahead heuristic; information gain about the complete model; information gain about the optimal action; and random experiments.
- Four independent scenarios, 1,000 new seeds each: many irrelevant nuisance bits, XOR-complementary decision bits, generic randomized reward/observation matrix, and completely uninformative probes with respect to decisions.
- The mathematical XOR check verified for 100 further independently generated instances that each individual probe has immediate Bayes decision value zero to floating-point precision, while a two-probe contingent plan has positive **net** value.

**Mean final regret plus query expense (lower is better):**

| Family | Exact Bayes DP | Greedy immediate value | Full-model MI | Optimal-action MI | Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nuisance-heavy | 0.06830 | 0.06830 | 0.16146 | 0.08143 | 0.17458 |
| XOR-complementary | 0.10666 | 0.20761 | 0.22589 | 0.17782 | 0.22221 |
| Generic random tables | 0.08607 | 0.08642 | 0.10757 | 0.10380 | 0.10644 |
| No task-relevant probe information | 0.18955 | 0.18955 | 0.21708 | 0.21320 | 0.22281 |

Nuisance-heavy DP minus full-model MI paired net-cost mean -0.09316, SE 0.00695. XOR DP minus greedy mean -0.10094, SE 0.00845. The model-likelihood assumptions deliberately make exact DP unusually strong; the point is to **falsify** a simplistic novelty narrative about myopic decision-value querying, not to propose DP as a discovery.

**Mechanism:** For a uniform prior on two independent hidden bits and XOR outcome, observing either bit separately leaves the optimal action equiprobable. Observing both is informative. Greedy one-step value of information is zero and stops; proper multi-step planning initiates the complementary experiment sequence. Full-model information gain can be wasted on nuisance factors. This is classical Bayes decision theory and parity information, NOT a newly discovered theorem.

Negative conclusion: neither pure predictive information gain nor one-step decision value is a general substitute for sequential task-aware information planning.

## Cycle 33: exact model-optimal planning may be unsafe after unannounced sensor changes

Files: `cycle33_model_misspecification.py`, `cycle33_misspecification_results.json`.

1,600 independent seeds per regime. The same original nuisance-bit world and reward tables from Cycle 32, but the **real observation likelihood** undergoes a hidden shift. All candidates and algorithms continue to plan and perform Bayes updates with the old likelihood; no oracle model revision. Strong comparator: **no observations**, which costs zero but incurs partial-information policy regret.

| Sensor regime | Old-model Bayes DP: regret + costs | No-probe regret |
| --- | ---: | ---: |
| Correct calibration | 0.07144 | 0.19532 |
| Some channels corrupted | 0.17145 | 0.19873 |
| Half the channels inverted | 0.21622 | 0.19097 |
| All channels inverted | **0.35031** | **0.18534** |
| Signals made uninformative | 0.20751 | 0.20095 |

Under global inversion, DP-minus-no-probe paired difference **+0.16497**, SE **0.00604**: taking what the old model believes is an informative measurement is decisively worse than abstaining. A method can be Bayes-optimal *inside its assumed class* yet fail catastrophically when that class is wrong.

This does not establish that all Bayesian methods fail. Model-uncertainty mixtures, change detection, reference calibration, or online reward feedback may avoid the failure; these stronger explanations motivated Cycles 34-36.

## Cycles 34-36: identifiability, calibrated intervention rights, and a finite certificate

### Cycle 34: reward anchors are needed when sensor orientation is unidentifiable

Files: `cycle34_reward_anchor.py`, `cycle34_reward_anchor_results.json`.

Model: six uniform independent hidden binary bits; six noisy sensors with **unknown, shared global polarity**, one reward-optimal bit. A known family of 64 hidden states is doubled to 128 candidate (state,polarity) combinations for an uncertainty-aware Bayes learner. All prior, observation and action-reward likelihoods are **supplied**. Binary experimental feedback includes normal sensors and, for allowed methods, actions whose stochastic reward can be observed. The true hidden state/polarity is used in the simulator and final regret only.

With polarity 0 normal (800 new seeds), exact Bayes net regret was 0.05979 if the polarity was provided; 0.19193 if unknown and only normal sensors could be probed; 0.11581 with a costed reward-feedback probe. With polarity 1 inverted (800 new seeds), the oracle-polarity net was 0.05707, **incorrectly assumed normal** was 0.35717, probe-only uncertain model was 0.20250, and uncertain with reward feedback was 0.12683. No-probe risk was 0.19132 and 0.20076, respectively.

The fully uncertainty-aware planner chose **a reward-feedback probe**, not normal sensors. With the specified uniform symmetry and only two queries, adding normal sensor probes to its menu supplied no extra useful decision information beyond reward queries, because sensor orientation remained globally aliased.

### Cycle 35: a reference intervention breaks the ambiguity, but may itself fail

Files: `cycle35_reference_intervention.py`, `cycle35_reference_results.json`.

Add a new reference query: produce a reading from a sensor whose latent reference bit is independently known to be zero. This gives a response probability 0.08 under normal sensor polarity and 0.92 under inverted polarity. It is a **new experimental right and strong supplied prior**, not a spontaneously discovered operation. The learner has the same two-query cap and 0.012 per-query cost.

1,200 further seeds in each of four conditions:

| True polarity / reference validity | Known-polarity oracle | Unknown-polarity reference planner | Reward-feedback-only | No probe |
| --- | ---: | ---: | ---: | ---: |
| Normal / valid | 0.05269 | **0.08854** | 0.12928 | 0.18805 |
| Inverted / valid | 0.05835 | **0.09642** | 0.11414 | 0.19382 |
| Normal / reference becomes uninformative | 0.06189 | **0.21916** | 0.13021 | 0.19374 |
| Inverted / reference becomes uninformative | 0.05413 | **0.22830** | 0.13632 | 0.18857 |

Values are final decision regret plus probe cost; smaller is better.

A valid reference changes the identifiable information about the optimal action and dramatically improves decisions. **A silently broken reference makes this very same intervention actively harmful.** We did not simulate truly arbitrary reference failure; this is a targeted model-assumption negative control.

### Cycle 36: general finite-hypothesis identifiability certificate

Files: `cycle36_identifiability_certificate.py`, `cycle36_identifiability_certificate.json`.

Given any supplied finite response-kernel table Q_e(m) and reward matrix U(m,a), group candidate latent models with the same distribution over **every currently accessible experiment**. Mark a group as *decision ambiguous* if there are models in that class with different unique optimal actions. Check which potential extra experiments split those ambiguities.

For 100 independent instances of the polarity model: 128 candidate models collapsed into 64 indistinguishable two-model classes under six normal sensors. Every such class contained **opposite optimal actions**. Either reward-action probe or reference probe split all 64; a dummy uninformative query split zero. All 100 instances passed the certificate.

The script treats response-vector equality to 10 rounded decimal places, presumes the entire candidate table is accurate and finite, assumes a unique preferred action in this instance, and tests **structural** rather than finite-data identification. It is a useful automated impossibility/checking facility, not a new theorem or an operation that discovers unknown model classes.

### Elementary formal limit (known statistical equivalence principle)

Let M be a finite set of fixed candidate environments and E a menu of experiments. For each environment m and experiment e, let Q_e(·|m) denote the complete response distribution, and let A*(m) be the set of optimal terminal actions.

Define observational equivalence m ~ m' iff Q_e(·|m)=Q_e(·|m') for **every** permitted experiment e.

**Necessity:** If some equivalence class C has empty common optimal-action intersection `intersection_{m in C} A*(m) = empty`, no adaptive sequence of experiments using E can reliably choose one action that is optimal in every m in C. Proof: by induction, the probability law for the entire adaptive transcript is the same in every m in C, since each adaptively chosen query has the same response law. Thus the algorithm cannot produce distinct action distributions tailored to m.

**Sufficiency (asymptotic under strong assumptions):** If every equivalence class C has a common optimal action, the models are finite, query responses are stationary and independent conditional on m and chosen e, and the agent can repeat each experiment arbitrarily often, empirical response distributions converge to the correct class. Because there are finitely many distinguishable pairs, an arbitrarily accurate class estimate can be obtained and then a common optimal action selected. This says nothing about finite-budget cost and cannot apply directly without assumptions to arbitrary nonstationary POMDP histories.

For a pair of indistinguishable models with opposite unique optimal actions and positive action gap, the regret of any algorithm on at least one model is bounded away from zero. This remains true even with infinitely many ordinary observations; adding a genuinely distinguishing intervention changes the experimental *information geometry*, not merely the training-set size.

The concrete polarity gauge transformation pairs (binary state bits, normal polarity) with (complemented state bits, inverted polarity). Sensor observations match exactly, while the reward-optimal action flips. A known-state reference or a reward observation violates the symmetry. **This is classical identifiability and active experimental design reasoning; novelty status: not established.**

### Crucial limitation and decision

A learner cannot automatically invent a universally effective diagnosis if its permitted observations contain no information about the needed distinction. Under the standing research standard, this is a specification of **what new intervention access or structural assumptions must be declared**, not permission to invent claims of unrestricted knowledge discovery.

Stop iterating on this finite supplied-model generator. Value-directed planning, reference calibration and equivalence-class certificates all have established antecedents. A consequential next algorithm must handle *model discovery and model criticism from raw data*, without receiving the right candidate likelihood table.

## Cycle 37: real-world randomized controlled test of decision-focused representations

Files: `cycle37_hillstrom_policy.py`, `cycle37_primary_result.json`, `cycle37_hillstrom_policy_results.json`.

**Data:** 64,000 customers in the 2008 Kevin Hillstrom MineThatData three-arm randomized email campaign, available from a public GitHub mirror of the original data and stored locally at `research/local_datasets/hillstrom.csv`. Original web origin failed TLS verification; the mirror downloaded under proper HTTPS without disabling certificate checks. SHA256:
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

Upstream data redistribution terms are not clearly permissive; **do not push the CSV**. The directory is Git-ignored. The experiment contains 21,306 No E-Mail, 21,307 Mens E-Mail, and 21,387 Womens E-Mail customers, consistent with 1/3 allocation. Outcome is a binary website visit. Observed rates: No 0.10617, Mens 0.18276, Womens 0.15140. Important: these are randomized *population averages*, not individual treatment effects.

**Estimand:** expected two-week website visit probability of a policy choosing among the three arms, minus an explicitly **hypothetical, probability-scale 0.012 penalty** per email (NOT empirical dollar spend or actual business costs). Uses known random assignment probabilities 1/3. Three-arm direct outcomes/policies; do NOT collapse the two emails into an unspecified generic treatment.

**Protocol:** eight train/validation/test splits with 60/20/20 stratified arm allocation. Only pre-treatment fields `recency, history, mens, womens, newbie, zip_code, channel` may enter features. Outcomes and assigned campaign never enter the input representation. Models fit train data only; validation chooses the policy, heldout customers evaluate once. The same 64,000 customers are reused in different sensitivity splits, so these are NOT eight independent field trials. Bootstrap resamples only the customers in the final test set, with trained models held fixed. Therefore CIs omit training-data/selection instability beyond the evaluation split.

**Model portfolio:** constant No/Mens/Womens; S-learner with logistic main effect + treatment interactions; T-learner separate logistic; S- and T-boosted outcome trees; T-random-forest outcome models; doubly robust transformed/pseudo-outcome gradient-boosted learner with a disjoint within-training nuisance split. A separate T-HGB nuisance model supports heldout doubly robust evaluation, without touching validation/test labels during training.

For action policy pi and arm propensity e_a=1/3, the *heldout* unit-level DR value is

```text
m_hat_pi(x) + 1[a==pi(x)]*(y-m_hat_a(x))/e_a - 0.012*1[pi(x) != no_email].
```

All models select action maximizing predicted visit probability minus assumed email penalty. Validation DR chooses a single candidate from 3 constants + 6 trained models; sensitivity to other assumed penalties is exposed but not tuned on test.

**Evidence:** In the primary split, the validation-selected S-logistic interaction policy achieved heldout estimated value 0.17801 vs 0.18027 for always-Mens Email, difference -0.002261, paired SE 0.004602, bootstrap 95% interval [-0.01130,0.00756]. Among eight distinct but overlapping splits, the chosen learner was S-logistic 4 times, S-boosted once, and constant Mens three times. **No nonzero selected difference had a paired 95% interval excluding zero.** One split's selected learner was +0.00381, but its CI included zero; some were negative.

Scientific interpretation: No reliable improvement over a competent constant policy on this real, randomized, observed decision objective. A flexible predictor is not necessarily a valuable action policy. This does NOT refute uplift modeling in general; the available covariates may not identify stable heterogeneity, website visits are a partial reward, and sample sizes may be insufficient to resolve small conditional effects.

## Cycle 38: externally collected randomized treatment data under send quotas

Files: `cycle38_budgeted_uplift.py`, `cycle38_budgeted_uplift_results.json` (initial exploratory), `cycle38_conservative_results.json` (corrected stronger fallback).

**Question:** Maybe individualized policy choice is useful when sending Mens E-Mail to everyone is infeasible. At fixed hypothetical 20% or 40% send quotas, is it better to rank which customers get Mens Email versus sending it uniformly at random?

Models: T-boosted, T-logistic, S-logistic, a direct DR-pseudo learner, and two simple pre-treatment heuristics (historical spend descending; purchase recency ascending). Exactly the required number are selected at each split. The stochastic uniform-random comparator sends with quota q **in expectation**, which gives a lower-variance DR reference rather than randomly drawing an arbitrary rank.

Validation chooses one ranking based on known-propensity DR score; test estimates its value relative to uniform random quota. **Important correction:** the first exploratory script forced selection among rankers. After reviewing this, a stronger and more realistic version added the stochastic uniform-random fallback as an eligible validation choice. This change was made AFTER inspecting the exploratory results; the corrected eight-split study is still the same underlying field trial and is sensitivity research, NOT fresh independent confirmatory field data.

Corrected selection, across eight reused-data split scenarios:

| Quota | Heldout improvement over uniform random across splits (ranges) | Calibration |
| --- | --- | --- |
| 20% Mens Email | from -0.00552 to +0.00863 | Some splits positive, others negative; no stable method selected |
| 40% Mens Email | from -0.00085 to +0.00533 | Mostly small positive point estimates with paired CIs crossing zero |

Two selected policies at 20% had intervals excluding zero on **individual** splits, but these were among multiple exploratory analyses with overlapping data and model selection; they do **NOT** justify a confirmed general targeting gain. At 40%, none of eight selected-policy intervals excluded zero. There is no robust, repeatedly winning representation or targeting algorithm.

**Economic limitation:** Quotas are hypothetical, visit is not profit, and the observed group contrast has limited two-week scope. We must not infer treatment personalization has proven business value here.

## Cycle 39: independent evaluator/experiment validity sanity checks

Files: `cycle39_evaluator_sanity.py`, `cycle39_evaluator_sanity.json`.

On 120 independently generated randomized three-arm datasets with 12,000 units each and known true conditional treatment responses, compare DR policy-value estimates with the exact simulated mean under **deliberately misspecified** nuisance models. Mean DR estimation error +0.000511 with SE 0.000574, statistically consistent with unbiasedness; raw IPS error +0.000798 with SE 0.000618. Verify by unitwise identity that a stochastic mixed mailing rule's DR score equals the convex combination of deterministic constant-arm DR scores.

On Hillstrom itself, confirm three-arm counts [21306,21307,21387], chi-square 0.20253 against uniform planned randomization (p=0.90369), no post-treatment label or arm included in features, no reported null/empty pre-treatment feature rows.

These checks reduce the risk of an implementation-driven false research claim. They are **not** proofs that every inferential/model assumption is correct. In particular the individual causal effect remains unobservable.

## Cross-cycle assessment: what is consequential, what is not

**Strongest established constraints**

1. Task-oriented information gain can require long-horizon planning; immediate policy improvement can be exactly zero despite a jointly valuable experiment sequence.
2. Optimal inference *inside an incorrect class* is not robust by definition, and in our tests old-model probes sometimes more than doubled final regret relative to abstention.
3. When accessible observations make reward-distinct models observationally equivalent, extra samples alone cannot help; a known-state reference or outcome feedback can change identifiability. But assuming a reference remains valid without testing can itself be harmful.
4. A simple general finite-model certificate can flag such ambiguities when all candidate likelihoods are supplied, distinguishing structural identification from mere finite-sample estimation error.
5. On a real, genuine randomized field experiment, strong individualized prediction and direct treatment-effect learners did **not** establish a consistent heldout policy advantage over reasonable simple alternatives.

**What was not achieved**

- No representation family or intervention operation was invented without structural templates.
- No robust online detector and repair learner faced a continuous unknown-onset nonstationary environment.
- No method beat known exact optimal Bayesian experimental design on its own assumptions.
- No real-world individualized uplift result survived a stringent repeated holdout interpretation.
- No new theorem beyond established statistical equivalence/value-of-information reasoning.
- No evidence of a compelling default over strong reward-machine, R-PSR, causal representation, active model learning or deep model-based RL systems.
- Synthetic model-access and controlled intervention rights remain far more favorable than typical practical deployments.

**Novelty classification:** Established mathematical principles and controlled empirical counterexamples with an independent real-data negative control. Useful methodological progress in specifying what a genuine new learner would need. **Not a proven breakthrough or a new paradigm.**

## Decision and next high-upside research specification

STOP expanding the current fixed finite hypothesis and sensor-polarity suite. A version-space planner with known likelihoods, optimally selecting a supplied reference, is already explainable by Bayes design; more toy wins would not constitute consequential novelty. Likewise stop fitting ever more arbitrary Hillstrom variants unless a proper held-out independent RCT or a declared prospective validation is obtained.

A genuinely stronger next research effort must operationalize **model-class criticism, creation of new predictive distinctions, and construction of useful diagnostic actions from raw history**, with strong existing alternatives. No-free-lunch results forbid unrestricted success without structural assumptions. Declare those assumptions explicitly.

A minimum decisive protocol must include: (i) multiple **different continuous partially observed decision environments** beyond the old short-episode simulator; (ii) unannounced changes, including nuisance-only drift, reward-relevant coupling, violated reference assumptions and unknown mechanisms; (iii) observation/reward/intervention privileges strictly equal across algorithms; (iv) real cumulative task reward and total training, probing, compute and calibration cost; (v) strong R-PSR/reward-machine, recurrent/attention, learned latent-state, online changepoint, causal policy and optimistic Bayesian or posterior-sampling baselines as appropriate; (vi) out-of-grammar mechanisms and explicit failure conditions; (vii) true train/validation/test and independent task-family splits.

Promising but **still unverified** organizing target: a learning agent that can declare its current decision abstraction non-identifying, construct or request a scientifically testable new measurement operation to break the relevant ambiguity, validate that the operation's own semantics have not drifted, and reuse learned distinctions across tasks with less *total* interaction/computation than established systems. The distinguishing consequence would be a new **reliable operational capability** in environments where a static observation/action interface is inadequate. Closest predecessors include reward machines, Bayesian design, active automata learning, R-PSRs, POMDP CEGAR, online change detection and causal structure learning. The idea could reduce to integration unless experiments demonstrate a substantive new mechanism or capability.

**Avoid:** claiming all-generic representation invention, feature engineering disguised as novelty, or treating an accurate simulator oracle as accessible learning data.


## Reproducibility and local machine state

All code and calculations ran on the user's MCP-connected Linux PC, from `/home/dtdav/dev/learn`, in the **existing** `.venv`, not the assistant container. Required packages: Python 3.12.3, NumPy, SciPy, scikit-learn, Torch, POPGym/Gymnasium, and pandas 3.0.6 (installed in the existing environment for the real-data audit). `requirements.lock.txt` records exact installed dependency versions.

The external RCT mirror CSV is cached **locally only** in `research/local_datasets/` (explicitly Git-ignored), with precise SHA256 recorded; anyone reproducing the RCT study must separately download the same upstream data subject to its terms. Do not commit or redistribute the raw data. Model scripts never read user private records or secrets.

To rerun from the repository root, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```bash
.venv/bin/python research/cycle32_decision_design.py --seeds 1000
.venv/bin/python research/cycle32_xor_validation.py
.venv/bin/python research/cycle33_model_misspecification.py --seeds 1600
.venv/bin/python research/cycle34_reward_anchor.py --seeds 800
.venv/bin/python research/cycle35_reference_intervention.py --seeds 1200
.venv/bin/python research/cycle36_identifiability_certificate.py
.venv/bin/python research/cycle37_hillstrom_policy.py --splits 8 --bootstrap 600
.venv/bin/python research/cycle38_budgeted_uplift.py --splits 8 --bootstrap 600 --output research/cycle38_conservative_results.json
.venv/bin/python research/cycle39_evaluator_sanity.py
```

Random seeds, environment generation, exact intervention and estimated intervention propensity, cost assumptions, train/test split structure, algorithms and means/paired standard errors are in the program source and JSON results. The `*.log` files are Git-ignored. Several scripts rerun to their default named JSON output and will overwrite its current content; use `--output` to retain prior artifacts.

The original forced-ranking exploratory result for Cycle 38 remains saved in `cycle38_budgeted_uplift_results.json` and is not silently replaced by the corrected stronger reference. Likewise `cycle37_primary_result.json` preserves the early single-split audit. Only complete scripts and aggregate results, not hidden latent task truth or customer-level raw data, are versioned.

**Session boundary:** No continuous background research or scheduled execution is claimed. Subsequent research should restart from this report and `STANDING_RESEARCH_STANDARD.md`, verify Git state, independently review the methods and results, and **pursue a distinct high-upside question rather than continuing these saturated toy benchmarks**.
