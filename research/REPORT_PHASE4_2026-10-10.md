# Phase IV: External partial-observability, temporal program induction, and identifiability

**Date:** 2026-10-10. **PC:** `/home/dtdav/dev/learn`. **Governing standard:** `STANDING_RESEARCH_STANDARD.md`.

**Status:** Cycles 24-28 completed on an independent external POPGym environment. Cycle 29 separately tests measurement-noise robustness. No new general-purpose learning algorithm, new mathematical theorem, or breakthrough is established.

## Scientific objective and independently formulated competing hypotheses

The earlier Phase III experiments established that a researcher-specified interaction model can beat a restricted HMM at decision making and a stronger latent model reverses part of that ranking. Their central weakness was selecting a *supplied* adequate representation rather than inventing one.

We deliberately moved to a third-party benchmark of **genuine multi-step trajectories and observed rewards**, not the old independently reset three-step synthetic POMDP. We test:

- **H1 (supplied correct prior):** A learner that is handed precisely the right family of memory operations can greatly outperform a generic neural memory baseline with modest data. This is likely *engineering/inductive bias*, not a novel learning principle.
- **H2 (ordinary strong baseline):** A conventional learned reward predictor with those same features eliminates much of the apparent advantage. If so, reject claims that the program selection rule is itself novel or superior.
- **H3 (transferable mechanism vs fitting):** A program inferred on one environmental configuration may be unidentifiable as a transferable abstraction. Diversity of environments can disambiguate equivalent formulas at fixed total information cost; equally, an appropriately chosen prior may resolve the ambiguity without diversity.
- **H4 (finite data/noise):** Structural identifiability does not guarantee statistical reliability under noisy outcomes. Test this separately rather than treating a noiseless result as general.

## Tools and external sources

- On the user's PC, `popgym==1.0.6`, `gymnasium==1.4.0` installed **into the existing `.venv`** using pip; NumPy/SciPy/scikit-learn/Torch already installed. No other environment created.
- POPGym benchmark: https://github.com/proroklab/popgym and https://arxiv.org/abs/2303.01859 . Independent third-party tasks include Repeat First, Repeat Previous and Autoencode, with actual observation/action history and reward feedback.
- POPGym's own RLlib-based original baseline package lists dependency incompatibilities; therefore we wrote transparent modern PyTorch/sklearn baselines and do not compare against unreproduced published scores.
- The experiments give neither true state labels nor the correct action on training steps. Actions are uniformly randomized; learners see ONLY the chosen action's reward and observation sequence. Evaluation is by actions actually taken in fresh live episodes.
- **Important unequal inductive priors and compute:** The symbolic and linear models receive deliberately structured temporal feature libraries, while GRU receives raw observation, previous logged action and reward. This is not a fair test of inherent algorithm-class superiority. The comparison demonstrates exactly why stronger structural controls are needed.

## Cycle 24: external POPGym, supplied memory program versus GRU

Source: `cycle24_popgym_memory.py`
Data: Eight fresh trials per task, 350 random-action training episodes and 100 independently seeded evaluation episodes, GRU with 48 hidden units and heldout reward BCE early stopping. The program method selects the reward-maximizing expert from a supplied library: `first`, `lag1` ... `lag64`, `mode`, scored using unbiased inverse propensity reward estimates. Program family directly contains the correct operator for each tested task.

Mean live-episode returns (POPGym normalized range near -1 to 1, larger is better):

| Task | Supplied program | GRU reward predictor | Memory-free tabular |
| --- | ---: | ---: | ---: |
| RepeatFirstEasy | 1.000 | 0.979804 | -0.480343 |
| RepeatPreviousEasy (k=4) | 1.000 | 1.000 | -0.494531 |
| RepeatPreviousMedium (k=32) | 1.000 | -0.492743 | -0.506319 |

All eight program selections in each task found the correct index. This is expected given a hand-coded complete grammar. Cycle 24's recorded program fitting time includes training-data collection, while the GRU fitting time excludes it. The timings are not directly comparable. In particular, the GRU result does **not** establish a neural learning impossibility. No effort matched computational budget or tuned an extensive recurrent architecture family.

## Cycle 25: adequate ordinary linear model is sufficient

Source: `cycle25_linear_memory.py`.
A standard sklearn logistic regression learns action reward probability from temporal action-match features. A 20% validation split chooses regularization C in {0.1, 1, 10}, then the model is refit to all logged training examples. No test selection.

Eight novel trials, 350 training + 100 test episodes per trial:

| Task | Full first/lag1..64 features | Deliberately missing lags>16 |
| --- | ---: | ---: |
| RepeatFirstEasy | 1.000 | 1.000 |
| RepeatPreviousEasy | 1.000 | 1.000 |
| RepeatPreviousMedium | 1.000 | -0.460382 |

Thus ordinary logistic regression matches the symbolic program learner exactly when given the right temporal features, while both classes fail when the necessary lag is absent. This **removes support for a special new algorithm in Cycle 24**. Successful inductive bias is not automatic construction of that bias.

## Cycle 26: arithmetic program induction across external tasks

Source: `cycle26_memory_program_synthesis.py`.

Instead of naming `first`, `lag4`, or `reverse`, the programmer supplies an **expression grammar**:

```text
index(t,W) = a*t + b*W + c
a in {-1,0,1}; b in {-2,-1,0,1,2}; c in {-70,...,70}
```

The observed `t` is the current time index, and `W` the first observed phase-switch index plus one (zero if no phase switch). Candidate action is the observed suit at the computed index, with a default action for invalid indices. Fit selects the expression with the greatest randomized-action inverse propensity reward estimate. True states and correct actions are never supplied.

This is a small but genuine **composition of a useful temporal operator out of arithmetic primitives**. It is textbook enumerative program synthesis, not an unconstrained learner that invented the primitives.

Independent smoke trials found optimal rules for `RepeatPreviousEasy` and `AutoencodeEasy`. Additional 3 trials on AutoencodeEasy/Medium/Hard found perfect within-length returns, but distinct same-length expressions and a transfer ambiguity. Those results led to the stronger, predeclared Cycle 27 test with a widened grammar.

## Cycle 27: perfect within-task fit does not identify a transferable operator

Source: `cycle27_transfer_identifiability.py`.
Widened grammar: a in {-1,0,1}, b in {0,1,2}, c integer [-325,325]. There are 5,859 programs. Under this class, the correct family for the observed Autoencode task is

```text
f(t,W)=2*W-t-2.
```

At **fixed W=w0**, all coefficients satisfying `b*w0+c=constant` are observationally equivalent on every possible time index. Thus no amount of additional reward-bearing observations at the same W can identify b and c separately. Two distinct values W1 != W2 make the design matrix [[W1,1],[W2,1]] full rank, distinguishing the parameters, provided rewards cover enough of the actions/time steps. This is elementary structural identifiability and existing system-identification theory, NOT a new theorem.

Four groups use exactly 6,240 scored training actions, plus independent heldout evaluation in AutoencodeHard (W different from either training length):

| Training group | Rewarded examples | Mean Hard return, 48 trials | Selected program |
| --- | ---: | ---: | --- |
| Easy only (120 episodes) | 6,240 | -0.500000 | (-1,0,102) |
| Medium only (60 episodes) | 6,240 | -0.502891 | (-1,0,206) |
| Easy 60 + Medium 30 | 6,240 | 1.000000 | (-1,2,-2) |
| Easy only + favor simple coefficients in exact IPS ties | 6,240 | 1.000000 | (-1,2,-2) |

The one-length groups each had three empirically tied operators. Mixed lengths had one. Distinct tie-breaking choices led to dramatically different heldout policies. **A favorable MDL-like prior achieved perfect transfer without mixed training.** Therefore do not claim variation is universally necessary for successful transfer; it is necessary for data-only identification in this family without a distinguishing prior.

## Cycle 28: cost of one informative environmental variation

Source: `cycle28_diversity_experiment.py`. Prespecified m in {0,1,2,5,10,20,30} Medium episodes; replace two Easy episodes per Medium episode to hold rewarded training actions EXACTLY at 6,240 for every condition. Each method uses the same reward-only program-fitting procedure, followed by unseen AutoencodeHard testing.

In 48 new trials, m=0 chose (-1,0,102) and had mean return -0.5. For **every** m>=1, all 48 trials chose (-1,2,-2), with mean Hard return 1.0. One varying environment episode sufficed in this deterministic suite with this high-signal observation/reward protocol.

This is evidence for **experimental-design efficiency**, not proof of a general one-intervention recovery rate. The reward is noise-free apart from randomized action feedback, the operator form is researcher-supplied, and the environment parameter W is visibly provided by the observation transition. **Cost caveat:** exactly 6,240 rewarded training decisions are matched, but the total number of environment steps differs: 120 Easy episodes use 12,360 steps; replacing two Easy episodes with one Medium episode adds one unscored step per replacement, or 30 extra steps (+0.24%) at m=30. This small unmatched cost is disclosed.


## Cycle 29: structural versus statistical identifiability under noisy reward measurements

Source: `cycle29_reward_noise.py`; results: `cycle29_noise_diversity_results.json`.

We did NOT claim POPGym has stochastic or corrupt rewards. This is a separate **synthetic measurement-noise stress test**: each *training-only* nonzero reward sign is flipped independently with probability eta in {0,0.20,0.40,0.45}. Action and observation sequences remain native POPGym; Hard evaluation uses intact rewards. The program learner is unaware of the sign flip. All conditions keep exactly 6,240 nonzero training reward observations; a Medium episode replaces two Easy episodes.

Across 32 NEW seeds:

| Reward-sign error rate | 0 longer episodes | 1 longer episode | 5 longer episodes | 20 longer episodes |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 0% | 100% | 100% | 100% |
| 0.20 | 0% | 100% | 100% | 100% |
| 0.40 | 0% | 87.5% | 100% | 100% |
| 0.45 | 0% | 62.5% | 93.75% | 100% |

Entries give exact transferable formula selection frequency. The 30-longer-episode group also achieved 100% at each tested corruption level. At eta 0.45, the 1-longer-episode mean Hard return was about 0.4381, while 5 reached 0.9056 and 20 achieved 1.0.

**Interpretation:** structural rank removes infinite-data ambiguity, but finite noisy observations still require adequate identification evidence. The apparent one-experiment sufficiency of Cycle 28 is sensitive to feedback noise. This is established statistical identification and does not constitute an independent new learning principle. The apparatus changes logged reward measurements after generation and therefore is not itself an externally defined noisy POMDP.

## Cycle 30: heldout further-horizon transfer check

Source: `cycle30_extended_transfer.py`; results: `cycle30_extended_transfer_results.json`.

For 24 genuinely fresh training seeds, fit the same selected affine program from 60 Easy + 30 Medium episodes (6,240 rewarded observations), without using unseen episode-length results for tuning. Evaluate on `popgym.envs.Autoencode(num_decks=d)` for **d=3,4,6,8**; each uses 24 independent test episodes per seed.

Every training seed selected `(-1,2,-2)`. Mean episode return was **1.0 for all four heldout deck counts**, including length considerably beyond those seen in training. This is a valid *within-PoPGym-Autoencode-family* extrapolation finding. It cannot establish transfer to other POMDP mechanisms or unrestricted program invention.

## Scientific conclusions

1. **A prior containing the correct operator explains Cycle 24 success.** An ordinary logistic reward model given equivalent temporal features removes the proposed learner's apparent superiority (Cycle 25). The correct conclusion is useful structural features, not a new learning algorithm.
2. **A compositional program can be inferred from reward-only feedback.** Searching a general affine index-expression grammar found a reverse-memory expression without enumerating `reverse` as a named atomic operation (Cycle 26). This remains established program synthesis in a small researcher-defined DSL.
3. **Task success does not establish a reusable representation.** In a single fixed-length environment, a whole equivalence class of parameterizations has exactly the same observable behavior for all possible training data, yet divergent heldout decisions. More same-distribution data cannot disambiguate it.
4. **Environmental variation and a good prior are alternative explanations for transfer.** Two lengths make coefficients identifiable from data; a favorable simplicity prior selects the same transferable rule without diverse data. Neither is universally sufficient outside the given model family.
5. **Identifiability and finite-sample power must be separated.** One informative environment variation suffices under clean reward feedback but can fail at high feedback noise.
6. **The researcher's decisions still supply most of the solution.** The affine grammar, observation encoding, selectable lengths and transfer target were furnished by the investigator. The learner did not discover which interventions to create, which representation language to construct, or when to change its language; no unknown-change adaptation is tested.
7. **Real-world consequence is not yet established.** These are executable external benchmark and formal-identification controls in elementary card-memory tasks. No state-of-the-art reinforcement learning comparison, direct transfer to continuous control, compute-matched advantage, or newly established theorem supports a breakthrough claim.

**Classification:** Valid conditional method/representation identifiability evidence and a useful negative control against superficial novelty; no novel general principle demonstrated.

## Verified predecessors and open originality boundary

The following were independently located in current public sources, not inferred solely from the earlier reports:

- Ljung and Glad (1994), *On global identifiability for arbitrary model parametrizations*: https://doi.org/10.1016/0005-1098(94)90029-9 . Structural parameter identifiability and persistent excitation are well-established. Our affine rank example is an instance, not a novel theorem.
- Rajendran, Reizinger, Brendel and Ravikumar (2024), *An Interventional Perspective on Identifiability in Gaussian LTI Systems with Independent Component Analysis*: https://proceedings.mlr.press/v236/rajendran24a.html . Shows the significance of choosing informative interventions for system identification in another formal class, not identical to the POPGym setting.
- Bing et al. (2024), *Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions*: https://proceedings.mlr.press/v236/bing24a . Multi-environment intervention diversity is a close conceptual precedent for identifying otherwise ambiguous representations.
- Varici et al. (2024), *General Identifiability and Achievability for Causal Representation Learning*: https://proceedings.mlr.press/v238/varici24a.html . Additional formal coverage of multiple interventions and identifiable latent representations.
- DreamCoder (Ellis et al., 2021), https://doi.org/10.1145/3453483.3454080 ; program induction/library learning explicitly predates our bounded expression search.
- Foster, Krishnamurthy and Luo (2019), *Model Selection for Contextual Bandits*: https://papers.neurips.cc/paper_files/paper/2019/hash/433371e69eb202f8e7bc8ec2c8d48021-Abstract.html . Strong precedent for adaptive policy-class selection in reward-feedback decision problems.
- Krishnamurthy, Propp and Athey (2024), *Towards Costless Model Selection in Contextual Bandits*: https://proceedings.mlr.press/v238/kumar-krishnamurthy24a.html . Shows bias-variance-adaptive model selection, including limits and benign cases.
- Alver, Rahimi-Kalahroudi and Precup (2025), *Partial Models for Building Adaptive Model-Based Reinforcement Learning Agents*: https://proceedings.mlr.press/v274/alver25a.html . Close prior on localized model repair.
- Oh et al. (2025), *Discovering state-of-the-art reinforcement learning algorithms*, Nature: https://www.nature.com/articles/s41586-025-09761-x . DiscoRL learns update targets and prediction semantics with meta-gradients across broad environments, reporting strong Atari and heldout benchmark performance. This is an especially consequential predecessor for any claim of autonomous invention of learning rules. It does not automatically solve the precise problem of identifying the cheapest decision-discriminating interventions.
- POPGym (Morad et al., 2023), https://arxiv.org/abs/2303.01859 . Third-party continuous-trajectory partially observable benchmarks.

This remains a **targeted, not exhaustive, audit**. It is improper to claim the current elementary experiment is unprecedented, or to dismiss a future intervention/representation invention system as identical to the above solely because some ingredients overlap.

## High-upside hypotheses for the next distinct phase

**A. Decision-targeted experiment and representation co-design (leading candidate, conditional).** A learner should retain a *version space* of plausible decision-relevant abstractions. Where candidate abstractions agree on all seen data but disagree on actions that matter in plausible deployment settings, use its intervention rights to select a low-cost environment or action probing that disagreement; only then revise its abstraction. The defining mechanism would be **actively breaking decision-relevant observational equivalence, rather than optimizing generic prediction entropy**.

Distinctive test: The learner must (i) propose/construct its own new diagnostic features or memory operators, not simply choose among supplied exact solutions, (ii) select interventions without being told which domain parameter resolves ambiguity, and (iii) outperform strong adaptive-experiment-design, Bayesian latent-model, PSR, program-induction, and standard RL/meta-learning baselines on a diverse set of unseen environments at comparable total data and compute budgets. Negative controls must include non-identifiable environments and variation that changes observations but not decision equivalence. If conventional active model selection explains every improvement, classify the result as useful integration, not a new paradigm. The present Cycle 27-30 results DO NOT satisfy any of those three conditions.

**B. Learned update rules and prediction semantics (independent competitor).** Rather than construct memory programs directly, meta-learn a general update rule across large task families. This has obvious breakthrough upside but is directly preceded by DiscoRL. Given one RTX 3060 Ti (8GB), a small toy reproduction cannot support comparison to published Atari/ProcGen-scale evidence. A differentiated research target and realistic computational plan would be required before spending substantial resources.

**C. Partial observable predictive sufficient statistics (alternative).** Learn controlled reward-/policy-relevant representations from raw histories and unknown change points. Strong predecessors include reward-predictive PSRs, value-directed POMDP compression, structured Bayesian filtering, and recent partial-model RL. The prior research's single synthetic task does not establish progress over these. Independent continuous-control tasks, full trajectory conditioning and memory/computation budgets are prerequisites.

## Next decisive research protocol and stopping conditions

1. Benchmark at least three heterogeneous partially observable task families, including one with *no correct operator in a predeclared grammar* and one with task-irrelevant observation drift. Keep heldout environments, nuisance mechanisms, and intervention schedules sealed before model tuning.
2. Give methods identical observation/reward/intervention rights and report total trajectory steps including rewardless setup, expensive querying and model search. If a method requires knowledge of possible intervention parameters or operator templates, state it as a prior and give strong existing methods equivalent access.
3. Compare the strongest actual alternatives including efficient recurrent/attention RL, online model selection with uncertainty control, latent predictive models, value-aware alternatives and program synthesis. Tune each with a declared, similar budget, and record action-policy regret or cumulative returns, not only predictive log loss.
4. In the best candidate, study the *mathematics of task-directed experimental distinguishability* before implementing: which counterfactual decision classes are statistically identifiable with the available actions, what minimal coverage is necessary, and what intervention-cost bounds apply? Treat impossibility results and established theorems as existing knowledge.
5. If the candidate does not learn new distinctions without supplied right answers or cannot outperform strong established approaches in a consequential decision-cost regime, **stop optimizing the synthetic grammar and redirect** rather than multiplying operators.
6. Preserve a failed attempt or negative result with an explanatory counterexample. Never reframe a model-class privilege advantage as a learning breakthrough.

## Reproduction and exact files

From `/home/dtdav/dev/learn` with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```bash
.venv/bin/python research/cycle24_popgym_memory.py --trials 8 --train 350 --test 100 --epochs 50 --tasks RepeatFirstEasy RepeatPreviousEasy RepeatPreviousMedium
.venv/bin/python research/cycle25_linear_memory.py --trials 8 --train 350 --test 100
.venv/bin/python research/cycle26_memory_program_synthesis.py --trials 3 --train 60 --test 30 --tasks AutoencodeEasy AutoencodeMedium AutoencodeHard --output research/cycle26_autoencode_results.json
.venv/bin/python research/cycle27_transfer_identifiability.py --trials 48 --eval 80
.venv/bin/python research/cycle28_diversity_experiment.py --trials 48 --test 40
.venv/bin/python research/cycle29_reward_noise.py --trials 32 --eval 30
.venv/bin/python research/cycle30_extended_transfer.py
```

Aggregated result JSON files are checked in alongside executable scripts; logs are Git-ignored. Smoke checks and interim results are separately named. No privileged latent truth was used in model fitting. All study computations ran on the user's Linux PC via Terminal MCP, not in the assistant container. Research expenses were small CPU workloads; GPU energy was not measured. Git supplies continuity. `requirements.lock.txt` was updated to include POPGym, Gymnasium and Farama Notifications for reproducibility.

**Governing conclusion:** Useful progress is learning which *distinctions and intervention rights* make transferable representations identifiable and computationally worthwhile. The external benchmarks show that model-family priors and environmental variation can dominate data quantity. Their known mathematical ingredients and small task scope preclude a breakthrough claim.

## Addendum, Cycle 31: automatic experiment selection, not a new design rule

Source: `cycle31_active_experiment.py`; result: `cycle31_active_experiment_results.json`. A learner first receives 118 Easy randomized-action episodes and extracts the entire set of empirically tied highest IPS-scoring affine programs from the existing grammar (three candidates). It is offered two experiment choices with known intended lengths, an Easy W=52 episode option or a Medium W=104 episode option. **No true reward distribution, unseen target labels or correct model coefficients are supplied.**

For each candidate experiment, compute the fraction of test-time indices on which plausible programs would select different observable memory positions; divide by episode interaction cost and choose the maximum. If Medium is selected, add one Medium episode; if Easy is selected, add two Easy episodes, holding total rewarded training observations fixed at 6,240. Refit the same program search and evaluate on AutoencodeHard.

In 48 fresh seeds, every run discovered three tied training hypotheses, scored Easy at zero inter-hypothesis disagreement, scored Medium at positive disagreement, selected Medium, and recovered `(-1,2,-2)`. All 48 achieved heldout normalized return 1.0.

This **implements an automatic information-seeking choice** rather than having the researcher always supply the informative length. However, its environment menu, observable length W, hypothesis grammar, and disagreement objective are supplied. Standard active learning / version-space experiment design predicts exactly the same selection. It is a useful prototype of an end-to-end measurement-selection loop, **not** evidence of a novel algorithm or superiority to established optimal information design.

The high-impact unresolved question is whether such a learner can **construct novel diagnostic hypotheses and interventions** without hand-written operator/access assumptions, with safe change handling and competitive total learning cost. Do NOT extend the fixed Autoencode grammar merely to make more positive results.


Additional reproduction:

```bash
.venv/bin/python research/cycle27_symbolic_validation.py
.venv/bin/python research/cycle31_active_experiment.py --trials 48 --test 80
```

**Research decision after Cycle 31:** STOP running more variations of this simple fixed affine memory-task grammar. Neither the compositional operator search nor automatic experiment selector has established novel scientific principles beyond existing program synthesis, persistent excitation and active design. The correct next work is an independently designed test of **construction of new, decision-relevant hypotheses and interventions** in heterogeneous partially observable environments, including wrong-prior and unknown-change regimes. Require strong established baselines and report real resource costs before making any breakthrough claim.
