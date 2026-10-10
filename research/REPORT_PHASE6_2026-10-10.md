# Phase VI: raw-history learners, unseen reward rules, and independent control benchmark preparation

**Date:** 2026-10-10. **Workspace:** `/home/dtdav/dev/learn/research`.
**Standard:** `STANDING_RESEARCH_STANDARD.md`.
**Status:** Cycles 40–44 independently executed, with corrected stronger early-stopping controls and a separate published POPGym Arcade benchmark pilot. **No new general-purpose algorithm, theory, or breakthrough established.**

## Deliberate research redirection

Phase V conclusively showed that known finite-model Bayesian experimental design outperforms simplistic heuristic experiment selection, that old likelihoods can be actively harmful under unknown sensor drift, and that many latent states are unidentifiable without fundamentally new intervention rights. It also supplied a real randomized-data negative control. Further improving the fixed finite model would manufacture more positive results without scientific novelty.

This phase asks:

1. **Raw representation acquisition:** Can a conventional strong model learn previously hand-supplied memory distinctions *only from observed histories and randomized chosen-action rewards* in independent external POMDP tasks, without a correct index being given as a feature?
2. **Within-episode feedback adaptation:** Can it infer the currently relevant memory/reward rule from its own experienced feedback when the rule ID is hidden and candidate tasks are familiar?
3. **Out-of-class revision:** Does that behavior persist for a genuinely withheld reward-rule mechanism? If it fails, can standard target-only fine-tuning discover it cheaply, rather than needing a full new model grammar?
4. **Operational benchmark and strong priors:** Can we test any future method on heterogeneous, action-dependent, partially observable control tasks with official, competitive recurrent baselines and heldout returns, rather than only card-memory reward prediction?

None of these questions presupposes a new paradigm. Both neural memory architectures and Bayes rule selection have strong established precedents. We explicitly test a **positive known-architecture explanation** before proposing any novel mechanism.

## Relevant verified predecessors

- POPGym, Morad et al. (2023), https://arxiv.org/abs/2303.01859 and https://github.com/proroklab/popgym , supplies third-party partially observable decision processes and established memory baselines.
- Wang et al. (2025), *Investigating Memory in Model-Free RL with POPGym Arcade*, https://arxiv.org/abs/2503.01450 and https://github.com/bolt-research/popgym-arcade , supplies a heterogeneous, pixel-based, common-interface POMDP benchmark with official PPO/PQN/DQN and recurrent memory controls. Its published scores/algorithms must be used as the real comparator for future learning proposals.
- Toro Icarte et al. (2023), *Learning reward machines: A study in partially observable reinforcement learning*, https://doi.org/10.1016/j.artint.2023.103989 , explicitly learns reward-memory automata from experience. Its provided atomic property detectors represent a documented assumption, not a reason to dismiss its central advance.
- Wu, Liu and Zhang (2025 preprint), *Inferring Reward Machines and Transition Machines from Partially Observable Markov Decision Processes*, https://arxiv.org/abs/2508.01947 , reports efficient inferred automata for both non-Markovian rewards and transitions, with substantial computational speedups. This raises the novelty and efficiency thresholds for any new symbol-induction claim; as a preprint, published performance claims require checking.
- *Learning Latent and Changing Dynamics in Real Non-Stationary Environments* (IEEE TKDE 2025), https://doi.org/10.1109/TKDE.2025.3535961 , already combines latent dynamics modeling and online change detection for adapting model-based RL to nonstationary environments. A simple change detector plus world model is not new.
- Existing causal Transformers, GRUs, TCNs, episodic meta-learning/in-context reinforcement learning, and Bayesian finite-model selectors are established. A positive showing by one is not a novel scientific principle.

## Cycle 40: generic memory learners versus supplied memory operators

Files: `cycle40_raw_memory_baselines.py`, `cycle40_raw_memory_results.json`; separate `cycle40_smoke.json`.

Use **native** POPGym 1.0.6 `RepeatFirstEasy`, `RepeatPreviousMedium`, `AutoencodeEasy`, and `AutoencodeMedium`. Actions are randomized during training. Models see only the current/past **observed card, phase marker and normalized step count**. No model sees a hand-engineered `lag4`, `lag32`, `first`, or `reverse` operator, a true hidden state, the correct action, or a privileged reward oracle. A reward is observed only for the randomly chosen action.

Architectures: memoryless two-layer MLP, 96-hidden-state GRU, 8-layer causal dilated temporal convolution with maximum 511-step receptive field, and a two-layer, 4-head causal Transformer with learned absolute positional embeddings (64 hidden dimensions). All optimized with masked binary cross-entropy on observed chosen-action rewards, with early stopping on an independent 20% of training episodes. Same logged training episodes, **NOT compute- or capacity-matched architectures or a complete hyperparameter sweep**.

Training: six independent seeds/task, 700 random-action episodes/seed, 70 maximum epochs, separate 140 live heldout episodes. Every method receives identical episode observations and chosen-action reward outcomes. The benchmark tasks were verified *action-independent in their future observation process* across independent seeds; the evaluation replays actual live actions and checks observation consistency at each step. No reward counterfactual labels are used to train. The policy depends only on observation history, never on previous actions/rewards, so this replay is a valid live evaluation on these tasks.

**Mean normalized live episode return (higher is better, optimum 1.0):**

| Native POPGym task | Memory-free | GRU | TCN | Causal Transformer |
|---|---:|---:|---:|---:|
| Repeat First Easy | -0.50696 | 1.00000 | 1.00000 | **1.00000** |
| Repeat Previous Medium (lag 32) | -0.49845 | -0.48892 | -0.24610 | **1.00000** |
| Autoencode Easy | -0.47766 | -0.33864 | -0.04583 | **1.00000** |
| Autoencode Medium | -0.48871 | -0.41264 | -0.28523 | **0.00304**, SE 0.31527 |

At Autoencode Medium, the Transformer achieved optimum in three of six seeds and near-chance in three; averaging these outcomes hides **bimodal optimization instability**. Claims of reliable learning or architecture dominance would be unwarranted. The Transformer mean fitting time was roughly 2.5–3.25 seconds/seed per task on the existing NVIDIA GeForce RTX 3060 Ti, but the less capable alternatives used less time, so this is a resource trade-off.

**Scientific significance before stronger controls:** Correct hand-built temporal features are not necessary for a conventional causal Transformer to fit three of these tasks. The longer Autoencode case appeared unreliable under the first training schedule. **Cycle 44 subsequently demonstrated this was primarily premature early stopping**, as detailed below. Both results are preserved rather than silently substituting the more favorable one. This is a conditional architecture/training observation, not a new model or a claim of architecture-level impossibility.

The task distribution is still narrow: card histories with rewards independent of earlier actions; no control-dependent dynamics, no unknown environmental change, no new intervention creation, no generality over partially observable control.

## Cycle 41: in-context hidden-rule identification and out-of-class failure

Files: `cycle41_hidden_reward_context.py`, `cycle41_hidden_reward_results.json`; separate smoke result.

**Important:** This is a *researcher-created composite task*, NOT the native POPGym reward function. Card observation trajectories are sampled from POPGym `RepeatFirstEasy`; a synthetic action-independent reward mechanism scores every step. Hidden reward rule within an episode is either `first` or `lag4` during training; **lag7 is withheld for testing**. All tasks have identical input spaces, episode lengths, scored-action onset and card-generation dynamics, eliminating a trivial distinction through episode duration or start-of-reward timing. The hidden rule ID is never an input to neural learners.

Training: six independently seeded experiments, 1,200 randomized-action episodes/seed from first/lag4 tasks, independent validation and test, 85 max epochs, 96 heldout live evaluation episodes per rule/seed. Raw input includes card, normalized timestep, previous *chosen* action and previous *observed* reward, with no correct-action labels. On test episodes, the model takes actual actions, receives feedback, and updates its context **without gradient steps**. Comparisons: GRU, causal Transformer, an otherwise identical Transformer without previous-action/reward context (negative control), exact Bayesian selection restricted to the two supplied true training rules, and exact Bayesian selection with the withheld lag7 rule added as a **privileged comparator**.

**Mean normalized return (higher better, 1.0 oracle):**

| Withheld/evaluation rule | GRU contextual | Transformer contextual | Transformer without feedback | Bayes 2-rule | Bayes 3-rule (includes heldout truth) |
|---|---:|---:|---:|---:|---:|
| Known first | 0.94901 | 0.94288 | 0.36417 | 0.97964 | 0.98032 |
| Known lag4 | 0.96378 | 0.91728 | 0.25776 | 0.97971 | 0.96078 |
| Unseen lag7 | **-0.32605** | **-0.33517** | -0.32135 | -0.31992 | **0.98100** |

Strong conventional neural models can infer and exploit **a known family** based on feedback, while failing to revise their semantics within a single episode when the correct rule was not represented in training. The Bayesian 3-rule control wins on unseen lag7 because **the researcher gave it that candidate model**. It is not a valid proof of superior autonomous class invention by Bayes.

The source observation trajectories and rewards are in a deliberately researcher-controlled discrete memory task. This finding demonstrates a specific *out-of-training-distribution mechanism failure*, not general impossibility of online neural representation discovery. All candidate-specific policies use the same heldout cards, but neural training seeds and model computational cost differ.

## Cycle 42: does ordinary postchange fine-tuning build an unseen mechanism efficiently?

Files: `cycle42_ooc_gradient_adaptation.py`, `cycle42_ooc_adaptation_results.json`; smoke `cycle42_smoke.json`.

Continue the **same composite externally generated card trajectories with synthetic reward overlay** from Cycle 41, not a new third-party task family. Initially train GRU and Transformer with 1,200 reward-logged episodes drawn from rule-first / rule-lag4 hidden tasks. Without telling either network the new rule, withhold `lag7` as a separate shift mechanism.

At target-only training budgets 0, 4, 16, 64, and 256 *independent randomized-action episodes* (each exactly 51 rewarded steps, hence 0/204/816/3264/13056 observed bandit outcomes), compare a copy of the source-pretrained model updated by ordinary AdamW gradient descent, versus the SAME architecture retrained from scratch with the SAME target logged actions, same validation splits and fixed update hyperparameters. Both methods use an independent heldout set of live episodes with no intervention oracle. Report six independent seeds and up to 75 epochs (early stopping). These methods are **NOT total-compute matched**: the fine-tuned model additionally consumed 1,200 source episodes and the original pretraining computation, while scratch did not.

| Target-only training episodes | GRU fine-tune | GRU scratch | Transformer fine-tune | Transformer scratch |
|---:|---:|---:|---:|---:|
| 4 | -0.1414 | -0.4968 | -0.3101 | -0.5079 |
| 16 | -0.0716 | -0.4922 | -0.2925 | -0.4895 |
| 64 | 0.0913 | -0.2799 | -0.1879 | -0.3267 |
| 256 | 0.7062 | 0.4901 | 0.9754 | **1.0000** |

All values are heldout mean episode returns averaged over six fresh training seeds (higher is better). At 256 target episodes, the pretrained Transformer no longer beats a newly initialized Transformer with enough target data. Pretraining assists adaptation at smaller budgets (particularly GRU) but does NOT confer a reliable general ability to immediately infer an unseen reward mechanism; both families are data-hungry relative to the privileged Bayes rule-selector comparator.

**Comparison discipline:** The oracle-like Bayes three-rule baseline in Cycle 41 scores about 0.981 with NO task-specific pretraining, but it is **given the true withheld lag7 operator** as a possible program. The unknown-rule neural learners are not. That is an inductive-prior/feature-class comparison and is not an equal-prior assertion of algorithm superiority.

**Scientific decision:** Closed as an investigatory control. Further adding lag lists, changing early stopping or finding favorable synthetic reward overlays would have declining value. Test unannounced shifts on genuine action-dependent POMDP control environments, with model criticism, replacement and strong baseline control. No algorithm proposed here is novel.

## Cycle 43: move to independent action-dependent pixel POMDP control

POPGym Arcade https://github.com/bolt-research/popgym-arcade by Wang et al. supplies fully- and partially observed variants of diverse Atari-style tasks with **common five-action interface and observation shape**. This avoids the obvious problem in Cycle 41 where the card observation process was independent of all actions, allowing an unusually easy reward-label prediction/one-step policy problem.

**Environment and provenance:**

- Created a dedicated `.venv_arcade` under the project root using Python 3.12, isolated from the main research `.venv`. Installed `popgym-arcade==0.0.6` and the published `[baselines]` extras into it; package versions are independently frozen to `research/requirements_arcade.lock.txt`.
- Downloaded reference code (temporary non-project clone) from https://github.com/bolt-research/popgym-arcade to examine its published `pqn.py`, `pqn_rnn.py`, configuration, and reproducible testing method.
- The official `popgym_arcade.train` PQN/MLP and recurrent PQN/LRU training entrypoints both passed isolated CPU smoke tests (4 parallel environments, 8 rollout steps, 128 total steps, 128×128 RGB, NoisyCartPoleEasy POMDP, cloud tracking disabled). This verifies ability to use **the authors' own controls**, not re-created weaker substitutes.
- The initial official PQN/MLP partial-observation NoisyCartPoleEasy 8,192-step CPU pilot, using 4 parallel environments, 32 rollout steps, one update epoch and four minibatches, required **125.88 seconds including compilation**. The resulting mean training completed-episode return rose from approximately 3.28 in the early quarter to 5.63 in the last quarter (under the published wrapper's raw return, which may differ from the paper's normalized return). These are **on-training trajectories, not heldout scores**, one seed, severely undertrained; no performance or novelty claims are justified.

`cycle43_arcade_official_baselines.py` calls the original authors' unmodified `make_train` for PQN and PQN_RNN, with an external standardized configuration and no cloud upload. The harness records training runtime, actual interaction budget, algorithm, memory type, early/late training completed-episode scores and (if enabled) periodic heldout evaluation. Unlike the published `single_run` wrapper, the harness need not write learned model weights to the repository. Model weights accidentally produced by smoke tests are local only and excluded from Git.

**Hardware constraint:** JAX CPU training is too slow for a fully powered benchmark on this machine. An isolated CUDA 13 JAX installation is being validated against NVIDIA driver 580.126.09 and RTX 3060 Ti (8 GB). If the GPU runtime cannot be made reliable, do not claim results from an underpowered 8k-step pilot as a substantive comparison. A valid next research step is to scale official controls to ~100k–1M environment interactions and multiple seeds/tasks, using comparable compute and appropriate validation, before introducing or praising a new algorithm.

## Cycle 44: decisive audit of early-stopping failure

Files: `cycle44_early_stop_control.py`, `cycle44_early_stop_results.json`.

The six-seed `AutoencodeMedium` raw-Transformer result in Cycle 40 was bimodal (three perfect, three near chance). Inspection showed failed seeds terminated at about epoch 11, whereas successes needed roughly 43–46 epochs. Before inferring an architecture limitation, Cycle 44 **held the data and seed streams fixed** and increased early-stopping patience from 9 to 45 non-improving epochs with a cap of 140 training epochs, rather than the original 70. This was a **post hoc stronger-baseline audit** motivated by inspecting the original results; it is not an independent preregistered scientific discovery.

Same native POPGym AutoencodeMedium, same six training/eval seeds, same 700 randomized logged reward-only training episodes and 140 live heldout episodes per seed, and the same GRU, TCN and Transformer classes/hyperparameters except the schedule.

| Architecture | Original short-patience mean return | Longer-patience mean return | New mean fitting time |
| --- | ---: | ---: | ---: |
| GRU | -0.41264 | -0.41264 | 2.87 sec |
| TCN | -0.28523 | -0.27230 | 5.31 sec |
| Causal Transformer | 0.00304 | **0.98743**, SE 0.01254 | **13.85 sec** |

The three previously failing Transformers now learned strong policies; five seeds scored essentially 1.0 and one scored approximately 0.925. This shows the old **three-out-of-six failure conclusion was substantially an early-stopping artifact**, not evidence that a Transformer intrinsically cannot learn the longer reverse-memory task. It is a more computationally expensive but stronger conventional baseline. GRU and TCN remain weak in this exact tested recipe; absence of an extensive learning-rate/hidden-size/architecture sweep means their class-level limitations are NOT established.

**Scientific calibration:** The older Cycle 40 table remains preserved because it is reproducible. Do not silently overwrite it with the more favorable tuned result. Both protocols, with compute cost and post-hoc origin, appear in this report. This test supplies an important correction against prematurely discarding a familiar method.

## Cycle 43 final: official GPU-based pixel-POMDP baseline pilot across two tasks and three seeds

Files: `cycle43_arcade_official_baselines.py`, `cycle43_arcade_aggregate.py`, `cycle43_arcade_pilot_summary.json`, twelve individual run JSON files for two tasks × two algorithms × three seeds; reproducibility: `ARCADE_BENCHMARK_PROTOCOL.md`, `requirements_arcade.lock.txt`, `arcade_pqn_rnn_diagnostic_fix.patch`.

### Actual protocol

In an isolated `.venv_arcade` with JAX CUDA13 on the RTX 3060 Ti, ran the **published POPGym Arcade package's original PQN memory-free and PQN_RNN LRU learning programs**, except for the exact one-line RNN *churn diagnostic* patch above. Everything affecting model predictions/training loss/updater/exploration is taken from the installed published package, though training config is a small exploratory setup rather than paper defaults.

For each algorithm × task × seed: 32,768 online actual environment steps in a partially observable 128×128 RGB environment with five actions; eight parallel actors, 32 rollout steps, two update epochs, four minibatches, initial seed in [43221026,43221027,43221028], same number of environment interaction opportunities, and separate rollout test evaluations of 256 steps with eight test actors at four evenly spaced checkpoints. Metric is the package's reported `test_returned_episode_returns` at the **last heldout checkpoint**. All cloud logging disabled. Test RNG streams are not guaranteed pairwise identical between algorithms and environment-step budgets are short. Hyperparameter tuning and architecture parameter counts were not matched.

**Final test metric, three independent seeds; higher is better:**

| Native POMDP task | Memory-free PQN mean ± SE | Recurrent PQN/LRU mean ± SE | PQN mean wall time | Recurrent mean wall time |
| --- | ---: | ---: | ---: | ---: |
| CountRecallEasy | 0.04291 ± 0.02148 | **0.16854 ± 0.03051** | 26.80 s | 38.77 s |
| NoisyCartPoleEasy | **0.72708 ± 0.04114** | 0.22590 ± 0.03192 | 24.49 s | 37.21 s |

CountRecallEasy individual PQN scores [0.06624,0,0.06250]; recurrent [0.14423,0.22917,0.13221]. NoisyCartPoleEasy PQN [0.77312,0.64500,0.76312]; recurrent [0.19074,0.19734,0.28963]. The means summarize **distinct training seeds**; standard errors quantify between-training-run variability (n=3, enormous uncertainty). The final-return metric comes directly from benchmark authors' test routine and may be internally normalized; do not equate it to Cycle40 POPGym -1..1 card-episode returns or official paper's full-budget SOTA values without additional validation.

**Interpretation:** The relative advantage reverses across these environments at the intentionally short interaction budget: memory helps on CountRecall, but the memory-free control is far stronger in this NoisyCartPole pilot. This is a common pattern of environment-specific memory/sample-efficiency tradeoffs, NOT evidence of universal failure of recurrent controllers; the recurring LRU may need a more appropriate learning-rate/update schedule or more interactions. These runs are one or two orders of magnitude short of a serious benchmark learning-curve comparison. Only the feasibility claim, not an algorithm ranking at convergence or a novelty claim, is scientifically warranted.

The GPU acceleration was necessary: a single authors' PQN 8,192-step run on CPU took 125.9s (plus time for script startup), whereas a GPU run of 32,768 steps took roughly 25–27s including compilation. The unpatched PQN_RNN original version failed on extended runs because its metric compared arrays of shapes (4,) and (4,36,2). This was explicitly **not** hidden: we preserved the exact patch and recorded diagnostic churn as unavailable. The authors' package's short smoke tests passed without encountering the shape error, illustrating why longer integration tests matter.

### Future benchmark prerequisites (do not call a 32k-step test decisive)

Prior to claiming any algorithmic advancement, extend to:
- At least five independently seeded fully trained learning curves per method/task and several different task families, including CountRecall, CartPole, navigation and a nontrivial game with real action-dependent transitions. Use enough interaction to reach credible performance plateaus; the published authors' methodology takes far more than 32k steps.
- Compare authors' tuned recurrent, nonrecurrent, attention/memory models, and relevant finite predictive-state/reward-machine/change-detection controls, with comparable tuning privileges, model sizes, and end-to-end computational accounting.
- Unknown-onset observation-only nuisance drift **and** reward/transition mechanism changes with heldout shift schedules, plus controls whose distinctions cannot be identified with the permitted observations.
- Test a candidate that constructs a **new task-relevant distinction** using observed data rather than selecting only among supplied program/factor templates; audit closest precedents at the level of mechanism and consequence. Evaluate true cumulative online reward (and false-positive revision, diagnostics cost), not merely final reward prediction.
- Exclude oracle-provided intervention models, manually supplied exact memory rules, and trivial detection shortcuts. If baselines explain any apparent advantage, stop optimizing and update the research conclusion.

## Final Phase VI classification

**Scientifically supported:** Strong conventional causal attention can learn important memory operators from raw observation and randomized reward alone in some third-party native POPGym environments. The apparently unstable long-memory result was mostly an early-stopping confound; correcting a strong baseline changed its mean return from ~0.003 to ~0.987 at greater compute. Conventional in-context neural learners can adapt within an episode among previously encountered reward rules, while the tested withheld rule required target-only gradient training or a privileged candidate menu. Independently, official learned control baselines run in modern POPGym Arcade pixel POMDPs on the user's GPU, and multiple task-dependent tradeoffs are observable even in early training.

**Not supported:** An autonomously constructed new representation/operator for genuinely unseen nonstationary action-dependent POMDP mechanisms, a formal new theorem, a sample-efficient broadly superior decision learner, compute-matched adoption advantage, or a new general paradigm.

**High-impact direction retained provisionally:** Develop and falsify a mechanism for *detecting when an existing representation is decision-inadequate, autonomously proposing predictive counterexamples and candidate new distinctions, and acquiring data that validates them*, with long-horizon information complementarity and explicit calibration under observational drift. It must be compared with existing reward/transition machine inference, online causal model learning, Bayesian experimental design, change-detection world models, recurrent RL and reward-predictive representations. If this is just known abstraction refinement + active experimental design with no distinctive consequence, classify it as integration and redirect. **No candidate is being designated a breakthrough.**

**Research continuation handoff:** Read `STANDING_RESEARCH_STANDARD.md`, this report, `ARCADE_BENCHMARK_PROTOCOL.md`, the actual baseline source code/results, and recent model-inference literature. Do not continue tuning native card-memory task results to manufacture novelty. The next proposal should begin with a concrete falsifiable impact/identifiability mechanism and a task whose correct representation class is not supplied. Use genuine environment interaction, cost and robust negative controls. The authors' benchmark infrastructure is prepared, but a fully trained multi-seed algorithm comparison remains unperformed and must not be implicitly claimed.
