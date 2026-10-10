# Phase VII: limits and recoverability of autonomous sensor-model repair

**Date:** 2026-10-10. **Workspace:** `/home/dtdav/dev/learn`.
**Standard:** `research/STANDING_RESEARCH_STANDARD.md`.
**Status:** Cycles 45–47 completed on independent controlled latent-action streams, Cycle 46 exact finite-history proof check passed. Cycle 48 uses real action-dependent pixel observations from POPGym Arcade to audit existing unsupervised input adaptation. **No novel general-purpose learning algorithm or breakthrough established.**

## Consequential research question

Can an agent repair an unannounced sensory-representation change by exploiting dynamics learned before the change, without requiring an expensive reference sensor or immediate task reward? Which assumptions make that possible, and when is it **information-theoretically impossible** to distinguish sensor corruption from changes to the environment itself?

This is a materially relevant choice for sensor-dependent decision systems, but prior art already includes Bayesian system identification, hidden-state relabeling symmetries, PSR/model-based RL, test-time feature alignment, domain adaptation and active costly supervision. The present study should be judged by its *precise conditions and reproducible empirical consequences*, not an invented novelty label.

### Distinctive hypotheses and falsifiers

- **H1 conditional self-calibration:** Action-conditioned predictive structure can distinguish a sensor inversion even if the unconditional sensor marginal is uninformative and unrelated nuisance observations drift. A simple action-conditioned first-moment model should be given a fair chance; pairwise models may add little.
- **H2 observational gauge:** If the agent's transition dynamics can also change, there exist distinct sensor/transition changes yielding exactly the same laws for every possible adaptive action-observation transcript yet requiring opposite optimal actions. No observation-only method can universally repair both. Exact transition conjugacy supplies a mathematical counterexample.
- **H3 additional evidence:** A small paid number of randomized chosen-action reward observations can break the ambiguity, but its net decision value depends on costs and feedback noise. Compare explicit no-adaptation, marginal-only, learned action-conditioned and pair-predictive policies, minimax random policy, and polarity oracle.
- **H4 native visual control:** An established feature-normalization adapter should be tested on actual independent action-dependent partially observable pixel-control environments; if it fails relative to strong conventional controls, stop equating observation distribution alignment with validated mechanism reconstruction.

## Close predecessors, not an exhaustive novelty audit

- Zhang et al. (2024), *Provable Representation with Efficient Planning for Partially Observable Reinforcement Learning*, https://proceedings.mlr.press/v235/zhang24bd.html , establishes learned predictive representation + efficient planning under partial observation; do not claim learning a belief representation is new.
- Zhang et al. (2019), *Learning Causal State Representations of Partially Observable Environments*, https://arxiv.org/abs/1906.10437 , learns history abstractions approximating predictive causal states and relates them to bisimulation.
- Wang et al. (2025), POPGym Arcade, https://arxiv.org/abs/2503.01450 and https://github.com/bolt-research/popgym-arcade , independent action-dependent pixel-POMDP and memory baseline suite.
- Wu, Liu and Zhang (2025), *Inferring Reward Machines and Transition Machines from POMDPs*, https://arxiv.org/abs/2508.01947 , directly targets automaton discovery from unknown dynamics and rewards.
- *Observable quotient world models for knowledge-state abstraction under partial observability* (2026), https://www.sciencedirect.com/science/article/pii/S0950705126014942 , appears to address the same fundamental observable-quotient modeling question, including nuisance, sensing rights and observational distinction. A recent, close predecessor; only its public summary/abstract was inspected, so **do not assert substantive identity or difference without full-text analysis**.
- *Reusable Consequence States Under Partial Support and Model Uncertainty* (2026 preprint), https://kadubon.github.io/github.io/papers/2026-08-30-reusable-consequence-states-under-partial-support-and-model-uncertainty-22170023/ , further cautions against making broad claims about new reusable, uncertainty-aware task representation principles.
- Standard two-state HMM gauge symmetries, controlled state-space system identification, change detection, and Bayesian value of information already explain the mathematics of Cycles 45–47. No independent theorem claim.

## Cycle 45: learning action-conditioned sensor-calibration signatures

Code: `cycle45_sensor_self_calibration.py`. Data: `cycle45_self_calibration_results.json`.

Independent controlled Markov two-state POMDP: hidden binary state X, observed noisy sensor S with symmetric bit error epsilon=0.12, chosen action a in {0,1} changes the probabilities of X flipping. Actions are randomized during source training and diagnostic episodes. Another observable nuisance bit N follows an independent Bernoulli process, p=.2 initially and p=.8 under a separate nuisance-only drift.

In the **asymmetric** source transition class, action a=0 has P(0→1)=.08 and P(1→0)=.28, while action a=1 reverses the two rates. Uniform random actions keep the *unconditional sensor marginal* at ~0.5, but action-conditioned S_next probabilities and adjacent sensor-pair distributions are orientation-sensitive. In the **symmetric** class, each action has equal zero-to-one and one-to-zero switching rates, leaving both simple sensor orientation signatures invariant to a global bit flip.

The learner fits (1) unconditional sensor mean, (2) action-conditioned next-sensor mean, and (3) adjacent sensor-pair conditional distribution from **7,000 prechange randomized steps**, with additive smoothing. The source action/reward reveals the source action-sensor convention: training rewards are given for the randomized chosen action, so recovered source hidden labels are deterministically implied by observed source chosen action + reward; no hidden state label is independently supplied. This source reward source does NOT apply during postchange diagnosis.

The postchange stage supplies **64 observed sensor/action diagnostic steps** with randomized actions, **no intermediate reward feedback**, and then a *single terminal decision reward* for choosing the correct current hidden bit. All sensor-only candidates use these same diagnostic observations. A separately labeled 3-reward-query comparator receives three additional chosen-action reward readouts, an extra privilege not available to the other candidates. A privileged oracle knows the true sensor orientation. The learner's permitted repair family is **explicitly only source orientation vs global binary inversion**; the general transformation mechanism is NOT invented.

256 independent **source-model seeds** × 32 postchange episodes per condition, eight conditions. Means below are terminal accuracy / probability of correct action; reported SE for each mean about 0.003–0.006 across 256 independent source seeds.

| Test change | Frozen original sensor | Unconditional marginal | Learned action mean | Learned pair predictor | 3 reward readouts (extra right) | Polarity oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Asymmetric, no shift | .883 | .505 | .790 | .798 | .853 | .883 |
| Asymmetric, sensor inverted | .129 | .493 | **.795** | **.803** | .845 | .871 |
| Asymmetric, nuisance-only drift | .885 | .498 | .791 | .802 | .853 | .885 |
| Asymmetric, sensor inverted + nuisance drift | .120 | .498 | **.788** | **.797** | .848 | .880 |
| Symmetric, no shift | .874 | .500 | .493 | .498 | .850 | .874 |
| Symmetric, sensor inverted | .125 | .487 | .494 | .494 | .850 | .875 |
| **Action dynamics swapped, no sensor inversion** | **.879** | .505 | **.209** | **.198** | .850 | .879 |
| Action dynamics swapped + sensor inverted | .115 | .496 | .201 | .186 | .855 | .885 |

**Conditional positive:** Action-conditioned signatures can repair an unexpected inversion using observation-only temporal control statistics, even when the sensor marginal is uninformative. Nuisance-only drift does not degrade these sensor-only scores. However, they have nontrivial false inversion errors even in unchanged regimes, and the no-adaptation policy wins there.

**Decisive failure:** When action semantics shift correspondingly, the exact same repair confidently chooses the WRONG polarity. The pair model offers little advantage over the simplest action-conditioned mean; a complex representation is not necessary for the positive results.

**Interpretation:** These are small researcher-designed Markov systems with a supplied inversion option and delayed terminal reward, not a native external environment or a new general model critic.

## Cycle 46: an exact, action-dependent observational non-identifiability certificate

Code: `cycle46_observational_gauge.py`. Result: `cycle46_action_observation_equivalence.json`.

Let latent X_t ∈ {0,1}. Let original action-indexed hidden transition matrix be T_a. Define the label-complement permutation

```
C = [[0,1],[1,0]],
T'_a = C @ T_a @ C.
```

With original flip rates T_0=(.08,.28), T_1=(.28,.08), this conjugacy *is exactly the transition response of swapping action-conditioned transition rows*: T'_0=T_1 and T'_1=T_0.

Consider two possible new mechanisms, both with uniform initial state and 12% independent symmetric observation flip noise:

- World A: hidden dynamics T_a, **globally inverted** sensor S_t = X_t XOR 1 XOR noise.
- World B: dynamics T'_a, normal sensor S_t = X'_t XOR noise with X'_t = 1 - X_t.

For any action trajectory, couple X'_0=1-X_0 and give both worlds identical action sequence and emission noises. Then by T'=CTC, X'_t=1-X_t throughout the coupled trajectory, and the observed S_t are **identical for every t**. Since the agent's next actions depend only on observations/actions available to it, the coupling applies by induction even to arbitrary randomized adaptive policies. It is not restricted to uniformly randomized action probes.

But terminal reward in world A is R_A(a)=1[a=X_t], while in world B it is R_B(a)=1[a=X'_t]. Consequently, **R_A(a)+R_B(a)=1 for any shared terminal action**. No observation-only learner, no matter how computationally powerful or how long it observes, can achieve better than **0.5 expected terminal correctness in BOTH worlds simultaneously**. A shared strategy may do very well in one and poorly in the other, exactly as Cycle 45 observed.

The exact arithmetic verification checked **10,922 action-observation strings**, covering ALL binary action strings paired with ALL binary observation strings for history lengths 1 through 7. Every paired world had the **same likelihood to machine precision (largest difference 0)**, and the transition conjugacy identities passed. This finite check supports, but does not replace, the induction argument for arbitrary horizons.

**Critical restrictions:** The result assumes two binary state labels, a complement-symmetric prior, symmetric emissions, a model family admitting both the specified sensor and transition changes, no distinguishing reference measurement and no reward feedback during diagnosis. If the transition process is known not to have changed, the World B explanation is excluded and self-calibration can be possible. If chosen-action reward is observed during diagnosis, the observation histories are not equivalent and the impossibility does not apply.

This is a classical latent-state permutation/gauge identifiability argument specialized to an unknown sensor-versus-dynamics change with opposite decision labels, **not a new mathematical theorem**. Its practical consequence is explicit: *the right observation model cannot generally be identified without declaring stable physical semantics or obtaining reward/reference evidence.*

## Cycle 47: what extra reward feedback buys after correcting for cost

Code: `cycle47_reward_calibration_value.py`, result `cycle47_reward_calibration_results.json`.

128 independent source-training seeds, 40 distinct test episodes per seed in each of four conditions, 64 postchange randomized, rewardless sensory diagnostic steps per episode, and one terminal decision. Sources receive 8,000 prechange interactions to estimate an action-conditioned sensor model. Compare:

- Frozen source sensor orientation (zero postchange reward query cost).
- Learned action-conditioned next-sensor model and learned adjacent-pair model (no query cost).
- A minimax-oriented random terminal choice, defensible when two equally plausible but decision-opposite worlds are observation-equivalent.
- Extra **1, 3 or 5 chosen-action reward queries** from available diagnostic steps, each costing 0.025 terminal-reward units. Each query reveals only the outcome of the action actually chosen, not a hidden state label or counterfactual; the binary reward and known action may together logically determine the state in this deliberately simple family.
- Privileged oracle orientation (no query charge; impossible to deploy without a declared source of truth).

**Terminal reward minus explicit additional reward-query fees; mean±SE over independent sources:**

| Actual mechanism | Frozen | Learned action mean | Learned pair | Minimax random | 1 reward query | 3 reward queries | 5 reward queries | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sensor inverted only | .126 | .792 | .804 | .494 | .764 | **.776** | .739 | .874 |
| Transition actions swapped only | **.879** | .207 | .199 | .486 | .766 | **.774** | .742 | .879 |
| Neither shifted | **.876** | .792 | .802 | .504 | .760 | .768 | .739 | .876 |
| Dynamics swapped AND sensor inverted | .119 | .205 | .195 | .485 | .764 | **.775** | .746 | .881 |

All standard errors approximately .004–.007. The three-reward strategy is robust across the two observationally ambiguous changes, although it needlessly pays query costs under no change. Under the chosen fee 0.025 per query, five queries are inferior to three, which demonstrates **meaningful cost/accuracy trade-offs**. The fixed-count query strategies are a conventional baseline, not an autonomously synthesized query protocol or an optimal Bayes decision policy.

**Important identification of the real contribution:** This illustrates which *additional observations* can restore useful decisions when no sensor-only repair is universally possible. It does not claim the additional channel was discovered or created by the agent, nor that its fixed querying schedule is novel. The magnitude depends on the constructed deterministic binary reward and assumed query fee; it does not transfer automatically to delayed/noisy reward in live control.

## Research conclusion after Cycles 45–47

- **Conditional success:** Learned action-conditioned observational dynamics can serve as a no-reward sensor reference when the controlled transition mechanism is stable, even when marginal appearance provides no calibration signal and an independent nuisance channel drifts.
- **Exact impossibility:** The right transformation can become observationally underidentified when physical dynamics and sensor mapping may change together. More observation-only training cannot resolve the gauge; a reward or reference query is a genuinely new information source.
- **Decision value of evidence:** Three paid outcome observations gave strong conditional net utility across the confounded cases, while zero-query calibration has catastrophic false repairs in one of them.
- **Novelty boundary:** The exact effect is understood through existing HMM label non-identifiability, controlled system identification, Bayes experimental design and information theory. A proposal must transcend supplied two-state/inversion structural classes and beat strong prior methods in consequential applications to qualify as original.

**Stopping decision:** Stop optimizing the manually constructed binary orientation model now that its exact equivalence and costed rescue are established. Only an independent external action-dependent benchmark with raw observations and strong controls can justify continuing this representation-repair line. Cycle 48 tests that, but it is a standard *unlabeled pixel normalization* baseline, not a proposed novel solver.

## Cycle 48: native pixel POMDP sensor corruption and conventional per-channel normalization

Code: `cycle48_arcade_observation_adapter.py`; numeric JSON: `cycle48_pixel_adapter_results.json`. Smoke: `cycle48_smoke.json`.

**External environment and base controller:** POPGym Arcade 0.0.6 `NoisyCartPoleEasy`, actual native action-dependent CartPole dynamics, partial 128×128 RGB observations. The base policy is the published package's PQN/Multi-Layer-Perceptron control learner, **not a proposed algorithm**. Three independent seeds, each trained on exactly 65,536 environment interactions with the publisher's Q learning algorithm (2 update epochs, 8 parallel actors, 32 steps/update, four minibatches); GPU JAX CUDA13, RTX 3060 Ti. Source control is then frozen.

**What changes:** Only the image returned to the controller is transformed at evaluation time. CartPole dynamics and native reward functions remain unchanged. The shift family was constructed by the researcher: no shift; all RGB intensities inverted, x→255−x; brightness +48 with clipping; channel permutation (R,G,B)→(B,R,G); zero out blue channel. These artificial corruptions are not claimed to be native nonstationary POPGym Arcade environments or naturally arising hidden changes.

**Unlabeled calibration:** Each source model receives 512 random-action, rewardless source RGB calibration images and 512 separate random-action, rewardless target-domain images per shift. The ordinary baseline computes per-channel means and standard deviations from *unpaired* images, and applies an affine, positive-sign normalization to shifted test images. This is the established AdaBN / marginal moment-matching baseline, not a new learner. Strong comparisons: frozen source model without repair; a privileged oracle knowing the true shift-specific inverse for invertible transforms. For erased blue the supposed oracle cannot invert the shift and therefore intentionally leaves it unchanged; it is NOT a full-information clairvoyant control upper bound.

**Strictly heldout control:** 32 parallel independent test actors, 512 steps each per policy and shift. The reported return is the average *completed episode* `returned_episode_returns` from the benchmark's own logging wrapper, averaged with equal weight over **three independent source-model seeds**. Test resets share seeds but policies take different actions, thus only approximate paired experimental conditions. Final episode count differs drastically when a policy fails quickly. **The 512+512 calibration observations are additional interaction costs**, not included in training steps, and reward readouts were explicitly withheld.

| Artificial visual corruption | Frozen | Positive-scale moments | Privileged known inverse |
| --- | ---: | ---: | ---: |
| Identity | 0.836 | 0.835 | 0.836 |
| RGB inversion | 0.047 | 0.205 | 0.836 |
| Brightening +48 | 0.312 | **0.842** | 0.842 |
| RGB permutation | 0.099 | 0.213 | 0.836 |
| Erase blue | 0.120 | 0.129 | 0.120 (no valid inverse) |

**Conclusion:** Ordinary per-channel moment alignment succeeds on pure brightness shift but is not a general representation repair algorithm. It cannot infer the sign of contrast or rearrangement of semantically distinct channels from marginal mean/std alone. A group oracle recovers performance with access to shift semantics. Missing-channel information is not bijectively recoverable through this class.

**Cost warning:** Full wall clock was 240.1 sec for three seeds and all evaluation sweeps (including JAX compilation). A significant share was test evaluation, and user-provided GPU was used. This is a proof of comparative behavior on a single task family, not a compute-matched adoption study or significant multi-task performance claim.

## Cycle 49: recover invertible palette changes without the shift label

Code `cycle49_palette_alignment.py`; results `cycle49_palette_alignment_results.json`; smoke `cycle49_smoke.json`.

To challenge the weak baseline, we built a **stronger existing-class alignment method**, NOT an unrestricted novel representation learner. It is given a researcher-defined family of `3!×2³=48` signed RGB channel permutations plus continuous per-channel mean/std affine parameters. It is **NOT given the identity of the actual shift**. It matches target to source through average image *spatial templates* and per-channel normalized third central moments (skewness), evaluating all 48 candidates under a fixed discrepancy:

```text
loss = mean( (source standardized spatial mean
             - signed/permuted target standardized spatial mean)^2 )
     + 0.25 * mean( source skewness - signed/permuted target skewness )^2.
```

No reward/correct-action/simulator latent state, shift flag or true transformation is consulted by the selector. The spatial coordinate system, full prechange image palette, paired per-channel affine family and finite search alphabet are **explicit privileged structural priors**. The method performs test-time preprocessing, not controller weight updates. It has 512 source and 512 target random-action, rewardless calibration images, as Cycle 48, and 32 actors×512 heldout steps per method/shift.

**Three new model-training seeds (distinct from Cycle 48), all 65,536 control training interactions each:**

| Sensor shift | Frozen PQN | Positive-scale moments | **Unlabeled spatial/signed alignment** | Privileged true inverse |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.864 | 0.865 | 0.865 | 0.864 |
| Invert RGB | 0.143 | 0.152 | **0.866** | 0.864 |
| Brighten +48 | 0.089 | 0.851 | **0.851** | 0.845 |
| Permute RGB channels | 0.590 | 0.457 | **0.863** | 0.864 |
| Erase blue | 0.154 | 0.158 | **0.123 (worse)** | 0.154 (cannot invert) |

In all **three independent seeds per recoverable shift** it selected identity under no shift, all three negative contrast signs under RGB inversion, identity with positive signs under brightness and inverse channel permutation [1,2,0] for the [2,0,1] shift. In all three *erase blue* cases, the learner forced a false transform [0,2,1] with signs [1,-1,1], often harming reward.

**Interpretation:** Recoverability is conditional on the shift belonging to the supplied invertible family. Within that family, unlabeled structure is sufficient for near-oracle recovery under this task's stable RGB spatial distribution. This is a useful engineering capability and controls against the weak-moment straw man, but image standardization, distribution matching, signed permutation search and test-time adaptation have many existing predecessors; **no novel general learning algorithm is established**. Results are only three source-model seeds on one synthetic visual-corruption suite over real benchmark dynamics, without unknown change times or physical camera hardware validation. The experiment does **not** demonstrate autonomous creation of a transformation outside its fixed grammar.

The total 3-seed wall time was 284.1 seconds including GPU model training, JAX compilation and all heldout evaluation. The adapter searches 48 candidate transformations and carries full source spatial templates and three moments; its calibration is not literally zero-cost.

## Cycle 51: abstention from unsupported reconstruction

Code: `cycle51_open_set_rejection.py`; result `cycle51_open_set_rejection_results.json`.

The signed transformation search gives a fit residual. For identity and all three invertible shifts across the 3 Cycle 49 training seeds, the best-fit residual lay between 0.0006 and 0.0164. For erased blue, it lay between **2.4058 and 2.43**. In this particular image domain, the algorithm therefore has a very clear signal of **out-of-class sensory corruption**. Its open-set confidence can be treated as a gate: accept repair only when residual ≤ fixed threshold; otherwise retain original input and flag that new information/modeling is required.

We swept **eight thresholds** in {0.005,0.01,0.02,0.05,0.1,0.25,1.0,3.0} on the ALREADY INSPECTED three seeds and reused the frozen policy's recorded heldout returns. At each threshold from 0.02 to 1.0, the gate accepted all reversible shifts and the identity, rejected erased blue in all three source seeds, and avoided the harmful forced correction (raw erased-blue return 0.154 versus false repair 0.123). At threshold 3.0 it accepted erasure and degraded.

**No independent heldout validation:** This is openly **post-hoc exploratory threshold analysis** on the same outcomes. It is NOT a novel rejection algorithm, not proof of calibration in a new task family, and not evidence that all irreversible failures produce a large residual. The source spatial templates, transform grammar, loss scale and threshold privilege require careful declaration. A fresh sealed task-family test and measurement-cost analysis are prerequisites for any consequential reliability claim.

*Operational implication:* A representation repair method should be able to say not only which available explanation best matches observations, but whether **any** allowed explanation matches well enough to deploy. This is the established open-set recognition/model criticism principle, now grounded in a real action-dependent POMDP control experiment. It is especially important for shifts that require a qualitatively different reconstruction, rather than forcing a visually plausible but wrong representation.

## Cycle 50: missing-channel recovery using source color redundancy

Code: `cycle50_missing_sensor_imputation.py`. Results: `cycle50_missing_sensor_results.json`; smoke saved independently.

**This changes the interpretation of an earlier negative control:** Zeroing a channel is not a globally invertible transformation on the space of arbitrary RGB pixels, but a missing channel can nevertheless be **predictable on the actual source-image support**. If the task renderer uses a small palette whose projection onto two surviving channels uniquely determines the third, the erased color can be reconstructed from a source-trained conditional model without obtaining any postshift reward. That is *source redundancy*, not violation of information theory.

We tested this distinction on a frozen **native** NoisyCartPoleEasy POMDP policy trained using the benchmark authors' PQN implementation. Three independent 65,536-interaction training seeds, 32 live evaluation actors ×512 steps each and unmodified dynamics/rewards. **Source training resources:** for each model, 512 independent rewardless source-domain random-action RGB images were collected to estimate a simple global color model; 128 independent rewardless target-domain images diagnosed which RGB channel became constant. These are extra data/interaction costs, explicitly uncounted in the original 65,536 policy-training steps. No correct action, counterfactual image or target episode reward is passed to the conditional imputer. The original RGB policy weights remain frozen.

A source-trained model predicts the missing color from the other two using either (1) a global mean, (2) ordinary linear regression, or (3) a 65,536-entry discretized conditional RGB palette lookup `E[channel | surviving_channel_1, surviving_channel_2]`, with smoothing. The source palette contains **only nine observed distinct two-channel color pairs for every missing-channel identity** in each training seed. This is a *highly constrained renderer*; the lookup's stunning effectiveness is explained by its narrow support, not an exotic new structural ability.

Heldout mean completed-episode returns across three independent trained source policies (higher better):

| Actual erased channel | No reconstruction | Global mean fill | Linear regression | Source conditional color LUT | Privileged intact RGB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Red | .081 | .367 | .449 | **.847** | .849 |
| Green | .257 | .099 | .657 | **.849** | .849 |
| Blue | .115 | .103 | .626 | **.846** | .849 |

No-shift identity: original and unrepaired return .849. The missing-channel detector found the correct erased channel in each trial because the channel was **zeroed exactly**, yielding near-zero empirical variance, while intact channels had appreciable variance. This is an easy supplied corruption/detection prior, not generic autonomously created failure diagnosis.

**Scientific correction:** Earlier reports saying channel erasure is “irrecoverable” must be read as “there is no global pixelwise inverse without source/support assumptions.” Actual decisions CAN be restored when source colors are redundant. The exact feasibility condition depends on the conditional distribution `P(missing channel | remaining observable channels)` and, more weakly, whether uncertainty changes the policy's chosen action. Under squared-error reconstruction loss, the minimum achievable MSE from the surviving channels is `E[Var(missing | surviving)]`, a standard conditional expectation identity, **not a new theorem**. Decision identification can remain possible even when pixel reconstruction is imperfect if all source-consistent completions agree on the optimal action.

**External validity limitation:** These are computer-generated pixel sprites with an **extremely small, nine-combination color palette** and one task's rendering convention. They are not complex natural images or real damaged cameras. They give evidence for a practical emergency imputer on a discrete-color renderer, not a general capability to infer missing sensory modalities or novel world representations.

**Closest concrete prior art:** Huang, Su, Yang, Lin & Xu (2025), *Mitigating Missing Feature Channels at Inference Stage: Test-Time Adaptation Through Self-Training With Data Imputation*, IEEE Signal Processing Letters 32, 2414–2418, DOI https://doi.org/10.1109/LSP.2025.3542696 , directly covers imputation during test-time sensor-channel failure. Skand, Pandit, Kim, Fuxin & Lee (2025), *Simple Masked Training Strategies Yield Control Policies That Are Robust to Sensor Failure*, https://proceedings.mlr.press/v270/skand25a.html , is a stronger robotics/control precedent. Existing test-time domain alignment and source-based multimodal completion further constrain claims of originality.

**Decision:** STOP tuning this nine-color environment as evidence of novel sensor-representation discovery. It is useful as a validation and diagnostic fixture. Cycle 53 audits color support on heterogeneous external tasks to select genuinely challenging settings; Cycle 52 tests a strong established *mask-during-training* alternative to retrofitting.

## Cycle 52: proactive random sensor-dropout training is a strong existing alternative

Code: `cycle52_masked_policy_control.py`, results `cycle52_masked_policy_results.json`. The masked-training implementation uses the **same official PQN controller architecture and Q-learning update rule** as Cycles 48–50, but a wrapper independently chooses at each episode reset whether the observation is intact or has its R/G/B channel zeroed (each option 25% probability). It holds this missing-channel choice for the episode. The Q learner sees that corrupted observation with the native reward while learning. No test-time imputation, source-color lookup, shifted-environment reward query, or policy retraining at deployment is used.

**Fairness/privilege distinction:** All methods use **65,536 actual control training steps per seed**, and the control test uses 32 actors ×512 steps. But proactive masked training has a *strong prior*: it is told during training that any of the three RGB channels can fail and sees examples of each failure throughout reinforcement learning. The retrofit method in Cycle 50 learns only from uncorrupted observations and **receives 512 additional source observations plus 128 test-time diagnosis observations** for a postfailure imputer. These are materially different training and deployment rights; the methods are not equal-privilege or fully equal-compute. The masked baseline does not exactly reproduce the authors' 2025 Skand et al. multimodal robust encoder, so it cannot be described as the full published strongest method.

Three independently seeded training runs, mean completed-episode return:

| Condition | Proactive random channel-masking training | Retrofit source conditional color LUT (Cycle 50) |
| --- | ---: | ---: |
| Sensor intact | 0.839 | 0.849 |
| Red missing | 0.837 | 0.847 |
| Green missing | 0.857 | 0.849 |
| Blue missing | 0.844 | 0.846 |

Between-run SEs for proactive masked training approximately .0063, .0130, .0102, .0069, respectively.

**Scientific result:** Simple, established sensor-dropout augmentation is already sufficient to maintain high control reward on every tested channel failure in this narrow rendered environment, at exactly the base policy training interaction budget. It matches the frozen-policy retrofit method within these small-sample scores. Neither is novel. The useful practical choice is **anticipatory robustness versus after-failure repairs when a frozen controller cannot conveniently be retrained**, conditional on rights and costs.

## Cycle 53: falsifying generality across independent renderers

Code: `cycle53_palette_support_audit.py`, results `cycle53_palette_support_audit.json`.

The conditional LUTs in Cycle 50 encountered only **nine distinct RGB colors**, explaining why missing one channel can be reconstructed by an ordinary nine-entry discrete palette mapping. Before claiming any broader sensor-recovery capability, we sampled **independent source and heldout random-action observation trajectories from eight distinct, third-party POPGym Arcade environments**. Each task sampled four source and four test actors, 16 timesteps each, with 128×128 RGB pixels. The audit calculates unique *source* RGB colors and heldout normalized color-reconstruction RMSE from the two surviving channels. It does not supply reward feedback or evaluate a new control policy.

| POPGym Arcade task | Distinct observed RGB colors | Relative heldout reconstruction RMSE, missing R/G/B |
| --- | ---: | --- |
| NoisyCartPoleEasy | **9** | .0013 / .0013 / .0008 |
| NavigatorEasy | 8 | .1437 / .0736 / .3531 |
| BattleShipEasy | 8 | .0243 / .0129 / .0008 |
| CountRecallEasy | 7 | .2259 / .0003 / .0004 |
| MineSweeperEasy | 7 | .0001 / .0001 / .0001 |
| SkittlesEasy | 11 | .1559 / .2189 / .5400 |
| TetrisEasy | 11 | .0683 / .0003 / .3565 |
| BreakoutEasy | 10 | .0001 / .0000 / .0000 |

**Decisive outcome:** Every one of these eight observed source environments uses **only 7–11 distinct RGB colors** in the sampled random-action histories. This is not the whole POPGym Arcade catalogue and says nothing about as-yet-unvisited states, but it demonstrates that the current sprite-renderer sensor-correction task is unusually low-entropy and often easily solved by elementary discrete palette lookup. Some tasks have ambiguity even in their tiny palettes (e.g., Skittles blue conditional RMSE 0.54), so missing-channel recovery is NOT universally perfect.

**Branch stopping decision, reinforced:** Do not spend additional GPU compute tuning RGB color lookup, channel permutation recovery or masked training on these constrained palettes to manufacture novelty. We have a working comparative and negative-control fixture. The high-upside unsolved target requires richer continuous sensing, true hidden-model discovery, intervention rights not supplied by a tiny finite grammar, robust unknown onset and a strong established control-learning comparison. We should move to realistic or procedurally rich pixel/physical observations and fully independent task families.

## Phase VII final calibration, standing-standard assessment

**Useful evidence established**

1. Observation-only repair can be powerful when stable action-dependent dynamics and a known candidate transformation group identify the sensor orientation. The simplest action-conditional mean already explains much of the gain.
2. With unknown sensor and transition changes, we found a **verified exact observational equivalence with opposite terminal actions**. The right recovery decision is mathematically impossible from allowed observation/actions alone; additional outcome feedback or a declared physics constraint is necessary. This is a known identifiability principle, not a novel theorem.
3. On native external pixel-control tasks, strong per-channel moment matching repairs brightness, while **unlabeled, researcher-class-constrained signed spatial alignment** recovers inverse contrast and channel permutations with near-oracle results. But a rigorous out-of-class residual gate is needed before deploying a forced candidate; its reported 3-seed threshold sweep was post hoc, not independently validated.
4. Missing channels can be reconstructed on highly redundant image supports even though raw pixel transforms are not globally invertible. Source-trained conditional color lookup restored near-original NoisyCartPole control; a strong standard *mask during training* alternative independently matched its performance. This is not a new learning mechanism.
5. **Every audited renderer sampled contained only 7–11 RGB colors**, exposing a major narrow-benchmark shortcut and preventing overgeneralization to realistic images or camera failures.

**What is NOT demonstrated**

- No invented unrestricted sensor or dynamics transformation outside a researcher-given grammar.
- No autonomous new diagnostic action/measurement right, fully general identifiable-model discovery or novel theorem.
- No externally validated safe online calibration with unknown change time, calibrator validity drift, natural camera images, natural sensor failure mechanisms or realistic hardware costs.
- No matched published state-of-the-art test-time adaptation comparison against full SATA, Skand et al., missing-channel data-imputation or modern domain generalization on high-dimensional naturalistic visual tasks.
- No robust multi-task advantage of a new mechanism over established Bayesian active experiment design, reward machines, causal-state/PSR representations, pretrained robust RL, existing masked training or test-time imputation.

**Novelty classification:** Correct conditional evidence, decisive counterexample, strong negative controls and an operationally useful demonstration of support-limited recoverability. **Not** a new algorithm, new field or breakthrough.

**Research decision:** CLOSE the nine-color sprite restoration branch after this evidence. Open a distinct high-upside investigation on realistic or procedurally rich continuous visual/physical partial observability, first establishing that the rendering/support, class changes and evidence rights actually require more than a finite palette or supplied inverse group. Any new method must show an independent capability or meaningful adoption trade-off against strong proper methods, including all data, diagnostic, retraining, compute and failure costs.

**Reproducibility:** All Cycle 45–47 simulation computations and 48–53 pixel-control computations ran on the user's connected Linux PC, not the assistant container. Source code and aggregate JSON outputs are versioned; large raw image trajectories and isolated virtual environments are not. GPU experiments used the isolated `.venv_arcade` JAX 0.11.2 CUDA13 environment on RTX 3060 Ti, benchmark package `popgym-arcade==0.0.6` and an official PQN learning kernel without weight-update modifications. The original `research/requirements_arcade.lock.txt` and Phase VI protocols document reproduction. The published PQN_RNN churn diagnostic patch from Phase VI does not apply to these memory-free PQN visual repair experiments. Every external sensor corruption is implemented by our scripts, **not claimed to be a native benchmark environmental change**.
