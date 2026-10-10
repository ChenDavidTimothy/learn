# Phase III research audit and independent tests (2026-10-10)

**Location:** `/home/dtdav/dev/learn/research`
**Standing standard:** `STANDING_RESEARCH_STANDARD.md`
**Prior work:** `REPORT_2026-10-10.md`, `REPORT_CONTINUATION_2026-10-10.md`, `REPORT_PHASE2_2026-10-10.md`.
**Status:** A stronger conventional baseline was implemented and evaluated; the original decision-loss divergence was independently replicated. No new general learning principle has been established.

## Research decision and falsifiable questions

The prior research emphasized an 84% action-regret reduction from a targeted interaction representation relative to a learned two-state HMM, even though the HMM had better predictive log loss. This is a **real comparison but asymmetric inductive assumptions**: a two-state target-only model cannot represent effects mediated by an unobserved nuisance bit.

The new study tests two competing explanations:

- H1 (general decision-relevance advantage): the interaction learner exploits a distinction that a conventional learned latent-state learner with matched observations cannot adequately acquire.
- H2 (correct model-class advantage): the poor HMM policy results from its missing coupling. A learned two-factor generative HMM, without being told which nuisance matters or the true parameters, recovers comparable policies.

H2 receives stronger support if the expanded latent model reliably closes the policy gap, also behaves sensibly under null shifts, and does so without using test labels. A positive result does NOT imply automatic invention of a new representation: the expanded class is provided by the investigator.

## Source audit, old experiments, limitations

Inspected all three period reports and README, environments, code implementing Cycles 01, 02, 03, 04, 08, 09, 12, 13, 15, 16, 17, 19, 20, with function-level scrutiny of the decisive data generator, oracle, HMM, interaction features, and policy evaluation. This is **not** a line-by-line independent audit of all approximately 4,000 lines or a rerun of all old trials.

Key verified distinctions:

- Cycle 01 knows exact prechange truth tables, priors, and structure. Bayesian mixture controls explain much of the apparent advantage.
- Cycles 02-04 rely on finite researcher-given grammars. The parity barrier is a known Fourier/Jensen result; anytime-valid likelihood ratios use established inference.
- Cycles 05-11 document advantages erased by stronger prior baselines and the failure of compact shift fingerprints.
- Cycles 12-20 draw independently reset three-step partially observed episodes; they are not continuous trajectories, and training uses randomized actions.
- The target sensor index is estimated from *prechange training labels*. The partner nuisance index is NOT provided to the methods.
- The cycle15 feature map `targeted(...)` manually includes target-by-action-by-nuisance interaction terms. Its key family was not autonomously invented.
- The two-state HMM in `cycle13_latent_hmm.py` ignores all nuisance observations by construction and therefore cannot represent action-conditioned nuisance dependence.
- The exact oracle in `cycle13_oracle.py` is used solely for evaluation, not for training or feature selection.
- Cycle 19's confidence threshold is pointwise; repeated overlapping adaptive checkpoints are not covered by an anytime-valid guarantee.
- The previous neural MLP and boosted tree results cover one chosen training recipe each, not optimized algorithm families.

## Cycle 21: stronger learned two-factor latent-state baseline

Files: `cycle21_latent_joint.py`, `cycle21_validation.py`.
Data: 960 prechange + 8,192 postchange supervised episodes, seven nuisance factors, 3,500 independent test episodes; each method sees the same training examples and selected target index. Four conditions, 48 entirely new seeds per condition, seed start 91921026, stride 917.

The new method assumes a class of four latent configurations `(target_bit, one_nuisance_bit)` with unknown binary-emission probabilities, unknown nuisance transition rate, and four separate target flip rates conditional on previous action and nuisance bit. The initial latent bits are assumed independent uniform. This is a **researcher-specified, highly favorable family** for this task; it is not discovered from raw history without a prior.

For each of seven possible nuisance observation channels, fit weighted conditional-label likelihood using bounded L-BFGS-B with exact PyTorch autodiff gradients and two starts. Select the candidate on an independent 20% slice of postchange training examples. Refit the selected candidate to all original training examples. Neither true partner index, true transition rates, test labels, nor oracle outcomes enter fit/selection. The metric `correct_nuisance_selected` evaluates with true identity only AFTER training. The full training and evaluation time is recorded.

The original two-state HMM and original targeted logistic learner are refit to precisely the same training data with their previous code and hyperparameters. All have the same test examples and metric definition. They are **NOT compute-matched**: the new method tests more candidates and is much slower.

`cycle21_validation.py` substitutes the true generator parameters **only into a standalone analytic consistency test** and verifies exact agreement of the four-state forecast with `prob_oracle` in all four regimes (maximum absolute error <= 2.3e-16). This is a numerical check of the model formula, not proof that training reliably learns it.

### 48-seed results

Mean one-step expected policy regret (lower is better):

| Regime | Restricted 2-state HMM | Targeted interaction | Learned 4-state joint |
|---|---:|---:|---:|
| stationary | 0.000000 | 0.006649 | 0.000043 |
| nuisance shift | 0.000000 | 0.006321 | 0.000000 |
| contextual action coupling | 0.031636 | 0.004867 | 0.003294 |
| target transition shift | 0.000000 | 0.007344 | 0.001419 |

Paired difference, learned joint minus targeted, coupling regime: -0.001573, SE 0.001211 (48 paired independent seeds). Thus the 48-seed experiment does not establish significantly lower regret for the joint model under coupling. It does show a conventional larger latent class can close almost all of the restricted HMM's deficit.

Coupled regime predictive log loss: restricted HMM 0.658168, targeted 0.658927, learned joint **0.654948**. The expanded latent model is better on both observed loss and regret *in this restricted data generator*. In stationary/nuisance regimes the restricted HMM remains very strong and more efficient.

The joint model's correct partner-channel selection frequency is 44/48 (91.7%) under action coupling, but about chance under null conditions, where the partner is not decision-relevant. This should not be taken as proof of representation invention: a fixed class of exactly one nuisance partner was searched.

Mean coupling fitting/evaluation times: restricted HMM 0.0567s, targeted interaction 0.0218s, joint latent 1.5671s per trial on this PC. The joint model is approximately 72x slower than the targeted method, and its computational cost is a serious adoption trade-off. No energy measurements.

**Independently frozen replication:** `cycle21_independent_contextual.json`, 120 further fresh seeds (start 97111026), independently sampled, same model family and fixed hyperparameters. In coupling, restricted HMM regret 0.031757, targeted interaction 0.004877, learned joint latent **0.003188**. Joint minus targeted paired difference -0.001689, SE 0.000731, 95% Student-t interval **[-0.003137, -0.000241]**. This supports a conditional advantage of the joint model in decision regret on this generator. Joint latent predictive log loss 0.655824, compared with 0.659007 (restricted HMM) and 0.660040 (targeted). Partner selection succeeded in 110/120 trials; mean fitting time 1.527s. It is **NOT** evidence of a new learning paradigm, compute-matched advantage, or general superiority: its structural prior includes exactly the form of the coupling, and it has far higher fitting cost.

## Cycle 22: diagnostic replication and explanatory inequality

Files: `cycle22_policy_loss_diagnostics.py`, `cycle22_policy_loss_results.json`.
Protocol: 120 further independent seeds per regime, start 94921026, disjoint from earlier reported seeds and Cycle 21. All models/hyperparameters unchanged. Measures observed label loss, true conditional KL excess loss, action-contrast error, and regret.

Under binary randomized actions and proper conditional predictive probabilities p and fitted q, a greedy policy's expected one-step regret obeys

```text
regret(h) <= |p_0(h)-q_0(h)| + |p_1(h)-q_1(h)|
R <= sqrt(2 * E_{h,a~uniform} KL(Ber(p_a(h)) || Ber(q_a(h)))).
```

Proof: greedy policy inequality, Bernoulli Pinsker, Cauchy-Schwarz, Jensen. This is established plug-in decision theory and NOT a novel theorem. It is only a one-step statement assuming the correct logged action distribution/identifiability and a conditional observation model. It does NOT imply that a method with lower finite-data log loss has lower finite-data regret.

Independent coupling-regime results:

- Two-state HMM regret 0.031783; targeted interaction regret 0.004250.
- Paired targeted minus HMM regret -0.027533, SE 0.000303, 95% Student-t interval [-0.028132,-0.026934].
- Paired targeted minus HMM observed predictive log loss +0.000920, SE 0.000185, 95% interval [+0.000554,+0.001287]. **The ranking reversal survives independent replication.**
- Exact oracle-conditional excess log loss HMM 0.003751 vs targeted 0.004578. Lower predictive loss is genuine when averaged over both randomized actions, not just realized-label variance.
- Wrong-action rates about 49.9% (restricted HMM) and 10.0% (interaction), confirming the sign of the action contrast is the major discriminator.
- The oracle KL-based regret upper bound is satisfied numerically but loose (about 0.087 for the HMM).

Negative regimes reproduce the opposite adoption ranking; the HMM has 0 regret when stationary or nuisance-only, while targeted learner has regret around 0.0065.

This establishes the performance distinction for these two trained predictors, NOT a fundamental impossibility of making good decisions from accurate predictions. The stronger Cycle 21 model changes the attainable trade-off.

## Cycle 23: omitted-case sensor-identification audit

Files: `cycle23_channel_audit.py`, `cycle23_channel_audit.json`.
Some original experiment scripts assert the estimated target sensor matches the generator's true target index, raising a selection-bias concern. We audited 1,000 further random prechange seeds for each nuisance count 3, 7 and 11, using only the existing observation-based selector. **Zero failures among 3,000 trials.** This is strong empirical evidence that the channel identification substep is easy in the generator, though it does not prove universal identification or invalidate the concern in more realistic settings.

## Novelty audit: verified closest predecessors

1. Poupart & Boutilier (2002), *Value-Directed Compression of POMDPs*. https://papers.nips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html . Precedent for decision-preserving compression.
2. Baisero & Amato (2021), *Reconciling Rewards with Predictive State Representations*. https://www.ijcai.org/proceedings/2021/299 . Precedent for reward-predictive PSRs and a mismatch between observation-only PSR policies and POMDP policies.
3. Foster, Krishnamurthy & Luo (2019), *Model Selection for Contextual Bandits*. https://arxiv.org/abs/1906.00531 . Precedent for adaptation across model classes under partial feedback and decision-relevant estimation.
4. Zhang et al. (2024), *Provable Representation with Efficient Planning for Partially Observable Reinforcement Learning*. https://proceedings.mlr.press/v235/zhang24bd.html . A strong partial-observability representation-learning benchmark.
5. Alver et al. (2025), *Partial Models for Building Adaptive Model-Based Reinforcement Learning Agents*. https://proceedings.mlr.press/v274/alver25a.html . Direct predecessor on localized adaptation and partial world models.
6. POPGym (Morad et al., 2023). https://github.com/proroklab/popgym . Independent partial-observability benchmark with multiple memory baselines; useful external test bed, but archived-baseline dependencies need scrutiny.
7. Memory Gym (Pleines et al., 2023/2024). https://arxiv.org/abs/2309.17207 . Independent memory-sensitive decision benchmark.

The relevant literature has genuine results about value-relevant compressions and adaptive model-class selection. It is not valid to declare the present objectives unprecedented. This is a targeted audit, NOT an exhaustive claim of equivalence or originality.

## Assessment and next independent research phase

**What is solid:** Experimentally verified one-step decision regret can rank trained predictors differently from log loss; a hand-featured method beats a structurally restrictive HMM under coupling; a stronger learned generative latent model erases much of that disadvantage and has better negative-control behavior. Mathematically, sufficiently low oracle-conditional cross-entropy does upper-bound decision regret.

**What is not demonstrated:** autonomous discovery of new operators, an unknown-onset robust learner, generalization to continuous trajectories, compute-matched advantage, external benchmark value, a new theorem, or a novel learning paradigm.

**Highest-priority open problem:** Discover and validate new *decision-relevant distinctions* online from raw histories when the number and type of latent couplings are NOT prelisted and the environmental change time is unknown. A credible solution should improve **total reward per interaction and computation**, not solely one-step regret on a researcher-tailored process.

Discriminating next protocol:

- Use publicly available partially observable benchmarks with continuous trajectory history and stochastic rewards (for example selected POPGym environments), while preserving the original task's two different kinds of shift: observation-only drift and policy-relevant mechanism drift.
- Compare reasonably tuned GRU/LSTM policy or predictive models, learned latent-state models, reward-predictive PSR/control methods where practical, generic nonlinear policy learning, and known contextual-bandit/model-class selectors. Disclose compute/tuning disparities.
- Test an autonomous candidate-construction mechanism, not just expansion among an investigator-defined family. Predeclare which observations may be accessed, intervention rights, unknown change schedule, memory/compute/sample budgets, relevant returns, and failure conditions.
- Include impossibility/identifiability controls where randomized actions or sufficient observational coverage are withheld.
- No algorithm or paradigm declaration without independent reproducible advantages on such tests.

## Reproduction, provenance

From repository root, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```bash
.venv/bin/python research/cycle21_validation.py
.venv/bin/python research/cycle21_latent_joint.py --seeds 48 --seed-start 91921026 --output research/cycle21_latent_joint_results.json
.venv/bin/python research/cycle21_latent_joint.py --seeds 120 --seed-start 97111026 --conditions contextual_action --output research/cycle21_independent_contextual.json
.venv/bin/python research/cycle22_policy_loss_diagnostics.py --seeds 120
.venv/bin/python research/cycle23_channel_audit.py
```

The additional JSON artifacts preserve aggregated statistics and protocol metadata; logs (`*.log`) and `__pycache__` are Git-ignored. This work ran solely on the user's MCP-connected Linux PC with Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and PyTorch 2.14.1. A GeForce RTX 3060 Ti is available but unnecessary for these small tabular numerical experiments.

**Interpretation policy:** These experiments supply stronger controls and improve scientific calibration; they do not justify announcing an algorithmic breakthrough.
