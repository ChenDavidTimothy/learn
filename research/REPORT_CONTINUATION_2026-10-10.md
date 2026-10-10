# Research continuation: cycles 05 through 11

Status: October 10, 2026. Local PC only.
Root: /home/dtdav/dev/learn
Python: /home/dtdav/dev/learn/.venv/bin/python
Read together with REPORT_2026-10-10.md and README.md.

## Governing result

No consequentially novel learning mechanism has been demonstrated.
We obtained reproducible, bounded mathematical and experimental results
and falsified several seemingly promising shortcuts.
Scientific novelty, viable mechanism, and practical advantage remain
different claims. No general superiority over neural networks exists
in the evidence gathered here.

## Cycle 05: Classical parity learning under membership queries

File: cycle05_interventions.py; output: cycle05_results.json.
For d=32, independent label noise epsilon=0.10, exactly 297 oracle
observations, 48 trials per environment class:
- Sparse 4-bit parity: targeted basis queries 0 test error,
  exhaustive passive 4-sparse parity ML 0 test error.
- Dense 12-bit parity: targeted basis 0 test error,
  deliberately 4-sparse passive ML near chance (0.501).
- Majority 11 bits: both near chance.
- Five-variable conjunction: targeted basis approx 0.031,
  passive sparse-parity ML approx 0.499.
The dense-parity comparator was structurally misspecified, so that
comparison cannot establish a general advantage. Selection geometry
is a real issue, but classical membership-query methods address it.

## Cycle 06: Family selection, initially weak passive comparators

File: cycle06_gate.py; output: cycle06_results.json.
Each learner receives 1,024 noisy labels. The targeted system spends
297 labels on repeated basis queries and the rest on uniform inputs.
Logistic regression and random forests are selected on held-out labels.
These controls lacked a noise-robust parity learner. Their apparent
parity disadvantage was an invalid basis for high-impact claims.
The adaptive policy suffered on majority and conjunction shifts.

## Cycle 07: Correct parity-capable passive comparison

Files: cycle07_ransac.py, cycle07_fair_policy.py.
Output: cycle07_fair_results.json.
A random-subset GF(2) equation solver with full-sample validation
was added as an established noisy-parity competitor.
Selection among parity RANSAC, logistic regression, and random forest
used held-out labels. Again, same 1,024 oracle observations.

Mean excess predictive log loss over 36 independently seeded tasks:
                               adaptive       strong passive
4-variable parity               0.0000             0.0000
12-variable parity              0.0000             0.0000
11-variable majority            0.1062             0.1011
5-variable conjunction          0.0490             0.0460
parity XOR conjunction          0.0550             0.0639
The pure-parity predictive benefit vanished with the stronger
baseline. The composite-task difference was small and statistically
uncertain. The initially spectacular improvement over generic models
was mainly a model-class comparison failure.

## Cycle 08: Intervention versus passive recovery scaling

File: cycle08_scaling.py; output: cycle08_scaling.json.
Dimension d       active exact/18      passive exact/18
32                       18                  18
64                       17                   6
96                       16                   0
128                      17                   0

Both algorithms were given the parity algebra but not the true mask.
Active membership querying repeated zero and unit-vector probes
11 times. Passive RANSAC received the same number of independent,
uniform random labels, with a limit of 1,500 random equations
subsets. This is a comparative algorithm experiment, not a
complexity-theoretic lower bound for all passive learning.

For d input bits, R odd repeated probes and noise epsilon:
p_majority = sum_{j=(R+1)/2}^{R} C(R,j) eps^j (1-eps)^(R-j).
The exact active recovery probability under independent noise is
(1-p_majority)^(d+1); the intercept is estimated by zero queries.
For R=11, epsilon=0.1, p_majority=0.00029571.
Thus exact probability is about 0.9903 at d32, 0.9626 at d128.

For this RANSAC method, a single random full-rank subset of d+1
rows can recover the exact true rule only if all chosen labels are
clean. Approximate success per draw under independent random rows:
p_draw = (1-epsilon)^(d+1) product_{i=1}^{d}(1-2^{-i}).
It falls from 0.008924 at d32 to 3.61e-7 at d128.
These predictions align with the bounded algorithmic comparison.
They DO NOT rule out other passive decoders or enhanced computing.
## Cycle 09: Candidate-free evidence of representation mismatch

File: cycle09_generic_detector.py; output:
cycle09_generic_detector.json.
Rather than specifying a new rule family, define Q as a fair-coin
predictive distribution and H0 as ANY single rule in the original
1,335-candidate grammar. The sequential statistic

  E_t = Q(y1:t | x1:t) / max_(h in H0) P_h(y1:t | x1:t)

has anytime-valid type-I control for every fixed correct null h,
under known independent label noise and predictable interventions.
This is an established universal-inference construction.
It does NOT generate an explanatory alternative or adapt predictions.

Replication used 150 fresh seeds per regime and up to 192 postshift
random queries, with six Bonferroni-corrected output tests (alpha=.05):
- In-class: zero alerts observed.
- New parity4: both changed outputs alerted in 150/150 trials.
  Mean time when both were alerted: 36.51 queries.
- Dense threshold: global grammar mismatch caused many alerts,
  typically including unchanged mechanisms. At 192 queries,
  mean total alerts = 5.37 outputs out of six.

Key failure: absolute model inadequacy is NOT change localization.
A detector that ignores historical prechange behavior can mistake
an always-unsupported mechanism for a changed mechanism.

## Cycle 10: Relative conditional two-sample testing

File: cycle10_relative_shift.py; outputs:
cycle10_relative_shift_d0.json through d4.json.
Independent pre/post samples: 96 and 192, noise .05, d=10 input bits.
For each output, form z=(2y-1)*chi_S(x), Walsh features degree<=r.
Test the maximum shift of empirical feature means. Use Hoeffding:
P(|mu_post-mu_pre| >= t) <=
  2 exp[- t^2 / (2*(1/npre+1/npost))].
Bonferroni across six outputs and all prespecified features controls
the familywise false-alarm rate at .05 without a grammar assumption.
Results: correct localization of BOTH shifts out of 160 trials:
feature degree max     parity4 shift    dense threshold shift
0                        38                 13
1                        43                117
2                        58                 99
3                       121                 74
4                       160                 62

Degree-3 features cannot represent a fourth-order parity rule but
often detect disappearance of an old lower-degree mechanism.
However, monitoring more features raises statistical thresholds.
On dense thresholds, degree-1 substantially outperforms degree-4.
This is a known bias/variance and multiple-testing trade-off.
The task calls for careful signature selection, NOT simply
unboundedly expanding the monitor's representation.

## Cycle 11: Test the apparent compact-signature opportunity

File: cycle11_fingerprint.py; output: cycle11_fingerprint.json.
A selective monitor uses 48 prechange labels to choose the top
correlated Fourier feature(s), then AN INDEPENDENT 48 prechange
calibration labels and 192 postchange labels to test for a shift.
Splitting protects type-I validity even though the signature is
data-dependent. Bonferroni corrects over selected signatures/outputs.

Tested max degrees 1,3,4 and 1,2,4 selected features.
Best degree-3 single-feature monitor localized both parity4 shifts
in only 116/160 trials, versus 121/160 for the full degree-3
monitor. On dense thresholds, the compact degree-4 configurations
localized BOTH shifts in only 1 to 8 of 160 trials, versus
62/160 for the full degree-4 monitor and 117/160 with degree-1.

This REFUTED our first compact-monitoring implementation as a
general improvement. Feature selection from limited prechange
data can omit useful change-sensitive directions. Protecting
false-alarm rates did not guarantee sensitivity or practicality.
No evaluation-condition-specific threshold tuning was performed.
## What actually changed in our understanding

(A) Knowing that a model is inadequate is mathematically distinct
from locating changes and distinct again from learning a replacement.
Different information and experimental access are needed.

(B) Before/after observed signatures can localize some shifts even
when new mechanisms are NOT representable by the monitored features.
A new operator need not be known to detect disappearance of old facts.

(C) The best monitoring feature family is environment-dependent:
more expressive tests sometimes reduce power under finite data.
Naively selecting compact signatures can be substantially worse.

(D) Active experimental rights can strongly change the computational
resources needed to learn a mechanism. But classic query-learning and
noisy parity theory already explain much of the apparent benefit.

(E) Several gains vanish under competent structured baseline models.
This underlines why neural-vs-nonneural generality cannot be inferred
from a benchmark chosen for one side's inductive bias.

## Mathematical research boundary

If a target function is sampled uniformly from all Boolean functions
over N possible input points, labels at distinct unseen points are
independent fair bits conditional on any observed dataset.
Thus no algorithm, neural or not, achieves expected error <1/2
on genuinely unseen inputs without structural assumptions or transfer
from a non-uniform task distribution. This is a standard no-free-lunch
observation, not our discovery. Asking for a learner that invents ANY
mechanism with no prior regularities is mathematically ill-posed.
## Research decision and next serious direction

Stop treating finite Boolean-rule expansion as the breakthrough target.
More low-impact modifications to the fixed grammar would violate the
standing preference for consequential, differentiated hypotheses.

Candidate next organizing problem: learning *decision-relevant,
intervention-sensitive predictive abstractions* in unknown partially
observed environments with localized dynamics changes. The goal is
not to reconstruct an assumed latent state but to learn predictive
equivalence classes that support actions, detect shifts, and update
only implicated future predictions. A formal model could use:

  hidden Markov state s_t; actions a_t; raw observations o_t;
  unknown transition mechanisms; m potentially interacting factors;
  source tasks with some shared but unidentified regularities;
  train/test changes not announced to the learner.

Decisive comparative test: Is there a family for which a learner
constructs a transferable test/intervention basis from experience,
then has lower *total interaction plus computation* to recover
decision performance than carefully tuned latent-state models,
spectral predictive-state representations, causal modular models,
and neural sequence learners? The evaluation must include structure
discovery costs, unknown change time, task value/regret, uncertainty
calibration, and adversarial coupling. Without this comparison the
candidate is only a research question, not a novel paradigm.

Nearest predecessors to audit before implementing:
- Littman, Sutton, Singh, Predictive Representations of State (2001).
- Spectral PSR learning / observable operator models / Hankel rank.
- DreamCoder program abstraction induction and wake-sleep learning.
- Sparse causal representation/shift work, including Lopez et al.
- Bayesian changepoint detection and model-based active identification.
- Adaptive model selection, anytime-valid e-process inference.

Avoid stating these are equivalent without detailed mechanistic
comparison; but their existence significantly raises the novelty bar.
## Verified predecessors and sources

- Blum, Luby, Rubinfeld, Self-testing/correcting with applications
  to numerical problems (1993):
  https://doi.org/10.1016/0022-0000(93)90044-W
- Kushilevitz and Mansour, Learning Decision Trees Using the
  Fourier Spectrum (1993): https://doi.org/10.1137/0222080
- Chen, Shu, Zhou, Algorithms for Sparse LPN and LSPN Against
  Low-noise (COLT 2025): https://arxiv.org/abs/2407.19215
- Wang, Dillig, Singh, Program Synthesis using Abstraction
  Refinement (POPL 2018): https://arxiv.org/abs/1710.07740
- Lopez et al., Learning Causal Representations of Single Cells
  via Sparse Mechanism Shift Modeling (CLeaR 2023):
  https://proceedings.mlr.press/v213/lopez23a.html
- Wu et al., Adaptive Nonparametric Perturbations of Parametric
  Models with Generalized Bayes (JMLR 2026):
  https://www.jmlr.org/beta/papers/v27/25-1832.html
- Littman et al., Predictive Representations of State (2001):
  https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf

This is a targeted and incomplete literature audit, not a claim
that every nearby article has been reviewed in full.

## Reproduction and record

cd /home/dtdav/dev/learn
.venv/bin/python research/cycle05_interventions.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/cycle06_gate.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/cycle07_fair_policy.py
.venv/bin/python research/cycle08_scaling.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/cycle09_generic_detector.py
for d in 0 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 FEATURE_DEGREE=$d .venv/bin/python research/cycle10_relative_shift.py
done
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/cycle11_fingerprint.py

Running these commands overwrites the same named result JSON files;
existing fixed-seed raw result summaries remain saved.
No files outside /home/dtdav/dev/learn were modified by research scripts.
No calculations used the assistant's Python container.
