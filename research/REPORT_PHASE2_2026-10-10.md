# Phase II Research Handoff: Decision-Relevant Learning Under Partial Observability

**Date:** 10 October 2026.
**Execution:** user's Linux PC through Terminal MCP, not assistant container.
**Project root:** /home/dtdav/dev/learn
**Python:** /home/dtdav/dev/learn/.venv/bin/python
**Reports to read first:** REPORT_2026-10-10.md,
REPORT_CONTINUATION_2026-10-10.md, then this document.

## Executive outcome

Cycles 12-20 explored predictive-state compression, learned
controlled HMMs, unknown factor couplings, policy learning, and
unknown-onset adaptation. We obtained one robust, consequential
*conditional* result: a task-centered interaction model can greatly
reduce one-step action regret after a hidden environmental coupling,
despite slightly worse overall predictive log loss than a learned
compact HMM. The preference reverses in the independent regime.
None of these mechanisms is conceptually unprecedented. Closest
predecessors include PSRs, value-directed POMDP compression,
reward-predictive PSRs, causal/treatment-effect meta learners, and
Bayesian/POMDP model learning.

This IS NOT a breakthrough in general-purpose machine learning.
It is a reproducible and properly bounded research finding which
clarifies a path to consequential future research.

## Formal mathematical target

Environment histories h are noisy action-observation sequences;
the hidden state s may have independent or coupled factors.
There are actions a in {0,1} and a binary observed target Y.
Let mu_a(h)=E[Y | H=h, do(A=a)] under the environment. The task
is to choose an action maximizing mu_a(h), not merely to model
every component of P(next observations | history, action).

Define treatment/action contrast tau(h)=mu_1(h)-mu_0(h).
The optimal one-step decision is pi*(h)=1{tau(h)>0}.
For any policy pi, its expected one-step regret equals

  R(pi)=E[ |tau(H)| * 1{pi(H) != pi*(H)} ].

Thus policy selection requires the right SIGN of the action
contrast where it matters. Aggregate cross-entropy of a predictor
may be good while its action contrast is badly misspecified.
This identity is elementary expected-utility algebra and is
NOT an original mathematical theorem.

## Closest verified research predecessors

- James and Singh (ICML 2004), Learning and discovery of PSRs:
  https://doi.org/10.1145/1015330.1015359
- Kulesza, Jiang and Singh (AAAI 2015), spectral PSRs with
  insufficient statistics:
  https://doi.org/10.1609/aaai.v29i1.9635
- Poupart and Boutilier (NeurIPS 2002), Value-Directed Compression
  of POMDPs:
  https://papers.nips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html

- Baisero and Amato (IJCAI 2021), Reward-Predictive State
  Representations:
  https://doi.org/10.24963/ijcai.2021/299
- Li et al. (AISTATS 2020), Efficient planning with unnormalized
  Q functions and spectral learning:
  https://proceedings.mlr.press/v108/li20h.html
- Ross, Chaib-draa and Pineau (NeurIPS 2007), Bayes-Adaptive POMDPs:
  https://papers.nips.cc/paper/3333-bayes-adaptive-pomdps
- Künzel et al. (PNAS 2019), Meta-learners for estimating
  heterogeneous treatment effects using machine learning:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6410831/
- Nie and Wager (2021), Quasi-oracle estimation of heterogeneous
  treatment effects:
  https://doi.org/10.1093/biomet/asaa076
- Boots et al. (2013), Hilbert space embeddings of PSRs:
  https://proceedings.mlr.press/r11/boots13a.html

These precedents prevent claims that predictions-as-state,
task-directed compression, reward-sensitive state representation,
and estimating action contrasts are original discoveries.
This is a targeted, not exhaustive, novelty audit.

## Cycle 12a: exact rank separation

Script: cycle12_rank.py
Result: cycle12_rank.json

Independent two-state Markov target and m independent nuisance bits.
Each bit produces a separate noisy observable. For a fixed action,
the joint one-step observable past/future matrix is a Kronecker
product of full-rank 2x2 matrices, hence rank 2^(m+1).
The task-only matrix containing predictions of target observations
can be represented with rank 2 because nuisance histories convey
no information about the independent target future given target
history. Numerically verified from m=0 through m=8 (full rank 512,
task-only rank 2 at m=8).

This is a restricted case of classical predictive-state theory.
The experiment did not learn PSR operators from raw trajectories.
It did not prove a general bound across coupled models or actions.

## Cycle 12b: fair conditional prediction with hidden states

Script: cycle12_partial_obs.py
Result: cycle12_partial_obs.json

A 2-step noisy history of controlled hidden binary factors,
observed through permuted noisy channels, is used to predict the
next target observation. Training includes the supervised target
observation, never hidden state labels. Target channel index is
not revealed and is identified from PRE training data.
Each method gets 960 prechange and 192 postchange observations.
Twenty seeds, nuisance factors m=3,7,11; source task and shifts
from independently resampled three-step episodes.
The methods are a selected-target 16-cell predictive table,
full L1/L2 logistic models, full interaction logistic, and trees.

At seven nuisance factors, final mean predictive log losses:
  stable: task table .63788 vs full L1 .63512;
  nuisance shift: task .63788 vs full L1 .63619;
  target coupling: task .66864 vs full L1 .66356;
  target transition shift: task .70644 vs L1 .69878.

There was no predictive advantage from simply using fewer factors.
Strong sparse full-input models can learn relevant features.

## Cycle 12c: task-rank jumps under coupling

Script: cycle12_coupling_rank.py
Result: cycle12_coupling_rank.json

Exactly evaluated a two-latent-bit model. Matrix entries were
two-step FUTURE task-observation test probabilities conditioned
on a current jointly observed noisy target/nuisance pair.
Independent dynamics: rank 2 and zero conditional mutual
information from current nuisance observation about future task.
Target dynamics weakly coupled to nuisance: rank 4, while the
extra conditional mutual information in two-step task forecasts
is only about 0.00370757 bits under the specified parameters.

Exact algebraic rank can jump from 2 to 4 for weak coupling even
when decision-relevant information remains small. Rank alone
is not a sufficient measure of practical significance.

## Cycle 13: structured latent-state HMM baseline

Script: cycle13_latent_hmm.py
Result: cycle13_latent_hmm.json

A learned controlled two-state HMM fits unknown observation noise
and two transition rates by weighted likelihood on matched noisy
target-input/output histories. Target observation channel is
selected from training data. No true latent state is supplied.

Numerical audit caught a uint8 unsigned-negation overflow in
the ORIGINAL implementation of the likelihood objective.
That experiment was INVALID and was corrected and rerun.
The current saved result is from the corrected floating-label
likelihood. Do not use preliminary erroneous scores.

At seven nuisance factors, HMM postchange predictive loss:
  stationary .62805
  nuisance-only shift .62805
  coupled target .66097
  target rate shift .69843

These improved on the task-table and full L1 logistic controls.
A correctly chosen compact LATENT model can be better than a
naive compact PREDICTIVE model.

## Cycle 13b: oracle information ceiling

Script: cycle13_oracle.py
Result: cycle13_oracle.json

The oracle knows exact hidden factor identity, transition rates,
observation noise, and nuisance coupling. Used ONLY to quantify
available information, not as an attainable learning baseline.
At seven nuisance factors with target coupling, full oracle
mean log loss was .64698; reduced-misspecified oracle .65827.
Thus the coupled nuisance carries usable predictive information.

## Cycle 14: evidence requirement for expanding a task model

Script: cycle14_interaction_data.py
Result: cycle14_interaction_data.json

Baseline learned HMM compared with ordinary hand-featured target
interaction logistic, full logistic with L1 and full degree-2 L1.
All get the SAME pre/post labeled histories and old-data discount.
16 seeds, 11 nuisances. Post budgets 192,768,3072,12288.
In coupled environment, mean log losses:

Budget        HMM      targeted interactions   full degree2
192          .66244        .66519                .67449
768          .66013        .65924                .67001
3072         .65911        .65368                .66171
12288        .65880        .65193                .65552

With little new evidence, a restrictive HMM is favorable; with
more evidence, including interactions helps. The representation
expansion was manually supplied as a feature family, so this is
not autonomous discovery of a new primitive.

## Cycle 15: direct decision task, hidden contextual action effect

Script: cycle15_decision.py
Result: cycle15_decision.json

New regime "contextual_action": target hidden bit's flip probability
is .08 if chosen action matches a hidden nuisance bit and .32
otherwise. The reward is whether the next noisy target observation
equals one. The optimal action now depends on joint target/nuisance
history. Action labels are randomized during training, ensuring
positivity. The agent is evaluated by one-step expected policy regret
using the true model ONLY at evaluation, not training.

12 seeds, 7 nuisances, 960 initial and up to 8192 new samples.
After 8192 postchange samples:
  two-state HMM predictive loss .65812; regret .03175;
  target-action interactions loss .65890; regret .00349;
  full degree3 L1 logistic regret .01373;
  full L1 logistic regret .03134.

Despite worse global log loss, targeted interaction learning chose
much more valuable interventions in this particular environment.
This is a consequential, conditional engineering outcome, not
new causal learning theory.

## Cycle 16: strong causal treatment-effect controls

Script: cycle16_treatment_effect.py
Result: cycle16_treatment_effect.json

Classical randomized treatment-effect meta learners, trained on the
same observations. For randomized A with probability one-half and
binary Y, Z=(2A-1)(2Y-1) is an unbiased transformed outcome for
tau(h)=E[Y|A=1,h]-E[Y|A=0,h].
Direct ridge, direct Lasso and two-model T-learner all improved
decisions, with 8192-sample regrets .00846, .00624, .00796.
The targeted S-style interaction learner reached .00349.
This is method selection within established causal learning.

## Cycle 17: neural and nonlinear baselines

Script: cycle17_neural_tree.py
Result: cycle17_neural_tree.json

An MLP with 128+128 hidden units was trained using PyTorch on
the local NVIDIA RTX 3060 Ti; its epoch count was selected by
held-out weighted validation. HistGradientBoosting was also tested.
Both receive all raw history/actions and outcome labels; no hidden
states or target sensor index. The target interaction learner from
Cycle 15 does use a more favorable hand-designed feature grammar.

12 seed results for contextual action coupling, budget 8192:
  MLP policy regret .02944
  HistGradientBoosting policy regret .01460
  targeted interaction model policy regret .00349
  learned two-state HMM policy regret .03175

This result DOES NOT prove neural methods fundamentally incapable;
it tests one trained MLP and one tree-boosting setup, both potentially
improvable through more tuning or more data.

## Cycle 18: unannounced change in a stream of short episodes

Script: cycle18_online_selection.py
Result: cycle18_online_selection.json

Time of environment change randomized between stream samples 1200
and 1750. Learners see 960 prior samples plus 4096 randomized-action
samples in order, with rolling windows of 2048. They are not told
when the change happens. IMPORTANT: The stream is independent noisy
3-step POMDP episodes, NOT a single continuous controlled trajectory.

At each checkpoint, refit the HMM and interaction candidate.
Select model either on heldout predictive log loss or doubly robust
policy value estimated with known randomization propensity one-half.
An independent recent validation slice of <=384 observations is
used for selection.

At t=4096, across 24 seeds in hidden action-coupling:
  always HMM regret .03180
  always interaction regret .01521
  logloss-selected regret .03092
  DR-policy-value-selected regret .02174

The DR gate reduced decision regret in the coupled regime but
selected the interaction model unnecessarily in stable/no-target-
change regimes. Selecting methods by policy value without
uncertainty control is not enough for robust adaptation.

## Cycle 19: confidence-aware switching, independent replication

Script: cycle19_confident_gate.py
Result: cycle19_confident_gate.json

48 NEW seeds per regime. Gate interaction only if its heldout
paired doubly robust policy value exceeds the HMM by more than
1.645 estimated standard errors (a predeclared one-sided margin).
This is NOT an anytime-valid confidence-sequence guarantee, since
windows are reused and adaptively revisited across checkpoints.

The confidence-gated selector reduced some false model switches on
stationary tasks, but missed important coupling changes. At t=4096,
across 48 new seeds in coupled environments:
  always HMM regret .03200;
  always interaction regret .01554;
  logloss gate regret .02972;
  unguarded policy-value gate regret .01926;
  conservative confidence gate regret .02865.
A naive significance threshold is NOT a general solution to
unknown-change autonomous representation selection.

## Cycle 20: 120-seed independent confirmation

Script: cycle20_independent_replication.py
Result: cycle20_independent_replication.json
Novel seeds disjoint from preceding trials, with model structures and
hyperparameters frozen. Every learned model received the same 960
prechange and 8192 postchange examples. No hidden state or causal
interaction location was supplied to learners. Oracle knowledge was
used ONLY to calculate expected policy regret.

On hidden action-dependent coupling:
  learned 2-state HMM: log loss 0.658070; regret 0.0318855.
  targeted interaction model: log loss 0.658778; regret 0.0050915.
  direct Lasso effect learner: regret 0.0068841.
  two-model effect learner: regret 0.0081853.
The interaction method reduced regret relative to HMM by 0.0267940
reward units, approximately 84.03%; paired SE 0.0003593.
Approximate 95% Student-t CI for interaction minus HMM regret:
[-0.0275054, -0.0260826].
This is a robust and conditional synthetic-task finding, not a
general superiority claim.

Negative controls on the SAME 120-seed protocol:
                       HMM regret  interaction regret
  stationary            0.00000         0.00666
  nuisance-only shift   0.00000         0.00685
  target rate shift     0.00027         0.00717
  action coupling       0.03189         0.00509

We have NOT demonstrated autonomous invention of an interaction
family, a well-calibrated successful unknown-onset switching method,
an advantage over all RL / PSR / causal policy-learning predecessors,
or value on external real-world benchmarks. The methods use
researcher-specified feature families and independently sampled,
short action-observation episodes, not continuous trajectories.

## Research decision

The most consequential open issue is discovering, validating and
adapting decision-relevant predictive distinctions under unknown
change and partial observability, with total interaction and
computational cost measured against strong latent-state, reward-
predictive PSR and causal policy-learning baselines.
Do not continue optimizing the same synthetic task to manufacture
novelty. Move to independent benchmark environments and theoretically
meaningful mechanisms with explicit decisive comparisons.

## Reproduction

From /home/dtdav/dev/learn (reuses existing .venv):

    .venv/bin/python research/cycle12_rank.py
    .venv/bin/python research/cycle12_partial_obs.py
    .venv/bin/python research/cycle12_coupling_rank.py
    .venv/bin/python research/cycle13_latent_hmm.py
    .venv/bin/python research/cycle13_oracle.py
    .venv/bin/python research/cycle14_interaction_data.py
    .venv/bin/python research/cycle15_decision.py
    .venv/bin/python research/cycle16_treatment_effect.py
    .venv/bin/python research/cycle17_neural_tree.py
    .venv/bin/python research/cycle18_online_selection.py
    .venv/bin/python research/cycle19_confident_gate.py
    .venv/bin/python research/cycle20_independent_replication.py

All scripts run on the user's PC. Python environment, logs, and
caches are excluded from Git. The checked-in *.json files contain
aggregated numerical output and protocol metadata, including
replication results. Avoid interpreting a successful numerical
experiment as evidence that its conceptual ingredients are novel.
