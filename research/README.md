# Research program: consequential learning mechanisms

**Latest research phase (2026-10-10):** Read `STANDING_RESEARCH_STANDARD.md`,
then `REPORT_PHASE7_2026-10-10.md` for Cycles 45-53: sensor/dynamics
observational equivalence, costed reward calibration, native pixel-POMDP
sensor adaptation, strong proactive masked-training controls, and an eight-task
color-palette generality audit. **Stop extending the 7-11-color sprite
restoration branch; pivot to richer partially observed sensing.** Phase VI
covers Cycles 40-44, Phase V Cycles 32-39, Phase IV Cycles 24-31, Phase III
Cycles 21-23, and prior reports Cycles 01-20. No general learning breakthrough
has been established.

Last verified: 2026-10-10. Workspace: /home/dtdav/dev/learn.
Python: /home/dtdav/dev/learn/.venv/bin/python (3.12.3).
Cycle 01 historical status: initial formalization and controlled sanity experiments
only. Current findings and limits are in Phase VII, VI, V, IV and III reports.

## Governing question

If a learning principle works, what changes in understanding, capabilities,
or informed choices, and what evidence would establish that change?
Do not confuse engineering integration, novelty, and demonstrated advantage.

## Candidate portfolio (none selected as final)

A. Selective revision: learn reusable mechanisms, diagnose a change, update
only implicated knowledge. Primary risk: existing Bayesian modular filtering,
active diagnosis, and continual learning may already capture the principle.
B. Reusable causal abstractions: learn transferable causal variables,
operators, or intervention-sensitive compositions. Primary risks:
identifiability, expensive search, prior program/causal learning.
C. Predictive-state learning: acquire minimal actionable predictive
representations that transfer across environments. Primary risks:
established predictive state representations and learned world models.
Potentially explore structure discovery + robust revision only after
substantive literature audit, strongest coherent mechanism, and baselines.

## Cycle 01: formal environment

There are m=16 independent modules. Each module has two input entries.
An entry is queried by one action a in {0,...,2m-1}, and returns a noisy bit.
The prechange truth table b in {0,1}^{2m} is known EXACTLY to every learner
(no training/sample cost is charged; privileged knowledge, Stage A only).
After the shift exactly k=2 modules change. Every query returns
y = b[a] XOR d[a] XOR e, e ~ Bernoulli(0.1), independently per query.
Single regime: exactly one entry flips per changed module.
Coupled regime: both entries flip per changed module.
Learners select one action per intervention and do not observe other entries.
Each trial supplies the same underlying base/shift and a shared noise table
to all methods, with distinct policy random seeds.
All methods receive 64 interventions. 80 fixed, independent trial seeds.

## Formal assumptions, hypothesis classes, and key distinction

For the restrictive model the changed subset H is uniform over
N = choose(m,k) * 2^k = 480 hypotheses. Its exact posterior is updated
by Bayes rule given noise rate, base table, k, and its hypothesis class.
The flexible independent estimator keeps a posterior flip probability per
entry, using prior k/(2m), and does not enforce k or module consistency.
The mixture model puts prior mass 1/2 on the restrictive regime and 1/2
on a coupled regime (exactly both entries flip in each changed module).
There are 480 + 120 = 600 hypotheses, posterior updated exactly.
It is given a richer hypothesis class, NOT inferred from raw observations.
All active methods greedily choose the single action with maximum
one-step mutual information about the hypothesis under binary symmetric
noise. Since epsilon is fixed, this is maximal p_flip*(1-p_flip).
This is not a globally optimal experiment-design policy.
Random controls have identical inference but uniform random actions.
Frozen and tabula-rasa reset are intentionally weak diagnostic controls,
NOT substitutes for competent neural or modular baselines.

## Information-theoretic sanity check

With uniform H over N hypotheses and a queried output bit passing through
independent Bernoulli(epsilon) noise, each observation carries at most
1 - h2(epsilon) bits about H, even with adaptive action choice.
Fano's inequality gives for fixed T and identification error <= delta:
T >= [log2 N - h2(delta) - delta*log2(N-1)] / [1-h2(epsilon)].
This is a loose NECESSARY bound only. It does not establish achievable
adaptation cost, exact recovery with noise, or superiority of any learner.
It does not generalize unchanged to other observation models, priors,
query bandwidths, or feedback structures.

## Predeclared questions (before observing this follow-up)

1. Within the correct restrictive model, does greedy intervention choice
   reduce prediction loss compared with random interventions at equal budget?
2. When single-entry structure is false, does the restrictive model
   underperform a less brittle independent local learner?
3. When given the correct wider family as a prior, can Bayesian model
   averaging recover and erase much of the apparent advantage?

## Experimental results after 64 queries

Mean expected excess cross-entropy in nats per action-output coordinate
minus the irreducible Bernoulli(0.1) observation entropy. Evaluation covers
all 32 coordinates, not only the selected actions. Test truth is
available for evaluation ONLY, not policy decisions. Lower is better.

Method                    single-regime     coupled-regime
structured_active         0.004993          0.109854
structured_random         0.036873          0.109846
mixture_active            0.008090          0.001387
mixture_random            0.042539          0.020463
independent_active        0.006007          0.012717
reset_random              0.172000          0.169872
frozen                    0.109861          0.219722

The original experiment results are preserved as cycle01_results_v1.json.
The revised results (including mixture controls) are cycle01_results.json.
WARNING: adding models changed method-index RNG streams for some methods,
so v1 and revised results must not be interpreted as paired comparisons.

## What the results justify

Engineering: models implemented and tested in a tiny noise-controlled task.
Within-assumption experiment: active action selection helps the exact model.
Out-of-class experiment: a narrower hypothesis class can fail badly.
Stronger control: conventional Bayesian mixtures can handle the broader
change family and outperform the independent updater in coupled conditions.
Novelty: NO NEW PRINCIPLE established; these techniques have predecessors.
Generality: no claims for hidden structure, raw data, learned factorization,
continuous dynamics, real applications, or deep neural baselines.
Energy: not measured. Wall-clock time is not energy consumption.

## Closest verified prior research, initial bibliography

- Adams and MacKay (2007), Bayesian Online Changepoint Detection:
  https://arxiv.org/abs/0710.3742
- Hauser and Buehlmann (2014), Two optimal strategies for active learning
  of causal models from interventional data:
  https://doi.org/10.1016/j.ijar.2013.11.007
- Ellis et al. (2021), DreamCoder: Bootstrapping Inductive Program
  Synthesis with Wake-Sleep Library Learning:
  https://doi.org/10.1145/3453483.3454080
- Schoelkopf et al. (2021), Toward Causal Representation Learning:
  https://doi.org/10.1109/JPROC.2021.3058954
- Littman, Sutton, and Singh (2001), Predictive Representations of State:
  https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf
- Lake, Salakhutdinov, and Tenenbaum (2015), Human-level concept
  learning through probabilistic program induction:
  https://doi.org/10.1126/science.aab3050
These do not constitute an exhaustive novelty audit.

## Next decisive research cycle

Stage B: do not provide the factorization or truth table. Give all methods
equal pre-change interaction budget and measure FULL costs of discovering
structure, locating shifts, and adaptation. Include a competently tuned
neural predictor, structured statistical baseline, modular neural baseline,
and flexible Bayesian mixture / change-detection control. Include mixed,
nonlocal, coupled, noisy, and partially observed shifts with held-out seeds.
Separate identifiability from computational issues. Use matched action
access, tuning budgets, and computational metrics; no GPU energy claims.
A new mechanism requires more than reproducing known Bayesian diagnosis.


## Post-run verification and uncertainty (2026-10-10)

Run: cd /home/dtdav/dev/learn &&
  .venv/bin/python research/cycle01_sparse_repair.py
Check: .venv/bin/python research/cycle01_validate.py
Validation: hypothesis counts, posterior normalization, marginal count
invariants, and an analytical one-observation Bayes update passed.
For restrictive N=480, epsilon=0.1, target error at most 0.05, the
loose Fano lower bound is T >= 15.395917 fixed-budget observations.

Paired difference in final excess log loss (nats/output coordinate):
Single: structured_active - structured_random = -0.031881, SE 0.003696.
Single: mixture_active - mixture_random = -0.034449, SE 0.004371.
Coupled: structured_active - structured_random = +0.000008, SE 0.003022.
Coupled: mixture_active - mixture_random = -0.019076, SE 0.004500.
Coupled: mixture_active - independent_active = -0.011330, SE 0.002792.
These standard errors use 80 paired, independently seeded environments.
Approximate normal intervals, NOT formal universal performance guarantees.
Full values and commands are in cycle01_validation.txt.

Reasoned conclusion: active selection is useful within this toy class,
but the broad mechanism is already captured by standard Bayesian inference
once an adequate hypothesis space is explicitly supplied. The harder
open question is how that space and its computationally tractable updates
could be learned from unstructured observations without privileged access.
No claim of a breakthrough is justified by Cycle 01.
