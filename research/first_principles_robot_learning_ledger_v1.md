# First-Principles Robot Learning: Living Research Ledger

**Version:** 1  
**Created:** 2026-10-20  
**Scope:** Derive and experimentally test the minimal structure needed for a robot to learn general principles through action and observation.  
**Status:** Research notebook, not a claim of a new algorithm or a universal theory of intelligence.  
**Companion artifacts:** `car_learning_benchmark_v1.py`, `car_learning_benchmark_v1_results.json`.

## Read this first: current position (V1)

1. A robot cannot infer which outcome is *desirable* from physical trajectories alone when different desirability rules are compatible with the same physics. Some preference/goal/judgment information must be supplied or assumed.
2. A finite dataset does not uniquely determine a universal rule. Every learner requires an inductive bias, hypothesis class, prior, or analogous structural assumption.
3. A useful *state* is information from history sufficient to predict the task-relevant effects of future **interventions**. The canonical exact predictive state can be defined as an equivalence class of histories with identical distributions of future observations under every chosen future action sequence.
4. A robot cannot learn a variable or parameter through data that never distinguishes its possible values. Experiments must provide informative variation. This is a rank/identifiability question, not only an optimization question.
5. The learned latent coordinates are not uniquely determined. Equivalent invertible reparameterizations yield identical physical predictions. Demand correct transferable behavior, not human-named hidden variables.
6. The world model needed for all possible tasks is generally more detailed than the world model needed for one particular goal. Learning should distinguish uncertainty that changes decisions from uncertainty that does not.
7. In a toy simulation with no direct velocity sensor, a small expression grammar selected `x_t - x_(t-1)` as a useful predictor. It did **not** discover arithmetic or physical concepts from pixels, and its expression grammar is an explicit inductive bias.
8. In a small nonlinear variant, providing `v*abs(v)` in the expression grammar enabled exact recovery of a nonlinear drag law. This demonstrates expressibility within the grammar, not an unrestricted nonlinear-learning breakthrough.
9. No candidate approach (reward maximization, pure prediction, symbolic model discovery, active experimentation, or goal-conditioned models) has been demonstrated superior overall. Retain parallel alternatives and design falsifying comparisons.

## 0. Research discipline and boundaries

**Objective:** Start from basic logical and mathematical requirements for learning, then consult established research to avoid relabeling old results as new. Keep the example a simple car, expandable to nonlinear dynamics and later robotics.

**Included:** partial observation, state discovery, causal action-outcome models, labeled goodness, hypothesis revision, experiment choice, goal transfer, nonlinear identification.

**Deferred intentionally:** safety, alignment, humanoid embodiment, large pretrained models, optimization scaling, hardware details, reward hacking. These are important independently but would obscure the foundational question at this stage.

**Epistemic labels:**
- **PROVEN under assumptions:** mathematically derived from specified assumptions.
- **TESTED in simulation:** code ran and reproducible numeric results are saved.
- **EXISTING RESEARCH:** established by cited literature, not claimed as new.
- **HYPOTHESIS:** proposed architecture or generalization, not demonstrated.
- **OPEN:** unresolved and requires new reasoning, mathematics, or experiments.
- **REJECTED / CORRECTED:** a previous claim was too strong, mismatched the equations, or failed a test.

## 1. Question history, decisions and corrections

| ID | Question or proposed principle | Result/status | Next AI should know |
|---|---|---|---|
| Q01 | How can labeled car movements imply "forward is good"? | PROVEN conditionally | Positive displacement fits, but infinitely many laws fit finite training labels. |
| Q02 | Is maximizing reward the fundamental built-in mechanism? | OPEN; no universal proof | An objective may select behavior without specifying a representation-learning method. The reverse claim, that reward is useless, is also unsupported. |
| Q03 | Could prompting experiences to an LLM suffice? | INSUFFICIENT AS A FOUNDATION | A prompt supplies information, while its interpretation depends on a previously learned inference mechanism. |
| Q04 | Could pure world prediction suffice? | NEGATIVE for arbitrary preference learning | Identical physics permits opposite desirability rules. Prediction is powerful but does not uniquely supply goals. |
| Q05 | Can the robot discover state variables from prediction failures? | TESTED in small grammar | Position alone aliases different momentum states; adding history or its difference resolves this toy ambiguity. |
| Q06 | Does the robot necessarily discover *velocity* uniquely? | REJECTED | `(x, x_prev)` and `(x, x-x_prev)` are invertibly related. More generally latent coordinates are non-unique. |
| Q07 | Can informative actions help identify parameters? | PROVEN conditionally; TESTED | The input/velocity design matrix must have full column rank for unique noiseless recovery of `(alpha,beta)`. |
| Q08 | Must the robot model every physical detail? | NEGATIVE for a fixed restricted goal | Two distinct models can imply the same adequate action. Different goals can require finer distinctions. |
| Q09 | Does a single state-potential always encode "good transitions"? | REJECTED | Directed cycles of strictly good transitions cannot all increase a state-only scalar potential. A transition reward can nevertheless encode them. |
| Q10 | Can the same principle extend beyond linear physics? | TESTED on one nonlinear grammar; general case OPEN | A sparse library containing `v*abs(v)` recovered nonlinear drag; unknown basis, noise and generalization remain hard. |
| Q11 | Are safe experiments needed for the foundational derivation? | DEFERRED by request | Do not reintroduce as a leading topic without a specific need. |

**Correction to previous discussion:** Two integration conventions were used across explanations. Under explicit Euler `x_next=x+v`, `v_next=v+a`, the two-step trajectory `(x,v)=(8,1)` to `(10,0)` uses actions `(0,-1)`. Under **semi-implicit Euler** `v_next=v+a`, `x_next=x+v_next`, the same actions go from `(8,1)` to `(9,0)`, **not** `(10,0)`. V1 uses semi-implicit Euler consistently. This was caught by executable assertions and corrected; do not silently mix conventions again.

## 2. Basic formal setup and what is given

At time `t` an agent has a history `h_t=(o_0,a_0,o_1,...,a_(t-1),o_t)` and may receive judgments `j_t` about transitions or goals. A physical system obeys an unknown transition law. We distinguish:

- **Observation:** `o_t`, in our car usually measured position `x_t`.
- **Intervention:** chosen motor command `a_t`.
- **Transition rule:** how the physical state changes under the chosen command.
- **Judgment/goal:** external information about what is desirable, when supplied.
- **Representation:** `z_t=phi(h_t)` summarizing the history.
- **Model class:** the allowed expressions/distributions for laws; always an inductive bias.
- **Experiment-selection rule:** how the agent chooses which intervention to try.
- **Action-selection rule:** how the agent pursues a supplied goal using its knowledge.

There is no free inference of human goals from physics alone. In two worlds with identical physical trajectories and opposite judgment maps, all observations lacking judgments are indistinguishable.

## 3. The benchmark car: enough complexity without unnecessary complications

**Main V1 environment:** a friction/momentum car on an unrestricted 1D line. It has position `x_t`, hidden velocity `v_t`, and motor command `a_t`:

```text
v_(t+1) = alpha*v_t + beta*a_t
x_(t+1) = x_t + v_(t+1)
o_t     = x_t
```

The agent sees `x` and remembers action histories, **not** a direct velocity sensor. `alpha` is momentum persistence; `beta` is motor authority. Simulation values: `alpha=0.8`, `beta=1.3` unless otherwise stated. When velocities are formed from consecutive positions, `v_t = x_t - x_(t-1)` under these update conventions.

We choose semi-implicit position integration because it keeps the observation-only expression concise. The selection is a modeling convention, **not** a principle of learning.

**Minimal variants to expose specific capacities:**

- B0: `x_next=x+a`, no memory needed; easy to memorize a button; insufficient benchmark by itself.
- B1: unknown preference over displacement, judgments supplied intermittently; distinguish `a>0` from `x_next>x`.
- B2: momentum/drag with position-only observation (main benchmark); forces useful history.
- B3: unknown physical parameters; requires informative interventions.
- B4: changing target `g`; examine transfer via `e=g-x`.
- B5: velocity-dependent nonlinear drag `v_next=0.9v-0.15v|v|+1.1a`; inspect expression discovery and extrapolation.
- Later, only when B1-B5 are solved honestly: partial noisy observation, stochastic disturbances, a simple pendulum/cart-pole, then higher-dimensional articulated robots.

## 4. Precise mathematical findings

### F01. Finite evidence does not entail a unique universal law [PROVEN]

For observations with all starting positions below 10, these two hypotheses fit equally well:

```text
H1: Good iff x_next > x
H2: Good iff x_next > x and x < 10
```

No finite set that excludes starting positions at or above 10 can select between them. Simplicity, symmetry, grammar limitations, a prior, or new experiments are necessary. It is incorrect to claim a completely assumption-free universal rule discoverer.

### F02. Current observation may not be a predictive state [PROVEN]

For `alpha=0.8`, `beta=1.3`, at `x=5` with `a=0`:

```text
v=+1 -> x_next=5.8
v=-1 -> x_next=4.2
```

The same `(x,a)` maps to two outcomes. A deterministic predictor based only on `(x,a)` cannot be exactly correct in both cases. Distinguishing the preceding positions (or their difference) suffices in this toy:

```text
d_t = x_t - x_(t-1)
x_(t+1) - x_t = alpha*d_t + beta*a_t
```

This is not proof that velocity is the uniquely correct latent coordinate.

### F03. An exact predictive state can be defined by future experiments [PROVEN as a definition]

Histories `h` and `h'` are equivalent if, for **every** permitted future action sequence `A` and future observation sequence `Y`, they imply the same future distribution:

```text
h ~ h'  iff  P(Y | h, do(A)) = P(Y | h', do(A))  for all A,Y.
```

The equivalence classes partition histories into the coarsest classes that preserve *all* these interventional predictions. If two histories produce distinguishable distributions for even one available test, any exact all-tests predictor must distinguish them.

Caveats:
- This is a mathematical target, not a finite-sample algorithm.
- It depends on which observations and future experiments are admitted.
- Finite data, noise, untested action sequences, and computational limits prevent directly computing the universal partition.
- For a narrower goal family or finite horizon, a coarser quotient may suffice.

Connections: predictive state representations (Littman, Sutton, Singh), causal states (Shalizi, Crutchfield), automata state minimization, bisimulation.

### F04. For this linear car, whether velocity is *needed* has an exact rank criterion [PROVEN]

Write `s=(x,v)^T`, `s_next=A s+B a`, `x=C s`, with

```text
A = [[1, alpha], [0, alpha]]
B = [[beta], [beta]]
C = [[1, 0]]
```

The two-step observability matrix is

```text
O = [ C ; C A ] = [[1,0], [1,alpha]]
```

and `det(O)=alpha`. If `alpha != 0`, the physical 2D state is observable from position measurements with known inputs. If `alpha=0`, the current velocity does not influence future dynamics; carrying it separately is unnecessary for future physical position predictions. For `beta != 0`, the controllability matrix `[B,AB]` has determinant `-beta^2` and full rank. Thus for this linear model, `alpha != 0` and `beta != 0` give a minimal two-dimensional observable/reachable realization. This is a special-case conclusion, not a general claim about nonlinear systems.

**Key philosophical point:** an internal variable is warranted by its role in distinguishing futures, not by a human insistence that the variable must exist in the model.

### F05. Parameter recovery requires experimentally independent variation [PROVEN]

From velocity observations:

```text
[v'_1, v'_2]^T = [[v_1,a_1],[v_2,a_2]] [alpha,beta]^T
```

Two noiseless experiments identify `(alpha,beta)` if and only if

```text
det([[v_1,a_1],[v_2,a_2]]) != 0.
```

A concrete experiment pair `(v,a,v')=(1,0,0.8)` and `(0,1,1.3)` recovers `(0.8,1.3)` exactly. If every action satisfies `a=2v`, the design matrix has rank 1 and the data determine only `alpha+2beta=3.4`. The parameter pairs `(0.8,1.3)` and `(3.4,0)` are indistinguishable under these restricted inputs.

No number of repeated rank-one measurements uniquely identifies two independent parameters under this model. This is the finite-dimensional version of the need for informative system excitation. Stochastic/noisy estimation additionally needs conditioning and sufficient samples.

### F06. The "true" coordinates are not uniquely recoverable [PROVEN]

Let `z_next=F(z,a)` and `o=O(z)`. For any invertible `psi`, let

```text
z_tilde       = psi(z)
F_tilde       = psi o F o psi^(-1)
O_tilde       = O o psi^(-1)
```

Then the transformed model produces the same observations under every action sequence. For example `(x,v)` and `(x,2v)` are equally valid coordinate systems once their equations are transformed. A purely behavioral learner cannot infer a unique privileged gauge or name for the hidden variables.

To select interpretable coordinates we may impose convenience, translation symmetry, sparsity, units, locality, separability, or other inductive biases. These choices must be documented, not portrayed as consequences of observations alone.

### F07. Prediction sufficiency depends on the goal [PROVEN by counterexample]

Assume `v=0`, `alpha` irrelevant for this step, unknown `beta` is either `0.5` or `1.0`, and available actions are `{1,2}`.

- Goal A, *move positively*: both actions work under both models. Identifying beta is unnecessary to succeed once.
- Goal B, *move exactly one unit*: beta=0.5 requires action 2; beta=1.0 requires action 1. The optimal action depends on the unknown law.
- Executing action 1 once and observing displacement 0.5 or 1 distinguishes the two models.

Define candidate optimal action sets `A*_M(z,g)`. If their intersection across current candidate models is nonempty, a universally satisfactory action exists for this goal (given the chosen success criterion). If the intersection is empty, information about the model may change the decision. This motivates **goal-relevant** experiment selection, not learning every parameter for its own sake.

This connects to decision-aware/value-equivalent modeling. It is not a claim that information-seeking is always the correct next action; it is a rule for identifying relevant uncertainty.

### F08. Good transitions do not necessarily increase a state potential [PROVEN]

Suppose a 1D shuttle is praised for *moving*, so both `0->1` and `1->0` are Good. A state-only progress potential `U(x)` satisfying `U(x_next)>U(x)` for every Good transition would require `U(1)>U(0)` and `U(0)>U(1)` simultaneously, impossible.

This refutes **state-potential increase as a universal description of goodness**. It does not refute rewards: the transition reward `R(x,x_next)=1[x_next != x]` encodes the labels perfectly. A reward may also depend on action or history. Distinguish a restricted type of value function from a general transition reward.

### F09. Predictive disagreement and labeled disagreement are different [PROVEN conditionally]

Candidate judgment rules:

```text
H1: Good iff a>0
H2: Good iff x_next-x>0
H3: Good iff x_next-x>0 and x<10
```

Judged transitions under main car physics:

```text
(x,v,a)=(0,0,1)  -> displacement +1.3, Good: H1,H2,H3 survive
(x,v,a)=(20,0,1) -> displacement +1.3, Good: H3 rejected
(x,v,a)=(3,2,0)  -> displacement +1.6, Good: H1 rejected
```

H2 is the only survivor among the **listed** hypotheses. No claim follows about all conceivable rules. The judgment is provided externally; it is not entailed by physics.

### F10. The target "reach and stop" requires looking beyond immediate movement [PROVEN for toy]

For the frictionless semi-implicit variant:

```text
v_next=v+a
x_next=x+v_next
allowed a in {-1,0,1}
start (x,v)=(8,1); goal (x,v)=(9,0), exactly 2 steps
```

Enumerating 9 two-action sequences yields exactly one: `(0,-1)`. Translating all positions and the goal by +100 yields the identical action sequence from `(108,1)` to `(109,0)`.

This demonstrates the utility of a goal-relative representation `e=g-x` and planning over consequences. It does not show an agent discovered such a representation independently; the exhaustive planner was supplied.

### F11. A small nonlinear extension can be attacked with the same program-search principle [TESTED]

Simulated law:

```text
v_next = 0.9*v - 0.15*v*abs(v) + 1.1*a
x_next = x + v_next
```

The feature candidates were `[v,a]`, `[v,v^2,a]`, `[v,v^3,a]`, `[v,v*abs(v),a]`. Train on `v in [-1.5,1.5]`; test on wider `[-2.5,2.5]`.

- Linear model out-of-range RMSE: approximately 0.1984.
- Cubic approximation out-of-range RMSE: approximately 0.1154.
- The correct `v*abs(v)` library member: RMSE approximately `1.06e-15` and coefficients `(0.9,-0.15,1.1)`.

This does **not** prove that an arbitrary nonlinear physical law can be recovered with a finite grammar. It demonstrates why choosing or growing the candidate function library is a real foundational design problem.

## 5. Empirical ledger: exact V1 tests

Code: `car_learning_benchmark_v1.py`  
Data: `car_learning_benchmark_v1_results.json`  
Dependencies: Python 3, NumPy.  
Reproduce: `python car_learning_benchmark_v1.py > car_learning_benchmark_v1_results.json`  
RNG seed: `20261020` for primary experiments and `20261021` for the nonlinear experiment.

All experiments are synthetic, deterministic, and built from known benchmark equations. Independent random tuples of position, velocity, and action are sampled using a simulator reset, so the fitting set is not a single correlated driving trajectory.

**Displacement prediction using 200 training and 1,000 test transitions:**

| Learned predictor input | Held-out RMSE | Interpretation |
|---|---:|---|
| Motor action alone | 1.399674 | Fails to account for momentum |
| Current position + motor action | 1.407207 | Position-only state is inadequate |
| Current + prior position + motor | 2.83e-15 | Enough history to reconstruct relevant momentum |
| `(current - prior)` + motor | 1.18e-15 | Compact derived coordinate sufficient under the selected dynamics |

A grammar of simple single-feature expressions selected `x_t-x_(t-1)` **or its negative** when paired with an action term. Both signs are behaviorally equivalent after coefficient adjustment. This is another demonstration of coordinate non-uniqueness.

**Parameter excitation:** full-rank two-probe design recovered `(alpha,beta)=(0.8,1.3)` exactly. Confounded probes `a=2v` have rank 1; alternative `(3.4,0)` makes identical predictions on those probes.

**Goal transfer:** exact two-step action pair `(0,-1)` worked for both `(8,1)->(9,0)` and the translated `(108,1)->(109,0)` with the same dynamics and shifted target. This is a planned demonstration, not learned policy transfer.

**Numerical tests:** the companion script asserts the expected prediction and rank properties, rule elimination, goal-planning uniqueness, and nonlinear recovery. Floating-point errors at `1e-15` should be interpreted as numerical zero.

## 6. Three rival foundation-level approaches, not yet collapsed

### Candidate C1: Pure prediction plus goal-directed planning [RETAIN as baseline]

Learn an action-conditional predictive model of observations, then plan toward externally supplied goals. Advantage: physical knowledge can transfer across tasks. Weakness: exact prediction of every physical detail may be unnecessarily expensive, and the model cannot identify desirability without goal data.

### Candidate C2: Counterexample-driven symbolic state and law discovery [RETAIN as main experimental hypothesis]

Maintain candidate expressions/models. Seek distinct histories or interventions that falsify current representations. When predictive aliasing occurs, generate additional features from a grammar. Prefer compact models that generalize on holdout experiments. Advantage: transparent, falsifiable, feasible to test precisely on simple systems. Weakness: grammar bias and combinatorial explosion; exact falsification fragile under noise; no demonstrated scalability.

### Candidate C3: Task-conditioned predictive abstraction [RETAIN as substantive alternative]

Preserve only distinctions required to choose appropriate actions for specified goal families/horizons. Advantage: can discard task-irrelevant physics. Weakness: risks being too narrow when a novel goal requires a previously discarded distinction. Need quantify the tradeoff by systematic goal transfer.

**Reward maximization, imitation learning, world models, in-context prompting, active inference** should be included later as implemented baselines when a concrete benchmark can discriminate them. Mathematical re-expression of a successful policy as a reward optimizer is not by itself evidence about its learning efficiency or representation quality. Equally, C1-C3 can be implemented within reward-based algorithms; new names are not novelty.

**Comparison requirement:** same action budget, access to sensor data, access to judgments, candidate grammar/inductive structure, compute budget, and held-out regimes where feasible. Do not compare a hypothesis-rich symbolic agent to a deliberately crippled black-box model and call the result a discovery.

## 7. Working algorithm: intervention-driven representation refinement [HYPOTHESIS]

```text
Input:
    observations o; permitted actions A; occasional judgments j or goals g
    expression grammar G (explicit inductive bias)
    hypothesis set or parametric model family M
Initialize:
    smallest feature set z = phi(h) (e.g. current position)
Loop:
    1. Predict future observations conditional on candidate actions using M.
    2. Check for state aliasing: similar z and action, incompatible outcomes.
    3. If aliasing persists, search G for history-derived features that
       separate the conflicting cases without harming held-out prediction.
    4. Maintain several compatible laws rather than only one when data
       do not identify a unique model.
    5. For the current goal, determine whether all plausible models
       admit a common satisfactory action.
    6. If a common action exists, it may be taken without full ID.
       If not, find an accessible experiment that distinguishes
       goal-relevant models, subject to specified experimental costs.
    7. Execute, observe, acquire any available judgment, revise beliefs.
    8. Retest on held-out conditions and revise the representation.
```

**Known omissions:** no general guarantee of termination, efficient grammar generation, approximate state splitting under noise, optimal experimental design, or scaling to images. No claim of a novel algorithm. Some components overlap established active learning, symbolic regression, system identification, and planning.

## 8. What would falsify or improve the working hypothesis?

**Priority experiment E1 (state discovery):** Supply only `(x_t, a_t, judgment when available)` and finite history. Allow the algorithm to grow expressions such as `x_t-x_(t-1)` from a declared small grammar; compare against fixed-memory and fixed-state baselines. Criterion: whether it autonomously proposes a predictive feature and improves held-out causal prediction.

**E2 (identifiability-aware action selection):** Compare passive action sequences to experiment-selection rules for learning `alpha,beta`, using equal numbers of experiments and identical candidate sets. Metric: parameter error and time to distinguish models. Deliberately include a confounded action sequence that produces rank-deficient data.

**E3 (coordinate/generalization):** Train at positions around zero; test shifted origins and a changed destination. Check whether the learned expression rather than just the policy generalizes. Rotate the representation by a known invertible transformation and check if performance remains equivalent.

**E4 (task relevance):** Two possible motor gains produce the same adequate actions for a coarse progress goal but different actions for an exact displacement goal. Compare full identification versus task-aware identification on experiments needed to succeed. Do not assume task-aware ID always wins once a distribution over future tasks is considered.

**E5 (nonlinear capacity):** With the same small grammar, add velocity-dependent drag; then add a nonlinear term **outside** the grammar. The latter should falsify any claim that the grammar is universally sufficient. Compare grammar growth, fixed sparse libraries, and a generic nonparametric predictor, under equal evidence.

**E6 (judgment-rule inference):** Maintain multiple rules for goodness (command-positive, actual-displacement-positive, position-limited). Choose experiments that make their predictions disagree. Count external judgment queries. The judge remains an independent data channel.

**E7 (novel goal transfer):** Test abstract representations over goal families with varying precision, horizons, and reversals. A representation minimal for the old task may fail a new task. This should guide whether full predictive and task-conditioned models need to coexist.

## 9. What this ledger does *not* establish

- That maximal compression, predictive state equivalence, or information gain alone defines all intelligent behavior.
- That a universal minimal representation exists independent of observations, interventions, horizons, and tasks.
- That the symbolic grammar itself can emerge without built-in inductive biases.
- That a self-built mechanism outperforms well-designed reinforcement learning, self-supervised world models, or any existing baseline.
- That the toy demonstration scales to uncertain nonlinear robotics or raw visual observations.
- That a preference label on actions, transitions, or trajectories is automatically representable by a monotonic state potential.
- That active experiments always improve task performance enough to justify their cost.

## 10. Research links (primary sources and directly relevant technical material)

These references supply context and guard against accidental claims of novelty. They are **not** presented as proof that the proposed V1 agent is superior.

1. **Littman, Sutton, Singh, _Predictive Representations of State_** (NeurIPS): action-conditional future observations as state. https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
2. **Shalizi and Crutchfield, _Computational Mechanics: Pattern and Prediction, Structure and Simplicity_** (2001): equivalence of pasts by future predictions and statistical complexity. https://doi.org/10.1023/A:1010388907793
3. **Angluin, _Learning regular sets from queries and counterexamples_** (1987): active rule identification using counterexamples with strong assumptions about query access. https://doi.org/10.1016/0890-5401(87)90052-6
4. **Ljung, _System Identification: Theory for the User_, 2nd ed.** (1999): model class, data design, identifiability, validation. https://www.mit.bme.hu/system/files/oktatas/targyak/9132/Ljung_L_System_Identification_Theory_for_User-ed2.pdf
5. **Tedrake, _Underactuated Robotics_, System Identification chapter**: accessible robotics-oriented discussion of model learning. https://underactuated.mit.edu/sysid.html
6. **Rissanen, _Modeling by shortest data description_** (1978), and **Grunwald and Roos, _Minimum Description Length Revisited_**: explicit representation/compression assumptions. https://doi.org/10.1016/0005-1098(78)90005-5 ; https://arxiv.org/abs/1908.08484
7. **Schmidt and Lipson, _Distilling Free-Form Natural Laws from Experimental Data_** (2009): symbolic-law discovery from physical measurements. https://doi.org/10.1126/science.1165893
8. **Brunton, Proctor, Kutz, _Discovering governing equations from data by sparse identification of nonlinear dynamical systems_** (2016): sparse dynamics identification. https://doi.org/10.1073/pnas.1517384113
9. **Locatello et al., _Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations_** (ICML 2019): need for inductive biases; latent identification limitations. https://proceedings.mlr.press/v97/locatello19a.html
10. **Grimm et al., _The Value Equivalence Principle for Model-Based Reinforcement Learning_** (NeurIPS 2020): predictions relevant to planning, not necessarily full physics. https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html
11. **Silver, Singh, Precup, Sutton, _Reward is enough_** (2021): a competing unifying hypothesis, explicitly presented by the authors as a hypothesis. https://doi.org/10.1016/j.artint.2021.103535
12. **Bongard, Zykov, Lipson, _Resilient Machines Through Continuous Self-Modeling_** (2006): a real robot inferring its own body model from motor-sensation data. https://doi.org/10.1126/science.1133687
13. **Classical linear realization/observability reference**: observability, reachability, state minimality. https://doi.org/10.1016/S0377-0427(00)00341-1

## 11. Instructions for the next AI / next ledger release

1. **Read all of V1 before extending.** First check whether an alleged new idea is already F01-F11, the research links, or a corrected old mistake.
2. **Create a new filename for each release:** `first_principles_robot_learning_ledger_v2.md`, then `_v3.md`, etc. Never overwrite a released version.
3. Copy V1 into V2 as historical material or retain all its contents through a well-labeled versioned history. Mark entries `SUPERSEDED` with a reason and replacement. Do not silently delete failed ideas, corrections, or negative results.
4. Put a concise *version delta* at the beginning of each new release: new theorems, empirical evidence, corrections, rejected hypotheses, new alternatives, outstanding problems.
5. Maintain the status labels from Section 0. The distinction between a theorem, a simulation, a conjecture, and an assumption is mandatory.
6. Any new numerical result must name the environment equations, observation channel, action set, sample size, random seed, train/test split, and code or sufficient reproducibility instructions.
7. Do not claim novelty until checking relevant primary literature. Cite the closest competing approaches, not only supportive work.
8. Prioritize **new discriminatory experiments** over adding more terminology. An experiment is useful if it could prove one proposed mechanism wrong or distinguish it from a serious competitor.
9. Continue expanding the car only when the simpler version cannot reveal the property being investigated. Prefer changing exactly one source of difficulty at a time.
10. Main unresolved question: **How can an agent autonomously enlarge its representation language when no expression in the current grammar is adequate, and how can it tell the difference between insufficient features and an inadequate dynamics model?**

### End of V1
