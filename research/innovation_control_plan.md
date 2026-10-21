# Frozen exploratory screen: innovation memory for learned adaptive control

2026-10-21. This is a new bounded control hypothesis, not a revival of the three
retired prediction screens, a new general framework, or a v18 release. The
independent direction critic endorsed one screen with conventional adaptation
and absolute competence gates. Planning began approximately 02:07 UTC.

## Operation and first-principles purpose

An agent needs the effects of its actions to use a new actuator arrangement.
Natural action–motion correlation mixes intervention and state-dependent drift.
Use an independent, known Gaussian probe xi in the action preactivation:
z = mu_theta(observation, memory) + .35 xi + sigma_theta epsilon; u=tanh(z).
xi and epsilon are independent fresh standard Gaussians; the policy chooses mu
before seeing the CURRENT xi. Only past probes enter its observation/memory.
The PPO action likelihood is conditional on xi, with mean mu+.35 xi. Thus the
probe is an external randomized intervention, not a parameter-dependent residual
reconstructed from the actor's sampled action. This matters for correct gradients.

After an observed transition, update H <- .98 H + .02 (delta-v/dt) xi^T.
H starts at zero for each unfamiliar body and persists through its goal changes.
A two-layer 64-unit tanh actor/critic reads H and ordinary observations to choose
actions. Its readout and policy are learned by transition rewards using PPO;
no inverse controller, plant parameters, derivative labels or simulator gradient
are supplied. H itself is a supplied statistical learning bias.

Under conditional independence and smoothness, Stein's identity gives
E[(delta-v/dt) xi_j | history] = .35 E[(d(delta-v/dt)/du_j)(1-u_j^2)|history].
This is a behavior-weighted local sensitivity, NOT a sufficient invariant body
description. Unforced motion cancels in conditional mean, not finite-sample
variance. Smoothing mixes configurations; saturation/delays/nonstationarity can
make H uninformative. No theorem of successful control follows.

The precise hypothesis is a useful simplicity or adaptation advantage from this
explicit statistic in actual feedback control. Generic outer-product memory is
established differentiable plasticity, and sensorimotor covariance adaptation is
established adaptive control. DEP-RL also learns sensorimotor correlations, with
an inverse map encoding sensor–actuator correspondence. No novelty claim is made
for IV/Stein identities, plasticity, PPO, or pairing a statistic with a network.

PREFIT CORRECTION (no fitting/control results yet): the independent math critic
identified that permanent external probing would burden late precision relative
to RLS's finite calibration. Use .35xi ONLY for the first32 steps, then xi=0 and
H is held, not decayed, for the remaining208 steps. This tests initial acquired
response memory reused over goal changes, not continued identification. During
training the learned Gaussian epsilon exploration remains; during validation/test
epsilon=0 (standard deterministic actor evaluation), apart from the first32
external probes. Both learned policies have the same schedule. The actor never
sees currentxi. This correction precedes all fitting and all control scoring.
The displayed identity applies in the active-probe phase; its two sides are zero
when probe magnitude is zero. Persistent-probe formulation is not tested.

## Supplied information, experiment and rivals

Reuse the validated horizontal two-link point-mass plant via
error_transport_data.step (dt .05, eight RK4 substeps). Hide masses, lengths,
damping, signed actuator permutation and gains from all ordinary learners.
Exact q/v sensing, issued actions, previous xi and finite measured delta-v are
supplied. All methods are offered all three future joint goals and their times;
the adaptive reference chooses to use the current goal only.

Each body runs 240 steps (12 seconds), with independent goals in [-.8,.8]^2
lasting 80 steps each, no reset between goals. Initial q in [-.4,.4]^2,
v in [-.2,.2]^2. Train/validation/fresh ID masses and lengths [.8,1.2],
damping [.08,.2], actuator gains [.8,1.2]. Fresh OOD masses/lengths independently
choose [.6,.75] or [1.3,1.5], damping [.03,.3], gains [.6,1.5]. Signs and
permutation are randomized in all splits; this favors the calibration bias and
does NOT establish transfer to arbitrary nonlinear mechanisms or new dimensions.

Per-step cost is mean wrapped joint error squared + .1 mean v squared
+ .01 mean issued u squared, evaluated after the action against its active goal.
Precision requires BOTH joint errors <=.15 rad and BOTH speeds <=.15 rad/s at
ALL last 20 samples of a goal. Score mean cost per body and precise-goal fraction;
retain failures. There are no oracle recovery resets or state clipping.

Rivals:
* A competent history-based policy: shared 64-unit tanh input layer, GRU64,
  actor/value heads, full-episode backpropagation. Same ordinary observations,
  probes, Gaussian action rule, interactions and PPO settings. It can learn the
  correlation statistic but is not given H. Report parameter counts.
* Conventional online affine RLS acceleration identification with forgetting
  .98 or .995, ridge initialization 1, followed by regularized inverse control
  to desired acceleration 4*wrapped-error -4*v. First32 independent uniform
  calibration actions count toward the same240-step budget. It receives no true
  signs/correspondence or physics. Choose forgetting on validation only. This is
  a conventional supplied controller, not the capability claimed acquired by ours.

The recurrent PPO contrast is an EARLY screening comparator, not a full modern
meta-RL benchmark. Current strong accounts include AMAGO/off-policy sequence RL;
a surviving result must face a faithful modern implementation before any broad
competitive claim. Do not call a homemade PPO-GRU pilot SOTA.

## Training, selection and resource boundaries

Candidate and GRU each receive two learning rates (.0003/.001), one declared
training seed, 128 updates of 64 full240-step episodes =1,966,080 interactions
per fit. Total four fits=7,864,320 training interactions /393,216 simulated seconds.
Body/goal draws match across methods/settings; policies induce their own data.
Report meta-training separately from online adaptation; it is not free.

PPO: gamma .99, GAE lambda .95, four epochs, four minibatches of16 complete
episodes, clip .2, advantage normalization, value coefficient .5, global grad
clip .5, Adam eps1e-5, linear LR decay, learned logstd initialized -.7, no entropy
bonus, no physics gradients. Reward is negative declared cost; no task curriculum
or supplied stabilizing controller. Current-goal time and all future goals are
ordinary policy inputs. No parameter/state normalization learned from test.

Select checkpoints at updates32/64/96/128 and rate by mean cost on the SAME32
validation bodies (seed roles separate from training/test). Count all validation
interactions. RLS forgetting uses the same validation set; its PD/ridge/calibration
are fixed above. Freeze selections before inspecting32fresh ID/32fresh OOD bodies.
Matched evaluation randomness for stochastic learned policies; adaptive RLS uses
its declared calibration then deterministic control. Calibration/noise effort is
charged in task cost. Evaluate selected candidate with H zeroed as an experience
ablation, under otherwise matched random numbers, on its own closed-loop paths.

First run tiny likelihood/recurrent-replay/environment fixtures, then benchmark
one training update. Budget up to45 elapsed minutes for all four fits; if measured
runtime exceeds this, prospectively reduce identical update budgets before ANY
substantive fitting or retain an explicitly unfinished checkpoint. Do not truncate
only a competitor. Target first result within two hours of the direction review.
No extra LR, seed, decay, feature-bank or environment rescue after seeing test.

## Frozen continuation decision

Proceed beyond this screen ONLY if candidate precision >=75% on ID and >=50% on
OOD, and its mean body cost is <=90% of BOTH selected rivals on BOTH splits.
Also require candidate mean cost <=90% of its zero-H ablation on BOTH splits.
These are screening thresholds, not significance/general-framework claims.
Success earns fresh-seed/strong-modern-comparator confirmation; it does not
establish a breakthrough. Failure retires this formulation. Preserve mixed wins,
all failures, learning/tuning costs and any implementation intervention.

## Primary methods checked

* Differentiable plasticity, Section2/Equations1–2:
  https://proceedings.mlr.press/v80/miconi18a/miconi18a.pdf
* DEP-RL Sections4.1–4.2 (actual correlation/inverse-map/switching mechanisms):
  https://arxiv.org/pdf/2206.00484
* CleanRL continuous PPO and recurrent minibatching/BPTT source checked directly:
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
* AMAGO primary abstract/source located, full method not yet evaluated:
  https://arxiv.org/abs/2310.09971
  https://github.com/UT-Austin-RPL/amago

Independent critic recommendation: the narrow test is defensible if calibration
handles unknown correspondence, pretraining is visible, and both absolute
competence and consequential advantage are required. It recommends retirement on
failure without adding lag banks, richer features or a new body family.
