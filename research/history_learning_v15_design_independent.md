# Independent design challenge: learning useful history in continuous motion

Date: 2026-10-20. Latest published ledger isv14; the supplied-slope history pilot is a validated instrument, not the intended learning milestone. This note proposes a bounded next experiment, not canonical settings or performance claims.

## What uncertainty is worth resolving?

The consequential question is whether ordinary regression/model selection can acquire a reusable distinction in continuous experience that changes sustained decisions, rather than being handed that distinction as a slope/state. A validation-selected integer lag is a conventional instance of this, not autonomous discovery of a universal observation language. If it suffices here, accept that result; if it fails, diagnose whether the failure is information, finite representation, acquisition coverage or decision optimization.

An important alternative framing is that the body already supplies sufficient information in a very short history, and the challenge is statistical use rather than inventing new state. In the deterministic smooth second-order plant with held torque and unwrapped positions, backward flow from(qt,vt) under the known previous command has

    q[t-1]=q[t]-dt*v[t]+O(dt^2),
    derivative(q[t-1])/derivative(v[t])=-dt*I+O(dt^2).

For sufficiently small dt on a bounded smooth region this derivative is invertible. The inverse-function theorem locally reconstructs vt from(q[t-1],q[t],u[t-1]); then q[t+1] is a function of those quantities and candidate u[t]. This does not prescribe velocity as a learned variable, prove a global embedding, hold with wrapped positions/high-speed ambiguity, or imply finite histories are exact sufficient statistics under sensor noise. It does explain why an ordinary short-delay predictor is a serious competitor and why adding many lags alone is not a fundamental advance. Actuator memory would change this information problem, but should not be introduced merely to manufacture a favorable result.

## Smallest useful controlled comparison

Use the same continuous acquisition streams for every learner: noisy joint positions, issued bounded commands and sampling times; no velocities, accelerations, mechanics formula, evaluator rewards or exact current preparation. All ordinary controllers get the same sensor history at test time. Supply/reset only a COMMON initial condition if needed to start each stream, then change goals without resetting physical state or controller memory. Record this preparation capability separately. No goal-change oracle velocity is available.

Fit the same regularized nonlinear conditional predictor at a small prescribed lag set such as1,2,4,8 samples. Current-pose-only(h1), a fixed short-history competitor, and validation-selected history should differ only by their allowed inputs; use the same scaling, hyperparameter budget, planning cost, solver, constraints and fallback. Histories must include past ISSUED actions. Omitting past actions confounds hidden velocity with a known input's acceleration effect. Declare any supplied periodicity, displacement coordinates or action-affinity prior; normalize only using training records.

At minimum add one strong conventional alternative: a causal filtered finite-difference/observer estimate followed by the same nonlinear learned consequence relation and same planner. Tune its filtering and regression from the SAME training/validation streams. It receives no true velocity or mechanics. This tests whether finite-delay selection does more than approximate an already simple conventional motion estimator. A generic weak fixed PD alone is inadequate. A true-dynamics/true-state controller may separately diagnose task feasibility or optimizer limitations, but remains privileged.

The most decisive minimal matrix is current-only, fixed short history, selected history and the filtered conventional alternative. Extra architectures are not needed before these have been fairly evaluated. An episodic trajectory-retrieval controller is a genuinely different possible next formulation (rank logged future action segments by consequences at matched causal histories), but its coverage/action-sequence restrictions must be explicit; do not add it merely to multiply baselines.

## Prediction is a diagnostic; sustained decisions are the test

Select widths/ridges/history using independent continuous VALIDATION streams or contiguous heldout blocks, with boundary guards at least the largest lag plus evaluated rollout horizon. Randomly splitting adjacent overlapping windows leaks almost identical sensor/physical histories and gives optimistic generalization. Cluster uncertainty over independently acquired streams, not adjacent windows. Shared-data pairing is appropriate across methods.

Prescribe two predictive diagnostics: one-step observation error and open-loop multistep position error under the LOGGED future actions at a meaningful common horizon. The latter uses real future observations only for scoring, never as inputs after rollout initialization. It has no controller-induced data advantage and reveals whether teacher-forced one-step selection is sufficient for recursive use. One selected-history rule is enough initially, but decide before outcomes whether it uses one-step or multistep validation. Comparing both fixed rules later is an interpretable diagnostic if this distinction matters, not permission for an optimizer ladder.

Run sustained new-goal feedback episodes with identical initial conditions/noise per paired learner, switched goals without reset, and fixed timing/bounds/cost. Primary cost should be the actual declared tracking/effort objective; report large errors, constraint/fallback use and finite-horizon settling separately. Do not require a forced unannounced coast or future goal knowledge. If the planner is given a task deadline or reference preview, every relevant comparator receives it. Endpoints need no special stopping claim. Evaluate sufficiently many contiguous transitions after goal switches that ignoring motion cannot pass by matching the current position.

Training commands need sufficient excitation over the tested pose/speed/action region. Uncontrolled random torque in a frictionless body can produce an expanding velocity distribution; a bounded observation-only acquisition stabilizer plus dither is a permissible supplied resource if common and explicit. That stabilizer cannot then be sold as learned competence. Separate an in-support sustained task from any unsupported transfer condition; do not silently infer identifiability from a richly populated pose plot while velocity/action coverage is missing.

## Statistical and causal interpretation

With fresh independent zero-mean endpoint sensor noise, raw NEXT-position squared loss estimates observation prediction risk up to a constant output-noise term conditional on the causal history. Input noise is part of the conditional prediction problem; one need not pretend that noisy lagged inputs are exact physical states. Subtracting current position to form a residual target is algebraically valid for conditional prediction when that known offset is restored, but it shares sensor noise with inputs: the v14 independent-response EIV formulas cannot simply be copied. A supplied persistence/ballistic residual connection is an inductive bias, not proof of unbiased physical dynamics estimation.

For genuinely nested information summaries S contained in H and an unrestricted Bayes predictor, the minimum squared prediction risks satisfy

    R(S)-R(H)=E||E[Y|H]-E[Y|S]||^2 >=0.

Finite fitted/regularized model risks need not be monotone in history length. A selected lag is an empirical risk-optimal choice within a supplied family/budget, not proof of the physically minimal necessary memory. A long lag may denoise sensors rather than reveal additional mechanical state. A small noiseless-sensor diagnostic can distinguish those explanations if that ambiguity becomes consequential; it need not become a separate release.

Logged action endogeneity is harmless only when the conditioning history contains what the acquisition policy used and the desired intervention is supported. For an observation-based randomized acquisition policy, conditional prediction under a chosen current action can transfer causally within support because no additional hidden action-selection information is supplied. A predictor trained where particular actions occur only at particular histories may be unconstrained for the planner's counterfactual actions. Independent randomized dither and action-support diagnostics matter more than generic in-sample fit quality.

Recursive planning with conditional mean observations is certainty equivalence. Nonlinear uncertainty does not propagate by repeatedly substituting means, and a noisy-history predictor trained on sensor histories may be evaluated by the planner on unrealistically noise-free predicted histories. This is a declared approximation, not a Bayes-optimal control method. A poor controller despite good one-step prediction is scientifically informative and must not trigger repeated architecture changes before coverage, recursive rollout error and solver quality are checked.

## Relevant established work and stopping the detour

Romeres et al., *Derivative-free online learning of inverse dynamics models* (2018), explicitly compares learned position-history features with derivative-based alternatives on real iCub-arm data. Its causality discussion distinguishes prediction from feedforward use of known desired trajectories. It is stronger existing evidence that the history idea itself is not novel; our supplied-family continuous comparison would not reproduce that full method or establish superiority. Primary paper: https://arxiv.org/pdf/1809.05074 (read for this audit).

One complete continuous same-experience comparison can establish conventional sufficiency or a concrete failure requiring revision. Its scientific value is connecting acquired history distinctions to sustained action, not improving another covariance theorem or naming an internal state. After the bounded experiment, decide what evidence actually changed about general learning and redirect if only kernel tuning remains.
