# Independent design assessment: useful history in continuous coupled control

2026-10-20. This is a design critique, not an experimental result. No canonical experiment source or settings are changed here. It follows the verified v14 ledger and audited preliminary history pilot. The proposed scientific question is consequential: can experience select a causal summary that predicts interventions and enables sustained goal behavior? It should not become another demonstration that a scientist-supplied slope approximates velocity.

## First-principles requirements

The controller's information is the measured position/action history and the currently announced goal. A proposed summary is useful only relative to a specified consequence and decision, not because it resembles a named physical state. Two candidate summaries can predict similarly on a collector's trajectories while disagreeing under the actions chosen by a new controller.

Action history matters even without actuator memory. For an exact zero-order-held double integrator with sampling interval D,

\[
q_{t+1}=2q_t-q_{t-1}+\tfrac12D^2B(u_{t-1}+u_t).
\]

Thus even the noiseless two-position/current-action summary can omit relevant information if the preceding action is hidden. All serious delay models should have causal issued-action history available. A short position history alone is not automatically an exact Markov state; noise, unobserved disturbances and limited windows generally leave uncertainty. Calling a delay vector a “state” does not establish sufficiency.

For nested summaries S=f(H), the population squared prediction risks of their optimal conditional means obey

\[
R(S)-R(H)=E\|E[Y\mid H,U]-E[Y\mid S,U]\|^2\geq0.
\]

This is an elementary projection identity, not a claim that finite fitted models attain either risk. A richer nonlinear regressor may improve because of approximation or regularization rather than an isolated information gain. The aliasing pilot supplies a separate mechanistic example; the new experiment must establish actual acquired usefulness.

## Acquisition and validation hazards

Prefer a shared prerecorded exogenous bounded excitation bank for the first intervention comparison. A stabilizing collector using past positions to estimate motion makes current issued torque informative about that motion. A pose-only conditional predictor can exploit that association and fail when a planner varies the torque counterfactually. Adding independent dither does not by itself remove omitted-history confounding. A true-state-assisted safety collector is additional privileged information and must not be hidden.

If exogenous continuous excitation drives broad motion, retain its full trajectories and actual input bounds, rather than discarding inconvenient segments. The body has no joint limits and finite bounded torque does not cause finite-time energy blowup, but numerical feasibility, sampling, velocity coverage and winding conventions still need checks. If safety intervention changes commands, record actual commands and intervention rules. Initial resets may be used at separately counted episode starts; resetting filters/physical state at goal changes defeats the desired test.

Use whole trajectories or separated temporal blocks for train/validation/test; overlapping lag windows share sensor errors. A gap of at least the largest history plus forecast horizon removes exact window overlap, although it does not create independent physical trajectories. Normalization, landmarks and compressed features must be fitted on training only. Selection of lag, rank, ridge and filter constants uses validation only; test worlds, goals and precision results remain unused.

One-step prediction from noisy history to the next raw measurement has an operational conditional-risk interpretation because the next sensor error is independent of current causal features. A residual target such as q[t+1]−q[t] instead shares current sensor noise with its inputs; treat it as a conditional prediction problem, not uncritically as a noiseless structural acceleration regression. For counterfactual multi-step validation, exogenous future action blocks are particularly clean. Conditioning on a complete feedback-generated future command sequence can encode future sensing noise. Future goals/noise must not initialize summaries or planners.

## Strong simple conventional alternatives

The minimal core comparison should include a current-position/action ablation, a fixed reasonably rich position/action delay model, and a selected-history version. Selection should have a genuinely nested current-only option, rather than presupposing that more history must win. A linear MIMO ARX/delay model is a cheap serious diagnostic: full cross-joint coefficients, train-only scaling, ridge/order selection and a common controller. It may be adequate for slow small motion and should be allowed to win.

A generic nonlinear NARX model on causal position/action delays is the stronger default. Kernel/RBF regression or a small conventional MLP is acceptable; fixed-history and selected-history versions should share prediction targets and decision rules. Predictive validation must include autonomous multi-step behavior, because one-step persistence can look excellent at small sampling intervals while learned rollouts are unstable. Supplying a second-difference integrator skeleton changes prior knowledge and should be disclosed if used.

Learned linear compression of a full history, with current position retained and remaining projections optimized from training/validation prediction, is a serious alternative to raw-delay order selection. It need not be interpreted as discovered velocity. Reduced rank, projection regularization and optimizer starts must be retained. A nonlinear latent observer is another possibility, but is not necessary merely to make this first comparison larger. A model-informed EKF or moving-horizon estimator is a meaningful structured competitor only if its extra dynamics/order/noise priors are explicit and it uses measured positions/issued commands online. An observer given the true inertia and a raw-delay learner given none is a prior comparison, not equal-information evidence for observers.

A conventional causal filtered-position PD controller is also important. It can be a fixed, openly supplied baseline, with any tuning charged to training/validation or declared virtual compute. It must not be tuned on heldout hidden-state scores. A strong result may be that such a simple controller is sufficient, in which case the new learned mechanism has not earned a capability advantage. A weak arbitrary PD gain is not an adequate competitive baseline.

## Feasibility before performance interpretation

There is a simple privileged fixed-goal feasibility controller for this body. Let e=q−g and

\[
u_i=-a\tanh(k_p e_i)-b\tanh(k_d v_i),\quad a+b\leq1,\quad a,b,k_p,k_d>0.
\]

For true velocity available only to a diagnostic, define

\[
V=\tfrac12v^TM(q)v+\sum_i(a/k_p)\log\cosh(k_p e_i).
\]

The mechanical energy identity gives `Vdot=−b sum(v_i tanh(kd v_i))<=0`. In the fixed-goal, ideal continuous, fully actuated instrument, uniform positive inertia and a proper potential give compact sublevels; the only invariant subset with v=0 has e=0. LaSalle therefore supplies asymptotic fixed-goal regulation within the torque bounds. This uses direct generalized torque/work structure but no numerical inertia formula in the controller. It does not certify a chosen finite switch deadline or noisy filtered-position implementation. Those require actual sustained-task checks. It provides a useful simple alternative to declaring arbitrary goals impossible or adding a privileged numerical optimizer by default.

All ordinary methods should receive the same current goal and history, same initial observation protocol, same sample interval, bounds and announced horizon/deadlines. Filter/history memories continue through goal switches. Position and late precision/velocity outcomes should both be retained; average tracking alone already failed to describe stationkeeping in v13. Goal conventions must say whether joint winding matters or errors are periodic. Hidden truth may score outcomes and diagnose feasibility, but may not tune ordinary control online.

## Relevant established research

[Romeres et al.2018](https://arxiv.org/pdf/1809.05074) explicitly studies raw, weighted, learned reduced-rank and structured position-history features for inverse-dynamics learning, with hyperparameters chosen by validation or marginal likelihood. Its reduced-rank construction is directly relevant to a serious compressed-history competitor; an arbitrary fixed slope is weaker. This is precedent, not a requirement to inherit its physics priors or reproduce its experiments.

[Dalla Libera et al.2020](https://arxiv.org/html/2002.10621) learns transitions from position histories using derivative-free GP models and physically informed kernels, with ball-and-beam/Furuta experiments and control. It supplies a closer precedent for learned history-to-future-position models than rest inverse calibration alone. Its displayed trajectory-optimization control is open loop, so a sustained noisy feedback experiment would have a different protocol; neither novelty nor superiority follows. Its finite-history Markov wording should not be adopted as an unconditional theorem for noisy arbitrary systems.

[Tedrake's output-feedback notes](https://underactuated.csail.mit.edu/output_feedback.html) emphasize that estimator dynamics affect control and that task-relevant information need not equal a complete physical state. [The system-identification chapter](https://underactuated.mit.edu/sysid.html) distinguishes fitting input/output behavior from learning a model useful for decisions. These are established frameworks to test, not mandatory foundations.

The decisive result is not a selected lag or a low regression error by itself. It is a preserved, independently audited demonstration that the chosen summary improves useful sustained decisions from actual available experience—or an honest negative showing that a simpler conventional alternative is sufficient or the learned summary fails under intervention.
