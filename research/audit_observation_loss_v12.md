# Independent v12 audit: observation-only improvement and shared-experience fitting

Date: 2026-10-20 (Europe/Berlin). This file derives claims independently and reviews the explicitly unreleased `direct_experience_v12_pilot.py`. It is not independent numerical reconstruction of the pilot or a completed competitive learning comparison. No canonical source or prior ledger was edited.

## 1. Lagged physical timing makes the proposed observation loss usable

**PROVEN UNDER ASSUMPTIONS.** Assume

    v[t+1]=f(v[t])+m[t]
    m[t+1]=h(m[t],v[t],u[t])+xi[t]
    x[t+1]=x[t]+v[t+1],  o[t]=x[t]+e[t].

Sensor errors e[t] are temporally independent, identically distributed, ZERO MEAN, variance R<infinity, and independent of the physical initial state, physical-noise stream and policy random coins. Commands are causal functions of observations/goals/commands through the present time, with no future sensor information. Fixed-horizon expected losses are finite. For comparing normalized goal objectives, goal schedules, time weights and normalization are fixed/exogenous to the policy and sensor errors.

The true x[t+1],v[t+1] depend on commands only through u[t-1] and earlier, and hence on sensor errors only through e[t-1] and earlier. Therefore they are independent of the fresh e[t],e[t+1]. This remains true for history-adaptive policies or within-episode learning rules as long as every action/update is genuinely causal and sensor errors are fresh.

Write a=x[t+1]-g[t], v=v[t+1], d=e[t+1]-e[t]. The difference between observable and true stage losses is exactly

    2*a*e[t+1] + e[t+1]^2 + .6*v*d + .3*d^2.

The expected cross terms vanish and E[d^2]=2R. The issued-action effort cancels pointwise, even though u[t] depends on the current sensor error. Thus

    E[(o[t+1]-g[t])^2+.3*(o[t+1]-o[t])^2+.03*u[t]^2]
      = E[(x[t+1]-g[t])^2+.3*v[t+1]^2+.03*u[t]^2] + 1.6*R.

For T fixed steps and a fixed goal normalization D=1+mean(g^2), the mean normalized episode losses differ in expectation by 1.6*R/D. A fixed exogenous distribution of goal schedules gives a policy-independent expected offset too. Knowing R is unnecessary for ranking expected policies; subtracting it is useful only for numerical comparison with a true-state evaluator. No Gaussian assumption is needed for this identity. Gaussianity matters separately for a Gaussian fitting likelihood.

This supplies an experience-based objective without a physical/predictive model or hidden-state reward. It does not imply that any finite policy class/search rule learns efficiently, that the observation loss is low variance, or that models are unnecessary for every learning problem.

## 2. Assumptions that cannot be dropped

**Counterexample: nonzero sensor bias.** If e has mean mu!=0, the position term adds E[e^2]+2*mu*E[xnext-goal], which depends on the policy. Differencing removes a constant bias in the velocity proxy, but does not repair the absolute-position objective. Merely saying iid finite variance is insufficient.

**Counterexample: current actuation / motor-first timing.** Let x0=v0=m0=0, motor-first v1=beta*u0, and u0=-k*e0. Then

    E[(o1-o0)^2-v1^2] = 2R+2*beta*k*R.

For a bounded policy u0=-k*tanh(e0/sqrt(R)), k within the action bound and R>0,

    E[(o1-o0)^2-v1^2]
       = 2R+2*beta*k*E[e0*tanh(e0/sqrt(R))].

The full position-plus-.3-velocity stage offset at goal0 is therefore

    1.6R+.6*beta*k*E[e0*tanh(e0/sqrt(R))],

which is policy dependent. Independence from the future e1 is not sufficient; correlation with the current e0 is the problem. The bounded cross moment is nonnegative and positive for any nondegenerate centered error distribution. Gaussianity is unnecessary for the counterexample.

Temporally correlated errors can likewise correlate current sensor error with a past-feedback-driven motor state in the lagged system. State/action-dependent sensor errors, sensor/physical-noise correlation, anticipatory policies or adaptive stop/normalization rules can invalidate the constant-offset ranking. With data-dependent episode termination, one cannot simply average the fixed-stage identity over a selected random horizon. The pilot's fixed86-step exogenous goal protocol avoids these issues.

## 3. Finite-fitness noise and winner selection

The identity is about expectations, not equality on individual trajectories. Squared sensor errors, physical position error times sensor error and correlated neighboring displacement errors can make fitness noisy. Finite second moments ensure the expected identity; finite variance of the squared-loss estimator generally needs stronger moments, such as fourth moments. A finite-variance sensor can still produce an infinite-variance squared-loss fitness. Within-episode stage errors are dependent, so uncertainty should use independent episode/fit-seed clusters rather than count every time step as an independent trial.

A policy fixed independently of a FRESH evaluation noise stream obeys the identity. If search repeatedly reuses the same finite noise-seed bank, the selected parameters can depend on that bank's future sensor errors; the selected training score is not an unbiased fixed-policy estimate. Even with fresh noise for every candidate, selecting the minimum noisy score makes its recorded winning score optimistic. Re-evaluating elites on independent episodes or evaluating the final frozen policy on held-out fresh goals/noise is necessary. Offset subtraction does not remove winner's curse. Common random numbers may reduce variance for a predeclared candidate comparison, but adaptive re-use of a bank needs the same caution and is not a free hardware capability.

The pilot uses fresh candidate episodes and fresh test streams and labels winning-fitness optimism explicitly. Its global best can retain an unusually favorable earlier noisy score; this is a search limitation, not hidden-state reward leakage. Testing across fresh independent seeds is necessary before a capability claim.

## 4. Can shadow OE fit the same feedback experience? Yes, with a precise likelihood

**PROVEN UNDER ASSUMPTIONS.** Training process noise is ZERO. Each episode resets to the supplied known physical zero state, sensor errors are iid Gaussian, and all issued commands/observations from every candidate episode are retained. Let theta denote a hypothetical physical parameter. Given the issued causal command prefix, the physical trajectory x_theta is deterministic. Let pi_t be the actual experimenter's causal action rule, including direct-search parameters chosen from earlier collected experience.

The JOINT/SEQUENTIAL density factors as

    p_theta(o[0:T],u[0:T-1])
      = p_theta(o0) * product(t=0..T-1)
          pi_t(u_t | recorded history through o_t)
          * N(o[t+1]; x_theta[t+1](issued command prefix), R).

For a deterministic action rule, the policy factor is the corresponding point mass; on the observed support it still contributes no theta-dependent term. Conditioning on history makes the actual algorithm's chosen action known. Its dependence on earlier observations and earlier candidate outcomes does not make this policy factor a function of the hypothetical physical theta. One does not rerun the experimenter with hypothetical hidden parameters when evaluating the likelihood of the recorded data.

Therefore minimizing the sum of squared raw-position residuals across ALL reset segments is a valid Gaussian sequential/joint maximum-likelihood objective. Derivatives propagate through each segment's deterministic simulated physical dynamics while the recorded commands are held fixed. Feedback does not by itself invalidate this argument. It applies to both lagged and motor-first deterministic plants; the lag restriction is specific to the observation-only velocity-cost identity above.

Crucial distinctions:

- This is NOT a claim that p_theta(observations | the ENTIRE future command sequence) is an iid Gaussian product. Future commands encode earlier observations, and that conditional likelihood has a generally theta-dependent input-distribution normalizer. Full-command exogeneity must not be asserted.
- All episodes must be retained, including failed/nonelite candidates. Keeping only trajectories selected by their later fitness raises an additional selection-likelihood problem; the simple full-experiment factorization then cannot just be asserted for the selected sample alone.
- A reset to known zero is supplied physical preparation, not an inferred property. Restart sensitivities and simulated states at EACH episode boundary; do not concatenate independent resets as one continuous plant trajectory.
- Nonzero unobserved process noise makes the conditional physical state uncertain. A deterministic OE SSE is no longer this exact likelihood; filtering/latent-state integration or a justified alternative is needed.
- Valid likelihood does not imply full rank, unique physical labels, global optimizer success, finite-sample accuracy or good control. Direct trajectories may lack informative excitation. Keep those as empirical questions.

This shared-experience shadow learner is a useful fairness test: it compares direct behavior improvement with model learning using the same acquired observations/actions, rather than giving one approach a better experimental dataset. A separate model learner using budget-matched designed exploration is also legitimate, but tests acquisition plus representation together.

## 5. Source scope and minimum meaningful comparison

The pilot explicitly searches a supplied three-parameter PD/smoothing family by conventional CEM. It is a preparatory construction, not a new controller, a fair modern-RL benchmark or completed comparison. Hidden true states enter only simulator/scoring; selection reads observation-only fitness. Continuing feedback throughout all86 test steps removes the prior forced-control-cutoff task. This changes the task and intervention opportunities, so pilot outcomes must not be presented as improvements over v10/v11 stopping counts.

For an actual comparison, use a causal dynamic feedback family with enough observation and past-command memory to address actuator lag, rather than handicapping direct search with memoryless/PD-only behavior. A bounded finite-history controller or compact stable dynamic linear controller is a meaningful conventional class. Its input arithmetic, memory length, parameter bounds and initial guess remain supplied bias. Conventional antithetic random search/CEM or another established policy-search optimizer is a baseline, not novelty; retain failures and evaluate search noise honestly.

Count EVERY actual environment transition and physical reset/episode used for search, retesting, tuning and identification. Model-based virtual fitting/planning/tuning does not add physical data, but report its computation separately. Fit OE/filter/MPC to the SAME direct-search trajectories as the strongest immediate shared-data comparison; optionally add budget-matched designed exploration. Give both methods the same observation channel, bounds, task/action metadata and held-out goals. Use independent final evaluation and comparable hyperparameter-development budgets. A fixed generic feedback controller is useful as a descriptive floor, not the only competing learner. No modern-method superiority follows from beating one handpicked coarse PD or a poorly optimized local search.

Do not publish the pilot as a resolved model-based-versus-direct-learning result. The scientific uncertainty is whether different uses of the same experience produce reliably different transferable behavior, what assumptions/compute they require, and which failure is actually limiting. If established simple mechanisms suffice, accept that outcome and move toward coupled model revision or missing function vocabulary rather than adding architecture for its own sake.
