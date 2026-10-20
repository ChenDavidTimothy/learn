# Independent v12 comparison-design audit

Date: 2026-10-20 (Europe/Berlin). Read continuation_handoff.md, the unreleased direct PD pilot, the observation-loss audit and the evolving shared-experience design before new comparison outcomes. No released ledger, pilot or canonical experiment source was edited. This is a prospective adversarial design report, not a performance claim.

## 1. Selected smallest useful extension beyond PD

The selected direct controller is

    sx_t = (1-alpha)*sx_prev + alpha*o_t
    sv_t = (1-alpha)*sv_prev + alpha*(sx_t-sx_prev)
    u_t = clip(kp*(g_t-sx_t)-kd*sv_t+bv*sv_prev
               -bu1*u[t-1]-bu2*u[t-2], -1, 1).

Its six searched coordinates are log(kp), log(kd), logit(alpha), bv, bu1, bu2. Both sensor smoothing memories and command history are internal information derived from experience; no hidden true force/velocity is supplied. PD is included exactly by setting bv=bu1=bu2=0. Memory must continue through goal switches and restart only at the supplied episode reset. Save sv_prev BEFORE updating sv; use actual issued clipped command history.

This is preferable to merely enlarging the PD gain grid: the new velocity-lag/action-history terms can compensate for actuator memory. They are supplied features, not discovered concepts. Bounds and parameter signs remain substantial bias. Positive kp assumes the known positive control direction, and positive kd excludes some dynamic compensators. The chosen finite bounds may exclude useful gains. The model family likewise supplies positive beta and a correct tanh vocabulary; equal observed experience does not make these priors identical.

## 2. Exact actuator-compensation capacity check, with honest restrictions

**PROVEN UNDER ASSUMPTIONS.** For the linear lagged plant with noiseless observations and zero process noise,

    v_t=x_t-x[t-1]
    m[t-1]=v_t-rho*v[t-1]
    m_t=lambda*(v_t-rho*v[t-1])+beta*u[t-1].

Hence any clipped linear state-feedback action

    clip(Kp*(g-x)-Kv*v-Km*m,-1,1)

can be expressed by the selected finite-history controller in the alpha=1 limit via

    kp=Kp
    kd=Kv+Km*lambda
    bv=Km*lambda*rho
    bu1=Km*beta
    bu2=0,

provided the resulting kp/kd signs and all coefficient bounds are permitted. No physical parameter must be estimated by direct search to use that coefficient family. The algebra establishes useful expressive capacity; it does not claim the optimizer discovers those coefficients efficiently.

A separate independent 86-step noiseless closed-loop check used rho=.65, lambda=.7, beta=1.2, Kp=.08, Kv=.6, Km=.2 and goals(2,-2,1) switching at6/28. Mapped coefficients are(.08,.74,.091,.24,0). Independently propagated true-state feedback and observation/action-history feedback agree to8.64e-15 in command and1.87e-14 in state. Alpha=1 is OUTSIDE the actual alpha<=.98 search bound, so this is a limiting structural identity, not a claim that the executed class contains that controller exactly.

With independent zero-mean motor process noise but exact positions, the same formula reconstructs the CONDITIONAL MEAN of m_t, not its actual realization: the latest xi[t-1] has not reached position. Therefore the same limiting family contains feedback on that optimal exact-position conditional mean for the linear case. It cannot recover unknowable current motor noise.

With noisy positions, the single double-smoothed derivative is not an arbitrary Kalman filter. With c!=0, eliminating m introduces an additional tanh(v[t-1]) term absent from this linear feature family. The selected family therefore does NOT contain arbitrary observer feedback, nonlinear state-space controllers, bounded MPC or every stabilizing output-feedback law. Its positive/bounded parameters and alpha ceiling further restrict it. A performance difference must not be interpreted as a universal direct-versus-model learning distinction.

## 3. Stronger representation check if general observer containment is claimed

For a fixed-gain linear observer let its posterior state q obey

    q_t=F*q[t-1]+G*u[t-1]+K*o_t,
    a_t=k_g*g_t-L*q_t.

For a three-state observer, Cayley-Hamilton yields a finite order-three autoregression for a driven by observation/actual-action/goal history. In the UNSATURATED case a=u, an order-three input-output controller with past three commands, current/past observations and current/past goals can represent this fixed linear observer-feedback law exactly. Coefficients follow from det(I-F*z^-1) and its adjugate. Arbitrary small PD/action-memory classes need not contain it.

When clipping is active, preclip a and issued u differ; an exact elimination must retain preclip history separately or retain the observer state. Replacing all preclip lags by clipped commands is not justified. Bounded nonlinear MPC is a still different policy class. This establishes why the six-parameter family is a useful simple baseline without licensing an overstrong observer-equivalence claim.

A diagnostic projection of a frozen learned observer/controller onto the direct feature family could reveal approximation error, but it must be labeled a capacity diagnostic. If its labels/coefficients initialize or train the direct learner, that learner uses model-derived supervision and is a hybrid, not the original independent direct-search method.

## 4. Shared-data OE under feedback is statistically legitimate

The full derivation is in audit_observation_loss_v12.md. Training process noise is zero, each episode has a supplied known-zero physical reset, and every policy candidate's commands/positions is saved. Given a hypothetical physical theta and the issued command prefix, the physical position is deterministic. The actual experimenter's causal policy factor, conditional on the recorded observed history, is independent of the hypothetical physical theta. The joint/sequential likelihood is therefore the product of those policy factors and Gaussian raw-position innovation densities.

Sum the position residual SSE over ALL segments, restarting states and sensitivities at each episode's reset. Do not concatenate resets into one trajectory, rerun the policy when evaluating hypothetical theta, treat the entire future input series as exogenous, or discard nonelite episodes. Future commands encode past observations, so a likelihood conditional on the whole command sequence is a different object; it is unnecessary. The sequential/joint likelihood remains valid for adaptive policy-search data collection across episodes because each algorithmic action is fixed given the previous recorded history.

This argument does not ensure excitation, identifiability, global optimizer convergence or control success. Nonzero unobserved training process noise would require a latent-state likelihood or a separately justified estimator. A fit to all same-data segments is the strongest immediate control for differences in how experience is USED. A method choosing its own exploration under the same physical budget tests acquisition and use together; that is a distinct comparison.

## 5. Loss, horizon and information-access fairness

The lagged observation-only fitness has a policy-independent EXPECTED offset only with fresh zero-mean iid sensor errors, fixed/exogenous goals/normalization and the declared timing. It is not an exact per-episode hidden-state reward. Fresh test noise is essential; a winner's noisy training score remains optimistic, even when all candidate episodes used independent noise. Report clustered variation and retain poor candidates.

All methods should receive identical task facts, action bounds, observation channel, actual training segments and test goals/noise. Initial physical reset and unit/noise metadata are supplied capabilities. Model fitting, synthetic tuning and planning consume no new physical data but do consume computation. Report both interaction and computation costs. Model-directed virtual computation is an intended benefit of that method, not leakage.

Continuous feedback for all86 test steps differs fundamentally from v10/v11 forcedzero coasting. Do not compare its station-keeping success counts as if the previous task had been solved. Its fresh motor disturbances continue while feedback is available; stopping here means keeping a specified neighborhood under control, not removing force and coasting without disturbances.

The evolving design gives MPC the remaining86-t horizon, while the selected direct controller is time invariant and has no remaining-time feature. This is a policy-class/task-metadata-use difference worth stating. In the lagged plant the final command u85 cannot affect any scored x/v before time86, so its only finite-episode consequence is positive action effort. A common wrapper u85=0, a common finite-horizon metadata feature or an explicit limitation can resolve interpretation. No need to claim that giving both programs the same current goal makes every internal use of the task horizon equivalent. Conversely, an episode merely ending is not itself an announced physical loss of control after the end; define which finite-horizon task is actually being scored.

MPC's cost/horizon should use the same stage semantics and available commands. A nonlinear optimizer success flag is not a global certificate. A standard model/observer can fit the full correct tanh family while a direct linear-history policy cannot exactly express the resulting nonlinear policy; this is unequal inductive bias, which must remain part of the result rather than be called equal architecture capacity.

## 6. Search strength, literature and admissible conclusions

A credible comparison must separate policy class, finite search efficiency, model prior, acquisition and planner/observer quality. The six-coordinate extension and substantially increased candidate budget are meaningful improvements over the preparatory PD pilot, but a poorly optimized direct learner remains one method, not all model-free learning. Changing search bounds/initializations after outcomes requires disclosure and a fresh evaluation, not retrospective seed/plant selection. Report checkpoint learning curves from actual accumulated experience rather than compare a selected best run with a mean baseline.

**EXISTING RESEARCH:** [Mania, Guy and Recht, Simple random search of static linear policies is competitive for reinforcement learning (NeurIPS2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf) is a primary example of a serious simple policy-search comparator and emphasizes wide seed/hyperparameter variation. It does not validate this experiment or imply that CEM implements their ARS algorithm. A CEM experiment should be called CEM. The present scalar supplied-family study cannot establish superiority to modern robotics/RL methods.

The meaningful question is whether conventional direct behavior improvement and predictive learning, given identical full experience, yield different transferable capability in the declared setting, and what assumptions/compute explain it. Either outcome is acceptable. If a simple established mechanism succeeds, use it; if one fails, identify the actual finite family/search/information limitation before escalating architecture. The broader general-learning Goal remains unresolved.

## 7. Review of executed shared-data and prescribed reuse designs

Reviewed `shared_experience_v12.py` before inspecting its corrected outcomes. The selected six-coordinate policy uses old sv before updating it and actual issued command history correctly. Segmented OE restarts physical states/sensitivities per episode and fits observations/commands only, despite true physical states being retained for independent scoring. Each checkpoint fit receives exactly its accumulated prefix. Training completes independently of test evaluation, and evaluation outcomes do not enter CEM or model selection. No implementation error or hidden-state fitness leak was found in this source review; numerical reconstruction belongs to the artifact auditor.

An initial replica-seed overlap was caught before interpretation, its complete pilot artifacts retained and excluded. Corrected base ranges use plant stride1e9 and replicate stride1e6, separating candidate episode offsets100000..119101 from test offsets500000/500001 and other replicas. Preserve this correction in publication; never count the overlapped pilot as independent replication.

The prescribed `model_reuse_v12.py` diagnostic fits the SAME six-parameter direct family by the SAME8x24x2 CEM protocol on each fitted nonlinear model, with training sensorSD.2/motorSD0 and separate seed range +1e10. It adds33024 virtual steps per fit and no physical training, then uses the original held-out streams for exploratory comparison. This is a model-using hybrid, not model-free direct learning. It changes the route to parameters and adds offline computation; a stochastic restart can differ even with a perfect learned model.

The first70-step score was prescribed before corrected outcomes were read. The MPC horizon stays16 throughout this prefix, so its end-of-episode horizon shrinkage cannot causally explain prefix differences. This is a useful diagnostic, not a retrospectively substituted primary metric. Continuous feedback means this remaining clock question is distinct from the earlier physical command-cutoff error.

After the corrected outcomes became available, I independently read the saved summaries. Final direct / nonlinear MPC mean costs by plant are respectively:

    .314099 / .267673
    .331481 / .274189
    .401574 / .266358
    .357066 / .274551.

The corresponding mean direct-minus-nonlinear paired differences are .046426, .057292, .135216 and .082515, with clustered MC SEs .008424, .011525, .009540 and .014324. These support better mean task performance for this supplied predictive pipeline than this specific finite CEM/direct family on the declared four cells. Both methods use the identical acquired experience. This does not establish equal priors/capacity, optimum policy search or broad method superiority.

The same-family virtual-CEM mean costs are .321943, .311855, .393994 and .406712. Original-direct minus reused-direct differences are -.007844+/-.014065, +.019626+/-.012586, +.007579+/-.017026 and -.049646+/-.027202 (clustered MC SE). There is no consistent benefit in these finite runs. This does not prove simulated experience is useless: family, CEM search, random restart, finite fitness noise and learned-model bias remain. The MPC advantage also appears on the prescribed first70-step metric, so end-clock shrinkage is not its sole explanation.

## 8. One bounded optimizer diagnostic, then leave this scalar comparison

A single exact-pathwise-gradient diagnostic within the SAME six-coordinate family is scientifically defensible if it is prescribed and bounded. It can test whether finite CEM search rather than family capacity is preventing useful behavior. Optimize on the learned model, verify derivatives independently, preserve actual clipping derivatives almost everywhere (no straight-through replacement), maintain the same parameter bounds and training task/noise initially, use fixed multiple starts and separate virtual validation, and disclose all added compute.

A good fresh physical rollout from such a policy would constructively falsify the claim that the family cannot achieve that behavior; it would support a limitation of the earlier search/experience use. Failure of another local optimizer does NOT establish representational incapacity. Neither a small gradient nor convergence certifies a global policy optimum. If test outcomes already inspected motivate this repair, call it exploratory and use fresh physical evaluation streams for any stronger confirmatory claim.

Do not escalate through an indefinite list of optimizers or force a desired winner. Set a one-diagnostic stopping rule, retain an OPEN capacity-versus-search conclusion if unresolved, and move to a coupled/missing-vocabulary or context-learning experiment with a consequential scientific discriminator. The point is to understand and construct learning mechanisms, not to optimize this scalar benchmark indefinitely.
