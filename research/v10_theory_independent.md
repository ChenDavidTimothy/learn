# Independent pre-experiment theory audit: v10 construct-and-control cycle

Date: 2026-10-20 (Europe/Berlin). This is a prospective critique, not a claim that v10 has run. No canonical source or ledger was edited. The proposed experiment compares fitted linear/nonlinear output-error models, observation-only filters and predictive control against model-tuned/fixed PD; privileged true-model/state variants diagnose limitations.

## Information and claim boundary

All fair fitted controllers receive the same 512 command/noisy-position rows. The exact tanh family, state dimension, initial rest, lagged timing, sensor noise law and test motor-noise scale are supplied assumptions. Model-tuned PD uses simulations from its fitted model; this consumes no additional physical observations and is a valid common-data comparator, but is not model-free. True-model variants must remain explicitly privileged diagnostics. Test target sequences/noise cannot tune controllers; future goal switches should be revealed only when prescribed by the common goal interface.

Success can demonstrate a complete, conventional experience-to-control mechanism under supplied structural assumptions. It does not establish discovery of that structure, superiority over modern nonlinear control, necessity of the nonlinear family or necessity of learning. A fixed nominal-model version of the same filter/planner is the clean optional learning-benefit ablation; generic PD alone changes both knowledge and controller.

## Lagged timing and action sensitivities

For state s=[x,v,m], f(v)=rho*v-c*tanh(v), a=f'(v),

    s_next = [x+f(v)+m, f(v)+m, lambda*m+beta*u]
    A = [[1,a,1], [0,a,1], [0,0,lambda]]
    B = [0,0,beta]^T.

With additive motor innovation, Q=diag(0,0,sigma_xi^2). Current u has zero effect on next x and v; its first positional effect occurs at t+2. Motor-first sensitivities would be incorrect here. Independently finite-difference predicted action-cost gradients, including fixed-zero tail propagation. Record local-optimizer failures/iterations/gradient diagnostics so numerical planning failures are not attributed to missing learned physics.

## Exact linear realization-invariance check

At c=0 set d=rho-lambda and q=m+d*v. Swapping rho/lambda preserves x,v. With

    T=[[1,0,0], [0,1,0], [0,d,1]],

A_swapped=T*A*T^-1, B_swapped=T*B=B, C_swapped=C*T^-1=C and Q_swapped=T*Q*T^T=Q. A Kalman filter initialized at transformed mean/covariance therefore yields identical observable predictions. Exact known-zero initial state with covariance zero already satisfies this. Reusing an arbitrary diagonal uncertain covariance in both coordinates generally does not.

Any MPC cost on x,v,u and a zero-command tail is invariant under this transform. Its predicted objective, action gradients and optimized commands should match to numerical tolerance with the same solver initialization. Do not penalize hidden m directly: that would introduce a coordinate-dependent objective rather than a physical stop requirement. Nonlinear EKF approximations need not be invariant to nonlinear state transforms; their differences do not resolve physical component ambiguity.

## A disturbance confound in forced-coast success

**PROVEN UNDER ASSUMPTIONS / NUMERICALLY CHECKED:** Even an ideal controller starting at exact rest cannot maintain position under indefinitely forced zero command if actuator-process noise continues. For the linear c=0 cases, propagate P_next=A*P*A^T+Q from P=0 with rho=.65 and sigma_xi=.03. After 16 uncontrollable coast steps:

| lambda | position SD | velocity SD |
| --- | ---: | ---: |
| .2 | .36674516 | .04591858 |
| .7 | .85677724 | .09029993 |

The zero-mean Gaussian final position alone exceeds a .3 tolerance frequently; requiring every one of the 16 positions within tolerance is stricter. This is unavoidable disturbance after control is prohibited, not a learning failure. The numbers were independently obtained using the displayed 3x3 covariance recursion with local NumPy. They do not depend on observation error because the state begins known and no feedback is allowed.

Recommended separation: noisy active control measures disturbance rejection; then a common process-noise-free zero-command coast measures residual stored velocity/actuator force. This shutdown is an explicit diagnostic evaluation condition, not a realistic universal stopping guarantee. Alternatively retain noisy coast as open-loop survival and interpret it against the privileged oracle rate, without calling it reliable stopping. Sensor noise can remain; true x/v are evaluator-only scores.

A three-step coast is also insufficient to expose slow stored force: lambda=.7 retains .343 of it after three steps. A 12-step predicted tail and 16-step evaluation coast are much better minimal choices. Observed consequences over that tail test settling without defining a physical force coordinate.

## Minimal reporting and falsification

Report tracking cost, terminal position/velocity error, success, effort and prospective prediction errors separately. Primary normalization and stopping tolerances must be fixed before outcomes. Preserve all seeds/failures. Pair test exogenous streams across controllers for variance reduction; controllers still generate their own observation histories and only the simulator sees the shared innovations. Model-tuned PD versus learned MPC isolates planning differences conditional on the learned model; true-model/EKF versus true-model/true-state separates filtering effects. If linear fitted control or PD already performs adequately, that is evidence to defer active uncertainty machinery rather than manufacture a setting that requires it.

## Final source review and resolved conditions (before outcomes)

Read `research/goal_control_v10.py` after the full run started, before inspecting outcomes. The source correctly implements the three-state transition/Jacobian above, an observation-only Joseph-form EKF correction, lagged motor-noise prediction, current-goal-only planning, bounded commands, known-rest initialization and no goal-switch resets. True-state assignment occurs only in the labeled `oracle_state_MPC` branch; true plant parameters are otherwise used by simulator/scoring and labeled oracle branches. PD tuning runs only against the learned nonlinear model, using supplied noise scales; it has no extra physical interactions. All fair controllers within a plant/fitted-model seed share the identical 512 physical training rows. Different plant cells use different seed streams, so cross-plant comparisons are not matched-data pairs.

The coast-noise concern was resolved BEFORE execution: motor innovations are zero during the 16 evaluation-coast transitions, while sensor errors remain. MPC optimizes 8 controls followed by 12 fixed-zero transitions and penalizes only x-goal, v and u. A fixed nominal-parameter MPC was added as a same-planner learning ablation. Optimizer records retain success/status, iterations, objective improvement/fallback and projected-gradient diagnostic. No remaining actual implementation error was identified in this independent source review. Independent numerical reproduction/action-gradient checks belong to the separate artifact auditor.

Two limitations are deliberately retained with explicit scope, rather than changing the frozen source after launch. PD tuning minimizes precisely the primary active normalized tracking cost; MPC uses its planning surrogate including a zero-command continuation. Consequently stopping-success comparisons do not isolate planning or learning, since PD was not tuned for stopping. The settling diagnostic is a zero-based qualifying action-index delay: its 0 corresponds to the first post-switch scored state, one physical transition after goal revelation. It should not be called elapsed settling transitions. Neither issue changes the correctly computed primary active tracking metric.

One unavoidable disturbance remains at the active/coast boundary. The last active command is chosen before its motor innovation xi[69] is realized; removing subsequent innovations does not remove this stored final shock. For c=0, propagate d0=[0,0,1] through 16 copies of A: the coefficient times .03 gives final-coast position SD .10701715 for lambda=.2 and .28219306 for lambda=.7, due to this final innovation alone. In the slow linear plant, even perfect conditional centering therefore gives at most erf(.3/(sqrt(2)*.28219306))=.71226429 probability of the final position lying within .3. The full all-times/velocity success event is no more likely. This is a limit of the declared uncontrollable stopping evaluation, not an error or a request to modify the frozen experiment. Interpret stop rates relative to the privileged diagnostics; do not impose near-perfect success as a learning requirement.

The resulting scientific question remains consequential but narrow: does conventional experience-fitted predictive control acquire sufficient observable consequences for this bounded, supplied-family new-goal task, and how much do filtering, representation restriction and control choice matter? The design now constructs and tests an actual learner/controller. It does not justify adding active uncertainty machinery merely because that architecture was originally contemplated; the observed failures must first distinguish an information problem from model, estimator, optimizer or stop-evaluation limits.

## Explicit final-innovation Gaussian bound

**PROVEN UNDER ASSUMPTIONS.** Assume c=0, rho=.65, additive independent xi_t~N(0,.03^2), all controls past-adapted, and the declared last active action u69 followed by 16 exactly zero-command/no-process-noise transitions. Let G contain the entire true history, policy random coins and u69, immediately before xi69 is drawn. Including hidden history only strengthens the controller for this bound, so it includes the privileged true-state controller. The last transition has

    s70 = s70_bar(G) + e_m*xi69,   e_m=[0,0,1]^T.

Since the following transitions are linear, autonomous and disturbance free,

    x86 - g = mu(G) + h16*xi69,
    h16 = (A^16)[0,2].

Sensor errors in the coast cannot affect this true-state trajectory because future commands are fixed zero. For rho!=lambda,

    hL = sum(k=1..L) (rho^k-lambda^k)/(rho-lambda)
       = [rho*(1-rho^L)/(1-rho)
          -lambda*(1-lambda^L)/(1-lambda)]/(rho-lambda).

For any conditional mean mu, a fixed-width symmetric interval has greatest Gaussian probability when mu=0. Therefore

    P(|x86-g|<=a | G)
       <= erf(a/(sqrt(2)*.03*abs(h16))).

The same bound holds after averaging G. The full stopping event is a subset of this final-position event. At a=.3:

| lambda | h16 | final-shock position SD | upper bound on stop probability |
| --- | ---: | ---: | ---: |
| .2 | 3.5672382580412054 | .10701714774123616 | .9949415690842329 |
| .7 | 9.406435335003787 | .2821930600501136 | .7122642919368134 |

These are population probability bounds, not hard limits on a finite empirical success fraction. A finite sample can exceed its corresponding bound by chance. The bound does not apply unchanged to c!=0, anticipatory information about xi69, process-noise cancellation at the final active step, or a policy allowed to act during the coast.

## F44 audit: accurate consequences imply an accurate local decision

**PROVEN UNDER ASSUMPTIONS; classical approximate-optimization fact.** Fix a current history h, the same nonempty candidate future-action set A and the same expected trajectory cost semantics in the true and learned evaluations. Let J(a) be true conditional expected cost and Jhat(a) its approximation. Suppose

    sup(a in A) |J(a)-Jhat(a)| <= epsilon

and the selected action sequence ahat is globally delta-optimal for the learned cost:

    Jhat(ahat) <= inf(a in A) Jhat(a) + delta.

Then

    J(ahat) <= Jhat(ahat)+epsilon
            <= inf_A Jhat + delta + epsilon
            <= inf_A J + delta + 2*epsilon.

Thus true suboptimality is at most 2*epsilon+delta, whether or not either infimum is attained. This theorem concerns decision consequences, not physical labels. Prediction-equivalent physical explanations have identical J for the same permitted ordinary actions, so distinguishing them is unnecessary for this local decision.

Important limits: marginal one-step RMSE, average rollout RMSE, parameter error, a fitted physical label or training SSE does not establish the required uniform expected-cost bound over candidate actions and the current history. Error at a different training-history distribution is insufficient. Optimizer success or a small projected gradient is not a global delta certificate for a nonconvex learned cost. Receding-horizon replanning does not turn the statement into a closed-loop guarantee without bounds at repeated encountered histories and an argument relating the local surrogate to the actual long-horizon objective. In v10, deterministic nominal rollouts and a finite fixed-goal coast are a planning surrogate; they are not a generally exact noisy expected-cost model or an oracle for future unannounced goal switches. V10 does not supply epsilon or a global delta. F44 therefore explains what useful learned consequences would need to control; it does not prove the experiment's success in advance or constitute a new learning theorem.
