# Independent theory/source audit of the bounded v13 diagnostic

Date: 2026-10-20 (Europe/Berlin). Read gradient_reuse_v13.py/design before optimization outcomes, derived the sensitivities independently, and constructed a separate PyTorch forward-autodiff implementation. Canonical code is used only as a final numerical comparison target; no canonical sensitivity formula enters the independent autodiff. No canonical source, executed design or prior ledger was edited.

## 1. Exact fixed-noise pathwise derivative

Let searched coordinates be p=(log(kp),log(kd),logit(alpha),bv,bu1,bu2). The learned physical parameters rho/lambda/c/beta are fixed during this diagnostic. For any coordinate perturbation,

    dkp=kp*dp0, dkd=kd*dp1, dalpha=alpha*(1-alpha)*dp2,
    dbv=dp3, dbu1=dp4, dbu2=dp5.

At a step let old controller memories be S,V,U1,U2 and physical state x,v,m. Fixed virtual sensor error is e; current goal g is exogenous. Define

    Snew=(1-alpha)*S+alpha*(x+e)
    Vnew=(1-alpha)*V+alpha*(Snew-S).

Then

    dSnew=(1-alpha)*dS+alpha*dx+(x+e-S)*dalpha
    dVnew=(1-alpha)*dV+alpha*(dSnew-dS)
           +(Snew-S-V)*dalpha.

For raw command

    r=kp*(g-Snew)-kd*Vnew+bv*V-bu1*U1-bu2*U2,

its differential is

    dr=(g-Snew)*dkp-kp*dSnew-Vnew*dkd-kd*dVnew
       +V*dbv+bv*dV-U1*dbu1-bu1*dU1-U2*dbu2-bu2*dU2.

For a=clip(r,-1,1), da=dr when |r|<1 and da=0 when |r|>1. There is no ordinary derivative at exactly +/-1; assigning0 there is a convention, not a claim of differentiability. The source correctly uses actual issued clipped actions and shifts their derivatives into command memory. A straight-through derivative would be a different optimizer and is not used.

Lagged physical timing gives

    vnew=rho*v-c*tanh(v)+m
    mnew=lambda*m+beta*a
    xnew=x+vnew,

hence

    dvnew=(rho-c*sech(v)^2)*dv+dm
    dmnew=lambda*dm+beta*da
    dxnew=dx+dvnew.

The action does not affect vnew/xnew at the same step. Process noise is zero for these virtual training rollouts; fixed additive noise would have zero pathwise derivative too, but would change the training distribution. Reset ALL state/controller values and derivatives at each episode. The source implements these chain rules correctly.

For batch size B, T=86 and goal normalization D_b=1+mean_t(g_bt^2), residuals divide x-error, sqrt(.3)*v and sqrt(.03)*a by sqrt(B*T*D_b). Their squared sum is exactly the mean normalized model stage cost across episodes. The residual/Jacobian flattening orders agree. Goal normalization is parameter independent; no missing derivative term arises. The model-state cost uses consequences of an already learned model, not privileged real-plant state reward.

This is the exact derivative of a fixed finite-noise-bank objective almost everywhere. It is not automatically the gradient of the true expected plant objective, a global optimization guarantee, or a derivative through the earlier model-fitting procedure. No derivative through the frozen physical model parameters is required here.

## 2. Independent autodiff and clipping checks

`audit_gradient_pathwise_v13.py` implements the physical/controller recursion separately in existing local PyTorch, float64, and obtains the entire residual Jacobian by forward autodiff. It tests generic, command-memory and strongly saturated coordinates across all four declared plants, plus the exact16-episode canonical check setup. Results and every comparison error are retained in `audit_gradient_pathwise_v13_results.json`. No dependency was installed.

Regular generic/memory tests agree at approximately1e-14 componentwise scaled error. Moderately saturated cases remain close; an extreme long-horizon saturated stress case exposes numerical conditioning: independent algebra/elementary-function ordering yields state differences up to1.24e-7 and command differences1.16e-7, while the maximum absolute autodiff Jacobian entry is2.95e7. Its maximum absolute Jacobian discrepancy is .1798, but componentwise scaled discrepancy is1.99e-8. The initial audit's strict1e-8 scaled assertion rejected that stress case; the correction and strict-agreement=false flag are preserved rather than claiming machine-precision agreement. All errors are reported, with a broader1e-6 stress tolerance. This is an explicit finite-precision conditioning limit, not evidence of a missing chain-rule term. It reinforces that large policy sensitivity can make finite differences unreliable even without changing the clip active set.

For the exact canonical check setup (theta=.65/.7/.2/1.2, seed75071013,16 episodes), independent autodiff agrees with the manual Jacobian within1.11e-13 absolute and7.98e-15 scaled error, with state discrepancy4.45e-14. This directly verifies the manually derived derivative despite the failed coarse finite difference.

The source's first epsilon1e-6 test changed one clipped/unsaturated decision for the bv coordinate; its .331 error is not a derivative approximation within one smooth region. Shrinking the fixed step to1e-7 and1e-8 preserves active sets and produces errors1.09e-8 and1.66e-7 respectively; the smaller step's larger error is consistent with subtraction roundoff. Preserve the failed coarse check as done in gradient_reuse_v13_check_correction.md. A corrected numerical check must independently establish the active-set explanation, not merely loosen tolerance. The separate autodiff/raw-active-set test provides that independent check.

The three-step finite-difference check evaluates one baseline plus36 perturbed rollouts, so derivative-verification accounting is37*16*86 virtual transitions, not the original13*16*86. The source metadata was corrected to37 before publication. These are virtual computations, not physical experience.

## 3. Source, selection and resource audit

The three starts are fixed physical-CEM winner, virtual-CEM winner and generic coordinates. Each of32 learned models uses its own fixed16-episode fit bank; all starts share it. The source retains all96 final fit attempts, including one nonconverged attempt, and chooses among their outputs and the two original winners using a separate16-episode virtual validation bank. Including unoptimized originals prevents a forced replacement on that validationbank, but does NOT guarantee improved expected model or physical cost after selection. Five-way validation selection remains optimistic.

Fit/validation/test namespaces at physicalseed+20e9/+30e9/+40e9 are disjoint from each other and earlier v12 roles given the declared plant/replicate strides. Test sensor/motor streams are fresh, paired across all four controllers, and never read by fitting/validation. Initial physical/controller memories remain zero and no target-switch reset occurs. Goal magnitudes repeat the previously specified evaluation family, but noises are new; this is exploratory method development following v12, not a new blind broad task benchmark.

Fixed-bank parameter optimization can overfit virtual noise. Independent validation reduces this problem; fresh physical testing is the actual transfer check. The observation-loss constant-offset identity is not needed to pretend optimized fit-bank scores are unbiased: this optimizer uses predicted model states, and selected parameters depend on that bank. Model bias and selection variance remain.

No new actual-plant training is acquired. Virtual objective calls, validation calls, retained-trajectory replays and derivative checks are separately accounted. The cache counter records actual rollout evaluations/cache misses; it need not equal the number of globally distinct parameter vectors. Added physical test interactions are evaluation, not fitting. The method uses a learned simulator, extra computation and a supplied correct function family; it is a model-using hybrid, not an improved model-free policy search given free true dynamics. TRF convergence or a small projected gradient is not global policy optimality in this clipped nonlinear closed loop.

## 4. Interpretation after the bounded experiment

Independently read the saved summary after execution. Mean gradient-selected costs across the four cells are .265958/.275643/.306701/.289054, versus physical-CEM .313822/.330627/.402863/.352047 on the SAME fresh streams. The corresponding paired improvements are .047864+/-.007912, .054984+/-.010174, .096162+/-.008644 and .062994+/-.013799 (clustered MC SE, eight model/fit-seed clusters). All32 selected outputs come from optimized candidates. This constructively demonstrates better attainable primary cost within the same six-coordinate family in these runs; prior finite CEM search had left useful behavior unfound. It does not prove an optimal stationary controller or globally identify the sole earlier bottleneck.

Nonlinear MPC costs are .266410/.271958/.266985/.263184. The fast-linear cost difference is small relative to its paired variation; the slow cases retain substantial differences. Model/observer/planner policy capacity, parameter bounds, filtering and local optimization remain relevant. No equivalence to MPC or universal model necessity follows.

Crucially the strict last10-step station-keeping event succeeds only1/64 times for gradient-selected policies, versus14/64 physical-CEM,16/64 virtual-CEM and25/64 MPC. Lower average quadratic episode cost therefore is NOT uniformly improved task capability. It can emphasize transients/effort while sacrificing late precision, and training motor noise remains0 while testing motor noise is .03. Objective mismatch, finite-bank overfit, distribution shift and controller properties are competing explanations; no additional optimizer run is warranted merely to recover a desired success story. The observed-cost ranking theorem also never guaranteed a particular precision-event probability.

The diagnostic has served its bounded purpose: it falsifies the inference that the earlier direct family's entire primary-cost deficit was representational incapacity, while exposing a different capability loss. Failure to reach a controller/precision objective still cannot prove family incapacity. Preserve this mixed result and leave scalar optimizer tuning as promised. The next coupled or missing-function-vocabulary experiment should test a meaningful revision of acquired consequences, with context/task facts explicitly distinguished from learned physical regularities.
