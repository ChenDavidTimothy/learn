# Independent v14 mechanics, comparison-design and drift audit

Date: 2026-10-20 (Europe/Berlin). Read coupled_context_v14_question.md and independently derived the horizontal two-link mechanics. This is a design/physics audit, not an executed learning comparison. No canonical experiment or prior ledger was edited. Independent mechanics code/results are audit_coupled_shortpulse_v14.py and audit_coupled_shortpulse_v14_results.json.

## 1. What question is consequential?

A defensible question is whether experience can reveal a missing dependency in what a command does, and whether retaining that dependency improves NEW goal-directed choices. A two-link body provides genuine coupled physics beyond the supplied scalar family. But merely calibrating a smooth known one-dimensional context map at rest is standard local system calibration, not a deep new learning mechanism. Comparing a context learner only to independent joints would predetermine the answer.

Time-box the rest-pulse instrument: verify mechanics, train useful context-aware alternatives on the same observations, test actual requested displacements, then advance to continuous coupled motion or a genuine model-revision question. Do not publish a sequence of increasingly specialized inertia-identification formulas. The masses/cosine terms belong to simulator derivation and privileged diagnostics; recovering their named values is not the destination.

A stronger next capability, after this calibration gate, is learning when an existing experience summary is insufficient: e.g. the same pose/action under different incoming motion requires history, and the learner must revise its consequence predictor rather than receive a newly hand-selected physical feature. This is still established system-learning territory, but it tests construction and revision rather than another static counterexample. Do not assume a world-model architecture is mandatory; a direct goal-conditioned controller or episodic rule can also retain the needed distinction.

## 2. Lagrangian derivation and continuous-time response

Use massless links of lengths l1=l2=1, endpoint masses m1=m2=1, absolute first angle q1 and RELATIVE elbow angle q2. Horizontal motion removes gravity; there are no contacts, joint limits, friction or actuator lag. Let

    r1=[cos(q1),sin(q1)]
    r2=r1+[cos(q1+q2),sin(q1+q2)].

Expanding kinetic energy gives

    T=.5*2*v1^2+.5*(v1+v2)^2+cos(q2)*v1*(v1+v2)
     =.5*v^T*M(q2)*v,

    M=[[3+2*cos(q2),1+cos(q2)], [1+cos(q2),1]].

Lagrange's equations d/dt(dT/dv)-dT/dq=u yield

    M*qdd + h = u,
    h=sin(q2)*[-(2*v1*v2+v2^2), v1^2].

M is symmetric positive definite: M22=1 and detM=2-cos(q2)^2>=1. At q2=0 its inverse is [[1,-2],[-2,5]], while at q2=pi it is identity. Those are INITIAL acceleration maps, not complete finite-pulse maps.

**EXISTING RESEARCH:** [Tedrake's multibody notes](https://underactuated.mit.edu/multibody.html) provide the general point-mass two-link Lagrangian and manipulator equations; the horizontal unit-parameter equations above were independently expanded. The result is classical mechanics, not novelty.

For a constant torque pulse from exact rest,

    a0=M(q0)^-1*u, q'''(0)=0
    q(T)-q0=.5*T^2*a0 + T^4*q''''(0)/24 + O(T^6).

The odd terms vanish by local time-reversal symmetry of this smooth constant-force, no-damping system. A damping term generally creates a nonzero cubic term, and actuator lag changes the leading command response; neither can be silently introduced later. A semi-implicit Euler step instead gives T^2*a0, a factor-of-two discrepancy from continuous displacement, so its discrete convention cannot be called the continuous short-pulse law.

Writing M2=dM/dq2 and H(a)=sin(q2)*[-(2*a1*a2+a2^2),a1^2],

    q''''(0)=-M^-1*(a2*M2*a+2*H(a)).

At q2=pi/2 and u=(0,1), a0=(-.5,1.5) and q''''(0)=(1.75,-3), so

    deltaq1=-.25*T^2+(7/96)*T^4+O(T^6)
    deltaq2=.75*T^2-(1/8)*T^4+O(T^6).

At q2=0/pi, sin(q2)=0 and the fourth-order term vanishes; leading approximation error can start at T^6. Using only those special poses would conceal the generic convergence order. In particular at q2=pi, joint1 initial acceleration under joint2-only torque is zero, but finite-time joint1 motion is generally NONZERO after q2 changes. Do not turn an initial zero coefficient into an exact finite-time decoupling claim.

Independent DOP853 integration (rtol2e-13, atol1e-15) confirms halving ratios approaching16 for the quadratic approximation's absolute error at generic poses and64 at q2=0/pi. The code also checks the two independent conservation identities

    kinetic_energy(T)=u dot (q(T)-q0)
    M11(q2)*v1+M12(q2)*v2=0 when u1=0 and initial rest.

Maximum energy-work discrepancy is2.26e-14 and cyclic-momentum discrepancy5.51e-15 across retained cases. At q2=pi, T=.2, u=(0,1), actual deltaq1=-1.333e-6 rather than zero. These are mechanics/numerical checks, not learning evidence. Solver-tolerance refinement should accompany the canonical pilot, particularly before shrinking T enough that displacement subtraction reaches roundoff.

## 3. What context is actually needed?

At rest and leading order, absolute q1 is a global-rotation nuisance, and response depends on the scalar cos(q2). Thus q2 and -q2 have the same leading response. Finite-pulse fourth-order terms generally involve sin(q2) too, so cos(q2) alone need not specify exact finite-duration outcomes. Velocity/rest preparation and actuator condition are also context; eliminating them experimentally does not prove they are generally unnecessary.

If the investigator passes only q2 to a learner, relevance has been supplied. Giving both joint encoders and varying q1 independently lets a generic learner confront nuisance versus consequential context. A q2-only or exact-M predictor can be a clearly privileged diagnostic. Do not give masses, inertia formulas or hand-selected cosine features to the ordinary learner and then claim it discovered them. Circular angle topology is a separate supplied bias; either state it explicitly or first use a declared contiguous pose interval without pretending to have learned global wraparound.

A position preparation setpoint is not automatically an exact measured context. Either label exact known preparation pose as supplied, or provide only noisy encoder context and model its errors. Known physical zero velocity is also an extra reset capability. Count preparation/reset and sensor-read resources. Direct generalized torque in this instrument is an ideal calibrated actuator, different from the earlier hidden-motor command interface.

## 4. Two noise traps capable of deciding the comparison artificially

If context is qhat0=q0+e0 and training response is qhatT-qhat0, then the shared -e0 term is correlated with the context regressor. Even at zero torque a flexible predictor can learn regression-to-the-mean effects unrelated to physical displacement. Use an independent initial encoder read for the displacement baseline (the extra read is supplied to all methods and counted), or a correct joint observation model. Independent read noise does not eliminate context errors-in-variables, but it removes this artificial direct noise correlation. True q is scorer-only unless exact preparation is explicitly supplied.

Naive direct inverse regression is not automatically as statistically well specified as forward response regression. In one dimension, y=a*u+epsilon with exact random u and noisy response gives inverse OLS gain

    Cov(u,y)/Var(y)=a*Var(u)/(a^2*Var(u)+Var(epsilon)),

rather than1/a. Acting on a clean requested target using this shrunk inverse underachieves even with unlimited samples. A forward learner has response noise in its output; the inverse has it in its input. Beating a naive inverse therefore cannot establish a necessity of predictive models.

A fair strong inverse alternative can use calibrated response-noise covariance/replicate measurements and errors-in-variables correction. For a locally constant invertible2x2 map y=B*u and zero means, local moments

    G=E[u*y^T]*(E[y*y^T]-Sigma_noise)^-1

recover B^-1 under independent output noise and sufficient input rank. Finite estimates can be indefinite/weak; regularization/failures must be reported, not silently clipped into success. Forward and inverse formulations can become equivalent in this limit, which is a legitimate simplifying result. Generic retrieval without a physical predictor is also possible, but its noise/sampling limitations must not be hidden.

## 5. Smallest useful same-experience task

A minimal comparison needs a strong FULL coupled/context-aware baseline, not only independent joints:

1. A constant2x2 coupled response model, as a diagnostic of omitted context.
2. A generic context-conditioned local/RBF response learner, choosing bounded torque to match a requested small joint displacement.
3. A direct context/goal inverse or episodic retrieval learner, with its response-noise issue handled or clearly delimited.

The response learner may use a local torque-linear map justified by the short-pulse approximation, but that linearity is an explicit supplied prior; it is not unconstrained function discovery. A generic joint kernel over context and torque is a useful stronger alternative if finite-pulse nonlinearity matters. Both ordinary formulations receive exactly the same context/issued torques/encoder responses. Hyperparameters must be chosen using only declared training/validation data, with any extra physical goal trials counted; fitting compute need not be equal, but it must be reported. Supplied exact mechanics is a privileged feasibility/planning diagnostic, not the ordinary comparison baseline.

Use matched broad-pose versus narrow-pose acquisition at the same pulse/reset/measurement budget. Include independent joint and mixed torque probes with action rank, and vary q1 independently rather than correlating it with q2. Hold out BOTH requested displacement directions/magnitudes and actual pose regions. Same-context interpolation, new-q1 invariance transfer and new-q2 transfer are different tests and should be reported separately. Do not call a narrow-region fit's failure far outside support surprising, or claim arbitrary extrapolation without an assumption enabling it.

Use the SAME desired displacement at multiple poses and preferably a conservative common feasible goal set. If goals are generated by hidden random torque at each pose, they can themselves encode pose/context and change the inverse-regression target distribution. Such goal generation is valid only if declared, and is weaker evidence that the explicit context channel matters.

Leading-order feasibility is u=(2/T^2)*M(q)*delta. A simple conservative common box over all q2 has ||delta||_infinity <= T^2*Umax/14, because the maximum infinity row norm of M is7. Add a finite-pulse margin and verify feasibility with the privileged continuous simulator; do not classify unreachable targets as learning failure or silently remove only inconvenient method cases. An endpoint oracle also separates poor local approximation from actual command constraints.

The primary result is actual achieved displacement error and torque effort under unseen targets/poses, with all failures retained. A short pulse ends with nonzero velocity: it is NOT stable reaching, stopping or continuing articulated-body control. Do not inflate this capability claim.

## 6. Drift audit and recommendation

This route is defensible as the first coupled physical learning instrument, but static context calibration is close to trivial under supplied rest, encoder coordinates, torque calibration, smoothness and a one-dimensional relevant context. Its scientific value depends on what changes the construction of a learner: which missing dependency is acquired and reused for new actions, with what assumptions and failure modes? A successful kernel fit alone is not a deep principle.

Prefer one compact complete same-experience capability test over many mass-identification or kernel-tuning releases. If ordinary local response/inverse methods succeed, accept their sufficiency and move promptly to continuous coupled motion or learning when to revise an inadequate summary. If they fail, first distinguish noisy context, inverse-regression bias, unsupported extrapolation and infeasible goals from a failure of general learning. Do not force a new architecture merely to improve this calibration toy. The ultimate research objective remains much broader and unresolved.

## 7. Review of prescribed compact comparison before execution

Read `coupled_learning_v14_design.md` before learning outcomes. Exact known rest preparation q0 is now explicitly supplied to every learner, including both joint encoders. The noisy response uses endpoint-minus-initial readings but the initial reading is NOT used as context. This resolves the shared-context-noise issue by a declared extra preparation capability, not by learning an observation model. No noisy-state-inference claim is justified or intended.

For T=.1, encoderSD=1e-4, normalized response d=(oT-o0)/(.5*T^2) has covariance R=.0008*I under independent encoder noise. With phi(q) exact and inverse design features phi tensor d, the Gram bias is exactly sum(phi*phi^T) tensor R. Subtracting it gives an unbiased unregularized moment estimate under the declared zero-mean independent response errors. The inverse validation statistic

    ||K(q)*dobs-u||^2 - trace(K(q)*R*K(q)^T)

is unbiased for fixed-candidate inverse squared error on the true finite-pulse response. The subtraction identity itself does not require that physical response be exactly torque linear; finite-pulse nonlinearity instead creates model-family approximation error. This is not a universal likelihood or finite-sample optimality result. Corrected Gram indefiniteness, eigenfloor and ridge effects must be retained as prescribed.

The constant FULL coupled model and both context-aware forward/inverse families are appropriate conventional competitors. Periodic RBF features over BOTH encoders avoid researcher-supplied relevance selection; periodicity, basis size, smoothness/torque affinity and exact preparation remain supplied priors. Width/ridge selection uses independent validation, but equal grids do not imply equal effective regularization across differently scaled forward/inverse regressors. Corrected inverse validation can be negative and noisy for high-norm K; selecting its minimum is optimistic. Save all scores/eigenvalues and rely on separate actual goal tasks rather than interpret a negative score as negative physical risk.

Two caveats were sent before execution:

1. The proposed q2 grid includes both -pi and+pi, identical physical contexts, and omits0. Replacing one endpoint duplicate with0 preserves160 tasks while adding narrow-region interpolation and avoiding implicit double weight on the folded pose. If retained, explicitly disclose repeated physical context; do not call the160 tasks unique independent contexts.
2. Forward bounded least squares and clipping the inverse command are different constrained decision rules. If K=B^-1 and K*z lies outside the box, clipping minimizes command-space distance while forward optimization minimizes displacement error with metric B^T*B. Common true goals have substantial feasibility margin, so this primarily affects poorly estimated/extrapolated models. Report saturation as planned and avoid attributing all differences to the representation of knowledge. Inverting a learned inverse to use a common constrained response optimizer is a legitimate optional diagnostic, but unnecessary for the limited primary four-pipeline comparison if this caveat is explicit.

The strongest permitted interpretation is that equally acquired pulse experience can or cannot support these specified generic context-conditioned goal decisions, and how coverage/noise/decision rules affect them. It does not discover a physical function vocabulary, prove forward models necessary, or establish a general learner. The present design is a defensible compact gate to continuous coupled learning; no further calibration/optimizer ladder should follow merely to force one competitor to win.

## 8. Accepted pre-execution corrections and remaining statistical limits

Before any learning outcomes were inspected, the investigator replaced the +pi duplicate test context with0, retaining eight unique periodic q2 poses and160 context-goal tasks. All learned relations now use the SAME bounded displacement-error decision solver: forward B directly, inverse via B=pinv(K). This removes coordinatewise-clipping versus goal-metric differences from the prescribed comparison. The result compares learned relation parameterizations/statistical treatment, not a pure model-free-versus-model-based dichotomy.

Retain the singular values and pseudoinverse threshold for K: near-singular inverse relations can amplify estimation noise or discard directions. Equal numbers of RBF coefficients do not make the two function classes equivalent, because the inverse of an RBF matrix function need not lie in the same RBF span. Shared feature count/grid and common decision metric improve fairness but do not establish equal expressivity or regularization.

For fixed fitted relations and independent validation measurement error epsilon with E[epsilon]=0, Cov(epsilon)=R:

    E[||B(q)*u-(dtrue+epsilon)||^2 | q,u]
        = ||B(q)*u-dtrue||^2 + trace(R)

    E[||K(q)*(dtrue+epsilon)-u||^2 - trace(K(q)*R*K(q)^T) | q,u]
        = ||K(q)*dtrue-u||^2.

The inverse Gram subtraction yields unbiased Gram/cross-MOMENTS for the noise-free regression problem, not unbiased finite coefficient estimates after matrix inversion, eigenfloor, ridge or hyperparameter selection. Those operations, finite-pulse approximation, limited context support and the constrained decision conversion all remain material. Very negative corrected validation scores are possible sampling outcomes, not negative physical risks or proof of a particularly good inverse.

A new privileged diagnostic starts from M(q)*z and optimizes the actual continuous endpoint error under the same torque bounds. A small residual plus the retained bounded command supplies an actual feasibility witness for that task; numerical optimizer failure would not prove it unreachable. Every oracle simulation call should be counted separately from ordinary acquisition/evaluation, and no oracle outcome may tune ordinary learners. This is stronger than treating the leading-order analytic command as an exact finite-time endpoint oracle.

Broad acquisition covers the evaluation pose support, so new exact poses there mainly test interpolation and nuisance handling. Narrow acquisition tested at outer q2 values probes unsupported extrapolation. Report those distinctions, including the newly included q2=0 in-support diagnostic; do not combine them into a claim that a learner either can or cannot generalize arbitrarily. Generic periodic kernels supply angle topology and a smooth function vocabulary, even though no physical inertia formula is handed to the learner.

These changes resolve the material pre-execution statistical/decision concerns. No further algorithm or comparator expansion is needed before the compact experiment. Preserve the restrictions, execute and audit actual goal consequences, then continue to continuous coupled learning as prescribed rather than reopening a calibration optimizer ladder.
