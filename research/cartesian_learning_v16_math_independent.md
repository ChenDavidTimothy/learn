# Independent grouped Cartesian-learning and three-pulse decision audit

Date: 2026-10-20. Pre-execution mathematics/design audit. No canonical implementation was edited. All constant-B statements below are exact for the declared local Cartesian double-integrator approximation, not for the full finite-time articulated ODE.

## Why grouped experience is necessary for the general covariance comparator

Use output-row-major b=vec(B)=(B11,B12,B21,B22). A single response y=B*u+eta has observation matrix H(u)=I2 tensor u^T. Its conditional covariance is H*Gamma*H^T+R. A general symmetric4x4 Gamma has an unobservable direction: the skew part of its2x2 cross-output block. For skew A, u^T*A*u=0. For example Gamma=I4 plus or minus[[0,A],[A^T,0]] is positive definite for sufficiently small nonzero A and gives identical single-action marginals for every u. Yet cross-action covariances H(v)*Gamma*H(u)^T differ. Those affect a probe-conditioned future decision. One-pulse population fitting cannot claim to identify the complete Gaussian matrix prior.

Rank-one covariance is a restriction that can remove that ambiguity, but it must be supplied honestly. The proposed GROUPED continuous two-pulse acquisition is the stronger repair. From known Cartesian p0 and rest, with timestep tau and two independent commands u0,u1,

    p1-p0=tau^2*B*(.5*u0),
    p2-p0=tau^2*B*(1.5*u0+.5*u1).

Stack these4 scalar outputs with matrices H1,H2 to obtain y=Hgroup*b+eta. Hgroup is invertible whenever u0,u1 are noncollinear, which occurs almost surely for independent continuous random commands. Thus repeated unlabeled groups can identify the full covariance, including cross-time/output information, in the population local model. One group is not a precise covariance estimate. No reset occurs between its two pulses; B remaining constant is a supplied local approximation as the true body moves.

With exact p0 and independent endpoint camera errors, Rgroup=SD^2*I4. If p0 is instead a noisy common baseline, the two displacements share its error and Rgroup has off-diagonal blocks; conditioning on the initial reading also matters. Openly supplying exact CARTESIAN preparation/rest is a strong common resource, but it does not reveal the joint branch. Keep that resource explicit rather than recreating the earlier avoidable-noise error.

## Fitting moments and persistent latent alternatives

For independent commands and a group-persistent b with mean mu and covariance Gamma,

    E[y|H]=H*mu,
    E[(y-H*mu)(y-H*mu)^T-Rgroup|H]=H*Gamma*H^T.

A symmetric10-parameter moment least-squares fit can use basis matrices Eii and Eij+Eji. The operator must include the full grouped covariance, not just each endpoint's marginal. Packing only the upper triangle is permissible, but differs in finite-sample weighting from Frobenius loss unless off-diagonal weights are included. Specify that choice. Fitting mu on the same records introduces finite-sample residual-covariance bias; subsequent PSD projection/ridge/selection is not an unbiased covariance estimator or Gaussian MLE. A negative raw eigenvalue is not hidden evidence of a negative physical variance. Preserve raw/projected spectra and regularization.

For a K-component point mixture over b, each WHOLE GROUP has density

    sum_k pi_k*N(y;H*b_k,Rgroup).

EM responsibilities use the4-dimensional grouped likelihood and remain one latent assignment per episode. The M-step is responsibility-weighted linear least squares in b_k (plus any explicitly declared ridge), and pi_k is the mean responsibility. Apply stable log-sum-exp. Do not assign independent component labels to the two rows/times. Fixed positive sensor covariance prevents the usual free-covariance mixture collapse to an infinite likelihood, but local maxima, vanishing components and overfitting remain. Retain starts, training criteria and all validation-selected K1/2/3 candidates.

Whole independent groups must be split for fit/validation/test. Model selection may use grouped observation likelihood; it must not read true B, branch labels, joint state or selected task outcomes. More than two selected components can approximate finite-time nonlinear residuals; it does not prove the physical body has that many modes. Known Gaussian sensor noise is not a justification for treating all local-model approximation error as Gaussian noise.

A Gaussian matched to a learned point mixture has mu=sum(pi*b) and Gamma=sum(pi*(b-mu)*(b-mu)^T). This is a valuable shape control. Compare it separately from an independently moment-fitted general Gaussian: the latter can differ in first/second moments and calibration, not merely multimodality. A low-rank continuous Gaussian prior may already exploit the same reusable variation; discreteness is not logically mandatory.

## Conditioning on the real first pulse

For a chosen first command u0, the observed first displacement is y0=H0*b+eta, H0=.5*tau^2*(I2 tensor u0^T). Under a Gaussian prior,

    S=H0*Gamma*H0^T+R,
    mu_post=mu+Gamma*H0^T*S^-1*(y0-H0*mu),
    Gamma_post=Gamma-Gamma*H0^T*S^-1*H0*Gamma.

Use linear solves, symmetrization/numerical PSD tolerances and positive camera noise. Under a point-mixture prior, posterior weights are proportional to pi_k*N(y0;H0*b_k,R); component matrices themselves remain fixed. These are ordinary conditional distributions, not a new learning principle. Conditioning is approximate for actual nonlinear flow.

The real first endpoint observation updates B. Do NOT also replace p1 and v1 by independent exact values inferred from that same noisy endpoint while ignoring their correlation with B. In this local model p1=p0+.5*tau^2*B*u0 and v1=tau*B*u0; propagating the posterior from known p0/rest uses all information consistently. No true Cartesian or joint velocity should enter ordinary planning.

## Convex remaining-control problem

Commit the two remaining bounded torques u1,u2 after observing the first pulse. Under the local model,

    terminal_displacement=tau^2*B*a, a=2.5*u0+1.5*u1+.5*u2,
    terminal_velocity=tau*B*c, c=u0+u1+u2.

For posterior mean Bbar and Q positive semidefinite, define SQ=E[B^T*Q*B]. With covariance output blocks Gamma_ij,

    SQ=Bbar^T*Q*Bbar+sum_ij Qij*Gamma_ij.

For diagonal identity Q this uses Gamma11+Gamma22. For a mixture compute SQ by weighted component B_k^T*Q*B_k. Then with desired displacement d,

    E||tau^2*B*a-d||_Qp^2
      =tau^4*a^T*SQp*a-2*tau^2*d^T*Qp*Bbar*a+d^T*Qp*d.

Add the velocity term lambda_v*tau^2*c^T*SQv*c and declared quadratic effort. This is a convex quadratic in the4 remaining action coordinates. Positive effort weight makes it strictly convex and the box optimum unique. An independent complete check can enumerate3^4=81 free/lower/upper active sets, solve each free block and check bound/KKT signs. This is preferable to accepting a local optimizer status as the only evidence when the inner problem has an elementary global certificate.

Mixture and matched Gaussian with the SAME first two moments give identical expected costs for any fixed OPEN-LOOP sequence. Any shape-dependent control difference must arise through conditioning on observations or anticipating that future conditioning, not merely evaluating this quadratic before feedback. With u0=0, H0=0, so the observation conveys no B information; matched priors retain identical moments and unique remaining actions. This is a useful implementation consistency check.

Choosing an informative first u0 while anticipating the posterior/remaining optimum is an OUTER nonlinear decision problem; inner convexity does not make that full optimization convex. A maximum-separation or variance probe is a heuristic, not optimal task-aware dual control. Compare equal effort/duration and include its actual task excursion. A random probe can have equal torque energy yet different task cost. A predeclared common finite action set can simplify this engineering comparison without a global-outer-optimum claim.

## Fixed-probe feasibility can fail before learning does

With invertible true B, exact terminal stopping after all three local pulses requires

    z=B^-1*d/tau^2,
    u1=z-2*u0, u2=u0-z.

Therefore every tested branch/goal/first-probe combination needs |u1|,|u2|<=1 as well as |u0|<=1. Informative large probes can consume stopping authority. For opposite radial targets' branch-normalized commands z and-z, a common probe must satisfy componentwise |u0_j|<=(1-|z_j|)/2; when |z_j|>1 no common probe allows both branches' exact three-pulse stop. This is a local evaluator feasibility restriction, not true B supplied to an ordinary learner.

Before inspecting learning outcomes, verify actual nonlinear bounded endpoint position/velocity witnesses for the prescribed branch/goal/probe grid, including nonzero motion after the first pulse. The remaining2torques give4 variables for4 Cartesian terminal constraints; local nonsingularity gives a reasonable existence route near the approximation, not a finite-time numerical certificate by itself. Retain witnesses and count their privileged calls. Do not discard failed goals after seeing controller outcomes.

Goal-success tolerances must be stricter than the displacement itself; three short pulses can have millimeter-scale targets, so the old .03 threshold could make zero action trivially successful. Evaluate actual Cartesian position AND instantaneous velocity, action effort and every interaction, with latent truth only for external scoring. Sensor-derived velocity reward is not automatically unbiased under this direct-actuation timing.

## Scientific scope

This two-stage, three-pulse contract deliberately commits the final two actions without using intermediate feedback. All methods must share it; record any acquired-but-unused intermediate observations. It is a bounded first engineering comparison, not a demonstration of optimal sensing, sustained camera-only feedback, discovered state, general robotics or a universal need for probabilistic modes. A credible broader milestone requires an actual acquired useful distinction and, if warranted, sustained feedback beyond the prepared short-pulse contract. Preserve ordinary Gaussian/local-Jacobian sufficiency or random-probe equality as genuine possible outcomes.
