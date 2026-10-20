# Independent joint Cartesian-state/response belief extension

Date: 2026-10-20. Pre-implementation derivation and falsification checks. This is exact only for the supplied constant-B Cartesian local model. The articulated DOP853 plant remains the external test, so its state-dependent B/curvature can invalidate exact-model calibration.

## Normalization and persistent dynamics

Keep one fixed Cartesian origin p0, timestep T and state

    z=(s,w,b), s=(p-p0)/T^2, w=v/T,
    b=(B11,B12,B21,B22), H(u)=I2 tensor u^T.

For issued torque u,

    s_next=s+w+.5*H(u)*b,
    w_next=w+H(u)*b, b_next=b.

Thus the proposed8x8 matrix with blocks[[I,I,.5H],[0,I,H],[0,0,I4]] is correct. Observation is y=(p_observed-p0)/T^2=[I,0,0]*z+eta with R=(SD/T^2)^2*I2. Only initial preparation may supply exact p0/rest; no state, covariance, mode weight or physical body reset occurs at later goals. Goal g=(p_goal-p0)/T^2 uses that SAME origin throughout.

With known realized u, Gaussian prediction is mu_minus=A*mu, P_minus=A*P*A^T (plus only an explicitly supplied process-noise model). Gaussian conditioning uses innovation y-C*mu_minus, S=C*P_minus*C^T+R, K=P_minus*C^T*S^-1 and the usual joint mean/covariance update. Preserve every s/w/b covariance block. An adaptively chosen u is known conditional on the past, so exact within-model Gaussian filtering remains valid even though unconditional action/state distributions need not be Gaussian.

A Gaussian mixture propagates/conditions each FULL8D component and updates weights by its predictive observation density. Per-component observation covariance is C*A*P*A^T*C^T+R, not merely R. No hard branch label/classification is needed. During unobserved issued pulses, propagate each joint component and leave weights unchanged. Do not collapse to separate parameter/state filters or rebuild s/w from a noisy position as if independent of B.

Freeze the observation schedule: after the informative first pulse, during committed pulses, at batch boundaries, and before the next goal's first action. State whether a terminal camera reading updates the NEXT batch. Different observation opportunities change the information problem even when no extra control pulse is issued. All algorithms need the same schedule and all readings counted.

## Calibration uncertainty and coefficient ordering

Conditional on component responsibilities and a point-mixture weighted linear fit, let Gk=sum_episode,t responsibility_k*X*X^T. A local Gaussian curvature approximation for each output row's fitted coefficients is R*Gk^-1. For current OUTPUT-ROW-MAJOR b, the4x4 covariance is

    I_output tensor (R*Gk^-1).

The alternative (R*Gk^-1) tensor I_output belongs to input-major ordering and is WRONG with H=I_output tensor u^T. Include declared ridge consistently in Gk; do not divide by episode count again if Gk is a sum.

This curvature is a plug-in approximation conditional on soft assignments, not full parameter posterior uncertainty under latent-assignment/model error. It can represent finite calibration uncertainty within each learned component. It is not automatically a learned physical within-mode random-coefficient covariance. Tiny components have low effective sample size and potentially large uncertainty; suppressing that uncertainty after observing poor performance would be a hidden intervention. Validation selection must use the actual predictive law if this covariance is added.

When forming a matched Gaussian, aggregate FULL joint moments:

    mu=sum_k pi_k*mu_k,
    P=sum_k pi_k*(P_k+(mu_k-mu)*(mu_k-mu)^T).

This includes between-mode s/w/B correlations. Initially s/w may be exact0, but after action those correlations carry precisely the information being tested. Matching only B moments and separately averaging state loses them.

## Remaining two-action expected-cost QP

At the posterior AFTER the first pulse, let U=(u1,u2), A=[1.5I,.5I], D=[I,I]. Terminal normalized state is

    s_terminal=s+2w+B*A*U,
    w_terminal=w+B*D*U.

Define d=s+2w-g, G=E[B^T B], h_d=E[B^T d], h_w=E[B^T w]. With the previous normalized cost, the quadratic coefficients are

    Q=(A^T*G*A+WV*D^T*G*D)/DS^2+WU*I4,
    q=(A^T*h_d+WV*D^T*h_w)/DS^2.

The linear terms include correlations. With row-major index b_(o,i)=b_(2o+i),

    (h_d)_i=(meanB^T*(mean_s+2mean_w-g))_i
             +sum_o[P_b,s(2o+i,o)+2*P_b,w(2o+i,o)],
    (h_w)_i=(meanB^T*mean_w)_i+sum_o P_b,w(2o+i,o).

G is meanB^T*meanB plus the sum of the two diagonal output-row covariance blocks. The objective is U^T*Q*U+2*q^T*U+constant, with positive WU giving a unique global box optimum as before. A direct augmented-state sigma-point/component loss and finite-difference gradient provide independent implementation checks.

The constant is NOT optional for choosing FIRST actions:

    [E||d||^2+WV*E||w||^2]/DS^2+WU*||u0||^2.

Here E||d||^2=||mean_d||^2+trace(P_ss+4P_ww+2P_sw+2P_ws). This constant depends on the predicted/conditioned state and first action, although it does not affect the inner minimizer. Dropping it would corrupt information/action selection. Do not double-count the first torque or add the previous batch's actions to current effort.

## Testable consistency and scientific limitations

Known noiseless B and initially exact s/w must reduce to ordinary deterministic double-integrator propagation/control. With initial rest and zero state covariance, the FIRST batch must reduce to the already audited parameter-only three-pulse formulas, including their first-action cost. A single Gaussian represented as an identical mixture must be numerically identical. With no future observations, matched full8D Gaussian/mixture moments must give identical expected costs and unique open-loop actions. Across goal changes, changing only g must not alter the carried belief. A zero observation innovation should not change its mean but should still reduce the appropriate uncertainty.

The exact constant-B joint KF is a strong conventional alternative, not a weak independent point estimate. Its limitation is distributional shape/approximation, not a missing state-parameter correlation. A learned multimode advantage should survive that competitor; a tiny-component calibration or quadrature artifact must not be called general intelligence.

Six successive small goals with carried belief address a real omission of the initial batch, but are still local prepared-body control. A repeated two-stage batch is not a fully receding sensor/action policy; name it precisely. Constant-B certainty can become overconfident under real state-dependent mechanics, and a failure over time may reflect that approximation rather than inability to retain useful knowledge. Preserve it and avoid changing process noise, mode count, filtering or action horizons until the frozen experiment reveals which uncertainty actually matters.
