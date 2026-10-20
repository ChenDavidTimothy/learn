# Prospective action-effect objectives: independent derivation and falsification

2026-10-20. This note follows the independently verified weak heldout command
contrasts. It proposes NO change to that pilot, its models, selection, goals or
fork scores. No new method is implemented. A later learning test would need its
own fresh/frozen protocol. These are conventional conditional-projection and
orthogonal-risk identities, not a new learning architecture or milestone.

## What random renewal really supplies

Let H be an available causal history or chosen history summary at a renewal,
U~Q=Uniform([-.5,.5]^2), independently of the full physical/past information,
and Y the next two ordinary camera positions (a4-vector, for example). The action
is held for.2sec; its law is known. Sigma=E[UU^T]=I2/12. Condition throughout on
an independently fitted model/training sample. Future camera noise has zero
conditional mean and does not bias these moments. Whole trajectories, rather
than randomly interleaved overlapping rows, must define training/validation and
cross-fitting folds.

Under affine conditional mean m(H,U)=d(H)+G(H)U,

    G(H)=E[(Y-dhat(H))U^T | H] Sigma^-1

for ANY genuinely past-measurable dhat. Mean-zero randomization eliminates its
cross moment exactly. More generally, WITHOUT affine truth, the same identity
identifies

    G_projection(H)=E[YU^T|H] Sigma^-1,

the conditional best linear projection underQ. It is not automatically the
current body's instantaneous J*M^-1, the derivative at zero command, the response
of the particular hidden state, or the finite difference at command.4.

For a fixed gain candidate, an ordinary validation score is

    S_G = tr(Ghat Sigma Ghat^T) - 2(Y-dhat)^T Ghat U.

Its expectation equals E|| (Ghat-G_projection)Sigma^(1/2) ||_F² minus the SAME
projection-norm constant for all candidates. The quadratic term can instead use
||Ghat U||², with the same expectation. This identifies the gain part of a class
without allowing a drift-only predictor to dominate its loss. It is an estimand/
objective choice, not a guarantee that the finite data contain enough information.

Regressing the pseudo-outcome Z=(Y-dhat)U^T Sigma^-1 onH has the same conditional
mean. But if different candidates use different dhat, their raw pseudo-outcome
squared losses have different noise constants. Comparing such raw losses without
accounting for this can select a nuisance/noise level rather than a better gain.
The cross-product score above avoids that expectation-level problem.

## Nonlinearity is a real limitation of the affine projection

- Y=U² in one dimension with symmetric commands gives G_projection=0 despite a
  strong action effect. A zero moment need not mean the learner has no useful
  action consequences to acquire.
- Y=U³ gives G_projection=E[U⁴]/E[U²]=3a²/5 for Uniform[-a,a], whereas the derivative
  at0 is zero. Changing the randomizer bound changes the projection of the SAME
  physical response. A controller concentrating on different commands can have a
  different best affine approximation even with perfect history sufficiency.
- The articulated body is affine in torque at the instantaneous acceleration
  level, but its finite.2sec endpoint map is generally nonlinear in command:
  the command changes the configurations/velocities that determine later response.
  'Small dt' is an approximation to quantify, not a license to call the projection
  physical inertia or an exact localB.

An affine moment can be a useful deliberately restricted relation. It must be
judged as such rather than used to explain every failed nonlinear contrast.

## General action-centered risk without affine truth

Let m(H,u)=E[Y|H,u], m0(H)=E_Q[m(H,U)], and

    g_true(H,u)=m(H,u)-m0(H).

For a FIXED candidate response mhat, center it under the same known randomizer:

    ghat(H,u)=mhat(H,u)-E_Q[mhat(H,U)].

Then E_Q[ghat|H]=0 and, for any past-measurable dhat,

    R(ghat)=E[||ghat(H,U)||²-2(Y-dhat(H))^T ghat(H,U)]
           =E||ghat-g_true||²-E||g_true||².

Proof: the dhat term vanishes by centering/randomization; E[Y^T*ghat|H,U]
replacesY bym; the action-independentm0 term vanishes; complete the square.
The result needs square-integrability and independent fitting/validation, not
an affine mean or a discovered physical state.

Numerical centering is not innocuous if it leaves an error delta(H). Write a
noncentered implementation q=ghat+delta. Its expected score adds

    E[||delta||²-2(m0-dhat)^T delta].

A small residual mean can be multiplied by a much larger drift. Centering accuracy
therefore needs checking relative to the action signal and remaining drift, not
only an absolute integration convergence flag. This is not a reason to start
another quadrature/optimizer ladder.

### A competing formulation avoids a centering integral entirely

Draw ONE independent auxiliary V~Q at the SAMEH; it is a virtual proposed command,
not another physical observation. For Dhat=mhat(H,U)-mhat(H,V), use

    S_pair=.5||Dhat||²-2(Y-dhat(H))^T Dhat.

Conditional onH,

    .5 E||mhat(U)-mhat(V)||² = trace Cov_Q(mhat,mhat),
    E[(Y-dhat)^T(mhat(U)-mhat(V))] = trace Cov_Q(m,mhat).

Hence E[S_pair] is EXACTLY the same centered-mean risk above. It does not need a
numerical action-average integral or another physicalY. The factor1/2 on its
quadratic term is necessary. Scoring ||Dhat||² without it adds a variation penalty
and changes the estimand. Common predeclared auxiliary draws across candidates
support paired comparisons. Auxiliary Monte Carlo and finite validation still
create variance; this is an unbiased score, not a perfect selector or guarantee.
It is a potentially simpler prospective diagnostic, not an implemented rescue.

## Nuisance precision, leakage and variance

Any past-measurable dhat leaves EXPECTATION unchanged. It does NOT automatically
reduce variance. For example Y=0,ghat(U)=U,dhat=C adds the random term2CU;
its variance becomes arbitrarily large as|C| grows. A good drift estimate can
reduce variance substantially, while a bad one can increase it. Conditional-mean
drift is a reasonable control variate but is not universally variance optimal
for every vector/model score.

If dhat is fitted on the same row's future outcome and can memorize it, the
residual can be identically zero even forY=U. 'Not supplyingU as an input' does
not make such a trained prediction past-measurable. Cross-fitting must preserve
trajectory independence or strict temporal causality. Similarly a candidate
chosen/tuned on the same validation responses is no longer fixed for an unbiased
post-selection performance claim. Independent test/fresh replication is needed.

The moments can have high variance: dividing by Sigma amplifies residual noise;
heterogeneous moving drift can dwarf the action signal. Cross-fitting, centering
and known randomization do not increase the number of independent trajectories.
Statistical uncertainty should be clustered at that level. No new guarantee of
root-n inference follows from the algebra alone.

## What these objectives still miss

Full-history randomization does not make a16-lag summary policy sufficient. The
conditional hidden-state mixture given that summary may change under a later
controller or after long quiet periods. The fitted effect is a training-law
conditional average, not a physical law guaranteed to transport under policy shift.

Even perfect centered MEAN estimation need not capture task-relevant response
variation. Let latentZ be equally likely+/-1, absent fromH, withY=ZU. Then
m(H,U)=0 and g_true=0 for EVERYU, although conditional variance isU² and an
informative earlier/later observation can change decisions. The mean risk correctly
favors zero but does not learn all distributional/interventional knowledge. V16
already demonstrates why uncertainty coherence can matter. This counterexample
prevents treating a focused mean objective as a universal control foundation.

The current exogenous law also does not persist in deterministic closed-loop
control: U may become a fixed function of history, its conditional covariance
can collapse, and dividing by acquisitionSigma is then invalid for online update.
Any future online moment use needs its actual policy randomization/positivity or
additional structural assumptions. No mandatory exploration policy follows.

## Is this worth pursuing after the frozen control pilot?

Potentially, if the next question is whether drift-dominated training/validation
allocated a finite model class poorly, rather than whether the camera records
fundamentally lack action information. Known experimental randomization makes a
more directly relevant mean-effect objective possible without true fork labels.
This could distinguish an objective/variance explanation from some alternatives.

It would not establish a new architecture or bypass representation/sample limits.
A joint drift+varying-coefficient regression under mean-zero randomization has
the same population affine normal equations. A fair follow-up should therefore
separate changed estimand/loss from changed class/capacity, keep honest conventional
partialling-out/conditional regression alternatives, and use fresh frozen evidence.
Do not retroactively select or rescale current models using their heldout forks.
The original charged-control pilot should first establish what its imperfect
forecasts actually do. No experiment or new candidate menu is proposed here.

## Established-research mapping, consulted after derivation

[Nie and Wager, Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://arxiv.org/abs/1712.04912)
develop a residualized loss isolating heterogeneous treatment effects and tune it
with cross-fitting. Their displayed setup uses binary treatments and independent
examples. It supports the connection to established residualized objectives; its
rates do not automatically apply to two-dimensional continuous torques, controlled
history summaries, overlapping rows or the pairwise score derived here.

[Chernozhukov et al., Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060)
combine orthogonal scores and cross-fitting to handle nuisance estimation and
overfitting. This provides established context for the nuisance-robust moment
perspective, not a ready-made guarantee for this dynamical learning experiment.
No originality or competitive-method claim is warranted by rediscovering these
covariance/centering identities.
