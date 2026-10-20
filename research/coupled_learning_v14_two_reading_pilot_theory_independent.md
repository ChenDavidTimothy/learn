# Independent v14 relation-learning mathematics/source/result audit

Date: 2026-10-20 (Europe/Berlin). Read the prescribed coupled_learning_v14 design and source before inspecting learning outcomes. Afterwards independently read retained summaries and computed the prescribed q1-dependence diagnostic. This report does not replace the separate artifact auditor's numerical reconstruction. No canonical source, experiment settings or historical ledger was edited.

## 1. Information and physical scope

All ordinary learners receive exact supplied rest PREPARATION q0, both joint encoders, command bounds, pulse duration and response-noise scale. They do not infer noisy current pose or hidden velocity, and receive no mass/inertia formula. Exact context is a strong physical preparation capability, common to all. The noisy initial reading is not the context input, resolving the shared-context-error issue from the early proposal.

The physical flow is the verified continuous horizontal point-mass two-link system, no gravity/friction/actuator memory, constant torque forT=.1 from rest. Normalized response is d=(oT-o0)/f, f=.5*T²=.005. Independent zero-mean encoder errors with SD1e-4 give additive normalized response noise eta with covariance R=.0008*I. Actual endpoint response need not be exactly torque linear; the learned matrix relations are declared approximations. Short pulses end moving and are not feedback control, reaching/stopping or a general robot learner.

The source imports physical flow for acquisition/evaluation and mass only for the privileged oracle start. Ordinary fitting uses known preparation, issued commands and noisy response, not evaluator states. Test commands depend on exact supplied preparation/goal and fitted relation; noisy test readouts are retained but do not feed an online observer. Thus this experiment does not establish noise-robust current-state inference or closed-loop skill.

## 2. Coefficient indexing and forward/inverse relations

For F context features, coefficients W have shape(2F,2), with flattened row(f,i) and output column o. Source features X[n,(f,i)]=phi[n,f]*source[n,i] imply

    predicted[n,o]=sum(f,i) phi[n,f]*W[f,i,o]*source[n,i].

Therefore the context matrix must have entries A[o,i]=sum_f phi[f]*W[f,i,o]. Both `relation` einsum(f,fio->oi) and validation einsum(nf,fio->noi) implement exactly this output/input orientation. For forward fits source=u,target=d; inverse source=d,target=u. No transposition or reshape error was found.

There are65 features: constant plus64 periodic Gaussian centers over BOTH angles. Generic angle periodicity and torque/response affinity are supplied priors. Equal feature counts do not make the forward and inverse function classes equivalent under matrix inversion. Leading mechanics already illustrates why: M entries are affine in cos(q2), while M^-1 has denominator2-cos(q2)^2. A generic RBF span can approximate these with different difficulty without any implementation bias or universal preference for one representation.

## 3. EIV moment correction and validation

At exact preparation q, inverse regressor noise is phi(q) tensor eta. In the source's(f,input) flattening,

    E[X_noise^T*X_noise | contexts]
        = sum_n phi_n*phi_n^T tensor R.

Since R=.0008*I, the implemented RESPONSE_VARIANCE*kron(phi.T@phi,I2) is correct. The cross moment X^T*u has no corresponding additive bias when eta is independent of context/issued command. This produces unbiased noise-free Gram/cross MOMENTS, not unbiased finite fitted coefficients after inversion, regularization or selection.

For a fixed candidate B/K and a separate validation pulse,

    E[||B*u-dobs||² | q,u]=||B*u-dtrue||²+trace(R)
    E[||K*dobs-u||²-trace(K*R*K^T) | q,u]=||K*dtrue-u||².

The source subtracts RESPONSE_VARIANCE*sum(K²), exactly the trace term. Both naive and corrected inverse candidates use this corrected validation loss. These identities require zero-mean independent response noise and fixed candidate given training; exact Gaussianity is unnecessary. Finite-pulse nonlinearity does not invalidate the noise identities, but does create approximation error in the chosen affine relation classes.

Selection among widths/ridges remains optimistic. Corrected validation can be negative from sampling variation, particularly for large-norm K. It is not a negative physical risk. The separate160-task evaluation is essential. Validation remains within the acquisition context distribution; it cannot certify extrapolation to unobserved q2 regions.

## 4. Eigenfloor, pseudoinverse and bounded decisions

For corrected or raw Gram G=V*diag(lambda)*V^T, source denominator

    max(lambda+ridge*scale, ridge*scale)
       = max(lambda,0)+ridge*scale

first projects negative eigenvalues to0, then adds positive ridge. Thus W globally solves a positive-definite EIGENFLOORED regularized quadratic; it does not solve the original possibly indefinite corrected normal equations. The raw Gram trace defines the scale, which is retained. Ridge, floor and basis conditioning add bias and can affect unsupported contexts.

The recorded `training_normal_equation_residual` is ||G*W-rhs|| for the UNREGULARIZED, possibly indefinite G. It need not be near zero even for a numerically exact regularized fit. It must not be interpreted as a solver error certificate. Independent numerical checking should use the floored-plus-ridge system if a stationarity check is wanted.

Forward fits supply B directly. Inverse fits supply K; source uses pinv(K,rcond=1e-8) as B, then EVERY ordinary method solves the same convex bounded goal-displacement least squares. This resolves the former action-space-clipping/goal-space-metric mismatch. Rank cutoff is a further explicit regularization; near singular K can amplify errors or lose response directions. Source retains K singular values, B condition and bounded-solver status. No exact reciprocity or global physical validity of learned matrices is assumed.

For full-rank B the bounded quadratic has a unique minimizer; rank deficiency can make it nonunique while remaining convex. Solver status and actual endpoint consequences remain the evidence. No private mechanics enters these ordinary decision solves.

## 5. Acquisition, evaluation and numerical scope

Eight independent datasets per broad/narrow condition contain512 fit+128 separate validation pulses, with known-zero-velocity reset at EVERY pulse. Conditions have equal acquisition budgets; narrow q2 support is[-.4,.4], broad covers the full angle circle, and q1 varies in both. Seed role/replicate/condition ranges are disjoint. All candidates and eigenvalues/settings are retained, with no winner-based dropping of acquisition rows.

The160 predetermined context-goal tasks now use eight unique periodic q2 values, including0, crossed with four q1 values and five common goals. They are not160 independent stochastic test datasets. MC SE must cluster by the eight independent acquisition datasets, as implemented. Broad held-out exact poses are mainly in-support interpolation; outer q2 tests after narrow training are unsupported extrapolation. The128 fresh broad predictive pulses additionally probe a different random-command distribution, not the same decision risk as the small requested goals.

The privileged continuous endpoint solver uses true mass only to initialize its bounded torque search; actual flow residual supplies feasibility. It found a bounded command for every task, maximum normalized residual3.24e-13. This is a numerical witness with retained commands, not an analytic global feasibility theorem; failed optimization would not prove unreachability. All1456 feasibility-search rollouts and repeated actual evaluations are counted separately from ordinary acquisition. Ordinary fit/selection never reads oracle solutions.

No fatal source/mathematical error was identified. The appropriate comparison is among these supplied relation/statistical/decision pipelines. Equal physical data does not equate priors, representational complexity, regularization, numerical conditioning or computation.

## 6. Actual findings and bounded interpretation

After execution, independently read saved summaries. Broad-acquisition taskMSEs are:

    constant coupled forward .008981408
    context forward .0000702875
    naive inverse .0000179176
    corrected inverse .00000941635.

The inverse advantage is legitimate evidence within these parameterizations and tasks. It is not a model-free victory, proof that prediction is unnecessary or discovery of an inertia law. Both inverse relations are converted into a response matrix for the same decision solver, and their finite-family/statistical geometry differs from the forward family. Calibrated noise correction improves broad taskMSE beyond naive inverse by8.50123e-6+/-1.13781e-6 (paired clustered MC SE).

Narrow-acquisition taskMSEs are .0196274/.0455312/.0231309/.0254856 in the same order. Context-conditioned learners do not generalize reliably to the unsupported broad pose grid; the more elaborate forward fit is worse on average than constant calibration, with high seed variation. This does not show context is unnecessary, complexity is generally bad or no learner can transfer under additional physical priors. Corrected inverse is WORSE than naive by .00235467+/-.000669447 on this OOD grid. Unbiased moment subtraction does not solve missing coverage or guarantee better transfer after inversion/regularization.

Broad task saturation counts are zero; narrow counts are49/1280 for context forward and12/1280 for each inverse. True targets have retained feasible witnesses, so those saturations are properties of fitted decision relations, not evidence that the original goals were physically unreachable. No method-specific difficult goals were removed.

## 7. q1 symmetry was tested, not presumed

True joint-displacement dynamics are exactly invariant to absolute q1 in this horizontal instrument. A learned periodic fit over both encoders need not discover that exact invariance. `audit_coupled_q1_variation_v14.py` groups actual retained commands at the SAME(q2,goal) across four q1 values and retains every per-seed statistic/results JSON.

The primary reported diagnostic is

    sqrt(mean_{seed,q2,goal,q1} ||u(q1,q2,goal)-mean_q1 u||_2²).

Broad pooled Euclidean RMS command variation is .0227039/.00471410/.00525049 for context forward/naive inverse/corrected inverse. Narrow values are .152682/.0411531/.0443252. Constant calibration has0 by construction despite its poor task accuracy; privileged oracle variation is about2.82e-13. Thus zero nuisance dependence alone is not sufficient knowledge, and none of the learned context relations has exact inferred symmetry.

A separate legitimate diagnostic averages per-seed componentwise RMS AFTER square roots; it gives different numbers and can change the naive/corrected ordering because of seed heterogeneity. Both definitions/values are retained in the independent JSON. Do not conflate these pooling conventions or claim one method is universally more invariant from one aggregate. The robust conclusion is that better broad goalMSE does not prove exact symmetry discovery.

## 8. Direction audit

This is a useful first complete coupled consequence-to-goal test and supports context-sensitive use of experience under declared preparation/basis/noise assumptions. It does not acquire an observation language, discover dynamics from arbitrary sensors, perform continuous articulated control or establish a general learning framework. Rest preparation, encoder coordinates, periodic kernels, affinity, loss and resets are supplied.

Do not run another inertia-identification or calibration-optimizer ladder. The compact experiment has supplied both a positive capability and negative extrapolation result. Move to continuous coupled motion or a meaningful missing-history/model-revision question, letting simple established alternatives remain genuine competitors. Additional physical/context priors may help, but should be acquired or supplied explicitly rather than silently inserted to manufacture extrapolation success.
