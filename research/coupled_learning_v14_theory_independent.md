# Independent v14 available-information repair audit

Date: 2026-10-20. This report supplements, and does not replace, the complete original two-reading audit preserved as `coupled_learning_v14_two_reading_pilot_theory_independent.md`. The original source, raw records, fits, results and design remain under the corresponding `two_reading_pilot` names. The earlier comparison was internally valid for its specified response statistic, but inefficiently used information explicitly available to every method.

## Noise and statistical identities

All learners receive exact preparation q0 and start at rest. Let f=.5*T^2=.005, d=(qT-q0)/f and oT=qT+eT, with E[eT]=0 and Cov(eT)=SD^2 I. The repaired statistic is

    dobs=(oT-q0)/f=d+eta,
    E[eta | q0,u]=0,
    R=Cov(eta | q0,u)=SD^2/f^2 I=.0004 I.

The original `(oT-o0)/f` is also unbiased under the declared independent sensor errors, with covariance .0008 I. Since exact q0 is supplied, however, e0=o0-q0 is known; retaining it adds avoidable noise. Initial readings are still preserved. This is not a demonstration of learning exact preparation, inferring hidden velocity or overcoming state uncertainty. If preparation were uncertain, correlated with endpoint noise or unavailable, the new covariance and identities would require revision.

For fixed training contexts and issued commands, inverse design-row noise is phi(q) tensor eta. In the source's (feature,input) flattening,

    E[Xobs^T Xobs]-Xtrue^T Xtrue
        =sum_n phi_n phi_n^T tensor R
        =.0004*kron(Phi^T Phi,I2).
    E[Xobs^T U]=Xtrue^T U.

Thus the repaired Gram subtraction is correct. It estimates noise-free moments without additive sensor-noise bias; it does not make fitted coefficients unbiased after eigenspectrum projection, ridge, inversion or model selection. Finite-pulse nonlinearity remains approximation error, not a violation of the additive-noise identity.

For a candidate B or K fixed given training and independent validation noise,

    E[||B u-dobs||^2 | q,u]=||B u-d||^2+trace(R),
    E[||K dobs-u||^2-trace(K R K^T) | q,u]=||K d-u||^2.

The source's inverse validation subtraction `.0004*sum(K*K)` is the required trace term for BOTH naive and corrected inverse candidates. Forward validation has the same candidate-independent +.0008 total two-output offset. Gaussianity is unnecessary; conditional zero mean, covariance and independent validation are sufficient. Validation selection still introduces optimism, corrected sample losses can be negative, and none of these identities establishes out-of-support accuracy.

## Repaired source and retained scope

The current `main` loads the archived original NPZ, copies all raw records and replaces only each acquisition/validation/predictive `response` field by `(observed_final-known_preparation)/f`. It uses `.0004` in both corrected-inverse training and both inverse validation scores. Contexts, commands, states, initial/final sensor records, splits, features, grids, bounds and selection rule remain the same. The unused legacy `experience` helper still constructs the old two-reading response, but `main` does not call it; the actual executed path is the explicit archived-data repair.

Coefficient orientation, EIV Kronecker ordering, spectral floor plus ridge, pseudoinverse cutoff and common bounded goal-displacement least-squares decision solver are unchanged from the original audit. The inverse and forward function classes remain different despite equal generic context-feature counts. Their performance cannot establish an architecture-wide preference or a model-free advantage.

Original acquisition and predictive raw pulses are reused: zero new training, validation or predictive physical interactions. The old 160 privileged feasibility witnesses remain valid for identical physical tasks; no new oracle optimization is needed. Executing the repaired commands entails 12,800 additional task-evaluation pulses, including repeated evaluation of the reused oracle commands, not additional learner acquisition. Raw-data identity and exact reconstructed numerical fits are the separate artifact auditor's responsibility; this report checks the mathematics and executed source path.

The repair was chosen after original outcomes and an information audit. It is an exploratory correction on the same fixed task grid, not a fresh confirmatory benchmark. No changed task result should be presented as independently sampled confirmation of the original ranking. New sensor readouts in task evaluation do not alter commands because context is supplied exactly; the deterministic goal outcomes are not made statistically independent by regenerating those unused readouts. Monte Carlo SE remains clustered over the eight acquisition datasets, not the 160 fixed tasks.

## Invariance diagnostic and interpretation

The canonical q1 diagnostic source was regenerated from its preserved two-reading version by replacing ONLY the input result filename; formulas remain unchanged. It measures retained commands at identical (q2,goal) across four supplied absolute q1 values. Its primary pooling is

    sqrt(mean_seed,q2,goal,q1 ||u-mean_q1(u)||_2^2).

The alternative mean of per-seed componentwise RMS is retained separately. Nonlinear pooling can change cross-method order. Zero q1 dependence is not sufficient accuracy (constant calibration has it by construction), and low goal error alone does not prove exact symmetry discovery.

This is still one compact rest-calibration relation-learning test, with supplied preparation, encoder coordinates, periodic smooth features, torque/response affinity, noise calibration, resets and goal metric. Finish auditing the repaired comparison and retain the complete old one, then move to continuous coupled motion or missing-history/model revision. Repeating covariance or calibration refinements would not by itself establish a more general learning mechanism.

## Independent retained-data check and observed repaired outcomes

After writing the mathematical/source audit above, read the repaired results and regenerated q1 diagnostic. Independently checked all160 non-response NPZ arrays against the original archive with exact element equality; all agree. Every response array exactly equals `(observed_final-known_preparation)/(.5*.1**2)`. This is a direct raw-record/response check, not an independent full numerical reconstruction of fitting or continuous plant integration.

Repaired broad taskMSE, in order constant forward/context forward/naive inverse/corrected inverse, is

    .0089823602 / .00003182548 / .00000898605 / .00000452803.

The paired naive-minus-corrected difference is +4.45802e-6 with clustered MC SE1.08175e-6; all eight dataset differences are positive. This supports the calibrated correction within this specific broad-support experiment. It does not establish universal risk dominance or equivalence of the forward/inverse function classes.

Repaired narrow-training broad-grid taskMSE is

    .01948820 / .05480937 / .01448839 / .01478161.

The corresponding naive-minus-corrected difference is -.000293217 with SE .000602516. Unlike the original two-reading comparison, these repaired results do not resolve a harmful corrected-inverse effect on this OOD grid. Do not copy the original stronger OOD-harm numerical claim into the repaired experiment. Narrow context forward remains particularly unstable (MC SE .02089068 and66/1280 saturated tasks); inverse saturation counts are0. Narrow inverse errors remain orders of magnitude above broad-trained inverse errors, although their means are below narrow constant calibration. The proper conclusion is failure of accurate broad-support transfer from these narrow datasets, not failure of every aspect of transfer or superiority of the constant fit over every contextual learner.

Primary pooled Euclidean q1 command-variation RMS, in order context forward/naive inverse/corrected inverse, is broad

    .01576066 / .00388909 / .00408140,

and narrow

    .22783029 / .02541424 / .02289744.

Constant calibration has0 by construction and the retained oracle has approximately2.82e-13. None of the ordinary context-dependent fits is exactly invariant. Both diagnostic pooling conventions remain explicitly available in the canonical JSON; cross-method ordering should be tied to the stated convention.

Noise reduction is an information-use repair, not a monotonicity theorem for the entire selected pipeline. In fact narrow context-forward taskMSE increases relative to the preserved two-reading result despite the smaller observation covariance. Finite realization, model selection, regularization and unsupported extrapolation can change with this statistic. Both full results must remain available rather than reporting only improved cells.
