# Independent pre-outcome Cartesian study delta audit

2026-10-20. No canonical source was edited. This is an audit of the frozen study
changes before fresh acquired-corpus/control results, building on the independent
joint-state derivation, direct sigma-point/Torch cost checks, precision-matrix
filter checks, and actual pilot quadrature falsification. It is not an independent
reimplementation of the entire learning/controller experiment.

`audit_cartesian_feedback_v16_study_delta.py` and its actual JSON results are the
executable record. Source hashes identify the checked code. AST comparisons show
that `transition`, the entire `Belief` class, `future_qp`, `plan` and nonlinear
nominal-witness code are unchanged from the separately audited feedback pilot.
The imported base `box_qp` is unchanged from the audited refinement base.

## Numerical correction

The four rectangles [-7,0]x[-7,0], [-7,0]x[0,7], [0,7]x[-7,0], [0,7]x[0,7]
cover exactly the old standardized domain, with overlaps only on measure-zero
boundaries. Per-quadrant absolute tolerance is total_atol/4. Weighted component
errors are summed, and the previously derived analytic outside-domain bound is
unchanged. Gk21 estimates remain numerical estimates, not rigorously enclosing
bounds; partitioning and tighter rescoring are empirical safeguards.

Seven ACTUAL prior-pilot belief/action fixtures were checked through the new
`expected_first`, against a separately orchestrated tighter partitioned gk21
calculation using the old independently algebra-audited core. Fixtures include
first-rest winners/runners and each of the three specific Genz–Malik failures.
Maximum absolute difference is4.37e-12. No fresh nonlinear control experiment was
run to obtain this check. This verifies the numerical source change on meaningful
actual examples, not just the analytically convenient synthetic fixture.

The new design re-evaluates the original lowest three candidate values at tighter
tolerances, stores those records and recomputes the selected minimum afterward.
The assertion allows a declared finite discrepancy; it is not a proof that the
other candidates' ordering is correct. A fourth candidate could in principle
become relevant if a top value changes. The explicit future requirement to audit
actual selected/nearby forecasts and retain failures is therefore appropriate.
This does not justify reintroducing an unqualified 'certified optimum' claim.

## Menus, covariance diagnostic and likelihood

The geometric action menu is common: zero plus sixteen uniform torque directions
at each of two radii. Each prior additionally receives its own cautious-plan first
action. There are34 candidates. The full action sets need not be identical across
priors; this is a fair common algorithmic option, not strict identical-candidate
counterfactual isolation. The two random-radius averages correctly use candidates
1:17 and17:33, respectively, and the actual random angle uses the same sixteen
uniform directions. Learned eigenvectors no longer add candidates. The remaining
eigenvalue calculation is diagnostic only and does not influence actions.

Isotropic initialization preserves the learned general Gaussian mean exactly and
replaces Gamma by trace(Gamma)/4 times I4; executed checks verify the mean, trace
and matrix transformation. This changes more than off-diagonal entries. Writing
row-major Gamma in2x2 row blocks, the prior second control moment is

    E[B^T B]=meanB^T meanB+Gamma11+Gamma22.

Under isotropy the covariance contribution becomes trace(Gamma)/2 times I2.
Consequently even open-loop expected quadratic costs generally change. This is a
diagnostic of directional/coupling structure beyond mean and total variation,
not a pure removal of cross-action coherence or a uniquely attributable mechanism.
The study design states this limitation correctly.

By contrast the full moment-matched Gaussian and mixture have identical first and
second augmented-state moments initially. Fixed open-loop expected quadratic
costs/actions then agree, as independently derived and tested earlier. Shape can
matter through conditioning/anticipating future observations; no theorem makes
explicit mixture modes necessary. At later histories their filtered moments may
diverge, which is an experimental consequence rather than an error.

The generalized one/two-read predictive likelihood uses the correct output count,
H construction and known-R covariance. For the general Gaussian, the new grouped
NLL agrees with the separately existing Gaussian NLL and old two-read mixture-NLL
implementation to floating precision. Joint minus first-read average NLL is the
average second-given-first NLL under the SAME prior law. Heldout forecasting does
not enter fit, selection or action decisions. Re-evaluating all17 already fitted
mixtures under the actual curvature-augmented predictive law on validation groups
is legitimate, with the existing approximate-curvature caveat preserved.

## Freshness, resources and claim limits

Acquisition uses independent train/validation/test group roles1/2/3, EM-start role5,
and replicateIDs100..107 within the630 namespace. Control uses the separate640
namespace and replicate ID. Fixed derivative-check directions in the imported
Gaussian refinement use a separate role8 path with rep0; these repeated numerical
check directions do not couple acquired noise or control outcomes across corpora.
They are numerical diagnostics, not fresh physical samples or extra fit starts.
One fixed group law and two fixed hidden preparations are sampled; eight corpora
are not eight different physical systems.

No early physical state or coefficient reset is introduced at later goals.
Preparation/rest is exact and supplied only initially; all later ordinary updates
use actual actions and camera observations. Full state/B cross covariance is
retained. Under the constant-B approximation the Gaussian filter is exact
conditionally on realized causal actions. Under the true nonlinear arm it is an
approximation that can become overconfident; no process/drift uncertainty was
silently added. Nominal nonlinear witnesses do not certify every failed recovery.

The geometry count34/design and tighter rescoring count3/design are distinct.
The source's integration-point total includes both sets of integrations; resource
reports should distinguish unique candidates from expected-first numerical calls.
Acquisition resources are stored separately from feedback resources. Feedback's
'parent acquisition reused' field means reuse within this feedback stage, including
when the same invocation created that acquisition artifact; consolidated study
cost must include the acquisition record. All exact repeated preparation is an
unpriced strong resource, as the design explicitly acknowledges.

No blocking mathematical/source issue was found. The study is a defensible bounded
construction test: acquired local response alternatives are carried through causal
Cartesian control. It does not establish general learning of unknown physical
vocabularies, necessity of multimodality/probing, arbitrary per-pulse feedback,
modern-method superiority, or a universal mechanism. Honest Gaussian catch-up,
null isotropic effects, approximation failures and all eight corpus results must
remain visible. The next framing should revisit the supplied preparation/local
response family instead of extending an optimizer or covariance ladder.
