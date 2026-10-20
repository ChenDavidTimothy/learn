# Independent persistent Cartesian-feedback source audit

Date: 2026-10-20. Audited source/design before physical control outcomes. No canonical code was edited. `cartesian_joint_belief_v16_math_independent.md` contains the derivation; this records implementation falsification tests and remaining limits.

## Findings

No substantive transition, filtering, correlated-QP or normalization error was found. The8D transition has the correct row-major H=I2 tensor u^T. Gaussian component conditioning uses the complete joint covariance, including s/w/B cross blocks. Mixture weights use the correct predictive likelihood. Covariances are symmetrized, goals retain a fixed Cartesian origin, terminal observations update the following goal, and no physical/controller state is reset at switches. Initial design caching is legitimate because all trials share the same prior/current first goal there; later action designs use each trial's current belief.

Calibration curvature uses I_output tensor (R*Gk^-1), the correct row-major order. Re-evaluating all17 fixed mixture candidates under their changed predictive Gaussian-component law is proper: otherwise selecting the original point-mixture candidate would use a different validation density from deployment. That is conventional predictive uncertainty/model selection, not a fully integrated empirical-Bayes posterior or discovery of physical labels.

`future_qp` correctly includes mean and covariance contributions to E[B^T B], E[B^T(s+Nw-g)] and E[B^T w]. The trace terms in E||s+Nw-g||^2 and E||w||^2 are present. The first-action effort is charged once. This matters particularly for outer first-action comparisons, where these apparently constant inner terms vary with current action/observation. The physical normalized cost and posterior cost agree in units.

Known-pulse transition is linear in the augmented state ONLY under the supplied constant-B model; adaptation of issued actions to past observations does not invalidate conditional filtering because each realized action is known. The true articulated body is nonlinear and state dependent; exact within-model filtering is not an exact physical posterior.

## Independent executed falsification tests

`audit_cartesian_feedback_v16_theory.py` imports the canonical module only as the implementation under test. It separately propagates deterministic16-point Gaussian sigma sets through the normalized s/w/B equations and sums terminal losses over components. Torch differentiates those direct losses, rather than using the implementation's QP moment expansion. Synthetic beliefs contain dense state/parameter cross covariance and exercise two/three remaining pulses, including active action bounds.

- Direct expected-cost discrepancies<=1.07e-14; projected-gradient residuals<=9.00e-14.
- Full8D moment-matched Gaussian/mixture open-loop actions agree<=4.45e-13.
- Independent sigma-point transition moments agree<=2.23e-16.
- Conditioning computed independently through the inverse JOINT precision matrix and SciPy Gaussian densities agrees<=3.34e-16.
- Initially exact rest/state reduces to the already audited parameter-only first-batch continuation within5.00e-16.

A separate known-B Gaussian-state fixture keeps all action bounds inactive throughout the integration rectangle. Its posterior-optimal cost is quadratic in observation, so moment points integrate the untruncated expectation analytically. Adaptive integration differs by1.14e-13, within its reported estimated error and analytic tail allowance. This checks one nontrivial integration case; it is not a verification of every future mixture integrand or selected first action.

## Tail derivation and numerical meaning

For independent standard normal coordinates and a=7, the code's quantities bound

    P(outside[-a,a]^2)<=4*Phi(-a),
    E[||Z||^2*1_outside]<=4*a*phi(a)+8*Phi(-a).

Given a generating predictive component j, each updated component's conditional mean is affine in its standard-normal z. Expected ZERO-remaining-command cost is a quadratic plus a covariance trace. Applying ||x+Jz||^2<=2||x||^2+2||J||op^2*||z||^2 yields a_k+b_k*||z||^2. The posterior mixture of such costs is at most max_k(a_k)+max_k(b_k)*||z||^2; optimized nonnegative continuation loss is at most this feasible-zero cost. The weighted analytic tail formula is therefore valid in the declared model.

Finite-domain cubature error is an adaptive numerical ESTIMATE, not a rigorous interval certificate. Nearby candidate values can have unresolved orderings even after a convergence flag. First-action selection remains a finite-candidate approximate dual-design heuristic, not globally optimal stochastic control. Retain its error/tail records and avoid precise rank claims for overlapping numerical uncertainty. Numerical clipping of tiny negative loss roundoff is harmless only when direct nonnegativity/cost checks remain valid; it must not conceal a substantive quadratic error.

## Foundational and experimental scope

The observation schedule is explicit and equal: after the first and third pulse of each goal; the third read updates the next batch only. This is continuous physical two-stage-per-goal feedback with persistent belief, not arbitrary per-pulse feedback. Nominal nonlinear witnesses use their own feasible zero-first trajectories; they do not establish recoverability from every state reached by a failed controller or arbitrary selected probe.

One parent dataset, two fixed goal sequences and two paired sensor repeats are engineering evidence. There are four noise paths, shared across methods and hidden branches, not64 independently learned bodies. No new calibration data or true state enters ordinary decisions. Cartesian preparation/rest, fixed local linear B family, known noise/duration/bounds, candidate action grids and loss remain substantial supplied assumptions. Learning a useful multimodal prior within them would not establish a universal representation or need for explicit modes.

Without process/drift noise the local-B filter can become overconfident as the true body moves. Poor long-run behavior may indicate model approximation rather than failed memory or a fundamental information impossibility. Conversely ordinary joint Gaussian adaptation may catch up after the first observation; that is a serious alternative outcome. Do not tune extra mode counts/noise/filter variants merely to retain an apparent mixture advantage.

One reporting detail was flagged: the resource field `evaluated_action_candidates` sums persisted design entries, including initial designs reused from cache across branches/repeats/modes. That is a count of referenced candidate records rather than unique integration evaluations. Actual computational calls need a separate cache-aware count or explicit labeling. This does not change control decisions.

## Resource-count correction and actual-integral audit (after partial outcomes)

The resource-count concern above was corrected BEFORE the feedback run: attached
candidate records and unique actually computed designs/integration points are now
separate fields. Cache identity is counted before JSON serialization. This resolved
reporting issue does not affect the policy.

The analytic Gaussian fixture did NOT establish the reliability of Genz–Malik on
actual posterior-dependent controller costs. Following an independently discovered
false-convergence example in the earlier batch experiment, I checked actual
feedback candidate integrals. These are independent numerical quadrature checks
of the separately algebra-audited implementation; they are not another independent
controller derivation or reproduction of nonlinear physical outcomes.

Executable sources are `audit_cartesian_feedback_v16_quadrature.py` and
`audit_cartesian_feedback_v16_actual_integrals.py`. Saved JSON records identify
input/source hashes, original values/error estimates, all generating-component
integration results, integration point counts, convergence status, and differences.
The input was a partial outcome snapshot containing all24 dual-policy episodes
(8 each for mixture, general Gaussian and matched Gaussian). I checked:

- All22 initial candidates for both goal sequences and all3 priors:132 integrals.
- The original top3 candidates at each of5 later phases in all8 dual episodes per
  prior:360 integrals.
- Separate tighter, quadrant-partitioned gk21 checks of initial mixture winner and
  runner-up, plus three actual problematic integrals and their nearby decisions.

The first sweep used tensor Gauss–Kronrod `gk21`, atol1e-8, rtol1e-7, max4000
subdivisions on the SAME truncated standard-normal domain[-7,7]^2, component by
component. Every checked component reported convergence. No initial selected
candidate changed among ALL22 checked candidates; no later selected candidate
changed among the checked original top3. This does not certify that an unchecked
later candidate or random-radius average cannot change. Nearly equal numerical
minima also do not imply meaningfully different physical commands or costs.

Substantive Genz–Malik error-estimate failures WERE found:

| Actual record | Original value | gk21 value | Original estimated error |
|---|---:|---:|---:|
| General Gaussian, sequence1 initial, candidate3 | .0034018316882792774 | .0032990266784155794 | 9.990883251814944e-7 |
| Mixture, episode6 phase5 candidate9 | .0007566573173783163 | .0007354486956037184 | 8.815532615388832e-7 |
| Matched Gaussian, episode39 phase1 candidate15 | .36034618720469336 | .3603360337798136 | 4.492253577185983e-6 |

Here episode and phase indices are ZERO based, and identify the stored pilot
snapshot (the full records also include sequence, branch and repeat). These are
actual posterior/goal/action integrals, not synthetic fixtures. Splitting the
standardized rectangle into four quadrants and using gk21 atol1e-10, rtol1e-9 gives,
respectively, .0032990266901477307, .0007354486956036915, and .36033603378237056.
Agreement with the first gk21 sweep is at most1.18e-11 for these three records,
whereas disagreement with the original Genz values reaches1.02805e-4. The analytic
tail bound does not explain these in-domain discrepancies.

Across initial and later sweeps, the largest difference for an actually selected
action was1.38672e-6 (general Gaussian episode21 phase1); ordering among the checked
candidates remained stable. Larger errors occurred in unselected candidates.
Therefore this audit rejects treating the original integrator's convergence/error
flags as reliable certificates. It does NOT show that the checked feedback policy
choices or physical outcomes were numerically wrong. Both findings must survive.

A practical correction is to use quadrant-partitioned gk21 for ordinary finite
candidate design and independently check winners/near runners with tighter
quadrature or additional partitioning on actual integrands. Preserve the old run;
any correction after observing its outcomes is exploratory. No adaptive quadrature
estimate is a rigorous enclosure. Report finite candidate search, numerical error
estimates and analytic truncation bounds separately. If selected/near-tied actions
remain unresolved, refine or report a numerical tie rather than claiming an exact
minimizer. Do not add controller/model variants to recover a favored outcome.
