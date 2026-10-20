#!/usr/bin/env python3
"""Build v6 once, preserving the complete v5 text verbatim."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
r=json.loads((root/'context_geometry_v6_results.json').read_text())
prior=(root/'first_principles_robot_learning_ledger_v5.md').read_text()
doc=r'''# First-Principles Robot Learning: Living Research Ledger v6

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited identifiability theorem, constructive intervention tests and corrected baseline comparison. **Code/results:** `research/context_geometry_v6.py`, `research/context_geometry_v6_results.json`; independently executable rank check `research/context_rank_v6.py` and its results JSON. Full v5 and all prior history follows unchanged. V5 was verified on SSH origin/main at `a342bb6510c248659a4f15ff6f846102c9b4c6e3` before this cycle began.

## Why this question superseded a toy efficiency comparison

V5 proved that biased first-order interventions recover one bilinear family. The planned next step was to optimize their efficiency against centered mixed moments. Before that comparison, an independent counterexample exposed a more consequential question: **Which dimensions of a history-dependent mechanism remain invisible when all contexts are scalar operating-point shifts?** A richer estimator cannot reconstruct information absent from the retained summary, while more samples of a restricted intervention family may not create the missing information.

**PROVEN under assumptions:** For a quadratic conditional mean of a d-dimensional command history, exact output means and complete first-order response maps at every mean in a k-dimensional subspace and every common iid Gaussian amplitude leave a precisely characterized nullspace. If h=d-k, its dimension is `max(0,h(h+1)/2-1)`. Thus constant mean shifts plus all common amplitudes cannot identify every quadratic history interaction once history dimension is at least three.

**TESTED / CONSTRUCTIVE:** A three-command actuator-memory term is invisible to all those means/footprints. Periodic mean schedules expose it through phase-specific first-order gains; changed temporal correlation exposes it through the output mean; second-order regression learns it from centered probes.

**REJECTED/CORRECTED before release:** An exploratory comparison suggested phased probes were more efficient than centered second-order estimation. The independent auditor supplied an equally informed, much more efficient centered regression. It reversed the ranking against the simple phased estimator; the same efficient regression on both datasets shows no consistent winner. Structural identifiability and estimator efficiency are separate questions.

The result concerns necessary information for particular model/statistic/probe classes. It is established Gaussian-score/Wiener/linear-algebra mathematics, not a new universal learning algorithm or evidence of humanoid intelligence. The overall Goal remains active.

## Observables, assumptions and competing explanations

The robot issues a scalar command a, observes only noisy position, and remembers issued commands and the programmed mean schedule. Consider two temporal contrast coordinates of its last three commands:

~~~text
A_t=(a_t,a_(t-1),a_(t-2))
v=(1,-1,0)/sqrt(2), w=(1,1,-2)/sqrt(6)
F_t=(v^T A_t)(w^T A_t)=A_t^T Q A_t
Q=(1/sqrt(12))*[[1,0,-1],[0,-1,1],[-1,1,0]].

p_(t+1)=rho p_t+beta a_t+eta F_t+w_t
x_(t+1)=x_t+p_(t+1), o_t=x_t+e_t.
~~~

Here rho=.45,beta=.9,eta=.7; w_t is independent process noise (distinct from the contrast vector w), SD .08. Position-sensor SD is declared per condition. This is a finite command-memory actuator drive with stable velocity filtering, not a claim that drag or contact passively generates this force. Finite action histories and stable filtering yield stationary velocity/displacements under stationary probes, with finite moments. Periodic probes yield a phase-dependent stationary regime. Raw position is not stationary. No privileged state, sensor-noise realization or counterfactual world is supplied to estimators.

Competing formulations and discriminators:

1. **Operating-regime sufficiency:** First-order responses at all scalar command biases and amplitudes suffice to reveal nonlinear actuation. Discriminator: two systems eta=0 and eta!=0 with identical summaries over that entire family.
2. **Temporal context geometry:** Constant biases move all history coordinates together; a mechanism depending on temporal contrasts can remain invisible. Discriminator: time-patterned means change first-order gains although constant means never do.
3. **Richer observable tests:** Keep centered probes but retain quadratic/Hermite response statistics. Discriminator: mixed/diagonal response kernels identify eta without changing probe context.
4. **Covariance experiment:** Preserve centered Gaussian commands but change temporal correlation. Discriminator: first-order blindness may persist while the output mean changes.

These alternatives concern the same mechanism. Their existence prevents a claim that either higher-order statistics or any particular mean pattern is universally necessary.

## F28. Response order is a derivative in context [EXISTING RESEARCH / DERIVED]

Let a finite command vector A~N(mu,sigma²I), let `E[Y|A]=f(A)` have polynomial growth, and let measurement error have conditional mean zero. Differentiating the Gaussian density gives

~~~text
M(mu,sigma)=E[Y]
H(mu,sigma)=Cov(A,Y)/sigma²=gradient_mu M.
~~~

For multi-index alpha, higher context derivatives are expectations of Y times the corresponding probabilists' Hermite polynomial of `(A-mu)/sigma`, divided by `sigma^|alpha|`. Distinct-index second moments use centered products; repeated indices require subtracting sigma² from the centered square. This is the same information family viewed as higher-order statistics at one context or first-order sensitivities across contexts. It does not say a low-dimensional set of contexts spans all necessary directions, nor provide stable finite-data inversion.

## F29. Exact nullspace theorem for quadratic conditional means [PROVEN]

Assume the declared finite-history family

~~~text
f(A)=c+B^T A+A^T Q A, Q symmetric, A in R^d.
~~~

Observe exact M and the **full d-component** H for every mu in a known linear subspace S of dimension k and all scalar variances sigma²>0. Then

~~~text
M=c+B^T mu+mu^T Q mu+sigma² trace(Q)
H=B+2Q mu.
~~~

H at zero identifies B. Varying sigma at mu=0 identifies c and trace(Q). Varying mu through S identifies Q restricted to S, including cross components outside S. Two coefficient matrices therefore give the same retained information iff their difference Delta obeys

~~~text
Delta S=0, trace(Delta)=0.
~~~

Symmetry implies Delta is supported entirely on S-perp. With h=d-k, this is a symmetric h-by-h block with one trace constraint: nullity `h(h+1)/2-1` for h>=1, zero for h=0. Equivalently nullity is `max(0,h(h+1)/2-1)`. This proves both necessity and sufficiency for this summary-identification problem: all coefficients are identified iff codimension(S)<=1. For only common command means, S=span(ones); at d=3,6,12 the invisible quadratic dimensions are respectively 2,14,65.

If only first-order responses are retained without output means or variance changes, the trace constraint is absent and the nullspace is larger. With known c or freely scalable covariance families, more general covariances constrain the remaining block through `trace(Delta Sigma)`. Full identification requires the projected covariance matrices on S-perp to span its symmetric-matrix space. With unknown c and a fixed non-scalable covariance family, use covariance **differences** relative to a reference, because c absorbs a shared trace offset. Do not silently assume c known.

**Precise scope:** This is a population theorem for quadratic finite-history conditional means, known Gaussian intervention distributions, exogenous measurement noise, complete response components, and arbitrary exact exploration over the stated family. It does not identify unspecified latent nonlinear dynamics, prove finite-data confidence, or imply that full raw observations are uninformative. The counterexample's nonidentifiability survives stable output filtering; the theorem's sufficient condition is not automatically a sufficient condition for identifying any unknown filter or hidden system.

Independent random-subspace numerical rank checks at d=3,6,12 and every k (24 cases) exactly match the formula. They validate the implementation of the constraint map, not the proof by themselves.

## F30. A stable actuator-memory counterexample [PROVEN]

The displayed Q satisfies `Q*ones=0` and `trace(Q)=0`. Under every common iid Gaussian command distribution `a_t~N(mu,sigma²)`,

~~~text
E[F_t]=0
Cov(F_t,a_s)=0 for EVERY s
H_l=beta rho^l for eta=0 and eta!=0, at every mu and sigma.
~~~

For the covariance claim, subtract the shared mean from each command: Q annihilates that mean vector and F is quadratic in centered Gaussian variables. Multiplying by any centered action gives an odd joint Gaussian moment, including actions in overlapping windows. Stable filtering preserves zero covariance. Independent process and sensor errors contribute zero. Thus all output means and centered first-order footprints match over this entire scalar-mean/common-amplitude family. Distinct higher moments and full trajectory distributions still distinguish the worlds.

Window overlap changes output variance: `Var(F_t)=sigma^4`, `Cov(F_t,F_(t-1))=-sigma^4/6`, and covariances at all further lags vanish. Hence the omitted interaction variance of any affine predictor using the full command history is

~~~text
eta² sigma^4 (1-rho/3)/(1-rho²).
~~~

At sigma=.8 the noiseless RMSE floor is .46251107. This formula differs from v5's white bilinear driving term; copying the old formula would be wrong. It excludes predictors using output histories or nonlinear features.

## F31. Three constructive repairs [PROVEN under the stated model]

**Timed means:** Use period-three means `(h,-h,0)` and independent Gaussian innovations. Within phase k, the current innovation is independent of prior output, and the immediate footprint is

~~~text
H0_phase=(beta+2eta h/sqrt(3), beta-eta h/sqrt(3), beta-eta h/sqrt(3))
eta=[H0_0-(H0_1+H0_2)/2]/(sqrt(3)*h).
~~~

Phase centering and phase-specific intercepts matter: pooling raw phase means can produce deterministic confounding. At h=sqrt(.48), predicted gains are exactly (1.46,.62,.62). This is first-order covariance with a time/context label, not a stochastic product of action tags.

**Second-order centered tag:** At zero mean,

~~~text
E[Y_t(a_t²-sigma²)]=2 eta Q00 sigma^4.
~~~

Current action does not enter earlier output, so inertia creates no extra term. This is an identifying diagonal Hermite moment, although it is statistically inefficient relative to joint regression.

**Correlated Gaussian input:** For centered stationary AR(1) commands with correlation r_a and fixed marginal variance sigma²,

~~~text
E[F_t]=2 sigma² (r_a-r_a²)/sqrt(12)
Delta E[p]=2 eta sigma² (r_a-r_a²)/(sqrt(12)*(1-rho)).
~~~

First-order covariances still ignore the quadratic component because of global sign symmetry; the output mean now distinguishes it. This is another reason to say exactly which summaries are retained.

## E17. Common-mean/amplitude falsification

Six conditions mu=-.8,0,.8 crossed with sigma=.35,1.25. Each has 32 independent 4095-row trajectories (exactly1365 per phase) after 408 generated initial rows, with sensor SD .5. Seeds `20261020+800000+1000*j+i`, j=0..5,i=0..31. Worlds eta=0 and .7 share exogenous seed streams solely for paired simulation audit; no learner receives paired-world observations. Fit eight command lags plus intercept separately by phase, then average the first-order coefficients. Maximum absolute mean paired coefficient differences across each condition range .001013 to .006734. Mean-output differences range -.009611 to .005155. These finite numbers are consistent with the exact zero population result; no post-hoc threshold or significance guarantee is claimed. Full paired per-seed differences and MC SEs are in JSON.

## E18. Matched-energy repairs and a reversed preliminary conclusion

For each sensor SD .1,.5,1.5, run 64 independent seeds `20261020+900000+1000*j+i`. Every policy gets 8190 fit rows, exactly2730 per phase, plus the same 408 initial rows. Commands:

- Centered iid SD .8.
- Constant bias sqrt(.32), innovation SD sqrt(.32).
- Period-three means `(sqrt(.48),-sqrt(.48),0)`, innovation SD sqrt(.32).

All have expected command energy .64 per measured step. Realized sample energies differ; their actual averages are saved and stay close to .64. Gaussian commands are unbounded. Equal expected energy and time are not guarantees of equal state excursions or equal optimal information.

Every estimator receives noisy displacement and logged command histories only. Q's shape is supplied to the raw Hermite inversion, phase contrast and structured baseline; eta is unknown. No true rho/beta is used by any estimator. Estimators:

- Centered raw Hermite estimate from F31.
- Phased first-order estimate: phase-specific eight-lag OLS with intercept, then immediate-gain contrast.
- Complete degree-two Volterra OLS: eight linear and 36 quadratic history terms; phase data additionally uses phase-specific linear terms/intercepts. Eta comes from current-action times lag-two coefficient divided by `2Q02`. This supplies a quadratic dictionary, not discovered operators.
- Equally Q-informed structured FIR regression: **identical 17-column form on both designs**, three phase intercepts, eight centered linear action lags, and six Q evaluated on **raw** three-command windows. This preserves the known relation between phase sensitivities and quadratic coefficients. Current Q coefficient identifies eta; rho is not supplied. The finite Q window leaves a small tail and coefficients are not tied by powers of rho, so negligible truncation bias can remain. No tuning, oracle state or threshold is used. Output noise is colored by differencing but independent of all action features.

{{ESTIMATORS}}

Table entries are RMSE of eta across independent seeds; eta=.7. The elementary phased estimator beats the raw centered moment, but **this is not probe-design superiority**. A stronger centered regression beats that phased estimator at every noise level. Applying the same supplied-Q regression to both designs yields comparable errors and no consistent winner. The complete quadratic estimator also beats the raw moment. All estimated means, Monte Carlo SEs and median errors are retained. Different nonlinear priors are explicit; the matched structured comparator is the fairest design comparison here.

The raw-moment-versus-phase experiment preceded addition of the efficient baselines; these are exploratory corrections, not a preregistered method contest or independent confirmatory efficiency study. The structural theorem and mechanism formulas were derived before their tests. No general advantage over nonlinear system identification, kernels, recurrent models or optimal experiment designs is claimed. V5 already compared a flexible kernel predictor on the prior coupled system; this cycle tests exact identifiability and efficiency confounding, not a new competitive learner.

## E19. Covariance-only repair

64 independent 8190-row runs per AR coefficient0,.5,.75, marginal command SD .8, sensor SD .5. Seeds `20261020+1000000+1000*j+i`. Paired worlds share seed streams only to verify the analytic mean difference, not as an oracle for a learner.

{{COVARIANCE}}

Values are paired mean difference +/- MC SE over trajectories. This establishes the derived intervention effect, not a fully autonomous single-environment test of unknown stochastic disturbances. A policy-dependent unknown bias could confound such a test without the stated exogenous-noise assumptions.

## Audit, prior work and interpretation limits

`independent_theory` independently found the scalar-mean counterexample, audited the rank theorem, derived overlap covariances and phase formulas, and checked all24 numerical ranks through a separate random-subspace construction. Crucially, it refuted our preliminary efficiency interpretation by constructing the efficient equally informed baseline. `v5_adversarial_audit` independently reran the finalized artifacts, checked phases, energy, centered/raw feature handling, formulas and saved summaries. Detailed evidence is retained in `research/audit_v6.md`; the main agent integrated corrections and owns final publication.

**EXISTING RESEARCH:** Gaussian-score derivative features and higher-order moments are established; see [Janzamin, Sedghi and Anandkumar, Score Function Features for Discriminative Learning](https://arxiv.org/abs/1412.6514). [Schoukens, Vaes and Pintelon, Linear System Identification in a Nonlinear Setting](https://arxiv.org/abs/1804.09587) studies input-dependent best linear approximations and nonlinear distortions. [Schoukens and Ljung, Nonlinear System Identification: A User-Oriented Roadmap](https://arxiv.org/abs/1902.00683) provides the broader established design/model/validation framework. [Regularized Volterra identification](https://arxiv.org/abs/1804.10435) is a stronger structured alternative to raw moment fitting. Our rank proof is independently derived elementary Gaussian moment algebra; no priority or novelty is claimed. Its value here is an exact specification of what this research program's proposed summaries can and cannot identify.

No conclusion transfers automatically to arbitrary nonlinear sensors, feedback-dependent interventions, unknown input noise, contact switches, unbounded hidden memory or articulated robotics. The tests are simple because they already falsify the overbroad sufficiency hypothesis. A learner still needs to choose a model/test vocabulary and intervention family; the record does not show autonomous discovery of Q or a universally correct repair.

## Reproduction and precise continuation

Dependencies: existing `.venv`, Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No package installed or external service changed.

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/context_geometry_v6.py > /tmp/context_geometry_v6_check.json
cmp research/context_geometry_v6_results.json /tmp/context_geometry_v6_check.json
.venv/bin/python research/context_rank_v6.py > /tmp/context_rank_v6_check.json
cmp research/context_rank_v6_results.json /tmp/context_rank_v6_check.json
~~~

**Next consequential question:** Can a learner detect and repair response-summary blindness when the nonlinear form Q is not supplied and the same plant must remain under noisy feedback? Begin with the v5 stable two-actuator model, replace the linear gain q by `tanh(q)` or a unilateral smooth contact gain, and superimpose logged independent perturbations on a fixed known stabilizing controller. Derive which immediate and delayed moments still describe the plant versus controller, before choosing a learner. Compare classical instrumental/closed-loop state-space prediction, full quadratic response regression, and an expressive nonlinear predictor with the same trajectories and fixed intervention energy. Validate on unseen mean/time-pattern/correlation schedules and changed tracking goals; separate parameter identification, counterfactual sequence prediction and actual control. The current rank theorem does not authorize extrapolating bilinear parameter ratios to a new functional class.

Keep the overall Goal active: the audited rank limitation changes the design of this candidate learner but remains a class-specific result within established mathematics. A broader independently verified advance in learning mechanisms under progressively nonlinear robotics-relevant conditions is still outstanding.

---

# Historical archive: Complete original v5 ledger, including all earlier ledgers

'''
rows=['| Sensor SD | Raw centered order2 | Phased order1 | Full Volterra centered / phased | Matched Q-FIR centered / phased |',
      '| --- | --- | --- | --- | --- |']
for sd,d in r['equal_energy'].items():
    e=lambda k:f"{d[k]['rmse_from_eta']:.6f}"
    rows.append(f"| {sd} | {e('centered_eta2')} | {e('phased_eta1')} | {e('centered_volterra2')} / {e('phased_volterra2')} | {e('centered_structured')} / {e('phased_structured')} |")
doc=doc.replace('{{ESTIMATORS}}','\n'.join(rows))
rows=['| Command AR coefficient | Analytic difference | Measured difference +/- MC SE |','| --- | --- | --- |']
for d in r['covariance_change'].values():
    rows.append(f"| {d['corr']} | {d['theoretical_mean_difference']:.6f} | {d['mean_difference']:.6f} +/- {d['mc_se']:.6f} |")
doc=doc.replace('{{COVARIANCE}}','\n'.join(rows))
out=root/'first_principles_robot_learning_ledger_v6.md'
if out.exists():raise SystemExit('Refusing to overwrite released v6')
assert '{{' not in doc
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
