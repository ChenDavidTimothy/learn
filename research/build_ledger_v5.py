#!/usr/bin/env python3
"""Build v5 once from checked data, embedding the full v4 verbatim."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'coupled_response_v5_results.json').read_text())
prior=(root/'first_principles_robot_learning_ledger_v4.md').read_text()
doc=r'''# First-Principles Robot Learning: Living Research Ledger v5

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited representation limitation, exact family identification, noisy coupled trajectory tests. **Code/results:** `research/coupled_response_v5.py`, `research/coupled_response_v5_results.json`. Full v4/v3/v2/v1 follows unchanged. V4 was published over existing SSH origin/main at `d1b5895e129314fae9fe16882e3aa90af059a10f`, and remote equality was verified before this investigation began.

## Result and narrow interpretation

**PROVEN under assumptions:** All centered first-order command-response moments are blind to an even action-history component under a globally sign-symmetric intervention distribution. Arbitrarily many amplitude changes do not fix this. A stable two-actuator bilinear system has exactly the same first-order footprints with or without a pose-dependent actuation term, yet transfers differently when commands acquire a mean. This is a limitation of the statistic class plus probe family, not an impossibility of learning from the full logged data.

**PROVEN within the supplied family / TESTED:** Three mixed second-order kernels expose the coupling, the otherwise invisible persistence of a hidden coordinate, and the velocity persistence. A classical Volterra model and a generic nonlinear kernel predictor learn useful response structure from exactly the same noisy-position/action trajectories on which a linear footprint model fails. Neither is a new algorithm.

**Important counterweight:** First-order responses at deliberately biased operating conditions can also identify this bilinear family. Thus higher-order statistics are one repair, not universally necessary. Representation and experiment choice must be considered jointly.

This rules out a serious overreading of v3: first-order footprints plus amplitude adaptation are not sufficient foundations for unrestricted nonlinear transfer. It does not establish a uniquely minimal learner, autonomous representation discovery, humanoid intelligence or completion of the overall research Goal.

## First-principles question and competing formulations

What minimal information distinguishes a hidden actuator interaction from dynamics that have the same linear responses? Consider a cart-like displacement driven by a thrust command u and an unobserved gain coordinate q controlled by a second command b. This is a small interpretable model of configuration-dependent actuation, not a full rigid-body or contact simulator.

~~~text
q_(t+1) = r q_t + b_t
p_(t+1) = rho p_t + beta u_t + eta q_t u_t + w_t
x_(t+1) = x_t + p_(t+1)
o_t = x_t + e_t.
~~~

All units and timestep are normalized. Assume |r|,|rho|<1, additive independent centered Gaussian w,e, and commands logged exactly. The agent receives noisy positions and both commands; no q or p, velocity sensor, state reset or true parameter is supplied to fitting. The experimenter knows the simulator and uses noiseless output only for evaluation. The q update precedes its use at the next step, so b_t first influences the interaction at time t+1.

Competing framings:

1. **First-order sufficiency hypothesis:** Rich randomized commands and varying amplitude make linear response propagation sufficient for transfer. Discriminator: two systems with identical population footprints should behave identically for the target transfer.
2. **Conditional physical gain:** q changes the effect of u. The state has no independent additive b-to-output path, so one must inspect joint interventions or change the operating mean. Discriminator: mixed command-response moments and shifted-mean responses change with eta although every centered first-order footprint stays fixed.
3. **Irreducible-noise explanation:** Extra residual variance might simply be physical randomness. Its magnitude alone cannot decide; correlation with a mixed intervention tag distinguishes our interaction under the independence assumptions. This experiment does not solve arbitrary unknown noise mechanisms.

No paradigm is selected by name until the distinguishing information is derived below.

## F24. Symmetry creates a first-order information nullspace [PROVEN]

Let A denote the entire relevant command history with `A equal-in-law to -A`. Let Q be any integrable even causal functional, `Q(-A)=Q(A)`, with adequate product moments. Every coordinate A_j has mean zero, and

~~~text
E[Q(A) A_j] = E[Q(-A)(-A_j)] = -E[Q(A) A_j] = 0.
~~~

Thus `Cov(Q,A_j)=0`, even when Q has a nonzero mean. Every centered jointly Gaussian command law has this symmetry, including arbitrary covariance structure and variances. Changing amplitude while preserving symmetry cannot recover Q from first-order moments. If output is a linear response plus Q plus independent noise, its first-order action covariances match the system without Q.

In our model q is linear in b and the eta-dependent contribution to p is quadratic in (u,b), hence even under simultaneous sign reversal. Additive independent process noise has no command covariance. Under white mutually independent zero-mean channels,

~~~text
H_u(i)=Cov(p_(t+1),u_(t-i))/sigma_u^2=beta rho^i
H_b(j)=0,  for all i,j>=0, independently of eta and r.
~~~

The same holds for measured displacement; sensor differences add no cross moment. For colored or cross-correlated Gaussian commands the formulas change because tags are correlated, but equality between eta=0 and eta!=0 still holds for centered first-order covariances. Means or other output statistics can expose parts of Q; the theorem is specifically about first-order action covariance information. Feedback selected from observations may violate the declared command symmetry/exogeneity and is outside this proof.

## F25. Mixed responses reveal a hidden mode [PROVEN within family]

Under stationary independent white commands,

~~~text
q_t = sum_(m>=0) r^m b_(t-1-m)
p_(t+1) = sum_(i>=0) rho^i [beta u_(t-i)
           + eta u_(t-i) sum_(m>=0) r^m b_(t-i-1-m) + w_(t-i)].
~~~

The sums converge in every finite L^k: q is a stable Gaussian linear filter, the innovation containing q*u has finite moments, and the rho filter is absolutely summable. This gives a unique stationary law for q,p and displacement, not for integrated x. The velocity distribution is generally non-Gaussian.

Multiply the measured displacement by `u_(t-i)b_(t-j)`. Independence eliminates every term except the matching two commands:

~~~text
K_ij = E[(o_(t+1)-o_t)u_(t-i)b_(t-j)]/(sigma_u^2 sigma_b^2)
     = eta rho^i r^(j-i-1) if j>=i+1; otherwise 0.
~~~

Consequently, for eta!=0,

~~~text
beta=H_u(0), eta=K_01, r=K_02/K_01, rho=K_12/K_01.
~~~

This identifies coefficients in this *declared bilinear family*, with the b-to-q gain fixed to one, not a unique physical coordinate or the correct family autonomously. If that gain were unknown, only its product with eta would be identifiable because q can be rescaled. Ratios become weakly identified near eta=0; no finite-data certificate of absence follows. Full trajectory distributions, output variance and sampling fluctuations can retain information absent from population first-order summaries. The physical q mode can be entirely absent from first-order responses while visible in mixed responses. This extends v3's cancellation warning: invisibility can be statistical-order blindness, not physical uncontrollability.

For affine predictors of the **entire command history**, the quadratic part is orthogonal to every regressor and has variance

~~~text
C = eta^2 sigma_u^2 sigma_b^2 / [(1-r^2)(1-rho^2)].
~~~

Indeed `q_t u_t` is temporally uncorrelated under independent white u, and its variance is `sigma_u^2 sigma_b^2/(1-r^2)`; filtering by rho supplies the other factor. Therefore this class has noiseless-output RMSE at least sqrt(C). At SD=.8 for both commands, sqrt(C)=0.57927133. This is **not** a lower bound for all linear ARX predictors using past outputs, arbitrary nonlinear predictors, or all possible learners.

For L chronological command lags, omitted quadratic variance equals C minus

~~~text
eta^2 sigma_u^2 sigma_b^2 *
 sum_(i=0)^(L-2) rho^(2i) sum_(m=0)^(L-i-2) r^(2m).
~~~

At L=8 the omitted coefficient is .0001733013 times `sigma_u^2 sigma_b^2`, about .0212% of interaction energy. Linear-history truncation also remains. It cannot explain the much larger linear-model failure.

## F26. Experiment choice can replace response order [PROVEN, next test]

For means mu_u,mu_b, with otherwise independent iid channels,

~~~text
E[q]=mu_b/(1-r)
E[p]=[beta mu_u + eta mu_u mu_b/(1-r)]/(1-rho).
~~~

With mu_u=.4,mu_b=.5, true mean p is 1.16363636; the centered-data first-order model predicts .65454545, a systematic gap .50909091.

But re-identifying first-order responses at biased means can repair the missing mechanism:

~~~text
H_u(i)=[beta+eta mu_b/(1-r)]rho^i
H_b(j)=eta mu_u sum_(i=0)^(j-1)rho^i r^(j-1-i), j>=1.
~~~

At mu_u!=0,mu_b=0, beta and rho come from H_u; eta=H_b(1)/mu_u and r=H_b(2)/H_b(1)-rho. Thus biased first-order probes and centered mixed probes are genuine competing solutions. The next investigation must compare them at matched total time and actuation energy, rather than declaring a particular moment order essential.

## F27. No fixed response order certifies unrestricted sufficiency [PROVEN, limited scope]

For any finite m, choose m+1 distinct delayed iid centered command coordinates and add a finite-memory response proportional to their product. Its mean is zero, and its cross moment with any input monomial of total degree <=m is zero: at least one of the m+1 independent centered factors remains unpaired. This holds for all independent centered amplitudes with finite required moments. Yet command sequences with those coordinates fixed nonzero produce a different output.

Therefore even all input-output polynomial moments up to fixed finite order cannot certify transfer over unrestricted finite-memory nonlinear functions. This is a limitation of truncated response summaries, not of full predictive distributions, non-polynomial tests, changed probe means or learners with additional structural assumptions. It is elementary orthogonal-expansion mathematics; it does not imply that robots need arbitrarily high interactions in practice.

## E15. Noise, amplitude replication and identification

Coefficients r=.5,rho=.45,beta=.9,eta=.7. Process SD .08, position sensor SD .5. Trajectories are continuous within each run, with 400 burn steps; no mid-trajectory reset. For two amplitudes .35 and 1.25, 64 independent 8192-row runs compare eta=0 and .7 using common random numbers within pairs. Seeds `20261020+500000+1000*j+i`, j=0,1, i=0..63. Fit intercept plus eight lags of both commands. Raw mixed moments use known commanded variances, centered measured output, and the exact kernel normalization. No threshold is tuned and no seed is discarded. All per-seed outputs are in JSON.

{{MOMENTS}}

The true K01,K02,K12 are .7,.35,.315. Larger amplitude improves finite-sample precision without making the population first-order footprint informative about eta. The maximum absolute mean paired first-order coefficient difference is .000587 at SD .35 and .003126 at SD1.25. These finite estimates do not prove equality; the symmetry theorem does. The L2 metric uses parameter order beta,eta,r,rho and fixed normalized units. It is not a universal parameter metric.

## E16. Fair predictive comparison and transfer failures

32 independent training seeds `20261020+600000+100*i`, i=0..31. Each model receives the exact same 1024-row training trajectory, separate 512-row validation trajectory (base+1), and four fresh 2048-row tests (base+10..13). Train/validation command SD .8, zero means, white independent channels. Test conditions: iid matched; means (.4,.5); temporal AR(1) correlation .75 within each channel with same marginal SD .8 and independent channels; and amplitude1.25. Hyperparameters use observed validation displacements, never test or latent data.

Models: FIR ridge on 16 command-history coordinates; quadratic Volterra ridge on all 16 linear and 136 degree-two monomials; exact RBF kernel ridge on the same 16 coordinates. Each includes an intercept. Polynomial features and RBF are **supplied nonlinear vocabularies**, not discovered operators. Ridge grids are [.001,.1,10,100] on summed squared loss; RBF gamma is [.25,1,4]/16. Feature scaling/centering uses training data only. RBF has 12 validation configurations, versus four for each polynomial degree; compute is disclosed, not matched. Eigensolves are exact and one BLAS thread is used. These are established nonlinear competitors, not implementations of the specialized multidimensional GP priors in cited modern Volterra work. No claim of superiority over those methods or recurrent/state-space alternatives is made.

Primary score compares predictions to noiseless velocity generated by the same test commands, with independent process noise removed for evaluation only. Because models use only command history as predictors, this is their conditional-mean target up to history truncation. Actual measured-output RMSE and signed mean errors are also saved. Models could use past measurements in a broader experiment; this restriction is intentional for testing the response-summary claim and limits conclusions.

{{PREDICTION}}

Values are mean noiseless-output RMSE +/- Monte Carlo SE across 32 seeds. FIR's matched RMSE .582922 is close to the proven .579271 population floor. Its mean-shift signed bias is -.525123, close to -.509091. Nonlinear models improve all four tested conditions; the quadratic dictionary matches this simulator's operator class. Both nonlinear models still suffer substantial amplitude transfer error: Volterra .683538, RBF .913199. A finite noisy dataset and supplied finite vocabulary do not solve general transfer. RBF matches quadratic performance closely on iid data, an explicit counterweight to any special superiority narrative.

## Audit, literature and novelty assessment

The `independent_theory` auditor derived the kernels, lower bound, finite-history error and biased-probe counterexample independently. It caught an overbroad possible interpretation of the lower bound (past-output ARX predictors are outside the bound) and insisted on the biased-mean repair. A second `v5_adversarial_audit` auditor inspected and independently reproduced code/results; full findings are retained in `research/audit_v5.md`. No auditor wrote canonical artifacts. The main agent checked reported tables, saved seeds and full history preservation before publishing.

**EXISTING RESEARCH:** [Schoukens et al., Identification of linear systems with nonlinear distortions (2005)](https://www.mit.bme.hu/system/files/oktatas/targyak/9132/Identification_of_linear_systems_with_no.pdf) explicitly warns that nonlinear systems can pass linear correlation and residual-whiteness tests. [Schoukens and Tiels' survey](https://arxiv.org/abs/1607.01217) explains how best linear approximations guide nonlinear identification while remaining input-dependent. The mixed kernels here are classical Wiener/Volterra information. [Birpoutsoukis et al., Regularized Nonparametric Volterra Kernel Estimation (2017; arXiv 2018)](https://arxiv.org/abs/1804.10435) supplies more structured regularization than our isotropic ridge. [Dalla Libera et al., Kernel-based methods for Volterra series identification (2021)](https://www.research.unipd.it/retrieve/3384812c-4a68-4754-a539-298d6a1c6a10/Automatica_Dalla%20Libera_Carli_Pillonetto.pdf) addresses expressive kernels and scaling. These precedents substantially limit originality; our contribution is a precise falsification and reproducible correction of the current research framing, not a new theory of nonlinear identification.

## Reproduction and exact next question

Python 3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl in existing `.venv`; no packages installed. No GPU. All calculations on owner's PC. Reproduce without overwriting releases:

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/coupled_response_v5.py > /tmp/coupled_response_v5_check.json
cmp research/coupled_response_v5_results.json /tmp/coupled_response_v5_check.json
~~~

**Immediately next:** Compare centered order-two moments against biased order-one experiments, with exactly equal total interaction steps and expected command energy. Include a conventional full quadratic ridge estimator on each dataset and the same family assumptions for parametric inversion. Determine which noise/actuator regimes favor experimental mean shifts, which favor richer test functions, and whether either repair reliably reveals a weak hidden mode. Then test failure under unknown saturating configuration gain and noisy closed-loop feedback; do not extrapolate bilinear-family ratios to those environments.

**Still open:** Autonomous choice between experiment change, statistic expansion and state construction; finite-data confidence for declaring missing structure; transfer under unseen contacts/goals; nonlinear observation maps; real articulated hardware. Keep state-first, response-first and goal-first foundations alive. This milestone is too close to established mathematics to justify declaring the persistent foundational objective achieved.

---

# Historical archive: Complete original v4 ledger, including all earlier ledgers

'''
rows=['| SD | Mean K01 | Mean K02 | Mean K12 | Median parameter L2 |', '| --- | --- | --- | --- | --- |']
for a,d in r['moments'].items():
    rows.append(f"| {a} | {d['k01']['mean']:.6f} | {d['k02']['mean']:.6f} | {d['k12']['mean']:.6f} | {d['parameter_l2_error']['median']:.6f} |")
doc=doc.replace('{{MOMENTS}}','\n'.join(rows))
rows=['| Model | iid | mean shift | correlated | amplitude shift |','| --- | --- | --- | --- | --- |']
for model,d in r['prediction']['summary'].items():
    cells=[f"{d[c]['noiseless_rmse']['mean']:.6f} +/- {d[c]['noiseless_rmse']['mc_se']:.6f}" for c in ('iid','mean_shift','correlated','amplitude_shift')]
    rows.append('| '+model+' | '+' | '.join(cells)+' |')
doc=doc.replace('{{PREDICTION}}','\n'.join(rows))
out=root/'first_principles_robot_learning_ledger_v5.md'
if out.exists(): raise SystemExit('Refusing to overwrite released v5')
assert '{{' not in doc
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
