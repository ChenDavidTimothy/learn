#!/usr/bin/env python3
"""Build v7 exactly once from audited final results and preserved v6."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'feedback_virtual_zero_v7_results.json').read_text())
prior=(root/'first_principles_robot_learning_ledger_v6.md').read_text()
doc=r'''# First-Principles Robot Learning: Living Research Ledger v7

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited conditional-intervention mechanism, precise impossibility result and coupled nonlinear feedback experiments. **Code/results:** `research/feedback_virtual_zero_v7.py`, `research/feedback_virtual_zero_v7_results.json`. Full v6 and all earlier ledgers follow verbatim. V6 was published and remotely verified on SSH origin/main at `1444da20638391b852a644615deeabcf94705c50` before this investigation began.

## Outcome and the standard of the claim

**PROVEN under assumptions:** One can identify a constant passive recurrence coefficient from noisy position under feedback, without reconstructing an arbitrary hidden, state-dependent actuator gain. The necessary construction is a conditional regression-intercept weight followed by an instrumental moment. For polynomial actuation, a finite Hermite reproducing kernel cancels the unknown actuator coefficients. The kernel is classical mathematics, not a new invention.

**PROVEN limitation:** Under a continuous Gaussian probe distribution, no fixed integrable context/input weight can preserve the passive term while exactly canceling every bounded continuous actuator function that vanishes at zero. Such a weight would have to transform an absolutely continuous measure into a point mass. This is a limitation of exact universal weighting, **not** an impossibility of learning from the full data or of consistent nonparametric estimation with regularity assumptions.

**CONSTRUCTIVE / TESTED:** Give the action distribution an actual randomized atom at zero. The ordinary inverse-probability weight on this atom cancels arbitrary zero-at-zero actuation, including saturation. A policy can do this while matching the Gaussian policy's conditional action mean and variance. This changes intervention support rather than increasing representation order.

**TESTED negative:** More cancellation order is not universally better: finite polynomial weights can retain saturation bias and have higher variance; even the correct exact moment can be less precise than ordinary series IV. Small probes and strong sensor noise produce large errors. No superiority over modern nonlinear identification or full model learning is established.

This is a useful class-specific mechanism and a consequential restriction on exact response isolation. It does not recover the full gain law, learn a complete state representation, solve nonlinear passive drift, demonstrate control transfer, or complete the overall research Goal.

## Starting question, assumptions and competing formulations

V6 leaves open how an agent can isolate plant structure when it cannot simply treat feedback commands as independent probes. Begin with a scalar passive coefficient but allow an unknown nonlinear actuator with hidden coupled history:

~~~text
p_(t+1)=rho p_t+F_t(u_t)+w_t
x_(t+1)=x_t+p_(t+1), o_t=x_t+e_t
D_t=o_t-o_(t-1), Y_t=D_(t+1)
u_t=pi_t+z_t.
~~~

The robot logs measured positions, issued commands, the pre-probe controller command pi, and the randomization. F_t may depend arbitrarily on pre-probe hidden state, earlier commands and noise. It must be determined before the current probe; current-probe-dependent actuator state is not included. Assume `F_t(0)=0`, constant rho, additive exogenous process/sensor errors with the stated moments, and sufficient finite product moments. Baseline Gaussian z is centered, variance sigma²>0 and independent of the pre-probe history. The controller can be nonlinear/time-varying and depend on noisy observations, but must select pi before seeing z.

Position differencing gives the exact equation

~~~text
Y_t=rho D_t+F_t(u_t)+epsilon_t
epsilon_t=w_t+e_(t+1)-(1+rho)e_t+rho e_(t-1).
~~~

A valid previous-probe instrument V_t=z_(t-1) has `E[V_t epsilon_t]=0` under exogenous iid sensor/process errors and sequentially centered probes. It must also have `E[V_t D_t]!=0`. Weak relevance must not be mistaken for a detected absence of physical memory.

Competing framings:

1. **State/gain first:** Reconstruct the hidden gain and fit it jointly with rho using an IV or state-space model. This may be statistically efficient. However finite feature approximation or an erroneous generated state can contaminate the rho moment even after sample splitting.
2. **Isolate passive response:** Construct a current-probe weight that annihilates the actuator term conditionally on all pre-probe history, leaving rho estimable without a gain model.
3. **Change intervention support:** If the actuator cannot be canceled by the chosen finite vocabulary, observe the plant at an actual zero command with known random probability instead of pretending a finite response expansion is universal.

Discriminators: arbitrary hidden gain changes should leave the passive-moment identity intact; quadratic actuation should break an affine-only weight; saturation should reveal finite polynomial limitations; an actual zero-action atom should remove those actuator-class biases while showing its variance cost.

## F32. Conditional intercept weighting removes unknown gain [PROVEN]

Suppose first `F_t(u)=G_t u`, with G_t any pre-probe measurable gain. Define

~~~text
w1_t=1-pi_t*z_t/sigma².
~~~

Condition on the pre-probe history, including G_t,D_t,pi_t,V_t. Then `E[w1|history]=1` and

~~~text
E[w1*(pi+z)|history]=pi-pi*E[z²]/sigma²=0.
~~~

Current randomization is independent of the additive error conditional on its pre-probe and fresh exogenous noise components, so `E[V w1 epsilon]=E[V epsilon]=0`. Therefore

~~~text
rho = E[V_t w1_t Y_t] / E[V_t D_t].
~~~

The denominator uses `E[Vw1D]=E[VD]`; its unweighted form avoids unnecessary sampling variance. No Gaussianity is needed for this affine weight: centered probes with known conditional variance and appropriate exogeneity suffice. G_t may depend on previous z, true state, unobserved configuration or the feedback trajectory. It need not be independent of the entire probe sequence. This was independently derived by the main agent and auditor, with the conditional cancellation removing an unnecessary stronger independence assumption in an earlier product-moment argument.

The conditional squared weight is `1+pi²/sigma²`. Small randomized probes relative to the feedback command amplify noise. The sample ratio is generally biased at finite n; the theorem is a population identity. Consistency additionally needs relevance, averaging/mixing assumptions and sufficient moments.

## F33. Polynomial actuator classes, minimum-norm weights [PROVEN / EXISTING RESEARCH]

For Gaussian `z=sigma Z`, Z~N(0,1), let He_k be probabilists' Hermite polynomials. Define

~~~text
w_m(Z;pi)=sum_(k=0)^m He_k(-pi/sigma)*He_k(Z)/k!.
~~~

Every polynomial h of degree<=m has a finite Hermite expansion in Z. Orthogonality reproduces its value at `Z=-pi/sigma`, hence

~~~text
E[w_m*h(pi+sigma Z)|history]=h(0).
~~~

Thus the rho ratio from F32 remains valid for arbitrary history-dependent coefficients in `F_t(u)=sum_(j=1)^m G_(t,j)u^j`. A nonzero actuator constant term is not removed. For m2 the weight is

~~~text
1-pi*z/sigma² + .5*(pi²/sigma²-1)*(z²/sigma²-1).
~~~

Its conditional norm is

~~~text
E[w_m²|history]=sum_(k=0)^m He_k(-pi/sigma)^2/k!.
~~~

Any other square-integrable weight satisfying the same polynomial reproduction constraints has these same first m Hermite coefficients and can only add an orthogonal component. Therefore w_m has minimum conditional L2 norm for these constraints. This is exactly a classical Christoffel–Darboux reproducing kernel, **not a new kernel construction**. The statement is not optimality of the full rho estimator: serial dependence, instrument choice, noise covariance and denominator estimation also affect its variance.

For a quadratic actuator, an affine weight leaves `G_(t,2)*(sigma²-pi²)` conditionally. More finite order is not monotonically better for a genuinely non-polynomial actuator. A wrong actuator class is an alternative explanation for apparent passive-dynamics changes.

## F34. Continuous probes cannot provide exact universal point isolation [PROVEN]

Fix a context and a continuous Gaussian distribution P for U. Suppose an input/context-only integrable signed weight w satisfied `E_P[w]=1` and `E_P[w f(U)]=0` for every bounded continuous f with f(0)=0. Choose bounded continuous f_n, zero at zero, between0 and1, converging to1 at every nonzero U. Because P(U=0)=0, dominated convergence with |w| gives

~~~text
0=lim E[w f_n(U)]=E[w]=1,
~~~

a contradiction. Equivalently w dP is absolutely continuous and cannot equal a Dirac mass at0. Thus no such L1 weight exists; a finite-variance weight would in particular be L1 and is also impossible. The theorem concerns exact fixed universal weighting, not arbitrary procedures, complete-data identifiability, or asymptotic localized estimation. With continuity/smoothness assumptions, shrinking neighborhoods or nonlinear function learning can estimate zero-input response with a bias/variance trade-off. The theorem does not prohibit those.

## F35. Actual randomized zero input changes the information [PROVEN]

Choose U=0 with a known positive probability p conditional on every pre-probe history. Then `w_atom=1{chosen zero branch}/p` has conditional mean1, squared norm1/p, and `E[w_atom F_t(U)|history]=F_t(0)=0` for every zero-at-zero actuator. The same IV ratio identifies rho whenever its instrument is relevant and additive-error moments hold.

The randomization may change the action law. To match the Gaussian policy's conditional mean pi and variance sigma², use p at0 and, otherwise,

~~~text
U ~ Normal(pi/(1-p), sigma²/(1-p)-p*pi²/(1-p)^2).
~~~

The component variance must be positive. For p=.1, |pi|<=.5 and sigma>=.2, its minimum is .01358025. Direct expansion proves `E[U|history]=pi` and `Var(U|history)=sigma²`. The resulting innovation `z=U-pi` is conditionally centered, not iid independent of history. That is sufficient: previous sensor error is already in the previous pre-probe history, and future sensor/process noise is exogenous, so `E[zprev epsilon]=0` still holds. Gaussian Hermite weights are not assumed valid for this atom law.

Conditional moment matching does not equate the complete action distribution, state trajectories or unconditional action energy. All realized energies are saved. Neither all nonlinear systems nor all actuator functions guarantee instrument relevance. If rho varies with state, the ratio need not be a universal physical coefficient; if unknown additive passive forcing remains beyond rho*p, zero-input isolation does not identify it as a constant rho.

## E20. Executed coupled nonlinear feedback worlds

All calculations used the owner's existing local `.venv`, Python3.12.3 and NumPy2.5.3 plus threadpoolctl; no package installation. The source computes measured displacements directly, algebraically identical to differencing noisy integrated positions. Only D,Y,issued u,probe z,controller pi and secondary command b go to estimators; no true q,p,gain or actuator formula is supplied. The simulator is

~~~text
rho=.65, pi_t=-.5*tanh(D_t)
gain_t=1.2+.7*tanh(q_t)
q_(t+1)=.6*q_t+.3*tanh(p_(t+1))+b_t+qnoise_t
b_t~N(0,.6²), qnoise_t~N(0,.1²), w_t~N(0,.06²).
~~~

The semi-implicit q coupling makes the current gain depend on the preceding probe, explicitly testing the weaker theorem. It remains selected before the current probe. Bounded gains/controller and |rho|<1 give uniformly finite state moments under these finite-moment action laws. We do not infer a full mixing theorem solely from a finite burn-in.

Four cases, each48 independent seeds `20261020+1200000+1000*case_index+i`:

1. Affine actuator `gain*u`, sigma=.7,sensorSD=.5.
2. Quadratic actuator `gain*u+(.2+.15*tanh(q))*u²`, same noise/probe.
3. Saturation `gain*tanh(u)`, same noise/probe.
4. Affine with weak probes sigma=.2 and sensorSD1.5.

Gaussian and atom policies share exogenous random streams within each seed but generate different trajectories. Each gets65536 retained samples after604 discarded initial rows, with also an8192 prefix analysis. Prefixes are nested, not independent sample-size replications. Atom branch probability .1; conditional action mean/variance match as derived. All seeds are retained, including outliers.

Comparisons use identical observations within a policy: OLS on[D,u]; constant-gain conventional IV with instruments[zprev,z]; conventional series IV using u times ten quadratic features of observed (D,bprev,bprev2); richer action-polynomial series IV2/4 using u^j times those same history features and current Gaussian Hermite instrument features; reproducing weights m1,m2,m4; and randomized-atom IV. The action-polynomial series regressions have21 and41 coefficients; they cover richer actuator classes but still approximate hidden history. No true scheduling state is supplied. Methods are conventional finite dictionaries, not a claimed implementation of optimized neural or LPV identification. There is no tuned hyperparameter, fitted feature selection, oracle threshold or omitted bad seed. Degrees are reported separately; the learner is not given an automatic rule choosing the correct one.

Gaussian/atom comparisons are experiments with different support and matched conditional first two moments, not literally the same trajectories. Numerical weights use known designed sigma; mislogged controller/probe distributions would invalidate identities. All per-seed estimates, denominators, energies and squared weight norms are retained.

{{MAIN_TABLE}}

Each entry reports mean rho estimate / RMSE from true .65 at65536 samples. Full Monte Carlo SEs and10th/90th quantiles are in JSON. A finite-data estimate consistent with .65 is not proof of exact finite-sample unbiasedness. OLS exhibits sensor/model bias. Ordinary and richer series IV can be more precise than cancellation weights, particularly in the valid affine world and weak-probe condition. These negative comparisons matter. Saturation can leave fixed-degree weighted biases; m2 can be worse than m1, so increasing a response grammar blindly is not justified. Atom estimates retain higher sampling variance but avoid the actuator-class obstruction under the stated assumptions.

18 independent Gaussian quadrature cases (pi=-.5,0,.4; sigma=.2,.7; degree1,2,4) check every reproduction constraint through the selected degree. Maximum errors are below1e-10. They verify algebra/implementation, not the completeness of the actuator class.

## Independent audit and precedence

`independent_theory` independently derived the conditional cancellation and the Hermite extension, found the continuous-weight impossibility and zero-atom alternative, and refuted unnecessary q/probe independence. It also warned about finite-basis IV contamination, weight variance and state-dependent passive drift. `v5_adversarial_audit` independently reconstructed simulations, checked adaptive-probe exogeneity, quadrature and final results. Detailed numerical findings and final-source replication status are in `research/audit_v7.md`. The main agent owns integration and publication.

**EXISTING RESEARCH:** The weighting kernel is established polynomial reproduction; [Cichon and Szafraniec, Christoffel–Darboux Kernels in Several Real Variables](https://link.springer.com/article/10.1007/s00009-023-02464-5) discusses reproducing kernels. [D-optimal polynomial design via Christoffel functions](https://arxiv.org/abs/1703.01777) provides related design/variance theory. Closed-loop IV precedents include [Gilson and Van den Hof2005](https://doi.org/10.1016/j.automatica.2004.09.016) and [Toth et al., closed-loop LPV IV2012](https://doi.org/10.1016/j.automatica.2012.06.037). Those methods address scheduling/feedback identification beyond the particular elementary moment here. A relevant modern strong alternative is [Neural Identification of Feedback-Stabilized Nonlinear Systems2025](https://arxiv.org/abs/2503.22601); it has not been implemented in this experiment, so no advantage over it is claimed. All these precedents constrain novelty interpretations. The scientific milestone is a fully audited derivation of what this candidate mechanism can isolate, its exact limitation, and a reproducible intervention-support repair.

## Reproduction, limitations and exact next investigation

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/feedback_virtual_zero_v7.py > /tmp/feedback_virtual_zero_v7_check.json
cmp research/feedback_virtual_zero_v7_results.json /tmp/feedback_virtual_zero_v7_check.json
~~~

Only one constant passive coefficient is identified. No full nonlinear plant, new goal policy, long-horizon control performance, operator discovery or humanoid capability is demonstrated. Zero action must be physically meaningful and feasible; these simulations supply that intervention. Probe support, actuator vocabulary, additive error assumptions and nonzero relevance are built-ins/capabilities, not learned facts.

**Immediately next:** Replace rho*p by a globally stable nonlinear passive drift, initially the exact dissipative flow from v4. Randomized zero-action atoms still isolate the zero-input transition but no longer turn noisy displacement into noiseless state. Derive what is identifiable from noisy position histories, attempt a counterexample separating drift from sensor noise/state aliasing, and compare a correctly specified observation-model/state-space filter against naive drift regression and an expressive history predictor on equal data. Keep a separate exact constant-rho reproduction as a regression check. The question becomes learning nonlinear passive dynamics after actuator nuisance cancellation, rather than relabeling a weighted average as a physical parameter.

The overall Goal remains active. This release is not a stopping condition; it establishes an actionable mechanism/limitation under declared assumptions while broader autonomous learning and robotics transfer remain unproven.

---

# Historical archive: Complete original v6 ledger, including all earlier versions

'''
cases=list(r['experiments'].values())
rows=['| Method | Affine | Quadratic | Saturating | Weak probe / noisy sensor |',
      '| --- | --- | --- | --- | --- |']
for method in ('OLS','constant_gain_IV','series_IV','action_series_IV2','action_series_IV4',
               'weight_m1','weight_m2','weight_m4','atom'):
    cells=[]
    for d in cases:
        x=d['summary']['65536'][method]
        cells.append(f"{x['mean']:.6f} / {x['rmse_from_rho']:.6f}")
    rows.append('| '+method+' | '+' | '.join(cells)+' |')
doc=doc.replace('{{MAIN_TABLE}}','\n'.join(rows))
out=root/'first_principles_robot_learning_ledger_v7.md'
if out.exists():raise SystemExit('Refusing to overwrite released v7')
assert '{{' not in doc
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
