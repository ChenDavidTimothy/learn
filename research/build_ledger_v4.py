#!/usr/bin/env python3
"""Build v4 exactly once; preserve v3 verbatim."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
r = json.loads((root / 'dissipative_flow_v4_results.json').read_text())
prior = (root / 'first_principles_robot_learning_ledger_v3.md').read_text()
doc = r'''# First-Principles Robot Learning: Living Research Ledger v4

**Date:** 2026-10-20, Europe/Berlin. **Status:** audited mathematical correction and executed replacement experiment. **Code/results:** `research/dissipative_flow_v4.py`, `research/dissipative_flow_v4_results.json`. Complete v3, including v2/v1, follows verbatim. Historical files are unchanged.

## Consequential correction before further theory

**REJECTED/CORRECTED:** V3's nonlinear simulator does not have the stationary velocity distribution invoked in its population interpretation. Its map `f(v)=.75v-.12v|v|` amplifies sufficiently large velocities. With nondegenerate iid Gaussian forcing it has **no invariant probability distribution**. This is a property of the mathematical simulator, not evidence that physical quadratic drag injects energy.

**PROVEN under assumptions / TESTED:** Exact unforced dissipative flow followed by independent motor/process impulses supplies a globally contractive replacement with a unique invariant velocity distribution. The first delayed footprint still measures the stationary average derivative. A direct derivation and independent audit precede experiment design.

**Narrow conclusion:** Stable finite rollouts cannot certify stationary validity of a nonlinear learning benchmark. The v3 finite numbers remain reproducible, but their stationary nonlinear-population interpretation is withdrawn. This is a substantive correction of our evidence, not a new general learning mechanism or completion of the persistent research Goal.

## Problem, observables and competing explanations

The agent observes positions `o_t=x_t+e_t` and iid Gaussian commands `a_t`; it does not observe velocity, process disturbance or sensor error. The quantity estimated is `Cov(o_(t+2)-o_(t+1),a_t)/Var(a_t)`, divided by the immediate coefficient. We need a stable velocity process before interpreting a long time average as a stationary physical derivative. Integrated position itself need not be stationary.

Two genuinely competing explanations for v3's benign nonlinear rollouts were:

1. Dissipative drag produces a stationary process; the sampled derivative estimates that process's expectation.
2. The discrete simulator is only locally dissipative. Finite samples miss a globally unstable region, so apparent stationary agreement is not evidence for a stationary expectation.

A counterexample starting at `v=20` with no later forcing decisively distinguishes these explanations. A theorem then determines whether Gaussian excitation can leave this pathology irrelevant asymptotically. A second framing treats the experiment as exact unforced physical flow plus impulses, instead of approximating damping by an unrestricted explicit polynomial update.

## F22. No invariant probability for the v3 nonlinear map [PROVEN under assumptions]

Let `X_(t+1)=rho X_t-c X_t|X_t|+xi_t`, where `c>0`, rho is finite, and xi are iid nondegenerate Gaussian variables. In v3, xi combines independent Gaussian motor and process noise. For any bounded set `B=[-M,M]`, choose

~~~text
R > M,   R >= max(2|rho|/c, 8/c).
~~~

When `|X|>=R` and `|xi|<=c X^2/4`, the reverse triangle inequality gives

~~~text
|rho X-cX|X|+xi| >= c X^2-|rho||X|-|xi|
                  >= c X^2/4 >= 2|X|.
~~~

From any point in B, a Gaussian innovation reaches `X_1>=R` with uniformly positive probability because the unforced map is bounded on B. Conditional on that event, require for every `n>=0`

~~~text
|xi_(n+1)| <= (c R^2/4) 4^n.
~~~

The independent event has positive probability: its Gaussian tail failure probabilities are summable, hence the infinite product of success probabilities is positive. Induction gives `|X_(n+1)|>=2^n R`. Thus there is positive probability of leaving B forever.

If an invariant probability pi existed, some bounded B would have `pi(B)>0`. In the resulting stationary path measure, Poincare recurrence implies almost every path starting with `X_0 in B` returns to B infinitely often. The positive-probability escape event contradicts this. No invariant probability exists. This does not assert that a short run at v3's stated amplitudes is likely to escape; no rare-event frequency is estimated here.

**Scope of correction:** Gaussian integration by parts remains valid at each fixed finite time (states are functions with finite polynomial-growth bounds of finitely many Gaussians). The identity then concerns a time-specific distribution. It does not supply a stationary law or an ergodic n-to-infinity interpretation for this map. The recorded near-agreement of empirical averages on benign v3 paths does not repair that distinction.

## F23. Exact dissipative flow gives a valid replacement [PROVEN under assumptions]

Start from the unforced physical law `dv/dt=-k v-d v|v|`, `k>0,d>0`. Sign is preserved. For `s=|v|`, `ds/dt=-ks-ds^2`. Separating variables over one unit of time yields

~~~text
r = exp(-k),  alpha = (d/k)(1-r)
f(v) = r v/(1+alpha |v|)
f'(v) = r/(1+alpha |v|)^2.
~~~

The derivative at zero is r by continuity. Therefore f is globally r-Lipschitz, `0<r<1`, and `|f(v)|<=r/alpha`. Declare the new physical experiment precisely:

~~~text
v_(t+1) = f(v_t) + beta a_t + w_t
x_(t+1) = x_t + v_(t+1)
o_t = x_t + e_t.
~~~

This is an exact damping flow followed by Gaussian impulses, with a discrete position update. It is **not** exact integration of continuously forced Langevin dynamics. Synchronous coupling contracts distances by at least r each step; the bounded f plus finite-moment Gaussian innovations yields existence, uniqueness and finite moments of the stationary velocity distribution. Burn-in error contracts geometrically.

For iid Gaussian exogenous commands, the finite-time Stein calculation gives

~~~text
H0=beta
H1=beta E[f'(v_(t+1))]
H1/H0=E[r/(1+alpha|v|)^2]  (stationary version).
~~~

Independent additive sensor noise increases sampling variance but contributes no action covariance. No unique physical-coordinate discovery or universal nonlinearity test follows.

The comparison uses `k=.25,d=.12,beta=1.2`, giving `r=0.7788007831`, `alpha=0.1061756241`. The old rho=.75 is the explicit-Euler coefficient `1-k`; the exact flow changes the transition law and its numerical sensitivities. We do not claim identical stationary responses or identical parameterization across the two simulators.

## E14. Executed stress test and replacement identity

Dependencies: Python 3.12.3, NumPy 2.5.3 in the existing `.venv`; no installation. Global seed 20261020. For each amplitude .35 and 1.25, 64 independent 6000-transition trajectories follow 600 burn-in steps, with process SD .06 and position-sensor SD 1.0. Seeds are `20261020+400000+1000*j+i`, j=0,1, i=0..63. Four action lags plus intercept fit measured displacement, exactly as a finite-lag first-order footprint; excluding three initial lags leaves 5997 fit rows. Latent v is exposed only to audit the derivative identity. All per-seed coefficients and identity errors are retained in JSON.

{{TABLE}}

The uncertainty column is the Monte Carlo standard error across independent trajectories for `(echo ratio - latent derivative average)`, not a confidence guarantee for a single trajectory. Agreement is consistent with the exact theorem; finite-sample ratio bias is not removed. We make no superiority claim over IV, subspace, recurrent or nonlinear estimators, so this is not a competitive learning benchmark.

With initial v=20 and zero subsequent forcing, the old map produces `20,-33,105.93,-1267.092288,...,-3.15646e140` within nine updates. The exact flow produces `20,4.98670,2.53922,1.55761,...,.0127912` after 20. This stress intervention is deliberately outside the original typical state range; it demonstrates global invalidity, not its nominal incidence.

On a grid of 12001 points in [-30,30], the exact flow semigroup composition error is below 5e-15 and the finite-difference derivative error below 9e-7. These checks catch flow/index implementation mistakes; the algebra establishes global behavior. Assertions ran successfully. Original v3 was rerun separately to check its saved finite data, without overwriting released results.

## Independent adversarial audit and provenance

The independent `independent_theory` subagent derived the escape argument, recurrence contradiction and exact damping flow without editing canonical artifacts. It specifically challenged stationary claims, the difference between continuous forcing and impulses, the distinction between exact-flow and Euler parameters, derivative indexing and the interpretation of finite-time Stein identities. Artifact-specific numerical audit is recorded in `research/audit_v4.md` before publication. The main agent owns integration and verification.

**EXISTING RESEARCH:** Hutzenthaler, Jentzen and Kloeden's [divergence analysis](https://arxiv.org/abs/0905.0273) establishes important Euler pathologies for non-globally-Lipschitz stochastic equations. Their [multilevel Monte Carlo analysis](https://arxiv.org/abs/1105.0226) further explains rare-event numerical failures. Those are related precedents, not a claim that their exact theorem is the elementary invariant-law proof above. Our finite-grid and stress checks are narrower than their results. The exact scalar damping solution is elementary ODE mathematics. Novelty is not claimed.

## Reproduction and precise continuation handoff

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/dissipative_flow_v4.py > /tmp/dissipative_flow_v4_check.json
cmp research/dissipative_flow_v4_results.json /tmp/dissipative_flow_v4_check.json
~~~

**Next investigation (begin immediately after publication):** Determine whether arbitrarily rich amplitude variation can compensate for the information discarded by first-order action covariances. Derive an exact stable coupled bilinear counterexample, compare first-order footprints with second-order Volterra and a flexible kernel predictor on identical noisy-position/action data, and test transfer under changed command means/coordination. Independent symmetry under global command sign reversal suggests that an even action-history component is invisible to every first-order footprint. This is a testable architectural limitation, not evidence against all response-based or state-based learning.

**Open:** A model class or statistical summary must be challenged separately from simulator validity. Nothing in this correction solves nonlinear partially observed feedback control, autonomous operator discovery or humanoid intelligence. The overall research Goal remains active.

---

# Historical archive: Complete original v3 ledger (including complete v2 and v1)

Verbatim v3 follows; corrections above supersede its stationary nonlinear interpretation without rewriting history.

'''
rows = ['| Motor SD | Mean H0 | Mean H1/H0 | Mean derivative | Mean error +/- MC SE |',
        '| --- | --- | --- | --- | --- |']
for d in r['stable_identity'].values():
    rows.append(f"| {d['amplitude']} | {d['mean_h0']:.6f} | {d['mean_echo_ratio']:.6f} | {d['mean_derivative']:.6f} | {d['mean_identity_error']:.6f} +/- {d['identity_error_mc_se']:.6f} |")
doc = doc.replace('{{TABLE}}', '\n'.join(rows))
out = root / 'first_principles_robot_learning_ledger_v4.md'
if out.exists():
    raise SystemExit('Refusing to overwrite released v4')
out.write_text(doc + prior)
assert out.read_text().endswith(prior)
print(out)
