# First-Principles Robot Learning: Living Research Ledger v4

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

| Motor SD | Mean H0 | Mean H1/H0 | Mean derivative | Mean error +/- MC SE |
| --- | --- | --- | --- | --- |
| 0.35 | 1.210880 | 0.700012 | 0.707438 | -0.007426 +/- 0.007723 |
| 1.25 | 1.199769 | 0.596126 | 0.595778 | 0.000348 +/- 0.001795 |

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

# First-Principles Robot Learning: Living Research Ledger v3

**Date:** 2026-10-20, Europe/Berlin. **Status:** mathematical derivations plus executed numerical experiments.  
**Canonical project:** `/home/dtdav/dev/learn`. **Code:** `research/car_learning_benchmark_v3.py`. **Recorded evidence:** `research/car_learning_benchmark_v3_results.json`.  
**History:** Full v2 is appended verbatim at the bottom, including its complete verbatim v1. Originals remain untouched as separate files.

## Version delta: a change of perspective

Instead of beginning with a guessed latent coordinate such as velocity, ask a simpler question: **How does a chosen action perturb the distribution of future observations?** A randomized command acts as its own experimental tag. A time-ordered sequence of action–observation covariances is an experimentally accessible *causal footprint* of the car. Its measured meaning is a property of the permitted interventions and observation channel, not a privileged physical coordinate.

This framing yields the following results.

1. **PROVEN conditionally / TESTED:** In a linear car, three short action-response coefficients can recover momentum persistence, direct motor gain and a one-step delayed motor effect, despite noisy position readings. The sample estimator is a known correlation/impulse-response method. Ordinary autoregressive regression is strongly biased under the same noisy observations.
2. **PROVEN conditionally / TESTED negative:** If a mechanical recurrence and a delayed actuator term cancel in the action-to-observation map, a mode of physical velocity can be invisible to these action-response statistics. The algorithm should return *non-identifiable*, not construct fictitious detailed coordinates. Process disturbances or deliberately changed initial conditions can reveal additional information not present in the action-only footprint.
3. **PROVEN via Gaussian integration by parts / TESTED:** For nonlinear drag, the first delayed action footprint divided by the immediate footprint equals the **average local derivative** of the uncontrolled transition law over the state distribution induced by the probing policy. Changing probe amplitude changes this average for a nonlinear car, but not for a linear car.
4. **TESTED negative:** A single simple amplitude-change threshold is not fully trustworthy: it falsely labeled **10/64** independent clean-linear-mechanism validation experiments as drag under noisy observation. It correctly flagged **62/64** independently held-out nonlinear cases. The detection threshold was chosen *after exploratory trials* and evaluated on fresh random seeds; these are limited statistical metrics, not a formal error-controlled test.
5. **PROVEN conditionally / TESTED:** When motor commands are chosen using noisy position feedback, ordinary command-output correlations are confounded. Independent randomized *perturbations* superimposed on the controller can still identify the immediate actuator coefficient without a direct velocity sensor.
6. **CRITICAL LIMIT:** An input-response footprint does not, in itself, recover all latent physical modes, determine desirable goals, invent nonlinear operators, or solve long-horizon nonlinear control. The experiment supplies iid Gaussian probing commands and one-dimensional position measurement. Extending it requires new experiments, not rhetorical extrapolation.

**Primary new insight:** Under deliberate intervention, one can learn *local causal response structure* without first claiming to know the correct hidden state. It suggests a productive representation target for robotics: observable action-conditioned **response derivatives and their temporal propagation**, eventually indexed by operating regime. That suggestion is a **HYPOTHESIS**, not a proven universal architecture. Close mathematical relatives are classical impulse-response identification, Markov parameters, best linear approximation, and Stein/Bussgang identities; novelty is not claimed.

## F18. Deriving the action footprint from elementary covariance [PROVEN under assumptions]

Let the car obey, with optional delayed actuation,

~~~text
v_(t+1) = rho v_t + beta a_t + gamma a_(t-1) + w_t
x_(t+1) = x_t + v_(t+1)
o_t = x_t + e_t.
~~~

The robot observes **only** positions o_t and issued motor commands a_t, not hidden v_t, w_t, or sensor errors e_t. The motor commands in this theorem are iid, centered, variance sigma_a^2 > 0 and independent of process disturbances, sensor disturbances and pre-existing state. The physical recurrence is stable, effects are stationary after burn-in, and sensor noise is additive and independent of the actions. Define the observable displacement D_(t+1)=o_(t+1)-o_t, and intervention-response coefficients

~~~text
H_k = Cov(D_(t+1), a_(t-k)) / Var(a_t),   k = 0,1,2,...
~~~

By independence, noise terms contribute zero to each covariance; so the footprint is not biased simply by independent measurement error, though its **sampling variance increases**. Direct substitution gives

~~~text
H_0 = beta
H_1 = rho*beta + gamma
H_2 = rho*(rho*beta + gamma)
H_3 = rho^2*(rho*beta + gamma)
~~~

For H_1 != 0, a reconstruction *within this selected parametric family* is

~~~text
beta = H_0
rho = H_2/H_1
gamma = H_1 - rho H_0.
~~~

**Derivation, not a magic feature:** v_(t+1) directly contains beta a_t, giving H_0=beta. The previous command a_(t-1) reaches v_(t+1) through rho times its contribution beta to v_t, plus gamma directly, giving H_1=rho beta+gamma. Earlier commands enter by repeated multiplication by rho. Taking output covariances with independent Gaussian or non-Gaussian centered actions isolates each lag's effect, because other action samples and exogenous noise are uncorrelated with the tagged action.

This is a classical impulse-response/Markov-parameter identification mechanism. It is particularly natural when raw position differences have correlated sensor noise (v2 F12). Our implementation estimates four H values by ordinary least squares of observed displacement against four motor-command lags, which is a finite-sample multi-regressor version of the covariance calculation.

**What is identified:** coefficients of one particular minimal action-to-output dynamic family; not physical coordinate uniqueness, not any unobservable mode, and not the autonomous choice of the correct family.

## F19. A controlled identifiability failure is more informative than an invented variable [PROVEN]

If gamma = -rho beta, then H_1 = H_2 = ... = 0 even though rho can be nonzero. The action-driven portion becomes

~~~text
v_(t+1) = beta a_t + rho*(v_t - beta a_(t-1)) + w_t.
~~~

Without process disturbances, the residual mode q_t=v_t-beta a_(t-1) evolves q_(t+1)=rho q_t. When q=0 at stationary initialization, every randomized motor probe has a memoryless observable effect beta a_t regardless of rho. The ratio H_2/H_1 becomes 0/0. With process disturbances q can fluctuate and create detectable unforced output correlations; those are **not captured by action-output cross-covariance alone**.

A sensor signal might reveal the residual mode through spontaneous fluctuations or preparations with nonzero q, while random action pulses alone still cannot excite it. Therefore the correct fundamental principle is not “recover every hidden state”; it is “recover only distinctions for which the available observations and interventions have identifying evidence, and retain explicit uncertainty otherwise.” This is classical uncontrollable/observable realization logic. A finite threshold |estimated H_1|<0.06 was used to flag weak identification in the test. The cutoff is heuristic and scale-dependent.

## F20. Nonlinear action echoes are averages of local physical sensitivity [PROVEN]

Now use nonlinear drift f and direct motor actuation:

~~~text
v_(t+1) = f(v_t) + beta a_t + w_t
f(v) = rho v - c v|v|,    c >= 0,
~~~

with iid **Gaussian** exogenous commands, independent w, stationary dynamics, and adequate regularity/integrability. The immediate response is H_0=beta. For the first delayed response, write v_(t+1)=f(v_t)+beta a_t+w_t; its effect on the next displacement is through f(v_(t+1)).

For Gaussian a_t, integration by parts (Stein's identity) yields

~~~text
E[a_t g(a_t)] / sigma_a^2 = E[g'(a_t)]
~~~

for smooth enough g and centered a_t. Apply this conditional on the pre-action state and disturbances, taking g(a_t)=f(f(v_t)+beta a_t+w_t) to obtain

~~~text
H_1 = beta E[f'(v_(t+1))]
H_1/H_0 = E[f'(v_(t+1))]
        = rho - 2c E[|v_(t+1)|].
~~~

Here f'(v)=rho-2c|v| for v !=0, and its continuous value at zero is rho. This identity links a purely experimental **input-output response** to a physical **local Jacobian averaged over the visited states**, with no velocity supplied to the estimator. The true hidden velocity is used only afterward to verify the theoretical identity in simulation.

Under linear dynamics (c=0), H_1/H_0=rho regardless of Gaussian probe amplitude. Under nonlinear drag (c>0), stronger actions generally increase E|v|, reducing the measured average local derivative. The reduction need **not** be monotonic for every nonlinear system or action distribution, and it is not a universal test of nonlinearity. More generally H_1/H_0 is a policy-dependent average derivative, not the derivative at a particular state.

**Robotics significance:** A humanoid's response to a joint torque depends on operating regime, pose and contact. A family of localized action-response derivatives can be more directly testable than first insisting on one globally correct latent variable. This is a research framing, not evidence the car algorithm learns humanoid contact dynamics.

## F21. Closed-loop interventions must be separated from ordinary commands [PROVEN, TESTED]

Let the robot stabilize observed displacement with

~~~text
a_t = -kappa*(o_t-o_(t-1)) + z_t
~~~

where z_t is independent, centered, randomized motor perturbation issued **after** observing state and independent of e_t. Raw a_t depends on the noisy measurement e_t. Consequently Cov(D_(t+1),a_t)/Var(a_t) mixes the direct physical motor effect with endogenous state/noise correlations and generally differs from beta.

For direct additive actuation, however, conditioning on pre-action history shows the immediate term beta*z_t enters D_(t+1) while the remaining contemporaneous terms are independent of z_t, hence

~~~text
beta = Cov(D_(t+1), z_t) / Var(z_t).
~~~

**Important scope:** This identifies the immediate coefficient under our direct-actuation and exogenous-perturbation assumptions. It does *not* imply that every later footprint of z gives open-loop plant dynamics: future feedback actions react to that perturbation, and delayed causal echoes then describe the plant-controller combination unless the feedback law is accounted for.

This is related to established closed-loop instrumental identification. The new perspective is to treat deliberately randomized perturbations as the experimental unit, not naively treat all commanded actions as independent interventions.

## Experimental protocol and code

Execute within the existing project-local environment:

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/car_learning_benchmark_v3.py > research/car_learning_benchmark_v3_results.json
~~~

Python 3.12.3 and NumPy 2.5.3. Global seed 20261020; all seeds explicitly generated in the code. True linear coefficients rho=0.75, beta=1.2; default Gaussian motor SD=0.7 and physical disturbance SD=0.06. Sensor error is independent Gaussian at the stated SD. **Continuous trajectories** are generated, not independent randomized tuples. The initial 400 simulated steps are removed as burn-in; model methods only receive measured positions and logged actions. Our experimenter may inspect true state solely when verifying a mathematically derived identity, not when estimating model parameters.

### E11: continuous car identification and indistinguishability

Six conditions with 48 independent seeds each, 4400 retained steps per run. For condition index i in the code, seeds are 20261020+1000i+j, j=0..47. Conditions: clean linear (sensor SD 0), linear SD 1.2, linear SD 2.4, delayed actuator gamma=0.34 and sensor SD 1.2, cancellation gamma=-0.9 and sensor SD 1.2, nonlinear drag c=0.12 and sensor SD 1.2. Three estimators use the **same logged trajectories**:
- **OLS ARX:** regression of observed next displacement against previous observed displacement, present and previous actions.
- **IV:** instrumental regression of the previous observed displacement with earlier actions, while present and past motor commands remain regressors. No direct velocity access.
- **Footprint:** regress observed next displacement on four action lags, infer beta=H0, rho=H2/H1, gamma=H1-rho H0 only when |H1|>=0.06.

Metric: median Euclidean coefficient error over independently executed runs for the **linear assumed parameter family** (rho,beta,gamma). For nonlinear drag the “coefficient error” is relative to the simulator's linear-part parameters and therefore measures **model mismatch**, not the quality of a nonlinear physical model. For cancellation the footprint is often **not estimable**; report n separately instead of treating skipped trials as successes.

| World | Sensor SD | OLS median L2 | IV median L2 | Footprint median L2 | Footprint identified |
| --- | --- | --- | --- | --- | --- |
| linear_exact | 0.0 | 0.0019 | 0.0022 | 0.0135 | 48/48 |
| linear_sensor_1_2 | 1.2 | 1.4789 | 0.0993 | 0.0890 | 48/48 |
| linear_sensor_2_4 | 2.4 | 1.8176 | 0.2732 | 0.2405 | 48/48 |
| delayed_sensor_1_2 | 1.2 | 1.2238 | 0.1092 | 0.0837 | 48/48 |
| cancelled_sensor_1_2 | 1.2 | 1.9521 | 2.7690 | 1.7888 | 3/48 |
| nonlinear_sensor_1_2 | 1.2 | 1.6116 | 0.2093 | 0.2047 | 48/48 |


**Negative findings:** A noise-free OLS model is much more statistically efficient than the footprint and IV in this linear benchmark. At high sensor noise, action footprints are substantially more accurate than naive OLS but are broadly similar to IV, not demonstrably better than established methods. In the cancellation case 45/48 footprint estimates fail the predeclared weak-identification cutoff; the remaining 3 valid finite numbers are unstable and are *not evidence of success*.

### E12: amplitude probes and an independently held-out failure test

Only x position and actions are observed. The generator uses sensor SD=1.0 and physical noise SD=0.06. For each independent run, conduct two **separate stationary probes** with Gaussian motor SD 0.35 and 1.25, retaining 5200 steps per amplitude, each with 400 burn-in steps.

Training/exploration split: 48 pairs of runs with seeds 20261020+16000+400i+2j, j=0,1. The one-sided heuristic “nonlinear drag detected if [H1/H0 at larger amplitude] - [H1/H0 at smaller amplitude] < -0.12” was selected after looking at initial exploratory experiments and must **not** be presented as a pre-registered test.

Independent confirmation split: 64 wholly new pairs using seed base 20261020+50000+400i+2j. No threshold retuning between exploratory and independent confirmation; this is the main reported sensitivity/specificity evidence.

| Split | World | Pairs | Footprint small/large | Derivative prediction small/large | OLS rho small/large | Flag fraction |
| --- | --- | --- | --- | --- | --- | --- |
| exploratory | linear | 48 | 0.737 / 0.750 | 0.750 / 0.750 | -0.368 / 0.241 | 4.2% |
| exploratory | drag | 48 | 0.630 / 0.427 | 0.643 / 0.426 | -0.429 / -0.289 | 89.6% |
| independent confirmation | linear | 64 | 0.757 / 0.750 | 0.750 / 0.750 | -0.370 / 0.241 | 15.6% |
| independent confirmation | drag | 64 | 0.650 / 0.427 | 0.643 / 0.427 | -0.430 / -0.287 | 96.9% |


The physical-derivative identity is independently validated: mean measured H1/H0 tracks rho-2c E|v| computed from *latent velocities revealed only to the experimenter*. In the linear regime, ordinary ARX rho varies strongly with motor amplitude because sensor noise is a larger fraction of displacement at low amplitudes; that creates false apparent nonlinearity. The causal footprint is substantially more invariant, but its finite-sample variability remains. In confirmation, the -0.12 threshold falsely flagged **10 of 64** physically linear runs (15.625% false alarms); it identified **62 of 64** drag runs (96.875% detection) on these exact conditions. This is not acceptable as a reliable universal discovery test; do not claim the diagnostic problem is solved.

There are *two different claims*: the nonlinear expectation identity is a mathematical theorem under our assumptions; the threshold's ability to classify worlds from finite data is an imperfect empirical result. The latter needs statistically calibrated, replicated intervention tests.

### E13: noisy closed-loop control and tagged perturbations

48 independent 5200-step runs per policy, seed 20261020+28000+i, i=0..47, using SD 1.2 position error, Gaussian SD .7 external perturbation and physical noise SD .06. Compare open-loop a_t=z_t to feedback a_t=-0.35 D_t + z_t using exactly the same exogenous seed schedule. Both generate continuous car trajectories.

| Policy | Feedback gain | Raw action coefficient | Exogenous perturbation gain | True gain |
| --- | --- | --- | --- | --- |
| open_loop | 0.00 | 1.2058 | 1.2053 | 1.2000 |
| feedback | 0.35 | 1.7489 | 1.2085 | 1.2000 |


The raw-action 'instantaneous gain' is badly biased in feedback, whereas correlation with the **exogenous perturbation** remains near beta=1.2. This does not mean closed-loop plant identification is automatically solved: feedback modifies all later impulse responses, and conditions that make z endogenous invalidate the proof.

## Mechanisms now supported, and what is not yet established

**Supported mechanisms, relative to particular assumptions:**
- Recover action-prediction dynamics from intervention footprints before choosing unique latent coordinates.
- Declare structural parameters non-identifiable when causal echoes cancel.
- Detect a change in average physical sensitivity through independent amplitude interventions without treating raw differenced positions as noiseless.
- Preserve an exogenous intervention tag under a noisy feedback controller for immediate-actuation estimation.

**Open mechanism:** An agent that autonomously decides *whether* to repeat probes, expand operator vocabulary, infer a new state coordinate, adjust its measurement model, or spend actions pursuing a target. E12's false-positive rate directly warns against simply linking a threshold crossing to automatic addition of a nonlinear drag feature.

### Two competing research framings that remain alive

**State-first:** Learn a sufficient action-conditional history representation, then identify dynamic laws and plan goals. Classical PSRs, subspace identification, state-space filters, and learned sequence models are serious baselines.

**Response-first:** Learn what accessible interventions change in future measurements (initially action-response footprints, then state/operating-regime-conditioned response maps); construct compact state only when it improves counterfactual response prediction or goal decisions. Classical impulse-response/Markov representations, best linear approximation and causal interventions are antecedents. This v3 experiment validates a useful part of response-first for synthetic systems, but **does not establish superiority** over state-first or modern methods.

**Goal-first:** Learn only differences that change feasible goal decisions, expanding when goals change. V1's goal-relevance theorem supports considering it; v3 does not experimentally compare it.

The eventual robotics target is to recover compact, transferable response structure for nonlinear coupled actuation and contact, not to glorify this car. A productive next falsification uses **one continuous unknown environment** with mixed process/sensor noise and nonlinear lag, a controller that cannot be turned off, and small independent randomized motor perturbations. The agent must allocate a fixed probe budget and compare models/representation changes on held-out counterfactual action sequences. Baselines must include classical IV/subspace identification, finite impulse models, state-space filters and learned recurrent predictors with equal information access.

## Remaining open questions, prioritized

1. How can the agent choose *which experiment* will separate response-nonlinearity from measurement noise and hidden-state aliasing, with a finite confidence budget? Confirmation false positives must be lower without giving up sensitivity.
2. How can it identify **local response derivatives indexed by learned operating conditions**, not just state-averaged derivatives, while seeing only noisy observations?
3. Can it detect and model a new physical mode when action echoes cancel but disturbances or changed initial conditions reveal it?
4. Can it maintain causal interpretation of future action-response propagation under autonomous feedback, without an unnatural permanently open-loop assumption?
5. When does a goal-conditioned model suffice, and when does a richer predictive response model measurably transfer better to an unforeseen goal?
6. What minimum operator/observation vocabulary is required for similar discovery in coupled nonlinear articulated bodies, and can it be grown from evidence?

## Research precedence and citations

These foundations are **not** claimed as our novel inventions; they are cited to compare our derivations with serious established alternatives.

1. Classical Methods of System Identification (1971), survey of identification by random-input cross-correlation and impulse response. https://www.sciencedirect.com/science/article/pii/S0076539208623462
2. Picci & Katayama, *Stochastic realization with exogenous inputs and subspace-methods identification* (1996). https://doi.org/10.1016/0165-1684(96)00050-3
3. Markovsky et al., *Algorithms for deterministic balanced subspace identification* (2005). https://doi.org/10.1016/j.automatica.2004.10.007
4. Enqvist, *Linear Models of Nonlinear FIR Systems with Gaussian Inputs* (2003), Bussgang-related best linear approximations. https://doi.org/10.1016/S1474-6670(17)35033-4
5. Enqvist, *Linear approximations of nonlinear FIR systems for separable input processes* (2005). https://www.sciencedirect.com/science/article/abs/pii/S0005109804003310
6. Stein's Gaussian integration-by-parts identity, mathematically applied in F20. https://elliotpaquette.github.io/high-dimensional-probability/lectures/10-gaussian-processes.html
7. *Instrumental variable methods for closed-loop system identification* (2005), explicitly about endogenous disturbance-control correlation. https://www.sciencedirect.com/science/article/pii/S0005109804002845
8. *Minimal state-space realization in linear system theory: an overview* (2000), including pole-zero cancellation and hidden modes. https://doi.org/10.1016/S0377-0427(00)00341-1
9. Muandet, Jitkrittum, Kübler (2020), *Kernel Conditional Moment Test via Maximum Moment Restriction*: relevant alternative for testing whether a predictive law satisfies interventional moments under more expressive test functions. https://proceedings.mlr.press/v124/muandet20a.html

The full previous v2 and v1 source inventories follow unchanged below. Sources support intellectual context, not claims of novelty or of generalization to robots.

## Reproducibility integrity and explicit history policy

- Simulations executed through the owner's local Terminal MCP with /home/dtdav/dev/learn/.venv/bin/python; no assistant cloud sandbox calculations.
- Source v1 and v2 files and JSON remain unchanged and tracked in the repository.
- The exact full text of v2 follows; v2 already embeds the entire v1. A future v4 should append v3 in full rather than delete past corrections.
- The V3 experiment uses physically continuous one-dimensional car trajectories, but exogenous open-loop probes are still a **privileged experimental capability**; autonomous goal-achieving exploration is not demonstrated.
- Empirical classification thresholds have no established frequentist significance level or proof of optimality; their post-exploratory tuning is explicitly disclosed.
- All metrics are summaries over finite reproducible synthetic experiments, not real hardware measurements.

---

# Historical archive: Complete original v2 ledger (which embeds v1)

Verbatim v2 follows; it is not retroactively edited or silently corrected by this new ledger.


# First-Principles Robot Learning: Living Research Ledger v2

**Date:** 2026-10-20. **Status:** mathematical deductions and executed synthetic experiments, not a universal learning theory.  
**Code:** car_learning_benchmark_v2.py. **Recorded outcomes:** car_learning_benchmark_v2_results.json.  
**Historical preservation:** The complete original v1 ledger follows verbatim in the archive at the end; v1 Python and JSON remain separate unchanged files.

## What changed from v1

1. **PROVEN under stated assumptions / TESTED:** If actual positions are corrupted by independent sensor noise, the velocity-like position difference can make naive regression estimate much lower momentum persistence than actually exists, possibly even the wrong sign. Valid motor-action instruments correct this bias asymptotically under independent random interventions.
2. **TESTED:** A generic, bounded arithmetic grammar composes a nonlinear drag feature from observations, and alternatively selects a lagged motor action when the actuator has one-step memory. It does not invent operators. Both cases are standard forms of identifiable model and feature selection, not a novel universal algorithm.
3. **PROVEN / TESTED negative:** A previous-action policy a_prev=2v makes distinct physical coefficients indistinguishable. Withheld samples from that same policy do not reveal the problem; independent interventions do.
4. **TESTED negative:** Physical process noise leaves prediction error even with the correct compact state. Adding unnecessary nonlinear features can worsen out-of-range error.
5. **TESTED:** A short-horizon D-opt experiment-selecting rule improved parameter estimation over random actions with equal action budgets on one linear car, though it assumes the correct parametric family and cannot justify global dominance.
6. **CORRECTION DURING V2:** An early exploratory implementation incorrectly aligned the action index when estimating from noisy positions, falsely suggesting nearly zero motor gain. The error was identified, corrected, and all published v2 results rerun with the proper action_t. V1 reexecution differed from the saved original only in floating-point values of at most 1.6e-15, not byte-for-byte.

These findings move the open question from “Does residual prediction error imply that a new state variable is needed?” to “**Which of missing history, wrong functional vocabulary, unidentifiable parameters, sensor corruption, and irreducible randomness produces the error?**” This classification is a hypothesis for further study, not a mechanism already solved in v2.

## Epistemic contract

- **PROVEN:** algebra derived from declared dynamics, noise laws, and identifiability conditions.
- **TESTED:** scripts actually executed locally with saved seeds and JSON.
- **EXISTING RESEARCH:** system identification, instrumental variables, active input design, symbolic sparse identification and predictive representations.
- **HYPOTHESIS:** a compact diagnostic controller choosing which type of representation repair or experiment to perform.
- **REJECTED:** residual magnitude proves missing state; any expression language generalizes universally; success on confounded same-policy holdout establishes causal transfer.
- **OPEN:** learning the correct diagnosis without supplied environment labels; online trajectory and closed-loop feedback; nonlinear partial observation; high-dimensional perception; scaling.

## V2-F12: An exact feature can lead to a biased law [PROVEN, TESTED]

Preserve v1's semi-implicit car:
~~~
v_(t+1) = rho v_t + beta a_t
x_(t+1) = x_t + v_(t+1).
~~~
Under exact position sensing, d_t = x_t - x_(t-1) = v_t. Under measured position o_t=x_t+e_t with independent, mean-zero sensor noise of variance sigma_e^2, the constructed displacement instead obeys
~~~
D_t = o_t-o_(t-1) = v_t+e_t-e_(t-1)
D_(t+1) = rho v_t + beta a_t +e_(t+1)-e_t.
~~~
Two adjacent differences reuse the error e_t with opposite signs. Thus Cov(e_t-e_(t-1),e_(t+1)-e_t)=-sigma_e^2 and Var(e_t-e_(t-1))=2 sigma_e^2. With independent, identically distributed, exogenous zero-mean actions, independent sensor errors, stable rho, no process noise, and stationary v_t, population OLS regression of D_(t+1) on [D_t,a_t] has coefficients
~~~
rho_OLS = (rho Var(v_t)-sigma_e^2) / (Var(v_t)+2 sigma_e^2)
beta_OLS = beta
Var(v_t) = beta^2 Var(a_t)/(1-rho^2).
~~~
**Proof:** The covariance between observed successive displacements, after removing the independently varying current action, is rho Var(v_t)-sigma_e^2. Divide it by the observed regressor variance Var(v_t)+2 sigma_e^2. Independence of current action from v_t and sensor errors preserves beta's population coefficient. If sensor noise is strong enough that sigma_e^2 > rho Var(v), the estimated rho can have the opposite sign to its true physical value.

This is a bias in parameter *identification*, not merely an increase in test RMSE. A representation-invention strategy that trusts the derived difference as an exact physical quantity may construct false anti-momentum concepts from entirely ordinary sensor noise.

**Instrumented alternative:** Use design X_t=[D_t,a_t] and instrument Z_t=[a_(t-1),a_t]. Earlier motor action influences v_t but, under the stated open-loop randomization assumptions, does not correlate with sensor error. Solve (Z'X)theta=Z'Y for Y_t=D_(t+1). In population, E[Z'(Y-Xtheta)]=0, and the moment matrix has full rank when beta !=0 and Var(a)>0. This is standard instrumental-variable identification, not a new estimator. Under feedback policies conditioned on noisy observations, weak actuator gain, or correlated environmental disturbances, the instrument can be invalid; no unconditional repair is claimed.

**Car interpretation:** Encoders often report noisy position, and finite differences create correlated velocity-estimation errors even when independent sensor samples are unbiased. An agent must be able to represent an observation mechanism, not just add physical state.

## V2-F13: Residual variation has at least two sources [PROVEN]

For proposed representation Z, motor input A, outcome Y, and a hypothetically sufficient physical state S, the law of total variance gives
~~~
Var(Y | Z,A) =
 E[Var(Y | S,Z,A) | Z,A] +
 Var(E[Y | S,Z,A] | Z,A).
~~~
The first term includes irreducible physical or observation randomness conditional on S, and the second reflects mixtures of distinguishable S values that the representation Z aliases. A nonzero predictive residual alone cannot separate these terms. The decomposition itself is an identity; physically interpreting the terms requires a correctly chosen S and stochastic model, neither available automatically to the learner.

In deterministic noiseless physics, two different futures from the same proposed Z and same intervention falsify exact sufficiency, *assuming the model has the required functional expressiveness*. Under stochastic observations the same pattern may occur without any additional recoverable latent state. This is a fundamental reason not to grow state at every prediction failure.

## V2-F14: Policy-confounded dynamics are not identified by cross-validation [PROVEN, TESTED]

Extend the car with a one-step motor-action effect:
~~~
v_(t+1) = 0.85v_t + 1.1a_t + 0.28a_(t-1).
~~~
If a training policy always obeys a_(t-1)=2v_t, all its data satisfy
~~~
v_(t+1) = (0.85 + 2(0.28))v_t + 1.1a_t
         = 1.41v_t + 1.1a_t.
~~~
The regressor columns (v,a_prev) have rank 1. Every coefficient pair with theta_v+2theta_prev=1.41 generates the same training and same-policy validation outcomes; (0.85,0.28) and (1.41,0) are examples. No longer sample or computationally stronger predictor can identify the two terms from that restricted policy alone. The counterexample is broken by independently varying earlier actions and velocity, provided experiments are feasible.

**Correction to any overreading of v1:** good held-out prediction establishes only generalization over a tested distribution, not identification of counterfactual effects under interventions that were absent from it.

## V2-F15: Bounded operator composition grows useful variables [TESTED, EXISTING RESEARCH]

The search starts with observations x, previous x, and remembered previous motor command, and fits current motor command as a default input. It enumerates generic sums/differences of measured positions, absolute values, squares, cubes, and products of *each candidate*, its absolute value, current action, and past action. There is no given named velocity sensor or special preselected drag feature.

For each candidate addition, ordinary least squares fits coefficients on independent training data and checks RMSE on separately sampled validation data with translated position coordinates. Greedy search accepts at most three expressions, requiring validation improvement greater than max(0.008,0.04 times current validation RMSE), with a tiny complexity tie breaker. This can select the difference x-x_prev, then compose its product with its absolute value, or add previous action. No train or validation data are drawn from the final test distribution.

This is finite **syntax-guided search**, not primitive operator discovery. Its expressive power, operator vocabulary, finite history, candidate count, greedy budget and selection criterion are all programmer-supplied inductive biases. SINDy/SINDYc, symbolic regression and syntax-guided synthesis already explore closely related approaches. A task with a genuinely unexpressible law remains unsolved.

## V2-F16: Informative action changes precision, not just available samples [PROVEN conditionally, TESTED]

For r_t=[v_t,a_t], the information matrix is G=sum_t r_t r_t'. With a correctly specified linear model and independent homoskedastic Gaussian innovation of variance sigma_w^2, exogenous fixed full-rank design gives conditional least-squares covariance sigma_w^2 G^(-1). An adaptive choice of action changes the design itself; this formula is an approximation for our adaptive experiment rather than an exact conditional assertion for feedback-selected regressors.

A myopic D-opt heuristic chooses the next action that maximizes log det(G+r_t r_t'+0.01 I) among allowed discrete commands. It can improve the smallest eigenvalue under a short budget. But information gain is not inherently goal achievement, and a random action policy already often achieves sufficient excitation.

## Experiment E8: Build and reject internal features

Environment output is next displacement v'. Measurements are x_t, x_(t-1), remembered a_(t-1), and selected a_t. Reset-sampled velocities and actions are independent except in the deliberately confounded case. Starting positions are uniform over [-5,5], shifted by 20 for validation and 100 for final out-of-range test. Train 500 transitions (v in [-1.3,1.3], actions [-1,1]), validation 500, test 1600 (v in [-2.4,2.4], actions [-1.4,1.4]). Cases follow base v'=0.85v+1.1a, with nonlinear addition -0.14v|v|, motor lag addition +0.28a_prev, both, independent process noise N(0,0.12^2), or confounding a_prev=2v on train and validation only. Seeds are 20261020+100i for training and +1,+2 for validation,test, i=0..5 for linear, nonlinear, lag, mixed, noise, confounded. The simulator allows independent resets; **these are not learned online policies**.

**Actual independent test RMSE (smaller better):**

| Case | Selected expressions | d only | d + a_prev | Adaptive | Maximal library |
|---|---|---|---|---|---|
| linear | (x-x_prev) | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| nonlinear | (x-x_prev); ((x-x_prev)*abs((x-x_prev))) | 0.17899 | 0.17901 | 0.00000 | 0.00000 |
| lag | (x-x_prev); a_prev | 0.22614 | 0.00000 | 0.00000 | 0.00000 |
| mixed | (x-x_prev); a_prev; ((x-x_prev)*abs((x-x_prev))) | 0.31196 | 0.17972 | 0.00000 | 0.00000 |
| noise | (x-x_prev) | 0.12022 | 0.12031 | 0.12022 | 0.15836 |
| confounded | a_prev | 0.82821 | 0.84004 | 1.25711 | 0.84004 |


The adaptive fit reconstructs exactly expressible noiseless laws within numerical precision. The 'noise' case selects only the difference and leaves test RMSE near 0.12. A larger candidate set can make this worse outside training support. Confounded training selects a wrong proxy and has very large intervention-shifted error: the method is **not** generally reliable. The table's maximal-library model is not called an oracle because with independent noise extra irrelevant features may overfit.

## Experiment E9: Sensor error and instrumental identification

Independent actions a_t ~ N(0,0.65^2), true rho=.8 and beta=1.3. Each of 35 independent trajectories has 6000 transitions (starting at v=0), with independent position sensor noise at SD 0, 0.5 or 1.0. Seeds 20261320..20261354. Both estimators use correctly indexed a_t; the IV uses a_(t-1) to instrument D_t. Comparisons below average per-trajectory coefficients, not pooled correlations. The displayed expected OLS rho uses the measured latent velocity variance as a plug-in, with finite-sample differences expected.

| Sensor SD | OLS mean (rho,beta) | IV mean (rho,beta) | Expected rho OLS | OLS rho SD | IV rho SD |
|---|---|---|---|---|---|
| 0.0 | (0.8000, 1.3000) | (0.8000, 1.3000) | 0.8000 | 0.0000 | 0.0000 |
| 0.5 | (0.5359, 1.2997) | (0.8038, 1.3018) | 0.5366 | 0.0093 | 0.0155 |
| 1.0 | (0.1440, 1.2975) | (0.8077, 1.3036) | 0.1448 | 0.0140 | 0.0310 |


V2 confirms a substantial positive momentum-estimation bias correction under *these* instrument-valid conditions. OLS beta stays near 1.3. The IV estimator trades larger variability for less large-sample bias. The experiment has clean open-loop inputs and no process noise, so it does not establish robustness in realistic noisy closed-loop robotics.

## Experiment E10: Equal-budget controlled motor excitation

A linear car has v'=0.8v+1.3a+w, w~N(0,0.12^2), starting v=0. Motor actions are restricted to {-1,0,1}. Velocity is inferred from successive exact position samples, not supplied by a velocity sensor. Three exploration rules, zero, uniform random, and greedy determinant, are compared across 200 seeds (20262020..20262219), 12 steps per episode for all, then fitted by standard least squares. Each method sees the same family of possible actions and same observation channel, but its distinct actions yield distinct trajectories.

| Policy | Median L2 error | Mean L2 error | Median min eigenvalue |
|---|---|---|---|
| zero | 1.30691 | 1.33156 | 0.00000 |
| random | 0.03966 | 0.04431 | 6.98207 |
| d_opt | 0.02943 | 0.03344 | 11.94145 |


The zero-input controller fails to identify beta even with many independent episodes. D-opt gave lower median Euclidean coefficient error than uniform random under the specified short budget, but the results do not demonstrate asymptotic superiority, robustness to changing dynamics, or usefulness for achieving an external objective.

## Negative evidence and model-selection diagnosis

- **Correct but incomplete state:** A current-position-only representation aliases distinct velocities; adding the positional difference solves the deterministic linear case, as in v1.
- **Wrong functional basis:** A linear memory model fails on velocity-dependent drag; generic composition selects a nonlinear expression when that grammar permits it.
- **Missing motor history:** A model conditioned on current velocity and command fails if the previous command still influences actuation; adding a_prev can restore predictability.
- **Unidentifiable coefficients:** a_prev=2v removes independent evidence. Validation on the same policy *cannot* certify transfer; a new intervention distribution is required.
- **Process noise:** Even the correct state predicts noisy outcomes imperfectly. Extra terms can overfit; one-step validation improvement is not proof of new physical structure.
- **Sensor noise:** Velocity differences become errors-in-variables and induce a false momentum estimate. Instrumentation can help if randomized past actions are genuine valid instruments.
- **Not addressed:** symbolic language learning from pixels, long-run stability, nonstationarity, tasks with changed reward/goal labels, real hardware, and fully autonomous diagnosis of failure sources.

## Competing theories and what would discriminate them

**C1 full predictive representations:** Aim for sufficient conditional distributions of future observations under every action sequence, preserving transfer to new goals. This is a formal target, not an efficient finite-sample method. Closest established work: predictive state representations and causal-state equivalence.

**C2 bounded grammar expansion:** Try temporal history and symbolic composition only when tested predictive distinctions require them. V2 explicitly tests a small version. Lack of operator invention and noisy-confound vulnerability remain serious.

**C3 task-dependent representation:** V1 demonstrates that a task might not require separating all dynamics. V2 did *not* compare goal-conditioned representation quality or success under new goal families; it remains a legitimate alternative.

**C4 calibrated sensor and hidden-state estimation:** Explicit observation likelihood, Kalman/filter methods, instrumental variables and state-space identification challenge the premise that symbolic state growth is always the correct repair. V2 noisy-sensor experiment is evidence for this alternative under its assumptions.

**C5 adaptive experiment design:** Select interventions to break uncertainty rather than just observe more of the same actions. V2 includes a short D-opt baseline. It does not compare Bayesian designs or goal-conditioned exploration.

A useful future mechanism would maintain multiple *failure explanations* instead of automatically growing representation: possible missing history, missing nonlinear operator, confounded parameter, sensor artifact or stochastic innovation. It would select a discriminating observation or action, then accept a revised representation only after intervention-distribution transfer. **This meta-diagnostic controller is an OPEN HYPOTHESIS**, not an experiment completed here.

Falsification demands one learner unaware of the true environment class, with mixed unknown sensor noise, process noise, delayed actuator effects, and restricted resets, measuring misdiagnosis frequency, interaction budget, out-of-distribution conditional prediction, and task performance. Compare established ARX/IV, state-space filtering, recurrent predictors, sparse nonlinear identification, and goal-conditioned models under genuinely matched access to data, interventions and compute.

## Minimum built-ins suggested, not proven universal

- Temporal order, controlled intervention identity and observation history sufficient to form candidate distinctions.
- Some inductive structure to generate/test variables or predictors; a finite experience history alone cannot uniquely imply an unrestricted law.
- A way to test the generalization of causal predictions outside one behavior policy, with explicit acknowledgment of inaccessible experiments.
- Recognition of measurement uncertainty and irreducible stochasticity as alternatives to missing physical state.
- Separate preference/goal information for choosing desirable behavior; physics alone cannot imply what the robot should want (preserved from v1).
- Capacity to choose whether more observation, new expression, new memory, better instrumentation or parameter excitation is warranted.

The evidence does **not** establish that these are sufficient for general intelligence or humanoid robotics, nor that any one is the uniquely minimal implementation.

## Primary research verified for V2

1. Littman, Sutton, Singh, *Predictive Representations of State* (2002), NeurIPS. https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
2. Brunton, Proctor, Kutz, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems* (2016), PNAS. https://doi.org/10.1073/pnas.1517384113
3. Brunton, Proctor, Kutz, *Sparse Identification of Nonlinear Dynamics with Control (SINDYc)* (2016). https://doi.org/10.1016/j.ifacol.2016.10.249
4. Söderström, *Errors-in-variables methods in system identification* (2007), Automatica. https://doi.org/10.1016/j.automatica.2006.11.025
5. *On instrumental variable-based methods for errors-in-variables model identification* (2008). https://doi.org/10.3182/20080706-5-KR-1001.00072
6. MathWorks, *Picking Instrumental Variables for System Identification*. https://www.mathworks.com/help/ident/ug/picking-instrumental-variables-for-system-identification.html
7. Alur et al., *Syntax-guided synthesis* (2013). https://escholarship.org/uc/item/1g67m7hp
8. Abate et al., *Counterexample Guided Inductive Synthesis Modulo Theories* (2018). https://doi.org/10.1007/978-3-319-96145-3_15
9. Zhang et al., *Active learning for optimal intervention design in causal models* (2023), Nature Machine Intelligence. https://doi.org/10.1038/s42256-023-00719-0
10. Nguyen, Nghiem, *Active Learning for Optimal Experimental Design in Machine Learning-Based Building Energy System Identification* (2026). https://arxiv.org/abs/2606.25301

These are antecedents and competing approaches, not external validation of our particular tests. V1's additional 13 references and earlier rejected ideas are preserved verbatim in the archive below.

## Exact reproduction and historical integrity

From /home/dtdav/dev/learn:

~~~bash
.venv/bin/python research/car_learning_benchmark_v1.py > /tmp/v1_comparison.json
.venv/bin/python research/car_learning_benchmark_v2.py > research/car_learning_benchmark_v2_results.json
~~~

V2 uses Python 3.12.3 and NumPy 2.5.3 on the owner's PC. V2 JSON output was regenerated twice with identical SHA-256. V1 was reexecuted successfully with maximum numeric scalar difference <= 1.6e-15 from the preserved original JSON due to floating-point implementation differences. No v1 document or saved results file is overwritten.

V2 is a one-step synthetic demonstration, not evidence of learning from perception or of novel intelligence algorithms. Do not promote a numeric coincidence to a theorem, or confuse sufficient deterministic regression with identifying an interventional model under noise and policy restrictions. The following archive is complete; a v3 must preserve it along with all new v2 findings, including negative results and the caught indexing bug.

---

# Complete Historical Archive: Original Living Research Ledger v1

This is the unchanged v1 source in full. Earlier conclusions are preserved for auditability; corrections and additional limitations are documented above.


# First-Principles Robot Learning: Living Research Ledger

**Version:** 1  
**Created:** 2026-10-20  
**Scope:** Derive and experimentally test the minimal structure needed for a robot to learn general principles through action and observation.  
**Status:** Research notebook, not a claim of a new algorithm or a universal theory of intelligence.  
**Companion artifacts:** `car_learning_benchmark_v1.py`, `car_learning_benchmark_v1_results.json`.

## Read this first: current position (V1)

1. A robot cannot infer which outcome is *desirable* from physical trajectories alone when different desirability rules are compatible with the same physics. Some preference/goal/judgment information must be supplied or assumed.
2. A finite dataset does not uniquely determine a universal rule. Every learner requires an inductive bias, hypothesis class, prior, or analogous structural assumption.
3. A useful *state* is information from history sufficient to predict the task-relevant effects of future **interventions**. The canonical exact predictive state can be defined as an equivalence class of histories with identical distributions of future observations under every chosen future action sequence.
4. A robot cannot learn a variable or parameter through data that never distinguishes its possible values. Experiments must provide informative variation. This is a rank/identifiability question, not only an optimization question.
5. The learned latent coordinates are not uniquely determined. Equivalent invertible reparameterizations yield identical physical predictions. Demand correct transferable behavior, not human-named hidden variables.
6. The world model needed for all possible tasks is generally more detailed than the world model needed for one particular goal. Learning should distinguish uncertainty that changes decisions from uncertainty that does not.
7. In a toy simulation with no direct velocity sensor, a small expression grammar selected `x_t - x_(t-1)` as a useful predictor. It did **not** discover arithmetic or physical concepts from pixels, and its expression grammar is an explicit inductive bias.
8. In a small nonlinear variant, providing `v*abs(v)` in the expression grammar enabled exact recovery of a nonlinear drag law. This demonstrates expressibility within the grammar, not an unrestricted nonlinear-learning breakthrough.
9. No candidate approach (reward maximization, pure prediction, symbolic model discovery, active experimentation, or goal-conditioned models) has been demonstrated superior overall. Retain parallel alternatives and design falsifying comparisons.

## 0. Research discipline and boundaries

**Objective:** Start from basic logical and mathematical requirements for learning, then consult established research to avoid relabeling old results as new. Keep the example a simple car, expandable to nonlinear dynamics and later robotics.

**Included:** partial observation, state discovery, causal action-outcome models, labeled goodness, hypothesis revision, experiment choice, goal transfer, nonlinear identification.

**Deferred intentionally:** safety, alignment, humanoid embodiment, large pretrained models, optimization scaling, hardware details, reward hacking. These are important independently but would obscure the foundational question at this stage.

**Epistemic labels:**
- **PROVEN under assumptions:** mathematically derived from specified assumptions.
- **TESTED in simulation:** code ran and reproducible numeric results are saved.
- **EXISTING RESEARCH:** established by cited literature, not claimed as new.
- **HYPOTHESIS:** proposed architecture or generalization, not demonstrated.
- **OPEN:** unresolved and requires new reasoning, mathematics, or experiments.
- **REJECTED / CORRECTED:** a previous claim was too strong, mismatched the equations, or failed a test.

## 1. Question history, decisions and corrections

| ID | Question or proposed principle | Result/status | Next AI should know |
|---|---|---|---|
| Q01 | How can labeled car movements imply "forward is good"? | PROVEN conditionally | Positive displacement fits, but infinitely many laws fit finite training labels. |
| Q02 | Is maximizing reward the fundamental built-in mechanism? | OPEN; no universal proof | An objective may select behavior without specifying a representation-learning method. The reverse claim, that reward is useless, is also unsupported. |
| Q03 | Could prompting experiences to an LLM suffice? | INSUFFICIENT AS A FOUNDATION | A prompt supplies information, while its interpretation depends on a previously learned inference mechanism. |
| Q04 | Could pure world prediction suffice? | NEGATIVE for arbitrary preference learning | Identical physics permits opposite desirability rules. Prediction is powerful but does not uniquely supply goals. |
| Q05 | Can the robot discover state variables from prediction failures? | TESTED in small grammar | Position alone aliases different momentum states; adding history or its difference resolves this toy ambiguity. |
| Q06 | Does the robot necessarily discover *velocity* uniquely? | REJECTED | `(x, x_prev)` and `(x, x-x_prev)` are invertibly related. More generally latent coordinates are non-unique. |
| Q07 | Can informative actions help identify parameters? | PROVEN conditionally; TESTED | The input/velocity design matrix must have full column rank for unique noiseless recovery of `(alpha,beta)`. |
| Q08 | Must the robot model every physical detail? | NEGATIVE for a fixed restricted goal | Two distinct models can imply the same adequate action. Different goals can require finer distinctions. |
| Q09 | Does a single state-potential always encode "good transitions"? | REJECTED | Directed cycles of strictly good transitions cannot all increase a state-only scalar potential. A transition reward can nevertheless encode them. |
| Q10 | Can the same principle extend beyond linear physics? | TESTED on one nonlinear grammar; general case OPEN | A sparse library containing `v*abs(v)` recovered nonlinear drag; unknown basis, noise and generalization remain hard. |
| Q11 | Are safe experiments needed for the foundational derivation? | DEFERRED by request | Do not reintroduce as a leading topic without a specific need. |

**Correction to previous discussion:** Two integration conventions were used across explanations. Under explicit Euler `x_next=x+v`, `v_next=v+a`, the two-step trajectory `(x,v)=(8,1)` to `(10,0)` uses actions `(0,-1)`. Under **semi-implicit Euler** `v_next=v+a`, `x_next=x+v_next`, the same actions go from `(8,1)` to `(9,0)`, **not** `(10,0)`. V1 uses semi-implicit Euler consistently. This was caught by executable assertions and corrected; do not silently mix conventions again.

## 2. Basic formal setup and what is given

At time `t` an agent has a history `h_t=(o_0,a_0,o_1,...,a_(t-1),o_t)` and may receive judgments `j_t` about transitions or goals. A physical system obeys an unknown transition law. We distinguish:

- **Observation:** `o_t`, in our car usually measured position `x_t`.
- **Intervention:** chosen motor command `a_t`.
- **Transition rule:** how the physical state changes under the chosen command.
- **Judgment/goal:** external information about what is desirable, when supplied.
- **Representation:** `z_t=phi(h_t)` summarizing the history.
- **Model class:** the allowed expressions/distributions for laws; always an inductive bias.
- **Experiment-selection rule:** how the agent chooses which intervention to try.
- **Action-selection rule:** how the agent pursues a supplied goal using its knowledge.

There is no free inference of human goals from physics alone. In two worlds with identical physical trajectories and opposite judgment maps, all observations lacking judgments are indistinguishable.

## 3. The benchmark car: enough complexity without unnecessary complications

**Main V1 environment:** a friction/momentum car on an unrestricted 1D line. It has position `x_t`, hidden velocity `v_t`, and motor command `a_t`:

```text
v_(t+1) = alpha*v_t + beta*a_t
x_(t+1) = x_t + v_(t+1)
o_t     = x_t
```

The agent sees `x` and remembers action histories, **not** a direct velocity sensor. `alpha` is momentum persistence; `beta` is motor authority. Simulation values: `alpha=0.8`, `beta=1.3` unless otherwise stated. When velocities are formed from consecutive positions, `v_t = x_t - x_(t-1)` under these update conventions.

We choose semi-implicit position integration because it keeps the observation-only expression concise. The selection is a modeling convention, **not** a principle of learning.

**Minimal variants to expose specific capacities:**

- B0: `x_next=x+a`, no memory needed; easy to memorize a button; insufficient benchmark by itself.
- B1: unknown preference over displacement, judgments supplied intermittently; distinguish `a>0` from `x_next>x`.
- B2: momentum/drag with position-only observation (main benchmark); forces useful history.
- B3: unknown physical parameters; requires informative interventions.
- B4: changing target `g`; examine transfer via `e=g-x`.
- B5: velocity-dependent nonlinear drag `v_next=0.9v-0.15v|v|+1.1a`; inspect expression discovery and extrapolation.
- Later, only when B1-B5 are solved honestly: partial noisy observation, stochastic disturbances, a simple pendulum/cart-pole, then higher-dimensional articulated robots.

## 4. Precise mathematical findings

### F01. Finite evidence does not entail a unique universal law [PROVEN]

For observations with all starting positions below 10, these two hypotheses fit equally well:

```text
H1: Good iff x_next > x
H2: Good iff x_next > x and x < 10
```

No finite set that excludes starting positions at or above 10 can select between them. Simplicity, symmetry, grammar limitations, a prior, or new experiments are necessary. It is incorrect to claim a completely assumption-free universal rule discoverer.

### F02. Current observation may not be a predictive state [PROVEN]

For `alpha=0.8`, `beta=1.3`, at `x=5` with `a=0`:

```text
v=+1 -> x_next=5.8
v=-1 -> x_next=4.2
```

The same `(x,a)` maps to two outcomes. A deterministic predictor based only on `(x,a)` cannot be exactly correct in both cases. Distinguishing the preceding positions (or their difference) suffices in this toy:

```text
d_t = x_t - x_(t-1)
x_(t+1) - x_t = alpha*d_t + beta*a_t
```

This is not proof that velocity is the uniquely correct latent coordinate.

### F03. An exact predictive state can be defined by future experiments [PROVEN as a definition]

Histories `h` and `h'` are equivalent if, for **every** permitted future action sequence `A` and future observation sequence `Y`, they imply the same future distribution:

```text
h ~ h'  iff  P(Y | h, do(A)) = P(Y | h', do(A))  for all A,Y.
```

The equivalence classes partition histories into the coarsest classes that preserve *all* these interventional predictions. If two histories produce distinguishable distributions for even one available test, any exact all-tests predictor must distinguish them.

Caveats:
- This is a mathematical target, not a finite-sample algorithm.
- It depends on which observations and future experiments are admitted.
- Finite data, noise, untested action sequences, and computational limits prevent directly computing the universal partition.
- For a narrower goal family or finite horizon, a coarser quotient may suffice.

Connections: predictive state representations (Littman, Sutton, Singh), causal states (Shalizi, Crutchfield), automata state minimization, bisimulation.

### F04. For this linear car, whether velocity is *needed* has an exact rank criterion [PROVEN]

Write `s=(x,v)^T`, `s_next=A s+B a`, `x=C s`, with

```text
A = [[1, alpha], [0, alpha]]
B = [[beta], [beta]]
C = [[1, 0]]
```

The two-step observability matrix is

```text
O = [ C ; C A ] = [[1,0], [1,alpha]]
```

and `det(O)=alpha`. If `alpha != 0`, the physical 2D state is observable from position measurements with known inputs. If `alpha=0`, the current velocity does not influence future dynamics; carrying it separately is unnecessary for future physical position predictions. For `beta != 0`, the controllability matrix `[B,AB]` has determinant `-beta^2` and full rank. Thus for this linear model, `alpha != 0` and `beta != 0` give a minimal two-dimensional observable/reachable realization. This is a special-case conclusion, not a general claim about nonlinear systems.

**Key philosophical point:** an internal variable is warranted by its role in distinguishing futures, not by a human insistence that the variable must exist in the model.

### F05. Parameter recovery requires experimentally independent variation [PROVEN]

From velocity observations:

```text
[v'_1, v'_2]^T = [[v_1,a_1],[v_2,a_2]] [alpha,beta]^T
```

Two noiseless experiments identify `(alpha,beta)` if and only if

```text
det([[v_1,a_1],[v_2,a_2]]) != 0.
```

A concrete experiment pair `(v,a,v')=(1,0,0.8)` and `(0,1,1.3)` recovers `(0.8,1.3)` exactly. If every action satisfies `a=2v`, the design matrix has rank 1 and the data determine only `alpha+2beta=3.4`. The parameter pairs `(0.8,1.3)` and `(3.4,0)` are indistinguishable under these restricted inputs.

No number of repeated rank-one measurements uniquely identifies two independent parameters under this model. This is the finite-dimensional version of the need for informative system excitation. Stochastic/noisy estimation additionally needs conditioning and sufficient samples.

### F06. The "true" coordinates are not uniquely recoverable [PROVEN]

Let `z_next=F(z,a)` and `o=O(z)`. For any invertible `psi`, let

```text
z_tilde       = psi(z)
F_tilde       = psi o F o psi^(-1)
O_tilde       = O o psi^(-1)
```

Then the transformed model produces the same observations under every action sequence. For example `(x,v)` and `(x,2v)` are equally valid coordinate systems once their equations are transformed. A purely behavioral learner cannot infer a unique privileged gauge or name for the hidden variables.

To select interpretable coordinates we may impose convenience, translation symmetry, sparsity, units, locality, separability, or other inductive biases. These choices must be documented, not portrayed as consequences of observations alone.

### F07. Prediction sufficiency depends on the goal [PROVEN by counterexample]

Assume `v=0`, `alpha` irrelevant for this step, unknown `beta` is either `0.5` or `1.0`, and available actions are `{1,2}`.

- Goal A, *move positively*: both actions work under both models. Identifying beta is unnecessary to succeed once.
- Goal B, *move exactly one unit*: beta=0.5 requires action 2; beta=1.0 requires action 1. The optimal action depends on the unknown law.
- Executing action 1 once and observing displacement 0.5 or 1 distinguishes the two models.

Define candidate optimal action sets `A*_M(z,g)`. If their intersection across current candidate models is nonempty, a universally satisfactory action exists for this goal (given the chosen success criterion). If the intersection is empty, information about the model may change the decision. This motivates **goal-relevant** experiment selection, not learning every parameter for its own sake.

This connects to decision-aware/value-equivalent modeling. It is not a claim that information-seeking is always the correct next action; it is a rule for identifying relevant uncertainty.

### F08. Good transitions do not necessarily increase a state potential [PROVEN]

Suppose a 1D shuttle is praised for *moving*, so both `0->1` and `1->0` are Good. A state-only progress potential `U(x)` satisfying `U(x_next)>U(x)` for every Good transition would require `U(1)>U(0)` and `U(0)>U(1)` simultaneously, impossible.

This refutes **state-potential increase as a universal description of goodness**. It does not refute rewards: the transition reward `R(x,x_next)=1[x_next != x]` encodes the labels perfectly. A reward may also depend on action or history. Distinguish a restricted type of value function from a general transition reward.

### F09. Predictive disagreement and labeled disagreement are different [PROVEN conditionally]

Candidate judgment rules:

```text
H1: Good iff a>0
H2: Good iff x_next-x>0
H3: Good iff x_next-x>0 and x<10
```

Judged transitions under main car physics:

```text
(x,v,a)=(0,0,1)  -> displacement +1.3, Good: H1,H2,H3 survive
(x,v,a)=(20,0,1) -> displacement +1.3, Good: H3 rejected
(x,v,a)=(3,2,0)  -> displacement +1.6, Good: H1 rejected
```

H2 is the only survivor among the **listed** hypotheses. No claim follows about all conceivable rules. The judgment is provided externally; it is not entailed by physics.

### F10. The target "reach and stop" requires looking beyond immediate movement [PROVEN for toy]

For the frictionless semi-implicit variant:

```text
v_next=v+a
x_next=x+v_next
allowed a in {-1,0,1}
start (x,v)=(8,1); goal (x,v)=(9,0), exactly 2 steps
```

Enumerating 9 two-action sequences yields exactly one: `(0,-1)`. Translating all positions and the goal by +100 yields the identical action sequence from `(108,1)` to `(109,0)`.

This demonstrates the utility of a goal-relative representation `e=g-x` and planning over consequences. It does not show an agent discovered such a representation independently; the exhaustive planner was supplied.

### F11. A small nonlinear extension can be attacked with the same program-search principle [TESTED]

Simulated law:

```text
v_next = 0.9*v - 0.15*v*abs(v) + 1.1*a
x_next = x + v_next
```

The feature candidates were `[v,a]`, `[v,v^2,a]`, `[v,v^3,a]`, `[v,v*abs(v),a]`. Train on `v in [-1.5,1.5]`; test on wider `[-2.5,2.5]`.

- Linear model out-of-range RMSE: approximately 0.1984.
- Cubic approximation out-of-range RMSE: approximately 0.1154.
- The correct `v*abs(v)` library member: RMSE approximately `1.06e-15` and coefficients `(0.9,-0.15,1.1)`.

This does **not** prove that an arbitrary nonlinear physical law can be recovered with a finite grammar. It demonstrates why choosing or growing the candidate function library is a real foundational design problem.

## 5. Empirical ledger: exact V1 tests

Code: `car_learning_benchmark_v1.py`  
Data: `car_learning_benchmark_v1_results.json`  
Dependencies: Python 3, NumPy.  
Reproduce: `python car_learning_benchmark_v1.py > car_learning_benchmark_v1_results.json`  
RNG seed: `20261020` for primary experiments and `20261021` for the nonlinear experiment.

All experiments are synthetic, deterministic, and built from known benchmark equations. Independent random tuples of position, velocity, and action are sampled using a simulator reset, so the fitting set is not a single correlated driving trajectory.

**Displacement prediction using 200 training and 1,000 test transitions:**

| Learned predictor input | Held-out RMSE | Interpretation |
|---|---:|---|
| Motor action alone | 1.399674 | Fails to account for momentum |
| Current position + motor action | 1.407207 | Position-only state is inadequate |
| Current + prior position + motor | 2.83e-15 | Enough history to reconstruct relevant momentum |
| `(current - prior)` + motor | 1.18e-15 | Compact derived coordinate sufficient under the selected dynamics |

A grammar of simple single-feature expressions selected `x_t-x_(t-1)` **or its negative** when paired with an action term. Both signs are behaviorally equivalent after coefficient adjustment. This is another demonstration of coordinate non-uniqueness.

**Parameter excitation:** full-rank two-probe design recovered `(alpha,beta)=(0.8,1.3)` exactly. Confounded probes `a=2v` have rank 1; alternative `(3.4,0)` makes identical predictions on those probes.

**Goal transfer:** exact two-step action pair `(0,-1)` worked for both `(8,1)->(9,0)` and the translated `(108,1)->(109,0)` with the same dynamics and shifted target. This is a planned demonstration, not learned policy transfer.

**Numerical tests:** the companion script asserts the expected prediction and rank properties, rule elimination, goal-planning uniqueness, and nonlinear recovery. Floating-point errors at `1e-15` should be interpreted as numerical zero.

## 6. Three rival foundation-level approaches, not yet collapsed

### Candidate C1: Pure prediction plus goal-directed planning [RETAIN as baseline]

Learn an action-conditional predictive model of observations, then plan toward externally supplied goals. Advantage: physical knowledge can transfer across tasks. Weakness: exact prediction of every physical detail may be unnecessarily expensive, and the model cannot identify desirability without goal data.

### Candidate C2: Counterexample-driven symbolic state and law discovery [RETAIN as main experimental hypothesis]

Maintain candidate expressions/models. Seek distinct histories or interventions that falsify current representations. When predictive aliasing occurs, generate additional features from a grammar. Prefer compact models that generalize on holdout experiments. Advantage: transparent, falsifiable, feasible to test precisely on simple systems. Weakness: grammar bias and combinatorial explosion; exact falsification fragile under noise; no demonstrated scalability.

### Candidate C3: Task-conditioned predictive abstraction [RETAIN as substantive alternative]

Preserve only distinctions required to choose appropriate actions for specified goal families/horizons. Advantage: can discard task-irrelevant physics. Weakness: risks being too narrow when a novel goal requires a previously discarded distinction. Need quantify the tradeoff by systematic goal transfer.

**Reward maximization, imitation learning, world models, in-context prompting, active inference** should be included later as implemented baselines when a concrete benchmark can discriminate them. Mathematical re-expression of a successful policy as a reward optimizer is not by itself evidence about its learning efficiency or representation quality. Equally, C1-C3 can be implemented within reward-based algorithms; new names are not novelty.

**Comparison requirement:** same action budget, access to sensor data, access to judgments, candidate grammar/inductive structure, compute budget, and held-out regimes where feasible. Do not compare a hypothesis-rich symbolic agent to a deliberately crippled black-box model and call the result a discovery.

## 7. Working algorithm: intervention-driven representation refinement [HYPOTHESIS]

```text
Input:
    observations o; permitted actions A; occasional judgments j or goals g
    expression grammar G (explicit inductive bias)
    hypothesis set or parametric model family M
Initialize:
    smallest feature set z = phi(h) (e.g. current position)
Loop:
    1. Predict future observations conditional on candidate actions using M.
    2. Check for state aliasing: similar z and action, incompatible outcomes.
    3. If aliasing persists, search G for history-derived features that
       separate the conflicting cases without harming held-out prediction.
    4. Maintain several compatible laws rather than only one when data
       do not identify a unique model.
    5. For the current goal, determine whether all plausible models
       admit a common satisfactory action.
    6. If a common action exists, it may be taken without full ID.
       If not, find an accessible experiment that distinguishes
       goal-relevant models, subject to specified experimental costs.
    7. Execute, observe, acquire any available judgment, revise beliefs.
    8. Retest on held-out conditions and revise the representation.
```

**Known omissions:** no general guarantee of termination, efficient grammar generation, approximate state splitting under noise, optimal experimental design, or scaling to images. No claim of a novel algorithm. Some components overlap established active learning, symbolic regression, system identification, and planning.

## 8. What would falsify or improve the working hypothesis?

**Priority experiment E1 (state discovery):** Supply only `(x_t, a_t, judgment when available)` and finite history. Allow the algorithm to grow expressions such as `x_t-x_(t-1)` from a declared small grammar; compare against fixed-memory and fixed-state baselines. Criterion: whether it autonomously proposes a predictive feature and improves held-out causal prediction.

**E2 (identifiability-aware action selection):** Compare passive action sequences to experiment-selection rules for learning `alpha,beta`, using equal numbers of experiments and identical candidate sets. Metric: parameter error and time to distinguish models. Deliberately include a confounded action sequence that produces rank-deficient data.

**E3 (coordinate/generalization):** Train at positions around zero; test shifted origins and a changed destination. Check whether the learned expression rather than just the policy generalizes. Rotate the representation by a known invertible transformation and check if performance remains equivalent.

**E4 (task relevance):** Two possible motor gains produce the same adequate actions for a coarse progress goal but different actions for an exact displacement goal. Compare full identification versus task-aware identification on experiments needed to succeed. Do not assume task-aware ID always wins once a distribution over future tasks is considered.

**E5 (nonlinear capacity):** With the same small grammar, add velocity-dependent drag; then add a nonlinear term **outside** the grammar. The latter should falsify any claim that the grammar is universally sufficient. Compare grammar growth, fixed sparse libraries, and a generic nonparametric predictor, under equal evidence.

**E6 (judgment-rule inference):** Maintain multiple rules for goodness (command-positive, actual-displacement-positive, position-limited). Choose experiments that make their predictions disagree. Count external judgment queries. The judge remains an independent data channel.

**E7 (novel goal transfer):** Test abstract representations over goal families with varying precision, horizons, and reversals. A representation minimal for the old task may fail a new task. This should guide whether full predictive and task-conditioned models need to coexist.

## 9. What this ledger does *not* establish

- That maximal compression, predictive state equivalence, or information gain alone defines all intelligent behavior.
- That a universal minimal representation exists independent of observations, interventions, horizons, and tasks.
- That the symbolic grammar itself can emerge without built-in inductive biases.
- That a self-built mechanism outperforms well-designed reinforcement learning, self-supervised world models, or any existing baseline.
- That the toy demonstration scales to uncertain nonlinear robotics or raw visual observations.
- That a preference label on actions, transitions, or trajectories is automatically representable by a monotonic state potential.
- That active experiments always improve task performance enough to justify their cost.

## 10. Research links (primary sources and directly relevant technical material)

These references supply context and guard against accidental claims of novelty. They are **not** presented as proof that the proposed V1 agent is superior.

1. **Littman, Sutton, Singh, _Predictive Representations of State_** (NeurIPS): action-conditional future observations as state. https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
2. **Shalizi and Crutchfield, _Computational Mechanics: Pattern and Prediction, Structure and Simplicity_** (2001): equivalence of pasts by future predictions and statistical complexity. https://doi.org/10.1023/A:1010388907793
3. **Angluin, _Learning regular sets from queries and counterexamples_** (1987): active rule identification using counterexamples with strong assumptions about query access. https://doi.org/10.1016/0890-5401(87)90052-6
4. **Ljung, _System Identification: Theory for the User_, 2nd ed.** (1999): model class, data design, identifiability, validation. https://www.mit.bme.hu/system/files/oktatas/targyak/9132/Ljung_L_System_Identification_Theory_for_User-ed2.pdf
5. **Tedrake, _Underactuated Robotics_, System Identification chapter**: accessible robotics-oriented discussion of model learning. https://underactuated.mit.edu/sysid.html
6. **Rissanen, _Modeling by shortest data description_** (1978), and **Grunwald and Roos, _Minimum Description Length Revisited_**: explicit representation/compression assumptions. https://doi.org/10.1016/0005-1098(78)90005-5 ; https://arxiv.org/abs/1908.08484
7. **Schmidt and Lipson, _Distilling Free-Form Natural Laws from Experimental Data_** (2009): symbolic-law discovery from physical measurements. https://doi.org/10.1126/science.1165893
8. **Brunton, Proctor, Kutz, _Discovering governing equations from data by sparse identification of nonlinear dynamical systems_** (2016): sparse dynamics identification. https://doi.org/10.1073/pnas.1517384113
9. **Locatello et al., _Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations_** (ICML 2019): need for inductive biases; latent identification limitations. https://proceedings.mlr.press/v97/locatello19a.html
10. **Grimm et al., _The Value Equivalence Principle for Model-Based Reinforcement Learning_** (NeurIPS 2020): predictions relevant to planning, not necessarily full physics. https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html
11. **Silver, Singh, Precup, Sutton, _Reward is enough_** (2021): a competing unifying hypothesis, explicitly presented by the authors as a hypothesis. https://doi.org/10.1016/j.artint.2021.103535
12. **Bongard, Zykov, Lipson, _Resilient Machines Through Continuous Self-Modeling_** (2006): a real robot inferring its own body model from motor-sensation data. https://doi.org/10.1126/science.1133687
13. **Classical linear realization/observability reference**: observability, reachability, state minimality. https://doi.org/10.1016/S0377-0427(00)00341-1

## 11. Instructions for the next AI / next ledger release

1. **Read all of V1 before extending.** First check whether an alleged new idea is already F01-F11, the research links, or a corrected old mistake.
2. **Create a new filename for each release:** `first_principles_robot_learning_ledger_v2.md`, then `_v3.md`, etc. Never overwrite a released version.
3. Copy V1 into V2 as historical material or retain all its contents through a well-labeled versioned history. Mark entries `SUPERSEDED` with a reason and replacement. Do not silently delete failed ideas, corrections, or negative results.
4. Put a concise *version delta* at the beginning of each new release: new theorems, empirical evidence, corrections, rejected hypotheses, new alternatives, outstanding problems.
5. Maintain the status labels from Section 0. The distinction between a theorem, a simulation, a conjecture, and an assumption is mandatory.
6. Any new numerical result must name the environment equations, observation channel, action set, sample size, random seed, train/test split, and code or sufficient reproducibility instructions.
7. Do not claim novelty until checking relevant primary literature. Cite the closest competing approaches, not only supportive work.
8. Prioritize **new discriminatory experiments** over adding more terminology. An experiment is useful if it could prove one proposed mechanism wrong or distinguish it from a serious competitor.
9. Continue expanding the car only when the simpler version cannot reveal the property being investigated. Prefer changing exactly one source of difficulty at a time.
10. Main unresolved question: **How can an agent autonomously enlarge its representation language when no expression in the current grammar is adequate, and how can it tell the difference between insufficient features and an inadequate dynamics model?**

### End of V1
