#!/usr/bin/env python3
"""Single bounded diagnostic, complete prior v12 history preserved."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v12.md').read_text()
r=json.loads((root/'gradient_reuse_v13_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v13

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited bounded policy-search diagnostic, with negative precision results and retained failed checks/optimizer. V12 was committed and published over the existing SSH origin/main at `ace6041069d246656caa3bcffb3239db93a1bfa2`, verified before continuing. Complete v12 and all earlier ledgers follow verbatim. The general-learning Goal remains ACTIVE.

## Main result: separate family capacity from finite search, and average cost from capability

**TESTED:** Optimizing the SAME six-coordinate stationary controller on a model learned from v12's existing experience, with verified exact derivatives and separate virtual validation, improves fresh mean tracking cost over both physical CEM and virtual CEM on all four declared plants. No new plant training is used. On the faster two plants, the resulting mean cost approaches the supplied OE/EKF/MPC controller. A limited direct family can therefore do substantially better than its earlier finite CEM result; attributing the entire v12 gap to family incapacity is refuted constructively.

**TESTED NEGATIVE:** This does not yield reliable precise station keeping. The gradient-selected policies meet the all-late10 position/velocity criterion only1/64 fresh task trials, versus14/64 for physical CEM,16/64 for virtual CEM and25/64 for nonlinear MPC. Improving an aggregate finite-horizon score is not a general capability improvement. Velocity precision worsens relative to physical CEM on all four plants. Residual mean-cost gaps on slow plants remain; capacity versus search is not globally settled.

**OPEN:** The experiment does not identify a necessary learning architecture or show the best attainable direct policy, model, optimizer or planner. This is an exploratory repair prescribed after v12 outcomes, tested with new independent noise streams. Model-based computation uses supplied physical vocabulary and predicted states; it is a hybrid, not a model-free learning breakthrough. End the scalar optimizer investigation here, irrespective of the favorable mean-cost result.

## Why one more diagnostic was justified, and why stop this branch now

V12 found better mean behavior from a predictive pipeline fitted to the exact same experience, but changing both policy family and optimizer left a causal ambiguity. Its same-family virtual CEM restart did not consistently help. A single derivative-based fit can constructively falsify the strongest 'the family cannot do better' explanation. A failed local optimization could not prove that explanation. This asymmetric falsification criterion, fixed starts/banks/budgets and a one-diagnostic limit were recorded in `gradient_reuse_v13_design.md` before execution; independent review supported them.

The useful uncertainty is now partly resolved: finite CEM and its use of scalar episode fitness left attainable mean performance undiscovered in the very same family. That does not establish why every remaining task fails. More scalar optimizers would increasingly substitute gain tuning for general intelligent learning. The next problem must challenge supplied function vocabulary/coupling or another consequential foundation, rather than optimizing this score indefinitely.

## F52. Exact pathwise derivative through an actual bounded controller [DERIVED / EXISTING RESEARCH]

For fixed virtual sensor noise, the declared controller and deterministic plant define a finite map from six parameters to trajectory loss. For any computation `y=a*b`, propagate `dy=b*da+a*db`; for tanh(v), `d tanh(v)=(1-tanh(v)^2)dv`. Logs give dkp/dp0=kp and dkd/dp1=kd; logit coordinate gives dalpha/dp2=alpha(1-alpha). Sensor errors are held fixed, so do=o's modeled position derivative, not zero.

With old smoothing variables saved before each update,

~~~text
dsx_new=(1-alpha)*dsx_old+alpha*dx+(x+e-sx_old)*dalpha
dsv_new=(1-alpha)*dsv_old+alpha*(dsx_new-dsx_old)
        +(sx_new-sx_old-sv_old)*dalpha.
~~~

Differentiate all six terms in `raw=kp*(g-sx)-kd*sv+bv*sv_old-bu1*u_prev-bu2*u_older`, including parameter dependence of past ISSUED commands. Then

~~~text
u=clip(raw,-1,1), du=1{|raw|<1}*draw
vnew=rho*v-c*tanh(v)+m
 dvnew=(rho-c*(1-tanh(v)^2))*dv+dm
mnew=lambda*m+beta*u, dmnew=lambda*dm+beta*du
xnew=x+vnew, dxnew=dx+dvnew.
~~~

This is the ordinary chain rule/piecewise-smooth sensitivity, not a novel gradient estimator. At exactly raw=+/-1 the clip need not be differentiable; zero derivative there is a convention, not a theorem. No straight-through derivative is used for saturated commands. Bounded TRF least squares is still local, and tiny projected gradient cannot certify a global optimum, especially near kinks.

**CORRECTED VERIFICATION:** Initial fixed central-difference step1e-6 crossed ONE clip-active-set boundary in bv, giving discrepancy.331221, while the other coordinates agreed to1.3e-8. Source execution stopped before optimization. Instrumented checks preserve this failure and require both fixed steps1e-7/1e-8 to keep identical saturation masks and agree. They do: maximum discrepancies1.09e-8/1.66e-7. Controller and derivative formulas were unchanged. The correction and full check table remain in `gradient_reuse_v13_check_correction.md` and JSON. Independent autodiff/recurrence audits distinguish real algebra verification from simply shrinking a step until a test passes; extreme ill-conditioned trajectories remain explicitly qualified.

## E33. One bounded same-family gradient/validation experiment [TESTED]

Reuse the32 final nonlinear OE models from v12: four plants rho=.65,beta=1.2,lambda .2/.7,c0/.2, eight independent original training seeds each. Do NOT refit on test outcomes. The exact stationary six-coordinate controller, bounds, two training goal schedules and known-zero resets remain. All calculations use the owner's existing project environment.

For each learned model, draw a fixed16-episode virtual sensor-noise bank (SD.2), alternating the two training goals with eight draws each, motor process0. Minimize the model's mean normalized position/velocity/command cost, using predicted states rather than hidden actual plant reward. These virtual states are legitimate outputs of the acquired predictor. Optimizing this finite noise bank can overfit; it does not inherit an unbiased selected-policy fitness guarantee.

Three FIXED starts are physical-CEM winner, virtual-CEM winner and generic(.08,.6,.6,0,0,0). Bounded least_squares TRF, max160 evaluations/start, ftol/xtol1e-9,gtol1e-8, exact analytic residual Jacobian. Preserve every start, outcome, evaluation count and projected gradient. A separate fixed16-episode virtual validation bank scores all three optimized candidates plus both unoptimized winners; select minimum validation cost. No actual-plant test enters selection. All32 selected candidates are optimized:8 from the physical-winner start,11 virtual-winner start,13 generic start. Validation selection is itself finite/noisy and optimistic; only fresh plant tests support transfer.

RNG namespaces use each v12 physicalseed plus20billion for fitting,30billion for validation and40billion+task for NEW benchmark-plant evaluation. Role ranges are disjoint from all v12 streams. Each actual test uses the same underlying noise for physical CEM, virtual CEM, gradient selected and nonlinear MPC, two heldout goal-amplitude schedules. Motor process SD.03 and sensor SD.2 throughout,86 bounded feedback actions, no forced coast, goals switch6/28 without resetting memories. No future goal sequence is revealed. MPC retains its supplied finite horizon/clock while the three direct policies are stationary.

Primary cost is the same86-stage mean divided by1+mean(goal²). True states only score actual plant episodes. Means/SE average tasks within each original fit seed before summarizing eight clusters; these are fresh noise tests of reused fits, not32 newly trained independent models. Each plant has16 task trials with shared-fit dependence.

{{TABLE}}

Physical-CEM-minus-gradient paired differences by plant are.047864+/-.007912,.054984+/-.010174,.096162+/-.008644,.062994+/-.013799 (MC SE). Nonlinear-MPC-minus-gradient differences are+.000452+/-.003115,-.003685+/-.002267,-.039716+/-.003875,-.025869+/-.003686. The first two support close mean performance under this protocol, not a formal equivalence test. The slow plants retain clear gaps in these finite summaries. No global policy optimum or universal superiority follows. Optimization, fixed-bank sampling, validation selection and virtual compute change together; this experiment does not isolate derivatives alone as the cause of the gain.

## Mean performance and precision disagree [TESTED NEGATIVE]

The gradient policies have higher final10 velocity RMS than physical CEM in every cell:

{{LATE}}

Position RMS is mixed; lower full-episode quadratic cost can coexist with worse final velocity precision. The whole-event count depends on all ten position AND velocity tolerances, not only an average. Small task counts cannot calibrate population success. The decline is preserved as an inconvenient result, not hidden behind a better mean score.

Independent audit decomposes pooled equal-task cost: position contribution drops.339797 to.270589 from physical CEM to gradient selection, while velocity contribution rises.009869 to.013201 and effort.000174 to.000549. Pooled late position RMS is almost unchanged(.199636 to.200485), but late velocity RMS rises.110705 to.160634. Of64 gradient trials,27 fail only velocity,35 both position/velocity,1 only position,1 neither. Thus62/64 violate the velocity requirement. These descriptive component summaries identify what the scalar gain contains; they do not establish the best precision policy or a causal role for a particular sensor/parameter.

Training optimizes small goals with process0; actual heldout tasks have larger amplitudes and process noise. This, the finite objective, current policy class and optimizer can all contribute to task differences. This experiment does not isolate their causal shares. Do not add a post-hoc precision-trained method and call it confirmatory, or infer that all information requirements have been solved. The next scientifically useful step is beyond this scalar gain family.

## Audits, failures, exact budgets and reproduction

There are96 optimizer attempts;95 report convergence. The generic start on plant3/replicate7 reaches max160 with status0, cost.1226847333, projected gradient6.28e-6. It is retained and NOT selected (that fit selects candidate1). Maximum projected gradient across all attempts is.00257074; convergence tolerances include objective/step conditions and are not uniform stationarity guarantees. All fresh MPC action solves report convergence; no global nonlinear guarantee follows.

The saved unique optimizer rollout calls consume6595168 learned-model transitions. Validation consumes220160; retained selected-trajectory replay88064; final derivative checks50912. No new plant training. Actual fresh evaluation consumes22016 benchmark-plant transitions/256 resets. Debug/check failure and reexecution are development compute, not hidden new independent training replicas. Exact per-attempt function/Jacobian counts and original wall timings are retained; their sum is implementation-specific, not a fair cross-method runtime claim. Temporary byte-for-byte replay matches the saved JSON and NPZ; wall timings are separately nondeterministic.

Independent theory derives smoothing, coordinate chains, actual clipping and lagged dynamics derivatives and checks a separate PyTorch autodiff implementation. Independent artifact implementation checks finite-noise banks, all candidate endpoints/objectives/gradients/validation scores, selected virtual trajectories and fresh actual-plant controllers/EKF/MPC/metrics, source hashes and clustered summaries. It does not claim to reproduce every optimizer iteration or prove global solutions. Reports and executable audit/results are preserved; audit limitations are part of the claim. Raw generated metadata records pre-audit execution; separate reports certify subsequent verification.

~~~bash
.venv/bin/python research/gradient_reuse_v13.py --output-prefix /tmp/gradient_v13
cmp research/gradient_reuse_v13_results.json /tmp/gradient_v13_results.json
cmp research/gradient_reuse_v13_data.npz /tmp/gradient_v13_data.npz
.venv/bin/python research/audit_gradient_reuse_v13.py
~~~

**EXISTING RESEARCH:** Chain-rule simulation sensitivities and gradient-based trajectory/policy optimization are established. [Tedrake's direct-shooting/trajectory-optimization notes](https://underactuated.mit.edu/trajopt.html) explain related computational structure; [Peters/Schaal2006](https://people.csail.mit.edu/rplatt/IROS2006-Peters.pdf) discusses established policy-gradient approaches. Our exact known-learned-model derivatives are not a score-function estimator for an unknown physical plant. CEM, TRF, raw output-error fitting and EKF/MPC remain conventional methods; no novelty or modern robotics benchmark claim.

## Consequential interpretation and next direction

Experience can be used to optimize observed behavior without fitting dynamics; acquired predictive consequences can instead support counterfactual virtual computation. V12's one unchanged virtual CEM did not reliably exploit that possibility. V13 constructively shows that a different conventional optimizer extracts useful behavior in the same family from existing learned models. This supports a finite-search limitation in the earlier experiment, rather than declaring an architecture necessary. Precision failures show why successful learning must be judged against the intended capabilities, not only the convenient scalar average.

The objective remains general learning in unfamiliar nonlinear physical systems. Supplied scalar model order, positive control sign, correct tanh vocabulary, known noise/reset/time conventions and hand-selected policy coordinates remain major priors. No learned mechanism has discovered its own representational language or demonstrated coupled articulated control. More scalar optimizer variants now risk drift.

**Immediately continue beyond this branch:** Derive a minimal coupled articulated or unknown-function-vocabulary test. Identify which information and function dependencies the existing methods cannot acquire from their supplied families. Compare a genuinely general candidate with strong simple conventional alternatives, holding actual experience and task facts explicit, and use predictive validation AND decisions to test acquired knowledge. Start with the smallest decisive system; do not assume neural networks, physical state labels, rewards or world models are necessary foundations. Preserve failed candidates. No further scalar controller optimizer ladder is planned. The Goal remains ACTIVE through publication.

---

# Historical archive: Complete original v12 ledger, including all preceding versions

'''
rows=['| lambda,c | Physical CEM cost +/- SE | Virtual CEM cost +/- SE | Gradient-selected cost +/- SE | Nonlinear MPC cost +/- SE | Station keeping P/V/G/M |','| --- | ---: | ---: | ---: | ---: | --- |']
for s in r['summary']:
    values=[]
    for name in ('physical_CEM','virtual_CEM','gradient_selected','nonlinear_MPC'):
        a=s['controllers'][name]['cost'];values.append(f"{a['mean']:.6f} +/- {a['mc_se']:.6f}")
    lam=.2 if s['plant']<2 else .7;c=0 if s['plant']%2==0 else .2
    successes='/'.join(str(s['controllers'][n]['stationary_successes']) for n in ('physical_CEM','virtual_CEM','gradient_selected','nonlinear_MPC'))
    rows.append(f'| {lam},{c} | '+' | '.join(values)+f' | {successes} of16 |')
doc=doc.replace('{{TABLE}}','\n'.join(rows))
rows=['| Plant | Physical CEM late velocity RMS | Gradient late velocity RMS | MPC late velocity RMS |','| ---: | ---: | ---: | ---: |']
for pi in range(4):
    current=[a for a in r['runs'] if a['plant']==pi]
    values=[np.mean([t['controllers'][n]['late_velocity_rms'] for a in current for t in a['tests']]) for n in ('physical_CEM','gradient_selected','nonlinear_MPC')]
    rows.append(f'| {pi} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
doc=doc.replace('{{LATE}}','\n'.join(rows));assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v13.md'
if out.exists():raise SystemExit('Refusing to overwrite released v13')
if not (root/'audit_gradient_reuse_v13_results.json').exists():raise SystemExit('Independent audit required')
out.write_text(doc+prior);assert out.read_text().endswith(prior);print(out)
