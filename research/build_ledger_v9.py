#!/usr/bin/env python3
"""Build v9 once with complete v8 historical suffix preserved verbatim."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prior=(root/'first_principles_robot_learning_ledger_v8.md').read_text()
ident=json.loads((root/'nonlinear_identification_v9_results.json').read_text())
gauge=json.loads((root/'nonlinear_force_gauge_v9_results.json').read_text())
doc=r'''# First-Principles Robot Learning: Living Research Ledger v9

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited nonlinear force-allocation limitation, restricted-family recovery, and noisy-position identification experiments. Full v8 and every preceding ledger follow verbatim. The prior repository HEAD `ecdc206fd5032c0b828f8deb5e6b991424861b82` and released v8 `33be698bb90bf83c63c5d0328c55aa30c78d2ea1` were present on SSH origin/main before this investigation. The persistent research Goal remains ACTIVE.

## What changes the construction of a learner

**PROVEN UNDER ASSUMPTIONS:** Nonlinear curvature does not universally locate passive versus motor force. A two-state nonlinear actuator can exactly reproduce a nonlinear-body/linear-actuator system under every ordinary command/position experiment, including adaptive feedback and zero-command atoms. This is classical realization freedom, not a new mathematical discovery.

**PROVEN UNDER ASSUMPTIONS:** Within a supplied velocity-independent linear-actuator family, nonzero tanh passive curvature does label the motor pole when the observable regression has full rank. Removing that modularity assumption removes the conclusion. At zero curvature, v8's label ambiguity returns.

**TESTED:** On 72 noisy-position datasets, a conventional raw-position output-error fit predicts new sequences much more accurately than either unrestricted regression on differenced noisy positions or a comparator using exactly the same four physical parameters and bounds but noisy regressors. Small-amplitude training still gives poor amplitude transfer; reliable physical labels do not follow from good matched prediction. This is a class-specific identification demonstration, with supplied tanh law, noise type and zero initialization; no autonomous architecture discovery or modern-method superiority is claimed.

**REJECTED:** The inference that nonlinear passive curvature alone establishes physical component attribution from commands and position. The assumption of a linear velocity-independent motor is information supplied by the investigator, not a conclusion of these observations.

## Starting from experience rather than a component name

V8 asked whether nonlinear passive structure could break its linear pole swap. Before fitting a richer model, ask what physical distinction the observations actually measure. A force term can be placed on the body or inside the motor assembly without changing connected motion. The two explanations disagree about component removal, which is a new intervention with physical semantics beyond issuing commands.

Use **lagged** motor timing throughout this version:

~~~text
v_(t+1)=f(v_t)+m_t
m_(t+1)=lambda*m_t+beta*u_t+xi_t
x_(t+1)=x_t+v_(t+1), o_t=x_t+e_t
f(v)=rho*v-c*tanh(v).
~~~

This differs explicitly from v8's motor-first timing. True rho=.65, lambda=.4, beta=1.2, with c declared per experiment. In the equivalence experiments c=.2, xi SD=.06 and position-sensor SD=.5. Zero initial states are compatible. With 0<c<rho<1, f is globally rho-Lipschitz; the linear actuator is stable. Bounded feedback plus Gaussian probes gives finite state moments. No stationarity of integrated position is claimed.

## F40. An exact nonlinear force-allocation freedom [PROVEN UNDER ASSUMPTIONS]

More generally, let `vnext=f(v)+m`, `mnext=h(m,v,u)+xi`. For any fixed measurable function phi define `q=m+phi(v)` and `F=f-phi`. This triangular state transform is invertible even if phi is not. Then

~~~text
vnext=F(v)+q
qnext=h(q-phi(v),v,u)+phi(F(v)+q)+xi.
~~~

The phi argument is computed from the current state; there is no future sensor value or implicit future-noise dependence. The same additive actuator innovation survives because xi enters mnext after vnext is determined. With mapped initial states and common sensor errors/random policy coins, induction gives identical observed histories and commands for every observation-based adaptive policy. This proves equality of complete trajectory laws, not merely equality of some response moments.

For `phi(v)=-c*tanh(v)`, the alternative has linear passive drift:

~~~text
vnext=rho*v+q
qnext=lambda*q+lambda*c*tanh(v)+beta*u
      -c*tanh(rho*v+q)+xi.
~~~

It is causal, smooth, input-affine and still two-state. It is excluded by assuming the motor is velocity independent. That exclusion needs physical evidence if physical interpretation matters.

**Scope:** q0=m0+phi(v0) is required. Independently preparing the same nonzero velocity with both total force channels certified zero generally breaks the mapping. Direct mechanical noise in vnext changes the transformed noise law; preserving the same independent additive motor noise is timing/noise-placement specific. Known force bounds, sensor calibration, modularity restrictions or other interventions may also rule out an alternative. No impossibility for every nonlinear robot follows.

## F41. Coordinate freedom and physical component transfer [PROVEN UNDER ASSUMPTIONS]

A coordinate change alone cannot change any physically specified intervention. Under the same physical disconnection m=0, the correct transformed condition is q=-c*tanh(v); both descriptions then predict f(v). Setting q=0 instead would be a different operation. The independent auditors emphasized this distinction before publication.

To obtain genuinely different physical worlds, explicitly locate the same drag on different components. World A has nonlinear drag on the body and active motor force m. World B has only linear body drift and the nonlinear drag in the removable motor assembly; its **total delivered motor force** is q=m-c*tanh(v). Its underlying active motor state still evolves linearly. Connected trajectories match exactly. Physical removal of the complete motor assembly leaves the nonlinear drag only in A, giving respectively `vnext=f(v)` and `vnext=rho*v`. The component boundary/removal semantics are an explicit extra assumption. A sensor of active current alone need not locate shaft versus body drag; a calibrated sensor of total delivered force can distinguish these worlds.

For any pretraining-only learner, the two worlds induce the same prediction distribution before removal. If their post-removal velocity vectors are a,b, the triangle inequality gives `max_world E[RMSE(prediction,response)] >= RMSE(a,b)/2`. Equal-prior expected squared error is at least `RMSE(a,b)^2/4`. The component-location metadata does not disclose the hidden true location.

{{REMOVAL}}

The preparation and removal replay start at known zero state and use no process or sensor noise, a privileged deterministic evaluation setup. This bound concerns those fixed clean response vectors after the newly labeled physical intervention and before post-change observations, not ordinary command prediction or the ability to adapt after removal.

## F42. What the restricted model really identifies [PROVEN UNDER ASSUMPTIONS]

Elimination in the lagged model gives

~~~text
v_(t+2)=(rho+lambda)v_(t+1)-rho*lambda*v_t
        -c*tanh(v_(t+1))+lambda*c*tanh(v_t)+beta*u_t+xi_t.
~~~

With exact velocity and full rank of `[vnext,v,tanh(vnext),tanh(v),u]`, noiseless coefficients theta are unique. For c!=0, `lambda=-theta3/theta2`, `c=-theta2`, `rho=theta0-lambda`, and beta=theta4 (zero-based indices). The additional restriction theta1=-rho*lambda checks consistency. Under stochastic innovations a population identification claim also needs exogeneity and averaging assumptions; this cycle's clean identification tests set xi=0.

At c=0 the unrestricted feature matrix may still have full rank, but theta2=theta3=0 makes the label ratio undefined. The remaining linear coefficients determine an unordered pole pair. Thus full observable-regression rank does not by itself establish physical parameter identification.

For small velocities, tanh(v)=v-v^3/3+O(v^5); nonlinear columns almost duplicate linear columns. Curvature-specific information can be weak despite exact mathematical identifiability. No universal finite-sample rate is claimed. Singular values diagnose this particular observable regression, not full general-learning sufficiency.

## F43. Nonlinear observation uncertainty cannot reuse the linear IV proof [DERIVED]

Measured differences satisfy D_t=v_t+e_t-e_(t-1). Even with an exogenous instrument Z, the term `E[Z*(tanh(v_t)-tanh(D_t))]` need not vanish. Independence of sensor errors does not turn a nonlinear function of noisy D into a function of noiseless v. The affine IV cancellation in v8 therefore does not justify naive nonlinear IV. This is established errors-in-variables reasoning.

With **no process noise**, exogenous commands and known zero initial state, the simulated position x(theta,u) is deterministic. iid Gaussian sensor errors make `sum_t (o_t-x_t(theta,u))^2` a valid negative log-likelihood up to scale/additive constants. This supplies a conventional output-error comparator without falsely treating D as exact state. With nonlinear process noise the likelihood generally requires latent-state integration; the present fit is not that solution.

## E23. Connected equivalence under three policies [TESTED]

`nonlinear_force_gauge_v9.py` saves 16 independent seeds per policy, each 8192 steps: iid open loop, feedback `u=-.5*tanh(D)+z`, and that feedback with an independent probability .2 branch setting u=0. z is Gaussian SD .7; the zero-branch design is not an energy-matched superiority contest. Seeds `20261020+1600000+1000*policy_index+i`, i=0..15. Each pair shares exogenous streams only for a simulation equivalence audit; no learner receives paired-world data or true forces. Position is explicitly accumulated, and shared sensor errors imply the same position/observation difference. The first measured D uses previous observation zero, a declared initialization convention.

{{GAUGE}}

Nonzero total-force differences are retained alongside negligible command/position differences. Ordinary zero commands do not break the equivalence. The original saved v9 pilot remains unchanged as preliminary history and was independently reconstructed from an eliminated observable recurrence; the released code adds explicit positions, replications and physical-removal semantics.

## E24. Clean recovery and limited excitation [TESTED]

`nonlinear_identification_v9.py` tests c=0,.01,.2 crossed with command SD .03,.3,1.2, with 12 independent 4096-command sequences per cell. Seeds `20261020+1500000+10000*c_index+1000*amplitude_index+i`. Exact positions are differenced; 4094 regression rows are retained, using t=1..4094. There is no process noise. The exact tanh scale and complete candidate vocabulary are supplied.

{{CLEAN}}

All 108 clean coefficient vectors are saved, including undefined c=0 label ratios as null rather than successes. These are arithmetic/excitation checks, not evidence that finite-noise labels are reliable.

## E25. Noisy position and unseen command amplitude [TESTED]

Six conditions c=0,.02,.2 crossed with command SD .08,.7; 12 independent datasets each. Seeds `20261020+1500000+100000+10000*c_index+1000*amplitude_index+i`. Each has 512 open-loop commands, 513 noisy positions, iid position-sensor SD .2, **zero process noise**, and known x0=v0=m0=0. Every estimator receives the same observations and commands. No hidden states/noise realizations enter fitting.

Three conventional estimators:

- Unrestricted OLS on the five noisy-difference regressors from F42.
- Structured one-step least squares with the four physical parameters, same tanh family, bounds and initial starts as output error, but using noisy differenced positions as regressors.
- Raw-position output error in the four-parameter family. Exact forward parameter sensitivities are implemented and independently checked by finite differences.

Both structured methods use rho/lambda in [.05,.95], c in [0,.4], beta in [.3,2], three fixed starts `(.7,.3,.1,1)`, `(.3,.7,.02,1)`, `(.8,.6,.3,1.4)`, and at most160 function evaluations per start. These bounds keep the passive velocity map contractive and the actuator recursion stable, even when an estimated c exceeds rho; integrated position is not contractive. Selection uses only minimum training SSE. Every optimizer record and every seed is retained. Convergence is not a global-optimum certificate; the methods minimize different objectives. Unrestricted OLS additionally lacks parameter ties/stability bounds, so its comparison conflates observation treatment with those restrictions. The matched structured comparator controls those family/bounds differences.

After fitting, two fresh 256-command sequences use either training amplitude or SD .9. Both tests start from known zero physical states and have no physical/sensor noise; latent velocities are used only to score prediction. This is clean sequence replay, not filtering unknown process noise, goal control, or on-hardware learning. Evaluation seed streams continue after training draws; neither evaluation enters fit selection.

{{NOISY}}

Table reports mean velocity RMSE across 12 runs; all MC SEs, medians, position errors, parameter errors and individual fits are in JSON. Small-amplitude output-error fits can predict the matched regime well yet fail at new amplitudes. At c=.2 and training SD .7, mean absolute lambda error is .013503; at c=.02 it is .124103 despite good sequence prediction. At c=0 no command/position procedure can guarantee correct labels across the equivalent worlds; choosing the true ordering by chance does not establish identification. There is no monotonic precision claim across these small finite sets.

Unrestricted OLS has catastrophic extrapolation in some small-amplitude cases: mean velocity RMSE reaches approximately4.03e73; all finite numbers are retained without clipping, and medians remain available. The table's bounded structured comparator avoids presenting unstable OLS as the only alternative. Output-error fits still use a favorable correct supplied family; no comparison with an optimized neural state-space learner, unknown model order, unknown sensor law or mixed process noise is completed here. Existing methods are sufficient for this restricted experiment; no new algorithm is proposed.

## Independent checks, corrections and existing research

`v9_theory_audit` independently derived F40/F42, challenged noise placement/initialization and supplied the concrete body-versus-motor drag interpretation. `v9_artifact_audit` independently implemented the eliminated recurrence, integrated positions and feedback; tested mapped nonzero states and mechanical-noise counterchecks; audited final identification objectives, Jacobians, forecasts and summaries; and used a separately optimized finite-difference fit as an independent numerical check. The audit explicitly distinguishes fresh derivations/reimplementations from repeated execution. Evidence: `v9_theory_independent.md`, `audit_force_gauge_v9.py`/JSON, `audit_nonlinear_identification_v9.py`/JSON and `audit_v9_artifacts.md`.

**CORRECTED BEFORE RELEASE:** A coordinate redefinition alone does not establish different physical force-disconnection predictions; F41 explicitly supplies different component boundaries. Full regressor rank at c=0 does not identify pole labels. Independent audit comparisons of enormous OLS errors require relative floating-point tolerances; large absolute differences at1e74 are not evidence of meaningful predictive disagreement. A same-family bounded noisy-regression comparator was added to avoid attributing all OLS failures solely to observation-noise treatment. No previous released ledger or pilot was rewritten.

**EXISTING RESEARCH:** [Structural identifiability via symmetries (Yates, Evans and Chappell)](https://wrap.warwick.ac.uk/id/eprint/16955/) studies transformations preserving observations and model structures; our discrete triangular transform is elementary realization algebra. [Schoukens and Ljung's nonlinear identification roadmap](https://arxiv.org/abs/1902.00683) treats model structure, validation, physical versus behavioral models, and why process noise can invalidate an output-error objective. [Tiels, Schoukens and Schoukens](https://arxiv.org/abs/1612.04568) studies nonlinear information for block-dynamics allocation under specified structures. [Janny et al.'s output-error canonical approach](https://arxiv.org/abs/2206.04791) is a richer learned nonlinear-state alternative, evaluated also on real UAV measurements; it is not implemented here and no advantage over it is claimed. These precedents prevent a novelty or general-learning interpretation of the present algebra and parameter fitting.

## Reproduction, drift audit and exact next question

Use the existing local `.venv`; Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No packages installed; owner's PC only.

~~~bash
.venv/bin/python research/nonlinear_force_gauge_v9.py > /tmp/nonlinear_force_gauge_v9_check.json
cmp research/nonlinear_force_gauge_v9_results.json /tmp/nonlinear_force_gauge_v9_check.json
.venv/bin/python research/nonlinear_identification_v9.py > /tmp/nonlinear_identification_v9_check.json
cmp research/nonlinear_identification_v9_results.json /tmp/nonlinear_identification_v9_check.json
.venv/bin/python research/audit_force_gauge_v9.py > /tmp/audit_force_gauge_v9_check.json
.venv/bin/python research/audit_nonlinear_identification_v9.py > /tmp/audit_nonlinear_identification_v9_check.json
~~~

The fundamental uncertainty resolved is whether adding nonlinear passive curvature is enough to justify physical modular attribution. It is not; it resolves labels only relative to a restricted class, and finite noisy data can leave even that identification weak. This matters for transfer to changed bodies/actuators, but more variants of the same ambiguity would be research drift. We still have no general learner that discovers its function vocabulary, chooses adequate experiments, and reliably acts toward changing goals.

**Immediately next:** Construct a learner that acquires useful shared command consequences from noisy experience and uses them for reach-and-stop goals while retaining unresolved component explanations. Compare ordinary-command tasks against a truly new component intervention; assess whether selecting an additional experiment based on predictive disagreement improves task transfer over an equally budgeted passive experiment, with strong conventional identification/planning comparators. Do not score a learner by arbitrary hidden labels when the task depends only on the common behavior. Derive the information available under each experiment before implementation, and reject an active design if a simple established method suffices. No further gauge variant is needed to establish this version's point.

The broader Goal is not achieved. Publication is followed immediately by the next investigation.

---

# Historical archive: Complete original v8 ledger, including all previous versions

'''
t=gauge['component_removal']
doc=doc.replace('{{REMOVAL}}',f"The retained eight-command preparation yields common velocity {t['initial_common_velocity']:.6f}. Over the next12 removal steps the velocity-response distance is {t['response_velocity_rmse_distance']:.6f}, giving expected-RMSE lower bound {t['two_world_minimax_expected_velocity_rmse_lower_bound']:.6f}. Both full response vectors are saved.")
rows=['| Policy | Max command difference | Max position/observation difference | Max gauge error |', '| --- | ---: | ---: | ---: |']
for name,data in gauge['connected'].items():
    values=[max(r[k] for r in data) for k in ('max_command_difference','max_position_observation_difference','max_force_gauge_error')]
    rows.append('| '+name+' | '+' | '.join(f'{v:.3g}' for v in values)+' |')
doc=doc.replace('{{GAUGE}}','\n'.join(rows))
maximum=max(r['coefficient_max_error'] for r in ident['clean'])
ratios=[r['singular_values'][-1]/r['singular_values'][0] for r in ident['clean'] if r['command_sd']==.03]
doc=doc.replace('{{CLEAN}}',f"Maximum coefficient error is {maximum:.3g}. At command SD .03, smallest/largest singular-value ratios range {min(ratios):.6g} to {max(ratios):.6g}, versus substantially better conditioning at larger amplitudes. Exact recovery under negligible roundoff does not remove finite-noise weak identification.")
rows=['| c | Train SD | OE matched | Structured matched | OE SD .9 | Structured SD .9 |','| ---: | ---: | ---: | ---: | ---: | ---: |']
for row in ident['noisy']:
    s=row['evaluation_summary']
    values=[s[regime][method+'_velocity_rmse']['mean'] for regime in ('matched','amplitude_shift') for method in ('OE','structured')]
    rows.append(f"| {row['c']} | {row['command_sd']} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
doc=doc.replace('{{NOISY}}','\n'.join(rows))
assert '{{' not in doc
out=root/'first_principles_robot_learning_ledger_v9.md'
if out.exists():raise SystemExit('Refusing to overwrite released v9')
out.write_text(doc+prior)
assert out.read_text().endswith(prior)
print(out)
