#!/usr/bin/env python3
"""Publish verified continuous-history milestone, preserving complete v14 bytes."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
target=root/'first_principles_robot_learning_ledger_v15.md'
assert not target.exists(),'Released ledgers are immutable; do not overwrite.'
summary=json.loads((root/'continuous_history_v15_results.json').read_text())
audit=json.loads((root/'audit_continuous_history_v15_results.json').read_text())
assert audit['status']=='INDEPENDENTLY VERIFIED EIGHT-REPLICATE CONTINUOUS-HISTORY EXPERIMENT',audit.get('status')
for name,digest in audit['artifact_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for row in summary['inputs']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256']
    assert audit['artifact_sha256'][row['file']]==row['sha256']
assert audit['artifact_sha256']['continuous_history_v15.py']==hashlib.sha256((root/'continuous_history_v15.py').read_bytes()).hexdigest()
assert audit['artifact_sha256']['continuous_history_v15_design.md']==hashlib.sha256((root/'continuous_history_v15_design.md').read_bytes()).hexdigest()
prior=(root/'first_principles_robot_learning_ledger_v14.md').read_text()
s=summary['summary']
rows=['| Ordinary summary | One-step true-position MSE | Recursive1..10-step MSE | Sustained position MSE | Late position RMSE | Late instantaneous velocity RMS | Precision goals /96 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for key,cell in s.items():
    vals=[]
    for m in ('one_step_true_position_mse','ten_step_recursive_true_position_mse','position_mse','late_position_rmse','late_velocity_rms'):
        vals.append(f"{cell[m]['mean']:.8g} +/- {cell[m]['se']:.3g}")
    rows.append('| '+key+' | '+' | '.join(vals)+f" | {cell['precision_success_total']} |")
doc=r'''# First-Principles Robot Learning: Living Research Ledger v15

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited continuous coupled learning from noisy position/action history, conventional sufficiency and limits of history selection. V14 was published and verified over existing SSH origin/main at `93262d83e99c113b3d9542630d8d270fcc106ade`; the audited supplied-history pilot/handoff at `2902a57b3842db97f25d93a6d69acb3044897f51` preceded this investigation. Complete v14 and every earlier ledger follow verbatim. The broader general-learning Goal remains ACTIVE.

## Principal result and scientific boundary

**TESTED:** Conventional learned delay relations can acquire useful information from continuous moving two-joint experience and use it in sustained new-goal feedback, without supplied exact current pose, rest resets at goal changes, true velocity, acceleration, mass or inertia. Every ordinary method receives the same noisy angle/issued-action experience and uses the same bounded receding-horizon planner. Current-position-only prediction/control fails badly in this declared family. A rich causal history and a conventional polynomial-compressed history support useful control.

**TESTED NEGATIVE / LIMITED:** Selection from a supplied history menu is not discovery of a universal state language. The selected raw delay and a fixed-rich raw delay must be compared explicitly; selected delays do not automatically improve over the fixed-rich alternative. A conventional derivative/filter relation is a serious successful explanation, not an obsolete baseline to dismiss. Small next-step or average position error does not guarantee the joint position-and-instantaneous-velocity precision criterion. Numerical optimization success does not certify the physical model or a globally optimal action.

**EXISTING RESEARCH:** Position-history forward learning and physical action optimization already appear in Dalla Libera et al.2020; Romeres et al.2018 learns inverse relations/history features on robot data. No novelty, superiority to those complete methods, general robotics capability or general intelligence is claimed. The advance in this research lineage is constructing/auditing sustained coupled learning after removing v14's exact rest/pose preparation, while identifying what this does and does not establish.

## F55. A missing distinction need not be called velocity [PROVEN UNDER ASSUMPTIONS / CORRECTED]

The same current pose and goal can follow opposite moving histories. A shared pose-only action cannot condition on that distinction, whereas past positions can contain it. The supplied-history pilot uses the same horizontal conservative two-link mechanics as v14:

~~~text
qdot=v, M(q2)*vdot=u-h(q2,v)
M=[[3+2cos(q2),1+cos(q2)],[1+cos(q2),1]]
h=sin(q2)*[-(2v1v2+v2²),v1²], |u_j|<=1.
~~~

Let fplus(u),fminus(u) be exact endpoints from identical q and opposite v. With midpoint m and half-separation d, paired squared return error is exactly ||m-q||²+||d||². Time reversibility and smoothness on compact bounded neighborhoods give m-q=.5T²M^-1(u-h)+O(T4), d=Tv+O(T3), uniformly over bounded u. Thus leading paired risk is T²||v||²+O(T4). This does NOT give an unqualified finite-T global lower bound or prove zero torque optimal in the actual nonlinear plant. The earlier display omitted the common Coriolis term: the exact identity T²||v||²+.25T4||M^-1u||² applies only to its declared ballistic/no-Coriolis surrogate. The correction and stored field rename remain in source/reports.

Two exact unwrapped nearby poses and previous issued action locally determine motion for sufficiently small sampling time: derivative of previous pose with respect to current v is -dt*I+O(dt²), invertible by the inverse-function theorem under smooth bounded conditions. This is local, not a global noisy-history embedding theorem. Sensor noise, wrapping, actuator memory and approximation can alter sufficiency. Longer chosen history can mainly denoise rather than reveal a further independent physical variable. No named state is logically prescribed as a universal learning foundation.

## E36. Audited supplied-history instrument, not the learning milestone [TESTED]

`history_context_v15_pilot.py`/JSON constructs12 worlds: two current poses and three opposite-velocity pairs per pose. Four legitimate zero-torque past positions at times-.2,-.15,-.1,-.05 are generated by backward/forward verification; noisy past encoders have SD1e-4. Ordinary actions use supplied exact current q, a supplied anchored slope, and one of8 previously learned broad-data v14 inverse relations. The goal is return to current q after one .2sec pulse; all trajectories end moving. There is no new training or learned history selection here.

Eight parent fits x12worlds x16fresh noisy histories gives1536 rows/4608 method pulses. Mean squared endpoint error: zero-command descriptive reference1.666670962e-5; learned relation plus supplied slope2.201255135e-8 +/-3.714201761e-9 SE across8parent fits; privileged exact-state endpoint feasibility optimizer1.47272254e-30. Zero is not globally certified optimal among nonlinear pose-only policies. The slope has conditional noise covariance1.3333333e-7 I and a leading curvature bias; it is not exact inferred velocity. This instrument identifies the question without claiming the supplied slope was discovered.

The first run's additive seed base collided with24 earlier roles (8v14 acquisition+16v12 heldout); complete excluded source/results remain as `history_context_v15_seed_collision_pilot*`. The canonical stream range50,020,261,020..50,027,371,035 is disjoint from earlier production. Independent theory/artifact implementations check48 backward states,12 forward returns,12 oracle witnesses,1536 slopes/actions and4608 pulses. State discrepancies<3e-15 and reproduction is byte-identical. Per run4812ODE calls/958.8 absolute simulated seconds/12288 past scalar reads; excluded run adds the same development resources, not fresh statistical evidence. See both history-context audits and theory report.

## F56. Conditional prediction is useful but not equivalent to useful control [DERIVED / EXISTING RESEARCH]

For any fixed causal forecast f(H_t,u_t), fresh independent zero-mean encoder noise e_next of covariance sigma²I gives

~~~text
E||o_next-f||² = E||q_next-f||² + 2sigma².
~~~

Predicting an increment and restoring current o is algebraically the same next-observation loss. Current sensor noise is shared between conditioning inputs and increment targets; this does NOT yield unbiased physical-parameter regression. Validation selection remains optimistic; use separate whole-trajectory tests. For nested information summaries S contained in H and unrestricted squared-loss Bayes predictors, risk reduction is E||E[Y|H]-E[Y|S]||². Finite ridge model classes/selection need not improve monotonically with history. These are standard conditional-expectation facts, not a new universal learning principle.

Previous commands matter even with ideal direct actuation. An exact constant-inertia ZOH double integrator satisfies q_next=2q-q_prev+.5dt²B(u_prev+u_current). Omitting previous commands would handicap a history baseline. Exogenous excitation is used rather than a hidden-state feedback collector whose current action could encode omitted motion. Full scalar action support still does not establish local joint history/action support.

Recursive mean substitution in a nonlinear noisy-history predictor is certainty equivalence, not a Bayes-optimal future observation distribution. Test recursive forecasts separately from teacher-forced next steps. The MPC motion penalty uses step-average predicted displacement/dt, whereas stopping is scored using external instantaneous velocity; these are different quantities. A good one-step mean or lower position cost cannot certify station keeping.

## E37. Same-experience continuous learning/control comparison [TESTED]

Frozen exploratory design/source/results: `continuous_history_v15_design.md`, `continuous_history_v15_pilot.py/.json`. One acquired fit uses48training/12validation/12test trajectories of64commands, dt.1sec, noisy unwrapped joint positions SD.001, hidden initial v uniform[-.15,.15]^2 and q uniform[-pi,pi]^2. Exogenous torque uniform[-.35,.35]^2 is held two samples. All candidate rows start at8 so memory lengths share training/validation targets. This pilot selected rawH4 and polynomialH4; each met6/6 goal precision checks, fixedH1 met2/6, linearARX3/6 and pose-only0/6 on two engineering starts. Executed actions often exceeded the pilot's acquisition amplitude; the pilot is exploratory, not retroactively confirmatory.

Fresh protocol was appended BEFORE fresh outcomes:8 independent acquired datasets, replicate IDs100..107; each48fit+12val+12test complete independent96-command trajectories. Even-index acquisition trajectories use amplitude.35, odd-index amplitude1, balanced in every split. This increases duration as well as command support; it is not an isolated amplitude ablation. There is no stabilizing collector, true-state pruning, hidden velocity target or removed failure. Role-tagged SeedSequence paths `[20261020,15,610,replicate,role,item]` replace collision-prone small additive version offsets. Sensor paths are shared across methods within each dataset/task, giving paired comparisons; they are independent across acquired-data replicates and across tasks.

Ordinary supplied menus:

- Current-position-only nonlinear predictor, ridge selected withinH0.
- Fixed one-delay nonlinear predictor, ridge selected withinH1.
- Fixed richH4 nonlinear predictor, ridge selected withinH4.
- Raw nonlinear delay predictor, H0/1/2/4 and ridge chosen by heldout next-observation error.
- Full MIMO linear ARX, H1/2/4 selected, a contextual-capacity diagnostic.
- Conventional degree2 causal polynomial compression of3/5/9 measured positions (slope/curvature arithmetic supplied), with learned nonlinear next-position relation and all past issued commands; history/ridge selected from the same experience.

Nonlinear features are a symmetric13-term Fourier context of BOTH current angles (first/second harmonics and first-harmonic pair products), tensor linear causal lag/command features plus quadratic terms of the most recent lag difference or polynomial slope. Action affinity, angle periodicity, clock, joint coordinates and this menu are supplied priors. Raw differences are an invertible rewriting of the given delays when current o is retained. Fit-only RMS scaling and ridge1e-8/1e-5/1e-2 produce30 candidate fits per dataset; the fixed-rich comparison reuses existing fits. No mass, inertia formula, relevant-q2-only feature, true v/a or future sensor noise enters ordinary prediction/selection/control. True states are retained for diagnostic scoring only.

All six controllers use the same local bounded TRF residual planner, horizon15atdt.1, current goal only, shared position/motion/action residual weights1/.3/.003,40 function-evaluation limit and shifted warm starts. No body/history/plan reset at goal changes. A finite improved optimizer result is retained even when termination is unsuccessful; statuses and projected gradients are saved. Nonfinite/worse results revert to the warm plan. No global optimum or closed-loop stability certificate is claimed. Future actions remain available after the scored window; there is no hidden actuator cutoff.

The current-position-only MODEL does not make its entire MPC policy strictly memoryless: the shifted warm plan retains prior history and can affect local optimization. Its observed failure therefore does not establish an information-theoretic lower bound on every current-position policy, or on controllers with other memory channels. The separate opposite-motion instrument concerns an explicitly shared action; do not conflate that derivation with this implementation.

Four new starting contexts q0=(-2,-2.1),(-.5,-.7),(.8,.7),(2,2.1), each with hidden v=(.04,-.02), receive8zero-torque warm-up samples and then three goals q0+(.2,-.15),q0+(-.15,.15),q0+(.1,.1), each6sec. The controller sees only the currently announced goal and noisy position/action history. Exact initial pose is not separately supplied. Goals use world coordinates and therefore constrain initial-pose information, but do not reveal evolving true motion. Precision requires BOTH coordinates' |position error|<=.03rad AND |instantaneous velocity|<=.03rad/sec at EVERY final1sec sample of each goal period. Truth-based precision is evaluation, not a learner loss or selection signal.

Prediction tests use separate logged future commands; recursive diagnostics average physical error over all predicted horizons1..10, not only the tenth endpoint. Controls average metrics over four fixed contexts FIRST, then uncertainty across8independently acquired fits;96 goal checks are clustered consequences, not96 independently learned bodies.

{{TABLE}}

{{SELECTION}}

The fixed one-delay, rich-delay, selected-delay, linearARX and polynomial methods
all have overall position MSE near.008, yet precision counts are45,79,76,16and69
of96 respectively. The linear model's late velocity RMS is.07458 versus.00936 for
selected history despite similar average position error. This reinforces the
v13 warning: average position performance is not stopping capability. Selection
versus fixed-rich precision differs by-.03125 +/- .03499 cluster SE; selection
versus polynomial by+.07292 +/- .06385. These finite comparisons do not establish
a reliable superiority of selected history. Rich-minus-selected late-velocity
RMS is-.001006 +/- .000736; fixed-rich should not be described as significantly
better from these descriptive summaries. One selected fit choseH1 and reproduced
the fixedH1 controller exactly, with its poorer precision. Do not rescue the claim
by selecting history from the heldout control outcomes.

All raw trajectories, coefficients/scaling/validation scores, predictions, controls, plans, solver statuses, metrics, acquisition speed/angle/action coverage and paired cluster differences remain in `continuous_history_v15_rep100.json` through`rep107.json` and aggregate`continuous_history_v15_results.json`. Do not interpret scalar command extrema or broad q coverage as a certificate that every planned local speed/history/action combination is supported. The body is conservative and random torques produce broad motion; no high-speed records were silently pruned.

## Feasibility, independent audits and numerical limits

**PROVEN UNDER ASSUMPTIONS:** Slow continuous-input references through these three goals exist within torque bounds, including the moving first phase. Globally ||M||inf<=7, ||h||inf<=3||v||inf², ||M^-1h||inf<=11||v||inf². During.8sec zero-torque warmup fromV=.04, comparison gives V<=.0617283951 and joint drift<=.0394422348. A6sec quintic Hermite reference from that actual warmup q/v to firstgoal/rest has bounded speed.136554094 and acceleration.130993159, hence required torque<=.972893174<1. Subsequent6sec rest-to-rest quintics over maximum joint displacement.35 need<=.42881. This is a continuous-torque existence proof, not an exact.1sec ZOH endpoint certificate, a guarantee for arbitrary failed recovery states, or knowledge supplied to the learner. Derivation is independently checked in the theory report.

Independent derivative audit implements all10feature-family/history combinations with separately written float64 Torch automatic differentiation, comparing4and15step recursive Jacobians against source analytic derivatives: max5.81e-15. AST identity confirms unchanged pilot/production formulas. Independent artifact audits separately reconstruct features, ridge stationarity/selection, forecast recursions, warm/selected plan costs/projected gradients, RNG/data/metrics and mechanics using Hamiltonian coordinates. Read their actual reports for precise scope/errors, not merely file existence. The builder requires completed current audits and exact result hash coverage.

The failed pose-only pilot reaches high velocities and is numerically sensitive over long global replays: adaptive independent Hamiltonian Radau local-step discrepancy2.61e-11, but global18.8sec difference8.78e-7; coarse RK32/64 difference1.54e-5. Normal pilot controls replay within1.80e-11 and acquisition within2.44e-12. This qualifies numerical trajectories, not evidence of competent pose-only control. Production audit records its own full comparison bounds; do not copy the pilot bound as a universal solver guarantee.

Fresh production independently checks all55296acquisition intervals,36096control
intervals,240fits and34560selected plans/full complex-step gradients. Maximum fit
stationarity residual2.00e-15 and projected-gradient discrepancy9.64e-14; ordinary
non-fast control global physics discrepancy2.98e-11, acquisition2.61e-9. Fast failed
pose-only cases use adaptive Hamiltonian Radau local checks(max2.86e-11), with
global sensitivity as large as2.13e-7 rather than machine precision. All8optimizer
unsuccessful terminations belong to the pose-only model.608fresh SeedSequence
states have no collisions with74pilotpaths or27929earlier integer-seed states.

A separate post-pilot causal tanh-PD reference freezes nine common-joint gain/window settings before its own outcomes and reports ALL18episodes on original pilot tasks. It receives no true motion/model, uses a causal polynomial position/slope estimate and bounded torque. Precision counts out of6: windows3/5/9 with gains2/4/8 give0,1,2 /0,1,2 /0,0,0 respectively. No winner is selected as a new primary method. This is an exploratory diagnostic, not an optimized matched fresh-production baseline or proof that learning is necessary. The successful conventional polynomial+learned relation already provides a serious same-experience alternative.

## Resources, reproduction and preserved errors

Fresh production:576acquisition/validation/test trajectory resets,55296real simulator intervals/5529.6simsec,111744scalar encoder reads. All six learned methods together192control episodes,36096intervals(including warmups)/3609.6simsec,72576scalar reads. Thus768acquisition/control starts, no goal-change reset. Rich/selected identical cases are repeated deterministic evaluations, not independent learned bodies. Offline30ridge fits/dataset=240; forecast96768predicted transitions; canonical derivative checks816residual evaluations/3264predicted transitions, in addition to optimizer calls reported in aggregate. Simulated plant/virtual planning/reference mathematics are distinct resources. Wall timings are retained, not a runtime advantage claim.

Exploratory continuous pilot additionally72acquisition streams/4608intervals/9360scalar reads plus10control episodes/1880intervals/3780reads and30fits. PD bank adds18episodes/3384intervals/6804reads. Supplied-history pilot and its excluded collision run have the separate budgets above. Independent audits/refinements and deterministic reproduction repeat computations, not fresh independent training evidence. No hardware experiment is claimed; existing owner PC/project environment is used.

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/continuous_history_v15.py --replicate 100 --output-prefix /tmp/continuous_v15_rep100
.venv/bin/python research/summarize_continuous_history_v15.py
.venv/bin/python research/audit_continuous_history_v15_derivatives.py
.venv/bin/python research/audit_continuous_history_v15.py
~~~

Reproduce other acquired fits with replicate101..107, preserving their role paths. JSON wall timing is nondeterministic and must be excluded from exact numerical-payload comparison; actual values/paths/coefficients/controls remain deterministic in the fixed single-thread environment. Keep parent v14 physics source available. Frozen pilot source is separate; do not run production defaults and overwrite pilot evidence. Complete excluded seeds, theory corrections, inadequate/failed controls and independent reports are retained rather than silently repaired.

Primary sources, read for this investigation: [Romeres et al.2018](https://arxiv.org/pdf/1809.05074), position-history inverse learning/features; [Dalla Libera et al.2020](https://arxiv.org/pdf/2002.10621), derivative-free GP forward models and physical trajectory optimization. Our finite Fourier/ridge receding-horizon implementation does not reproduce their GP algorithms or establish superiority. Standard NARX/conditional prediction and polynomial differentiation are accepted tools, not discoveries. See the independent design/theory notes for competing formulations and source-specific limitations.

## Direction reassessment and exact continuation

What changed about general learning? Experience can select/use a useful retained distinction in this coupled continuous task without being handed true velocity or exact current pose. Conventional delay/filter methods suffice under substantial supplied vocabulary/coverage/planning. This resolves a construction gap in the lineage; it does NOT show autonomous invention of representations, competitive real robotics, cross-body transfer or broad intelligence. Choosing from a lag menu is a modest conventional mechanism, and fixed-rich history must be credited when equally capable.

The broad offline independent data, fixed body, ideal torque, known joint-angle semantics, supplied periodic basis/action affinity, short lag menu and hand-specified task/planning loss remain large assumptions. Adding more kernels/lags/solvers would increasingly study a specialized surrogate. End that branch here unless a decisive observed failure demands it.

**Immediately continue:** The consequential unresolved question is whether ongoing ordinary experience can revise useful behavior when old knowledge stops supporting a decision, under a limited interaction budget and without new calibration resets/change labels. First test frozen knowledge, ordinary task-driven continual updates and serious robust conventional feedback. Do not assume explicit change detection or active probing is necessary; unchanged observations at rest may make a physical change unknowable and irrelevant until an action excites it. Probes must earn their real tracking/time/resource cost against ordinary updates/equal-cost random alternatives. A genuinely different route challenges dependence on supplied encoder coordinates, with task/noise transformations handled fairly. `experience_revision_v16_question.md` and independent direction critique preserve these possibilities without mandating architecture or a manufactured sign-flip challenge. Derive the information/feasibility problem, then consult serious adaptive/dual-control literature and build the smallest decisive test. Continue the ACTIVE general-learning Goal after SSH publication; no release is a stopping condition.

---

# Historical archive: Complete original v14 ledger, including all preceding versions

'''
doc=doc.replace('{{TABLE}}','\n'.join(rows))
selection='Selected raw histories across8fits: '+str(s['selected_delay']['selected_h'])+'. Fixed-rich histories: '+str(s['rich_delay']['selected_h'])+'. '
selection+='Polynomial histories: '+str(s['polynomial']['selected_h'])+'. '
selection+='Total unsuccessful optimizer terminations by method: '+str({k:v['solver_failures'] for k,v in s.items()})+'. '
selection+='These statuses do not imply every unsuccessful result was discarded; source retains finite improved results.'
doc=doc.replace('{{SELECTION}}',selection)
target.write_text(doc+prior)
assert target.read_text().endswith(prior)
print(target)
