# First-Principles Robot Learning: Living Research Ledger v16

**Date:** 2026-10-20, Europe/Berlin. **Status:** acquired Cartesian response variation, purposeful evidence, and persistent local goal control; independently audited conventional constructions and explicit limits. V15 was SSH-published and remotely verified at `e64e48c72bb9bcde3a85257d9d54ee843c6394ac`; audited post-v15 payload evidence/question handoff at `2023af57f978ff0baab22e7a144a44f7cce6c095` preceded this investigation. Complete v15 and every earlier ledger follow verbatim. The broad general-learning Goal remains ACTIVE and unresolved.

## Principal finding and boundary

**TESTED:** Unlabeled action/Cartesian experience can acquire alternative local responses and useful covariance structure, then use a small amount of new evidence for bounded physical decisions while retaining knowledge and motion through six successive goals. Ordinary learners receive no joint encoders, evolving true velocity, mass, branch label or true response matrix; initial Cartesian position/rest are supplied. General Gaussian, matched Gaussian, finite mixture and isotropic diagnostics share the acquired records, camera/action clock, loss and correlated continuation solver. Their full failed cases remain.

**PROVEN UNDER ASSUMPTIONS:** More stationary Cartesian history cannot resolve the two present hidden preparations. Under the declared local coefficient model, single-action population marginals also need not identify coherent cross-action consequences. Fixed open-loop quadratic decisions need only first/two moments; anticipating evidence-conditioned decisions can depend on more of the prior law. These are separate statements. The general covariance counterexample does NOT prove grouped experience is necessary for this arm's restricted rank-one variation.

**LIMITED / EXISTING RESEARCH:** This is conventional hierarchical regression, joint Gaussian/mixture conditioning and a small finite-candidate dual-control calculation, not a new universal architecture. Exact initial Cartesian preparation/rest, known noise/timing, persistent grouping and a constant-response vocabulary are supplied. The articulated ODE generates actual nonlinear motion, but millimeter goals remain near one preparation. Two-stage feedback is repeated without goal resets; it is not arbitrary per-pulse feedback or stationkeeping. No modern robot-learning benchmark superiority, real-time feasibility or general intelligence is established.

## E38. The payload premise failed before adding adaptation [TESTED NEGATIVE]

After v15, `payload_revision_v16_pilot.py` attaches distal mass1->2 at the first6sec goal switch, preserving q/v through an external matched-velocity load. The change is not announced to the frozen controller. The single parent v15rep100richH4 and entire nine-setting causal tanhPD bank are retained; no online learner or active probe is added. Four fixed contexts and paired fresh role-tagged sensor paths give80episodes/15,040intervals. This instrument still uses noisy JOINT positions, not Cartesian sensing.

Frozen MPC passes the same7/8postevent precision windows with and without the load. Postevent positionMSE changes .0107440728->.0124800700 and late velocityRMS .0079659600->.0068941482. This does not prove noninferiority or universal robustness: it is one learned fit/four contexts. It does reject the premise that this specification already requires adaptation or probing. No disturbance was increased until a desired method won.

Independent variable-mass Lagrangian/Hamiltonian audits check torque/work, external momentum/energy exchange, unit-parent recovery, all80episodes,1,440MPCgradients and12,960PDactions. Maximum replay discrepancy8.77e-11; common preevent states and first loaded actions match exactly. Sources, actual results, audit and rejected premise remain. Goal/change clocks coincide and cannot later be described as an unknown-time detector experiment.

## F57. An intervention can expose what more passive history cannot [PROVEN UNDER ASSUMPTIONS]

For two unit links p(q)=(cos(q1)+cos(q1+q2),sin(q1)+sin(q1+q2)). Preparations qA=(-phi/2,phi),qB=-qA,0<phi<pi have the same p0=(2cos(phi/2),0). At zero motion/torque their Cartesian histories are exactly identical for any duration, including with the same independent noise law. Offline population experience cannot identify the CURRENT preparation from that stationary record alone.

The inertias agree because M depends on cos(q2). With B=J*M^-1, at phi=pi/2,

~~~text
BA=[[1,-3],[1,1]]/(2sqrt(2)),
BB=[[-1,3],[1,1]]/(2sqrt(2)).
p(T)-p0=.5*T²*B*u+O(T4), starting at rest.
~~~

Each B is invertible, while its equal-preparation mean has a zero leading radial row. Averaging can erase an attainable direction. The EXACT finite-time mean need not have a zero radial response: curvature contributes higher-order effects. Nor does this prove every common open-loop sequence fails. Tangential goals admit a common leading action u proportional to(3,1); resolving the hidden distinction is useful only when it changes relevant available decisions enough to repay the experience. See `cartesian_branch_v16_theory_independent.md` for derivation and counterexamples.

## F58. Marginals are not always a reusable relationship [PROVEN UNDER ASSUMPTIONS / DESIGN CORRECTED]

Let b=vec_row(B), H(u)=I2 tensor u^T and A=[[0,1],[-1,0]]. For0<epsilon<1, the two positive-definite Gaussian covariances

~~~text
Gamma_plus/minus = I4 +/- epsilon*[[0,A],[A^T,0]]
~~~

give H(u)*Gamma*H(u)^T=||u||²I2 for EVERYu, hence identical single-action marginals, even with the same sensor noise. Their cross-action laws differ: Cov(B e1,B e2)=+/-epsilon*A. An observation under one action therefore has different implications for another. No amount of independent one-pulse population data identifies this general missing covariance direction.

Both independent auditors raised this BEFORE Cartesian acquisition. The design was repaired to group two consecutive actions under the same hidden context, without an intermediate reset. From supplied p0/rest, normalized endpoints satisfy

~~~text
(p1-p0)/T² = B*(.5u0),
(p2-p0)/T² = B*(1.5u0+.5u1).
~~~

Their2x2 effective-input determinant is .25det(u0,u1), nonzero almost surely for independent continuous commands. The resulting full group exposes all coefficient coordinates under the local model. Nearly collinear groups remain statistically weak; invertibility is not finite-data precision. The code fits whole group likelihoods/moments rather than treating inverted noisy coefficients as truth. This general theorem does not establish grouping necessity in the special arm, whose leading variation can already be found from suitable single-action second moments under extra restrictions.

## F59. Useful conditioning must retain motion/response dependence [DERIVED / EXISTING RESEARCH]

Use normalized z=[s,w,b],s=(p-p0)/T²,w=v_cartesian/T, with one fixed p0. Under the supplied constant-B model, known issued u gives s'=s+w+.5Bu,w'=w+Bu,b'=b. This is linear in the augmented quantity. Gaussian conditioning is exact WITHIN that model; mixtures condition full components and their weights. Real finite nonlinear flow can invalidate the model. No process noise/drift repair is inserted in this study.

After the first pulse, with remaining controlsU=(u1,u2), terminal s=s_now+2w+B[1.5I,.5I]U and w_terminal=w+B[I,I]U. Expected quadratic loss depends on G=E[B^TB], h_d=E[B^T(s+2w-g)] and h_w=E[B^Tw]. The cross-covariance terms are necessary. The state-uncertainty constant E||s+2w-g||²+.25E||w||² is irrelevant to the INNER optimizer but necessary when comparing FIRST actions. A noisy camera reading cannot also be inserted as an independent exact state.

Positive effort weight makes the remaining boxQP strictly convex. The solver checks KKT and enumerates all81faces when the unconstrained four-coordinate solution is infeasible. Cautious three-pulse plans use the analogous six-coordinate problem. Mixture and FULL-moment matched Gaussian have identical fixed open-loop quadratic decisions; shape matters through conditioning or its anticipation. The outer first-action menu is a heuristic, not a globally optimal dual controller. Independent augmented sigma-point/Torch/direct raw-second-moment checks establish the formulas and active-bound gradients; all actual filters/continuations are independently replayed.

## E39. Acquired priors and preserved numerical failures [TESTED / CORRECTED]

The exploratory corpus contains256fit/64validation/64test independent two-action groups atT=.1, cameraSD=.0002, |u_j|<=.5. A group chooses one hidden rest preparation, independent uniform commands and noisy Cartesian endpoints. No q/v/B/branch labels enter ordinary fitting. The mean/full covariance and K1/2/3mixtures are learned from raw groups. Seventeen generic EM starts, three moment/PSD Gaussian choices and two full marginal-likelihood refinements are retained. Validation, not control outcomes, selects candidates. A matched Gaussian isolates prior shape from differing fitted moments. Gaussian factor initialization uses eigenfloor1e-8 to avoid trapping a covariance factor in a deficient subspace. Bounded L-BFGS termination does not establish global maximum likelihood.

The initial moment Gaussian was weaker than a conventional likelihood fit: its directed mean task cost .0741395 falls to .0360809 after likelihood refinement on the SAME acquired records, versus mixture .000947949. All9methods/1,728short-task evaluations and failures remain. The mean-only ablation has physically infeasible first probes in128cases and is not the primary competitor. Heldout mean response errors near.083 do not distinguish coherent prior quality; grouped/conditional likelihood does.

The original GH9x9integration changed some Gaussian action choices under refinement and was rejected. Adaptive Genz-Malik then falsely reported small errors on four of900candidate values: two mixture discrepancies about+.00194528/-.00182176 and two Gaussian discrepancies-.000102805. Independent tighter tensor Gauss-Kronrod21 and scrambled Sobol corroborate the failures. Analytic Gaussian tail bounds remain valid but do not bound cubature error. All18directed choices and all18random-radius choices happened to remain unchanged under complete rescoring.

Every pilot/refinement/adaptive source, design, raw dataset, result and failed audit is preserved separately. Corrected immutable `_gk21.py` batch agrees with all900independently reconstructed scores within9.97e-12; complete controls/priors/fits are exactly unchanged from the adaptive parent. It acquires no fresh evidence. Three pulses with one intermediate read are NOT sustained feedback or a separate learning milestone.

## E40. Persistent Cartesian construction pilot [TESTED, ENGINEERING]

`cartesian_feedback_v16.py` reuses that one acquired corpus and propagates full8D beliefs through six current-goal-only phases without motion/belief resets. Each phase has three pulses: read after first, commit lasttwo, read after third for the NEXTgoal. Two goal sequences, two hidden preparations, two paired fresh noise repeats and eight methods give64episodes/384endpoints. Initial p0/rest are supplied exactly only once; ordinary input thereafter is the actual noisy camera/action history.

Mixture components receive declared plug-in coefficient curvature covariance I_output tensor(R*weightedGram^-1). All17existing fits are reselected using their actual predictive law on validation groups. This is conditional curvature, not full Bayesian assignment/model-error uncertainty. Mixture/generalGaussian/matchedGaussian each use directed/cautious actions, with mixture/Gaussian random-direction references. The extra pilot learned-variation direction menu is preserved but removed from the fresh comparison below.

Pilot mean phasecost/precise endpoints of48: mixture directed .000376472/48, ordinary task .417843872/27, random .084058946/44; generalGaussian directed .034535144/41, task .566171917/19, random .074462837/41; matchedGaussian directed .033992108/42 and task .475099162/21. These are one learned corpus/four independent noise paths, not64independent bodies. All1152physical intervals,384full filter/continuation phases,17prior reconstructions,24nominal nonlinear feasibility trajectories and counters replay independently; maximum Hamiltonian state error8.5e-15 and witness normalized residual difference1.2e-12.

Separate actual-integral checks cover ALL132initial candidates plus360later topthree scores. No checked action ordering reverses, but Genz-Malik false error estimates recur, up to.000102805. Tight split-domain checks confirm them. Retain numerical failures rather than replacing their flags with a blanket audit pass. Production therefore changes numerical integration and requires fresh acquired corpora.

## E41. Fresh acquired-response reuse over uninterrupted goals [TESTED]

Frozen `cartesian_feedback_v16_study_design.md` precedes all fresh replicates100..107. Each independently acquires256fit/64validation/64test groups, refits the same17mixture/fiveGaussian candidates and deploys all10methods on the same physical/sensing/loss contract. No redundant short-task deployment is run on these corpora. Selected mixture component counts are [2, 2, 2, 2, 2, 2, 2, 3]; this is validation behavior, not discovery of a physical number of modes. Eighteen of136EMcandidate fits reach the150iteration limit; rep107's DEPLOYED K3candidate is among them. All other deployed K2candidates converge. These unfinished fits are retained and independently reproduced, not discarded or retuned. All16GaussianMLruns report ftol-success, not global/strict-gradient optimality.

The trace-matched isotropic diagnostic retains the general Gaussian's learned mean and total coefficient variance, replacing Gamma by trace(Gamma)/4*I4. It keeps the same full joint filter, memory, observations and continuation solver. It changes anisotropy, marginal variances and covariance spectrum as well as correlations; it is NOT a pure coherence ablation or an optimized from-scratch baseline. The serious primary Gaussian retains its full learned covariance and conventional updates.

All priors share16uniform directions at radii.05/.1, zero and each prior's own cautious first action:34candidates, common geometric grid plus model-specific cautious action. There is no extra learned-eigenvector action for the structured prior. Random direction is uniform over the same16, with radius chosen by model direction-average. Every method has equal sensor opportunities/clock/bounds, not necessarily equal actual energy. Effort is scored and separately retained.

First-action expectations use gk21 on four fixed standard-normal quadrants, totalatol1e-8/rtol1e-7, then reintegrate the lowestthree at1e-10/1e-9 before selection. Coarse/strict values/errors, points, subdivisions and analytic discarded-tail envelopes remain. Numerical errors are estimated, not certified; outer candidate search is finite. Independent differently partitioned/rule tests check actual fresh integrals, with exact scope recorded in the separate numerical report. These do not prove global optimality or uniform accuracy on every possible belief.

Same two sequences of offsets*L, L=.001meter:

~~~text
A:(1,0),(1,1),(0,1),(-1,1),(-1,0),(0,0)
B:(0,1),(1,1),(1,0),(0,0),(-1,0),(-1,1)
~~~

Each adjacent goal moves one coordinate step. Each phase costs ||p3-p0-goal||²/L²+.25||v3/(L/T)||²+.01sum_all3||u||². Precision requires percoordinate .0003m AND.003m/sec at the ENDPOINT, not a sustained final window. Only the current goal is disclosed. Exact nonlinear nominal zero-first-action witnesses establish their own trajectories' feasibility, not recovery from all states reached by another method. All actions, beliefs, states, camera readings and failed goals remain.

Each table row has384clustered endpoints, from eight separately acquired corpora, two fixed preparations, two sequences and two paired sensor repeats. Means are formed per corpus BEFORE descriptive SE across8corpora. Only32independent paired noise paths underlie the640method episodes. These are not a sample of640bodies or general robotic tasks. First/later goals are separated to expose initial learning cost and subsequent adaptation; full sequence/phase profiles and paired differences are saved.

| Same-data prior/action rule | Mean phase cost +/- corpus SE | First-goal cost | Later-goal cost | Precise endpoints /384 |
| --- | ---: | ---: | ---: | ---: |
| mixture_dual | 0.00842859025 +/- 0.00801 | 0.048574232 | 0.00039946189 | 383 |
| mixture_task | 0.397200451 +/- 0.0357 | 0.4932057 | 0.3779994 | 236 |
| gaussian_dual | 0.024777299 +/- 0.00949 | 0.093811244 | 0.01097051 | 363 |
| gaussian_task | 0.429290411 +/- 0.0293 | 0.49314908 | 0.41651868 | 204 |
| matched_dual | 0.0254258791 +/- 0.00999 | 0.095373346 | 0.011436386 | 363 |
| matched_task | 0.406457877 +/- 0.0345 | 0.49339844 | 0.38906976 | 216 |
| mixture_random | 0.0296628423 +/- 0.0122 | 0.14023351 | 0.0075487086 | 370 |
| gaussian_random | 0.0523004957 +/- 0.0166 | 0.18432533 | 0.025895529 | 346 |
| isotropic_dual | 0.177347389 +/- 0.0103 | 0.41758879 | 0.12929911 | 250 |
| isotropic_task | 0.698767814 +/- 0.0248 | 0.82066025 | 0.67438933 | 106 |

The data support the comparisons in this declared contract. Mixture directed achieves63/64first and320/320later endpoint checks; the general Gaussian52/64first and311/320later, with later cost dropping to.01097051 from first.09381124. The isotropic diagnostic28/64first and222/320later. The full Gaussian therefore already acquires and reuses useful structure, and improves markedly after initial evidence. These results do not establish universal superiority of mixtures or explicit modes: ordinary task actions can themselves gather useful evidence, and distributional shape is only one possible way to preserve useful conditional consequences.

One real rare case dominates the mixture's marginal cost SE: rep105/sequenceA/repeat0/hiddenpreparation1/firstgoal. The first noisy read favors the wrong fitted component with posteriorweight.8760; terminal radial error is-.00175235m, phasecost3.07187341, while it ends nearly stopped. The terminal camera read reverses the component weighting to.999999617on the other component, and all five subsequent phases are precise. This failed first decision is retained; no probe amplitude, model or noise was tuned to remove it. The paired generalGaussian-minus-mixture meancost difference is.01634871+/- .00195621SE across8corpora, despite the mixture's marginal .00801190SE, because both methods share the difficult sensor paths. The isotropic-minus-generalGaussian difference is.15257009+/- .00932290SE. These are descriptive fixed-study comparisons, not universal risk or population-normality guarantees.

Shape and interaction choice can matter with finite noise/budget, but improved precision is not a universal need for discrete physical identification. Retain the covariance-isotropy result with its spectrum/marginal-variance caveat. The full per-corpus and per-sequence profiles, costs, effort and precision are in `cartesian_feedback_v16_aggregate_results.json`; no failure is removed for a cleaner summary.

### Independent verification and resources

Every fresh acquisition group,17EM/fiveGaussian fit candidates per corpus, heldout predictive likelihood, actual filter/continuation/action/score and physical trajectory is independently reconstructed. The auditor derives continuation expectations through full raw8D second moments, not the implementation's expanded formula; Hamiltonian replay and Radau spots/witnesses provide independent physics. Maximum fresh physical state discrepancy1.18e-14, continuation-command discrepancy7.72e-14; independent20-coordinate GaussianMLfinite-difference gradients differ at most6.25e-9. Aggregate means, paired SEs and counts reconstruct to roundoff.

Independent actual-integral checks use gk15 on NINEboxes with edges[-7,-2,2,7], tighter1e-10/1e-9, independently reconstructed joint conditioning/raw8Dmoment QPs and no canonical controller imports. All1440checked integrals agree within1.76e-9:1184initial/256later, all64initial histories' topthree,128later selected/runner histories at prespecified phases2/5 and fixed branch/repeat strata, plus32complete34candidate initial menus on100..103. No checked ordering/radius choice changes. The initial plan to check2176initial integrals was curtailed after this balanced policy-relevant coverage: unchosen K3hypotheticals were expensive and no uniform certificate was claimed. All completed checks remain. Later random-radius averages and complete104..107initial menus are NOT independently reintegrated; rep107's tinyK3selected/nearby forecasts ARE checked. Partial107fixtures are revalidated against final complete tuples before final hash binding. This is finite numerical confidence, not analytic enclosure or global stochastic-control certification. Exact hashes/scopes remain in the independent numerical report. The ledger builder requires completed physical/statistical/numerical audits before writing this file.

Fresh learning acquisition totals3072groups/6144intervals/614.4simsec/18432scalar camera reads, including noisy initial reads although exact p0 is supplied. Fresh control totals640episodes/3840clustered endpoints/11520intervals/1152simsec/15360scalar reads,32paired noise paths/768distinct scalar noise draws. There are1984computed designs,67456initial candidate integrations,5952tighter candidate reintegrations and1,419,473,144hypothetical integrand rows. Additional privileged nominal witnesses require11088plant-flow calls; independent numerical checking adds71,240,234hypothetical rows. These are separate from ordinary experience; exact preparation remains an unpriced supplied resource.

An inherited reporting bug was caught DURING the firstfour runs: their resource dictionary describes control-phase parent reuse/new0 even though the invocation itself generated a separately saved acquired corpus. Freeze initial source and preserve raw results. Laterfour source changes only origin/cost/timing labels, not numerical behavior. Aggregate combined costs are reconstructed from each actual acquisition array. `wall_seconds` originally excludes acquisition; corrected source also names control and total invocation timing. No raw result is silently rewritten. Both executable source versions and the protocol erratum are retained.

The integrator can evaluate hundreds of millions of hypothetical rows per corpus. Planning computation is not charged as physical delay in this simulation. No real-time implementation, latency-aware physical controller or computational superiority is established. More sophisticated approximate planners might reduce that cost, but refining this optimizer is not the next scientific milestone.

## Existing research and reproducibility

Full primary method/experiment sections were inspected in the independent literature review, not just abstracts: [Klenske/Hennig2016](https://jmlr.org/papers/volume17/15-162/15-162.pdf), [Nagabandi etal.2019](https://arxiv.org/html/1803.11347), [Chaumette/Hutchinson2006](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf), [Przystupa etal.2021](https://arxiv.org/pdf/2106.06083), [PETS2018](https://arxiv.org/pdf/1805.12114). The first establishes dual-control/turn-off precedent; the others challenge adaptation, visual-interface and coherent uncertainty choices. None of their full implementations was reproduced. Torque-to-Cartesian acceleration is not a velocity-command visual Jacobian. No claim against complete modern competitors follows from this instrument. Root additionally inspected the joint-state/parameter definitions and nonparametric formulation of the dual-control paper; our exact within-model filter relies on its unusually simple linear-in-augmented-quantity transition.

~~~bash
cd /home/dtdav/dev/learn
# Preserve published artifacts: choose a new output path for reproduction.
.venv/bin/python research/cartesian_feedback_v16_study_initial.py --replicate 100 --output-prefix /tmp/cartesian_v16_rep100
.venv/bin/python research/cartesian_feedback_v16_study.py --replicate 104 --output-prefix /tmp/cartesian_v16_rep104
.venv/bin/python research/analyze_cartesian_feedback_v16.py
.venv/bin/python research/audit_cartesian_feedback_v16_study.py --replicates 100 101 102 103 104 105 106 107
.venv/bin/python research/audit_cartesian_feedback_v16_aggregate.py
~~~

Use existing owner PC/project.venv; no new packages/environments/services were installed. The immutable batch GH/refinement/Genz/gk21 artifacts, persistent pilot, fresh acquired corpora/results, independent executable audits and complete past negative results are durable research memory. Publication uses the existing SSH origin without force-push; remote commit verification is recorded in the continuation handoff after committing.

## Direction reassessment and best next question [OPEN]

This cycle resolves a real uncertainty: a passive missing distinction can require useful intervention, and acquired population variation can let that new evidence support subsequent decisions. It does NOT discover general laws from unrestricted experience. We supplied the context grouping, initial rest and local response vocabulary. Correct ordinary Bayes solves inference/decision once the joint law is supplied; that is a successful conventional explanation, not a deficiency to manufacture away.

The consequential next question is **which relationships in experience remain interchangeable for useful decisions when the preparation, response vocabulary and persistence boundaries are no longer supplied?** Retained structured estimation, episodic conditional completion, generic causal-history learning and direct decision reuse are serious competing formulations. No latent-state, model or policy architecture is logically required by that question. The next test must show a learned relation transferring to actual new interventions/conditions, or falsify its sufficiency, rather than add another coefficient/optimizer menu.

Unknown initial motion is a decisive diagnostic. In the constant-B surrogate, p1=p0+Tv0+.5T²Bu. Choosing vB-vA=.5T(BA-BB)u makes the first observation identical for distinct B. A probe that identifies a prepared response need not identify it without rest. Extra observations can remove this ambiguity; another cadence/rank theorem alone would repeat identification drift. Similarly, a camera-rotation/chirality construction can expose coherent grouped knowledge in a surrogate, but nonlinear leakage and engineered symmetry make it optional, not automatic escalation.

`cartesian_v16_direction_independent.md` and `after_cartesian_v16_question_independent.md` actively challenge the scope and propose observable falsifiers. Do not conclude universality from this local theorem/task, do not increase difficulty solely to defeat a conventional method, and do not indefinitely postpone a less supplied learning construction by refining integration. Continue the ACTIVE general-learning Goal after publication, preserving honest conventional sufficiency and negative evidence. The ultimate objective has not been achieved.

---

# Complete preceding living ledger, preserved verbatim

# First-Principles Robot Learning: Living Research Ledger v15

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

| Ordinary summary | One-step true-position MSE | Recursive1..10-step MSE | Sustained position MSE | Late position RMSE | Late instantaneous velocity RMS | Precision goals /96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pose | 0.0091053359 +/- 0.00163 | 0.30416221 +/- 0.0551 | 40.855731 +/- 12.6 | 4.4799153 +/- 0.655 | 1.6343133 +/- 0.107 | 0 |
| one_delay | 1.5335182e-05 +/- 2.35e-06 | 0.0015142148 +/- 0.000435 | 0.0079982955 +/- 2.75e-05 | 0.003375953 +/- 0.000346 | 0.015114882 +/- 0.000609 | 45 |
| rich_delay | 1.5061309e-05 +/- 8.1e-06 | 0.00048238211 +/- 0.000214 | 0.0080004894 +/- 1.62e-05 | 0.0015078297 +/- 0.000114 | 0.0083584123 +/- 0.000352 | 79 |
| selected_delay | 1.2087247e-05 +/- 3.96e-06 | 0.00047995072 +/- 0.000124 | 0.0080028866 +/- 1.72e-05 | 0.0017580102 +/- 0.000338 | 0.0093644345 +/- 0.000639 | 76 |
| linear_arx | 6.6398652e-05 +/- 1.74e-05 | 0.022683172 +/- 0.0043 | 0.0080339074 +/- 9.45e-05 | 0.0058548334 +/- 0.000486 | 0.074579543 +/- 0.00178 | 16 |
| polynomial | 1.1182448e-05 +/- 3.5e-06 | 0.00036887657 +/- 4.37e-05 | 0.0079890908 +/- 1.78e-05 | 0.0016654298 +/- 0.000113 | 0.010560333 +/- 0.000481 | 69 |

Selected raw histories across8fits: [4, 2, 2, 4, 4, 4, 1, 2]. Fixed-rich histories: [4, 4, 4, 4, 4, 4, 4, 4]. Polynomial histories: [2, 2, 2, 2, 2, 2, 2, 2]. Total unsuccessful optimizer terminations by method: {'pose': 8, 'one_delay': 0, 'rich_delay': 0, 'selected_delay': 0, 'linear_arx': 0, 'polynomial': 0}. These statuses do not imply every unsuccessful result was discarded; source retains finite improved results.

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

# First-Principles Robot Learning: Living Research Ledger v14

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited first coupled articulated learning instrument, context-dependent relation transfer, negative extrapolation results and explicit measurement corrections. V13 was published and verified on the existing SSH origin/main at `aeb569a16688c7290af88169d654c0f809b6c2a6` before continuing. Complete v13 and all earlier ledgers follow verbatim. The broader general-learning Goal remains ACTIVE.

## Principal result: relation choice and context coverage both matter

**TESTED:** With broad pose experience, conventional generic context-conditioned regression acquires useful two-joint action/displacement knowledge and uses it for new displacement targets/poses, without receiving the true inertia formula or named physical parameters. A learned inverse relation performs better than the learned forward relation within the selected finite basis. This accepts established learning tools rather than claiming a new foundation or superior general robot learner.

**TESTED NEGATIVE:** Matched interaction budgets confined near one pose do not support broad-pose transfer. The more flexible context learners can extrapolate worse than the full coupled constant baseline; some inverse relations produce severe prediction outliers. Correcting known response-measurement noise helps inverse decisions in the broad condition but does not reliably improve the narrow condition’s out-of-support score. Statistical moment correction is not reliable extrapolation, and local validation is not evidence for unseen-context validity.

**DERIVED / EXISTING RESEARCH:** Which relation is approximated changes its noise placement and physical-function class. Forward and inverse RBF coefficient counts being equal does not make their approximation capacity identical: the leading inverse ACTION relation K(q)=M(q) is affine in cos(q2), while the forward response M(q)^-1 is rational. A learned inverse relation remains a model; the common bounded solver uses its implied forward relation. This is not a model-free-versus-model-necessity result.

**LIMITATION:** All episodes begin at supplied exact preparation pose/rest, have calibrated ideal torque, known duration/noise/angle topology, and measure joint-angle displacement after one short pulse. Pulse endpoints are moving. This is not continuous reaching/stopping, unknown-state inference, model-language discovery or general articulated intelligence. End this calibration branch after the compact test and move to continuous coupled motion or consequential missing-history revision.

## Why this changes the research question

The scalar program supplied a correct four-parameter function family and concentrated increasingly on controller optimization. V13 bounded that branch: a better optimizer finds lower average cost in the same policy family, but precise station keeping collapses. Adding another scalar optimizer would not resolve the deeper dependency on supplied vocabulary and context.

Start from an intervention rather than a preferred architecture: does the same command have the same consequence at different body configurations? A coupled body gives a clear counterexample. Learn a reusable relation from positions/actions, then use it to choose commands for goals never issued during acquisition. Forward consequences, inverse action relations and episodic/direct alternatives are logically possible; states/world models are not assumed universal foundations. Here a compact conventional comparison is enough to decide whether a new mechanism is immediately needed. The evidence favors conventional sufficiency inside the sampled regime and exposes unsupported transfer outside it.

## F53. Continuous coupled rest-pulse mechanics [DERIVED / EXISTING RESEARCH]

Consider a horizontal two-link point-mass body with unit masses/lengths, direct bounded generalized torques, no gravity/friction. From Cartesian kinetic energy, Lagrange's equation gives

~~~text
M(q2)=[[3+2cos(q2),1+cos(q2)],[1+cos(q2),1]]
M(q2)*v_dot + h(q2,v)=u, q_dot=v
h=sin(q2)*[-(2*v1*v2+v2^2), v1^2].
~~~

M is SPD: detM=2-cos²(q2)>=1, and its global minimum eigenvalue is3-2sqrt(2)=.1715728753. For fixed torque from rest, time-reversal symmetry and smoothness give

~~~text
q(T)-q0 = .5*T^2*M(q0)^-1*u + O(T^4).
~~~

At q2=0 the leading torque map is [[1,-2],[-2,5]], while at q2=pi it is identity. Thus joint2 torque has a nonzero leading joint1 effect at the first pose and zero leading coefficient at the second. The finite-time effect at pi is NOT identically zero: evolving q2 restores coupling at higher order. At generic q2, a frozen linear response has a quartic absolute remainder. At0/pi its quartic term vanishes, hiding that generic rate. Adding damping or actuator lag would change the expansion and cannot be silently imported.

An early design used semi-implicit Euler's T² coefficient; BEFORE execution the instrument was changed to continuous DOP853 and the .5T² coefficient, preserving the draft/correction. This avoids mistaking discretization for physical response. Classical mechanics is a tool, not a novelty claim or a supplied formula for the ordinary learner.

## E34. Independently checked coupled mechanics instrument [TESTED, NO LEARNER]

`coupled_context_v14_pilot.py`/JSON retains153 rest pulses over17 q2 poses/three command directions/three durations,24 non-rest energy cases and2 coupling examples:179 distinct episodes,21.025 simulated seconds. Twenty-four tighter numerical reintegrations are solver checks, not extra independent data; total203 solver calls. Joint positions are measured at pulse BEGINNING/END in time, not Cartesian end-effector positions. Pilot sensor SD1e-5.

Generic duration-halving error ratios are15.936..16.064 then15.984..16.016, approaching16 as quartic remainder predicts. Symmetry cases are higher order or roundoff limited. Independent theory computes fourth time derivative of q at0=(1.75,-3) at q2=pi/2,u=(0,1). Energy change equals constant torque work, and cyclic momentum p1 obeys p1_dot=u1. The artifact auditor reconstructs all cases using separately derived Hamiltonian(q,p) dynamics and Radau; max state discrepancy2.62e-12, energy-work1.66e-12, cyclic momentum8.27e-13; measurement RNG exact. An independent short-pulse implementation also checks the special higher-order effect. Sources/results/reports remain.

**Measurement correction before learning:** If the same noisy initial encoder reading supplies context and the displacement baseline, Cov(e0,eT-e0)=-R. Independent context reads remove that shared-noise path but still leave noisy-context EIV. The pilot contains that correlated record, but trains no learner. The subsequent comparison openly supplies exact PREPARATION pose to every method; it does not pretend to infer exact context from noisy sensors. Both joint coordinates are supplied; selecting q2 alone would reveal the relevant context dimension. Angle periodicity is separately supplied to the generic basis.

## F54. Fair inverse regression needs a measurement model [DERIVED / EXISTING RESEARCH]

Define measured normalized response d_obs=(oT-q0)/k, k=.5T², using supplied EXACT preparation pose. At T=.1 and encoder SD1e-4, response-noise covariance is S=.0004 I. An earlier two-reading comparison used (oT-o0)/k and S=.0008 I, adding avoidable initial measurement noise; its complete source/data/results are preserved below. Let phi(q) be fixed features. A forward relation d≈B(q)u has exact issued-command regressors and noisy response outputs. An inverse u≈K(q)d instead has noisy regressors: naive least squares generally shrinks an inverse even with unlimited samples. It is not a fair sole direct baseline.

With feature ordering(context,input), X=phi⊗d_obs. Independent zero-mean response noise gives

~~~text
E[X_obs'X_obs | noiseless data] = X_true'X_true + sum(phi*phi') tensor S
E[X_obs'u | noiseless data] = X_true'u.
~~~

Subtract the known covariance term for corrected moments. These are unbiased MOMENTS, not unbiased finite coefficient estimates. Inversion, regularization, finite data and selection retain bias. The source explicitly projects negative corrected-Gram eigenvalues to zero then adds ridge; it does not solve an unregularized indefinite moment equation. The field named training_normal_equation_residual refers to unregularized G and need not vanish under this regularization.

For any fixed inverse candidate K evaluated on fresh validation measurement noise,

~~~text
E[||K*d_obs-u||² - tr(K*S*K')] = ||K*d_true-u||².
~~~

This correction is valid even when finite-pulse nonlinearities make the linear inverse family approximate; it removes measurement risk, not approximation error. Corrected validation values can be negative, and choosing their minimum is optimistic. All candidate scores/eigenvalues/floors are kept; independent fresh tasks, not a lucky validation minimum, determine evidence. These are standard errors-in-variables identities, not new statistical theory.

## E35. Compact same-experience relation-learning comparison [TESTED]

The original `coupled_learning_v14_two_reading_pilot_design.md` was prescribed before learning outcomes. Current design and `coupled_learning_v14_information_correction.md` preserve the later exploratory information-use repair, with all tuning grids/tasks unchanged. Two acquisition conditions: q1 uniform[-pi,pi] in both; q2 broad[-pi,pi] or narrow[-.4,.4]. Eight independently seeded datasets per condition,512 training+128 separate validation pulses each. Commands uniform[-1,1]^2, independently randomized. Each pulse has T=.1, zero initial velocity, known EXACT preparation pose supplied to every learner, and independent noisy initial/final joint-angle measurements. Ordinary learners receive preparation context, commands and these measurements; true states only score outcomes. No mass/inertia/Coriolis/relevant-encoder formula is supplied. Canonical learners use the available exact preparation rather than a noisy baseline when constructing their response.

All methods use the identical complete dataset within replicate. Alternatives:

- FULL coupled constant forward matrix, a serious simple context-omission baseline rather than independent joints.
- Generic context forward B(q), using65 periodic Gaussian RBF features (8x8 center grid over BOTH encoders plusconstant).
- Same feature vocabulary for naive inverse K(q), a descriptive measurement-bias baseline.
- Moment-corrected inverse with the known response covariance and PSD/ridge regularization.

Width(.5,1,2,4) and ridge(1e-6,1e-4,.01,.1) are chosen by separate declared validation; constant matrix has the ridge grid only. Forward selection uses response prediction SSE; both inverse variants use noise-corrected inverse risk. All832 candidate fits/coefficients/eigenspectra/selection scores per comparison are saved (1664 across original and information-corrected pipelines). Selection itself remains finite/noisy; neither moment subtraction nor equal feature counts gives equal physical prior or universal optimality.

For ALL ordinary methods, commands minimize the SAME bounded displacement metric. Forward uses B(q); inverse uses pinv(K(q),rcond1e-8), retaining conditioning diagnostics. This eliminates a command-clipping versus goal-metric projection confound. It also means the inverse is a learned relation used in planning, not a pure model-free policy search.

Fresh tasks: q1[-2.7,-1.1,.4,2.2] crossed with q2[-pi,-2.1,-1.2,-.4,0,.4,1.2,2.1] and five COMMON desired normalized displacement vectors(.1,0),(0,.1),(.1,-.1),(-.1,.1),(.05,.1), total160 context-goal pairs per fit/method. An initial proposed +pi duplicate was replaced by0 BEFORE execution, making the periodic poses unique and including narrow-region interpolation. Desired actual displacement=k*z. Test goals are not generated from hidden pose-specific random torques, so they do not encode the action relation being tested.

A privileged continuous endpoint optimizer constructs a bounded command witness for EVERY task, max residual3.24e-13 in normalized displacement,1456 recorded solver rollouts in the original comparison; canonical correction reuses the same witnesses with no new oracle search. It receives true mechanics and is excluded from ordinary learning comparisons. Leading-order required torque<=.5 supplies margin; exact numerical witnesses establish feasibility for these finite tasks, not a global reachability theorem.

Primary score is ||actual_displacement/k-z||², averaged over160 tasks then summarized across eight independent acquired-data seeds. Truth is evaluator-only. Task dynamics are deterministic after command issuance; noisy test readings are retained but do not enter one-pulse decisions or primary score. MC SE concerns acquired-data variation, not160 independent fits. Additional128 fresh broad-context/random-command pulses per fit test predictive transfer without updating learners.

| Acquisition | Relation | Task MSE +/- clustered SE | Broad-context prediction MSE +/- SE | Saturations /1280 |
| --- | --- | ---: | ---: | ---: |
| broad | constant_forward | 0.00898236019 +/- 0.000346067 | 0.852809 +/- 0.0662727 | 0 |
| broad | context_forward | 3.18254792e-05 +/- 3.7038e-06 | 0.000414215 +/- 3.29175e-05 | 0 |
| broad | naive_inverse | 8.98604851e-06 +/- 1.53095e-06 | 0.00013751 +/- 1.35219e-05 | 0 |
| broad | corrected_inverse | 4.52803178e-06 +/- 9.76327e-07 | 0.000116246 +/- 1.84266e-05 | 0 |
| broad | oracle_endpoint | 9.59917847e-27 +/- 0 | privileged feasibility witness | 0 |
| narrow | constant_forward | 0.0194882022 +/- 0.00013015 | 4.13 +/- 0.136407 | 0 |
| narrow | context_forward | 0.0548093739 +/- 0.0208907 | 28.2751 +/- 8.23767 | 66 |
| narrow | naive_inverse | 0.0144883938 +/- 0.00289956 | 0.212483 +/- 0.116562 | 0 |
| narrow | corrected_inverse | 0.0147816106 +/- 0.00262247 | 0.157805 +/- 0.0792404 | 0 |
| narrow | oracle_endpoint | 9.59917847e-27 +/- 0 | privileged feasibility witness | 0 |

In broad context, naive-minus-corrected inverse paired task MSE is +4.458017e-6 +/-1.081753e-6 SE. Corrected inverse therefore improves this finite declared pipeline, while the forward-vs-inverse contrast also changes physical-function approximation and decision conditioning. In narrow context, the same difference is -.000293217 +/- .000602516: the corrected mean is slightly worse, with large paired uncertainty. The stronger negative difference in the original two-reading comparison does not carry over as a strong claim about the repaired estimator. It does not repair missing context coverage. No philosophical conclusion about model necessity follows.

**Secondary support diagnostic:** Narrow acquisition still fits decisions inside q2[-.4,.4]: context-forward mean task MSE2.44037e-5, corrected inverse9.95402e-6 over60 familiar-region tasks. On100 outside-region tasks these become.0876804 and.0236446. The constant model is9.84515e-5 inside/.0311221 outside. This isolates support-related transfer failure descriptively; it does not prove no method could extrapolate with appropriate physical assumptions. Broad acquisition uses the same640 pulses per seed, not more interactions.

**Symmetry limitation:** Physics is invariant to q1 global rotation, but learned context relations have nonzero command variation when the SAME q2/goal is tested at different q1. Lower task error does not prove exact symmetry discovery. Both q1 diagnostics/pooling conventions and all commands are preserved independently. No supplied q2-only feature selection was passed to the ordinary learner. Unsupported extrapolation can amplify this nuisance dependence and nearly singular inverse predictions.

## Pre-publication information-use correction [CORRECTED / TESTED]

The first complete comparison used the unbiased two-reading displacement (oT-o0)/k even though q0 was supplied exactly. Independent audit identified that o0-q0 is observable sensor noise: using (oT-q0)/k removes that independent error and halves S from.0008 to.0004 I. The old statistics remain valid for their stated estimators, but that noise was not irreducible and those estimators did not use available information efficiently.

Before publication, correct ALL methods without changing features, width/ridge grids, conditions, validation logic, targets or bounds. Reuse EXACT original raw acquisition/validation/predictive data and original endpoint feasibility witnesses. Original source/default output metadata, full data/results/timings/design/theory/audits remain as `coupled_learning_v14_two_reading_pilot*` and corresponding audit names. The canonical source records exact parent hashes. All160 non-response arrays are bitwise identical;32 response arrays change only by the declared statistic. No new acquisition/predictive data or oracle search is added. New commands require12800 additional task episodes.

This repair follows an information audit after original outcomes were read; call it exploratory, not a fresh blind benchmark. Heldout tasks/priors are unchanged and no task outcome selects settings. Original broad task MSEs for constant/forward/naive/corrected are.00898141/7.02875e-5/1.79176e-5/9.41635e-6; original narrow values.0196274/.0455312/.0231309/.0254856. Lower effective measurement noise improves all broad context-aware means in the canonical run, but does not uniformly improve out-of-support selected pipelines. Selection/regularization still matter.

A post-hoc analytic reference is zero torque: rest remains exactly at rest, so average task score equals mean||z||²=.0145 with no extra simulation. Narrow inverse means(.0144884 naive/.0147816 corrected) are close to that uninformed reference at their large SEs; their aggregate broad-grid transfer is not reliable learned goal achievement. This reference is not a tuned new method or an equivalence test. Broad context-aware scores are orders of magnitude smaller. No more calibration variants are added.

## Audit, history corrections, budgets and reproduction

All source/design/pilot/actual results/independent reports and failed or rejected candidates remain. Original two-reading and canonical repaired JSON/NPZ re-execution are byte-identical within their respective pipelines. Original pre-audit metadata is retained; separate completed audit records subsequent verification. No numerical/task case is silently dropped. All ordinary bounded action solver statuses and selected condition numbers are audited; successful numerical optimization does not make an extrapolated relation correct.

Acquisition:16 datasets x640 pulses=10240 episodes/resets,1024 simulated seconds. Predictive tests2048 episodes/204.8 seconds. Each original/corrected comparison has12800 task episodes/1280 seconds (five methods including privileged endpoint oracle), total25600 episodes/2560 seconds during development. Oracle feasibility optimization additionally uses1456 deterministic plant-simulator rollouts/145.6 seconds once, not ordinary learner training. There are49152 joint-coordinate scalar sensor readings in the reused acquired/predictive episodes and102400 total in both task comparisons; the preparation mechanism supplies exact context/rest beyond those noisy readings. Pilot179 episodes/21.025 seconds and solver-refinement computation are additional development, not hidden acquisition. No hardware experiments are claimed. Wall timings are retained separately and do not constitute a runtime-efficiency theorem.

Independent theory derives Lagrangian mechanics, continuous short-pulse order, sensor-noise coupling, EIV/validation identities, source tensor ordering and physical-function-class limitations. The artifact audit independently reconstructs physics, RNG/data, all fits/validation/selection, commands, predictive and decision scores, budgets and clustered summaries, including extrapolation failures. It uses a separate implementation and qualifies numerical conditioning rather than rerunning source and calling that independent verification. See `audit_coupled_context_v14_pilot`, `audit_coupled_shortpulse_v14`, `audit_coupled_learning_v14`, q1-variation audit and `coupled_learning_v14_theory_independent` reports/results. Audit scope and unresolved concerns must be read before extending any claim.

~~~bash
.venv/bin/python research/coupled_context_v14_pilot.py > /tmp/coupled_pilot_v14.json
cmp research/coupled_context_v14_pilot_results.json /tmp/coupled_pilot_v14.json
.venv/bin/python research/coupled_learning_v14.py --output-prefix /tmp/coupled_v14
cmp research/coupled_learning_v14_results.json /tmp/coupled_v14_results.json
cmp research/coupled_learning_v14_data.npz /tmp/coupled_v14_data.npz
.venv/bin/python research/audit_coupled_learning_v14.py
~~~

**EXISTING RESEARCH:** [Tedrake's multibody derivation](https://underactuated.mit.edu/multibody.html) supplies classical mechanics context. [Nguyen-Tuong/Peters2011](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Nguyen_CP_2011.pdf) explicitly discusses forward/inverse/mixed robot models and ill-defined inverses. [Jordan/Rumelhart1992](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1603_1) addresses learning when unknown dynamics intervene between actions and outcomes. These precedents exclude novelty claims. [Romeres et al.2018](https://arxiv.org/abs/1809.05074) tests learned position-history features for inverse dynamics on real iCub-arm data, a relevant stronger method for subsequent continuous motion. Our calibration is not a reproduction or performance competitor. The literature note distinguishes accessed full papers from publisher abstracts.

## Research-direction reassessment and exact continuation

This is a modest construction beyond the supplied scalar/tanh family: a generic conventional learner acquires context-dependent coupled behavior and chooses new actions from noisy experience. The same data can support forward or inverse relations, with different function/noise/conditioning behavior. Statistical corrections and local validation have explicit boundaries. Simple established methods suffice for the supported pulse task; do not invent an architecture to rescue novelty.

The scientific value is conditional and limited: exact rest/pose preparation, ideal torque, fixed body, joint-angle semantics/periodicity, torque-affine finite basis, known sensor noise and short isolated episodes are all supplied. No unknown component language, cross-body transfer, continuous stability or articulated stopping is demonstrated. A successful calibration is not the ultimate scientific objective. Further calibration/cosine-fitting releases would drift.

**Immediately continue:** Remove rest preparation and investigate continuous coupled motion or how a learner detects/revises a summary that omits relevant history. Derive a decisive aliasing/transfer example, ensure actual task feasibility, and compare serious history-based/derivative-free conventional methods rather than giving only one method true velocities or accelerations. Learn from actual available positions/actions; distinguish a learned conditional predictor from a supplied physical state. Preserve first-principles alternatives and strong baselines. Complete and audit meaningful results, append complete v14 in the next ledger, commit/push SSH/verify and continue. The general-learning Goal remains ACTIVE.

---

# Historical archive: Complete original v13 ledger, including all preceding versions

# First-Principles Robot Learning: Living Research Ledger v13

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

| lambda,c | Physical CEM cost +/- SE | Virtual CEM cost +/- SE | Gradient-selected cost +/- SE | Nonlinear MPC cost +/- SE | Station keeping P/V/G/M |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.2,0 | 0.313822 +/- 0.008810 | 0.324677 +/- 0.007301 | 0.265958 +/- 0.002726 | 0.266410 +/- 0.002463 | 4/5/0/9 of16 |
| 0.2,0.2 | 0.330627 +/- 0.010697 | 0.306525 +/- 0.009847 | 0.275643 +/- 0.002668 | 0.271958 +/- 0.002738 | 8/9/1/14 of16 |
| 0.7,0 | 0.402863 +/- 0.011340 | 0.390278 +/- 0.014717 | 0.306701 +/- 0.004617 | 0.266985 +/- 0.004863 | 1/0/0/0 of16 |
| 0.7,0.2 | 0.352047 +/- 0.015794 | 0.399153 +/- 0.028823 | 0.289054 +/- 0.005431 | 0.263184 +/- 0.003437 | 1/2/0/2 of16 |

Physical-CEM-minus-gradient paired differences by plant are.047864+/-.007912,.054984+/-.010174,.096162+/-.008644,.062994+/-.013799 (MC SE). Nonlinear-MPC-minus-gradient differences are+.000452+/-.003115,-.003685+/-.002267,-.039716+/-.003875,-.025869+/-.003686. The first two support close mean performance under this protocol, not a formal equivalence test. The slow plants retain clear gaps in these finite summaries. No global policy optimum or universal superiority follows. Optimization, fixed-bank sampling, validation selection and virtual compute change together; this experiment does not isolate derivatives alone as the cause of the gain.

## Mean performance and precision disagree [TESTED NEGATIVE]

The gradient policies have higher final10 velocity RMS than physical CEM in every cell:

| Plant | Physical CEM late velocity RMS | Gradient late velocity RMS | MPC late velocity RMS |
| ---: | ---: | ---: | ---: |
| 0 | 0.109063 | 0.157306 | 0.081090 |
| 1 | 0.074898 | 0.111641 | 0.055618 |
| 2 | 0.134367 | 0.186690 | 0.139521 |
| 3 | 0.124491 | 0.186898 | 0.113396 |

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

# First-Principles Robot Learning: Living Research Ledger v12

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited shared-experience comparison, observation-only loss derivation, negative same-family model-reuse diagnostic, and preserved seed correction. Complete v11 and all earlier ledgers follow verbatim. V11 was published at `90b7e9b8198a1854982588a2ed3553c51681129c`; the subsequent audited preparatory pilot/handoff was published at `6dc0c5037b5f23ee62ed04e07914ac595a7fc6bb`, both verified on the existing SSH origin/main. The general-learning Goal remains ACTIVE.

## Principal findings: different uses of identical experience

**PROVEN UNDER ASSUMPTIONS:** A causal learner can rank expected goal behavior directly from noisy positions and its issued actions, without hidden-state reward or fitting dynamics, in this declared lagged system. The observed position/displacement quadratic loss differs from the true position/velocity loss by a policy-independent expectation offset. Timing, sensor independence/zero mean, and exogenous fixed-horizon objectives are essential. This is an elementary expectation identity, not a universal learning principle or novelty claim.

**TESTED:** Direct CEM search in a supplied six-parameter dynamic feedback family improves behavior. Conventional output-error models, EKF and bounded MPC fitted to the **exact same complete experience**, including poor policy trials, yield lower mean tracking cost at both retained interaction checkpoints on all four declared plants. The models do not receive true parameters or extra training episodes. The result compares specified mechanisms, with different supplied vocabulary and computation; it does not show that direct learning generally requires models or that these toy methods compete with modern robotics learners.

**TESTED NEGATIVE:** Repeating the identical direct CEM search inside the learned nonlinear model, with no extra simulator-plant training, does not consistently improve the same controller family. Thus the current evidence does not isolate model reuse, policy capacity, or finite search as the cause of the MPC advantage. Using the correct nonlinear family still does not consistently beat the learned linear model in tracking. Neither predictable behavior nor low average cost ensures precise station keeping.

**CORRECTED BEFORE RELEASE:** Expanding the first run's candidate budget accidentally exceeded its replicate seed spacing. Some adjacent training replicates reused sensor streams. Source inspection caught this before outcomes were reviewed; the full excluded run is retained, corrected disjoint seed domains are verified, and all reported inferential results use the rerun. No failed run was concealed.

## Why change the framing now?

The preceding sequence successfully used predictive consequences despite ambiguous physical labels, then corrected a planner/task-interface mismatch. That route could prematurely turn 'learning' into 'fit a supplied physical model.' Ask instead whether experience must be used that way. Directly improving behavior from observed consequences is a genuinely different possibility: no hypothetical dynamics, latent-state estimate, or online planner is required by its algorithm.

A fair discriminating experiment must separate data acquisition from its use. Comparing a direct learner's feedback traces to a model learner's specially designed excitation would change both. Here the predictive learner is a shadow of the direct search: it reads every action/observation already collected, respecting all episode resets. Its favorable result concerns reuse of the same data within strongly supplied families, not a necessary foundation of intelligence. This is a constructive learning-mechanism comparison, rather than another isolated scalar identification theorem.

## F50. Observation-only expected goal loss [PROVEN UNDER ASSUMPTIONS]

Use the lagged channel

~~~text
v[t+1]=f(v[t])+m[t]
m[t+1]=h(m[t],v[t],u[t])+xi[t]
x[t+1]=x[t]+v[t+1], o[t]=x[t]+e[t].
~~~

The sensor errors are fresh iid, zero mean, variance R<infinity, independent of initial state, physical noise and policy coins. Commands can depend causally on observations through the present. Goals, weights, horizon and normalization are fixed/exogenous, and expected losses finite. Because v[t+1] and x[t+1] depend on commands only through u[t-1], they are independent of e[t],e[t+1]. Expanding the squares gives

~~~text
E[(o[t+1]-g[t])^2+.3*(o[t+1]-o[t])^2+.03*u[t]^2]
 = E[(x[t+1]-g[t])^2+.3*v[t+1]^2+.03*u[t]^2]+1.6R.
~~~

For the fixed episode normalizer D=1+mean(g²), the mean episode offset is1.6R/D. Knowing R is unnecessary to rank expectations. No Gaussianity is needed for the identity; finite fitness variance generally needs fourth-moment conditions. Neighboring stage errors are dependent; independent training seeds, not timesteps, are the uncertainty units.

**Counterexamples / limitations:** Biased sensors add a policy-dependent position cross term. Colored sensing, state-dependent sensor laws, random stopping or data-dependent goals/normalizers can break the argument. With MOTOR-FIRST one-step v1=beta*u0 and bounded u0=-k*tanh(e0/SD), the full expected gap is `1.6R+.6*beta*k*E[e0*tanh(e0/SD)]`, dependent on k. The preserved pilot independently checks Gaussian quadrature and one million Monte Carlo samples: at SD.2/beta1.2, k0/.3/.9 gaps are.064/.09016648/.14249943.

The identity applies to a fixed causal policy on fresh noise. Adaptive repeated optimization of the same finite future-noise bank can invalidate that fixed-policy training interpretation. Even fresh noise per candidate leaves optimistic minimum selection. All final policies are frozen before fresh test episodes; their finite observed/true score gaps need not equal the expectation offset trajectory by trajectory. No hidden true-state reward enters policy selection.

## F51. Predictive fitting of feedback-collected experience [PROVEN UNDER ASSUMPTIONS]

Training has ZERO process noise and each episode begins at supplied known physical zero. Given theta and the issued command prefix, the position x_theta is deterministic. For the actual causal search/controller, the joint/sequential experiment likelihood factors as

~~~text
p_theta(o0:T,u0:T-1)
 = p_theta(o0) * product_t pi_t(u_t | recorded history through o_t)
                       * Normal(o[t+1];x_theta[t+1](issued prefix),R).
~~~

Conditional on recorded history, the actual action rule and preceding CEM updates are independent of the hypothetical physical theta. Deterministic policies supply theta-independent point masses on the recorded support. Therefore raw-position SSE over **all** known-reset segments is a valid sequential/joint Gaussian likelihood objective. Reset simulated states and parameter sensitivities at every segment; do not join separate physical resets into a continuous trajectory.

This does NOT assert that the likelihood conditional on the entire future command sequence is an iid Gaussian product: those future commands encode past sensor errors. It does NOT justify fitting only elites selected by subsequent outcomes. Nonzero unobserved process noise would require a latent-state likelihood or separate argument. Valid likelihood does not imply excitation, unique physical labels, a global optimum or successful decisions. These distinctions were derived independently in `audit_observation_loss_v12.md` and challenged again in `v12_comparison_design_independent.md`.

## E30. Preserved preparatory direct-learning pilot [TESTED, LIMITED]

`direct_experience_v12_pilot.py`/JSON and independent executable audit retain eight seeds on theta(.65,.7,.2,1.2). CEM searched a three-parameter supplied PD/smoothing class using8256 simulated-plant transitions per seed. Fresh test mean true cost.678067 +/- .185905 clustered SE;3/16 station-keeping events. A handpicked fixed PD was much worse, but was explicitly a weak descriptive comparator. This pilot alone did not warrant a new ledger or any method-superiority claim. It motivated a stronger dynamic class and an actual shared-data competitor.

## E31. Shared-experience controller comparison [TESTED]

Design was written before inspecting new outcomes in `shared_experience_v12_design.md`; its source-inspection seed correction is separately disclosed. Four lagged plants rho=.65,beta=1.2,lambda in{.2,.7},c in{0,.2}. Sensor SD.2; train motor process0; test motor process SD.03 on every step. Known zero reset per86-step episode. Current goals switch at6,28 without resetting any physical/controller memory. Feedback remains available throughout all86 test steps. This differs from v10/v11 forced-coast stopping and must not be called a solution of that task.

The direct policy is

~~~text
sx += alpha*(o-sx)
sv += alpha*((sx-sx_previous)-sv)
u = clip(kp*(goal-sx)-kd*sv+bv*sv_previous-bu1*u[t-1]-bu2*u[t-2],-1,1).
~~~

Six searched coordinates are logkp,logkd,logitalpha,bv,bu1,bu2. Bounds kp[.005,.5],kd[.02,4],alpha[.1,.98],remaining coefficients[-2,2]. It includes pilot PD exactly. In the alpha1 noiseless linear limit, m_t=lambda(v_t-rho*v[t-1])+beta*u[t-1]; clipped state feedback Kp*error-Kv*v-Km*m maps to kp=Kp,kd=Kv+Km*lambda,bv=Km*lambda*rho,bu1=Km*beta,bu2=0 when gain restrictions permit. Alpha1 is OUTSIDE the actual search bound. This useful limiting capacity check does not prove arbitrary observer/MPC containment, especially with noisy measurements or nonlinear drag. Positive signs, smoothing, vocabulary and finite bounds are supplied biases.

Diagonal CEM uses8 generations x24 candidates x2 actual simulated-plant training episodes, goals(1,-1,.5) and(-1,1,-.5);6 elites, fixed initial distribution/SD floors detailed in the design/source. Each candidate gets fresh noise; previous winner is retested as candidate0. Select current-generation winner, not a stale best-ever noisy score. No true state or test result enters selection. Save every poor trial and CEM draw/elite. Checkpoints after2/8 generations consume8256/33024 transitions and96/384 resets; the earlier checkpoint is a prefix, not extra acquisition. Eight training seeds per plant,32 clusters total.

At each checkpoint, shadow linear/nonlinear OE fits ALL observations/commands using separate reset segments, v9's three starts/bounds/max160 evaluations, analytic Jacobians, minimum raw SSE. Supplied families include the true one. Each model controls with EKF (supplied noise scales/zero prior) and bounded TRF MPC:16 free actions, truncated to min(16,86-t), current goal held constant in forecasts, no zero-command tail, same quadratic stage cost. Save fit records/curvature diagnostics and all plans/status/selected costs/gradients. All384 fit attempts and22016 action solves report convergence; success is not a global nonlinear optimality certificate. Maximum selected projected action-gradient magnitude is3.568e-5.

Fresh evaluations use goals(2,-2,1),(6,-4,3), common underlying noise across compared policies/checkpoints within seed, separate training/evaluation streams. Primary cost averages86 true-state stages then divides by1+mean(goal²). True states are external evaluator-only. Late10 station keeping requires all |position error|<=.3 and|velocity|<=.15. Means/SE first average both tasks within each training seed, then summarize eight seeds per plant. Counts have16 dependent-within-fit tasks; they are descriptive, not calibrated success probabilities.

| lambda,c | Plant training steps/seed | Direct cost +/- SE | Linear OE/MPC cost +/- SE | Nonlinear OE/MPC cost +/- SE | Station keeping: direct/linear/nonlinear |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.2,0.0 | 8256 | 0.357912 +/- 0.013839 | 0.267746 +/- 0.002460 | 0.267815 +/- 0.002449 | 4/5/5 of16 |
| 0.2,0.0 | 33024 | 0.314099 +/- 0.008798 | 0.267667 +/- 0.002475 | 0.267673 +/- 0.002478 | 4/5/5 of16 |
| 0.2,0.2 | 8256 | 0.338069 +/- 0.015625 | 0.268062 +/- 0.003984 | 0.274095 +/- 0.001664 | 5/8/8 of16 |
| 0.2,0.2 | 33024 | 0.331481 +/- 0.010001 | 0.277502 +/- 0.004051 | 0.274189 +/- 0.001661 | 3/8/8 of16 |
| 0.7,0.0 | 8256 | 0.461630 +/- 0.022391 | 0.266351 +/- 0.006057 | 0.266379 +/- 0.006054 | 1/1/1 of16 |
| 0.7,0.0 | 33024 | 0.401574 +/- 0.013085 | 0.266351 +/- 0.006056 | 0.266358 +/- 0.006058 | 2/1/1 of16 |
| 0.7,0.2 | 8256 | 0.529828 +/- 0.079620 | 0.272097 +/- 0.003159 | 0.274589 +/- 0.002871 | 1/0/0 of16 |
| 0.7,0.2 | 33024 | 0.357066 +/- 0.013753 | 0.272597 +/- 0.003205 | 0.274551 +/- 0.002883 | 2/0/0 of16 |

The final direct-minus-nonlinear-MPC paired mean differences are.046426+/-.008424,.057292+/-.011525,.135216+/-.009540,.082515+/-.014324 by plant order, with SE across eight seed clusters. These are Monte Carlo summaries for the fixed protocol, not a distribution-free superiority theorem. Learned linear and nonlinear control remain close; adding the correct nonlinear vocabulary does not consistently improve the task score. The original direct policy can have more station-keeping successes despite worse average cost; no single scalar certifies all capabilities.

**Metadata/capacity caveat:** MPC consumes remaining episode time; the direct policy is stationary without a clock feature. Both have current goals, bounds and reset/noise facts, but their internal vocabulary/use of metadata differs. The prescribed first70-step diagnostic avoids horizon truncation entirely: MPC still has its full16 actions through t70, so end shortening cannot causally explain prefix performance gaps. All four final prefix costs favor the model planners too; this rules out that one explanation, not capacity/compute confounding. The final direct command can only add effort, but broader differences cannot be reduced to this small single-action effect.

## E32. Same-family simulated experience reuse [TESTED NEGATIVE]

Before inspecting corrected outcomes, `model_reuse_v12_design.md` prescribed repeating exactly the six-coordinate CEM protocol on each learned final nonlinear model. Same training goals, sensor SD.2 and motor process0; independent virtual seeds,33024 additional VIRTUAL transitions per fit, no new simulated-plant training. Each selected stationary policy is tested on the same heldout actual-plant streams. This controls controller family and time features, but adds model-based computation and a strongly supplied physical vocabulary. Virtual-state access never supplies true plant-state reward.

| Plant index | Model-reused same-family cost +/- SE | Original direct minus reused +/- paired SE | Station keeping |
| ---: | ---: | ---: | ---: |
| 0 | 0.321943 +/- 0.007520 | -0.007844 +/- 0.014065 | 5/16 |
| 1 | 0.311855 +/- 0.008179 | 0.019626 +/- 0.012586 | 2/16 |
| 2 | 0.393994 +/- 0.022549 | 0.007579 +/- 0.017026 | 2/16 |
| 3 | 0.406712 +/- 0.024405 | -0.049646 +/- 0.027202 | 1/16 |

No consistent gain emerges from this one virtual-CEM restart; the large spread and eight seeds forbid treating small differences as equivalence. It does not prove simulation reuse is useless or the direct family incapable. Search can fail within an adequate family; a local simulation objective may not transfer; a model can be accurate but the chosen optimizer unhelpful. The MPC advantage therefore remains a result for complete specified mechanisms. One exact-gradient same-family capacity/search diagnostic is a defensible next investigation, after which move toward coupled/missing-vocabulary learning rather than endlessly polishing scalar gains.

## Audit, exact budgets, failures and reproduction

All plants are simulated on the owner's existing PC/.venv; these are not hardware robotics results. 'Plant experience' distinguishes actual benchmark environment transitions from rollouts inside a fitted model. The corrected shared experiment collects1056768 unique training transitions/12288 resets total. Its two-checkpoint test consumes33024 transitions/384 resets, never used to update a learner. The virtual-reuse diagnostic adds1056768 learned-model transitions/12288 virtual resets and5504 plant test transitions/64 resets, zero new plant training. Earlier PD pilot and excluded overlap pilot are additional development work, not hidden from the interaction record. Reproduction reruns repeat existing seeded instances and are not new independent scientific replicates.

The excluded first large run is `shared_experience_v12_seed_overlap_pilot.py`, full NPZ, gzip-compressed full JSON and timing. Its replicate spacing10000 overlapped candidate episode offsets spanning19101. It is not used for reported MCSEs. Corrected seeds use plantstride1000000000,replicatestride1000000,trainingoffset100000+100*index+task,testoffset500000+task,CEMbase; virtual runs add10000000000. Audit checks global role/replicate disjointness. True outcome similarity would not rescue the invalid independence assumption.

Independent theory actively challenged policy capacity, sensor assumptions, feedback likelihood and horizon metadata. The artifact audit uses separately implemented dynamics/controllers/sensitivities/filter/plan loss, reconstructs all24576 corrected/virtual training segments, all448 fresh test episodes and22016 plans, checks all384 fit records, CEM randomness/selection, seeds, budgets and clustered summaries. It distinguishes independent implementation from simply re-executing the source. The earlier PD pilot audit independently reconstructed800 episodes. Sources/results/reports are retained alongside the main experiments; no unresolved audit discrepancy may be treated as a release.

The raw generated metadata retains its original pre-audit execution status; the separate subsequent audit report/result records completed verification. The main and reuse experiments were each re-executed byte-for-byte for NPZ and result JSON. Timing is separately logged and is inherently nondeterministic. On the saved original corrected run, total wall sums are3.48s plant-training simulation,4.19s linear fits,5.79s nonlinear fits,15.29s linear-control tests,15.52s nonlinear-control tests; reuse virtual training3.43s. These are implementation-specific local measurements, not a fair runtime contest or scaling claim. BLAS threads limited to1; Python3.12.3,NumPy2.5.3,SciPy1.18.1, existing threadpoolctl; no environment changes.

~~~bash
.venv/bin/python research/shared_experience_v12.py --output-prefix /tmp/shared_v12
cmp research/shared_experience_v12_results.json /tmp/shared_v12_results.json
cmp research/shared_experience_v12_data.npz /tmp/shared_v12_data.npz
.venv/bin/python research/model_reuse_v12.py --output-prefix /tmp/reuse_v12
cmp research/model_reuse_v12_results.json /tmp/reuse_v12_results.json
cmp research/model_reuse_v12_data.npz /tmp/reuse_v12_data.npz
.venv/bin/python research/audit_shared_experience_v12.py
~~~

**EXISTING RESEARCH:** [de Boer/Kroese/Mannor/Rubinstein's CEM tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf) supplies established distribution/elite-based optimization. [Mania/Guy/Recht2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf) demonstrates serious simple policy search and emphasizes broad seed evaluation. Our algorithm is CEM, not ARS; eight-seed scalar cells are not comparable with their robotics results. OE, EKF, shooting/MPC and pathwise derivatives are conventional tools; no originality or modern benchmark superiority is claimed.

## Research direction audit and exact continuation

This milestone changes the evidence about constructed learning mechanisms: experience can improve behavior through observation-only scores, while the same full experience also supports conventional predictive fitting and better mean transfer under the present supplied model/planner. It does not establish a necessary architecture. It also gives a useful negative diagnostic: adding simulation rollouts with unchanged finite CEM is not automatically enough. Scientific consequences concern how data, vocabulary, computation and decision objectives must be distinguished when comparing learning.

The limitations are substantial: scalar fixed-family systems; known reset, control sign, order/timing/noise and functional vocabulary; no autonomously learned distinction language; no coupled bodies, unknown contact or genuine robot transfer; controller class/search not exhausted; no modern learner superiority. A release does not achieve the general-learning objective.

**Immediately continue:** One prescribed, independently checked exact-gradient optimization of the SAME controller family on learned dynamics, with fixed starts, virtual training/validation separation and NEW physical test streams, can constructively test whether finite CEM search is limiting. Success would show useful family capacity, failure would not prove incapacity. Do not add an unbounded ladder of optimizers. Then choose a consequential coupled or missing-function-vocabulary problem: derive competing minimal mechanisms, test from observation/action experience under explicit priors and strong applicable baselines, and retain negative evidence. Keep the Goal ACTIVE across publication.

---

# Historical archive: Complete original v11 ledger, including all preceding versions

# First-Principles Robot Learning: Living Research Ledger v11

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited information bounds, data/sensing ablation and a consequential correction to task-framing interpretation. Full v10 and every earlier ledger follow verbatim. V10 was published and verified on SSH origin/main at `450272cc1f34282b7dd2521db3972280982c2c7e` before this milestone. The persistent general-learning Goal remains ACTIVE.

## Principal result: question the task contract before adding learning machinery

**TESTED NEGATIVE:** In the selected slow linear actuator case, extending the same physical training trajectory from512 to8192 commands barely changes this controller's performance. Reducing current position-sensor SD tenfold helps, but strict stopping remains unreliable. The model family, noise laws and initial state remain supplied.

**PROVEN UNDER ASSUMPTIONS:** Known-parameter Gaussian conditioning gives a controller-independent terminal uncertainty floor and a strict upper bound on the protocol's stop probability. Even perfect position history cannot reveal the newest motor disturbance before its effect reaches position. Longer history and more offline training cannot recover that independent unseen innovation.

**CORRECTED INTERPRETATION / TESTED REPAIR:** Those valid bounds did **not** explain most of the original endpoint error. The executor already encoded a step70 cutoff, but its planner was not passed that constraint: it kept optimizing actions that would never be executed. Independent decomposition found that avoidable conditional-mean error dominated the average predicted endpoint MSE over the retained histories. Passing the already available executor deadline into the planner and removing unavailable planned actions reduces that bias by orders of magnitude, using the same model, data, sensors and optimizer. Merely increasing the predicted coast from12 to16 changes almost nothing. No new physical state or nonlinear feature is needed for this repair.

This extends v10's evidence rather than rewriting it: its actual numerical comparisons remain valid for its frozen pipeline, but its forced-coast results must be read as involving **unannounced command cutoff**. They do not establish an optimal stopping limit or a pure sensing failure. The earlier suggestion to prioritize sensor-selection machinery is superseded until planning actually respects the task's available actions.

The scientific gain is a falsified failure diagnosis and a constructive context repair, not a novel control algorithm or completion of general intelligent learning.

## Why this is the next question

V10's learned and true-parameter filters behaved similarly, while full-state diagnostics improved stopping. It was tempting to attribute the residual mainly to sensing. First distinguish three possibilities: inaccurate learned dynamics, genuine uncertainty unavailable in the current history, and a wrong description of the future task/action interface. They imply different repairs. More training, better sensors, and a corrected task contract are genuinely competing explanations here.

Use only the simplest case that supports exact uncertainty analysis: rho=.65, lambda=.7, c=0, beta=1.2 with lagged motor timing, initial x=v=m=0. This case was selected **after** v10's failures for explanation, not as a new independent benchmark for a general performance claim. Analytic tables also include lambda=.2; new control trials use only lambda=.7.

Physical law: `vnext=rho*v+m`, `mnext=lambda*m+beta*u+xi`, `xnext=x+vnext`, `o=x+e`. Training has xi=0, sensor SD .2. Active testing uses independent motor SD .03; after70 actions the next16 commands are zero and no new motor noise enters. The last active innovation remains. Future goals are not announced; the current goal sequence is revealed at steps0,6,28 as in v10.

## F47. Conditional information at the last possible action [PROVEN UNDER ASSUMPTIONS]

Let s=[x,v,m], A=[[1,rho,1],[0,rho,1],[0,0,lambda]], B=[0,0,beta]^T, Q=diag(0,0,sigma_xi²), C=[1,0,0]. Let H69 include **all position measurements through o69**, issued commands, independent policy coins and the supplied task information. The decision u69 is made using H69. There are no later nonzero commands.

With known parameters/initial state and independent Gaussian process/sensor noise, conditional s69 given H69 is Gaussian with mean mu69 and deterministic posterior covariance P69. This remains true for arbitrary causal observation-based control: condition on the observed history so each earlier input is a known shift; Gaussian prediction/update induction supplies the conditional law and Riccati covariance. Controls do not change this covariance in this declared linear additive-noise channel. An extra private sensor, anticipatory disturbance information or a state-dependent noise/measurement law would change the argument.

After one last active transition and16 disturbance-free coast transitions,

~~~text
E[x86-goal | H69] = C*A^17*mu69 + C*A^16*B*u69 - goal = b
V = C*A^17*P69*(A^17)'*C' + C*A^16*Q*(A^16)'*C'.
~~~

Thus `E[(x86-goal)^2|H69]=b²+V>=V`, and

~~~text
P(whole stop event | H69) <= P(|x86-goal|<=.3 | H69)
                        <= erf(.3/sqrt(2V)).
~~~

The interval probability is maximized at b=0. V is a lower bound on conditional MSE, not necessarily an achievable minimum with bounded commands. Even feasible endpoint centering does not optimize the whole path/velocity event. Future coast observations cannot retroactively improve the already-issued u69; the history cutoff is essential. These are elementary Gaussian-conditioning/control facts, not new theory or an optimality certificate for MPC.

| Position sensor SD | Terminal conditional SD | Endpoint/whole-stop ceiling |
| ---: | ---: | ---: |
| 0.2 | 0.586023 | 0.391297 |
| 0.05 | 0.482486 | 0.465913 |
| 0.02 | 0.439348 | 0.505285 |
| 0.005 | 0.406520 | 0.539468 |
| 0.0 | 0.399789 | 0.546984 |

The bounds are population upper bounds; finite trial fractions can exceed them by chance. They apply to the exact linear Gaussian protocol, not arbitrary nonlinear robotics or unknown noise laws. Knowledge of true parameters is more informative than finite independent training about those same parameters, so additional offline samples cannot remove these particular live-innovation terms.

## F48. Perfect positions still arrive too late for one motor innovation [PROVEN UNDER ASSUMPTIONS]

With exact x0:69 and known parameters, v69=x69-x68 and m68=v69-rho*v68 are exact. But

~~~text
m69=lambda*m68+beta*u68+xi68.
~~~

xi68 has not yet affected x69. Therefore P69=diag(0,0,sigma_xi²), even with arbitrarily long noiseless position history. Terminal error contains two independent unobserved innovations: xi68 with coefficient `(A^17)[0,2]` and xi69 with `(A^16)[0,2]`. For lambda=.7 this gives terminal SD .399789 and whole-stop probability upper bound .546984, versus .712264 with full current physical state known. Position precision cannot remove this timing gap. A timely additional motor-force measurement or changed action opportunities supplies different information; it is not equivalent to more precise positions. This conclusion depends on fresh independent motor noise and the declared lagged interface, not a universal requirement for force sensors.

## E28. More experience versus current sensing [TESTED]

`precision_information_v11.py` retains the exact12 v10 slow-linear fit seeds and all original512 commands/513 noisy positions. Each physical trajectory continues to8192 commands without a state reset. The extension uses independent seeded bounded held commands(1,3,8 steps), seedfit+4000000, and sensor SD .2. All old observations stay unchanged; the new first observation is at time513, so the boundary is neither duplicated nor overwritten. Raw-position linear OE is refit using the same bounds/starts as v10. These extra samples are a sixteenfold interaction increase, not a same-budget method comparison.

With short fits, long fits and true parameters, test position-sensor SD .2 or .02. The sensing intervention changes both actual noise and the supplied filter variance, with identical underlying standard-normal draws and motor noises. Training sensor SD stays .2. Same goals, test seeds, bounds, controller/optimizer and scoring as v10; no true state to ordinary controllers. There are144 episodes/10080 plans,12 fit-seed clusters; two task trials share a fit. The added training and improved sensor's physical/economic costs are not equated.

| Test sensor SD | Model knowledge | Active cost +/- clustered SE | Late position RMSE | Whole stops |
| ---: | --- | ---: | ---: | ---: |
| 0.2 | short512 | 0.310972 +/- 0.005353 | 0.652039 | 0/24 |
| 0.2 | long8192 | 0.310989 +/- 0.005343 | 0.651881 | 0/24 |
| 0.2 | known_parameters | 0.310986 +/- 0.005339 | 0.651956 | 0/24 |
| 0.02 | short512 | 0.298497 +/- 0.002079 | 0.482483 | 3/24 |
| 0.02 | long8192 | 0.298424 +/- 0.002056 | 0.482250 | 3/24 |
| 0.02 | known_parameters | 0.298428 +/- 0.002060 | 0.482232 | 3/24 |

Cost means/MC SEs average tasks within seed before summarizing12 seeds. Late position RMSE combines the final10 active and16 coast states, while endpoint MSE below concerns time86 alone; these metrics must not be conflated. All per-episode trajectories/selected plans,8192-row training data and fit records are retained. The .2 short-fit and known-parameter episodes exactly reproduce v10's counterparts.

Increasing data changes mean active cost by +.0000173 +/- .0000281 paired MC SE at sensor SD .2; this is evidence of little benefit in this case, not a formal universal equivalence result. Lowering sensor SD with the short fit changes it by -.0124751 +/- .0043141. Whole-stop outcomes remain0/24 or3/24; the small counts do not establish calibrated success guarantees or general benefit from sensor upgrades.

## F49. An uncertainty floor is not the cause of all observed error [DERIVED / TESTED]

The fixed-horizon planner at time69 still optimizes eight future commands before its predicted zero-command tail. The actual protocol applies u69 and then forbids all commands. Neither MPC's action function nor its fixed horizon receives the cutoff deadline. This is a planner/executor constraint mismatch in our benchmark implementation: future controls are being optimized that will not be available.

Independently re-filtering saved known-parameter episodes gives the b from F47. Over the24 retained known-parameter histories at each sensor level, at SD .2 mean b²=1.676571 while V=.343423: about83% of predicted endpoint MSE b²+V is conditional-mean error. At SD .02, b²=.603351 versus V=.193027, about76% is conditional-mean error. These are finite averages of conditional biases/risks, not exact population expected MSEs. Neither valid uncertainty floor explains that excess. Observed0/24 or3/24 whole stops cannot be attributed solely to those endpoint ceilings.

Counterfactually replacing just the final command by `u69-b/(C*A^16*B)` centers the conditional endpoint mean; every checked known-parameter case permits this within[-1,1]. This is a diagnostic holding earlier history fixed, not a claim that the replacement meets every stop constraint, is optimal for the full loss, or could use future information. It demonstrates that a substantial part of the existing bias was avoidable with the already available measurements.

## E29. Announce the deadline and remove unavailable planned actions [TESTED]

`deadline_control_v11.py` compares two added variants, both predicting16 zero commands after the planned control sequence. One keeps eight planned actions. The other receives the known time70 cutoff and uses `N=min(8,70-t)` action variables, truncating the warm plan causally. At t69 the one optimized command followed by16 zeros matches the actual available action schedule. All state/filter/smoother variables continue normally; no reset is introduced.

Use the same12 seeds, both goals, sensor SD .2/.02, short512-fit or true parameters, with no additional training. This adds192 episodes and13440 bounded optimizations. Only current goals and, for the aware variant, the shutdown deadline are supplied; future goal changes remain hidden. The shutdown time was already encoded in the executor. Passing it through the planning interface uses a supplied task constraint; it acquires no new external physical information, learns no new law and improves no sensor. The comparison is an exploratory diagnosis after v10, not an independent preregistered algorithm contest.

| Sensor SD | Same known-parameter model / metadata | Late position RMSE | Mean conditional endpoint bias squared | Whole stops |
| ---: | --- | ---: | ---: | ---: |
| 0.2 | known_parameters_unaware16 | 0.651956 | 1.676571 | 0/24 |
| 0.2 | known_parameters_aware16 | 0.331484 | 0.001556 | 1/24 |
| 0.02 | known_parameters_unaware16 | 0.482232 | 0.603351 | 3/24 |
| 0.02 | known_parameters_aware16 | 0.231494 | 0.000664 | 6/24 |

The original12-step predicted coast and fixed-eight/16-step coast have nearly identical results. Removing unavailable future controls changes the conditional mean dramatically while **leaving V unchanged**. The learned512-row and known-parameter variants remain similar. This refutes the immediate need for additional physical model complexity or many more training examples to correct this particular bias.

The primary active tracking cost changes little because it largely scores the earlier70-step tracking phase. Large improvements in coast error need a separate task score; optimizing/reporting only a convenient primary scalar can obscure a capability failure. Stopping still fails on many trials; conditional uncertainty, path constraints, the finite quadratic objective and bounded/causal control remain. The scalar terminal bound is not a complete characterization of achievable whole-stop probability, and the deadline-aware controller is not asserted optimal.

## Audit, corrections, and existing research

Independent theory derived the adaptive-policy conditioning, exact-position two-innovation floor, and the b²+V decomposition. It challenged the initial sensor-first interpretation and independently re-filtered saved data in `audit_precision_cutoff_v11.py`/JSON. `v11_theory_independent.md` preserves the earlier suggestion and its explicit correction, rather than silently replacing research history.

The artifact auditor independently constructed batch Gaussian conditioning, including Moore-Penrose/nullspace handling for exact position measurements, rather than just rerunning the same Riccati loop. It reconstructed nested trajectories without hidden-state resets, all144 experience/sensing episodes,10080 plans, filter histories, metrics and clustered summaries; source/results/report are `audit_precision_information_v11.py`/JSON/Markdown. Deadline-aware experiments and conditional-mean diagnostics are independently reconstructed in `audit_deadline_control_v11.py`/JSON and its audit report. Audit scope distinguishes fresh derivations/implementations from repeated execution.

**CORRECTED BEFORE RELEASE:** A JSON field originally named minimum conditional MSE was renamed conditional MSE **lower bound**, since bounded centering need not be feasible generally. Known-parameter variance is not a success guarantee. The unannounced-cutoff issue corrects interpretation of v10/v11 initial stopping failures; it does not invalidate the actual saved numbers or Gaussian bounds. The final source/result metadata records audit completion; the earlier experiments and released ledgers remain preserved.

**EXISTING RESEARCH:** [Kalman1960](https://people.math.harvard.edu/archive/116_fall_03/handouts/Kalman1960.pdf) establishes recursive estimation-error covariance and optimal linear prediction. [Georgiou and Lindquist, The Separation Principle in Stochastic Control, Redux](https://arxiv.org/abs/1103.3005) examines the subtleties of information under feedback; our finite discrete-time induction explicitly supplies the history/causality conditions. [Tedrake's MPC/direct-shooting notes](https://underactuated.mit.edu/trajopt.html) describe planning with specified dynamics/constraints. These precedents preclude novelty claims. Gaussian filtering or LQG references do not turn this bounded, fixed-horizon, multi-step stopping event into an optimal-control theorem. No advantage over modern general learners is established.

## Reproduction and drift audit

Owner's existing `.venv`: Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No installation or service changes.

~~~bash
.venv/bin/python research/precision_information_v11.py > /tmp/precision_information_v11_check.json
cmp research/precision_information_v11_results.json /tmp/precision_information_v11_check.json
.venv/bin/python research/deadline_control_v11.py > /tmp/deadline_control_v11_check.json
cmp research/deadline_control_v11_results.json /tmp/deadline_control_v11_check.json
.venv/bin/python research/audit_precision_information_v11.py > /tmp/audit_precision_information_v11_check.json
.venv/bin/python research/audit_precision_cutoff_v11.py > /tmp/audit_precision_cutoff_v11_check.json
.venv/bin/python research/audit_deadline_control_v11.py > /tmp/audit_deadline_control_v11_check.json
~~~

This cycle resolves an operational uncertainty relevant to learning: when a task fails, extra experience or model capacity may be irrelevant because the learner/planner is solving the wrong future-interaction problem. The experiments distinguish actual parameter uncertainty, live hidden innovations and omitted action-availability context. Correcting the latter is simpler and more effective than building new sensing-policy machinery in this case.

The mechanism still does not autonomously discover the failure diagnosis or task metadata; independent scientific investigation supplied the correction. It does not learn an observation language, select among unknown physical families, infer preferences from nothing, or demonstrate nonlinear articulated-body intelligence. More isolated covariance identities would risk drift. Further work should test whether acquired predictive consequences support reliable decisions when the function vocabulary, coupled dynamics or action availability must themselves be inferred from experience, using strong established alternatives rather than imposing an architecture prematurely.

**Immediately next:** Reassess the complete learning loop beyond this supplied scalar family. Preserve the distinction between learned experience and supplied task/interface facts. Prefer a decisive test of which missing distinction actually changes successful action: task-context inference across episodes or coupled dynamics requiring model revision are legitimate alternatives. Derive the simplest discriminating case first, compare conventional simple mechanisms, and only add complexity if the simpler explanation fails. The broader Goal remains active; publication is not a stopping condition.

---

# Historical archive: Complete original v10 ledger, including all preceding versions

# First-Principles Robot Learning: Living Research Ledger v10

**Date:** 2026-10-20, Europe/Berlin. **Status:** constructed and independently audited experience-to-control pipeline; retained failed numerical pilot and stronger baseline repair. Complete v9 and all previous ledgers follow verbatim. Prior v9 commit `426524a38a041b7e566fece00831b073cfdddab6` was pushed and verified on existing SSH origin/main before this cycle. The persistent scientific Goal remains ACTIVE.

## Outcome: useful behavior without resolving physical names

**TESTED:** Conventional output-error identification from noisy position, state filtering and bounded receding-horizon planning form a working experience-to-goal-action pipeline on four declared actuator-memory plants. Controllers handle new goal sequences without resetting hidden velocity or motor force at switches. Learned linear models already track well, including the nonlinear plants. This is construction with supplied dynamics families and standard methods, not discovery of a general architecture.

**TESTED NEGATIVE:** On the nonlinear plants, nonlinear models' large improvement in clean command-sequence prediction does not reliably improve the selected closed-loop tracking objective. Accurate physical model parameters are not the only bottleneck: true-parameter filters have stopping failures too, and more informative true-state access changes performance. Finite-horizon planning, state estimation, process noise and the success definition remain distinct limitations.

**CORRECTED BEFORE RELEASE:** Initial L-BFGS-B planning frequently failed to converge, even in convex linear cases. The complete failed source/results remain. Replacing only the action optimizer with bounded least squares removes all final convergence failures and restores linear pole-swap invariance numerically. A weak coarse PD grid was also challenged by broader continuous tuning using only learned dynamics; it improves some comparisons but remains imperfect. Neither correction is concealed or called a preregistered method contest.

The strongest next uncertainty is whether obtaining more training examples is useful when current state information and future disturbances dominate goal precision. Adding active model-disagreement machinery is premature until this is resolved.

## First-principles question and a fixed experimental contract

V8/v9 showed that component labeling can be impossible even while command consequences are predictable. Do useful actions actually require those labels? Conversely, does a more accurate nonlinear predictor necessarily make better goal decisions? Test behavior directly, rather than score hidden parameter names.

The physical plant and observation channel are

~~~text
vnext=rho*v-c*tanh(v)+m
mnext=lambda*m+beta*u+xi
xnext=x+vnext, o=x+e.
~~~

Four plants: rho=.65, beta=1.2, lambda=.2 or .7, c=0 or .2. Units/timestep/tanh scale, model order, lagged timing and independent Gaussian noise laws are supplied. Training has no process noise and position-sensor SD .2. Test active control has motor-process SD .03 and sensor SD .2. Every episode begins at known x=v=m=0; there is no hidden-state reset at goal changes. These assumptions remain substantial built-in structure, not discoveries.

Each plant has12 independent fit seeds `20261020+1700000+10000*plant_index+100*i`. For each, generate512 commands in[-1,1] by choosing independent uniform amplitudes held for randomly selected1,3 or8 steps, truncated at512. Both model families receive **identical data within plant/seed**:513 noisy positions and512 commands. Command streams differ across plants. Neither family receives latent v/m, simulator parameters, sensor realizations or test outcomes.

Learned nonlinear OE uses the frozen v9 four-parameter fitting procedure. Learned linear OE fixes c=0, fitting its two poles and beta by the same raw-position objective, parameter bounds, three starts and evaluation budget. Linear labels are not scored. All fit objectives, starts and chosen models are saved. The experiment supplies the function vocabulary; no operator discovery, automatic model-order growth or online parameter revision is demonstrated.

## F44. The decision-relevant accuracy requirement [PROVEN UNDER ASSUMPTIONS]

At a given observable history h, let J(a) be the true expected trajectory loss for each admissible future action sequence, and Jhat(a) a learned estimate. If `sup_a |Jhat(a)-J(a)|<=epsilon` and the selected action sequence is within delta of the **global** infimum of Jhat, then

~~~text
J(selected) <= inf_a J(a)+2*epsilon+delta.
~~~

Insert Jhat on either side and use the uniform bound twice; an approximate infimum sequence gives the same inequality if a minimizer does not exist. No physical-coordinate identification is needed for this statement. But a one-step RMSE on a sampled distribution does not supply the uniform bound over sequences selected by optimization, and a projected gradient is not a global delta certificate for a nonconvex problem. This elementary established approximate-optimization fact is not a new learning theorem.

V10's planner uses a deterministic certainty-equivalent rollout with a fixed current goal; it does not compute the true noisy expected loss or anticipate unknown future goal switches. No tested epsilon/delta or repeated-history error bound is supplied. Thus the proposition explains the uncertainty that matters; it is not a guarantee for this implemented closed-loop controller.

## F45. Exact linear prediction/control need not choose a pole label [PROVEN / TESTED]

At c=0 let `q=m+(rho-lambda)*v` and swap rho/lambda. With s=[x,v,m], A=[[1,rho,1],[0,rho,1],[0,0,lambda]], B=[0,0,beta]^T and C=[1,0,0], the triangular transform T obeys `A_swapped*T=T*A`, `T*B=B`, `C*T=C`, and `T*Q*T^T=Q` for motor-only Q. Transforming prior mean/covariance therefore preserves Kalman observation/velocity predictions. Exact known-zero initial state/covariance already maps correctly.

Every planned loss using x,v,u and the same zero-command tail is invariant. Penalizing arbitrary hidden m instead would break that property without defining a physical task. Independently derived similarity/filter identities and a paired implementation check verify this. Final swapped-label MPC actions and physical states differ by about1.63e-14. The initial truncated solver differed by .010663 in action; the exact objective had never changed. Numerical optimization artifacts do not provide evidence identifying the poles. Nonlinear EKF is not claimed invariant to arbitrary nonlinear coordinate changes.

## Controller construction and fair access

All predictive controllers use the same eight bounded action variables followed by12 prescribed zero commands, minimizing the sum of `(x-goal)^2+.3*v^2+.03*u^2` across the20 predicted transitions. Only the first chosen command is executed; the remaining plan shifts for the next warm start. The planner sees only the current goal. It gets no future target schedule or true state except in the explicitly labeled state oracle.

Action sensitivities follow the actual lagged timing: current u changes only mnext, not next x/v; its first position effect is two steps later. Final action optimization uses bounded residual least squares (TRF), analytic Jacobians, max80 evaluations, ftol/xtol1e-10 and gtol1e-8. Every selected8-action plan, warm/selected cost, projected-gradient norm, termination record and fallback is retained. This solves the linear convex cases accurately; nonlinear global optimality/stability is not established.

The observer is a conventional EKF with state[x,v,m], position measurements, known noise scales and exact-zero initial mean/covariance. Measurement update precedes command selection; prediction follows the issued command. The transition Jacobian is `[[1,f'(v),1],[0,f'(v),1],[0,0,lambda]]`, Q=diag(0,0,.03²), and position covariance uses a Joseph update. Estimated parameters remain frozen during tests. Parameter uncertainty and nonlinear future noise effects are ignored by this certainty-equivalent design.

Comparators are fixed PD; PD tuned offline on the learned nonlinear model; learned linear and nonlinear MPC; and a fixed nominal-parameter MPC(.6,.4,.1,1) with no training. The nominal comparison concerns that specific mismatch, not every robust no-learning controller. Two privileged diagnostics have true plant parameters with either the same observation-only EKF or the full current physical state. They are not same-information learning competitors.

PD uses causal smoothed position and a second smoothed difference, then clips `kp*(goal-smoothed_position)-kd*smoothed_velocity` to[-1,1]. The initial18-gain grid is kp=.08,.2,.5, kd=.2,.6,1.2, smoothing=.25,.6. Its two synthetic goal schedules are(1,-1,.5) and(-1,1,-.5). Each virtual rollout uses only the learned nonlinear model and supplied noise law; **no extra physical observations** tune PD. Thus this is model-tuned feedback, not a model-free baseline. It optimizes active tracking, whereas the MPC surrogate also includes predicted coasting; stopping comparisons alone cannot establish a planning advantage.

After poor coarse-grid PD pilot results, an exploratory baseline repair broadens bounds to kp[.002,.5], kd[0,4], smoothing[.05,.95], with Powell optimization from the coarse winner and(.02,1,.8), up to350 evaluations per start. Selection also retains the coarse candidate. One version tunes active tracking, another adds five times normalized late squared position/velocity error, explicitly testing a stopping-aware objective. All tuning remains virtual on the same fitted model and separate goal schedules. This increases offline compute, not physical data. Thirty-eight Powell starts hit their evaluation limit and are retained. Gain optima are not certified globally, and no claim against the best possible PD controller follows.

## E26. New goal sequences, active disturbances and genuine coasting

Each fitted model is tested on goal sequences(2,-2,1) and(6,-4,3), revealed at steps0,6,28. There are70 active steps, then16 forced zero commands at the final goal. Sensors remain noisy throughout. Motor disturbances stop during the final coast so the score tests stored motion/actuation rather than a continuing uncontrollable random walk. The last active innovation remains in motor state; the coast is not a known-rest reset. Turning disturbance off is a declared diagnostic evaluation intervention, not a universal real-world stopping guarantee.

Test seeds are fitseed+2000000+task_index. Exogenous sensor/motor streams are paired across controllers, but their actions and observation histories differ. No test goal or noise is used for fitting/tuning. There are96 task trials per controller, with two sharing each of48 learned models. The main run retains672 episodes; the stronger PD repair adds192. Arrays of actual position, velocity, hidden motor state, observations, estimated states, commands and goals are saved. True state enters only the physical simulator/scorer and the explicitly privileged state oracle.

Frozen primary score: mean active `(position_error²+.3*v²+.03*u²)` divided by `1+mean(active_goal²)`. Stop success requires position error<=.3 and |v|<=.15 for every final5 active and16 coast samples. Also report late position/velocity error, action energy and settling diagnostics. The saved settling delay is a **zero-based action-index delay**, not elapsed physical transitions. Success rates are empirical task counts, not calibrated confidence guarantees.

Each cost mean first averages the two tasks within each fit seed, then averages12 independent seeds; MC SE uses those12 seed clusters. Counts out of24 share fitted models and are not24 independent Bernoulli fits.

| lambda | c | Nominal MPC | Linear learned MPC | Nonlinear learned MPC | Continuous PD active |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.0 | 0.327857 | 0.312584 | 0.312605 | 0.377893 |
| 0.2 | 0.2 | 0.300083 | 0.303825 | 0.312742 | 0.372196 |
| 0.7 | 0.0 | 6.112022 | 0.310972 | 0.310981 | 0.751704 |
| 0.7 | 0.2 | 1.371502 | 0.300641 | 0.305501 | 4.082810 |

Cost table shows means. Full MC SEs and individual outcomes are in JSON. The broader PD repair substantially improves the coarse PD comparison in some slow-actuator cases, but can still be unstable or get stuck in poor synthetic tuning basins. This limits interpretations of relative PD performance.

| lambda | c | Linear MPC | Nonlinear MPC | PD active / stop-aware | True-model EKF | True-state oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.0 | 8/24 | 8/24 | 4/24 / 13/24 | 8/24 | 23/24 |
| 0.2 | 0.2 | 16/24 | 18/24 | 5/24 / 18/24 | 18/24 | 24/24 |
| 0.7 | 0.0 | 0/24 | 0/24 | 1/24 / 0/24 | 0/24 | 13/24 |
| 0.7 | 0.2 | 1/24 | 1/24 | 2/24 / 0/24 | 1/24 | 22/24 |

Stopping table gives successes/24. True-parameter EKF and learned-model EKF behave similarly here. True-state access improves these particular controllers, but even that privileged controller can fail. No information-theoretic impossibility or uniquely causal attribution to state estimation follows just from these comparisons: planning horizon, objective, noise, filtering approximation and forced-coast timing all matter.

## E27. Prediction accuracy and control performance disagree [TESTED]

Each model also predicts a fresh clean256-command uniform[-1,1] replay from known rest, seedfit+3000000. This evaluation never selects models or gains. It measures prediction under a changed temporal command pattern, not the controller's future conditional objective.

| lambda | c | Linear replay velocity RMSE | Nonlinear replay velocity RMSE | Linear minus nonlinear control cost +/- paired SE |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.0 | 0.004187 | 0.004213 | -0.000022 +/- 0.000011 |
| 0.2 | 0.2 | 0.115987 | 0.004293 | -0.008918 +/- 0.001798 |
| 0.7 | 0.0 | 0.001158 | 0.001269 | -0.000009 +/- 0.000006 |
| 0.7 | 0.2 | 0.202348 | 0.001878 | -0.004860 +/- 0.001769 |

The nonlinear model improves command-replay prediction by large factors on c=.2, yet its primary control cost is slightly higher than the simpler learned linear model in both nonlinear cells. This is a finite-grid result, not a theorem that linear models suffice for general nonlinear robotics or that prediction is useless. It falsifies the inference that lower ordinary replay RMSE necessarily improves this control pipeline's selected task metric.

## F46. Even a true-state controller cannot know the final fresh motor innovation [PROVEN UNDER ASSUMPTIONS]

For c=0, condition on all information before the final active motor innovation xi69, including the chosen u69 and policy coins. At final coast time86, `x86-goal=mu+h16*xi69`, where `h16=(A^16)[0,2]`. xi69 is independent Gaussian SD .03; future process noise is zero. The probability of any length.6 interval under this conditional Gaussian is maximized when mu=0. Thus **any nonanticipating controller**, even a true-state one, has whole-stop success probability no larger than

~~~text
P(|x86-goal|<=.3) <= erf(.3/(sqrt(2)*.03*abs(h16))).
~~~

For lambda=.2/.7, h16=3.567238258/9.406435335, giving population bounds .994941569/.712264292. The whole event includes additional constraints and may be much less probable. Finite empirical fractions can exceed a population bound by chance. This result is specific to linear Gaussian noise, the fixed shutdown timing and success tolerance; it is not asserted for c=.2. It explains why even perfect model/state access does not imply100% strict coast success and why adding offline training cannot reveal that future noise sample.

## Independent audit, failed pilot, and existing work

The theory auditor independently derived lagged action sensitivities, filter/physical-label invariance, noise/coast counterexamples and the final-innovation bound; proof and scope are in `v10_theory_independent.md`. The artifact auditor independently eliminated motor state and reconstructed physical trajectories, observation streams, EKF estimates, PD virtual tuning, prediction scores, fit objectives, every task metric and clustered summary. It checked all33600 final selected plans, warm/selected costs and projected gradients from saved action vectors. Full audit sources/results are `audit_goal_control_v10.py` and JSON plus `audit_goal_control_v10.md`; the stronger PD audit is retained as `audit_goal_control_PD_v10.py` and its JSON.

The **failed pilot** source/complete JSON are `goal_control_v10_lbfgs_pilot.py` and `_results.json`, with independent audit JSON. It had large projected gradients, many40-iteration terminations and parameter-order-dependent actions despite identical linear objectives. After diagnosis, only the action optimizer changed; training rows/fits, plant grid, goals, bounds, noise streams, primary metrics and success tolerance remained fixed. The final33600 solves all report convergence, no fallback, maximum projected-gradient norm about3.17e-5. This is numerical verification, not a nonlinear global-optimum certificate. Predictions/plant replays and all negative outcomes remain available.

Before seeing results, the independent auditor also exposed two score-design confounds: a three-step coast misses slow stored force, and continued process noise during forced zero command measures uncontrollable drift. The frozen executed protocol uses12 planned and16 evaluated coast steps, and distinguishes noisy active control from disturbance-free coast. The additional continuous/stopping-aware PD search is explicitly **post-pilot exploratory baseline repair**, preventing a claim against an unnecessarily weak gain grid. No released history is overwritten.

**EXISTING RESEARCH:** [Tedrake's trajectory-optimization notes](https://underactuated.mit.edu/trajopt.html) describe direct shooting, exact gradients and receding-horizon control. [Nguyen et al., Output-Feedback MPC with Online Identification](https://arxiv.org/abs/2009.10631) is a serious observable-history alternative to a separate latent observer, including sensor noise and task-relevant identification; not implemented here. [Zhou and Tzoumas, Simultaneous System Identification and MPC with No Dynamic Regret](https://arxiv.org/abs/2407.04143) provides richer online learning/control under explicit technical assumptions and real quadrotor experiments. Their results do not validate ours; they establish how much broader existing work already is. Our construction is an elementary conventional baseline with known structure, not a new learning/control method or competitive robotics result. [Schoukens and Ljung](https://arxiv.org/abs/1902.00683) already distinguish model structure, noise treatment and application-specific validation.

## Reproduce and continue

Existing owner's PC and `.venv` only: Python3.12.3, NumPy2.5.3, SciPy1.18.1, threadpoolctl. No installation/services. Full outputs are intentionally retained, including the failed pilot, so the artifact files are larger than earlier releases.

~~~bash
.venv/bin/python research/goal_control_v10.py > /tmp/goal_control_v10_check.json
cmp research/goal_control_v10_results.json /tmp/goal_control_v10_check.json
.venv/bin/python research/goal_control_PD_v10.py > /tmp/goal_control_PD_v10_check.json
cmp research/goal_control_PD_v10_results.json /tmp/goal_control_PD_v10_check.json
.venv/bin/python research/audit_goal_control_v10.py > /tmp/audit_goal_control_v10_check.json
~~~

This cycle closes an earlier construction gap: acquired command consequences are used for actual goal-changing feedback control under hidden motor memory. It also rejects a tempting nonlinear-model success narrative, and independently catches numerical and weak-baseline confounds. But it supplies the state dimension, observation model, noise scales, function family, controller class, objective and exploratory data. No autonomous language expansion, coupled articulated body, unknown observation law or competitive modern general-learning framework has been demonstrated.

**Immediately next:** Test whether more offline experience versus more informative current sensing changes the observed precision failures. Begin with the exact linear Gaussian case: derive the best conditional uncertainty after all observation history, propagate it through the stopping decision, and compare its information limit to actual learned/oracle controllers. Then run an equal-protocol data-versus-sensing ablation, keeping model family and planner fixed. Additional samples cannot reveal a future innovation, but finite learned-model error and recoverable state uncertainty need distinct tests. Avoid interpreting another parameter fit as a solution to the task; the question is which available experience or measurement changes useful decisions. Broader mechanism construction remains OPEN, and the Goal stays active after publication.

---

# Historical archive: Complete original v9 ledger, including all preceding versions

# First-Principles Robot Learning: Living Research Ledger v9

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

The retained eight-command preparation yields common velocity 6.341128. Over the next12 removal steps the velocity-response distance is 0.273386, giving expected-RMSE lower bound 0.136693. Both full response vectors are saved.

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

| Policy | Max command difference | Max position/observation difference | Max gauge error |
| --- | ---: | ---: | ---: |
| open_loop | 0 | 3.98e-13 | 1.33e-15 |
| feedback | 5.55e-16 | 3.41e-13 | 1.33e-15 |
| zero_atom | 5.55e-16 | 2.84e-13 | 8.88e-16 |

Nonzero total-force differences are retained alongside negligible command/position differences. Ordinary zero commands do not break the equivalence. The original saved v9 pilot remains unchanged as preliminary history and was independently reconstructed from an eliminated observable recurrence; the released code adds explicit positions, replications and physical-removal semantics.

## E24. Clean recovery and limited excitation [TESTED]

`nonlinear_identification_v9.py` tests c=0,.01,.2 crossed with command SD .03,.3,1.2, with 12 independent 4096-command sequences per cell. Seeds `20261020+1500000+10000*c_index+1000*amplitude_index+i`. Exact positions are differenced; 4094 regression rows are retained, using t=1..4094. There is no process noise. The exact tanh scale and complete candidate vocabulary are supplied.

Maximum coefficient error is 1.48e-13. At command SD .03, smallest/largest singular-value ratios range 0.000596026 to 0.000929914, versus substantially better conditioning at larger amplitudes. Exact recovery under negligible roundoff does not remove finite-noise weak identification.

All 108 clean coefficient vectors are saved, including undefined c=0 label ratios as null rather than successes. These are arithmetic/excitation checks, not evidence that finite-noise labels are reliable.

## E25. Noisy position and unseen command amplitude [TESTED]

Six conditions c=0,.02,.2 crossed with command SD .08,.7; 12 independent datasets each. Seeds `20261020+1500000+100000+10000*c_index+1000*amplitude_index+i`. Each has 512 open-loop commands, 513 noisy positions, iid position-sensor SD .2, **zero process noise**, and known x0=v0=m0=0. Every estimator receives the same observations and commands. No hidden states/noise realizations enter fitting.

Three conventional estimators:

- Unrestricted OLS on the five noisy-difference regressors from F42.
- Structured one-step least squares with the four physical parameters, same tanh family, bounds and initial starts as output error, but using noisy differenced positions as regressors.
- Raw-position output error in the four-parameter family. Exact forward parameter sensitivities are implemented and independently checked by finite differences.

Both structured methods use rho/lambda in [.05,.95], c in [0,.4], beta in [.3,2], three fixed starts `(.7,.3,.1,1)`, `(.3,.7,.02,1)`, `(.8,.6,.3,1.4)`, and at most160 function evaluations per start. These bounds keep the passive velocity map contractive and the actuator recursion stable, even when an estimated c exceeds rho; integrated position is not contractive. Selection uses only minimum training SSE. Every optimizer record and every seed is retained. Convergence is not a global-optimum certificate; the methods minimize different objectives. Unrestricted OLS additionally lacks parameter ties/stability bounds, so its comparison conflates observation treatment with those restrictions. The matched structured comparator controls those family/bounds differences.

After fitting, two fresh 256-command sequences use either training amplitude or SD .9. Both tests start from known zero physical states and have no physical/sensor noise; latent velocities are used only to score prediction. This is clean sequence replay, not filtering unknown process noise, goal control, or on-hardware learning. Evaluation seed streams continue after training draws; neither evaluation enters fit selection.

| c | Train SD | OE matched | Structured matched | OE SD .9 | Structured SD .9 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.08 | 0.006893 | 0.153633 | 0.181307 | 1.670807 |
| 0.0 | 0.7 | 0.004680 | 0.268586 | 0.005716 | 0.327806 |
| 0.02 | 0.08 | 0.003496 | 0.147167 | 0.139296 | 1.700865 |
| 0.02 | 0.7 | 0.006077 | 0.254384 | 0.008131 | 0.335560 |
| 0.2 | 0.08 | 0.006698 | 0.124989 | 0.234140 | 1.504885 |
| 0.2 | 0.7 | 0.006547 | 0.211823 | 0.009084 | 0.288947 |

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

# First-Principles Robot Learning: Living Research Ledger v8

**Date:** 2026-10-20, Europe/Berlin. **Status:** independently audited full input/output nonidentifiability, component-transfer limitation, noisy feedback falsification and constructive added-information tests. **Code/results:** `research/actuator_memory_v8.py`, `research/actuator_memory_v8_results.json`. Full v7 and all prior history follow verbatim. V7 was verified on SSH origin/main at `3199066fdca516cc56e369d16a781cf1eb5c4d07` before this investigation began.

## Consequential limitation beyond compressed statistics

**PROVEN under assumptions:** For a linear mechanical mode driven by an unobserved first-order actuator, every command-to-position law identifies only the unordered pair of time constants. Swapping the mechanical and actuator poles preserves **all** observable trajectory distributions under every admissible observation-based adaptive command policy, including arbitrary zero-command patterns. This is a full observation/intervention equivalence, stronger than v5/v6's first-order-summary blind spots.

**TESTED / CORRECTED interpretation:** Zero **command** does not imply zero delivered physical actuation. Applying v7's one-state passive-coefficient interpretation to an actuator with memory returns the sum of two poles, here1.05, although both actual physical modes are stable (.65,.4). V7's conditional theorem remains valid under its stated F_t(0)=0/one-state passive-law assumptions; this experiment shows why those assumptions need evidence.

**PROVEN / TESTED:** Standard AR2-IV learns useful predictive dynamics and an unordered pole set from the same observations. It cannot label which pole belongs to mechanics. An unused lagged intervention moment exposes the inadequate AR1 representation. A calibrated motor-force measurement or a physical intervention that truly disconnects force breaks the label ambiguity.

**TRANSFER LIMIT:** Two indistinguishable training worlds require different predictions after a labeled motor replacement. No learner using only the original command/position channel can guarantee that component-transfer prediction in both worlds. This does not prevent within-interface prediction/control; the AR2 model demonstrates it.

The limitation is classical commuting-cascade/realization mathematics, not a novel theorem or proof of general intelligence. Its relevance is a concrete change to this learning program: identify experimentally warranted dynamical distinctions before assigning physical component names, and test zero-input semantics rather than infer them from command labels. The broader autonomous-learning Goal remains active.

## Question, assumptions and competing explanations

The initially planned next experiment was nonlinear passive-drift fitting from coast trajectories. Before choosing a nonlinear model, derive the smallest alternative explanation for unusual coast curves: hidden actuator force. A command may set a motor target to zero while current, torque or pressure continues to decay.

Consider the declared normalized discrete-time system

~~~text
m_(t+1)=lambda*m_t+beta*u_t+xi_t
v_(t+1)=rho*v_t+m_(t+1)
x_(t+1)=x_t+v_(t+1)
o_t=x_t+e_t.
~~~

Here m is delivered motor force in velocity-increment units. Only issued u and noisy o are observed. Both poles have magnitude<1; beta!=0. Initial v0=m0=0, or compatible unknown initial laws generated by the same past command/noise history. xi is independent actuator-process noise; e is exogenous position-sensor noise. The command policy may depend on all past/current observations and independent random coins, but not unobserved true states. No force/current sensor, independently certified motor-state reset at nonzero velocity, pole-order prior or physical force-disconnect capability is initially available.

Two explanations are genuinely distinct:

1. Mechanical persistence is .65 and motor persistence .4.
2. Mechanical persistence is .4 and motor persistence .65.

An apparent nonlinear or history-dependent coasting curve is a third possible explanation in a wider model family. The two linear worlds alone already falsify any claim that command-only coasting identifies the mechanical pole. More elaborate nonlinear fitting is unnecessary to decide that claim.

## F36. Complete command/position equivalence [PROVEN]

Since `m_t=v_t-rho*v_(t-1)`, elimination gives

~~~text
v_(t+1)=(rho+lambda)v_t-rho*lambda*v_(t-1)+beta*u_t+xi_t.
~~~

Only S=rho+lambda and P=rho*lambda occur. Swapping rho and lambda leaves the recurrence unchanged. The zero initial state corresponds to v_(-1)=v0=0. Couple the worlds using identical xi,e and policy random coins. If their observed histories match through t, an admissible policy issues the same u_t; the common recurrence then gives the same next velocity/position/measurement. Induction proves identical observable paths, hence identical laws, for every such policy. The result includes Gaussian probes, randomized zero atoms, nonlinear feedback, arbitrary deterministic commands, changed amplitudes, and arbitrary zero-command runs.

The hidden motor forces generally differ even when all measurements match. More data, larger models, state-first versus response-first framing, neural predictors, or any computation over the same observations cannot resolve the component label in both worlds. Quantitatively any estimate T of physical rho has equal-prior average MSE `E[(T-.525)^2]+(.65-.4)^2/4 >= .015625`; worst-world MSE is at least .015625 and root-MSE at least .125, irrespective of data quantity. A correct minimal predictive state realization is still available; unique physical component interpretation is not.

**Scope:** Identical arbitrary nonzero physical initial states are not automatically compatible. A known independently prepared v0!=0,m0=0 would break the swap. Known mechanical process noise added directly to vnext can also break distributional equivalence: eliminating it introduces a lambda-dependent colored component. Our xi enters the motor state, giving the same eliminated additive noise in both worlds. Additional force bounds/calibration, pole-order priors, force/current observations or labeled component interventions are extra information, not consequences of the original IO channel.

## F37. Zero-command weighting can return an apparently unstable coefficient [PROVEN]

Let measured displacement D_t=v_t+e_t-e_(t-1), and Y_t=D_(t+1). Then

~~~text
Y_t=S*D_t-P*D_(t-1)+beta*u_t+epsilon_t
epsilon_t=xi_t+e_(t+1)-(1+S)e_t+(S+P)e_(t-1)-P*e_(t-2).
~~~

The position-error part is MA3, not the MA1 error of a one-state velocity law. Sequentially centered randomized probes remain orthogonal to these additive errors under the stated assumptions. Let V=z_(t-1), and use v7's affine Gaussian cancellation weight or actual zero-command weight. They remove the **current** beta*u term. But `E[V D_(t-1)]=0`, while `E[V D_t]=beta*sigma²`. Therefore the v7-style ratio has population target

~~~text
E[V*w*Y]/E[V*D]=S=rho+lambda.
~~~

With .65 and .4, the value is1.05. This is not evidence of a genuinely unstable physical mode. The omitted lag term `-P Dprev` stabilizes the second-order dynamics. A single well-estimated coefficient is not a complete dynamical law or validated physical concept.

Likewise a just-identified constant-gain AR1-IV fit using instruments[zprev,zcurrent] converges to (S,beta). Its fitting moments all vanish despite omitted memory. The unused lag-two probe detects the failure:

~~~text
E[z_(t-2)*(Y-S*D-beta*u)]/sigma²=-P*beta=-.312.
~~~

This suggests a concrete minimal diagnostic: test additional intervention-history moments instead of treating success on the fitted moments as a sufficiency certificate. It is an established overidentification/model-order principle, not a new universal state-discovery algorithm.

## F38. Predictive realization versus physical labeling [PROVEN / EXISTING RESEARCH]

Fit the correct AR2 regressors `[D,Dprev,u]` using instruments `[zprev,zprev2,zcurrent]`. The instrument relevance structure includes `E[zcurrent*u]=sigma²`, `E[zprev*D]=beta*sigma²`, `E[zprev2*Dprev]=beta*sigma²`, with the corresponding pre-command zero entries. Thus the population system is full rank for beta!=0 and positive probe variance. Independent motor/sensor errors give valid IV moments even though raw D has correlated measurement errors.

The estimated characteristic polynomial `r²-S*r+P` yields the unordered poles. There is no evidence in these data that its larger root should be named mechanical. A predictor can transfer to new command sequences within this interface without solving that impossible labeling problem.

If calibrated measured motor force `M=mnext+independent force-sensor error` is added, then

~~~text
rho=E[zprev*(Y-M)]/E[zprev*D].
~~~

This identifies the physical mechanical coefficient. The extra channel and its unit scale are privileged additional information. If the scale were unknown, it would require joint estimation or another calibration, not literal subtraction. Alternatively a physical intervention enforcing total delivered force=0 gives vnext=rho*v. It differs from issuing u=0, under which motor state still decays.

## F39. A precise component-transfer lower bound [PROVEN]

Suppose after training one is told that the motor pole is physically replaced by .2, preserving mechanics and beta. The two pretraining-equivalent worlds now have pole sets {.65,.2} and {.4,.2}. For a fixed test command sequence their clean response vectors a,b differ. Any learner using only pretraining observations has the same prediction distribution in the two worlds. The triangle inequality gives

~~~text
max_world E[RMSE(prediction,true_response)] >= RMSE(a,b)/2.
~~~

The equal-prior average squared error is likewise at least `RMSE(a,b)^2/4`. This is a two-world information lower bound, independent of algorithm strength. It concerns a **new labeled component intervention outside the command-only training interface**. It does not assert that prediction under all future ordinary commands is impossible.

For the retained20-step command sequence, response distance is 0.371675; the two-world expected-RMSE lower bound is 0.185837. Actual command/response vectors are retained in JSON.

The test starts both post-change systems at the exact zero state, a privileged controlled evaluation initialization. Replacement metadata specifies only the new pole, without revealing the old hidden motor pole. It illustrates ambiguity before the new response is observed; subsequent observations or a force sensor could resolve the changed world.

## E21. Executed noisy feedback/atom replication

Python3.12.3, NumPy2.5.3, threadpoolctl in existing `.venv`; v8 imports the frozen v7 weights. No installation. Three conditions: Gaussian probes+position sensorSD .5; atom probes+sensorSD .5; atom probes+sensorSD1.5. Every condition has48 independent seeds `20261020+1300000+1000*case+i`, 65536 fit rows after604 initial rows. Motor-processSD .06, probeSD .7, beta1.2. Controller `pi=-.5*tanh(D)`. The atom policy p=.1 matches the Gaussian policy's conditional mean/variance as in v7. Both swapped physical worlds use the exact same exogenous noise and policy coins.

Within a policy, all command/position estimators receive the same trajectories. Added-force estimates separately receive mnext with independent sensorSD .2. No hidden force is supplied to ordinary IV or weighted estimators. Maximum paired observable differences, hidden-force differences, every coefficient/pole/forecast and all non-real pole cases are saved.

| Condition | Apparent weighted coefficient +/- MC SE | Max paired observation difference | Real pole fits | Mean pole-set error (real fits only) |
| --- | --- | --- | --- | --- |
| gaussian_sensor0.5 | 1.049515 +/- 0.001397 | 1.78e-15 | 48/48 | 0.029302 |
| atom_sensor0.5 | 1.044516 +/- 0.003893 | 1.78e-15 | 48/48 | 0.027766 |
| atom_sensor1.5 | 1.061074 +/- 0.005283 | 3.55e-15 | 34/48 | 0.080035 |

The apparent weighted coefficient is a finite-sample estimate of1.05, not exactly equal in each run. Maximum path differences are at numerical precision. The high-noise condition yields real AR2 roots in only34/48 runs; the remaining14 are retained with null pole-error values, never treated as successes. Pole-set error means are conditional on the reported real-root subset. AR2 coefficient/forecast calculations use all48 runs, including complex estimated poles. Finite data can fail physical pole recovery even when population IO parameters are identifiable.

## E22. Forecast, added-information and physical-disconnect tests

For each fit, generate a fresh20-step iid Gaussian command sequence SD .7 with seed+2000000. Both true hidden states and predictor initial outputs are exactly zero; no process/sensor noise is added to this evaluation. This is a privileged known-initial-state, clean open-loop replay after closed-loop training, not a test of unknown-state filtering, goal control or on-hardware transfer. It compares errors from inadequate versus adequate observable dynamic order.

| Condition | AR1 | AR2 |
| --- | --- | --- |
| gaussian_sensor0.5 | 2.373744 +/- 0.219536 | 0.018901 +/- 0.002319 |
| atom_sensor0.5 | 2.668252 +/- 0.265481 | 0.022439 +/- 0.003577 |
| atom_sensor1.5 | 2.912286 +/- 0.284929 | 0.078220 +/- 0.008616 |

Values are mean20-step velocity RMSE +/- Monte Carlo SE. The established AR2 baseline gives much better sequence prediction than the intentionally misspecified AR1 model. No advantage over modern nonlinear methods is claimed; no method can label the physical poles from the stated training channel.

For the added force sensor, physical rho estimates average near .65 in the original world and .4 in the swapped world (full table/SEs in JSON). This validates the extra information's role rather than a fair same-observation algorithm contest. The physical-disconnect check prepares identical observable velocities using eight common commands1.3, then truly removes delivered force for10 steps. Velocities subsequently decay by .65 versus .4. There is no claim the original command-only learner can execute that intervention.

## Independent adversarial audit and prior work

`independent_theory` identified the actuator-memory confound before routine nonlinear coasting fits, derived the exact pole swap and apparent-sum coefficient, and checked initialization, noise placement, IV relevance and force-sensor scale. It corrected a possible MA2 description to MA3. `v5_adversarial_audit` independently reproduced simulation/estimates and checked the full adaptive-policy proof and component-transfer bound. Full evidence is in `research/audit_v8.md`; no auditor edited canonical artifacts.

**EXISTING RESEARCH:** Commuting linear cascades and nonunique physical factorizations are classical system realization/identification. [Tiels, Schoukens and Schoukens, Initial estimates for Wiener–Hammerstein models using phase-coupled multisines](https://arxiv.org/abs/1612.04568) explicitly distinguishes identifying the product of dynamic blocks from assigning dynamics to individual blocks; nonlinear inter-block structure can provide additional information in that broader setting. The present pure-linear counterexample is its elementary commutative limit, not a new result or an impossibility for every nonlinear robot. [Closed-loop IV methods](https://doi.org/10.1016/j.automatica.2004.09.016) underlie the predictive baseline. Historical v1-v7 distinctions between predictive adequacy, physical identification, intervention access and measurement noise remain essential.

## Reproduction and next scientific question

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/actuator_memory_v8.py > /tmp/actuator_memory_v8_check.json
cmp research/actuator_memory_v8_results.json /tmp/actuator_memory_v8_check.json
~~~

The strongest next question is whether **nonlinear passive structure breaks the pole-allocation symmetry under the original observation/intervention channel**, and at what signal strength that distinction becomes statistically usable. A minimal globally contractive family is `f(v)=rho*v-c*tanh(v)` with0<c<rho<1, followed by the same linear actuator. Eliminating m yields

~~~text
vnext=(rho+lambda)v-rho*lambda*vprev-c*tanh(v)
      +lambda*c*tanh(vprev)+beta*u+xi.
~~~

Within this supplied family and exact velocity observations, nonzero c would label lambda through the ratio of nonlinear lag/current coefficients. At c=0 ambiguity returns; near0 identification may be weak. Derive and audit that claim before implementation; test whether a nonlinear actuator alternative or errors-in-variables regression can imitate the signature. Compare complete observable nonlinear regression/IV or state-space estimation, not just favorable physics-coordinate scores. Preserve v8's exact linear-symmetry baseline and consider component-change predictions separately from ordinary command transfer.

This release is a substantive, independently verified limitation on physical modular transfer, but not a universal theory of learning or completion of the persistent Goal. Continue immediately from the nonlinear symmetry-breaking discriminator.

---

# Historical archive: Complete original v7 ledger, including all earlier versions

# First-Principles Robot Learning: Living Research Ledger v7

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

| Method | Affine | Quadratic | Saturating | Weak probe / noisy sensor |
| --- | --- | --- | --- | --- |
| OLS | 0.246353 / 0.403660 | 0.287061 / 0.362960 | 0.087772 / 0.562242 | -0.336275 / 0.986293 |
| constant_gain_IV | 0.636794 / 0.013938 | 0.644421 / 0.007194 | 0.632690 / 0.018272 | 0.648252 / 0.043268 |
| series_IV | 0.649850 / 0.004698 | 0.653741 / 0.006322 | 0.636993 / 0.014364 | 0.651545 / 0.045148 |
| action_series_IV2 | 0.649495 / 0.005015 | 0.651223 / 0.005798 | 0.602164 / 0.048235 | 0.658529 / 0.055506 |
| action_series_IV4 | 0.649339 / 0.006157 | 0.651159 / 0.006809 | 0.636028 / 0.016096 | 0.628381 / 0.161548 |
| weight_m1 | 0.650998 / 0.007348 | 0.652492 / 0.007385 | 0.637942 / 0.014689 | 0.665962 / 0.106701 |
| weight_m2 | 0.650435 / 0.007270 | 0.649329 / 0.006987 | 0.602309 / 0.048516 | 0.646773 / 0.121971 |
| weight_m4 | 0.651209 / 0.009841 | 0.647070 / 0.008431 | 0.636128 / 0.016817 | 0.667914 / 0.151897 |
| atom | 0.655574 / 0.017112 | 0.646673 / 0.017096 | 0.649698 / 0.019532 | 0.680747 / 0.107148 |

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

# First-Principles Robot Learning: Living Research Ledger v6

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

| Sensor SD | Raw centered order2 | Phased order1 | Full Volterra centered / phased | Matched Q-FIR centered / phased |
| --- | --- | --- | --- | --- |
| 0.1 | 0.084037 | 0.014816 | 0.005261 / 0.010529 | 0.003318 / 0.003102 |
| 0.5 | 0.092653 | 0.028243 | 0.021553 / 0.043046 | 0.012580 / 0.013860 |
| 1.5 | 0.111864 | 0.073514 | 0.057083 / 0.114791 | 0.041138 / 0.042669 |

Table entries are RMSE of eta across independent seeds; eta=.7. The elementary phased estimator beats the raw centered moment, but **this is not probe-design superiority**. A stronger centered regression beats that phased estimator at every noise level. Applying the same supplied-Q regression to both designs yields comparable errors and no consistent winner. The complete quadratic estimator also beats the raw moment. All estimated means, Monte Carlo SEs and median errors are retained. Different nonlinear priors are explicit; the matched structured comparator is the fairest design comparison here.

The raw-moment-versus-phase experiment preceded addition of the efficient baselines; these are exploratory corrections, not a preregistered method contest or independent confirmatory efficiency study. The structural theorem and mechanism formulas were derived before their tests. No general advantage over nonlinear system identification, kernels, recurrent models or optimal experiment designs is claimed. V5 already compared a flexible kernel predictor on the prior coupled system; this cycle tests exact identifiability and efficiency confounding, not a new competitive learner.

## E19. Covariance-only repair

64 independent 8190-row runs per AR coefficient0,.5,.75, marginal command SD .8, sensor SD .5. Seeds `20261020+1000000+1000*j+i`. Paired worlds share seed streams only to verify the analytic mean difference, not as an oracle for a learner.

| Command AR coefficient | Analytic difference | Measured difference +/- MC SE |
| --- | --- | --- |
| 0.0 | 0.000000 | -0.001079 +/- 0.000884 |
| 0.5 | 0.117570 | 0.117681 +/- 0.000624 |
| 0.75 | 0.088177 | 0.088575 +/- 0.000307 |

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

# First-Principles Robot Learning: Living Research Ledger v5

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

| SD | Mean K01 | Mean K02 | Mean K12 | Median parameter L2 |
| --- | --- | --- | --- | --- |
| 0.35 | 0.694537 | 0.354960 | 0.306227 | 0.172482 |
| 1.25 | 0.700159 | 0.349326 | 0.315323 | 0.049866 |

The true K01,K02,K12 are .7,.35,.315. Larger amplitude improves finite-sample precision without making the population first-order footprint informative about eta. The maximum absolute mean paired first-order coefficient difference is .000587 at SD .35 and .003126 at SD1.25. These finite estimates do not prove equality; the symmetry theorem does. The L2 metric uses parameter order beta,eta,r,rho and fixed normalized units. It is not a universal parameter metric.

## E16. Fair predictive comparison and transfer failures

32 independent training seeds `20261020+600000+100*i`, i=0..31. Each model receives the exact same 1024-row training trajectory, separate 512-row validation trajectory (base+1), and four fresh 2048-row tests (base+10..13). Train/validation command SD .8, zero means, white independent channels. Test conditions: iid matched; means (.4,.5); temporal AR(1) correlation .75 within each channel with same marginal SD .8 and independent channels; and amplitude1.25. Hyperparameters use observed validation displacements, never test or latent data.

Models: FIR ridge on 16 command-history coordinates; quadratic Volterra ridge on all 16 linear and 136 degree-two monomials; exact RBF kernel ridge on the same 16 coordinates. Each includes an intercept. Polynomial features and RBF are **supplied nonlinear vocabularies**, not discovered operators. Ridge grids are [.001,.1,10,100] on summed squared loss; RBF gamma is [.25,1,4]/16. Feature scaling/centering uses training data only. RBF has 12 validation configurations, versus four for each polynomial degree; compute is disclosed, not matched. Eigensolves are exact and one BLAS thread is used. These are established nonlinear competitors, not implementations of the specialized multidimensional GP priors in cited modern Volterra work. No claim of superiority over those methods or recurrent/state-space alternatives is made.

Primary score compares predictions to noiseless velocity generated by the same test commands, with independent process noise removed for evaluation only. Because models use only command history as predictors, this is their conditional-mean target up to history truncation. Actual measured-output RMSE and signed mean errors are also saved. Models could use past measurements in a broader experiment; this restriction is intentional for testing the response-summary claim and limits conclusions.

| Model | iid | mean shift | correlated | amplitude shift |
| --- | --- | --- | --- | --- |
| FIR | 0.582922 +/- 0.003567 | 1.077155 +/- 0.016494 | 1.171913 +/- 0.017977 | 1.425701 +/- 0.010338 |
| Volterra2 | 0.289815 +/- 0.002927 | 0.396586 +/- 0.010938 | 0.258428 +/- 0.005589 | 0.683538 +/- 0.008519 |
| RBF_KRR | 0.294420 +/- 0.003330 | 0.527108 +/- 0.012173 | 0.463173 +/- 0.015818 | 0.913199 +/- 0.011202 |

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
