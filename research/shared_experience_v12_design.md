# Frozen v12 shared-experience comparison

Written before running new results, 2026-10-20. Exploratory follow-up to the retained
PD pilot, not a preregistered general-method contest. No outcomes may be used to
select hyperparameters or drop plants/seeds. Subsequent changes must be disclosed.

Question: with exactly the same complete physical experience, does direct search
in a supplied dynamic feedback class yield the same goal transfer as a conventional
predictive learner? This tests USE of experience, not optimal acquisition or a
universal model-free/model-based dichotomy.

Four plants: rho=.65, beta=1.2, lambda in {.2,.7}, c in {0,.2}, lagged timing.
Eight independent training seeds each. Known physical zero reset each86-step episode;
sensor SD .2; train motor noise zero; test motor SD .03. Continuous bounded actions
throughout; current goals only, switches at6 and28. Same physical family/noise metadata
as prior identification work; direct controller is not given true parameters.

Direct family: u=clip(kp*(g-sx)-kd*sv+bv*sv_previous-bu1*u[t-1]-bu2*u[t-2],-1,1),
sx+=alpha*(o-sx); sv+=alpha*((sx-sx_previous)-sv). Six searched coordinates:
log(kp),log(kd),logit(alpha),bv,bu1,bu2. Bounds kp[.005,.5],kd[.02,4],
alpha[.1,.98],bv/bu[-2,2]. PD is included exactly at bv=bu1=bu2=0. This is a restricted supplied
rational dynamic controller, not an arbitrary observer or nonlinear policy.

Diagonal CEM:8 generations,24 candidates each,6 elites, two training tasks
(1,-1,.5),(-1,1,-.5) per candidate. Initial means(.08,.6,.6,0,0,0),
coordinate SD(1,1,1,.3,.3,.3). Candidate0 is the previous generation winner
(retested with fresh noise), candidate1 at generation0 uses(.02,1,.8,0,0,0).
Each new generation uses empirical elite mean/SD, floors(.15,.15,.15,.03,.03,.03).
Selection uses only mean observed fitness; current generation winner is checkpoint
policy. No stale global minimum is retained. Fresh unique noises each candidate.
Checkpoints after generations2 and8:8256 and33024 transitions,96 and384 resets.
All candidates, including poor policies, are kept. No true-state reward in selection.

Shadow linear and nonlinear OE fit ALL issued inputs and raw positions at each
checkpoint, restarting each known-zero segment separately. Sum raw SSE; three
v9 starts/bounds and max160 evaluations per start, analytic derivatives, deterministic
training. This is joint/sequential likelihood under feedback, not a likelihood
conditional on the whole future command sequence. Supplied functional families are
strong prior knowledge. No physical interaction added by model fitting/planning.

Each learned model controls with EKF using supplied noise scales/zero prior, bounded
TRF MPC using16 free actions (no artificial zero coast), known86-step task horizon
truncated to min(16,86-t), current goal held constant in imagined future. Exact
same per-step position/velocity/action cost. Shifted warm plan, analytic residual
Jacobian, max80 evals, ftol/xtol1e-10,gtol1e-8. Save solver status/gradients/plans.
Nonlinear optimization has no global guarantee. EKF is approximate for c!=0.

Freeze evaluations: two held-out goal amplitudes(2,-2,1),(6,-4,3), independent
fresh noises shared across methods/checkpoints within seed. Actual evaluator cost
mean over86 steps divided by1+mean(g^2); true states are evaluator-only. Also late10
position/velocity errors, all-late10 |position error|<=.3 and |velocity|<=.15 event.
Report per-plant means/MC SE clustered by8training seeds, paired differences,
individual outcomes, no seed dropping. Evaluation is not read by training/fitting.
Exact training resets/steps and test interaction costs reported separately; virtual
compute timed. Equal data is not equal runtime, vocabulary or hyperparameter budget.

Primary precedents: de Boer et al.2005 CEM tutorial;
https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf;
Mania/Guy/Recht2018 https://arxiv.org/abs/1803.07055 cautions about seed variability.
This experiment implements CEM, not ARS; it is not a modern robotics benchmark.
Accept either outcome. If direct search fails, distinguish family expressivity,
search/data efficiency and model-planner advantages before a broad interpretation.

Before any new run, independent design audit recommended replacing four command lags
with one velocity-memory lag and two command lags (six coordinates above). This
permits a force-compensating linear-feedback interpretation as alpha approaches1;
alpha<=.98 and gain bounds limit exact inclusion. It is not an arbitrary observer.

## Pre-outcome correction and metadata limitation

The first execution inadvertently used replicate stride10000 while candidate
episode seed offsets spanned19101. Adjacentreplicates reused some sensor streams.
This was found by source inspection BEFORE reviewing outcomes. Its full source,
NPZ data, compressed JSON and timings remain as shared_experience_v12_seed_overlap_pilot*.
It is excluded from all inferential summaries. Corrected execution uses plantstride
1000000000, replicatestride1000000, trainingoffset100000+100*candidateindex+task,
testoffset500000+task, and CEMseed atbase. All role/replicate streams are disjoint.

Independent audit also noted that model MPC consumes the known remainingepisode
time while the direct family is stationary and lacks a timefeature. Both terminate
at86, but this is an explicit controllerclass difference. Results compare complete
supplied mechanisms, not equally expressive classes or a universal benefit of
prediction. The lastaction can only affect effort in this lagged finitehorizon;
setting it tozero removes atmost .03/(86*(1+mean(g^2))) from directtestcost.
Broader horizon anticipation can still differ and must not be wavedaway.
