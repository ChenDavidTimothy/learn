# Continuous-history v15: frozen first pilot design

The unresolved scientific question is whether ordinary continuous experience can
reject an inadequate current-position summary, select a useful causal history,
and use it in sustained coupled decisions. Choosing a lag integer from a supplied
menu is conventional model selection, NOT autonomous discovery of physical state
or a novel universal learning principle. A negative prediction/control mismatch
is equally informative. No v15 release for the earlier supplied-slope pilot alone.

Same continuous conservative two-link plant as v14; direct bounded torque, no
gravity/friction, ideal unwrapped angle encoders with independent Gaussian SD.001.
Sampling/zero-order hold .1 seconds. Initial velocities are random and hidden.
Known sampling clock, command bounds, angle periodicity and feature menus are
supplied; no masses, inertia, true motion, acceleration or future noise is supplied
to ordinary models/controllers. True states are retained solely for evaluation.

All candidates share exactly the same noisy position/action trajectories. Entire
trajectories, not overlapping rows, are split into fitting, validation and testing.
First engineering pilot: one fit, 48/12/12 independent trajectories of64 commands;
initial q uniform[-pi,pi]^2, hidden v uniform[-.15,.15]^2, commands uniform[-.35,.35]^2
held for two samples. These resets are acquisition resources, NOT goal resets.
Independent role-tagged SeedSequence([20261020,15,610,replicate,role,trajectory])
replaces earlier collision-prone small additive offsets. All role paths retained.

Ordinary candidates predict NEXT measured position through an increment regression.
Raw histories use current q, lag differences (o_t-o_{t-k})/(k*dt), issued past
commands and the candidate current command. Include past actions: even an exact
ZOH double integrator satisfies q_next=2q-q_prev+.5dt²B(u_prev+u).
Supplied nonlinear features are a symmetric Fourier basis of BOTH current angles
(harmonics1/2 and first-harmonic pair products), tensor linear command/history
features and quadratic terms of the most recent lag difference. Coefficients are
ridge learned. Candidate delay lengths0/1/2/4, ridge1e-8/1e-5/1e-2. No named true
velocity or physical acceleration targets are learned. Feature scaling uses fit
data only. Fit normal equations and all candidate validation scores are preserved.

Comparisons: current-position-only; fixed one-delay nonlinear; validation-selected
raw delay nonlinear; validation-selected linear ARX (no pose context/quadratics);
conventional causal polynomial history compression with learned nonlinear next
position relation (quadratic polynomial fit to3/5/9 past/current readings, current
slope/curvature, past commands, same ridge grid). The last is a serious standard
derivative/filter alternative; its compression is supplied arithmetic. The raw
history coefficients learn a weighting of available lags. This is not an exact
implementation of Romeres2018 or Dalla Libera2020 GP algorithms, and no superiority
over those robotic methods will be claimed.

Selection minimizes heldout next-observation Euclidean squared position error.
For any fixed causal forecast f(H_t,u_t), fresh zero-mean e_next independent of
the past gives E||o_next-f||²=E||q_next-f||²+2sigma². Differenced-target fitting
does NOT make regressors noise-free: current sensor error is in both conditioning
input and increment target. Do not apply a classical physical-parameter unbiasedness
claim. Selected validation scores are optimistic; independent test required.
Also report recursive10-step forecasts with no future observations. Composing
conditional means is a certainty-equivalent heuristic in nonlinear noisy systems.

All learned controls use the SAME bounded residual least-squares receding-horizon
planner, current goal only, horizon15, warm-started issued-action plan, no hidden
velocity filter/state initialization. Residuals predicted position error,
sqrt(.3)*predicted displacement/dt and sqrt(.003)*command. A stationary current
goal is assumed within the planning window; future switches are not disclosed.
All future actions remain available, including after the scored window, so there
is no undisclosed actuator deadline. The planner is local, not a global certificate.
Analytic rollout derivatives must pass independent numerical checks before use.

Pilot control: two initial poses (0,pi/2),(1,.3), hidden v(.04,-.02), then three
successive goals q0+(.2,-.15),q0+(-.15,.15),q0+(.1,.1), each6 seconds. No reset,
no state clearing at switches. Eight warm-up zero-torque samples provide causal
history. Initial exact q is NOT supplied to ordinary controllers. Sensor streams
shared across methods, actions/trajectories consequently differ. Report total
position MSE, actual velocity RMS, action energy, final1-second joint position
and velocity precision, solver failures, saturation and actual resources. Full
q/v/u/o/goals/plans retained. Goal schedules use experimenter-defined world angles;
their origin does not reveal the evolving true state. Scoring may use hidden truth.

This first pilot may expose inadequate coverage or unstable recursive models. It
is exploratory engineering, not fresh confirmatory evidence. Retain all outcomes
before any revision; freeze a separate fresh protocol if a meaningful study follows.
Analytic continuous feasibility of smooth slow reference motions should be derived
separately; a failed learner trajectory does not make every encountered recovery
feasible. Neither small one-step prediction error nor return to a point establishes
stopping or general learning.

Existing research: Romeres et al.2018 https://arxiv.org/abs/1809.05074 learns
position-history inverse relations/compressions. Dalla Libera et al.2020
https://arxiv.org/abs/2002.10621 directly studies derivative-free position-history
forward models and physical trajectory optimization. Our intended contribution
here is a controlled diagnosis in this research lineage, not novelty of delays.

## Pilot outcome and fresh protocol, frozen before fresh results

The complete first run is preserved in continuous_history_v15_pilot.py/.json.
It selected rawH4 and polynomialH4; each met all6 late precision checks. FixedH1
met2/6, linearARX3/6 and current-only0/6. This is one acquired fit on two familiar
engineering poses, not replicated evidence. Ordinary executed commands often
exceeded acquisition|u|.35, a material coverage limitation: selectedraw89/720
coordinates, polynomial91/720, fixedH1151/720. A finite-step plant need not be
exactly command affine. Source and all outcomes remain unchanged.

Fresh protocol:8 independently acquired datasets, replicate IDs100..107, disjoint
SeedSequence paths from pilot0. Same total72 trajectories/dataset but96 commands
per trajectory, held2. Torque amplitudes are exogenously stratified by trajectory:
even-index trajectories span[-.35,.35], odd-index span[-1,1], for fit/val/test alike.
Broader commanded support avoids relying solely on the pilot's extrapolation.
No stabilizing collector, hidden-state feedback or acquisition pruning. Report
speed/pose/command coverage; arbitrary excursions are evidence, not silently dropped.

Add a fixed-rich rawH4 comparator (ridge selected within H4) to separate history
selection from merely supplying a rich window. No extra acquisition. Other menus,
selection criterion, planner/cost/horizon and .03rad/.03rad/sec precision tolerances
unchanged. Precisely30 ridge candidates in total as above; fixed-rich reuses
existing3 H4 candidates, no extra fits.
Control starts four poses q0=(-2,-2.1),(-.5,-.7),(.8,.7),(2,2.1), hidden initial
v(.04,-.02), then the same3 offset goals/6sec dwell with8 warm-up samples and no
reset. These differ from both pilot starts. Every method/dataset gets the SAME
noise path for each control task, not the same path across independently acquired
datasets. Whole acquisition trajectories and method-specific task trajectories
are retained, including all failed controllers and optimization diagnostics.

No hyperparameter selection from fresh task outcomes. Aggregate each method over
the4tasks FIRST, then report mean and SE across8 acquired-data replicates; precision
counts are clustered outcomes, not96 independently learned bodies. A history or
filtered-motion method may suffice without learned history selection improving
over the fixed-rich alternative. No claim of novelty or state discovery follows.
