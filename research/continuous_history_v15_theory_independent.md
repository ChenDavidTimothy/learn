# Independent continuous-history v15 pilot audit

Date: 2026-10-20. Audited frozen design and source before examining control outcomes. The first numerical pilot is now frozen as `continuous_history_v15_pilot.py`; the revised fresh protocol is separately recorded below. No canonical source/settings were edited. This report separates a correct conventional learning experiment from claims it cannot establish.

## Information and selection

The source acquires72 continuous64-command trajectories, split by whole trajectory48/12/12. Every candidate uses row times8..63. Separate role-tagged SeedSequence paths distinguish fitting, validation, testing, control and derivative-check streams. Each trajectory starts with independent hidden velocity and random pose; this acquisition reset capability is supplied. Evaluation switches goals without resetting body, observation history, issued-action history or warm plan.

Ordinary fitting/planning reads only noisy positions, issued actions, clock and supplied feature menu. True states are stored and used for validation/test diagnostics and external control scoring, but candidate selection explicitly uses observation validation error only. Shared current physical pose/rest is not supplied at test time. Warm-up provides8 causal zero-torque transitions. Goal origins are experimenter world coordinates; although they constrain initial-pose information, they do not reveal evolving true motion.

Current-only, fixed one-delay, selected nonlinear delays, selected linear ARX and selected polynomial-compressed histories share data and the same receding-horizon solver/cost. This is stronger than comparing only against current-only control. Polynomial histories fit degree2 to3/5/9 causal readings and include supplied slope/curvature arithmetic; all past issued actions are still retained. This is a supplied compression competitor, not autonomous discovery of physical variables. The nonlinear Fourier basis is generic over both encoders; neither the true inertia formula nor the known irrelevance of absolute q1 is inserted into fitting.

Regression predicts measured increments and restores the current reading. This is algebraically next-observation regression: DT*(predicted_increment-target_increment) equals predicted_next_observation minus actual_next_observation. For a fixed causal forecast f, independent zero-mean fresh e_next gives expected squared error against measured next position equal to physical conditional forecast error plus2*SD^2. It does not make lagged regressors noise-free or identify unbiased physical coefficients. Validation selection is optimistic and independent test trajectories are necessary. One acquisition fit is an exploratory engineering study, not independent replicated evidence about learning variability.

## Timing, features and analytic derivatives

Past differences and issued actions are indexed causally. Including previous actions is appropriate: even an exact ZOH constant-inertia double integrator satisfies

    q_next=2*q-q_prev+.5*DT^2*B*(u_prev+u_current).

Current candidate torque affects the NEXT physical position in this continuous ZOH plant. Importing the earlier scalar motor-lag timing would be wrong. Raw finite differences are an invertible re-expression of the supplied position delays given the current reading; longer finite histories may mainly denoise or improve approximation, not establish additional mechanical state.

Write the learned increment rate as p(q)^T*C*b(history,past_actions,u), then

    qnext=qcurrent+DT*sum_p,b p*C*b.

Its position derivative has both the derivative of b AND the current-pose derivative of p, plus the current-position identity. Its action derivatives follow through b. During recursive planning, total action sensitivity sums these local derivatives times all retained position/action sensitivities, and adds the direct current-action block. The source implements all terms, including action memory and quadratic-motion product rules. No indexing/transposition error was found.

`audit_continuous_history_v15_derivatives.py` independently implements features/rollout in float64 Torch and obtains Jacobians by automatic differentiation. It imports the FROZEN pilot only as the implementation under comparison, not to compute independent features or derivatives. All10 supplied family/lag combinations were checked at4 and15 predicted transitions on stable synthetic coefficient/history fixtures. Maximum Jacobian discrepancy5.81e-15; residuals agree. The audit also checks AST identity of context/base/features/Model/residual_plan/choose between frozen pilot and fresh production source, so these derivative checks apply to unchanged fresh formulas. They verify formulas and full-horizon chain rules, not the fitted models' predictive accuracy, conditioning, global optimization or stability at every encountered history. The canonical four-step finite-difference check on selected fitted models remains a complementary local check.

## Planner interpretation and action coverage

All learned methods use the same local bounded residual least-squares planning. Warm plans shift without being reset at goal changes. The planner assumes only the current goal persists; undisclosed future goal switches are common. All future actions remain available, avoiding the old unannounced coast deadline. There is no privileged physical state initialization in the learned rollout.

The motion penalty is predicted STEP-AVERAGE displacement/DT, not endpoint instantaneous physical velocity. These agree only in an appropriate small-step limit. Position MSE, instantaneous velocity RMS and action energy are separately scored physical diagnostics; their weights/normalizations should not be silently conflated into an exactly matched physical stage objective. A method can improve one component and worsen another. Success on final1sec position AND velocity is a separate, demanding evaluation condition rather than a training-selection criterion.

Acquisition commands are bounded by .35 and held for two samples, whereas planning permits magnitude1. Thus planning includes counterfactual actions outside the observed command support. The supplied action-affine feature structure extrapolates there; exact finite-time nonlinear flow is only approximately affine in torque. Even within .35, repeated holds create temporal correlation, although every other step draws fresh independent torque. An actual control failure can be action/history coverage or recursive approximation, not evidence that delay information is absent. Initial velocity range and subsequent conservative drift also matter; broad initial angles alone do not establish broad pose/speed/history coverage.

Open-loop ten-step tests correctly initialize only from causal observed history, then append predicted positions and logged actual future commands, with no future observations. Their saved average pools all horizons1..10, rather than measuring only the tenth endpoint; label it accordingly. Recursive conditional means are certainty equivalence, not a correct nonlinear belief evolution. Synthetic predicted histories lack the training sensor-noise distribution and may leave its support; a good teacher-forced one-step score does not remove this mismatch.

TRF is a local optimizer. A finite improved final solution is used even if its solver reports unsuccessful termination; that behavior is explicit and gradient/status traces are retained. Report unsuccessful termination and projected-gradient residuals rather than calling every such action a fallback to the warm plan or a converged optimum. The actual warm fallback applies when the final result is nonfinite or worse than the initial plan. Numerical rollout overflow could still raise before an ordinary record is returned; if this happens, preserve it as an engineering failure, not a missing experiment.

## Feasibility and a controller-level alternative

In the horizontal fully actuated mechanics, for every q,

    ||M(q)||_infinity <=7,
    ||h(q,v)||_infinity <=3*||v||_infinity^2.

A standard quintic rest-to-rest interpolation q=qi+(qg-qi)*(10s^3-15s^4+6s^5), s=t/D, has maximum scalar speed factor15/8 and acceleration factor10/sqrt(3). With D=6sec and maximum joint displacement .35, speed is at most .109375 and acceleration at most .056133. The required continuous torque M*qdd+h is therefore at most .42881 in infinity norm, below the common bound1. Slow reference transitions between the stated stationary goals are not intrinsically impossible.

This is a continuous-reference existence calculation, not a proof that the initial nonzero-velocity phase, arbitrary failed recovery states or exact .1sec ZOH endpoints have a feasible6sec stopping trajectory. Those distinctions remain necessary. It also does not supply the learner any mechanics or reference torque.

If all learned MPC variants fail, a useful separately declared controller-level diagnostic is bounded causal filtered PD. With exact physical velocity and constant goal, an ordinary unsaturated PD has energy derivative minus a positive velocity quadratic. Even a globally bounded conceptual alternative is

    u_j=-.5*tanh(2*kp*error_j)-.5*tanh(2*kd*v_j),

whose potential is sum_j log(cosh(2*kp*error_j))/(4*kp). Total kinetic-plus-potential derivative is -.5*sum_j v_j*tanh(2*kd*v_j)<=0. This establishes the simple control mechanism's physical relevance without using M in its action. With noisy estimated velocity, finite sampling and switched goals, that proof does not directly apply; the real alternative must be tested and tune only from the same allowed experience/virtual resources. Do not add a true-plant-tuned PD grid and then label it same-information.

## Scientific decision after this pilot

The current source is an honest first continuous same-experience learner and serious conventional history comparison. No severe information leak or derivative error was found. Correct one-step formulas, selected lag length and good multistep scores still do not establish successful sustained decisions; actual control is essential.

If the data reject pose-only prediction and a conventional history learner also sustains new-goal control, that supports conventional sufficiency under supplied menus. If prediction improves but control fails, retain the discrepancy and diagnose counterfactual support, recursive rollout and local planning before proposing a new representation. Do not convert this into another endless scalar-like optimizer/filter ladder or claim a universal history-selection principle from one exploratory fit.

## Fresh protocol and corrective literature context

After the complete exploratory pilot, a separate protocol was frozen: independently acquired replicate IDs100..107,96 commands per trajectory, exogenous balanced trajectory amplitudes .35 and1 in every split, four new initial poses, and the fixed-rich H4 comparator reusing the existing H4 candidates. This addresses the pilot's scalar action-amplitude extrapolation and tests whether selection adds value beyond a supplied rich history. It does not retroactively make the pilot confirmatory. Increased duration also changes speed/pose distribution; the fresh study is a new acquisition protocol, not a pure amplitude-only ablation.

Full command magnitude support does NOT establish joint history/action/speed support. Report actual task-state speed envelopes against acquisition and, if consequential, local proximity under a common declared periodic-pose/motion/action diagnostic. True state may be used for this evaluator diagnosis but must remain unavailable to fitting/selection/control. Global extrema or a minimum-distance summary alone do not certify counterfactual coverage. Whole-trajectory failures must remain, and no hidden filtering/clipping/projection of true motion was found in the production path. Aggregate first across the four fixed tasks, then uncertainty across the eight acquired-data replicates. The4*3 precision checks per model/dataset are clustered consequences, not independent learning replicates.

Dalla Libera et al. (2020), *Model-Based Reinforcement Learning for Physical Systems Without Velocity and Acceleration Measurements*, directly learns position-history forward dynamics using derivative-free Gaussian-process kernels, including physically inspired/semiparametric kernels, and demonstrates trajectory optimization on real ball-and-beam and Furuta-pendulum systems. Its reported learned controls were applied open loop. The authors explicitly relate their approach to GP-NARX and distinguish learned raw histories from supplied numerical differentiation. Primary paper, read for this audit: https://arxiv.org/pdf/2002.10621 . This directly precludes novelty claims for position-history forward prediction used to optimize physical actions; our ridge/Fourier receding-horizon study neither reproduces nor fairly outperforms their complete method.

Independent caveat: a finite noisy position history is not universally an exact Markov state merely because a finite-difference filter fits inside it. For this second-order plant acceleration is a consequence of(q,v,u), not a further independent necessary physical state; prior actions also carry information. Treat finite-history sufficiency as a bounded approximation/empirical question, even when reading established papers' stronger framing. This is precisely where first-principles independence from the literature remains important.

## Independent check of moving first-segment feasibility

The coordinator proposed extending the continuous-reference existence witness through the actual moving warm-up state. Independently rederived the global norm/comparison bounds rather than assuming that the later rest-to-rest construction applies initially.

Let V=||v||_infinity, c=cos(q2), det=2-c^2>=1. Since

    M^-1=(1/det)*[[1,-(1+c)],[-(1+c),3+2c]],
    |h1|<=3*V^2, |h2|<=V^2,

the first component of M^-1*h is bounded by5*V^2 and the second by11*V^2. Therefore the upper Dini derivative of V during zero-torque warm-up obeys Dplus V<=11*V^2. Starting at V0=.04, comparison on t<=.8 gives

    V(t)<=V0/(1-11*V0*t),
    ||q(t)-q0||_infinity<=-log(1-11*V0*t)/11.

The denominator at .8 is .648>0, hence actual warm-up velocity is at most .0617283950617 and displacement at most .0394422347845. The first goal offset has maximum component .2, so from the actual warm-up pose, first-goal displacement delta is bounded by .2394422347846. These are global over the prescribed starting poses, independent of sensor realizations.

For r=t/D in[0,1], D=6, take

    s(r)=10*r^3-15*r^4+6*r^5,
    b(r)=r-6*r^3+8*r^4-3*r^5,
    qref=qstart+delta*s(r)+D*vstart*b(r).

This matches actual starting q/v, sets initial acceleration0, and ends at the first goal with velocity/acceleration0. The initial acceleration can be changed by a bounded torque jump; there is no torque-rate constraint. Bounds are |s'|<=15/8, |s''|<=10/sqrt(3), |b'|<=1 (minimum-64/125), and

    b''=-12*r*(1-r)*(3-5*r), |b''|<=9.

Consequently, using the conservative warm-up bounds,

    ||vref||_infinity<=.136554093432,
    ||aref||_infinity<=.130993158901,
    ||M*aref+h||_infinity
       <=7*.130993158901+3*.136554093432^2
       <=.972893173603<1.

Subsequent prescribed goal displacements have maximum joint component .35 and start at reference rest, so the earlier .42881 torque bound applies. Reference q/v/acceleration join continuously at those goal boundaries. Thus the FULL ideal continuous goal schedule, including the moving warm-up initial condition, has a bounded continuous-torque realization. The calculation supplies no true motion/inertia to the learner.

This strengthens only the declared continuous-reference feasibility statement. It does not prove that a piecewise-constant .1sec command sequence exactly realizes that reference or endpoint tolerances, that learned/local MPC finds it, or that every state reached after a failed controller can recover within the remaining dwell time. The first-segment conservative torque margin is about .0271 and should not be silently treated as a discretization-error allowance.
