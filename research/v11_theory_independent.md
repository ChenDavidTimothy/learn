# Independent theory and direction audit for v11

Date: 2026-10-20 (Europe/Berlin). This audit derives the information limit independently and reviews `precision_information_v11.py`. It does not reproduce the controller experiment; a separate artifact audit handles numerical reconstruction. No canonical source or historical ledger was edited.

## 1. What history means under adaptive control

**PROVEN UNDER ASSUMPTIONS.** Consider the known linear c=0 plant

    s[t+1]=A*s[t]+B*u[t]+e_m*xi[t],  o[t]=C*s[t]+e[t]
    s=[x,v,m], e_m=[0,0,1]^T, C=[1,0,0]
    A=[[1,rho,1],[0,rho,1],[0,0,lambda]], B=[0,0,beta]^T.

Initial state is exactly zero; xi and e are mutually independent temporally iid Gaussians with supplied variances. Each u[t] is measurable from o[0:t], previous commands and independent policy coins; it has no anticipatory disturbance information. Under these assumptions, conditioning on a nonlinear/adaptive control policy does not destroy the Gaussian filtering recursion: the issued command is known when conditioning on the observation/action history. It shifts the next conditional mean, not its covariance. Independent policy coins can be included in the history. Hence posterior covariance P[t] after o[t] is deterministic and obeys

    Pminus[t+1]=A*P[t]*A^T+Q, Q=sigma_xi^2*e_m*e_m^T
    K[t]=Pminus[t]*C^T/(C*Pminus[t]*C^T+R)
    P[t]=Pminus[t]-K[t]*C*Pminus[t].

Known-zero initialization starts Pminus[0]=0. Position sensor variance R=0 requires an exact-conditioning convention when the observation variance is zero, not division by zero. Skipping that zero-variance redundant update is correct.

Offline training, independent of the test innovations conditional on known parameters, cannot further reduce this known-parameter covariance. An unknown-parameter learner can have additional uncertainty; it cannot beat this channel's known-parameter information ceiling by estimating those already supplied parameters more accurately. This is not a claim that every known-parameter controller dominates every learned approximation under a mismatched planning objective.

## 2. Last-action conditional MSE and interval probability

At time69, after observing o69 and before choosing the last active action, let s69|h69~N(mu69,P69). Include u69 in the conditioning; a history-adapted action adds no new information about the state. The subsequent schedule is one noisy active transition followed by16 exactly zero-command, zero-process-noise transitions. Therefore

    x86-g = e_x^T*A^17*s69 + e_x^T*A^16*B*u69 - g
            + e_x^T*A^16*e_m*xi69.

Conditionally this is Gaussian, with arbitrary history/action-dependent mean b and history-independent variance

    V = e_x^T*A^17*P69*(A^17)^T*e_x
        + sigma_xi^2*(e_x^T*A^16*e_m)^2.

xi69 is independent of s69 and h69. No coast observations can change the physical trajectory because further commands are fixed zero. Consequently

    E[(x86-g)^2 | h69,u69] = V+b^2 >= V
    P(|x86-g|<=a | h69,u69) <= erf(a/sqrt(2*V)).

A symmetric fixed-width interval has maximum Gaussian probability at b=0. Averaging over histories preserves both bounds for ANY admissible observation-based policy, including adaptive bounded control. The whole-stop event is a subset of final-position success, so its probability obeys the same upper bound.

**Attainability limitation:** V is an unavoidable MSE lower bound, not necessarily the achievable minimum under bounded commands. The probability bound is tight only for the isolated terminal-position event when the conditional mean can be centered, for example with an unbounded final action and nonzero terminal input coefficient. It is generally not tight for an entire coast path plus velocity constraints or for the actual finite-horizon planning surrogate. Finite empirical proportions can exceed a population probability bound by chance.

Independently evaluated R=.2^2, rho=.65, sigma_xi=.03, 70 active decisions and16 coast transitions:

| lambda | state-uncertainty contribution | final fresh-noise contribution | terminal SD | probability upper bound (a=.3) |
| --- | ---: | ---: | ---: | ---: |
| .2 | .04339596670905541 | .011452669910669566 | .2341978578461489 | .7997951514799881 |
| .7 | .26378961118971544 | .07963292314044698 | .5860226397761118 | .39129692942747146 |

These bounds establish a genuine information limitation. They do NOT, by themselves, explain why a particular controller's empirical whole-stop fraction is lower: the endpoint bound need not be attainable and the actual event imposes many additional constraints. Sensor/data interventions and conditional mean errors are necessary to distinguish information from controller deficiencies.

## 3. Perfect positions still leave a one-tick motor-information gap

**PROVEN UNDER ASSUMPTIONS.** With exact position history through time69, known parameters and the declared lagged timing,

    v69=x69-x68
    v68=x68-x67
    m68=v69-rho*v68
    m69=lambda*m68+beta*u68+xi68.

Thus x69 and v69 are known exactly, and m68 is known, but xi68 has not yet affected position69. It first affects v70/position70. Therefore

    P69=diag(0,0,sigma_xi^2),

rather than zero. Terminal error contains two independent unobserved innovations:

    (e_x^T*A^17*e_m)*xi68 + (e_x^T*A^16*e_m)*xi69.

At lambda=.7, the resulting terminal SD is .3997888420858881 and its symmetric .3-interval probability upper bound is .5469838390677454. This remains below the full-current-state bound .7122642919368134, whose only uncertainty is xi69. Perfect current velocity sensing also does not directly reveal m69. A calibrated current motor-force measurement can reduce uncertainty about xi68; a one-tick-delayed force measurement of m68 adds nothing to otherwise exact position history. No measurement at time69 can reveal the independent future xi69 without changing the timing/information assumptions.

Scope is essential: this statement assumes the specified independent motor-only process noise, physical lagged update, known state dimension/parameters and no contemporaneous force channel. Direct mechanical noise, correlated/predictable innovations, motor-first sampling, asynchronous readout or unknown sensor maps require a fresh argument. The result is elementary delay/observability reasoning, not a new universal limit on learning.

## 4. Source and experiment-design audit

The selected [.65,.7,0,1.2] plant is a defensible TARGETED follow-up: v10 showed severe precision failures there, and linear Gaussian assumptions permit an exact optimal filtering benchmark. Selection after seeing v10 does not make v11 an independent benchmark confirmation or evidence across nonlinear robots. Preserve that explanatory scope.

The training extension retains every original512 command and513 observation, appends commands to the same deterministic no-process-noise physical trajectory without resetting x/v/m, and adds only new position measurements. All short/long models get the same initial data and continuation law. The new command/measurement RNG seed is separate from the original/test streams. Increasing to8192 samples changes physical interaction count and offline fitting burden; the sensor intervention changes information access. Their financial/energy/time costs are not equated, and no efficiency ranking between those resources follows.

At test, changing `control.SENSOR_SD` changes both the actual position-sensor error and the correctly supplied Kalman R, while retaining the same plant, noise seeds, goals, controller, optimizer and metrics. Known-parameter variants supply privileged parameter knowledge. No true-state or test-outcome leakage was found in this source review. Exact-position R=0 is an analytic limiting table entry only; it is not a hidden extra controller experiment. The MSE field must be described as a lower bound, as corrected before release.

## 5. What this should change in the research direction

The useful distinction is not simply more samples versus a better sensor: it is which decision-relevant random quantity has physically reached the available observation channel by the decision time. Better precision cannot compensate for an observation that arrives too late. Conversely, accurate offline models do not reveal fresh independent motor innovations. The next mechanism should use and acquire information with this timing in mind, not add another gauge example or merely enlarge the identification fit.

A consequential next experimental question is whether an agent can learn WHEN a costly additional observation changes an action enough to meet a new precision demand. A concrete first comparison can hold the ordinary plant/model/planner fixed and contrast contemporaneous versus one-tick-delayed noisy motor-force readout under matched measurement counts and calibration, alongside periodic versus decision-triggered readout. The force channel is explicit additional information; its timing must be recorded. Start with known-parameter controls as a diagnostic, then require a learner using experience to estimate the sensor-to-consequence relationship rather than handing it exact latent force labels. Use new goal tolerances and targets; compare against always-measure, never-measure and matched-budget periodic sensing. Measure actual reliability and observation cost, not parameter uncertainty alone. This is a proposed constructive active-perception test, not a novelty claim or prescribed universal architecture.

If that proposal becomes another long sequence of scalar Kalman calculations, redirect immediately. The larger scientific uncertainty is whether learned history-dependent consequences and acquired observations support flexible action under partial information. The stopping toy is an instrument for constructing/falsifying that mechanism; it is not itself general robotics progress.

A further adversarial caution about the proposed next experiment: known linear-Gaussian covariance evolution is independent of realized controls and measurements. For quadratic expected-loss objectives, a deterministic covariance-based observation schedule can therefore be a strong conventional comparator. A learned history-triggered sensing rule is not justified merely by calling sensing active. Preserve that comparator and matched observation counts; if a fixed late readout already suffices, accept it. State-dependent value must be demonstrated under the actual bounded/precision-constrained task rather than assumed.

## 6. CORRECTION: the unannounced control cutoff must be examined first

A consequential protocol issue was identified after the original data/sensor runs. The original MPC at time69 still optimizes eight controls (u69 through u76) and then12 zeros, while the evaluator discards u70 onward and forcibly sets16 actual zeros. The controller receives the current goal but not the impending loss of command authority. After a long near-steady target this mismatch can still matter: its current correction may rely on later counteractions that will never execute. This is task/action-availability metadata, not a new physical law or a deficiency corrected by more parameter data.

The information inequalities above remain valid. However, they do not establish that information causes most observed failures. The exact diagnostic for the saved known-parameter Kalman controller is

    b = C*A^17*posterior_mean69 + C*A^16*B*u69 - goal
    conditional endpoint MSE = b^2 + V.

`audit_precision_cutoff_v11.py` independently replays the known-parameter filter from saved observation/command arrays, verifies the recorded posterior means (maximum discrepancy1.78e-15), and saves every diagnostic in `audit_precision_cutoff_v11_results.json`. It imports neither canonical controller nor simulator. These are actual retained-array calculations, not hypothetical outcomes:

| Original condition | V | Mean b^2 | Mean conditional MSE | Realized endpoint MSE | Mean conditional endpoint success probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 lambda=.2, sensorSD=.2 | .054848637 | .021569157 | .076417793 | .084923524 | .720470623 |
| v10/v11 lambda=.7, sensorSD=.2 | .343422534 | 1.676571170 | 2.019993704 | 1.731874564 | .172740278 |
| v11 lambda=.7, sensorSD=.02 | .193027072 | .603350744 | .796377816 | .870677940 | .302807120 |

The slow original case therefore has about83% of its average conditional endpoint MSE in the mean term; even the more precise sensor case has about76%. These are descriptive decompositions for these histories, not a uniquely causal attribution to the cutoff alone. Model fit, finite-horizon objective and actual available actions must be separated by the planned repair.

For every audited episode, replacing only the last command by

    u69_center = u69 - b/(C*A^16*B)

can center the conditional final-position mean within the allowed [-1,1] bound. The required ranges are [-.04508,.07332] in the fast original case, [-.16066,.14229] in the slow original case and [-.12794,.10687] in the slow precise-sensor case. Thus the endpoint mean term is not forced by command saturation at these histories. This substitution is an endpoint diagnostic, not a proof that all trajectory/velocity stopping constraints can be satisfied, nor necessarily the action that optimizes their joint objective.

This finding supersedes the priority of the active-sensing proposal above. First announce the action cutoff and test a planner respecting it. Question the task contract before adding sensor-selection machinery. A lower uncertainty floor is useful knowledge, but it is not the explanation of most error when the controller plans unavailable future interventions.

## 7. Source review of the exploratory deadline repair

Reviewed `deadline_control_v11.py` before inspecting its added outcomes. The two additional variants both predict16 zero-command coast states. One retains eight free controls regardless of the cutoff; the other uses min(8,70-t) controls. This separates coast-length changes from the new deadline metadata by retaining the original8+12 result as a third comparison. Training rows/fits, actual plants, goals, test exogenous streams, filtering and score definitions are unchanged. This is explicitly an exploratory protocol repair, not a preregistered algorithm contest.

The wrapper resets its action-call counter each episode and calls exactly70 times. The underlying TRF planner is dimension-generic; truncating the shifted warm start as available actions shrink correctly discards commands beyond the deadline. At time69 the aware variant optimizes one last action plus16 true zero-command transitions, matching the physical action opportunities. Future goals remain undisclosed; only the cutoff is new supplied task information. Endpoint diagnostics are restricted to the known-parameter controller, where the saved estimate is the exact conditional mean.

No actual implementation error was found in this source review. Within a sensor condition the exact known-parameter covariance V is independent of the control policy and deadline metadata, so changes in b^2 directly expose changed mean decisions rather than changed uncertainty. Bounds on final-position probability remain ceilings, not guarantees of population-optimal whole-stop performance. Global expected closed-loop optimality is not supplied by convex open-loop shooting, especially with bounded actions, future replanning and a stricter pathwise success event.
