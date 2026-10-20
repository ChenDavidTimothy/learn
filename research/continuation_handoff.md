# Active continuation after v11

The persistent Goal is **ACTIVE**. The latest verified living ledger is v11:
`first_principles_robot_learning_ledger_v11.md`. It preserves the complete v10
and earlier history verbatim. Commit `90b7e9b8198a1854982588a2ed3553c51681129c`
was pushed and verified on the existing SSH origin/main. V10 was published at
`450272cc1f34282b7dd2521db3972280982c2c7e`; v9 at
`426524a38a041b7e566fece00831b073cfdddab6`.

Read the new v11 section and independent audits before extending. The ultimate
general-learning objective remains unresolved; no release is a stopping condition.

## Verified v11 findings and correction

In the slow linear motor case, extending the same training trajectory from 512
to 8192 commands barely changes this controller's performance. Reducing position
sensor SD from .2 to .02 helps, but cannot reveal the latest hidden motor
innovation. Known-parameter conditioning gives a real information floor. Even
exact positions leave xi68 unobserved at action 69; future xi69 also matters during
the sixteen forced-zero transitions.

Crucial correction: that floor did not explain most original endpoint error. The
executor encoded a cutoff at step 70, but its fixed-eight-action planner was never
passed remaining availability. It optimized future commands that would not execute.
For the retained histories at SD .2, mean squared conditional endpoint bias was
1.676571, versus conditional variance .343423. Using N=min(8,70-t), with the same
sixteen predicted zero commands for both comparators, reduces bias squared to
.001556 and late position RMSE from .652 to .331. At SD .02 the corresponding bias
falls from .603351 to .000664; strict whole-stop counts improve from 3/24 to 6/24.

Passing the existing task constraint into planning was the needed repair; more
physical model complexity was not. This was an externally diagnosed correction,
not autonomous discovery. Stopping remains imperfect; no optimal policy or general
learning theorem is established. All prior numbers, failed pilots and corrected
interpretations remain preserved. Independent audits reconstructed all 336 new
control episodes and 23,520 plans, including nested data and conditional diagnostics.

Sources: `precision_information_v11.py`, `deadline_control_v11.py`, their actual
JSON results, and `audit_precision_information_v11`, `audit_precision_cutoff_v11`,
`audit_deadline_control_v11` sources/results/reports. Theory: `v11_theory_independent.md`.

## V12 already begun: genuinely competing use of experience

The next direction questions whether fitting physical/predictive dynamics is
needed for the present goal family. Direct improvement of behavior from experience
is a serious alternative. Conventional policy search is established, not novelty.
Do not continue accumulating isolated scalar covariance formulas.

`direct_experience_v12_pilot.py` and its actual JSON are an **UNRELEASED PILOT**.
Eight independent seeds train a supplied PD/smoothing family using CEM: four
generations, twelve candidates, two actual 86-step episodes each, totaling 8256
physical training steps per seed. Candidate fitness and all selection/elite updates
read only observations, issued commands and supplied goals. Hidden states and true
cost are logged for the evaluator and never used to select gains. Noise streams
are fresh per candidate and for held-out evaluation.

Training has zero motor process noise; testing has motor SD .03. Sensor SD is .2.
Feedback continues throughout all 86 test steps, with no forced cutoff. This changes
the task relative to v10/v11, so results cannot be presented as an improvement over
the earlier coast stopping counts. The fixed PD comparator is only descriptive and
weak, not a matched-data competing learner. The pilot's fresh true-cost mean is
.678067 +/- .185905 (SE across eight training seeds), with 3/16 stationary successes.
One plant, eight seeds and a restricted PD family do not establish generality or a
competitive advantage. No v12 ledger should be created solely from this pilot.

The independent theoretical audit is `audit_observation_loss_v12.md`; independent
artifact audit files are `audit_direct_experience_v12_pilot.py`, its
JSON, and the corresponding report. The independent artifact audit passed all 800 episodes and 384 candidate choices.
It does not complete the missing learning-method comparison. Preserve the pilot and its limitations when extending it.

## Observation-only fitness: useful identity and counterexamples

For the declared lagged dynamics, v[t+1] and x[t+1] depend on issued commands only
through u[t-1] and earlier. With fresh, independent, **zero-mean** iid sensor errors,
variance R, the expected observed stage loss

    (o[t+1]-g[t])^2 + .3*(o[t+1]-o[t])^2 + .03*u[t]^2

exceeds the true position/velocity stage loss by the policy-independent constant
1.6R. For a fixed exogenous goal schedule, divide by the same fixed normalizer.
Knowing R is unnecessary merely to rank expectations. Finite fitness remains noisy;
selecting a lucky winner makes its training score optimistic. Only fresh evaluation
of the frozen selected policy supports the fixed-policy expectation identity.
Biased or colored sensors, future-noise reuse, data-dependent normalization and
adaptive termination can invalidate the argument. See the independent proof.

A bounded MOTOR-FIRST one-step counterexample is executed: u=-k*tanh(e0/SD) gives
expected loss gap 1.6R+.6*beta*k*E[e0*tanh(e0/SD)], which depends on k. Quadrature
and one million Monte Carlo draws are retained. Do not generalize the lagged result.

## Exact next work for a meaningful V12 comparison

1. Read the pilot, actual results and both audits. Keep its preliminary status and
   negative findings. Do not mistake beating a fixed poor PD gain for progress
   against a strong learning method.
2. Strengthen the direct controller to a causal dynamic feedback class with goal
   error, multiple observation-difference lags and past-command lags, or compact
   learned filter memory. Bound outputs. Use serious conventional random search,
   ARS, CEM or another appropriate policy optimizer, with training-only tuning,
   fresh evaluation and every physical interaction counted.
3. Fit a **shadow OE/filter/MPC learner to the SAME collected experience**, including
   every failed candidate episode, rather than only elites. For training process
   noise zero and known zero resets, the joint/sequential closed-loop likelihood
   factors into theta-independent actual policy terms and Gaussian observation
   residuals. Sum raw-position SSE over all reset segments, restarting simulated
   states and sensitivities at each boundary. Do not assert exogeneity conditional
   on the entire future input sequence. Nonzero process noise requires a separate
   latent-state argument. The theory audit derives these distinctions explicitly.
4. Compare checkpoint interaction budgets and identical observations, commands,
   goals and task metadata. Model-based virtual planning/fitting is legitimate but
   report its compute separately. Both model and controller vocabularies are
   supplied inductive biases; neither is a universal foundation. Optionally compare
   separate budget-matched acquisition policies, explicitly separating acquisition
   from the use of the same experience.
5. Keep executor, planner and score consistent about future action availability.
   Freeze fresh evaluation goals/noise/metrics before inspecting outcomes. Treat
   continuous station keeping and forced shutdown as different tasks, not hidden
   implementation details. Count validation/retesting/reset episodes honestly.
6. Only after a meaningful audited comparison or another consequential finding,
   build v12 embedding complete v11, retain executable sources/results/audits,
   commit and push over existing SSH, verify, and immediately continue. Accept a
   result favoring simple established methods. Escalate coupling or unknown
   function vocabulary only to resolve a specific remaining uncertainty.

Primary literature investigated: [Rubinstein's CEM](https://doi.org/10.1023/A:1010091220143),
[CE tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf),
[Mania, Guy and Recht's random-search policies and evaluation variability](https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf),
and [Peters and Schaal's policy gradients](https://people.csail.mit.edu/rplatt/IROS2006-Peters.pdf).
The pilot does not implement ARS or compare contemporary robotics methods.

Use the owner's PC, canonical repository and existing `.venv`. No external blocker
or routine approval is needed. Keep the Goal active; do not manufacture success.
