# Independent audit of continuous-history learning and sustained control

**TESTED / independently verified, 2026-10-20.** This report concerns the fresh eight-replicate protocol in `continuous_history_v15.py`, with complete results `continuous_history_v15_rep100.json` through `rep107.json`. The earlier engineering pilot and fixed PD diagnostic remain separate development evidence.

```sh
.venv/bin/python research/audit_continuous_history_v15.py > research/audit_continuous_history_v15_results.json
```

The audit imports only its own independent pilot-audit helper. It imports no canonical mechanics, features, predictor, planner or scoring function. The executable audit retains per-replicate caches, checks their source/result/helper hashes before reuse, and reconstructs the aggregate from all eight complete canonical artifacts. The final audit JSON records canonical source/design hashes, every result hash and independent source hashes. The theory auditor executed the same independent helper for the last two replicates; all eight use the same independent checks and cache guards.

## Verified data and fitted relations

All 576 acquisition trajectories, 55,296 physical intervals, role paths, random initial positions/hidden velocities, held-two-sample commands and sensor records were independently reconstructed. The entire command sequence is exogenous. Amplitude is `.35` in even-index trajectories and `1` in odd-index trajectories. All acquired cases, including high velocities, are retained. Training/validation/testing are distinct whole trajectories; input scaling uses fitting data only.

The audit checks 608 distinct acquisition/control role paths and their generated 128-bit initialization states. No state collides with the 74 earlier continuous engineering-pilot role states or 27,929 enumerated earlier v12/v13/v14 and history-pilot integer-seed states. Control methods intentionally share a noise role for each fit/task. Their actions and physical trajectories differ. The fixed derivative-check random path is computation, not another independent training/test role.

All 240 candidate fits were independently reconstructed using equivalent Fourier feature algebra and batched moment computations. The largest ridge stationarity residual is `2.00e-15`; all feature scales, validation observation risks, retained true-state diagnostics and selected IDs agree. Hidden true validation states do not choose a candidate. Both nonlinear raw-history and polynomial-compression learners receive causal issued-action history as well as measured positions. The nonlinear vocabulary contains supplied periodic context features, history/current-action products and recent-motion quadratics; it does not discover the observation language or physical equations without priors.

Selected raw delays across the eight fits are `4,2,2,4,4,4,1,2`. Fixed-rich uses H4 and all polynomial fits select H2 (three current/past readings). Linear ARX selects H4. Where selected and rich IDs match, their complete controllers/trajectories are byte-identical. This equality was checked. It is not evidence that validation-driven history order is better than a fixed rich choice.

All independent one-step forecasts and recursive ten-step forecasts agree. Recursive predictions do not use future observed positions; their issued future torque blocks are exogenous. Composing noisy conditional means remains a certainty-equivalent approximation, not a proof of an exact finite-history Markov process. The differenced regression target shares current measurement noise with the inputs; the audit does not infer unbiased physical parameter estimates.

## Physics and action planning

Separate Hamiltonian RK4 at 32/64 substeps checks ordinary acquisition and low-speed control trajectories. Maximum global discrepancy from saved acquisition states is `2.61e-9`; low-speed control discrepancy is `2.98e-11`. Coarse 32/64 disagreement in that checked group is at most `4.14e-8`.

Thirty failed high-speed control trajectories require additional numerical care. Adaptive Hamiltonian Radau checks each saved local transition to `2.86e-11`; independent whole-trajectory Radau replay differs by at most `2.13e-7` over 18.8 seconds. Coarse global RK32/64 disagreement reaches `1.34e-5`. These are explicit failed-pose numerical-sensitivity diagnostics, not hidden truncations or dropped cases. No fine successful-regulation claim depends on matching those divergent failed trajectories to roundoff. The earlier pilot has a separately retained, larger `8.78e-7` global replay discrepancy.

All 192 control episodes, 36,096 physical intervals, observation streams, goals, commanded bounds and 34,560 selected plans were checked. Every selected plan cost, warm-start cost, nonincrease condition, shifted warm start, issued first action and reported projected gradient was independently recomputed. **All 34,560 full rollout Jacobians** were checked by complex-step differentiation of the separate predictor/rollout implementation. Maximum projected-gradient discrepancy is `9.64e-14`; plan-cost discrepancies are below `3.64e-12`.

Eight optimizer failures are retained, all in pose-predictor controllers, all status 0 at 40 evaluations. Their selected plans still have finite costs no larger than their warm plans. Failure locations and gradients are preserved in the audit JSON. Successful solver status can arise from step/objective tolerances and is not a global or uniform stationarity certificate. Internal failed proposal/iteration histories are not saved; selected plans and consequences are what were independently verified.

Ordinary predictor/planner calls use measured positions, issued commands and the currently announced goal only. Exact current position, velocity, inertia and future sensor samples are absent from ordinary control. Eight zero-torque warmup intervals initialize history from noisy observations. The body, histories and warm plan continue through goal switches. The “pose” label describes a current-position predictive model; its locally optimized warm-started planner still retains indirect controller memory. Its failure is not a general impossibility theorem for all memoryless policies.

## Independently recomputed results

Means/MCSE average the four fixed tasks within each acquired-data fit and then use eight independent fits as clusters. The 96 goal windows per method are dependent within clusters and are not 96 independently trained systems. Position MSE is mean squared Euclidean joint-position error; late position RMSE and velocity RMS are componentwise quantities. Precision requires every position and instantaneous velocity coordinate within `.03` for every sample in the final second of each six-second goal interval.

| Predictor | Position MSE ± clustered SE | Mean late position RMSE | Mean late velocity RMS | Precision goals /96 |
|---|---:|---:|---:|---:|
| Current position | `40.8557 ±12.6265` | `4.47992` | `1.63431` | 0 |
| One delay | `.00799830 ±.0000274573` | `.00337595` | `.0151149` | 45 |
| Fixed rich H4 | `.00800049 ±.0000161504` | `.00150783` | `.00835841` | 79 |
| Selected raw delay | `.00800289 ±.0000171694` | `.00175801` | `.00936443` | 76 |
| Linear ARX | `.00803391 ±.0000945344` | `.00585483` | `.0745795` | 16 |
| Polynomial compression | `.00798909 ±.0000177505` | `.00166543` | `.0105603` | 69 |

Simple conventional history learners achieve useful sustained behavior in this instrument. This is a constructive step beyond rest-pulse return demonstrations. It does not establish that a new general architecture is needed, that raw histories are universally preferable to observers, or that selecting history is superior to supplying a reasonable fixed rich window.

The paired fixed-rich-minus-selected position difference is `−2.39714e-6 ±9.30659e-6` SE. The one-delay-minus-selected difference is `−4.59112e-6 ±1.89615e-5`. Neither supports a mean tracking gain from selected order. Mean position tracking is similar across the non-pose alternatives, while precise motion regulation differs substantially. The linear ARX's low average tracking score coexists with 16/96 precision events and much higher late velocity. These are important limits on interpreting one scalar average.

Mean test ten-step forecast MSEs for one/rich/selected/linear/polynomial are `.00151421 / .000482382 / .000479951 / .0226832 / .000368877`. The polynomial learner has the lowest mean recursive error here but fewer precision events than fixed-rich. Selected raw history also has lower mean one-step test error than rich, without better precision. Prediction/decision disagreement survives real learning and a common planner; this is not a reason to silently select a new metric or tune another method on these heldout tasks.

## Costs and scope

Fresh acquisition uses 576 episode starts, 55,296 intervals, 5,529.6 simulated seconds and 111,744 scalar joint-coordinate readings. Control uses 192 episode starts, 36,096 intervals (including eight warmups each), 3,609.6 simulated seconds and 72,576 scalar readings. Planning records 274,343 residual calls, independently checked to equal the sum of each plan's `nfev+2`; at 15 prediction steps per call this is 4,115,145 learned-model steps. Forecasting adds 96,768 predictor steps and derivative checks 3,264; candidate fitting/linear algebra and independent numerical audits are separate computation. No new physical observations come from virtual planning.

The prior 72-trajectory engineering pilot and the 18-episode fixed PD bank add development resources, retained in their separate reports. PD is evaluated on the original two pilot tasks only, not on the fresh four-task/eight-fit protocol, so its failures cannot be imported as a matched fresh comparison or evidence that learning is necessary. Reexecutions reproduce seeded cases rather than adding independent learned fits. No hardware experiment or runtime-efficiency theorem is claimed.

Limits remain substantial: one ideal frictionless fully actuated body, direct calibrated torque, unwrapped joint-angle observations, known timing/noise/bounds, supplied nonlinear feature families and history menus, no process disturbance, four fixed evaluation poses and one hidden initial velocity vector. No cross-body transfer, language discovery, contact, underactuation or humanoid capability is demonstrated. Held-two-input training creates present/past-input dependence; the deficient pose summary cannot be assumed causally sufficient merely because the full input sequence is exogenous. Local MPC and mean prediction are approximations. All mathematical/empirical claims should retain these qualifications.

The defensible result is conventional acquired use of causal history for this coupled continuous task, with no demonstrated advantage from adaptive order selection over fixed rich history. The next important uncertainty concerns which distinctions and validation criteria support useful decisions under changed conditions, rather than another isolated filter/ridge calibration ladder.
