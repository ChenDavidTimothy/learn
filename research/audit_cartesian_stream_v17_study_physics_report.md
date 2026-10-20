# Independent final fresh-study physical artifact audit

2026-10-20. **PASS: all fixed fresh replicates200–207.** This is an artifact
correctness result, not a claim that the ordinary controllers solve the task
reliably. The fresh ordinary outcomes are substantially worse than the pilot.

The parameterized audit source was prepared and compile-checked before any fresh
outcome inspection and was unchanged for execution. It reuses the earlier
independent Hamiltonian RK4/Radau implementation, Christoffel torque/IK routines
and sustained-score reconstruction. This is new artifact verification, not a
claim of a new independent derivation. It imports no canonical plant, learner or
controller. Another auditor handles fitting, selection, ordinary regressions,
append content, plan costs, gradients and optimizer decisions.

Executed in the existing owner's environment, retaining first-batch caches:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_cartesian_stream_v17_study_physics.py --replicates 200 201 202 203
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_cartesian_stream_v17_study_physics.py --replicates 204 205 206 207
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/aggregate_cartesian_stream_v17_study_physics_audit.py
```

Each replicate has an immutable `audit_cartesian_stream_v17_repN_study_physics_results.json`.
The final `audit_cartesian_stream_v17_study_physics_results.json` verifies every
cache's source hash and all final source/dependency/acquisition/fit/raw-result/
raw-array hashes again. The aggregate executable performs arithmetic and hash
checks, not new fitting or model selection. The study source is verified against
its recorded full hash, with expected prefix `a213d5eb`. No completed corpus was
repeated to alter an outcome or audit scope.

## Complete physical and information scope

All **81,920 acquired intervals** and **37,632 control/reference intervals**
independently replay: **119,552 local intervals** total. All **512 complete
trajectories** (320 acquired,192 control/reference) also roll from their actual
initial states without replacing intermediate states. Local replay uses256 RK4
substeps and complete replay512. Maximum local state discrepancy is **3.19e-11**;
maximum complete-trajectory state discrepancy **1.24e-8**. Independent Radau
checks cover **430 acquisition** and **986 control** intervals, including fixed
initial/midpoint/goal-boundary locations and extrema of speed, joint motion,
Jacobian singularity and distal phase. Maximum Radau discrepancy: **1.36e-11**.

All **1,112** acquisition/bootstrap/control RNG role paths match their prescribed
namespaces and are distinct across corpora. Acquisition and control draws are
fully reconstructed; bootstrap draws are the separate fitting auditor's scope.
All acquired actions, noise, camera readings, initial preparations, ordinary
split/index arrays, energy/work and sampled support diagnostics agree. No
truth-based filtering is introduced. Every actual ordinary/reference observation
matches its physical position plus the prescribed shared camera-noise path.

Within every paired case, retained/frozen/reference initial conditions, warmup
physics/readings and sensor-derived goals coincide. Retained/frozen first actual
commands and resulting states coincide before append could change a decision.
All **5,760 privileged reference commands** are checked using independent IK,
nearest winding/branch choice, metric-derived Coriolis terms, computed torque and
clipping. All **11,520 ordinary decisions** are checked for command timing,
bounds, held-pair commitment and current-goal access; evaluating their learned
plans is a separate audit. Every sustained score profile, paired difference and
recorded resource counter reconstructs. The references have23 saturated decisions
and no clipped goals, and attain all192 precision windows.

## Actual fresh outcomes, including failures

Costs below are per-episode means within a corpus. Precise windows are out of24
per method/corpus. SEs below the table are descriptive sample SD over eight corpus
means divided by sqrt(8); frames, phases and paired noise repetitions are not
independent trained bodies.

| Corpus | Retained cost | Frozen cost | Frozen−retained | Retained precise | Frozen precise |
|---|---:|---:|---:|---:|---:|
|200|25.957540|31.285091|5.327551|3|4|
|201|4.393150|23.411875|19.018725|11|0|
|202|36.034507|47.099127|11.064620|6|4|
|203|33.823355|42.404271|8.580916|2|0|
|204|32.448449|55.584235|23.135786|7|3|
|205|7.002559|14.279107|7.276549|12|0|
|206|42.866271|55.451320|12.585050|2|0|
|207|0.087520|0.125768|0.038248|12|8|

Retained task cost is **22.826669 ±5.834942 SE**, versus frozen
**33.705099 ±7.100576**. The paired frozen-minus-retained mean is
**10.878431 ±2.625521**, positive in all eight corpora. Retained precision is
**55/192=28.6458%**, frozen **19/192=9.8958%**. The corpus-mean paired precision
increase is **18.75 ±6.7737 percentage points**, with one corpus negative.
Retained phase counts are7/64,19/64,29/64; frozen3/64,6/64,10/64. The privileged
reference is192/192 with mean task cost0.032796. It has more information and is
not an ordinary competing learner.

**All310 failed ordinary goal windows are retained and indexed** in the final
aggregate, with corpus/method/case/phase, score, late maxima and raw-artifact
references. The source/raw streams retain their full trajectories and errors.
No reference window failed. Neither recorded ordinary optimizer termination
failure nor fallback occurs, which does not establish a physically valid model,
a globally optimized plan, adequate acquisition support or reliable control.

The result supports a beneficial total closed-loop effect of appending ordinary
experience for this fixed construction and study. It simultaneously rejects
extrapolating the pilot22/24 result into a reliable fresh task-solving claim.
The frozen variant still uses an acquired library, causal history, contextual
refitting and feedback; it is not an unlearned or no-adaptation baseline.
Policies can visit different histories after their actions diverge, so these
contrasts do not isolate a single-row effect at matched histories or feedback's
separate contribution. Better later-phase counts alone cannot attribute change
to learning rather than elapsed motion, goals and feedback. No source, model,
seed scope, metric or goal was changed in response to these outcomes.

## Resources and limits

Acquisition uses8,192 simulated seconds and164,480 independent scalar camera
noise draws. Ordinary control adds128 episodes,25,088 physical intervals,
2,508.8 seconds,384 sustained goal windows and11,520 decisions. There are25,216
camera-read/50,432 scalar-read opportunities, but25,216 distinct scalar control
noise draws because methods share64 paired paths from32 preparations.
Retained experience contributes5,760 actual row events, copied into three member
libraries for17,280 append copies; these are not independent additional physical
experiences. References add64 episodes/12,544 intervals/1,254.4 seconds and no
ordinary training experiences. Actual recorded counts sum to2,320,108 acquisition
RHS evaluations,743,476 ordinary RHS evaluations,326,264 reference RHS evaluations
and126,281 modeled value/gradient calls. Counters reconstruct from artifacts;
this audit does not independently observe historical timing or establish real-time
control.

Precision uses all ten saved endpoints67–76,127–136,187–196 and is not a
continuous-time guarantee. Sampled singularity/velocity coverage is diagnostic,
not a support filter or a uniform envelope. The study concerns one fixed body
and the declared preparation/acquisition/task distributions, with supplied history
length, features, local estimation and planner. It neither proves architectural
incapacity from failures nor universal learning from the append benefit. Further
causal diagnosis and a consequential next experiment remain necessary; outcomes
are not model-selection input.
