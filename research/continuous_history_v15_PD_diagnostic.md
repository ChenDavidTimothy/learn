# Fixed causal tanh-PD diagnostic

**TESTED / exploratory reference, 2026-10-20.** The design was frozen in `continuous_history_v15_PD_diagnostic_design.md` before any PD outcomes, after the learned engineering-pilot outcomes were known. This is not a fresh confirmatory or optimized PD comparison. Reproduce:

```sh
.venv/bin/python research/continuous_history_v15_PD_diagnostic.py > research/continuous_history_v15_PD_diagnostic_results.json
```

The script imports only the independent audit's physics/scoring helpers, not canonical research implementations. It uses the engineering pilot's two tasks, initial states, warmup, goal timing, bounds and paired sensor-noise paths. It fits a causal degree-two polynomial to the last 3/5/9 measured positions and applies

`u = -.5*tanh(kp*(qhat-current_goal)) -.5*tanh(kd*vhat)`.

True positions/velocities/inertia do not enter the action. No future goals or observations are used. The initial exact pose is used by the environment and scoring only; ordinary control reads noisy history. No histories are cleared at a goal switch. All nine prescribed `(window,kp,kd)` settings and all 18 episodes are retained; no winner is selected for a new primary claim.

| History readings | kp=kd | Task 0 position MSE | Task 1 position MSE | Precision goals /6 |
|---:|---:|---:|---:|---:|
| 3 | 2 | .0260529 | .0344768 | 0 |
| 3 | 4 | .0160241 | .0217100 | 1 |
| 3 | 8 | .0151311 | .0158485 | 2 |
| 5 | 2 | .0259128 | .0342414 | 0 |
| 5 | 4 | .0158691 | .0211432 | 1 |
| 5 | 8 | .0145981 | .0137254 | 2 |
| 9 | 2 | .0254306 | .0383913 | 0 |
| 9 | 4 | .0152737 | .0439640 | 0 |
| 9 | 8 | .0116502 | .0642047 | 0 |

Longer polynomial windows do not uniformly help. In the second task the window-nine/gain-eight case has late velocity RMS `.590641`, despite a better first-task average position score. This demonstrates why both taskwise outcomes and final motion matter; it is not a general instability theorem for polynomial observers. Failure of this small fixed bank does not establish learning necessity or an advantage over a properly optimized robust controller.

The independent Hamiltonian RK64 implementation agrees with RK32 locally to `1.75e-11`; a Radau check of the worst local interval differs by `1.17e-12`. Full states, observations, actions, goals, scores and original paired role paths remain. The source, independent helper and parent result hashes are recorded. It uses no new training data, but adds 18 simulator evaluation episodes, 3,384 intervals, 338.4 simulated seconds and 6,804 scalar coordinate sensor readings. Only two paired noise paths, totaling 756 distinct scalar random draws, underlie those repeated counterfactual evaluations. Reproduction and independent numerical checks add computation, not new independent training replicas.

These results concern the original two-task/replicate-zero engineering pilot. The production protocol uses different tasks and acquisition seeds, so this diagnostic is not a matched comparator for its eight-replicate statistics. No extrapolation of these PD counts into production is justified.
