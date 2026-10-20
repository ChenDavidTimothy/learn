# Independent audit of the payload-change instrument

**TESTED / independently verified, 2026-10-20. Preliminary instrument only.** No adaptation, detector, online parameter update or active probe is evaluated. Reproduce:

```sh
.venv/bin/python research/audit_payload_revision_v16_pilot.py > research/audit_payload_revision_v16_pilot_results.json
```

The audit imports only the previously independent v15 predictor/rollout helper, never canonical mechanics or controller code. Source/design/result/parent/helper hashes are recorded. The parent is precisely the protocol-fixed replicate-100 rich-H4 candidate, whose result hash matches the completed v15 independent audit. It is one previously acquired model, not eight new learned systems or a best-of-eight selection.

## Variable-mass mechanics and the loading event

From Cartesian point velocities, the kinetic energy is

\[
T=\tfrac12\|\dot r_1\|^2+\tfrac12m\|\dot r_2\|^2.
\]

This gives `M11=1+2m+2m*cos(q2)`, `M12=m*(1+cos(q2))`, `M22=m`, with determinant `m*(1+m*sin(q2)^2)>0` for positive mass. In canonical coordinates `p=Mv`, Hamilton's equations give

\[
\dot q_1={p_1-(1+c)p_2\over1+m\sin^2q_2},\quad
\dot q_2={-(1+c)p_1+(1/m+2+2c)p_2\over1+m\sin^2q_2},
\]

`pdot1=u1`, `pdot2=u2-m*sin(q2)*(v1^2+v1*v2)`. Unit mass recovers the previous dynamics algebraically.

The independent vectorized Hamiltonian RK4 implementation at 32/64 substeps checks all 24 random mass-mechanics cases, all twelve unit-parent comparisons and both exact zero-torque/rest cases. Its mechanics final-state discrepancy from source is `4.20e-11`; independent Cartesian kinetic-energy change agrees with torque work to `9.37e-12`, and cyclic momentum to `9.30e-15`. Stored unit-parent equality diagnostics agree exactly. The two masses remain observationally identical at exact rest and zero torque; the instrument therefore cannot justify an earlier change alarm in such a state.

At the imposed mass jump, coordinates and velocities remain continuous. This is matched-velocity external loading, not a closed-system conservation event. The generalized momentum changes by `(Mnew-Mold)*v`. The kinetic energy increases by half the squared Cartesian tip speed for the newly attached unit mass. These changes are independently reconstructed for every loaded trial and retained in the audit JSON. Energy-work conservation is checked only over fixed-mass motion; no claim of conservation across the external attachment is made.

## Data, ordinary control and timing

All 80 episodes, 15,040 physical intervals, sensor streams, mass schedules and complete states were independently checked. Maximum global control-state discrepancy is `8.77e-11`; coarse RK32/64 disagreement is `1.17e-10`. Goals, noise, warmup and all scoring match. There are four paired control noise paths and one separate mechanics path, with no collision among them or with the 682 enumerated previous v15 engineering/production acquisition-control paths.

All 1,440 retained MPC plans, warm costs, selected costs, issued prefixes and full complex-step rollout gradients are checked using the separate v15 implementation. Maximum projected-gradient discrepancy is `2.35e-14`; no optimizer failure is recorded. All 12,960 ordinary PD actions are recomputed from their causal polynomial position/slope summaries; they agree exactly. Every torque is bounded. The frozen MPC never receives payload mass/change labels, true current velocity or future goals/noise. PD receives current goal and noisy past/current positions only. Histories and warm plans continue through the switch.

All 40 matched unchanged/loaded pairs have bit-identical states and observations through index 68, the state just before the changed interval. **Their first loaded commands are also bit-identical**, and their first 61 MPC plans agree exactly. This is stronger than the source's command-prefix assertion. The first load-dependent information arrives only after the changed plant executes an action. No ordinary method anticipates the loading condition through a hidden label.

## What the instrument establishes

Every saved pre-/post-change score and all precision events were independently recomputed. The frozen MPC meets the exact same seven of eight post-change goal windows in both conditions: the per-task counts are `[2,1,2,2]`. Its mean post-change position MSE over the four fixed contexts increases from `.0107441` to `.0124801`. Thus the physical change has a cost consequence, but does not create an additional precision failure in these tests. It does **not** establish that adaptive learning or probing is necessary.

The full nine-setting fixed PD bank remains retained:

| Window | Gain | Unchanged post-change precision /8 | Loaded precision /8 |
|---:|---:|---:|---:|
| 3 | 2 | 0 | 0 |
| 3 | 4 | 4 | 0 |
| 3 | 8 | 4 | 4 |
| 5 | 2 | 0 | 0 |
| 5 | 4 | 4 | 0 |
| 5 | 8 | 4 | 4 |
| 9 | 2 | 0 | 0 |
| 9 | 4 | 4 | 0 |
| 9 | 8 | 0 | 4 |

No winner is selected from these outcomes. The bank is not globally optimized and its failures cannot establish learning necessity. Loading can damp an oscillatory fixed controller as well as slow goal response. Equal event counts do not imply equal trajectories/costs, and these four fixed tasks/one parent fit do not supply a population confidence interval or general robustness guarantee.

The body has no gravity or persistent force disturbance, so payload loading leaves all rest goal equilibria at zero torque unchanged. Existing feedback can correct errors despite an inaccurate frozen dynamic predictor. Calling that online feedback “adaptation” in ordinary language should not blur it with the untested claim that acquired parameters or representations must change. This instrument is a useful negative check on that proposed necessity.

## Costs and scope

The source performs 80 evaluation episode starts, 15,040 intervals, 1,504 simulated seconds and 30,240 scalar coordinate sensor readings. The four intentionally paired noise paths contain 1,512 distinct scalar noise draws. Mechanics adds 38 solver calls and 12.8 simulated seconds (24 mass cases, twelve old-parent reintegrations, two rest checks). Frozen MPC records 9,934 residual calls, each containing fifteen learned prediction steps. Original parent acquisition is reused and no new learner training occurs. Numerical audit/reproduction adds computation rather than new independent learned systems. All canonical resource totals match.

The external loading occurs at a goal switch in both matched trials; the matched unchanged condition separates its effect from the new goal in this experiment. Continuous rest-to-rest torque feasibility does not certify a sampled-input endpoint, every moving-state recovery, or arbitrary future loading. This is one ideal body, one mass change, one protocol-fixed model and four tasks. No milestone claiming adaptive representation, active information gathering or general robotics follows from this pilot alone. Preserve the result even if the premise that these tasks require relearning is rejected.
