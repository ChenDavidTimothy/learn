# Fixed causal tanh-PD diagnostic, frozen before execution

2026-10-20. This is a separate exploratory reference for the completed continuous-history engineering pilot. Canonical learning/controller outcomes are already known; this is not a blind confirmatory comparison. No PD outcome has been evaluated when writing this design. It will not select settings from heldout outcomes or claim optimized baseline strength.

Use the identical two goals/tasks, initial states, warmup, sensor streams, timing and torque bounds. Ordinary control receives only the currently announced goal and past/current noisy position readings. A causal quadratic polynomial fitted to the last 3/5/9 readings provides smoothed current position and slope. No true velocity, mass, model derivatives, future goals or future observations enter its action. Continue histories through switches.

Freeze a 3x3 diagnostic bank: windows 3/5/9, common joint gains `(kp,kd)=(2,2),(4,4),(8,8)`. Every torque is `-.5*tanh(kp*(qhat-goal))-.5*tanh(kd*vhat)`. Report all nine settings, all 18 episodes and the same tracking/late precision/effort metrics. Do not select a winner as a new primary method. This bank is a conventional reference, not an optimized PD competitor or proof that learning is necessary if it fails. Numerical plant integration uses an independent Hamiltonian implementation.

No additional model-training experience is used. Count every new actual simulator evaluation and sensor reading, retain complete trajectories/actions/goals and noise role paths, and preserve all failures.
