# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v3](research/first_principles_robot_learning_ledger_v3.md), self-contained with full prior v2 and v1 preserved verbatim.

V3 investigates **action-response fingerprints**: what randomized motor perturbations change in future measured positions, without requiring that a guessed hidden velocity be exactly observed. The benchmarks test unbiased open-loop impulse response, sensor noise, identifiability cancellation, nonlinear response sensitivity to action amplitude, and perturbations under feedback control. **No claim of universal learning or robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/car_learning_benchmark_v3.py > research/car_learning_benchmark_v3_results.json
~~~

Python 3.12.3 and NumPy 2.5.3 for committed results. The local `.venv` is ignored by Git.

Previous releases: [v2](research/first_principles_robot_learning_ledger_v2.md) and [v1](research/first_principles_robot_learning_ledger_v1.md), with corresponding unchanged Python and JSON experiments.
