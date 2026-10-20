# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v9](research/first_principles_robot_learning_ledger_v9.md), self-contained with all prior ledgers preserved verbatim.

V9 shows that nonlinear motion still need not locate force on the body versus motor without physical restrictions or new interventions. In the restricted linear-actuator family, nonlinear terms identify labels with enough excitation; conventional output-error learning handles noisy position much better than naive nonlinear regression, while weak-excitation transfer still fails. V8 proves linear pole-allocation ambiguity, v7 studies conditional intervention weighting, v6 context/response geometry, v5 nonlinear coupling, and v4 a simulator correction. **No claim of universal learning or robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/nonlinear_force_gauge_v9.py > /tmp/nonlinear_force_gauge_v9_check.json
cmp research/nonlinear_force_gauge_v9_results.json /tmp/nonlinear_force_gauge_v9_check.json
.venv/bin/python research/nonlinear_identification_v9.py > /tmp/nonlinear_identification_v9_check.json
cmp research/nonlinear_identification_v9_results.json /tmp/nonlinear_identification_v9_check.json
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. The local `.venv` is ignored by Git.

Previous releases: [v8](research/first_principles_robot_learning_ledger_v8.md), [v7](research/first_principles_robot_learning_ledger_v7.md), [v6](research/first_principles_robot_learning_ledger_v6.md), [v5](research/first_principles_robot_learning_ledger_v5.md), [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
