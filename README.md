# First-principles robot learning

Research ledgers, reproducible one-dimensional car experiments, mathematical derivations, negative findings, and citations live in [research](research/).

**Current release:** [Living Research Ledger v5](research/first_principles_robot_learning_ledger_v5.md), self-contained with all prior ledgers preserved verbatim.

V5 tests a precise limitation of first-order action-response statistics: nonlinear actuator coupling can remain invisible at every centered probe amplitude. Mixed responses identify a hidden mode within a supplied bilinear family; established quadratic and kernel models test noisy transfer. V4 corrects v3's globally unstable drag simulator using exact dissipative flow. **No claim of universal learning or robotic-humanoid performance.**

**Run on the existing PC and project-local environment:**

~~~bash
cd /home/dtdav/dev/learn
.venv/bin/python research/coupled_response_v5.py > /tmp/coupled_response_v5_check.json
cmp research/coupled_response_v5_results.json /tmp/coupled_response_v5_check.json
~~~

Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 for committed v5 results. The local `.venv` is ignored by Git.

Previous releases: [v4](research/first_principles_robot_learning_ledger_v4.md), [v3](research/first_principles_robot_learning_ledger_v3.md), [v2](research/first_principles_robot_learning_ledger_v2.md), and [v1](research/first_principles_robot_learning_ledger_v1.md), with unchanged source/results and explicit corrections in later ledgers.
