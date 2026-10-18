# Artificial agency: warranted goal reconsideration pilot

This repository is used for local, reproducible experiments on warranted goal reconsideration under acquired knowledge (WGR-AK). The initial engineering pilot is a fictional greenhouse with explicit causal dynamics, observation uncertainty, authority, paired cases and logged actions. It is not a claim of novel theory or an ecologically realistic simulator.

Research handoff: the user-provided attachment artificial_agency_research_handoff.md (18 October 2026). Frozen pilot design: research/protocol.md. Checkpoint with observed results: research/checkpoint.md.

Use the existing project environment, not global Python:

    /home/dtdav/dev/learn/.venv/bin/python -m unittest discover -s tests -v
    /home/dtdav/dev/learn/.venv/bin/python -m agency_pilot.experiment

The runner writes run-level JSON/JSONL artifacts under runs/. Inspect run traces before drawing any conclusions. The pilot uses only Python's standard library and has no real-world actuator, service, or API integrations.

Scientific caution: deterministic rule-bound agents in a synthetic domain validate the evaluator and demonstrate attainable task behavior. They cannot settle whether existing planning approaches generalize to truly novel scientific situations or prove that a new agency mechanism is needed.