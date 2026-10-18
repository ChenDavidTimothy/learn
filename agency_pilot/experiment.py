"""Run paired synthetic cases, record actual traces, and summarize controls."""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .agents import AGENTS, decide
from .model import MAX_STEPS, Case, Greenhouse, make_cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "2026-10-18-wgr-pilot"

def run_case(case: Case, condition: str, agent: str) -> dict:
    env = Greenhouse(case, condition)
    for _ in range(MAX_STEPS):
        if env.ended:
            break
        view = env.observe()
        # Privileged state is passed ONLY to the upper-bound oracle.
        state = env.world if agent == "oracle" else None
        previous = tuple(t["action"] for t in env.trace)
        action = decide(agent, view, privileged_state=state, previous_actions=previous)
        env.act(action)
    return {
        "pair": case.pair, "variant": case.variant, "case_id": case.case_id,
        "condition": condition, "seed": case.seed, "agent": agent,
        "mission": case.mission,
        **env.result(),
    }

def run_all() -> list[dict]:
    return [
        run_case(case, condition, agent)
        for condition in ("A", "B")
        for case in make_cases()
        for agent in AGENTS
    ]

def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["agent"])].append(row)
    summaries = []
    for (condition, agent), items in sorted(grouped.items()):
        pair_values: dict[str, list[bool]] = defaultdict(list)
        for row in items:
            pair_values[row["pair"]].append(row["success"])
        paired = sum(len(vals) == 2 and all(vals) for vals in pair_values.values())
        summaries.append({
            "condition": condition, "agent": agent, "cases": len(items),
            "successes": sum(row["success"] for row in items),
            "pair_successes": paired, "pairs": len(pair_values),
            "prohibited_execute_events": sum(
                sum(e["action"] == "execute" and bool(e.get("violations")) for e in row["trace"])
                for row in items
            ),
            "inspection_actions": sum(row["inspections"] for row in items),
            "mean_steps": round(sum(row["steps"] for row in items) / len(items), 3),
        })
    return {
        "suite": "WGR-AK engineering pilot v0.1",
        "design": "11 paired families x 2 variants x 2 conditions x 6 policies",
        "seeds": [case.seed for case in make_cases()],
        "step_limit": MAX_STEPS,
        "summary": summaries,
    }

def main() -> None:
    rows = run_all()
    summary = summarize(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    trace_path = OUT / "traces.jsonl"
    summary_path = OUT / "summary.json"
    with trace_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("cases:", len(make_cases()), "policy episodes:", len(rows))
    print("traces:", trace_path)
    print("summary:", summary_path)
    for row in summary["summary"]:
        print(f'{row["condition"]} {row["agent"]:>15}: '
              f'{row["successes"]:>2}/{row["cases"]}, '
              f'pairs {row["pair_successes"]:>2}/{row["pairs"]}, '
              f'unsafe {row["prohibited_execute_events"]:>2}, '
              f'inspections {row["inspection_actions"]:>2}, mean steps {row["mean_steps"]}')

if __name__ == "__main__":
    main()
