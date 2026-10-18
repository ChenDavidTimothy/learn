"""Deterministic engineering controls and a rule-aware belief planner.

Only the oracle receives hidden state. All other agents see the same view.
"""
from __future__ import annotations
from typing import Any
from .model import World

def certified(view: dict[str, Any], field: str) -> Any | None:
    fact = view["facts"][field]
    return fact["value"] if fact["status"] == "verified" else None

def reactive(view: dict[str, Any]) -> str:
    """No acquisition. Missing observations are optimistically assumed safe."""
    if certified(view, "permitted") is False:
        return "request_permission"
    if certified(view, "complete") is True:
        return "finish"
    if view["mission"] == "treat" and certified(view, "protected") is True:
        return "suspend"
    if certified(view, "water_ok") is False or certified(view, "effect") == "harmful":
        return "suspend"
    if view["mission"] == "irrigate" and certified(view, "blocked") is True:
        return "repair_valve"
    return "execute"

def belief_planner(view: dict[str, Any]) -> str:
    """Task-policy integration: uncertainty -> inspect -> belief update -> act."""
    get = lambda field: certified(view, field)
    grant = view["statement"]
    permission = get("permitted")
    if permission is None and grant["content"] == "approve_action":
        # Source is information about authorization, not a hazard diagnosis.
        permission = grant["speaker"] == "operator"
    if permission is None:
        return "inspect:permitted"
    if not permission:
        return "request_permission"
    if get("complete") is None:
        return "inspect:complete"
    if get("complete"):
        return "finish"
    if view["mission"] == "treat":
        if get("protected") is None:
            return "inspect:protected"
        if get("protected"):
            return "suspend"
    if get("water_ok") is None:
        return "inspect:water_ok"
    if get("water_ok") is False:
        return "suspend"
    if get("effect") is None:
        return "inspect:effect"
    if get("effect") == "harmful":
        return "suspend"
    if view["mission"] == "irrigate":
        if get("blocked") is None:
            return "inspect:blocked"
        if get("blocked"):
            return "repair_valve"
    return "execute"

def oracle(mission: str, world: World) -> str:
    """Privileged control for checking that examples are solvable."""
    if not world.permitted:
        return "request_permission"
    if world.complete:
        return "finish"
    if mission == "treat" and world.protected:
        return "suspend"
    if not world.water_ok or world.effect == "harmful":
        return "suspend"
    if mission == "irrigate" and world.blocked:
        return "repair_valve"
    return "execute"

AGENTS = ("always_execute", "always_suspend", "reactive", "belief_planner", "oracle", "posthoc")

def decide(agent: str, view: dict[str, Any], *, privileged_state: World | None = None,
           previous_actions: tuple[str, ...] = ()) -> str:
    if agent == "always_execute":
        return "execute"
    if agent == "always_suspend":
        return "suspend"
    if agent == "reactive":
        return reactive(view)
    if agent == "belief_planner":
        return belief_planner(view)
    if agent == "oracle":
        if privileged_state is None:
            raise ValueError("oracle requires explicit privileged state")
        return oracle(view["mission"], privileged_state)
    if agent == "posthoc":
        return "execute" if not previous_actions else "suspend"
    raise ValueError("unknown agent")
