#!/usr/bin/env python3
"""
TRPC-0 Exhaustive One-File Validation Harness
==============================================

Purpose
-------
This script is a deliberately exact, finite, code-level falsification test for the
TRPC-0 reference realization in the distilled TRPC foundation.

It is NOT an RL benchmark and does NOT train a neural network.  It implements the
finite TRPC-0 chain end-to-end on a small sequential controlled system where every
relevant probability can be computed exactly:

    declared competence
      -> recursive task-relative predictive object
      -> exact task-relative quotient
      -> adequate-controller correspondence
      -> experiment-indistinguishability collapse
      -> controller-invalidating confusions
      -> Hellinger max-min fixed allocation
      -> finite MLE
      -> adequate task-conditioned controller.

The script also contains deliberate adversarial tests intended to fail if a core
TRPC-0 assumption is removed or implemented incorrectly.

Dependencies
------------
Python >= 3.10
numpy
scipy

Example
-------
    python trpc0_exhaustive_validation.py

Optional:
    python trpc0_exhaustive_validation.py \
        --report trpc0_validation_report.txt \
        --mc-trials 200000 \
        --seed 20260906

The full result is written to the report TXT file.  Exit code 0 means every
mandatory validation passed. Exit code 2 means at least one mandatory validation
failed. Unexpected exceptions produce exit code 1 and are written to the report.

Scientific scope
----------------
This script validates the exact finite TRPC-0 reference mechanism.  It does NOT
validate novelty, practical superiority, learned quotient discovery, approximate
representation learning, continuous control, irreversible single-life learning,
or any theorem outside the finite exact assumptions explicitly encoded here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linprog


# =============================================================================
# 0. Global benchmark definition
# =============================================================================

SCRIPT_VERSION = "TRPC0-EXHAUSTIVE-1.0"

TASKS: Tuple[str, ...] = ("copy_hidden_bit", "invert_hidden_bit")
POLICIES: Tuple[str, ...] = ("trust_sensor", "invert_sensor")
EXPERIMENTS: Tuple[str, ...] = ("probe_A", "probe_B")
SENSORS: Tuple[str, ...] = ("operational",) + EXPERIMENTS

DEFAULT_EPSILON = 0.11
DEFAULT_DELTA = 0.05
DEFAULT_SEED = 20260906
DEFAULT_MC_TRIALS = 100_000
DEFAULT_MAX_RECURSION_DEPTH = 3
DEFAULT_NUISANCE_SIZES: Tuple[int, ...] = (1, 2, 4, 8, 16, 64, 256)
DEFAULT_CURVE_MULTIPLIERS: Tuple[float, ...] = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)

# The benchmark is constructed so that:
#   H: g_plus is required
#   M: both base policies are within epsilon on each task, hence all four
#      task-conditioned controllers are epsilon-adequate
#   L: g_minus is required
# while probes A/B emphasize different harmful distinctions.
BENCHMARK_Q_SPECS: Dict[str, "QuotientSpec"] = {}

FLOAT_ATOL = 1e-12
SIGNATURE_DECIMALS = 12


# =============================================================================
# 1. Data structures and reporting
# =============================================================================

@dataclass(frozen=True)
class QuotientSpec:
    name: str
    theta_operational: float
    theta_probe_A: float
    theta_probe_B: float

    def sensor_accuracy(self, sensor: str) -> float:
        if sensor == "operational":
            return self.theta_operational
        if sensor == "probe_A":
            return self.theta_probe_A
        if sensor == "probe_B":
            return self.theta_probe_B
        raise KeyError(f"Unknown sensor/experiment: {sensor}")

    def experiment_success_prob(self, experiment: str) -> float:
        return self.sensor_accuracy(experiment)


BENCHMARK_Q_SPECS.update(
    {
        "H": QuotientSpec("H", 0.80, 0.58, 0.90),
        "M": QuotientSpec("M", 0.55, 0.82, 0.62),
        "L": QuotientSpec("L", 0.20, 0.20, 0.48),
    }
)


@dataclass(frozen=True)
class FullModel:
    """Full environment = relevant quotient dynamics + irrelevant nuisance channel."""

    name: str
    q: QuotientSpec
    nuisance_size: int

    def __post_init__(self) -> None:
        if self.nuisance_size < 1:
            raise ValueError("nuisance_size must be >= 1")


@dataclass(frozen=True)
class Controller:
    name: str
    mapping: Tuple[Tuple[str, str], ...]

    def policy_for(self, task: str) -> str:
        d = dict(self.mapping)
        return d[task]


@dataclass
class CompiledQuotient:
    name: str
    full_models: List[FullModel]
    representative_q: QuotientSpec
    task_signature: Tuple[float, ...]
    experiment_signature: Tuple[float, ...]


@dataclass
class ExperimentClass:
    name: str
    quotients: List[CompiledQuotient]
    law: Dict[str, float]
    adequate_intersection: Set[str]


@dataclass
class AllocationResult:
    name: str
    weights: Dict[str, float]
    kappa: float
    pairs: List[Tuple[str, str]]
    per_pair_rates: Dict[Tuple[str, str], float]
    lp_status: str
    coarse_grid_weight_probe_A: Optional[float] = None
    coarse_grid_kappa: Optional[float] = None


@dataclass
class ExactErrorResult:
    true_class: str
    controller_error: float
    class_id_error: float
    harmful_wrong_class_mass: float
    harmless_wrong_class_mass: float
    tie_mass: float
    output_controller_mass: Dict[str, float]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    mandatory: bool = True


class Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: List[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(str(text))

    def section(self, title: str) -> None:
        self.line()
        self.line("=" * 88)
        self.line(title)
        self.line("=" * 88)

    def subsection(self, title: str) -> None:
        self.line()
        self.line("-" * 88)
        self.line(title)
        self.line("-" * 88)

    def kv(self, key: str, value: object) -> None:
        self.line(f"{key:<42}: {value}")

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# =============================================================================
# 2. Generic probability helpers
# =============================================================================

def clamp_prob(p: float) -> float:
    if p < -FLOAT_ATOL or p > 1.0 + FLOAT_ATOL:
        raise ValueError(f"Invalid probability {p}")
    return min(1.0, max(0.0, float(p)))


def bernoulli_obs_prob(theta: float, z: int, obs: int) -> float:
    """P(obs | Z=z) for a binary symmetric sensor with correctness theta."""
    if z not in (0, 1) or obs not in (0, 1):
        raise ValueError("z and obs must be binary")
    return theta if obs == z else 1.0 - theta


def predictive_obs_one(p_z1: float, theta: float) -> float:
    """P(obs=1 | current belief P(Z=1)=p_z1)."""
    p_z1 = clamp_prob(p_z1)
    return p_z1 * theta + (1.0 - p_z1) * (1.0 - theta)


def bayes_update(p_z1: float, theta: float, obs: int) -> float:
    """Exact scalar recursive predictive state update."""
    p_z1 = clamp_prob(p_z1)
    num = p_z1 * bernoulli_obs_prob(theta, 1, obs)
    den = num + (1.0 - p_z1) * bernoulli_obs_prob(theta, 0, obs)
    if den <= 0.0:
        raise ZeroDivisionError("Conditioning on a zero-probability event")
    return clamp_prob(num / den)


def brute_force_posterior(
    q: QuotientSpec, history: Sequence[Tuple[str, int]], prior: float = 0.5
) -> float:
    """Posterior computed from scratch by enumerating Z in {0,1}."""
    weights = []
    for z in (0, 1):
        w = prior if z == 1 else 1.0 - prior
        for sensor, obs in history:
            w *= bernoulli_obs_prob(q.sensor_accuracy(sensor), z, obs)
        weights.append(w)
    den = weights[0] + weights[1]
    if den <= 0.0:
        raise ZeroDivisionError("History has zero probability")
    return clamp_prob(weights[1] / den)


def target_bit(task: str, z: int) -> int:
    if task == "copy_hidden_bit":
        return z
    if task == "invert_hidden_bit":
        return 1 - z
    raise KeyError(task)


def policy_action(policy: str, sensor_obs: int) -> int:
    if policy == "trust_sensor":
        return sensor_obs
    if policy == "invert_sensor":
        return 1 - sensor_obs
    raise KeyError(policy)


def exact_task_policy_value(q: QuotientSpec, task: str, policy: str) -> float:
    """Enumerate Z and operational observation exactly."""
    total = 0.0
    for z in (0, 1):
        pz = 0.5
        for s in (0, 1):
            ps = bernoulli_obs_prob(q.theta_operational, z, s)
            action = policy_action(policy, s)
            reward = 1.0 if action == target_bit(task, z) else 0.0
            total += pz * ps * reward
    return total


def state_task_action_success(p_z1: float, task: str, action: int) -> float:
    """Predict terminal task success from the recursive belief state."""
    if action not in (0, 1):
        raise ValueError("action must be binary")
    p_z1 = clamp_prob(p_z1)
    total = 0.0
    for z, pz in ((0, 1.0 - p_z1), (1, p_z1)):
        total += pz * (1.0 if action == target_bit(task, z) else 0.0)
    return total


def predictive_vector(q: QuotientSpec, p_z1: float) -> Dict[str, float]:
    """
    Explicit finite predictive witness used for K2 validation.

    It contains future observation predictions for all three sensor actions and
    terminal success predictions for both tasks and both terminal actions.
    """
    out: Dict[str, float] = {}
    for sensor in SENSORS:
        out[f"P({sensor}=1)"] = predictive_obs_one(
            p_z1, q.sensor_accuracy(sensor)
        )
    for task in TASKS:
        for action in (0, 1):
            out[f"P(success|task={task},action={action})"] = (
                state_task_action_success(p_z1, task, action)
            )
    return out


def controller_value(q: QuotientSpec, controller: Controller, task: str) -> float:
    return exact_task_policy_value(q, task, controller.policy_for(task))


def enumerate_controllers() -> List[Controller]:
    controllers: List[Controller] = []
    for choices in itertools.product(POLICIES, repeat=len(TASKS)):
        mapping = tuple(zip(TASKS, choices))
        d = dict(mapping)
        if (
            d["copy_hidden_bit"] == "trust_sensor"
            and d["invert_hidden_bit"] == "invert_sensor"
        ):
            name = "g_plus"
        elif (
            d["copy_hidden_bit"] == "invert_sensor"
            and d["invert_hidden_bit"] == "trust_sensor"
        ):
            name = "g_minus"
        elif all(v == "trust_sensor" for v in choices):
            name = "g_trust_both"
        elif all(v == "invert_sensor" for v in choices):
            name = "g_invert_both"
        else:
            name = "g_" + "__".join(choices)
        controllers.append(Controller(name=name, mapping=mapping))

    preference = {
        "g_plus": 0,
        "g_minus": 1,
        "g_trust_both": 2,
        "g_invert_both": 3,
    }
    controllers.sort(key=lambda c: (preference.get(c.name, 99), c.name))
    return controllers


def adequate_controller_names(
    q: QuotientSpec, controllers: Sequence[Controller], epsilon: float
) -> Set[str]:
    best_by_task = {
        task: max(exact_task_policy_value(q, task, pi) for pi in POLICIES)
        for task in TASKS
    }
    adequate: Set[str] = set()
    for g in controllers:
        worst_regret = 0.0
        for task in TASKS:
            value = controller_value(q, g, task)
            regret = best_by_task[task] - value
            worst_regret = max(worst_regret, regret)
        if worst_regret <= epsilon + FLOAT_ATOL:
            adequate.add(g.name)
    return adequate


def controller_regret_table(
    q: QuotientSpec, controllers: Sequence[Controller]
) -> Dict[str, Dict[str, float]]:
    best = {
        task: max(exact_task_policy_value(q, task, pi) for pi in POLICIES)
        for task in TASKS
    }
    return {
        g.name: {
            task: best[task] - controller_value(q, g, task) for task in TASKS
        }
        for g in controllers
    }


# =============================================================================
# 3. Exact recursive predictive-state validation
# =============================================================================

def exhaustive_recursion_validation(
    q_specs: Mapping[str, QuotientSpec], max_depth: int
) -> Tuple[int, float, float]:
    """
    Exhaustively compare sequential Bayes updates and predictive vectors against
    from-scratch hidden-state enumeration over all sensor/action-observation
    histories up to max_depth.

    Returns:
        number_of_histories_checked,
        maximum_posterior_absolute_error,
        maximum_predictive_absolute_error
    """
    histories_checked = 0
    max_post_err = 0.0
    max_pred_err = 0.0

    for q in q_specs.values():
        for depth in range(max_depth + 1):
            for sensor_seq in itertools.product(SENSORS, repeat=depth):
                for obs_seq in itertools.product((0, 1), repeat=depth):
                    history = list(zip(sensor_seq, obs_seq))
                    p_rec = 0.5
                    for sensor, obs in history:
                        p_rec = bayes_update(
                            p_rec, q.sensor_accuracy(sensor), obs
                        )
                    p_brute = brute_force_posterior(q, history)
                    max_post_err = max(max_post_err, abs(p_rec - p_brute))

                    vec = predictive_vector(q, p_rec)
                    # Cross-check each vector component by direct hidden-state sum.
                    for sensor in SENSORS:
                        direct = 0.0
                        for z, pz in ((0, 1.0 - p_brute), (1, p_brute)):
                            direct += pz * bernoulli_obs_prob(
                                q.sensor_accuracy(sensor), z, 1
                            )
                        key = f"P({sensor}=1)"
                        max_pred_err = max(
                            max_pred_err, abs(vec[key] - direct)
                        )
                    for task in TASKS:
                        for action in (0, 1):
                            direct = 0.0
                            for z, pz in (
                                (0, 1.0 - p_brute),
                                (1, p_brute),
                            ):
                                direct += pz * (
                                    1.0
                                    if action == target_bit(task, z)
                                    else 0.0
                                )
                            key = (
                                f"P(success|task={task},action={action})"
                            )
                            max_pred_err = max(
                                max_pred_err, abs(vec[key] - direct)
                            )
                    histories_checked += 1

    return histories_checked, max_post_err, max_pred_err


def recursive_policy_value_from_state(
    q: QuotientSpec, task: str, policy: str
) -> float:
    """One-step dynamic-programming evaluation using the recursive state."""
    prior = 0.5
    total = 0.0
    theta = q.theta_operational
    for obs in (0, 1):
        p_obs = (
            predictive_obs_one(prior, theta)
            if obs == 1
            else 1.0 - predictive_obs_one(prior, theta)
        )
        post = bayes_update(prior, theta, obs)
        action = policy_action(policy, obs)
        continuation = state_task_action_success(post, task, action)
        total += p_obs * continuation
    return total


def frozen_state_counterfactual_value(q: QuotientSpec, task: str) -> float:
    """
    Counterfactual K2 deletion test: ignore the observation and freeze p=1/2.
    The best terminal action then has value 1/2 on either task.
    """
    p = 0.5
    return max(state_task_action_success(p, task, a) for a in (0, 1))


# =============================================================================
# 4. Task-relative quotient compiler and exact interface checks
# =============================================================================

def rounded_tuple(xs: Iterable[float]) -> Tuple[float, ...]:
    return tuple(round(float(x), SIGNATURE_DECIMALS) for x in xs)


def task_predictive_signature(model: FullModel) -> Tuple[float, ...]:
    """
    Deliberately task/deployment-relative signature.  Nuisance does not enter.

    We include operational-sensor predictions and residual predictions after both
    operational observations, plus all supported task-policy values.
    """
    q = model.q
    vals: List[float] = []

    # Initial operational observation prediction.
    vals.append(predictive_obs_one(0.5, q.theta_operational))

    # Residual predictive state after operational obs 0/1.
    for obs in (0, 1):
        post = bayes_update(0.5, q.theta_operational, obs)
        vec = predictive_vector(q, post)
        # Only deployment/task-relevant entries belong to the task quotient here.
        vals.append(post)
        for task in TASKS:
            for action in (0, 1):
                vals.append(
                    vec[f"P(success|task={task},action={action})"]
                )

    # Supported task-policy semantics.
    for task in TASKS:
        for pi in POLICIES:
            vals.append(exact_task_policy_value(q, task, pi))

    return rounded_tuple(vals)


def enumerate_raw_experiment_success_prob(model: FullModel, experiment: str) -> float:
    """
    Derive P(Y_e=1 | M,e) from the raw controlled experiment, not by assertion.

    Raw episode:
      Z ~ Bernoulli(1/2)
      nuisance N ~ Uniform({0,...,K-1}), independent
      probe observation S_e passes Z through a binary symmetric channel
      experiment policy chooses terminal action A=S_e
      observable summary Y_e is terminal success 1{A=Z}

    N is explicitly summed out.  Therefore the resulting summary law must be
    invariant to nuisance size when the factorization assumptions hold.
    """
    if experiment not in EXPERIMENTS:
        raise KeyError(experiment)
    theta = model.q.sensor_accuracy(experiment)
    total = 0.0
    for z in (0, 1):
        pz = 0.5
        for nuisance in range(model.nuisance_size):
            pn = 1.0 / model.nuisance_size
            _ = nuisance  # raw observation is present but deliberately discarded by Y_e
            for s in (0, 1):
                ps = bernoulli_obs_prob(theta, z, s)
                action = s
                y = 1 if action == z else 0
                if y == 1:
                    total += pz * pn * ps
    return clamp_prob(total)


def exact_full_model_task_policy_value(model: FullModel, task: str, policy: str) -> float:
    """Raw full-model enumeration including nuisance; task ignores nuisance."""
    total = 0.0
    theta = model.q.theta_operational
    for z in (0, 1):
        pz = 0.5
        for nuisance in range(model.nuisance_size):
            pn = 1.0 / model.nuisance_size
            _ = nuisance
            for s in (0, 1):
                ps = bernoulli_obs_prob(theta, z, s)
                action = policy_action(policy, s)
                reward = 1.0 if action == target_bit(task, z) else 0.0
                total += pz * pn * ps * reward
    return total


def experiment_signature_for_model(model: FullModel) -> Tuple[float, ...]:
    return rounded_tuple(
        enumerate_raw_experiment_success_prob(model, e) for e in EXPERIMENTS
    )


def compile_task_relative_quotients(
    full_models: Sequence[FullModel],
    refine_for_experiment_interface: bool = True,
) -> Tuple[List[CompiledQuotient], int]:
    """
    Compile by task-predictive signature first. If a candidate quotient does not
    have a well-defined observable experiment law, optionally refine it by the
    experiment signature, exactly as required by the TRPC-0 execution contract.

    Returns (compiled_quotients, number_of_base_groups_that_required_refinement).
    """
    base: Dict[Tuple[float, ...], List[FullModel]] = {}
    for m in full_models:
        base.setdefault(task_predictive_signature(m), []).append(m)

    groups: List[List[FullModel]] = []
    refinement_count = 0
    for members in base.values():
        by_exp: Dict[Tuple[float, ...], List[FullModel]] = {}
        for m in members:
            by_exp.setdefault(experiment_signature_for_model(m), []).append(m)
        if len(by_exp) > 1:
            refinement_count += 1
            if not refine_for_experiment_interface:
                groups.append(members)
            else:
                groups.extend(by_exp.values())
        else:
            groups.append(members)

    compiled: List[CompiledQuotient] = []
    for members in groups:
        underlying_names = sorted({m.q.name for m in members})
        if len(underlying_names) == 1:
            qname = underlying_names[0]
        else:
            qname = "MERGED[" + ",".join(underlying_names) + "]"
        rep = members[0].q
        compiled.append(
            CompiledQuotient(
                name=qname,
                full_models=list(members),
                representative_q=rep,
                task_signature=task_predictive_signature(members[0]),
                experiment_signature=experiment_signature_for_model(members[0]),
            )
        )

    compiled.sort(key=lambda q: q.name)
    return compiled, refinement_count


def experiment_interface_is_well_defined(q: CompiledQuotient) -> bool:
    sigs = {experiment_signature_for_model(m) for m in q.full_models}
    return len(sigs) == 1


def uniform_fibre_adequate_intersection(
    q: CompiledQuotient,
    controllers: Sequence[Controller],
    epsilon: float,
) -> Set[str]:
    intersections: Optional[Set[str]] = None
    for m in q.full_models:
        g = adequate_controller_names(m.q, controllers, epsilon)
        intersections = set(g) if intersections is None else intersections & g
    return intersections or set()


# =============================================================================
# 5. Observable experiment classes and identifiability preflight
# =============================================================================

def collapse_experiment_equivalence(
    quotients: Sequence[CompiledQuotient],
    adequate_by_q: Mapping[str, Set[str]],
) -> List[ExperimentClass]:
    groups: Dict[Tuple[float, ...], List[CompiledQuotient]] = {}
    for q in quotients:
        groups.setdefault(q.experiment_signature, []).append(q)

    classes: List[ExperimentClass] = []
    for idx, sig in enumerate(sorted(groups.keys())):
        qs = sorted(groups[sig], key=lambda x: x.name)
        law = {e: sig[i] for i, e in enumerate(EXPERIMENTS)}
        inter: Optional[Set[str]] = None
        for q in qs:
            g = adequate_by_q[q.name]
            inter = set(g) if inter is None else inter & g
        name = "C[" + ",".join(q.name for q in qs) + "]"
        classes.append(
            ExperimentClass(
                name=name,
                quotients=qs,
                law=law,
                adequate_intersection=inter or set(),
            )
        )
    classes.sort(key=lambda c: c.name)
    return classes


def identifiability_preflight(classes: Sequence[ExperimentClass]) -> Tuple[bool, List[str]]:
    bad = [c.name for c in classes if not c.adequate_intersection]
    return len(bad) == 0, bad


def global_safe_controllers(classes: Sequence[ExperimentClass]) -> Set[str]:
    inter: Optional[Set[str]] = None
    for c in classes:
        inter = (
            set(c.adequate_intersection)
            if inter is None
            else inter & c.adequate_intersection
        )
    return inter or set()


def choose_safe_selector(
    classes: Sequence[ExperimentClass], controllers: Sequence[Controller]
) -> Dict[str, str]:
    preference = [c.name for c in controllers]
    selector: Dict[str, str] = {}
    for cls in classes:
        choices = cls.adequate_intersection
        if not choices:
            raise ValueError(f"Class {cls.name} has no safe controller")
        selected = next((g for g in preference if g in choices), None)
        if selected is None:
            selected = sorted(choices)[0]
        selector[cls.name] = selected
    return selector


def harmful_confusions(
    classes: Sequence[ExperimentClass], selector: Mapping[str, str]
) -> List[Tuple[str, str]]:
    by_name = {c.name: c for c in classes}
    pairs: List[Tuple[str, str]] = []
    for true_cls in classes:
        for chosen_cls in classes:
            if true_cls.name == chosen_cls.name:
                continue
            g = selector[chosen_cls.name]
            if g not in by_name[true_cls.name].adequate_intersection:
                pairs.append((true_cls.name, chosen_cls.name))
    return sorted(pairs)


# =============================================================================
# 6. Hellinger geometry and fixed-allocation LP
# =============================================================================

def bernoulli_hellinger_sq(p: float, q: float) -> float:
    p = clamp_prob(p)
    q = clamp_prob(q)
    return 1.0 - math.sqrt(p * q) - math.sqrt((1.0 - p) * (1.0 - q))


def pair_hellinger_vector(
    c1: ExperimentClass, c2: ExperimentClass
) -> Dict[str, float]:
    return {
        e: bernoulli_hellinger_sq(c1.law[e], c2.law[e])
        for e in EXPERIMENTS
    }


def solve_maxmin_allocation(
    name: str,
    classes: Sequence[ExperimentClass],
    pairs: Sequence[Tuple[str, str]],
) -> AllocationResult:
    if not pairs:
        return AllocationResult(
            name=name,
            weights={e: 0.0 for e in EXPERIMENTS},
            kappa=math.inf,
            pairs=[],
            per_pair_rates={},
            lp_status="NO_PAIRS",
        )

    by_name = {c.name: c for c in classes}
    m = len(EXPERIMENTS)

    # Variables: w_0,...,w_{m-1}, kappa. Max kappa <=> min -kappa.
    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0

    A_ub = []
    b_ub = []
    for a, b in pairs:
        hv = pair_hellinger_vector(by_name[a], by_name[b])
        row = [-hv[e] for e in EXPERIMENTS] + [1.0]
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = [[1.0] * m + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * m + [(0.0, None)]

    res = linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(A_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Allocation LP failed: {res.message}")

    weights = {e: float(res.x[i]) for i, e in enumerate(EXPERIMENTS)}
    kappa = float(res.x[-1])
    per_pair_rates: Dict[Tuple[str, str], float] = {}
    for a, b in pairs:
        hv = pair_hellinger_vector(by_name[a], by_name[b])
        per_pair_rates[(a, b)] = sum(weights[e] * hv[e] for e in EXPERIMENTS)

    result = AllocationResult(
        name=name,
        weights=weights,
        kappa=kappa,
        pairs=list(pairs),
        per_pair_rates=per_pair_rates,
        lp_status=str(res.message),
    )

    # Independent dense-grid cross-check for this two-experiment benchmark.
    if len(EXPERIMENTS) == 2:
        grid = np.linspace(0.0, 1.0, 100_001)
        best_k = -1.0
        best_w = None
        e0, e1 = EXPERIMENTS
        for w0 in grid:
            rates = []
            for a, b in pairs:
                hv = pair_hellinger_vector(by_name[a], by_name[b])
                rates.append(w0 * hv[e0] + (1.0 - w0) * hv[e1])
            kval = min(rates)
            if kval > best_k:
                best_k = kval
                best_w = float(w0)
        result.coarse_grid_weight_probe_A = best_w
        result.coarse_grid_kappa = best_k

    return result


def all_distinct_pairs(classes: Sequence[ExperimentClass]) -> List[Tuple[str, str]]:
    names = [c.name for c in classes]
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                pairs.append((a, b))
    return pairs


def theorem_scalar_budget(kappa: float, K: int, delta: float) -> int:
    if K <= 1:
        return 0
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(f"Invalid kappa={kappa}")
    return int(math.ceil(math.log((K - 1) / delta) / kappa))


def integer_episode_allocation(n: int, weights: Mapping[str, float]) -> Dict[str, int]:
    return {e: int(math.ceil(n * weights[e])) for e in EXPERIMENTS}


# =============================================================================
# 7. Exact finite MLE error enumeration
# =============================================================================

def safe_log_prob(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    return math.log(p)


def log_count_likelihood(k: int, n: int, p: float) -> float:
    """Candidate log likelihood up to the binomial coefficient."""
    if k < 0 or k > n:
        return -math.inf
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    return k * math.log(p) + (n - k) * math.log(1.0 - p)


def log_binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_choose + k * math.log(p) + (n - k) * math.log(1.0 - p)


def mle_class_from_counts(
    counts: Mapping[str, int],
    n_by_e: Mapping[str, int],
    classes: Sequence[ExperimentClass],
) -> Tuple[str, bool]:
    scores: List[float] = []
    for cls in classes:
        score = 0.0
        for e in EXPERIMENTS:
            score += log_count_likelihood(counts[e], n_by_e[e], cls.law[e])
        scores.append(score)
    max_score = max(scores)
    winners = [i for i, s in enumerate(scores) if abs(s - max_score) <= 1e-13]
    # Deterministic tie break: classes are sorted by name.
    return classes[winners[0]].name, len(winners) > 1


def exact_error_enumeration(
    true_class: ExperimentClass,
    classes: Sequence[ExperimentClass],
    selector: Mapping[str, str],
    n_by_e: Mapping[str, int],
) -> ExactErrorResult:
    class_by_name = {c.name: c for c in classes}
    controller_error = 0.0
    class_id_error = 0.0
    harmful_mass = 0.0
    harmless_wrong_mass = 0.0
    tie_mass = 0.0
    out_mass: Dict[str, float] = {}

    ranges = [range(n_by_e[e] + 1) for e in EXPERIMENTS]
    total_prob = 0.0
    for ks in itertools.product(*ranges):
        counts = {e: int(k) for e, k in zip(EXPERIMENTS, ks)}
        logp = 0.0
        for e in EXPERIMENTS:
            logp += log_binom_pmf(
                counts[e], n_by_e[e], true_class.law[e]
            )
        prob = 0.0 if logp == -math.inf else math.exp(logp)
        if prob == 0.0:
            continue
        total_prob += prob

        chosen_name, tied = mle_class_from_counts(counts, n_by_e, classes)
        chosen_controller = selector[chosen_name]
        out_mass[chosen_controller] = out_mass.get(chosen_controller, 0.0) + prob
        if tied:
            tie_mass += prob

        wrong_class = chosen_name != true_class.name
        if wrong_class:
            class_id_error += prob

        bad_controller = chosen_controller not in true_class.adequate_intersection
        if bad_controller:
            controller_error += prob
            if wrong_class:
                harmful_mass += prob
        elif wrong_class:
            harmless_wrong_mass += prob

    if abs(total_prob - 1.0) > 1e-9:
        raise RuntimeError(
            f"Exact probability enumeration for {true_class.name} summed to {total_prob}"
        )

    return ExactErrorResult(
        true_class=true_class.name,
        controller_error=controller_error,
        class_id_error=class_id_error,
        harmful_wrong_class_mass=harmful_mass,
        harmless_wrong_class_mass=harmless_wrong_mass,
        tie_mass=tie_mass,
        output_controller_mass=out_mass,
    )


# =============================================================================
# 8. Monte Carlo cross-check
# =============================================================================

def monte_carlo_errors(
    rng: np.random.Generator,
    true_class: ExperimentClass,
    classes: Sequence[ExperimentClass],
    selector: Mapping[str, str],
    n_by_e: Mapping[str, int],
    trials: int,
) -> Tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan

    counts = {
        e: rng.binomial(n_by_e[e], true_class.law[e], size=trials)
        for e in EXPERIMENTS
    }
    scores = np.zeros((len(classes), trials), dtype=float)
    for i, cls in enumerate(classes):
        for e in EXPERIMENTS:
            p = cls.law[e]
            k = counts[e]
            n = n_by_e[e]
            if p == 0.0:
                term = np.where(k == 0, 0.0, -np.inf)
            elif p == 1.0:
                term = np.where(k == n, 0.0, -np.inf)
            else:
                term = k * math.log(p) + (n - k) * math.log(1.0 - p)
            scores[i] += term

    winners = np.argmax(scores, axis=0)  # deterministic first-index tie break
    true_idx = next(i for i, c in enumerate(classes) if c.name == true_class.name)
    class_err = float(np.mean(winners != true_idx))

    bad = np.zeros(trials, dtype=bool)
    for idx, cls in enumerate(classes):
        mask = winners == idx
        g = selector[cls.name]
        if g not in true_class.adequate_intersection:
            bad |= mask
    controller_err = float(np.mean(bad))
    return controller_err, class_err


# =============================================================================
# 9. Deliberate adversarial / negative tests
# =============================================================================

def synthetic_quotient(
    name: str, theta0: float, pa: float, pb: float, nuisance_size: int = 1
) -> FullModel:
    return FullModel(
        name=f"{name}_N{nuisance_size}",
        q=QuotientSpec(name, theta0, pa, pb),
        nuisance_size=nuisance_size,
    )


def build_main_full_models(nuisance_sizes: Sequence[int]) -> List[FullModel]:
    models: List[FullModel] = []
    for q in BENCHMARK_Q_SPECS.values():
        for k in nuisance_sizes:
            models.append(FullModel(name=f"{q.name}_N{k}", q=q, nuisance_size=k))
    return models


def evaluate_problem(
    full_models: Sequence[FullModel],
    controllers: Sequence[Controller],
    epsilon: float,
) -> Tuple[
    List[CompiledQuotient],
    Dict[str, Set[str]],
    List[ExperimentClass],
    Dict[str, str],
    List[Tuple[str, str]],
    AllocationResult,
    AllocationResult,
    int,
]:
    quotients, refinements = compile_task_relative_quotients(full_models, True)
    adequate_by_q: Dict[str, Set[str]] = {}
    for q in quotients:
        adequate_by_q[q.name] = uniform_fibre_adequate_intersection(
            q, controllers, epsilon
        )
    classes = collapse_experiment_equivalence(quotients, adequate_by_q)
    ok, bad = identifiability_preflight(classes)
    if not ok:
        raise ValueError(f"UNLEARNABLE classes: {bad}")
    selector = choose_safe_selector(classes, controllers)
    harmful = harmful_confusions(classes, selector)
    readiness = solve_maxmin_allocation("TRPC readiness", classes, harmful)
    rep = solve_maxmin_allocation(
        "Exact observable-class identification", classes, all_distinct_pairs(classes)
    )
    return (
        quotients,
        adequate_by_q,
        classes,
        selector,
        harmful,
        readiness,
        rep,
        refinements,
    )


def adversarial_tests(
    controllers: Sequence[Controller],
    reporter: Reporter,
    main_classes: Sequence[ExperimentClass],
    main_selector: Mapping[str, str],
    readiness: AllocationResult,
    rep: AllocationResult,
    trpc_exact: Mapping[str, ExactErrorResult],
) -> List[CheckResult]:
    results: List[CheckResult] = []

    # A. K2 uniform fibre soundness: H and L cannot share a nonempty exact certificate.
    g_h = adequate_controller_names(BENCHMARK_Q_SPECS["H"], controllers, DEFAULT_EPSILON)
    g_l = adequate_controller_names(BENCHMARK_Q_SPECS["L"], controllers, DEFAULT_EPSILON)
    fibre_inter = g_h & g_l
    results.append(
        CheckResult(
            "K2 rejects an unsafe H/L representation fibre",
            len(fibre_inter) == 0,
            f"G(H) intersection G(L) = {sorted(fibre_inter)}",
        )
    )

    # B. Exact experiment-indistinguishability + disjoint answers => unlearnable.
    bad_models = [
        synthetic_quotient("BAD_H", 0.80, 0.50, 0.50),
        synthetic_quotient("BAD_L", 0.20, 0.50, 0.50),
    ]
    bad_qs, _ = compile_task_relative_quotients(bad_models, True)
    bad_g = {
        q.name: uniform_fibre_adequate_intersection(q, controllers, DEFAULT_EPSILON)
        for q in bad_qs
    }
    bad_classes = collapse_experiment_equivalence(bad_qs, bad_g)
    learnable, bad_names = identifiability_preflight(bad_classes)
    results.append(
        CheckResult(
            "Identifiability preflight rejects indistinguishable incompatible models",
            not learnable,
            f"learnable={learnable}; rejected classes={bad_names}",
        )
    )

    # C. Indistinguishable models with common adequate answers are valid and need no ID.
    good_models = [
        synthetic_quotient("GOOD_M1", 0.55, 0.50, 0.50),
        synthetic_quotient("GOOD_M2", 0.52, 0.50, 0.50),
    ]
    good_qs, _ = compile_task_relative_quotients(good_models, True)
    good_g = {
        q.name: uniform_fibre_adequate_intersection(q, controllers, DEFAULT_EPSILON)
        for q in good_qs
    }
    good_classes = collapse_experiment_equivalence(good_qs, good_g)
    good_learnable, _ = identifiability_preflight(good_classes)
    good_global = global_safe_controllers(good_classes)
    results.append(
        CheckResult(
            "Indistinguishable compatible models collapse to a zero-experiment decision",
            good_learnable and len(good_classes) == 1 and bool(good_global),
            f"classes={len(good_classes)}; global_safe={sorted(good_global)}",
        )
    )

    # D. Execution-interface refinement: same task semantics, different probe laws.
    iface_models = [
        synthetic_quotient("IFACE_1", 0.55, 0.30, 0.70),
        synthetic_quotient("IFACE_2", 0.55, 0.70, 0.30),
    ]
    no_refine, base_refine_count = compile_task_relative_quotients(
        iface_models, refine_for_experiment_interface=False
    )
    refined, refined_count = compile_task_relative_quotients(
        iface_models, refine_for_experiment_interface=True
    )
    results.append(
        CheckResult(
            "Observable experiment interface forces quotient refinement when needed",
            len(no_refine) == 1 and base_refine_count == 1 and len(refined) == 2 and refined_count == 1,
            (
                f"unrefined_groups={len(no_refine)}, refined_groups={len(refined)}, "
                f"flagged_base_groups={refined_count}"
            ),
        )
    )

    # E. K4 must be weaker than all-pairs ID on this benchmark, strictly.
    strict_gap = rep.kappa + 1e-10 < readiness.kappa
    results.append(
        CheckResult(
            "Controller-readiness separation is strictly easier than all-pairs ID",
            strict_gap,
            f"kappa_TRPC={readiness.kappa:.12g}; kappa_rep={rep.kappa:.12g}",
        )
    )

    # F. K3: wrong model/class can be harmless. M is designed so every controller is adequate.
    m_class = next(
        (c.name for c in main_classes if any(q.name == "M" for q in c.quotients)),
        None,
    )
    if m_class is None:
        results.append(CheckResult("Readiness-vs-identity harmless-error test", False, "M class missing"))
    else:
        er = trpc_exact[m_class]
        results.append(
            CheckResult(
                "K3 permits class-identification errors that are controller-harmless",
                er.class_id_error > er.controller_error + 1e-12,
                (
                    f"true={m_class}; class_id_error={er.class_id_error:.12g}; "
                    f"controller_error={er.controller_error:.12g}; "
                    f"harmless_wrong_mass={er.harmless_wrong_class_mass:.12g}"
                ),
            )
        )

    # G. K1 competence tolerance changes the answer geometry.
    g_m_009 = adequate_controller_names(BENCHMARK_Q_SPECS["M"], controllers, 0.09)
    g_m_011 = adequate_controller_names(BENCHMARK_Q_SPECS["M"], controllers, 0.11)
    all_061 = [
        adequate_controller_names(q, controllers, 0.61)
        for q in BENCHMARK_Q_SPECS.values()
    ]
    global_061 = set.intersection(*map(set, all_061))
    results.append(
        CheckResult(
            "K1 tolerance changes adequate-answer geometry and can remove acquisition",
            g_m_009 != g_m_011 and bool(global_061),
            (
                f"G_M(eps=.09)={sorted(g_m_009)}; "
                f"G_M(eps=.11)={sorted(g_m_011)}; "
                f"global_safe(eps=.61)={sorted(global_061)}"
            ),
        )
    )

    # H. K4 anti-decoration check: harmful set must be a strict subset of all ordered pairs.
    all_pairs = set(all_distinct_pairs(main_classes))
    harmful_pairs = set(harmful_confusions(main_classes, main_selector))
    results.append(
        CheckResult(
            "K4 is operative: benchmark really removes harmless pair constraints",
            harmful_pairs < all_pairs,
            f"harmful_pairs={len(harmful_pairs)}; all_pairs={len(all_pairs)}",
        )
    )

    reporter.subsection("Adversarial negative tests")
    for r in results:
        reporter.line(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}")
        reporter.line(f"       {r.detail}")
    return results


# =============================================================================
# 10. Report rendering helpers
# =============================================================================

def fmt_prob(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.12g}"


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    str_rows = [[str(x) for x in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line]
    out.append("|" + "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(line)
    for row in str_rows:
        out.append("|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(line)
    return out


def self_sha256() -> str:
    try:
        data = Path(__file__).read_bytes()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return "UNAVAILABLE"


def module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return f"ERROR: {exc}"


# =============================================================================
# 11. Main validation
# =============================================================================

def run_validation(args: argparse.Namespace, reporter: Reporter) -> int:
    t0 = time.perf_counter()
    checks: List[CheckResult] = []
    controllers = enumerate_controllers()

    reporter.section("TRPC-0 EXHAUSTIVE ONE-FILE VALIDATION REPORT")
    reporter.kv("script_version", SCRIPT_VERSION)
    reporter.kv("script_sha256", self_sha256())
    reporter.kv("generated_utc", datetime.now(timezone.utc).isoformat())
    reporter.kv("python", sys.version.replace("\n", " "))
    reporter.kv("platform", platform.platform())
    reporter.kv("numpy", module_version("numpy"))
    reporter.kv("scipy", module_version("scipy"))
    reporter.kv("seed", args.seed)
    reporter.kv("epsilon", args.epsilon)
    reporter.kv("delta", args.delta)
    reporter.kv("mc_trials_per_true_class", args.mc_trials)
    reporter.kv("max_recursion_depth", args.max_recursion_depth)
    reporter.kv("nuisance_sizes", list(args.nuisance_sizes))
    reporter.kv("curve_multipliers", list(args.curve_multipliers))

    reporter.section("0. WHAT THIS SCRIPT IS TESTING")
    reporter.line("K1  Declared competence: fixed tasks, supported policies/controllers, epsilon.")
    reporter.line("K2  Recursive predictive soundness: exact recursive state and exact task values.")
    reporter.line("K3  Terminal target: adequate controller, not quotient identity.")
    reporter.line("K4  Acquisition: only controller-invalidating confusions constrain design.")
    reporter.line()
    reporter.line("TRPC-0 finite machinery: exact supplied class, quotient compilation, observable")
    reporter.line("experiment laws, identifiability collapse, safe selector, Hellinger max-min")
    reporter.line("fixed allocation, finite MLE, fixed-confidence theorem budget.")

    # -------------------------------------------------------------------------
    # K1: competence contract
    # -------------------------------------------------------------------------
    reporter.section("1. DECLARED COMPETENCE CONTRACT (K1)")
    reporter.kv("tasks U", TASKS)
    reporter.kv("supported policies Pi", POLICIES)
    reporter.kv("epsilon", args.epsilon)
    reporter.kv("controller count |Pi|^|U|", len(controllers))
    reporter.kv("common implementation map L", "policy_action(policy, operational_sensor_observation)")
    for g in controllers:
        reporter.line(f"  {g.name}: {dict(g.mapping)}")

    # -------------------------------------------------------------------------
    # K2: recursive predictive state
    # -------------------------------------------------------------------------
    reporter.section("2. EXHAUSTIVE RECURSIVE PREDICTIVE-STATE VALIDATION (K2)")
    n_hist, max_post_err, max_pred_err = exhaustive_recursion_validation(
        BENCHMARK_Q_SPECS, args.max_recursion_depth
    )
    reporter.kv("histories checked", n_hist)
    reporter.kv("max recursive-vs-bruteforce posterior error", f"{max_post_err:.3e}")
    reporter.kv("max predictive-vector error", f"{max_pred_err:.3e}")
    recursion_pass = max_post_err <= 1e-12 and max_pred_err <= 1e-12
    checks.append(
        CheckResult(
            "Exact recursive predictive state agrees with brute-force enumeration",
            recursion_pass,
            f"posterior_err={max_post_err:.3e}, predictive_err={max_pred_err:.3e}",
        )
    )

    reporter.subsection("Dynamic-programming policy value check")
    value_rows = []
    max_value_err = 0.0
    for q in BENCHMARK_Q_SPECS.values():
        for task in TASKS:
            for pi in POLICIES:
                enum_v = exact_task_policy_value(q, task, pi)
                dp_v = recursive_policy_value_from_state(q, task, pi)
                err = abs(enum_v - dp_v)
                max_value_err = max(max_value_err, err)
                value_rows.append(
                    (q.name, task, pi, f"{enum_v:.6f}", f"{dp_v:.6f}", f"{err:.2e}")
                )
    reporter.lines.extend(
        ascii_table(
            ["q", "task", "policy", "enumerated J", "recursive J", "abs err"],
            value_rows,
        )
    )
    checks.append(
        CheckResult(
            "Task-policy values are actually computed through the recursive state",
            max_value_err <= 1e-12,
            f"max_abs_error={max_value_err:.3e}",
        )
    )

    reporter.subsection("Counterfactual if recursion is deleted")
    frozen_rows = []
    for q in BENCHMARK_Q_SPECS.values():
        best_true = max(
            exact_task_policy_value(q, "copy_hidden_bit", pi) for pi in POLICIES
        )
        frozen = frozen_state_counterfactual_value(q, "copy_hidden_bit")
        frozen_rows.append((q.name, f"{best_true:.4f}", f"{frozen:.4f}", f"{best_true - frozen:.4f}"))
    reporter.lines.extend(ascii_table(["q", "best recursive", "frozen p=.5", "lost value"], frozen_rows))

    # -------------------------------------------------------------------------
    # Full models, nuisance, quotient compilation
    # -------------------------------------------------------------------------
    reporter.section("3. FULL MODEL CLASS -> TASK-RELATIVE QUOTIENT")
    full_models = build_main_full_models(args.nuisance_sizes)
    reporter.kv("number of full models", len(full_models))
    reporter.kv("relevant quotient types", sorted(BENCHMARK_Q_SPECS))
    reporter.line("Full model = (relevant q, independent nuisance alphabet size K).")
    reporter.line("Nuisance is present in the raw model but absent from declared tasks and experiment summary.")

    (
        quotients,
        adequate_by_q,
        classes,
        selector,
        harmful,
        readiness,
        rep,
        refinements,
    ) = evaluate_problem(full_models, controllers, args.epsilon)

    reporter.kv("compiled task-relative quotient count", len(quotients))
    reporter.kv("base quotient groups requiring experiment-interface refinement", refinements)
    q_rows = []
    for q in quotients:
        q_rows.append(
            (
                q.name,
                len(q.full_models),
                sorted(m.nuisance_size for m in q.full_models),
                q.experiment_signature,
            )
        )
    reporter.lines.extend(ascii_table(["quotient", "full models", "nuisance K", "experiment signature"], q_rows))

    expected_q_count = len(BENCHMARK_Q_SPECS)
    # Raw full-model enumeration must reproduce both task values and experiment laws.
    max_raw_task_err = 0.0
    max_raw_exp_err = 0.0
    for m in full_models:
        for task in TASKS:
            for pi in POLICIES:
                max_raw_task_err = max(
                    max_raw_task_err,
                    abs(exact_full_model_task_policy_value(m, task, pi) - exact_task_policy_value(m.q, task, pi)),
                )
        for e in EXPERIMENTS:
            max_raw_exp_err = max(
                max_raw_exp_err,
                abs(enumerate_raw_experiment_success_prob(m, e) - m.q.experiment_success_prob(e)),
            )
    reporter.kv("max raw-vs-quotient task-value error", f"{max_raw_task_err:.3e}")
    reporter.kv("max raw-vs-quotient experiment-law error", f"{max_raw_exp_err:.3e}")

    quotient_pass = (
        len(quotients) == expected_q_count
        and all(experiment_interface_is_well_defined(q) for q in quotients)
        and max_raw_task_err <= 1e-12
        and max_raw_exp_err <= 1e-12
    )
    checks.append(
        CheckResult(
            "Task-relative quotient removes nuisance while preserving experiment factorization",
            quotient_pass,
            f"full_models={len(full_models)}, quotients={len(quotients)}, refinements={refinements}",
        )
    )

    # -------------------------------------------------------------------------
    # Adequacy sets and fibre soundness
    # -------------------------------------------------------------------------
    reporter.section("4. EXACT ADEQUATE-CONTROLLER CORRESPONDENCE (K2 -> K3)")
    for q in quotients:
        reporter.subsection(f"Quotient {q.name}")
        repq = q.representative_q
        reporter.kv("theta_operational", repq.theta_operational)
        for task in TASKS:
            for pi in POLICIES:
                reporter.line(
                    f"J({task}, {pi}) = {exact_task_policy_value(repq, task, pi):.12g}"
                )
        reporter.kv("G_epsilon(q)", sorted(adequate_by_q[q.name]))
        reporter.kv(
            "uniform fibre certificate intersection",
            sorted(uniform_fibre_adequate_intersection(q, controllers, args.epsilon)),
        )
        regrets = controller_regret_table(repq, controllers)
        for g in controllers:
            reporter.line(
                f"  {g.name} regrets: "
                + ", ".join(f"{t}={regrets[g.name][t]:.6f}" for t in TASKS)
            )

    expected_main_sets = {
        "H": {"g_plus"},
        "L": {"g_minus"},
        "M": {g.name for g in controllers},
    }
    adequacy_pass = all(adequate_by_q.get(k) == v for k, v in expected_main_sets.items())
    checks.append(
        CheckResult(
            "Benchmark adequate sets match the intended overlapping answer geometry",
            adequacy_pass,
            "; ".join(f"{k}={sorted(adequate_by_q.get(k, set()))}" for k in sorted(expected_main_sets)),
        )
    )

    fibre_sound = True
    fibre_details = []
    for q in quotients:
        certified = adequate_by_q[q.name]
        for m in q.full_models:
            gm = adequate_controller_names(m.q, controllers, args.epsilon)
            ok = certified <= gm
            fibre_sound = fibre_sound and ok
            if not ok:
                fibre_details.append(f"{q.name}->{m.name} violates subset")
    checks.append(
        CheckResult(
            "Exact quotient certificate is uniformly sound over every full-model fibre",
            fibre_sound,
            "all quotient certificates are subsets of every member's G_epsilon" if fibre_sound else "; ".join(fibre_details),
        )
    )

    # -------------------------------------------------------------------------
    # Experiment interface and identifiability collapse
    # -------------------------------------------------------------------------
    reporter.section("5. OBSERVABLE EXPERIMENT INTERFACE AND IDENTIFIABILITY PREFLIGHT")
    for q in quotients:
        reporter.line(
            f"{q.name}: " + ", ".join(f"P(success|{e})={q.experiment_signature[i]:.6f}" for i, e in enumerate(EXPERIMENTS))
        )
    reporter.kv("experiment-equivalence class count K", len(classes))
    class_rows = []
    for c in classes:
        class_rows.append((c.name, [q.name for q in c.quotients], c.law, sorted(c.adequate_intersection)))
    reporter.lines.extend(ascii_table(["class", "quotients", "law", "common adequate controllers"], class_rows))
    learnable, bad_classes = identifiability_preflight(classes)
    reporter.kv("preflight learnable", learnable)
    reporter.kv("bad classes", bad_classes)
    reporter.kv("globally safe controllers", sorted(global_safe_controllers(classes)))
    checks.append(CheckResult("Main benchmark passes exact decision-identifiability preflight", learnable, f"bad_classes={bad_classes}"))

    # -------------------------------------------------------------------------
    # Safe selector and harmful confusions
    # -------------------------------------------------------------------------
    reporter.section("6. SAFE SELECTOR AND CONTROLLER-INVALIDATING CONFUSIONS (K4)")
    for cname in sorted(selector):
        reporter.line(f"a({cname}) = {selector[cname]}")
    reporter.line()
    reporter.line("Harmful ordered confusions C_a:")
    for p in harmful:
        reporter.line(f"  {p[0]} <- estimated as {p[1]}")
    all_pairs = all_distinct_pairs(classes)
    reporter.kv("harmful ordered pair count", len(harmful))
    reporter.kv("all ordered distinct pair count", len(all_pairs))
    reporter.kv("constraints removed by TRPC readiness", len(all_pairs) - len(harmful))

    # -------------------------------------------------------------------------
    # Hellinger geometry and LP
    # -------------------------------------------------------------------------
    reporter.section("7. HELLINGER GEOMETRY AND MAX-MIN FIXED ALLOCATION")
    h_rows = []
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i < j:
                hv = pair_hellinger_vector(c1, c2)
                h_rows.append((c1.name, c2.name, *(f"{hv[e]:.12g}" for e in EXPERIMENTS)))
    reporter.lines.extend(ascii_table(["class 1", "class 2", *EXPERIMENTS], h_rows))

    def report_alloc(a: AllocationResult) -> None:
        reporter.subsection(a.name)
        reporter.kv("LP status", a.lp_status)
        reporter.kv("kappa", f"{a.kappa:.12g}")
        for e in EXPERIMENTS:
            reporter.kv(f"weight[{e}]", f"{a.weights[e]:.12g}")
        for pair, rate in sorted(a.per_pair_rates.items()):
            reporter.line(f"  pair_rate{pair} = {rate:.12g}")
        if a.coarse_grid_kappa is not None:
            reporter.kv("independent dense-grid kappa", f"{a.coarse_grid_kappa:.12g}")
            reporter.kv("dense-grid weight probe_A", f"{a.coarse_grid_weight_probe_A:.12g}")
            reporter.kv("LP-vs-grid |kappa|", f"{abs(a.kappa - a.coarse_grid_kappa):.3e}")

    report_alloc(readiness)
    report_alloc(rep)
    lp_crosscheck = (
        readiness.coarse_grid_kappa is not None
        and abs(readiness.kappa - readiness.coarse_grid_kappa) <= 2e-5
        and rep.coarse_grid_kappa is not None
        and abs(rep.kappa - rep.coarse_grid_kappa) <= 2e-5
    )
    checks.append(CheckResult("Hellinger LP agrees with an independent dense-grid optimization", lp_crosscheck, f"TRPC diff={abs(readiness.kappa - float(readiness.coarse_grid_kappa)):.3e}; rep diff={abs(rep.kappa - float(rep.coarse_grid_kappa)):.3e}"))
    checks.append(CheckResult("Readiness rate is at least the all-pairs representation-ID rate", readiness.kappa + 1e-12 >= rep.kappa, f"kappa_TRPC={readiness.kappa:.12g}; kappa_rep={rep.kappa:.12g}"))

    # -------------------------------------------------------------------------
    # Theorem budgets
    # -------------------------------------------------------------------------
    reporter.section("8. FIXED-CONFIDENCE THEOREM BUDGETS")
    K = len(classes)
    n_trpc = theorem_scalar_budget(readiness.kappa, K, args.delta)
    n_rep = theorem_scalar_budget(rep.kappa, K, args.delta)
    alloc_trpc = integer_episode_allocation(n_trpc, readiness.weights)
    alloc_rep = integer_episode_allocation(n_rep, rep.weights)
    bound_trpc = (K - 1) * math.exp(-n_trpc * readiness.kappa) if K > 1 else 0.0
    bound_rep = (K - 1) * math.exp(-n_rep * rep.kappa) if K > 1 else 0.0
    reporter.kv("TRPC scalar n", n_trpc)
    reporter.kv("TRPC integer episode allocation", alloc_trpc)
    reporter.kv("TRPC total episodes N", sum(alloc_trpc.values()))
    reporter.kv("TRPC theorem bound", f"{bound_trpc:.12g}")
    reporter.kv("all-pairs-ID scalar n", n_rep)
    reporter.kv("all-pairs-ID integer episode allocation", alloc_rep)
    reporter.kv("all-pairs-ID total episodes N", sum(alloc_rep.values()))
    reporter.kv("all-pairs-ID theorem bound", f"{bound_rep:.12g}")
    reporter.kv("scalar budget ratio n_rep / n_TRPC", f"{(n_rep / n_trpc if n_trpc else math.inf):.6f}")
    reporter.kv("episode ratio N_rep / N_TRPC", f"{(sum(alloc_rep.values()) / sum(alloc_trpc.values()) if sum(alloc_trpc.values()) else math.inf):.6f}")
    checks.append(CheckResult("TRPC theorem budget satisfies requested delta", bound_trpc <= args.delta + 1e-15, f"bound={bound_trpc:.12g}, delta={args.delta}"))

    # -------------------------------------------------------------------------
    # Exact finite error enumeration at theorem budget
    # -------------------------------------------------------------------------
    reporter.section("9. EXACT FINITE MLE ERROR ENUMERATION AT TRPC THEOREM BUDGET")
    trpc_exact: Dict[str, ExactErrorResult] = {}
    exact_rows = []
    for c in classes:
        er = exact_error_enumeration(c, classes, selector, alloc_trpc)
        trpc_exact[c.name] = er
        exact_rows.append(
            (
                c.name,
                f"{er.controller_error:.12g}",
                f"{er.class_id_error:.12g}",
                f"{er.harmful_wrong_class_mass:.12g}",
                f"{er.harmless_wrong_class_mass:.12g}",
                f"{er.tie_mass:.3e}",
            )
        )
    reporter.lines.extend(ascii_table(["true class", "controller error", "class-ID error", "harmful wrong mass", "harmless wrong mass", "tie mass"], exact_rows))
    max_exact_controller_error = max(er.controller_error for er in trpc_exact.values())
    max_exact_class_error = max(er.class_id_error for er in trpc_exact.values())
    reporter.kv("max exact controller error", f"{max_exact_controller_error:.12g}")
    reporter.kv("max exact class-ID error", f"{max_exact_class_error:.12g}")
    reporter.kv("theorem delta target", args.delta)
    reporter.kv("theorem union/Hellinger bound", f"{bound_trpc:.12g}")
    checks.append(CheckResult("Exact enumerated controller error is <= theorem bound", max_exact_controller_error <= bound_trpc + 1e-10, f"exact={max_exact_controller_error:.12g}, bound={bound_trpc:.12g}"))
    checks.append(CheckResult("Exact enumerated controller error is <= requested delta", max_exact_controller_error <= args.delta + 1e-10, f"exact={max_exact_controller_error:.12g}, delta={args.delta}"))

    reporter.subsection("Output-controller distributions")
    for cname, er in sorted(trpc_exact.items()):
        reporter.line(f"{cname}: {json.dumps(er.output_controller_mass, sort_keys=True)}")

    # -------------------------------------------------------------------------
    # Exact error curve
    # -------------------------------------------------------------------------
    reporter.section("10. EXACT CONTROLLER-ERROR CURVE AROUND THE THEOREM BUDGET")
    curve_rows = []
    for mult in args.curve_multipliers:
        n = max(1, int(math.ceil(n_trpc * mult)))
        n_by_e = integer_episode_allocation(n, readiness.weights)
        exacts = [exact_error_enumeration(c, classes, selector, n_by_e) for c in classes]
        max_ctrl = max(e.controller_error for e in exacts)
        max_id = max(e.class_id_error for e in exacts)
        theorem_bound = (K - 1) * math.exp(-n * readiness.kappa)
        curve_rows.append((f"{mult:.2f}", n, sum(n_by_e.values()), f"{max_ctrl:.12g}", f"{max_id:.12g}", f"{theorem_bound:.12g}"))
    reporter.lines.extend(ascii_table(["n/n*", "scalar n", "episodes", "max controller err", "max class-ID err", "theorem bound"], curve_rows))

    # -------------------------------------------------------------------------
    # Representation-ID baseline at its own theorem budget
    # -------------------------------------------------------------------------
    reporter.section("11. ALL-PAIRS REPRESENTATION-IDENTIFICATION BASELINE")
    rep_rows = []
    rep_exact_results: Dict[str, ExactErrorResult] = {}
    for c in classes:
        er = exact_error_enumeration(c, classes, selector, alloc_rep)
        rep_exact_results[c.name] = er
        rep_rows.append((c.name, f"{er.controller_error:.12g}", f"{er.class_id_error:.12g}"))
    reporter.lines.extend(ascii_table(["true class", "controller error", "class-ID error"], rep_rows))
    reporter.line("This baseline uses the SAME MLE and controller map but allocates against every distinct class pair.")
    reporter.line("Its stronger target is included to quantify what TRPC-0 refuses to require.")

    # -------------------------------------------------------------------------
    # Monte Carlo cross-check
    # -------------------------------------------------------------------------
    reporter.section("12. MONTE CARLO CROSS-CHECK OF THE EXACT ENUMERATION")
    rng = np.random.default_rng(args.seed)
    mc_rows = []
    mc_max_abs_diff = 0.0
    if args.mc_trials > 0:
        for c in classes:
            mc_ctrl, mc_id = monte_carlo_errors(rng, c, classes, selector, alloc_trpc, args.mc_trials)
            ex = trpc_exact[c.name]
            diff = abs(mc_ctrl - ex.controller_error)
            mc_max_abs_diff = max(mc_max_abs_diff, diff)
            # 6-sigma-ish tolerance plus a small floor. This is a sanity check, not a theorem.
            sigma = math.sqrt(max(ex.controller_error * (1.0 - ex.controller_error), 1e-12) / args.mc_trials)
            tol = max(8.0 * sigma, 5.0 / args.mc_trials)
            mc_rows.append((c.name, f"{ex.controller_error:.12g}", f"{mc_ctrl:.12g}", f"{diff:.3e}", f"{tol:.3e}", f"{mc_id:.12g}"))
            checks.append(CheckResult(f"Monte Carlo agrees with exact controller error for {c.name}", diff <= tol, f"diff={diff:.3e}, tolerance={tol:.3e}"))
        reporter.lines.extend(ascii_table(["true class", "exact ctrl err", "MC ctrl err", "abs diff", "MC tolerance", "MC class-ID err"], mc_rows))
    else:
        reporter.line("Monte Carlo skipped because --mc-trials=0. Exact enumeration remains authoritative.")

    # -------------------------------------------------------------------------
    # Nuisance invariance scaling
    # -------------------------------------------------------------------------
    reporter.section("13. EXACT NUISANCE-INVARIANCE SCALING TEST")
    scale_sets = [
        (1,),
        (1, 2),
        (1, 2, 4, 8),
        tuple(args.nuisance_sizes),
    ]
    nuisance_rows = []
    base_kappa = None
    base_weights = None
    nuisance_pass = True
    for ks in scale_sets:
        fm = build_main_full_models(ks)
        qs, gq, cs, sel, harm, rd, rp, refc = evaluate_problem(fm, controllers, args.epsilon)
        if base_kappa is None:
            base_kappa = rd.kappa
            base_weights = rd.weights
        same = (
            len(qs) == 3
            and abs(rd.kappa - base_kappa) <= 1e-12
            and all(abs(rd.weights[e] - base_weights[e]) <= 1e-12 for e in EXPERIMENTS)
            and refc == 0
        )
        nuisance_pass = nuisance_pass and same
        nuisance_rows.append((list(ks), len(fm), len(qs), f"{rd.kappa:.12g}", json.dumps(rd.weights, sort_keys=True), same))
    reporter.lines.extend(ascii_table(["nuisance sizes", "full models", "quotients", "kappa_TRPC", "weights", "invariant"], nuisance_rows))
    checks.append(CheckResult("TRPC-0 quotient and acquisition are exactly invariant to nuisance-class scaling", nuisance_pass, f"tested {len(scale_sets)} nuisance model-class scales"))

    # -------------------------------------------------------------------------
    # Adversarial tests
    # -------------------------------------------------------------------------
    reporter.section("14. DELIBERATE FAILURE-MODE / DELETION TESTS")
    checks.extend(adversarial_tests(controllers, reporter, classes, selector, readiness, rep, trpc_exact))

    # -------------------------------------------------------------------------
    # Final pass/fail summary
    # -------------------------------------------------------------------------
    reporter.section("15. FINAL VALIDATION SUMMARY")
    mandatory = [c for c in checks if c.mandatory]
    passed = [c for c in mandatory if c.passed]
    failed = [c for c in mandatory if not c.passed]
    for c in checks:
        reporter.line(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}")
        reporter.line(f"       {c.detail}")
    reporter.line()
    reporter.kv("mandatory checks", len(mandatory))
    reporter.kv("passed", len(passed))
    reporter.kv("failed", len(failed))
    reporter.kv("elapsed_seconds", f"{time.perf_counter() - t0:.3f}")

    reporter.section("16. INTERPRETATION GUARDRAILS")
    reporter.line("If every mandatory check passes, the result supports only this claim:")
    reporter.line("  The exact finite TRPC-0 reference mechanism is internally executable on this")
    reporter.line("  benchmark and exhibits its intended recursion, readiness, invalidator, and")
    reporter.line("  nuisance-invariance behavior under the encoded assumptions.")
    reporter.line()
    reporter.line("A pass does NOT establish:")
    reporter.line("  - publication novelty or superiority over established decision-learning methods;")
    reporter.line("  - learned quotient/encoder discovery;")
    reporter.line("  - continuous-action or nonlinear-state performance;")
    reporter.line("  - robustness to misspecification;")
    reporter.line("  - irreversible/single-life exploration;")
    reporter.line("  - sample efficiency outside the exact finite resettable interface.")
    reporter.line()
    if failed:
        reporter.line("FINAL VERDICT: FAIL")
        reporter.line("At least one mandatory TRPC-0 validation failed. Do not scale the algorithm yet.")
        return 2
    else:
        reporter.line("FINAL VERDICT: PASS")
        reporter.line("The exact finite TRPC-0 reference implementation survived this validation suite.")
        reporter.line("The next scientific question is not more finite machinery; it is which single")
        reporter.line("assumption/axis to relax while preserving the operative K1-K4 chain.")
        return 0


# =============================================================================
# 12. CLI and robust report-on-exception behavior
# =============================================================================

def parse_float_tuple(text: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated float list")
    return vals


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals or any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exhaustive one-file validation of exact finite TRPC-0."
    )
    p.add_argument(
        "--report",
        type=str,
        default="trpc0_validation_report.txt",
        help="Output TXT report path (default: trpc0_validation_report.txt)",
    )
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--mc-trials",
        type=int,
        default=DEFAULT_MC_TRIALS,
        help="Monte Carlo trials per true class; set 0 to skip (exact enumeration still runs).",
    )
    p.add_argument(
        "--max-recursion-depth",
        type=int,
        default=DEFAULT_MAX_RECURSION_DEPTH,
        help="Exhaustive sensor-history depth for recursive state validation.",
    )
    p.add_argument(
        "--nuisance-sizes",
        type=parse_int_tuple,
        default=DEFAULT_NUISANCE_SIZES,
        help="Comma-separated nuisance alphabet sizes.",
    )
    p.add_argument(
        "--curve-multipliers",
        type=parse_float_tuple,
        default=DEFAULT_CURVE_MULTIPLIERS,
        help="Comma-separated multiples of the TRPC theorem scalar budget for exact error curves.",
    )
    return p


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")
    if not (0.0 < args.delta < 1.0):
        raise ValueError("delta must lie in (0,1)")
    if args.mc_trials < 0:
        raise ValueError("mc-trials must be >= 0")
    if args.max_recursion_depth < 0 or args.max_recursion_depth > 8:
        raise ValueError("max-recursion-depth must be between 0 and 8 for exhaustive enumeration")
    if any(x <= 0.0 for x in args.curve_multipliers):
        raise ValueError("curve multipliers must be positive")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()
    reporter = Reporter(report_path)

    try:
        validate_cli_args(args)
        code = run_validation(args, reporter)
        reporter.flush()
        print(f"TRPC-0 validation finished. Report: {report_path}")
        print(f"Exit code: {code}")
        return code
    except Exception as exc:
        reporter.section("FATAL EXCEPTION")
        reporter.kv("exception_type", type(exc).__name__)
        reporter.kv("exception", str(exc))
        reporter.line()
        reporter.line(traceback.format_exc())
        reporter.line("FINAL VERDICT: ERROR")
        reporter.flush()
        print(f"TRPC-0 validation crashed. Report: {report_path}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
