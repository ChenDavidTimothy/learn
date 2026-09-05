#!/usr/bin/env python3
"""
Capability-Track Stage-1 Falsification Harness
==============================================

Purpose
-------
This is the frozen pre-neural, pre-Stage-2 experimental harness for the
agent-learning research program developed in the preceding mathematical work.
It is deliberately designed to test a narrow claim before any larger claim is
allowed:

    For a declared family of future decisions, direct learning of only the
    residual/decision-relevant causal distinctions can require fewer
    interactions than full environment identification or task-by-task
    relearning, while retaining post-hoc competence on every declared task.

The script does NOT attempt to prove that "RL is obsolete". It tries to falsify
or support the mathematically derived Stage-1 predictions under environments
whose assumptions are explicit and checkable.

What this one file runs
-----------------------
1. Finite Stage-1A theorem-matching controller-identification experiments.
   - Hellinger fixed-allocation Capability-Track.
   - Canonical, minimum-cell, and max-separation safe assignments.
   - Small product nuisance problem.
   - Full-identification comparison.
   - Optional adaptive controller-invalidating allocator (Stage 1B heuristic).

2. NoisyRelevantBits scaling benchmark.
   - Exact decision-relevant learning by sequential likelihood-ratio tests.
   - Full environment identification.
   - Fixed-sample Hoeffding theorem reproduction.
   - Nuisance scaling, relevant-dimension scaling, and confidence scaling.

3. InformationGap benchmark.
   - Holds representation dimension fixed while shrinking the statistical gap.
   - Tests the predicted ~1/eta^2 sample-complexity law.

4. TaskFamilyGrowth benchmark.
   - Redundant tasks versus genuinely new decision-relevant directions.
   - Tests transfer/reuse rather than merely counting task labels.

5. Product predictive-rank benchmark.
   - Numerically verifies full product rank versus task-relative rank.
   - This is a structural diagnostic, not a learning claim by itself.

6. RewardSparsity / CombinationLock benchmark.
   - Capability/transition-directed learner.
   - A structure-aware reward-free model-based learner.
   - Tabular Q-learning.
   - UCBVI-like optimistic model-based RL.
   - This benchmark is explicitly diagnostic: sparse-reward wins against weak
     RL do not establish the core hypothesis if strong model-based/reward-free
     methods match the structural learner.

Output
------
Exactly one primary text report is written. It contains:
- machine and dependency metadata;
- script SHA-256;
- exact command/configuration;
- theorem/assumption reminders;
- aggregate tables with confidence intervals;
- fitted scaling exponents;
- kill-criterion checks;
- an automatic evidence-level summary;
- all raw trial records as JSON Lines at the end.

Recommended command
-------------------
    python capability_track_falsification.py --profile deep

For a first smoke run:
    python capability_track_falsification.py --profile quick

Dependencies
------------
Required:
    numpy
    scipy
Optional:
    psutil   (better process-memory reporting)

Install, for example:
    pip install numpy scipy psutil

Scientific discipline
---------------------
- The task family is declared before exploration.
- Individual task identity may be revealed after exploration.
- Stage 1 is learn-then-deploy, fixed-confidence readiness, not low-regret
  online control.
- Strong structure-aware baselines are intentionally included.
- Failure on theorem-satisfying benchmarks is a reason to revise the idea,
  not to change the metric after observing the result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator
from scipy.optimize import linprog

try:
    import scipy
except Exception:  # pragma: no cover
    scipy = None

try:
    import psutil  # type: ignore
except Exception:  # optional
    psutil = None


SCRIPT_VERSION = "2026-09-05-precode-stage1-v1"
EPS = 1e-15


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "finite_trials": 80,
        "adaptive_trials": 30,
        "bit_trials": 40,
        "task_trials": 30,
        "rl_trials": 6,
        "noisy_k": [0, 8, 32],
        "noisy_d": [4, 8],
        "noisy_p": [0.15, 0.30],
        "info_eta": [0.30, 0.20, 0.12, 0.08],
        "task_counts": [1, 2, 4, 8],
        "product_n": [1, 2, 4, 8],
        "rank_n": [1, 2, 3, 4],
        "lock_h": [4, 6],
        "reward_prob": [1.0, 0.20, 0.05],
        "rl_max_episodes": 5_000,
        "raw_records": True,
    },
    "standard": {
        "finite_trials": 300,
        "adaptive_trials": 100,
        "bit_trials": 120,
        "task_trials": 100,
        "rl_trials": 12,
        "noisy_k": [0, 8, 32, 128, 512],
        "noisy_d": [4, 8, 16],
        "noisy_p": [0.10, 0.20, 0.30, 0.40],
        "info_eta": [0.30, 0.20, 0.15, 0.10, 0.07, 0.05],
        "task_counts": [1, 2, 4, 8, 16],
        "product_n": [1, 2, 4, 8, 16],
        "rank_n": [1, 2, 3, 4, 6, 8],
        "lock_h": [4, 6, 8],
        "reward_prob": [1.0, 0.30, 0.10, 0.03],
        "rl_max_episodes": 20_000,
        "raw_records": True,
    },
    "deep": {
        "finite_trials": 1_000,
        "adaptive_trials": 300,
        "bit_trials": 400,
        "task_trials": 300,
        "rl_trials": 24,
        "noisy_k": [0, 4, 16, 64, 256, 1_024, 4_096],
        "noisy_d": [2, 4, 8, 16, 32],
        "noisy_p": [0.05, 0.10, 0.20, 0.30, 0.40],
        "info_eta": [0.35, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.035],
        "task_counts": [1, 2, 4, 8, 16, 32],
        "product_n": [1, 2, 4, 8, 16, 32],
        "rank_n": [1, 2, 3, 4, 6, 8, 12],
        "lock_h": [4, 6, 8, 10],
        "reward_prob": [1.0, 0.50, 0.20, 0.10, 0.05, 0.02],
        "rl_max_episodes": 50_000,
        "raw_records": True,
    },
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def rng_for(base_seed: int, *parts: Any) -> Generator:
    return np.random.default_rng(stable_seed(base_seed, *parts))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_log(x: float) -> float:
    return math.log(max(float(x), EPS))


def kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    q = min(max(q, EPS), 1.0 - EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def binary_kl(p: float, q: float) -> float:
    return kl_bernoulli(p, q)


def categorical_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if np.any((q <= 0) & mask):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def hellinger_sq(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(max(0.0, 1.0 - np.sum(np.sqrt(np.maximum(p, 0.0) * np.maximum(q, 0.0)))))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) / n) + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mean_ci(values: Sequence[float], z: float = 1.959963984540054) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(arr))
    if arr.size == 1:
        return m, m, m
    se = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return m, m - z * se, m + z * se


def quantiles(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(arr, [0.25, 0.50, 0.75]))  # type: ignore


def loglog_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(xa.size)}
    lx = np.log(xa)
    ly = np.log(ya)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(xa.size)}


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(xa.size)}
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = slope * xa + intercept
    ss_res = float(np.sum((ya - pred) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(xa.size)}


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if math.isnan(xf):
            return "nan"
        if math.isinf(xf):
            return "inf" if xf > 0 else "-inf"
        if xf == 0:
            return "0"
        if abs(xf) >= 1e5 or abs(xf) < 1e-4:
            return f"{xf:.{digits}e}"
        return f"{xf:.{digits}f}"
    return str(x)


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_width: int = 32) -> str:
    data = [[str(h) for h in headers]] + [[fmt(v) for v in row] for row in rows]
    widths: List[int] = []
    for j in range(len(headers)):
        widths.append(min(max(len(r[j]) for r in data), max_width))

    def crop(s: str, w: int) -> str:
        return s if len(s) <= w else s[: max(0, w - 3)] + "..."

    out: List[str] = []
    out.append(" | ".join(crop(data[0][j], widths[j]).ljust(widths[j]) for j in range(len(headers))))
    out.append("-+-".join("-" * widths[j] for j in range(len(headers))))
    for r in data[1:]:
        out.append(" | ".join(crop(r[j], widths[j]).ljust(widths[j]) for j in range(len(headers))))
    return "\n".join(out)


def current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


class MemoryMonitor:
    """Optional low-overhead RSS peak sampler.

    psutil is optional. If absent, the monitor reports None and the rest of the
    experiment remains valid.
    """
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peak_mb: Optional[float] = current_rss_mb()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if psutil is None:
            return
        def loop() -> None:
            while not self._stop.is_set():
                val = current_rss_mb()
                if val is not None:
                    self.peak_mb = val if self.peak_mb is None else max(self.peak_mb, val)
                self._stop.wait(self.interval_s)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.interval_s))
        val = current_rss_mb()
        if val is not None:
            self.peak_mb = val if self.peak_mb is None else max(self.peak_mb, val)


@dataclass
class TrialRecord:
    suite: str
    method: str
    config: Dict[str, Any]
    seed: int
    interactions: int
    success: bool
    runtime_s: float
    regret: float = 0.0
    censored: bool = False
    model_success: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


class ResultStore:
    def __init__(self) -> None:
        self.records: List[TrialRecord] = []
        self.diagnostics: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def add(self, rec: TrialRecord) -> None:
        self.records.append(rec)

    def diag(self, suite: str, name: str, **values: Any) -> None:
        self.diagnostics.append({"suite": suite, "name": name, **values})

    def error(self, text: str) -> None:
        self.errors.append(text)


class ReportBuilder:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def title(self, text: str) -> None:
        self.lines.extend(["=" * 88, text, "=" * 88, ""])

    def section(self, text: str) -> None:
        self.lines.extend(["", "# " + text, "-"])

    def subsection(self, text: str) -> None:
        self.lines.extend(["", "## " + text])

    def text(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.lines.append(ascii_table(headers, rows))

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------
# Finite controller-identification problem
# ---------------------------------------------------------------------------

@dataclass
class FiniteControllerProblem:
    name: str
    theta_names: List[str]
    experiment_names: List[str]
    controller_names: List[str]
    # laws[theta][experiment] -> 1D categorical probabilities; all experiments
    # may have different outcome counts, therefore nested arrays are used.
    laws: List[List[np.ndarray]]
    adequate: np.ndarray  # shape [n_theta, n_controllers], bool
    theta_meta: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_theta(self) -> int:
        return len(self.theta_names)

    @property
    def n_exp(self) -> int:
        return len(self.experiment_names)

    @property
    def n_ctrl(self) -> int:
        return len(self.controller_names)

    def validate(self) -> None:
        assert self.adequate.shape == (self.n_theta, self.n_ctrl)
        assert len(self.laws) == self.n_theta
        for t in range(self.n_theta):
            assert len(self.laws[t]) == self.n_exp
            assert np.any(self.adequate[t]), f"theta {t} has no adequate controller"
            for e in range(self.n_exp):
                p = np.asarray(self.laws[t][e], dtype=float)
                assert p.ndim == 1 and p.size >= 2
                assert np.all(p >= 0)
                assert abs(float(np.sum(p)) - 1.0) < 1e-9


def make_overlap_problem() -> FiniteControllerProblem:
    theta_names = ["theta0", "theta1", "theta2"]
    experiment_names = ["probe_A", "probe_B"]
    controller_names = ["A", "B", "C"]
    bern = lambda p: np.array([1.0 - p, p], dtype=float)
    laws = [
        [bern(0.18), bern(0.50)],
        [bern(0.82), bern(0.18)],
        [bern(0.50), bern(0.82)],
    ]
    # Pairwise compatible but no controller works for all three.
    adequate = np.array(
        [
            [True, True, False],   # theta0: A or B
            [False, True, True],   # theta1: B or C
            [True, False, True],   # theta2: A or C
        ],
        dtype=bool,
    )
    p = FiniteControllerProblem(
        name="OverlappingControllerID",
        theta_names=theta_names,
        experiment_names=experiment_names,
        controller_names=controller_names,
        laws=laws,
        adequate=adequate,
        theta_meta=[{"z": i} for i in range(3)],
    )
    p.validate()
    return p


def make_small_product_problem(d: int, nuisance_models: int, flip_p: float) -> FiniteControllerProblem:
    assert 1 <= d <= 4
    assert nuisance_models >= 1
    assert 0 < flip_p < 0.5
    z_values = list(itertools.product([0, 1], repeat=d))
    theta_pairs = [(z, n) for z in z_values for n in range(nuisance_models)]
    controllers = ["z=" + "".join(map(str, z)) for z in z_values]
    exp_names = [f"relevant_bit_{j}" for j in range(d)] + ["nuisance_probe"]
    bern = lambda p: np.array([1.0 - p, p], dtype=float)
    laws: List[List[np.ndarray]] = []
    adequate = np.zeros((len(theta_pairs), len(controllers)), dtype=bool)
    meta: List[Dict[str, Any]] = []
    for ti, (z, n) in enumerate(theta_pairs):
        row: List[np.ndarray] = []
        for bit in z:
            prob1 = (1.0 - flip_p) if bit == 1 else flip_p
            row.append(bern(prob1))
        if nuisance_models == 1:
            row.append(np.array([0.5, 0.5], dtype=float))
        else:
            out = np.full(nuisance_models, flip_p / (nuisance_models - 1), dtype=float)
            out[n] = 1.0 - flip_p
            row.append(out)
        laws.append(row)
        ci = z_values.index(z)
        adequate[ti, ci] = True
        meta.append({"z": tuple(z), "n": n})
    problem = FiniteControllerProblem(
        name=f"SmallProduct_d{d}_N{nuisance_models}",
        theta_names=[f"z={''.join(map(str,z))};n={n}" for z, n in theta_pairs],
        experiment_names=exp_names,
        controller_names=controllers,
        laws=laws,
        adequate=adequate,
        theta_meta=meta,
    )
    problem.validate()
    return problem


def canonical_assignment(problem: FiniteControllerProblem) -> np.ndarray:
    out = np.empty(problem.n_theta, dtype=int)
    for t in range(problem.n_theta):
        out[t] = int(np.flatnonzero(problem.adequate[t])[0])
    return out


def minimum_cell_assignment(problem: FiniteControllerProblem) -> np.ndarray:
    ctrls = range(problem.n_ctrl)
    for k in range(1, problem.n_ctrl + 1):
        for subset in itertools.combinations(ctrls, k):
            covered = np.any(problem.adequate[:, subset], axis=1)
            if bool(np.all(covered)):
                assignment = np.empty(problem.n_theta, dtype=int)
                for t in range(problem.n_theta):
                    candidates = [c for c in subset if problem.adequate[t, c]]
                    assignment[t] = int(candidates[0])
                return assignment
    raise RuntimeError("no safe assignment exists")


def pairwise_hellinger(problem: FiniteControllerProblem) -> np.ndarray:
    out = np.zeros((problem.n_theta, problem.n_theta, problem.n_exp), dtype=float)
    for i in range(problem.n_theta):
        for j in range(problem.n_theta):
            for e in range(problem.n_exp):
                out[i, j, e] = hellinger_sq(problem.laws[i][e], problem.laws[j][e])
    return out


def pairwise_kl(problem: FiniteControllerProblem) -> np.ndarray:
    out = np.zeros((problem.n_theta, problem.n_theta, problem.n_exp), dtype=float)
    for i in range(problem.n_theta):
        for j in range(problem.n_theta):
            for e in range(problem.n_exp):
                out[i, j, e] = categorical_kl(problem.laws[i][e], problem.laws[j][e])
    return out


def solve_separation_lp(dist: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
    """Maximize kappa subject to kappa <= sum_e q_e dist[pair,e]."""
    n_exp = dist.shape[2]
    if not pairs:
        return np.full(n_exp, 1.0 / n_exp), float("inf")

    # Variables [q_0,...,q_E-1,kappa]. linprog minimizes, so minimize -kappa.
    c = np.zeros(n_exp + 1, dtype=float)
    c[-1] = -1.0
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    for i, j in pairs:
        row = np.zeros(n_exp + 1, dtype=float)
        row[:n_exp] = -dist[i, j, :]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = np.zeros((1, n_exp + 1), dtype=float)
    A_eq[0, :n_exp] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n_exp + [(0.0, None)]
    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"separation LP failed: {res.message}")
    q = np.maximum(res.x[:n_exp], 0.0)
    q /= np.sum(q)
    return q, float(res.x[-1])


def assignment_pairs(assignment: np.ndarray) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    n = len(assignment)
    for i in range(n):
        for j in range(i + 1, n):
            if int(assignment[i]) != int(assignment[j]):
                pairs.append((i, j))
    return pairs


def solve_assignment_lp(problem: FiniteControllerProblem, assignment: np.ndarray) -> Tuple[np.ndarray, float]:
    h2 = pairwise_hellinger(problem)
    return solve_separation_lp(h2, assignment_pairs(assignment))


def max_separation_assignment(problem: FiniteControllerProblem, max_assignments: int = 200_000) -> Tuple[np.ndarray, np.ndarray, float, int]:
    choices = [list(map(int, np.flatnonzero(problem.adequate[t]))) for t in range(problem.n_theta)]
    total = math.prod(len(c) for c in choices)
    if total > max_assignments:
        a = minimum_cell_assignment(problem)
        q, k = solve_assignment_lp(problem, a)
        return a, q, k, total

    best_a: Optional[np.ndarray] = None
    best_q: Optional[np.ndarray] = None
    best_k = -1.0
    checked = 0
    for tup in itertools.product(*choices):
        checked += 1
        a = np.asarray(tup, dtype=int)
        q, k = solve_assignment_lp(problem, a)
        # Secondary preference: fewer cells.
        if k > best_k + 1e-12 or (
            abs(k - best_k) <= 1e-12
            and best_a is not None
            and len(set(map(int, a))) < len(set(map(int, best_a)))
        ):
            best_a, best_q, best_k = a.copy(), q.copy(), k
    assert best_a is not None and best_q is not None
    return best_a, best_q, best_k, checked


def allocate_integer_counts(total_n: int, q: np.ndarray) -> np.ndarray:
    if total_n <= 0:
        return np.zeros_like(q, dtype=int)
    raw = total_n * q
    base = np.floor(raw).astype(int)
    remainder = total_n - int(np.sum(base))
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return base


def allocate_theorem_counts(base_n: int, q: np.ndarray) -> np.ndarray:
    """Ceiling allocation used for the finite Hellinger theorem.

    The proof requires N_e >= base_n * q_e for every experiment.  A standard
    fixed-total largest-remainder allocation can undershoot some coordinates,
    so the theorem-matching implementation pays at most one rounding sample per
    positive experiment and uses ceil(base_n*q_e).
    """
    if base_n <= 0:
        return np.zeros_like(q, dtype=int)
    return np.ceil(base_n * q - 1e-14).astype(int)


def sample_finite_data(problem: FiniteControllerProblem, true_theta: int, counts: np.ndarray, rng: Generator) -> List[np.ndarray]:
    obs_counts: List[np.ndarray] = []
    for e in range(problem.n_exp):
        n = int(counts[e])
        probs = problem.laws[true_theta][e]
        if n == 0:
            obs_counts.append(np.zeros(len(probs), dtype=int))
        else:
            obs_counts.append(rng.multinomial(n, probs))
    return obs_counts


def finite_loglik(problem: FiniteControllerProblem, obs_counts: List[np.ndarray]) -> np.ndarray:
    ll = np.zeros(problem.n_theta, dtype=float)
    for t in range(problem.n_theta):
        val = 0.0
        for e in range(problem.n_exp):
            counts = obs_counts[e]
            probs = np.maximum(problem.laws[t][e], EPS)
            val += float(np.dot(counts, np.log(probs)))
        ll[t] = val
    return ll


def profiled_controller_decode(problem: FiniteControllerProblem, ll: np.ndarray) -> int:
    scores = np.full(problem.n_ctrl, -np.inf, dtype=float)
    for c in range(problem.n_ctrl):
        idx = np.flatnonzero(problem.adequate[:, c])
        if idx.size:
            scores[c] = float(np.max(ll[idx]))
    return int(np.argmax(scores))


def full_identification_assignment(problem: FiniteControllerProblem) -> np.ndarray:
    # Each model is treated as its own answer. Used only for experiment-design
    # diagnostics, not as a task controller.
    return np.arange(problem.n_theta, dtype=int)


def run_fixed_finite_trial(
    problem: FiniteControllerProblem,
    assignment: np.ndarray,
    q: np.ndarray,
    n: int,
    true_theta: int,
    rng: Generator,
    decoder: str = "assignment_mle",
) -> Tuple[int, bool, bool, np.ndarray, np.ndarray]:
    counts = allocate_theorem_counts(n, q)
    obs_counts = sample_finite_data(problem, true_theta, counts, rng)
    ll = finite_loglik(problem, obs_counts)
    mle = int(np.argmax(ll))
    model_success = mle == true_theta
    if decoder == "profiled_controller":
        ctrl = profiled_controller_decode(problem, ll)
    else:
        ctrl = int(assignment[mle])
    success = bool(problem.adequate[true_theta, ctrl])
    return ctrl, success, model_success, counts, ll


def solve_local_kl_allocation(problem: FiniteControllerProblem, theta_hat: int, invalid: Sequence[int], kl: np.ndarray) -> np.ndarray:
    if not invalid:
        return np.full(problem.n_exp, 1.0 / problem.n_exp)
    pairs = [(theta_hat, int(j)) for j in invalid]
    q, _ = solve_separation_lp(kl, pairs)
    return q


def run_adaptive_finite_trial(
    problem: FiniteControllerProblem,
    assignment: np.ndarray,
    true_theta: int,
    rng: Generator,
    delta: float,
    max_steps: int,
) -> Tuple[int, bool, bool, int, np.ndarray]:
    """Stage-1B heuristic: D-tracking against controller-invalidating alternatives.

    This is intentionally labeled heuristic. The stopping threshold is a
    conservative time-uniform style threshold, not a claimed reproduction of
    every assumption in the asymptotic Track-and-Stop theorem.
    """
    kl = pairwise_kl(problem)
    loglik = np.zeros(problem.n_theta, dtype=float)
    counts = np.zeros(problem.n_exp, dtype=int)
    cache: Dict[Tuple[int, int], np.ndarray] = {}

    # Initial forced exploration: one sample from every experiment.
    t = 0
    for e in range(problem.n_exp):
        outcome = int(rng.choice(len(problem.laws[true_theta][e]), p=problem.laws[true_theta][e]))
        for th in range(problem.n_theta):
            loglik[th] += safe_log(problem.laws[th][e][outcome])
        counts[e] += 1
        t += 1

    while t < max_steps:
        theta_hat = int(np.argmax(loglik))
        ctrl = int(assignment[theta_hat])
        invalid = [j for j in range(problem.n_theta) if not problem.adequate[j, ctrl]]
        if not invalid:
            return ctrl, bool(problem.adequate[true_theta, ctrl]), theta_hat == true_theta, t, counts

        valid_for_ctrl = [j for j in range(problem.n_theta) if problem.adequate[j, ctrl]]
        best_valid = float(np.max(loglik[valid_for_ctrl]))
        best_invalid = float(np.max(loglik[invalid]))
        glr = best_valid - best_invalid
        beta = math.log(
            max(2.0, problem.n_theta * problem.n_ctrl * (t + 1) ** 2 * math.pi**2 / (6.0 * delta))
        )
        if glr >= beta:
            return ctrl, bool(problem.adequate[true_theta, ctrl]), theta_hat == true_theta, t, counts

        key = (theta_hat, ctrl)
        if key not in cache:
            cache[key] = solve_local_kl_allocation(problem, theta_hat, invalid, kl)
        target = cache[key]

        # Forced exploration avoids permanently starving an experiment.
        min_required = int(math.sqrt(t + 1)) // max(1, problem.n_exp)
        under = np.flatnonzero(counts < min_required)
        if under.size:
            e = int(under[0])
        else:
            deficit = (t + 1) * target - counts
            e = int(np.argmax(deficit))

        outcome = int(rng.choice(len(problem.laws[true_theta][e]), p=problem.laws[true_theta][e]))
        for th in range(problem.n_theta):
            loglik[th] += safe_log(problem.laws[th][e][outcome])
        counts[e] += 1
        t += 1

    theta_hat = int(np.argmax(loglik))
    ctrl = int(assignment[theta_hat])
    return ctrl, bool(problem.adequate[true_theta, ctrl]), theta_hat == true_theta, t, counts


# ---------------------------------------------------------------------------
# Noisy bit experiments and task-family growth
# ---------------------------------------------------------------------------


def sprt_single_bit(true_bit: int, flip_p: float, alpha: float, rng: Generator, max_queries: int = 10_000_000) -> Tuple[int, int, float]:
    """Symmetric SPRT between bit=0 and bit=1 under BSC(flip_p).

    Returns (decision, number_of_queries, final_log_likelihood_ratio_for_bit1_vs_bit0).
    The threshold is the standard symmetric Wald threshold. For the current
    binary simple-hypothesis benchmark this is the natural sequential baseline.
    """
    assert true_bit in (0, 1)
    assert 0.0 < flip_p < 0.5
    alpha = min(max(alpha, 1e-12), 0.49)
    step = math.log((1.0 - flip_p) / flip_p)
    threshold = math.log((1.0 - alpha) / alpha)
    llr = 0.0
    for n in range(1, max_queries + 1):
        noise = int(rng.random() < flip_p)
        y = true_bit ^ noise
        llr += step if y == 1 else -step
        if llr >= threshold:
            return 1, n, llr
        if llr <= -threshold:
            return 0, n, llr
    return (1 if llr >= 0 else 0), max_queries, llr


def hoeffding_fixed_n(num_bits: int, flip_p: float, delta: float) -> int:
    if num_bits <= 0:
        return 0
    gap = 0.5 - flip_p
    return int(math.ceil(math.log(num_bits / delta) / (2.0 * gap * gap)))


def fixed_majority_bit(true_bit: int, flip_p: float, n: int, rng: Generator) -> int:
    if n <= 0:
        return int(true_bit)
    flips = int(rng.binomial(n, flip_p))
    if true_bit == 1:
        ones = n - flips
    else:
        ones = flips
    if ones * 2 == n:
        return int(rng.integers(0, 2))
    return 1 if ones * 2 > n else 0


def solve_bits_sprt(bits: Sequence[int], flip_p: float, delta: float, rng: Generator) -> Tuple[List[int], int]:
    n_bits = len(bits)
    if n_bits == 0:
        return [], 0
    alpha = delta / n_bits
    out: List[int] = []
    total = 0
    for b in bits:
        d, n, _ = sprt_single_bit(int(b), flip_p, alpha, rng)
        out.append(d)
        total += n
    return out, total


def solve_bits_fixed(bits: Sequence[int], flip_p: float, delta: float, rng: Generator) -> Tuple[List[int], int, int]:
    n_bits = len(bits)
    if n_bits == 0:
        return [], 0, 0
    m = hoeffding_fixed_n(n_bits, flip_p, delta)
    out = [fixed_majority_bit(int(b), flip_p, m, rng) for b in bits]
    return out, m * n_bits, m


def noisy_bit_theory(d: int, k: int, p: float, delta: float) -> Dict[str, float]:
    gap = 0.5 - p
    cap_upper = d * math.log(d / delta) / (2.0 * gap * gap)
    full_upper = (d + k) * math.log(max(1, d + k) / delta) / (2.0 * gap * gap)
    kappa_p = (1.0 - 2.0 * p) * math.log((1.0 - p) / p)
    lower = d * binary_kl(1.0 - delta, delta) / kappa_p
    full_lower = (d + k) * binary_kl(1.0 - delta, delta) / kappa_p
    return {
        "cap_hoeffding_upper": cap_upper,
        "full_hoeffding_upper": full_upper,
        "cap_change_measure_lower": lower,
        "full_change_measure_lower": full_lower,
        "kappa_p": kappa_p,
    }


def run_noisy_bits_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, delta: float, verbose: bool) -> None:
    suite = "NoisyRelevantBits"
    trials = int(cfg["bit_trials"])
    # Main nuisance sweep at a fixed representative d,p.
    d_main = 8 if 8 in cfg["noisy_d"] else int(cfg["noisy_d"][0])
    p_main = 0.20 if 0.20 in cfg["noisy_p"] else float(cfg["noisy_p"][0])

    configs: List[Tuple[int, int, float, str]] = []
    for k in cfg["noisy_k"]:
        configs.append((d_main, int(k), p_main, "nuisance_sweep"))
    for d in cfg["noisy_d"]:
        configs.append((int(d), 64, p_main, "relevant_sweep"))
    for p in cfg["noisy_p"]:
        configs.append((d_main, 64, float(p), "noise_sweep"))

    seen = set()
    unique_configs = []
    for c in configs:
        key = (c[0], c[1], round(c[2], 12), c[3])
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    for ci, (d, k, p, sweep) in enumerate(unique_configs):
        theory = noisy_bit_theory(d, k, p, delta)
        store.diag(suite, "theory", d=d, K=k, p=p, delta=delta, sweep=sweep, **theory)
        if verbose:
            print(f"[{suite}] {ci+1}/{len(unique_configs)} d={d} K={k} p={p:.3f} {sweep}", flush=True)
        for rep in range(trials):
            rng = rng_for(base_seed, suite, d, k, p, sweep, rep)
            relevant = rng.integers(0, 2, size=d).astype(int).tolist()
            nuisance = rng.integers(0, 2, size=k).astype(int).tolist()

            t0 = time.perf_counter()
            dec, interactions = solve_bits_sprt(relevant, p, delta, rng)
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="Capability_SPRT",
                config={"d": d, "K": k, "p": p, "delta": delta, "sweep": sweep},
                seed=rep,
                interactions=interactions,
                success=dec == relevant,
                model_success=None,
                runtime_s=runtime,
                extra={"objective": "declared_tasks_only"},
            ))

            rng2 = rng_for(base_seed, suite, "full", d, k, p, sweep, rep)
            t0 = time.perf_counter()
            all_dec, full_interactions = solve_bits_sprt(relevant + nuisance, p, delta, rng2)
            runtime = time.perf_counter() - t0
            rel_ok = all_dec[:d] == relevant
            model_ok = all_dec == relevant + nuisance
            store.add(TrialRecord(
                suite=suite,
                method="FullIdentification_SPRT",
                config={"d": d, "K": k, "p": p, "delta": delta, "sweep": sweep},
                seed=rep,
                interactions=full_interactions,
                success=rel_ok,
                model_success=model_ok,
                runtime_s=runtime,
                extra={"objective": "full_environment_identification"},
            ))

            rng3 = rng_for(base_seed, suite, "cap_fixed", d, k, p, sweep, rep)
            t0 = time.perf_counter()
            fixed_dec, fixed_interactions, per_bit = solve_bits_fixed(relevant, p, delta, rng3)
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="Capability_HoeffdingFixed",
                config={"d": d, "K": k, "p": p, "delta": delta, "sweep": sweep},
                seed=rep,
                interactions=fixed_interactions,
                success=fixed_dec == relevant,
                runtime_s=runtime,
                extra={"samples_per_required_bit": per_bit},
            ))

            rng4 = rng_for(base_seed, suite, "full_fixed", d, k, p, sweep, rep)
            t0 = time.perf_counter()
            full_fixed_dec, full_fixed_interactions, per_bit_full = solve_bits_fixed(relevant + nuisance, p, delta, rng4)
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="FullIdentification_HoeffdingFixed",
                config={"d": d, "K": k, "p": p, "delta": delta, "sweep": sweep},
                seed=rep,
                interactions=full_fixed_interactions,
                success=full_fixed_dec[:d] == relevant,
                model_success=full_fixed_dec == relevant + nuisance,
                runtime_s=runtime,
                extra={"samples_per_all_bits": per_bit_full},
            ))


def run_information_gap_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, delta: float, verbose: bool) -> None:
    suite = "InformationGap"
    trials = int(cfg["bit_trials"])
    for ei, eta in enumerate(cfg["info_eta"]):
        eta = float(eta)
        p = 0.5 - eta
        if verbose:
            print(f"[{suite}] {ei+1}/{len(cfg['info_eta'])} eta={eta:.4f}", flush=True)
        for rep in range(trials):
            rng = rng_for(base_seed, suite, eta, rep)
            true_bit = int(rng.integers(0, 2))
            t0 = time.perf_counter()
            decision, n, llr = sprt_single_bit(true_bit, p, delta, rng)
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="Capability_SPRT",
                config={"eta": eta, "p": p, "delta": delta, "predictive_rank": 2},
                seed=rep,
                interactions=n,
                success=decision == true_bit,
                runtime_s=runtime,
                extra={"final_llr": llr, "inv_eta2": 1.0 / (eta * eta)},
            ))
        kl = kl_bernoulli(0.5 - eta, 0.5 + eta)
        store.diag(suite, "gap_theory", eta=eta, p=p, kl=kl, inv_eta2=1.0/(eta*eta))


@dataclass(frozen=True)
class BitTask:
    bits: Tuple[int, ...]
    negate: bool = False

    def answer(self, latent_bits: Sequence[int]) -> int:
        v = 0
        for idx in self.bits:
            v ^= int(latent_bits[idx])
        if self.negate:
            v ^= 1
        return v


def make_task_family(kind: str, count: int, total_d: int) -> List[BitTask]:
    if kind == "redundant":
        # Many task labels, same predictive direction. Negation changes the
        # controller answer but requires no additional environment information.
        return [BitTask((0,), negate=bool(i % 2)) for i in range(count)]
    if kind == "new_directions":
        return [BitTask((i,), False) for i in range(min(count, total_d))]
    if kind == "shared_parity":
        # Increasing labels reuse a fixed small two-bit predictive subspace.
        return [BitTask((0, 1), negate=bool(i % 2)) for i in range(count)]
    raise ValueError(kind)


def required_bits_for_tasks(tasks: Sequence[BitTask]) -> List[int]:
    return sorted(set(itertools.chain.from_iterable(t.bits for t in tasks)))


def run_task_family_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, delta: float, verbose: bool) -> None:
    suite = "TaskFamilyGrowth"
    trials = int(cfg["task_trials"])
    total_d = max(32, max(int(x) for x in cfg["task_counts"]))
    nuisance_k = 128
    p = 0.18
    kinds = ["redundant", "shared_parity", "new_directions"]

    for kind in kinds:
        for count in cfg["task_counts"]:
            count = int(count)
            tasks = make_task_family(kind, count, total_d)
            req = required_bits_for_tasks(tasks)
            if verbose:
                print(f"[{suite}] kind={kind} tasks={count} unique_bits={len(req)}", flush=True)
            for rep in range(trials):
                rng = rng_for(base_seed, suite, kind, count, rep)
                latent = rng.integers(0, 2, size=total_d).astype(int).tolist()
                nuisance = rng.integers(0, 2, size=nuisance_k).astype(int).tolist()
                true_answers = [t.answer(latent) for t in tasks]

                # Capability/reuse: learn every distinct required bit once.
                req_truth = [latent[j] for j in req]
                t0 = time.perf_counter()
                req_dec, n_cap = solve_bits_sprt(req_truth, p, delta, rng)
                runtime = time.perf_counter() - t0
                recovered = dict(zip(req, req_dec))
                cap_answers = [t.answer([recovered.get(i, latent[i]) for i in range(total_d)]) for t in tasks]
                store.add(TrialRecord(
                    suite=suite,
                    method="Capability_Reuse",
                    config={"kind": kind, "tasks": count, "unique_bits": len(req), "total_d": total_d, "K": nuisance_k, "p": p},
                    seed=rep,
                    interactions=n_cap,
                    success=cap_answers == true_answers,
                    runtime_s=runtime,
                ))

                # Task-specific no-reuse: each task is learned from scratch.
                rng2 = rng_for(base_seed, suite, "noreuse", kind, count, rep)
                total = 0
                answers: List[int] = []
                t0 = time.perf_counter()
                for ti, task in enumerate(tasks):
                    # Use a union-bound confidence budget over task labels. If a
                    # parity task has two bits, split its alpha again.
                    bits_truth = [latent[j] for j in task.bits]
                    task_delta = delta / max(1, len(tasks))
                    dec, n_t = solve_bits_sprt(bits_truth, p, task_delta, rng2)
                    total += n_t
                    v = 0
                    for b in dec:
                        v ^= b
                    if task.negate:
                        v ^= 1
                    answers.append(v)
                runtime = time.perf_counter() - t0
                store.add(TrialRecord(
                    suite=suite,
                    method="TaskSpecific_NoReuse",
                    config={"kind": kind, "tasks": count, "unique_bits": len(req), "total_d": total_d, "K": nuisance_k, "p": p},
                    seed=rep,
                    interactions=total,
                    success=answers == true_answers,
                    runtime_s=runtime,
                ))

                # Full environment identification: learns relevant + nuisance.
                rng3 = rng_for(base_seed, suite, "full", kind, count, rep)
                t0 = time.perf_counter()
                full_dec, n_full = solve_bits_sprt(latent + nuisance, p, delta, rng3)
                runtime = time.perf_counter() - t0
                lat_hat = full_dec[:total_d]
                full_answers = [t.answer(lat_hat) for t in tasks]
                store.add(TrialRecord(
                    suite=suite,
                    method="FullIdentification",
                    config={"kind": kind, "tasks": count, "unique_bits": len(req), "total_d": total_d, "K": nuisance_k, "p": p},
                    seed=rep,
                    interactions=n_full,
                    success=full_answers == true_answers,
                    model_success=full_dec == latent + nuisance,
                    runtime_s=runtime,
                ))


# ---------------------------------------------------------------------------
# Product predictive-rank diagnostic
# ---------------------------------------------------------------------------


def random_transition_block(rng: Generator, states: int, actions: int) -> np.ndarray:
    # shape [states, actions, next_states]
    out = np.empty((states, actions, states), dtype=float)
    for s in range(states):
        for a in range(actions):
            # A mild diagonal/action bias keeps matrices generic while avoiding
            # near-degenerate rows.
            alpha = np.ones(states, dtype=float)
            alpha[(s + a + 1) % states] += 2.0
            out[s, a] = rng.dirichlet(alpha)
    return out


def one_step_dynamics_matrix(P: np.ndarray) -> np.ndarray:
    states, actions, next_states = P.shape
    assert states == next_states
    cols: List[np.ndarray] = []
    for a in range(actions):
        for sp in range(states):
            cols.append(P[:, a, sp])
    return np.column_stack(cols)


def numerical_rank(M: np.ndarray, rtol: float = 1e-10) -> int:
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0:
        return 0
    tol = rtol * max(M.shape) * s[0]
    return int(np.sum(s > tol))


def run_product_rank_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, verbose: bool) -> None:
    suite = "ProductPredictiveRank"
    s_r = 3
    a_r = 2
    rng_r = rng_for(base_seed, suite, "relevant")
    P_r = random_transition_block(rng_r, s_r, a_r)
    D_r = one_step_dynamics_matrix(P_r)
    r_r = numerical_rank(D_r)

    for n_states in cfg["rank_n"]:
        n_states = int(n_states)
        rng_n = rng_for(base_seed, suite, "nuisance", n_states)
        P_n = random_transition_block(rng_n, n_states, 2)
        D_n = one_step_dynamics_matrix(P_n)
        r_n = numerical_rank(D_n)
        D_full = np.kron(D_r, D_n)
        D_task = np.kron(D_r, np.ones((n_states, 1), dtype=float))
        r_full = numerical_rank(D_full)
        r_task = numerical_rank(D_task)
        predicted = r_r * r_n
        if verbose:
            print(f"[{suite}] N={n_states} rank_R={r_r} rank_N={r_n} full={r_full} task={r_task}", flush=True)
        store.diag(
            suite,
            "rank",
            relevant_states=s_r,
            nuisance_states=n_states,
            rank_R=r_r,
            rank_N=r_n,
            rank_full=r_full,
            rank_task=r_task,
            predicted_product_rank=predicted,
            product_identity_ok=bool(r_full == predicted),
            task_invariance_ok=bool(r_task == r_r),
        )


# ---------------------------------------------------------------------------
# Finite Stage-1A / Stage-1B theorem reproduction
# ---------------------------------------------------------------------------


def assignment_label(problem: FiniteControllerProblem, a: np.ndarray) -> str:
    return ",".join(f"{problem.theta_names[i]}->{problem.controller_names[int(a[i])]}" for i in range(problem.n_theta))


def stage1_bound(n_hypotheses: int, delta: float, kappa: float) -> int:
    if math.isinf(kappa):
        return 0
    if kappa <= 0:
        return 10**18
    return int(math.ceil(math.log(max(1.0, (n_hypotheses - 1) / delta)) / kappa))


def run_finite_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, delta: float, verbose: bool) -> None:
    suite = "FiniteControllerIdentification"
    trials = int(cfg["finite_trials"])
    adaptive_trials = int(cfg["adaptive_trials"])

    # A. Overlapping correct-answer sets: tests assignment choice itself.
    problem = make_overlap_problem()
    assignments: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    a_can = canonical_assignment(problem)
    q_can, k_can = solve_assignment_lp(problem, a_can)
    assignments["canonical"] = (a_can, q_can, k_can)

    a_min = minimum_cell_assignment(problem)
    q_min, k_min = solve_assignment_lp(problem, a_min)
    assignments["minimum_cells"] = (a_min, q_min, k_min)

    a_max, q_max, k_max, checked = max_separation_assignment(problem)
    assignments["max_separation"] = (a_max, q_max, k_max)
    store.diag(suite, "max_assignment_search", assignments_checked=checked)

    for name, (a, q, kappa) in assignments.items():
        n = stage1_bound(problem.n_theta, delta, kappa)
        store.diag(
            suite,
            "overlap_assignment",
            assignment_rule=name,
            assignment=assignment_label(problem, a),
            q=q.tolist(),
            kappa=kappa,
            theorem_n=n,
        )
        if verbose:
            print(f"[{suite}] overlap {name}: kappa={kappa:.5f}, n={n}, q={q}", flush=True)
        for rep in range(trials):
            rng = rng_for(base_seed, suite, "overlap", name, rep)
            theta = int(rng.integers(0, problem.n_theta))
            t0 = time.perf_counter()
            ctrl, ok, model_ok, counts, ll = run_fixed_finite_trial(problem, a, q, n, theta, rng)
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method=f"Stage1A_{name}",
                config={"problem": problem.name, "delta": delta, "kappa": kappa, "n_bound": n},
                seed=rep,
                interactions=int(np.sum(counts)),
                success=ok,
                model_success=model_ok,
                runtime_s=runtime,
                extra={
                    "true_theta": theta,
                    "controller": problem.controller_names[ctrl],
                    "counts": counts.tolist(),
                    "ll_margin": float(np.max(ll) - np.partition(ll, -2)[-2]) if len(ll) >= 2 else 0.0,
                },
            ))

        # Adaptive heuristic on the same problem.
        if name in ("minimum_cells", "max_separation"):
            max_steps = max(100, 20 * max(1, n))
            for rep in range(adaptive_trials):
                rng = rng_for(base_seed, suite, "adaptive", name, rep)
                theta = int(rng.integers(0, problem.n_theta))
                t0 = time.perf_counter()
                ctrl, ok, model_ok, steps, counts = run_adaptive_finite_trial(
                    problem, a, theta, rng, delta, max_steps=max_steps
                )
                runtime = time.perf_counter() - t0
                store.add(TrialRecord(
                    suite=suite,
                    method=f"Stage1B_Adaptive_{name}",
                    config={"problem": problem.name, "delta": delta, "max_steps": max_steps},
                    seed=rep,
                    interactions=steps,
                    success=ok,
                    model_success=model_ok,
                    censored=steps >= max_steps,
                    runtime_s=runtime,
                    extra={"true_theta": theta, "controller": problem.controller_names[ctrl], "counts": counts.tolist()},
                ))

    # B. Small product nuisance: the experiment design should place zero mass on
    # nuisance-only probes when the answer is the relevant controller.
    for nuisance_models in cfg["product_n"]:
        nuisance_models = int(nuisance_models)
        # Keep finite enumeration modest for the LP.
        if nuisance_models > 32:
            continue
        prod = make_small_product_problem(d=2, nuisance_models=nuisance_models, flip_p=0.15)
        a = canonical_assignment(prod)  # one z-controller per model
        q_cap, k_cap = solve_assignment_lp(prod, a)
        z_count = len(prod.controller_names)
        n_cap_raw = stage1_bound(prod.n_theta, delta, k_cap)
        n_cap_quot = stage1_bound(z_count, delta, k_cap)

        # Full-identification design: every theta is its own answer cell.
        h2 = pairwise_hellinger(prod)
        all_pairs = [(i, j) for i in range(prod.n_theta) for j in range(i + 1, prod.n_theta)]
        q_full, k_full = solve_separation_lp(h2, all_pairs)
        n_full = stage1_bound(prod.n_theta, delta, k_full)

        store.diag(
            suite,
            "product_design",
            nuisance_models=nuisance_models,
            theta_count=prod.n_theta,
            controller_count=z_count,
            q_cap=q_cap.tolist(),
            k_cap=k_cap,
            n_cap_raw=n_cap_raw,
            n_cap_quotient=n_cap_quot,
            q_full=q_full.tolist(),
            k_full=k_full,
            n_full=n_full,
            nuisance_mass_cap=float(q_cap[-1]),
            nuisance_mass_full=float(q_full[-1]),
        )
        if verbose:
            print(
                f"[{suite}] product N={nuisance_models}: cap_q_nuis={q_cap[-1]:.4g}, "
                f"full_q_nuis={q_full[-1]:.4g}, nQ={n_cap_quot}, nFull={n_full}",
                flush=True,
            )

        for rep in range(trials):
            rng = rng_for(base_seed, suite, "product", nuisance_models, rep)
            theta = int(rng.integers(0, prod.n_theta))

            # Quotient/profiled capability decoder with quotient confidence term.
            t0 = time.perf_counter()
            ctrl, ok, model_ok, counts, _ = run_fixed_finite_trial(
                prod, a, q_cap, n_cap_quot, theta, rng, decoder="profiled_controller"
            )
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="Capability_QuotientProfile",
                config={"problem": "small_product", "N": nuisance_models, "delta": delta, "n_bound": n_cap_quot},
                seed=rep,
                interactions=int(np.sum(counts)),
                success=ok,
                model_success=model_ok,
                runtime_s=runtime,
                extra={"nuisance_probe_fraction": float(counts[-1] / max(1, np.sum(counts)))},
            ))

            # Full identification MLE.
            rng2 = rng_for(base_seed, suite, "product_full", nuisance_models, rep)
            t0 = time.perf_counter()
            counts_full = allocate_theorem_counts(n_full, q_full)
            obs = sample_finite_data(prod, theta, counts_full, rng2)
            ll = finite_loglik(prod, obs)
            mle = int(np.argmax(ll))
            ctrl_full = int(a[mle])
            runtime = time.perf_counter() - t0
            store.add(TrialRecord(
                suite=suite,
                method="FullIdentification_MLE",
                config={"problem": "small_product", "N": nuisance_models, "delta": delta, "n_bound": n_full},
                seed=rep,
                interactions=int(np.sum(counts_full)),
                success=bool(prod.adequate[theta, ctrl_full]),
                model_success=mle == theta,
                runtime_s=runtime,
                extra={"nuisance_probe_fraction": float(counts_full[-1] / max(1, np.sum(counts_full)))},
            ))


# ---------------------------------------------------------------------------
# Sparse-reward combination-lock benchmark
# ---------------------------------------------------------------------------

@dataclass
class LockRun:
    interactions: int
    success: bool
    episodes: int
    censored: bool
    learned_actions: List[int]
    extra: Dict[str, Any] = field(default_factory=dict)


class CombinationLock:
    def __init__(self, correct_actions: Sequence[int], reward_prob: float):
        self.correct = list(map(int, correct_actions))
        self.H = len(self.correct)
        self.dead = self.H + 1
        self.terminal = self.H
        self.reward_prob = float(reward_prob)

    @property
    def n_states(self) -> int:
        return self.H + 2

    def step(self, state: int, h: int, action: int, rng: Generator) -> Tuple[int, float]:
        if state == self.dead:
            return self.dead, 0.0
        if state == self.terminal:
            return self.terminal, 0.0
        if state != h:
            # In this construction a live trajectory's state equals its depth.
            return self.dead, 0.0
        next_state = h + 1 if int(action) == self.correct[h] else self.dead
        reward = 0.0
        if h == self.H - 1 and next_state == self.terminal:
            reward = 1.0 if rng.random() < self.reward_prob else 0.0
        return next_state, reward

    def policy_is_optimal(self, actions: Sequence[int]) -> bool:
        return list(map(int, actions[: self.H])) == self.correct


def capability_lock_solver(env: CombinationLock, rng: Generator) -> LockRun:
    # Uses transition consequences, not terminal reward. With deterministic
    # binary actions and the known "exactly one advances" structure, one probe
    # per depth suffices. Reaching depth h requires replaying the learned prefix.
    learned: List[int] = []
    interactions = 0
    episodes = 0
    for h in range(env.H):
        state = 0
        episodes += 1
        # Replay known prefix.
        for i, a in enumerate(learned):
            state, _ = env.step(state, i, a, rng)
            interactions += 1
        # Probe action 0. If it advances, 0 is correct; otherwise 1 is correct.
        next_state, _ = env.step(state, h, 0, rng)
        interactions += 1
        learned.append(0 if next_state == h + 1 else 1)
    return LockRun(interactions, env.policy_is_optimal(learned), episodes, False, learned)


def reward_free_structure_aware_lock(env: CombinationLock, rng: Generator) -> LockRun:
    # A strong structure-aware reward-free baseline. Under exactly the same
    # structural knowledge, it should match Capability-Track. This is
    # intentional: if it matches, the empirical advantage belongs to the
    # structural principle, not a unique algorithmic label.
    return capability_lock_solver(env, rng)


def reward_free_full_transition_lock(env: CombinationLock, rng: Generator) -> LockRun:
    # Learns both actions' transitions at every reachable depth even though one
    # probe would suffice under the complement structure. This is the broader
    # system-identification baseline.
    learned: List[int] = []
    interactions = 0
    episodes = 0
    for h in range(env.H):
        observed: Dict[int, int] = {}
        for probe_action in (0, 1):
            state = 0
            episodes += 1
            for i, a in enumerate(learned):
                state, _ = env.step(state, i, a, rng)
                interactions += 1
            ns, _ = env.step(state, h, probe_action, rng)
            interactions += 1
            observed[probe_action] = ns
        learned.append(0 if observed[0] == h + 1 else 1)
    return LockRun(interactions, env.policy_is_optimal(learned), episodes, False, learned)


def q_learning_lock(
    env: CombinationLock,
    rng: Generator,
    max_episodes: int,
    eval_every: int = 25,
) -> LockRun:
    S, A, H = env.n_states, 2, env.H
    Q = np.zeros((S, A), dtype=float)
    N = np.zeros((S, A), dtype=int)
    interactions = 0

    def greedy_actions() -> List[int]:
        return [int(np.argmax(Q[h])) for h in range(H)]

    for ep in range(1, max_episodes + 1):
        state = 0
        epsilon = max(0.02, 0.25 / math.sqrt(1.0 + ep / 100.0))
        for h in range(H):
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                # Random tie-breaking prevents an arbitrary action-0 prior from
                # dominating early behavior.
                qrow = Q[state]
                best = np.flatnonzero(np.isclose(qrow, np.max(qrow)))
                action = int(rng.choice(best))
            next_state, reward = env.step(state, h, action, rng)
            interactions += 1
            N[state, action] += 1
            alpha = 1.0 / math.sqrt(N[state, action])
            next_v = 0.0 if next_state in (env.dead, env.terminal) else float(np.max(Q[next_state]))
            target = reward + next_v
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
        if ep % eval_every == 0 or ep == 1:
            acts = greedy_actions()
            if env.policy_is_optimal(acts):
                return LockRun(interactions, True, ep, False, acts, {"epsilon": epsilon})
    acts = greedy_actions()
    return LockRun(interactions, env.policy_is_optimal(acts), max_episodes, True, acts)


def ucbvi_plan(
    N: np.ndarray,
    trans: np.ndarray,
    reward_sum: np.ndarray,
    H: int,
    dead: int,
    terminal: int,
    episode: int,
    delta: float,
    optimistic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    S = trans.shape[-1]
    A = 2
    V = np.zeros((H + 1, S), dtype=float)
    policy = np.zeros((H, S), dtype=int)
    logterm = math.log(max(3.0, 2.0 * H * S * A * (episode + 1) ** 2 / delta))

    for h in reversed(range(H)):
        Qh = np.zeros((S, A), dtype=float)
        for s in range(S):
            if s in (dead, terminal):
                Qh[s, :] = 0.0
                continue
            for a in range(A):
                n = int(N[h, s, a])
                if n == 0:
                    Qh[s, a] = 1.0 if optimistic else 0.0
                    continue
                rhat = float(reward_sum[h, s, a] / n)
                phat = trans[h, s, a] / n
                estimate = rhat + float(np.dot(phat, V[h + 1]))
                if optimistic:
                    # Conservative simple UCBVI-like bonus. This is a strong
                    # tabular optimistic model-based baseline, but not a claim
                    # of exact constant matching to a particular paper theorem.
                    bonus = 2.0 * H * math.sqrt(logterm / n) + 3.0 * H * logterm / n
                    estimate += bonus
                Qh[s, a] = min(1.0, estimate)
        policy[h] = np.argmax(Qh, axis=1)
        V[h] = np.max(Qh, axis=1)
    return policy, V


def ucbvi_lock(env: CombinationLock, rng: Generator, max_episodes: int, delta: float) -> LockRun:
    H, S, A = env.H, env.n_states, 2
    N = np.zeros((H, S, A), dtype=int)
    trans = np.zeros((H, S, A, S), dtype=int)
    reward_sum = np.zeros((H, S, A), dtype=float)
    interactions = 0

    for ep in range(1, max_episodes + 1):
        pol_opt, _ = ucbvi_plan(N, trans, reward_sum, H, env.dead, env.terminal, ep, delta, optimistic=True)
        state = 0
        for h in range(H):
            action = int(pol_opt[h, state])
            ns, r = env.step(state, h, action, rng)
            interactions += 1
            N[h, state, action] += 1
            trans[h, state, action, ns] += 1
            reward_sum[h, state, action] += r
            state = ns

        if ep % 5 == 0 or ep == 1:
            pol_emp, _ = ucbvi_plan(N, trans, reward_sum, H, env.dead, env.terminal, ep, delta, optimistic=False)
            acts = [int(pol_emp[h, h]) for h in range(H)]
            if env.policy_is_optimal(acts):
                return LockRun(interactions, True, ep, False, acts)

    pol_emp, _ = ucbvi_plan(N, trans, reward_sum, H, env.dead, env.terminal, max_episodes, delta, optimistic=False)
    acts = [int(pol_emp[h, h]) for h in range(H)]
    return LockRun(interactions, env.policy_is_optimal(acts), max_episodes, True, acts)


def run_reward_sparsity_suite(store: ResultStore, cfg: Mapping[str, Any], base_seed: int, delta: float, verbose: bool) -> None:
    suite = "RewardSparsityCombinationLock"
    trials = int(cfg["rl_trials"])
    max_episodes = int(cfg["rl_max_episodes"])
    methods: List[Tuple[str, Callable[[CombinationLock, Generator], LockRun]]] = [
        ("Capability_TransitionDirected", lambda e, r: capability_lock_solver(e, r)),
        ("RewardFree_StructureAware", lambda e, r: reward_free_structure_aware_lock(e, r)),
        ("RewardFree_FullTransition", lambda e, r: reward_free_full_transition_lock(e, r)),
        ("Qlearning", lambda e, r: q_learning_lock(e, r, max_episodes=max_episodes)),
        ("UCBVI_like", lambda e, r: ucbvi_lock(e, r, max_episodes=max_episodes, delta=delta)),
    ]

    for H in cfg["lock_h"]:
        H = int(H)
        for reward_prob in cfg["reward_prob"]:
            reward_prob = float(reward_prob)
            if verbose:
                print(f"[{suite}] H={H} reward_prob={reward_prob:.4f}", flush=True)
            for rep in range(trials):
                correct_rng = rng_for(base_seed, suite, "lock", H, reward_prob, rep)
                correct = correct_rng.integers(0, 2, size=H).astype(int).tolist()
                for method_name, solver in methods:
                    rng = rng_for(base_seed, suite, method_name, H, reward_prob, rep)
                    env = CombinationLock(correct, reward_prob)
                    t0 = time.perf_counter()
                    run = solver(env, rng)
                    runtime = time.perf_counter() - t0
                    store.add(TrialRecord(
                        suite=suite,
                        method=method_name,
                        config={
                            "H": H,
                            "reward_prob": reward_prob,
                            "max_episodes": max_episodes,
                            "objective": "reach_terminal_sparse_reward",
                        },
                        seed=rep,
                        interactions=run.interactions,
                        success=run.success,
                        runtime_s=runtime,
                        censored=run.censored,
                        extra={"episodes": run.episodes, "learned_actions": run.learned_actions, **run.extra},
                    ))


# ---------------------------------------------------------------------------
# Aggregation and scientific report
# ---------------------------------------------------------------------------


def config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def aggregate(records: Sequence[TrialRecord], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[TrialRecord]] = defaultdict(list)
    for r in records:
        key_parts: List[Any] = []
        for f in group_fields:
            if f == "suite":
                key_parts.append(r.suite)
            elif f == "method":
                key_parts.append(r.method)
            else:
                key_parts.append(r.config.get(f))
        groups[tuple(key_parts)].append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        row = dict(zip(group_fields, key))
        ints = [float(r.interactions) for r in rs]
        success_n = sum(bool(r.success) for r in rs)
        lo, hi = wilson_interval(success_n, len(rs))
        m, mlo, mhi = mean_ci(ints)
        q25, med, q75 = quantiles(ints)
        runtimes = [r.runtime_s for r in rs]
        row.update({
            "n": len(rs),
            "success_rate": success_n / len(rs),
            "success_lo": lo,
            "success_hi": hi,
            "mean_interactions": m,
            "mean_interactions_lo": mlo,
            "mean_interactions_hi": mhi,
            "median_interactions": med,
            "q25_interactions": q25,
            "q75_interactions": q75,
            "censor_rate": sum(r.censored for r in rs) / len(rs),
            "mean_runtime_s": float(np.mean(runtimes)) if runtimes else float("nan"),
        })
        model_vals = [r.model_success for r in rs if r.model_success is not None]
        if model_vals:
            row["model_success_rate"] = sum(bool(x) for x in model_vals) / len(model_vals)
        out.append(row)
    return out


def aggregate_by_config(records: Sequence[TrialRecord]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[TrialRecord]] = defaultdict(list)
    configs: Dict[str, Dict[str, Any]] = {}
    for r in records:
        ck = config_key(r.config)
        groups[(r.suite, r.method, ck)].append(r)
        configs[ck] = r.config
    rows: List[Dict[str, Any]] = []
    for (suite, method, ck), rs in groups.items():
        ints = [float(r.interactions) for r in rs]
        success_n = sum(r.success for r in rs)
        lo, hi = wilson_interval(success_n, len(rs))
        _, med, _ = quantiles(ints)
        row: Dict[str, Any] = {
            "suite": suite,
            "method": method,
            **configs[ck],
            "n": len(rs),
            "success_rate": success_n / len(rs),
            "success_lo": lo,
            "success_hi": hi,
            "mean_interactions": float(np.mean(ints)),
            "median_interactions": med,
            "censor_rate": sum(r.censored for r in rs) / len(rs),
            "mean_runtime_s": float(np.mean([r.runtime_s for r in rs])),
        }
        model_vals = [r.model_success for r in rs if r.model_success is not None]
        if model_vals:
            row["model_success_rate"] = sum(bool(x) for x in model_vals) / len(model_vals)
        rows.append(row)
    return rows


def select_rows(rows: Sequence[Mapping[str, Any]], **conditions: Any) -> List[Mapping[str, Any]]:
    out = []
    for row in rows:
        ok = True
        for k, v in conditions.items():
            if row.get(k) != v:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def diagnostics_for(store: ResultStore, suite: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
    return [d for d in store.diagnostics if d.get("suite") == suite and (name is None or d.get("name") == name)]


def summarize_finite(rb: ReportBuilder, store: ResultStore, rows: List[Dict[str, Any]], delta: float) -> Dict[str, Any]:
    rb.section("1. Finite theorem-matching controller identification")
    rb.text(
        "This is the Level-0 gate. Fixed-allocation Stage 1A uses the Hellinger theorem with "
        "ceil(n*q_e) integer rounding so every experiment count satisfies the proof condition. "
        "Stage 1B rows are heuristic/adaptive and are not treated as theorem-certified."
    )

    rb.subsection("Safe assignment / Hellinger LP diagnostics")
    diag = diagnostics_for(store, "FiniteControllerIdentification", "overlap_assignment")
    rb.table(
        ["assignment", "kappa", "base_n", "q", "mapping"],
        [[d["assignment_rule"], d["kappa"], d["theorem_n"], d["q"], d["assignment"]] for d in diag],
    )

    rb.subsection("Empirical fixed/adaptive outcomes")
    fr = [r for r in rows if r["suite"] == "FiniteControllerIdentification" and r.get("problem") == "OverlappingControllerID"]
    fr.sort(key=lambda x: x["method"])
    rb.table(
        ["method", "n", "success", "95% CI", "mean interactions", "model success", "censor"],
        [[
            r["method"], r["n"], r["success_rate"], f"[{fmt(r['success_lo'])},{fmt(r['success_hi'])}]",
            r["mean_interactions"], r.get("model_success_rate", "NA"), r["censor_rate"]
        ] for r in fr],
    )

    rb.subsection("Product nuisance experiment-design diagnostics")
    pd = diagnostics_for(store, "FiniteControllerIdentification", "product_design")
    rb.table(
        ["N nuisance", "theta", "cap nuisance q", "full nuisance q", "cap quotient n", "full n", "kappa cap", "kappa full"],
        [[d["nuisance_models"], d["theta_count"], d["nuisance_mass_cap"], d["nuisance_mass_full"],
          d["n_cap_quotient"], d["n_full"], d["k_cap"], d["k_full"]] for d in pd],
    )

    pr = [r for r in rows if r["suite"] == "FiniteControllerIdentification" and r.get("problem") == "small_product"]
    pr.sort(key=lambda x: (int(x.get("N", 0)), x["method"]))
    rb.table(
        ["N", "method", "success", "model success", "mean interactions"],
        [[r.get("N"), r["method"], r["success_rate"], r.get("model_success_rate", "NA"), r["mean_interactions"]] for r in pr],
    )

    fixed = [r for r in fr if r["method"].startswith("Stage1A_")]
    failures = [1.0 - float(r["success_rate"]) for r in fixed]
    theorem_gate = bool(fixed) and max(failures) <= max(2.0 * delta, delta + 0.03)
    nuisance_mass_gate = bool(pd) and max(abs(float(d["nuisance_mass_cap"])) for d in pd) < 1e-8
    return {
        "finite_theorem_gate": theorem_gate,
        "finite_max_failure_rate": max(failures) if failures else float("nan"),
        "product_cap_nuisance_mass_zero": nuisance_mass_gate,
    }


def summarize_noisy_bits(rb: ReportBuilder, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rb.section("2. NoisyRelevantBits: nuisance and relevant-dimension scaling")
    nr = [r for r in rows if r["suite"] == "NoisyRelevantBits"]

    rb.subsection("Nuisance sweep")
    ns = [r for r in nr if r.get("sweep") == "nuisance_sweep" and r["method"] in ("Capability_SPRT", "FullIdentification_SPRT")]
    ns.sort(key=lambda x: (int(x["K"]), x["method"]))
    rb.table(
        ["K", "d", "p", "method", "success", "mean interactions", "model success"],
        [[r["K"], r["d"], r["p"], r["method"], r["success_rate"], r["mean_interactions"], r.get("model_success_rate", "NA")] for r in ns],
    )

    fits: Dict[str, Dict[str, float]] = {}
    for method in ("Capability_SPRT", "FullIdentification_SPRT"):
        mr = [r for r in ns if r["method"] == method]
        mr.sort(key=lambda x: int(x["K"]))
        fits[method] = loglog_fit([int(r["K"]) + 1 for r in mr], [float(r["mean_interactions"]) for r in mr])
        rb.text(f"log-log fit interactions vs (K+1), {method}: {json.dumps(fits[method], sort_keys=True)}")

    rb.subsection("Relevant-dimension sweep")
    ds = [r for r in nr if r.get("sweep") == "relevant_sweep" and r["method"] in ("Capability_SPRT", "FullIdentification_SPRT")]
    ds.sort(key=lambda x: (int(x["d"]), x["method"]))
    rb.table(
        ["d", "K", "method", "success", "mean interactions"],
        [[r["d"], r["K"], r["method"], r["success_rate"], r["mean_interactions"]] for r in ds],
    )
    cap_d = [r for r in ds if r["method"] == "Capability_SPRT"]
    cap_d.sort(key=lambda x: int(x["d"]))
    d_fit = loglog_fit([int(r["d"]) for r in cap_d], [float(r["mean_interactions"]) for r in cap_d])
    rb.text(f"Capability log-log fit interactions vs d: {json.dumps(d_fit, sort_keys=True)}")

    rb.subsection("Noise sweep")
    ps = [r for r in nr if r.get("sweep") == "noise_sweep" and r["method"] == "Capability_SPRT"]
    ps.sort(key=lambda x: float(x["p"]))
    rb.table(
        ["p", "gap=0.5-p", "success", "mean interactions"],
        [[r["p"], 0.5 - float(r["p"]), r["success_rate"], r["mean_interactions"]] for r in ps],
    )

    cap_slope = fits.get("Capability_SPRT", {}).get("slope", float("nan"))
    full_slope = fits.get("FullIdentification_SPRT", {}).get("slope", float("nan"))
    nuisance_gate = bool(np.isfinite(cap_slope)) and abs(cap_slope) < 0.20
    separation_gate = nuisance_gate and bool(np.isfinite(full_slope)) and full_slope > 0.45
    return {
        "cap_nuisance_loglog_slope": cap_slope,
        "full_nuisance_loglog_slope": full_slope,
        "nuisance_invariance_gate": nuisance_gate,
        "nuisance_scaling_separation_gate": separation_gate,
        "cap_relevant_dimension_slope": d_fit.get("slope", float("nan")),
    }


def summarize_information_gap(rb: ReportBuilder, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rb.section("3. InformationGap: low rank is not enough")
    ir = [r for r in rows if r["suite"] == "InformationGap" and r["method"] == "Capability_SPRT"]
    ir.sort(key=lambda x: float(x["eta"]), reverse=True)
    rb.table(
        ["eta", "1/eta^2", "p", "success", "mean interactions"],
        [[r["eta"], 1.0 / (float(r["eta"]) ** 2), r["p"], r["success_rate"], r["mean_interactions"]] for r in ir],
    )
    fit = loglog_fit([1.0/(float(r["eta"])**2) for r in ir], [float(r["mean_interactions"]) for r in ir])
    rb.text(f"log-log fit interactions vs 1/eta^2: {json.dumps(fit, sort_keys=True)}")
    gate = bool(np.isfinite(fit["slope"])) and 0.60 <= fit["slope"] <= 1.40 and fit["r2"] >= 0.80
    return {"information_gap_fit": fit, "information_gap_gate": gate}


def summarize_task_family(rb: ReportBuilder, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rb.section("4. TaskFamilyGrowth: task labels versus new predictive directions")
    tr = [r for r in rows if r["suite"] == "TaskFamilyGrowth"]
    tr.sort(key=lambda x: (str(x["kind"]), int(x["tasks"]), x["method"]))
    rb.table(
        ["kind", "tasks", "unique bits", "method", "success", "mean interactions"],
        [[r["kind"], r["tasks"], r["unique_bits"], r["method"], r["success_rate"], r["mean_interactions"]] for r in tr],
    )

    metrics: Dict[str, Any] = {}
    for kind in ("redundant", "shared_parity", "new_directions"):
        cap = [r for r in tr if r["kind"] == kind and r["method"] == "Capability_Reuse"]
        cap.sort(key=lambda x: int(x["tasks"]))
        fit = loglog_fit([int(r["tasks"]) for r in cap], [float(r["mean_interactions"]) for r in cap])
        metrics[f"{kind}_fit"] = fit
        rb.text(f"Capability interactions vs task count, {kind}: {json.dumps(fit, sort_keys=True)}")

    redundant_slope = metrics.get("redundant_fit", {}).get("slope", float("nan"))
    new_slope = metrics.get("new_directions_fit", {}).get("slope", float("nan"))
    gate = bool(np.isfinite(redundant_slope) and np.isfinite(new_slope)) and abs(redundant_slope) < 0.20 and new_slope > 0.45
    metrics["task_family_gate"] = gate
    return metrics


def summarize_rank(rb: ReportBuilder, store: ResultStore) -> Dict[str, Any]:
    rb.section("5. Product predictive-rank diagnostic")
    ds = diagnostics_for(store, "ProductPredictiveRank", "rank")
    rb.table(
        ["N states", "rank R", "rank N", "rank full", "rank task", "product OK", "task invariant"],
        [[d["nuisance_states"], d["rank_R"], d["rank_N"], d["rank_full"], d["rank_task"], d["product_identity_ok"], d["task_invariance_ok"]] for d in ds],
    )
    gate = bool(ds) and all(bool(d["product_identity_ok"]) and bool(d["task_invariance_ok"]) for d in ds)
    return {"rank_identity_gate": gate}


def summarize_reward_sparsity(rb: ReportBuilder, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rb.section("6. RewardSparsity CombinationLock diagnostic")
    rr = [r for r in rows if r["suite"] == "RewardSparsityCombinationLock"]
    rr.sort(key=lambda x: (int(x["H"]), -float(x["reward_prob"]), x["method"]))
    rb.text(
        "Interpretation warning: a sparse-reward win over Q-learning is not evidence for the core claim. "
        "The structure-aware reward-free baseline is intentionally expected to match the capability learner "
        "under the same structural assumptions."
    )
    rb.table(
        ["H", "reward p", "method", "success", "censor", "mean interactions", "median interactions"],
        [[r["H"], r["reward_prob"], r["method"], r["success_rate"], r["censor_rate"], r["mean_interactions"], r["median_interactions"]] for r in rr],
    )

    # Compare the two structure-aware methods. Exact equality is expected in the
    # current deterministic construction because they share the same solver.
    ratios = []
    for H in sorted(set(int(r["H"]) for r in rr)):
        for rp in sorted(set(float(r["reward_prob"]) for r in rr)):
            c = [r for r in rr if int(r["H"]) == H and float(r["reward_prob"]) == rp and r["method"] == "Capability_TransitionDirected"]
            s = [r for r in rr if int(r["H"]) == H and float(r["reward_prob"]) == rp and r["method"] == "RewardFree_StructureAware"]
            if c and s and float(c[0]["mean_interactions"]) > 0:
                ratios.append(float(s[0]["mean_interactions"]) / float(c[0]["mean_interactions"]))
    match = bool(ratios) and max(abs(x - 1.0) for x in ratios) < 1e-12
    return {
        "structure_aware_baseline_matches": match,
        "structure_aware_interaction_ratios": ratios,
        "interpretation": (
            "If true, the sparse-reward advantage is a property of using transition/capability structure, "
            "not evidence that the current Capability-Track implementation uniquely dominates structure-aware reward-free RL."
        ),
    }


def automatic_verdict(metrics: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
    positives: List[str] = []
    cautions: List[str] = []

    level0 = bool(metrics.get("finite_theorem_gate")) and bool(metrics.get("rank_identity_gate"))
    if level0:
        positives.append("Level 0 gate passed: finite theorem-matching implementation and rank identities reproduced.")
    else:
        cautions.append("Level 0 gate failed or was inconclusive. Do not advance the scientific claim before debugging/retesting.")

    level1 = level0 and bool(metrics.get("nuisance_scaling_separation_gate")) and bool(metrics.get("information_gap_gate")) and bool(metrics.get("task_family_gate"))
    if level1:
        positives.append("Level 1 gate passed: synthetic scaling follows nuisance/task-family/information-gap predictions.")
    else:
        cautions.append("Level 1 scaling gate not fully passed. At least one preregistered signature is absent or statistically weak.")

    # Level 2 requires an advantage over a strong structure-aware baseline on the
    # same assumptions. In this first harness the structure-aware lock baseline
    # intentionally shares the same structural information and is expected to
    # match. Therefore matching does NOT award Level 2.
    level2 = False
    if metrics.get("structure_aware_baseline_matches"):
        cautions.append(
            "Strong structure-aware reward-free baseline matches Capability-Track in CombinationLock. "
            "This supports the structural principle but blocks any unique-algorithm superiority claim in that regime."
        )
    else:
        cautions.append(
            "No Level 2 conclusion from this harness alone. A same-assumption strong-baseline comparison must be interpreted manually."
        )

    level = 2 if level2 else (1 if level1 else (0 if level0 else -1))
    return level, positives, cautions


def build_report(
    store: ResultStore,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    start_time: float,
    end_time: float,
    script_path: Path,
) -> str:
    rb = ReportBuilder()
    rb.title("CAPABILITY-TRACK STAGE-1 FALSIFICATION REPORT")
    rb.text(f"Script version: {SCRIPT_VERSION}")
    rb.text(f"Script SHA-256: {sha256_file(script_path)}")
    rb.text(f"Profile: {args.profile}")
    rb.text(f"Base seed: {args.seed}")
    rb.text(f"delta: {args.delta}")
    rb.text(f"Wall time: {end_time - start_time:.3f} s")
    rb.text(f"Python: {sys.version.replace(os.linesep, ' ')}")
    rb.text(f"Platform: {platform.platform()}")
    rb.text(f"CPU count: {os.cpu_count()}")
    rb.text(f"NumPy: {np.__version__}")
    rb.text(f"SciPy: {getattr(scipy, '__version__', 'unknown')}")
    rb.text(f"psutil: {getattr(psutil, '__version__', 'not installed') if psutil is not None else 'not installed'}")
    rb.text(f"Final RSS MB: {fmt(current_rss_mb())}")
    mem_diag = diagnostics_for(store, "System", "peak_memory")
    rb.text(f"Peak RSS MB (sampled): {fmt(mem_diag[-1].get('peak_rss_mb') if mem_diag else None)}")
    rb.text(f"Command: {' '.join(sys.argv)}")
    rb.text(f"Resolved profile config: {json.dumps(dict(cfg), sort_keys=True)}")

    rb.section("Research hypothesis and scope")
    rb.text(
        "Hypothesis under test: for a DECLARED family of future decisions, learning only residual/decision-relevant "
        "causal distinctions can have lower representation and interaction complexity than full environment identification "
        "or task-by-task relearning, while retaining post-hoc competence on every declared task."
    )
    rb.text(
        "Non-claim: this report cannot by itself show that all reinforcement-learning methods are inferior. Modern reward-free, "
        "model-based, representation-aware, and task-family-aware RL can exploit related structure. Stage 1 optimizes fixed-confidence "
        "learn-then-deploy competence, not cumulative online regret."
    )

    if store.errors:
        rb.section("Execution errors")
        for e in store.errors:
            rb.text(e)

    rows = aggregate_by_config(store.records)
    metrics: Dict[str, Any] = {}
    metrics.update(summarize_finite(rb, store, rows, args.delta))
    metrics.update(summarize_noisy_bits(rb, rows))
    metrics.update(summarize_information_gap(rb, rows))
    metrics.update(summarize_task_family(rb, rows))
    metrics.update(summarize_rank(rb, store))
    metrics.update(summarize_reward_sparsity(rb, rows))

    rb.section("7. Preregistered gate / kill-criterion evaluation")
    gate_rows = []
    for key in (
        "finite_theorem_gate",
        "product_cap_nuisance_mass_zero",
        "nuisance_invariance_gate",
        "nuisance_scaling_separation_gate",
        "information_gap_gate",
        "task_family_gate",
        "rank_identity_gate",
        "structure_aware_baseline_matches",
    ):
        gate_rows.append([key, metrics.get(key)])
    rb.table(["criterion", "result"], gate_rows)
    rb.text("Detailed metrics JSON:")
    rb.text(json.dumps(metrics, indent=2, sort_keys=True, default=str))

    level, positives, cautions = automatic_verdict(metrics)
    rb.section("8. Automatic evidence-level summary")
    rb.text(f"Maximum automatically supported evidence level from this run: {level}")
    rb.text("Research claim ladder: -1=failed/inconclusive Level0, 0=theorem reproduction, 1=synthetic scaling, 2=strong-baseline survival.")
    for x in positives:
        rb.text("POSITIVE: " + x)
    for x in cautions:
        rb.text("CAUTION: " + x)

    rb.section("9. What I should evaluate when you return this report")
    rb.text("1. Whether finite Stage1A empirical error is consistent with the conservative Hellinger guarantee.")
    rb.text("2. Whether Capability_SPRT stays flat in nuisance K while full identification grows.")
    rb.text("3. Whether the advantage disappears when task requirements expand, rather than persisting by construction.")
    rb.text("4. Whether InformationGap follows the predicted inverse-information scaling at fixed representation rank.")
    rb.text("5. Whether task cost tracks unique predictive directions rather than task-label count.")
    rb.text("6. Whether strong structure-aware baselines erase the apparent advantage in regimes where they share the same assumptions.")
    rb.text("7. Whether computation/runtime or censoring makes the statistical advantage irrelevant.")
    rb.text("8. Whether any failed gate is a code/theorem-assumption issue or a genuine falsification.")

    rb.section("10. All diagnostics (machine-readable JSON Lines)")
    for d in store.diagnostics:
        rb.text(json.dumps(d, sort_keys=True, default=str))

    if bool(cfg.get("raw_records", True)) and not args.no_raw:
        rb.section("11. ALL RAW TRIAL RECORDS (JSON Lines)")
        rb.text("Each following line is one complete trial record. Preserve this section when returning the report.")
        for r in store.records:
            rb.text(json.dumps(r.as_dict(), sort_keys=True, default=str))
    else:
        rb.section("11. Raw records omitted")
        rb.text("Raw records were disabled by configuration/CLI. Aggregates remain above, but this is less useful for audit.")

    return rb.render()


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capability-Track Stage-1 falsification harness. Produces one exhaustive TXT report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard", help="Experiment scale. 'deep' is the intended serious run.")
    parser.add_argument("--seed", type=int, default=20260905, help="Base deterministic seed.")
    parser.add_argument("--delta", type=float, default=0.05, help="Target fixed-confidence failure probability.")
    parser.add_argument("--report", type=str, default="", help="Output TXT path. Default includes timestamp and profile.")
    parser.add_argument("--no-raw", action="store_true", help="Omit raw JSONL trial records from the TXT report.")
    parser.add_argument("--skip-rl", action="store_true", help="Skip the potentially slow CombinationLock RL suite.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.delta < 0.25):
        raise ValueError("--delta must be in (0, 0.25) for the intended fixed-confidence tests")


def run_all(args: argparse.Namespace) -> Tuple[ResultStore, Dict[str, Any], float, float]:
    validate_runtime_args(args)
    cfg = dict(PROFILES[args.profile])
    store = ResultStore()
    verbose = not args.quiet
    start = time.time()

    suites: List[Tuple[str, Callable[[], None]]] = [
        ("FiniteControllerIdentification", lambda: run_finite_suite(store, cfg, args.seed, args.delta, verbose)),
        ("NoisyRelevantBits", lambda: run_noisy_bits_suite(store, cfg, args.seed, args.delta, verbose)),
        ("InformationGap", lambda: run_information_gap_suite(store, cfg, args.seed, args.delta, verbose)),
        ("TaskFamilyGrowth", lambda: run_task_family_suite(store, cfg, args.seed, args.delta, verbose)),
        ("ProductPredictiveRank", lambda: run_product_rank_suite(store, cfg, args.seed, verbose)),
    ]
    if not args.skip_rl:
        suites.append(("RewardSparsityCombinationLock", lambda: run_reward_sparsity_suite(store, cfg, args.seed, args.delta, verbose)))

    for name, fn in suites:
        if verbose:
            print(f"\n=== Running {name} ===", flush=True)
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except Exception:
            err = f"SUITE FAILED: {name}\n{traceback.format_exc()}"
            store.error(err)
            print(err, file=sys.stderr, flush=True)

    end = time.time()
    return store, cfg, start, end


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = Path.cwd() / f"capability_track_report_{args.profile}_{stamp}.txt"

    print("Capability-Track Stage-1 falsification harness")
    print(f"Version: {SCRIPT_VERSION}")
    print(f"Profile: {args.profile}; delta={args.delta}; seed={args.seed}")
    print(f"Report will be written to: {report_path}")
    if args.profile == "deep":
        print("Deep profile selected. The RL suite may be computationally expensive on some PCs.")

    monitor = MemoryMonitor()
    monitor.start()
    try:
        store, cfg, start, end = run_all(args)
    finally:
        monitor.stop()
    store.diag("System", "peak_memory", peak_rss_mb=monitor.peak_mb)
    report = build_report(store, args, cfg, start, end, script_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\nDONE")
    print(f"Report: {report_path}")
    print(f"Trials recorded: {len(store.records)}")
    print(f"Diagnostics recorded: {len(store.diagnostics)}")
    print(f"Suite errors: {len(store.errors)}")
    print(f"Script SHA-256: {sha256_file(script_path)}")
    print("Return the TXT report to ChatGPT for evaluation. Preserve the raw-record section if possible.")
    return 0 if not store.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
