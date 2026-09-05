#!/usr/bin/env python3
"""
Capability-Track Stage 2B: Active Causal Relevance Discovery
============================================================

Purpose
-------
Stage 2A exposed a structural limitation of passive same-data linear regression:
with the same ridge design matrix, fitting a full next-observation model and then
projecting to task features is algebraically equivalent to fitting those task
features directly. Therefore a passive linear benchmark cannot, by itself,
establish a sample-complexity advantage for task-relative prediction.

This Stage 2B harness activates the second half of the proposed foundation:
*decision-directed experimentation*. The learner is NOT told which upstream
variables are relevant to the declared terminal task family. It is given only:

  1. a known candidate causal skeleton (possible parent sets),
  2. the declared terminal target variables / task family,
  3. a resettable one-step intervention oracle.

It must discover the actual causal ancestors of those targets by intervention.
The main structured regime has bounded candidate in-degree p while total system
size D grows. The no-locality control makes every variable a possible parent,
where the hoped-for nuisance-independence should disappear.

What this can establish
-----------------------
A positive result can show that, under explicit causal sparsity/locality and
beta-min identifiability assumptions, task-relative active learning can pay for
causal ancestors of the task family rather than for the whole environment.
It does NOT establish superiority over reinforcement learning as a whole.
Task-family-aware reward-agnostic / causal RL methods may exploit the same
structure. Strong RL comparisons remain a later stage.

Experiment interface
--------------------
The environment is a layered linear structural causal model:

    x_{h+1,j} = a_{h,j} x_{h,parent(h,j)} + noise

Each child has exactly one true parent, hidden among a known candidate set.
The task family is any linear scalarization of q known terminal target nodes.
At deployment the agent chooses a signed source intervention +/-e_i at layer 0.
The best source depends on the unknown causal paths to the terminal targets.

During exploration, an experiment chooses (layer h, source coordinate i),
sets x_h = +/-e_i, and observes the next-layer vector. Multiplying by the
intervention sign yields a noisy unbiased observation of column i of A_h.
Every learner has the same intervention type. Different learners may choose
where to spend interactions.

Main methods
------------
Capability-Frontier:
    Backward from terminal task nodes. At each layer, test only candidate parents
    of the currently relevant frontier. Expand the frontier only through edges
    that are statistically detected. Estimate only task-ancestor coefficients.

Full-ID:
    Query every source coordinate at every layer to estimate the whole causal
    model under the same candidate skeleton.

Direct-Source:
    Ignore recursive structure. Directly estimate terminal task consequences of
    every possible source action. This is task-aware but pays for all D sources.

Random-SameBudget:
    Spend exactly Capability's interaction budget uniformly over one-step
    interventions, then plan from the resulting partial model.

Oracle-Ancestor:
    Privileged lower bound. Knows the true task ancestry and estimates only it.

Frozen falsification logic
--------------------------
E0: Capability fails to achieve high deployment competence in the local sparse
    regime. Stop / debug / reject implementation.
E1: Capability succeeds and its interactions stay nearly flat as D grows.
E2: Full-ID and Direct-Source grow materially with D while Capability does not,
    AND the advantage collapses in the no-locality control.
E3: The weak-edge experiment follows the predicted ~1/beta^2 information law
    and task-family cost grows with actual task ancestry rather than total D.

Even E3 does NOT mean "better than RL". It licenses a later same-assumption
comparison against faithful reward-agnostic / causal RL baselines.

Dependencies
------------
Required: numpy
Optional: psutil

Usage
-----
    python capability_track_stage2b_active_causal.py --profile standard
    python capability_track_stage2b_active_causal.py --profile deep

Outputs one TXT report with aggregate tables, automatic evidence level, frozen
kill criteria, theorem diagnostics, and raw JSONL records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

EPS = 1e-12
SCRIPT_VERSION = "2026-09-06-stage2b-active-causal-v1"

# -----------------------------------------------------------------------------
# Reproducibility / utilities
# -----------------------------------------------------------------------------


def stable_seed(base: int, *parts: Any) -> int:
    h = hashlib.sha256(str(base).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFF


def rng_for(base: int, *parts: Any) -> Generator:
    return np.random.default_rng(stable_seed(base, *parts))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return None


def fmt(x: Any, digits: int = 5) -> str:
    try:
        y = float(x)
    except Exception:
        return str(x)
    if math.isnan(y):
        return "nan"
    if math.isinf(y):
        return "INF" if y > 0 else "-INF"
    return f"{y:.{digits}f}"


def ci95(xs: Sequence[float]) -> Tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(xs, dtype=float)
    m = float(np.mean(a))
    if len(a) <= 1:
        return m, m, m
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, m - 1.96 * se, m + 1.96 * se


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y > 0 and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    lx = np.log([p[0] for p in pairs])
    ly = np.log([p[1] for p in pairs])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 1.0
    return float(slope), float(r2)


def unit_weights(rng: Generator, n: int, q: int) -> np.ndarray:
    w = rng.normal(size=(n, q))
    w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), EPS)
    return w


# -----------------------------------------------------------------------------
# Layered sparse causal system
# -----------------------------------------------------------------------------


@dataclass
class LayeredCausalEnv:
    D: int
    H: int
    q_task: int
    candidate_degree: int
    beta_min: float
    beta_max: float
    noise_sigma: float
    locality: str  # "local" or "global"
    seed: int

    def __post_init__(self) -> None:
        if self.D < 2:
            raise ValueError("D must be >=2")
        if not (1 <= self.q_task <= self.D):
            raise ValueError("q_task out of range")
        if not (0 < self.beta_min <= self.beta_max):
            raise ValueError("invalid beta range")
        if self.locality not in {"local", "global"}:
            raise ValueError("locality must be local/global")
        rng = np.random.default_rng(self.seed)
        self.targets = np.asarray(rng.choice(self.D, size=self.q_task, replace=False), dtype=int)
        # Candidate parent sets and one true parent/weight for every child at every layer.
        self.candidates: List[List[np.ndarray]] = []
        self.true_parent: List[np.ndarray] = []
        self.true_weight: List[np.ndarray] = []
        for h in range(self.H):
            cand_h: List[np.ndarray] = []
            par_h = np.empty(self.D, dtype=int)
            wt_h = np.empty(self.D, dtype=float)
            for j in range(self.D):
                if self.locality == "global":
                    cand = np.arange(self.D, dtype=int)
                else:
                    p = min(self.D, max(1, self.candidate_degree))
                    cand = np.asarray(rng.choice(self.D, size=p, replace=False), dtype=int)
                parent = int(rng.choice(cand))
                mag = float(rng.uniform(self.beta_min, self.beta_max))
                sign = -1.0 if rng.random() < 0.5 else 1.0
                cand_h.append(cand)
                par_h[j] = parent
                wt_h[j] = sign * mag
            self.candidates.append(cand_h)
            self.true_parent.append(par_h)
            self.true_weight.append(wt_h)
        # Reverse skeleton: candidate children reachable from each source. This is
        # shared structural knowledge, not true-edge information, and avoids O(D^2)
        # scans in the local sparse regime.
        self.candidate_children: List[List[np.ndarray]] = []
        for h in range(self.H):
            rev: List[List[int]] = [[] for _ in range(self.D)]
            for child in range(self.D):
                for src in self.candidates[h][child]:
                    rev[int(src)].append(child)
            self.candidate_children.append([np.asarray(x, dtype=int) for x in rev])
        self.true_G = self._compute_true_task_influence()
        self.true_frontiers = self._compute_true_frontiers()

    def candidate_parents(self, h: int, child: int) -> np.ndarray:
        return self.candidates[h][child]

    def candidate_children_for_source(self, h: int, source: int) -> np.ndarray:
        return self.candidate_children[h][source]

    def coefficient(self, h: int, child: int, source: int) -> float:
        return float(self.true_weight[h][child]) if int(self.true_parent[h][child]) == int(source) else 0.0

    def true_column(self, h: int, source: int) -> np.ndarray:
        out = np.zeros(self.D, dtype=float)
        par = self.true_parent[h]
        rows = np.flatnonzero(par == int(source))
        out[rows] = self.true_weight[h][rows]
        return out

    def _compute_true_frontiers(self) -> List[np.ndarray]:
        fronts: List[np.ndarray] = [np.zeros(0, dtype=int) for _ in range(self.H + 1)]
        fronts[self.H] = np.unique(self.targets)
        for h in range(self.H - 1, -1, -1):
            fronts[h] = np.unique(self.true_parent[h][fronts[h + 1]])
        return fronts

    def _compute_true_task_influence(self) -> np.ndarray:
        # G_h maps x_h to terminal task features. Start with target selection at H.
        G = np.zeros((self.q_task, self.D), dtype=float)
        for k, j in enumerate(self.targets):
            G[k, int(j)] = 1.0
        for h in range(self.H - 1, -1, -1):
            prev = np.zeros_like(G)
            for child in range(self.D):
                g = G[:, child]
                if np.any(np.abs(g) > 0):
                    p = int(self.true_parent[h][child])
                    prev[:, p] += g * float(self.true_weight[h][child])
            G = prev
        return G

    def terminal_noise_cov(self) -> np.ndarray:
        """Exact q-task covariance from process noise, computed sparsely.

        Noise injected into x_{h+1} reaches the terminal task through the current
        task influence G_{h+1}. Hence its contribution is sigma^2 G G^T. This
        avoids constructing or multiplying D x D covariance matrices.
        """
        G = np.zeros((self.q_task, self.D), dtype=float)
        for k, j in enumerate(self.targets):
            G[k, int(j)] = 1.0
        cov = np.zeros((self.q_task, self.q_task), dtype=float)
        for h in range(self.H - 1, -1, -1):
            # Process noise is added after A_h, i.e. directly to x_{h+1}.
            cov += (self.noise_sigma ** 2) * (G @ G.T)
            prev = np.zeros_like(G)
            for child in range(self.D):
                g = G[:, child]
                if np.any(np.abs(g) > 0):
                    src = int(self.true_parent[h][child])
                    prev[:, src] += g * float(self.true_weight[h][child])
            G = prev
        return cov


# -----------------------------------------------------------------------------
# Intervention sufficient statistics
# -----------------------------------------------------------------------------


@dataclass
class ColumnStats:
    n: int
    sum_vec: np.ndarray

    @classmethod
    def empty(cls, D: int) -> "ColumnStats":
        return cls(0, np.zeros(D, dtype=float))

    def add(self, true_col: np.ndarray, batch: int, sigma: float, amp: float, rng: Generator) -> None:
        if batch <= 0:
            return
        # sign*y/amp has mean true_col and iid Gaussian noise sigma/amp.
        noise = rng.normal(scale=(sigma / amp) * math.sqrt(batch), size=true_col.shape[0])
        self.sum_vec += batch * true_col + noise
        self.n += int(batch)

    @property
    def mean(self) -> np.ndarray:
        if self.n <= 0:
            return np.zeros_like(self.sum_vec)
        return self.sum_vec / float(self.n)


@dataclass
class LearnedCausalModel:
    H: int
    D: int
    q_task: int
    targets: np.ndarray
    edge_maps: List[Dict[Tuple[int, int], float]]

    def task_influence(self) -> np.ndarray:
        G = np.zeros((self.q_task, self.D), dtype=float)
        for k, j in enumerate(self.targets):
            G[k, int(j)] = 1.0
        for h in range(self.H - 1, -1, -1):
            prev = np.zeros_like(G)
            em = self.edge_maps[h]
            # Iterate only estimated nonzero/retained edges.
            for (child, source), a in em.items():
                if abs(a) <= 0:
                    continue
                prev[:, int(source)] += G[:, int(child)] * float(a)
            G = prev
        return G


@dataclass
class LearnerResult:
    name: str
    interactions: int
    model: Optional[LearnedCausalModel]
    direct_G: Optional[np.ndarray]
    diagnostics: Dict[str, Any]

    def influence(self) -> np.ndarray:
        if self.direct_G is not None:
            return self.direct_G
        if self.model is None:
            raise RuntimeError("no learned object")
        return self.model.task_influence()


# -----------------------------------------------------------------------------
# Confidence calculations
# -----------------------------------------------------------------------------


def confidence_logterm(env: LayeredCausalEnv, delta: float) -> float:
    # Conservative familywise bound over all candidate coefficients in the known skeleton.
    if env.locality == "global":
        m = env.H * env.D * env.D
    else:
        m = env.H * env.D * min(env.D, env.candidate_degree)
    return math.log(max(4.0, 2.0 * m / max(delta, 1e-12)))


def required_n_for_radius(sigma: float, amp: float, radius: float, logterm: float) -> int:
    # Gaussian tail: |mean-mu| <= sigma/amp * sqrt(2 logterm / n).
    return int(math.ceil(2.0 * (sigma / amp) ** 2 * logterm / max(radius ** 2, 1e-16)))


def sample_column_rows(env: LayeredCausalEnv, h: int, source: int, rows: np.ndarray, n: int, amp: float, rng: Generator) -> np.ndarray:
    """Sample the sufficient-statistic mean only on requested child rows.

    A physical intervention observes the full next vector. We simulate only rows
    that the learner is permitted/needs to use because unreferenced Gaussian noise
    has no effect on inference but can dominate runtime in large-D sweeps.
    """
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return np.zeros(0, dtype=float)
    mu = np.asarray([env.coefficient(h, int(j), source) for j in rows], dtype=float)
    return mu + rng.normal(scale=env.noise_sigma / (amp * math.sqrt(max(1, n))), size=len(rows))


def combine_means(mean1: np.ndarray, n1: int, mean2: np.ndarray, n2: int) -> np.ndarray:
    if n2 <= 0:
        return mean1
    return (n1 * mean1 + n2 * mean2) / float(n1 + n2)


# -----------------------------------------------------------------------------
# Learners
# -----------------------------------------------------------------------------


def capability_frontier(
    env: LayeredCausalEnv,
    rng: Generator,
    delta: float,
    amp: float,
    coef_tol: float,
) -> LearnerResult:
    """Backward task-relative causal screening with unknown true ancestry."""
    logt = confidence_logterm(env, delta)
    # beta-min classification is sound once coordinate error <= beta_min/4.
    n_class = required_n_for_radius(env.noise_sigma, amp, env.beta_min / 4.0, logt)
    n_est = required_n_for_radius(env.noise_sigma, amp, coef_tol, logt)
    n_est = max(n_est, n_class)

    tested_keys: set[Tuple[int, int]] = set()
    edge_maps: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(env.H)]
    frontiers: List[np.ndarray] = [np.zeros(0, dtype=int) for _ in range(env.H + 1)]
    frontiers[env.H] = np.unique(env.targets)
    interactions = 0
    tested_sources: List[int] = []
    active_sources: List[int] = []
    false_pos = 0
    false_neg = 0

    for h in range(env.H - 1, -1, -1):
        next_front = frontiers[h + 1]
        cand_sources: set[int] = set()
        for child in next_front:
            cand_sources.update(int(x) for x in env.candidate_parents(h, int(child)))
        found: List[int] = []
        for source in sorted(cand_sources):
            key = (h, source)
            class_mean = sample_column_rows(env, h, source, next_front, n_class, amp, rng)
            interactions += n_class
            tested_keys.add(key)
            vals = class_mean
            detected = bool(np.max(np.abs(vals)) > env.beta_min / 2.0) if len(vals) else False
            truth = any(env.coefficient(h, int(child), source) != 0.0 for child in next_front)
            if detected and not truth:
                false_pos += 1
            if (not detected) and truth:
                false_neg += 1
            if not detected:
                tested_sources.append(source)
                continue
            # Active candidate: collect enough additional samples for coefficient precision.
            if n_est > n_class:
                extra = n_est - n_class
                extra_mean = sample_column_rows(env, h, source, next_front, extra, amp, rng)
                vals = combine_means(class_mean, n_class, extra_mean, extra)
                interactions += extra
            else:
                vals = class_mean
            # Retain only edges supported by beta-min threshold. This protects the
            # recursive product from accumulating nuisance noise.
            any_retained = False
            for child, val in zip(next_front, vals):
                if abs(float(val)) > env.beta_min / 2.0:
                    edge_maps[h][(int(child), source)] = float(val)
                    any_retained = True
            if any_retained:
                found.append(source)
                active_sources.append(source)
            tested_sources.append(source)
        frontiers[h] = np.asarray(sorted(set(found)), dtype=int)

    model = LearnedCausalModel(env.H, env.D, env.q_task, env.targets.copy(), edge_maps)
    true_sizes = [int(len(x)) for x in env.true_frontiers]
    learned_sizes = [int(len(x)) for x in frontiers]
    return LearnerResult(
        name="Capability-Frontier",
        interactions=interactions,
        model=model,
        direct_G=None,
        diagnostics={
            "n_class": n_class,
            "n_est": n_est,
            "tested_unique_layer_sources": len(tested_keys),
            "true_frontier_sizes": true_sizes,
            "learned_frontier_sizes": learned_sizes,
            "false_positive_sources": false_pos,
            "false_negative_sources": false_neg,
        },
    )


def full_id(
    env: LayeredCausalEnv,
    rng: Generator,
    delta: float,
    amp: float,
    coef_tol: float,
) -> LearnerResult:
    """Estimate the whole causal system under the known candidate skeleton."""
    logt = confidence_logterm(env, delta)
    n_est = required_n_for_radius(env.noise_sigma, amp, min(coef_tol, env.beta_min / 4.0), logt)
    edge_maps: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(env.H)]
    interactions = 0
    queried = 0
    for h in range(env.H):
        # One source intervention observes the full next vector. Query every source
        # that appears in at least one candidate-parent set. In these random skeletons
        # this is usually all D, but use the exact known skeleton for fairness.
        sources: set[int] = set()
        for child in range(env.D):
            sources.update(int(x) for x in env.candidate_parents(h, child))
        for source in sorted(sources):
            children = env.candidate_children_for_source(h, source)
            mean = sample_column_rows(env, h, source, children, n_est, amp, rng)
            interactions += n_est
            queried += 1
            # Use known candidate skeleton and beta-min threshold to suppress zeros.
            for child, val0 in zip(children, mean):
                val = float(val0)
                if abs(val) > env.beta_min / 2.0:
                    edge_maps[h][(int(child), source)] = val
    model = LearnedCausalModel(env.H, env.D, env.q_task, env.targets.copy(), edge_maps)
    return LearnerResult(
        name="Full-ID",
        interactions=interactions,
        model=model,
        direct_G=None,
        diagnostics={"n_est": n_est, "queried_layer_sources": queried},
    )


def random_same_budget(
    env: LayeredCausalEnv,
    rng: Generator,
    budget: int,
    delta: float,
    amp: float,
) -> LearnerResult:
    """Uniform one-step interventions with the exact Capability interaction budget."""
    pairs = [(h, i) for h in range(env.H) for i in range(env.D)]
    if budget <= 0:
        return LearnerResult("Random-SameBudget", 0, LearnedCausalModel(env.H, env.D, env.q_task, env.targets.copy(), [dict() for _ in range(env.H)]), None, {})
    base = budget // len(pairs)
    rem = budget % len(pairs)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    extra_set = set(int(x) for x in order[:rem])
    edge_maps: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(env.H)]
    for idx, (h, source) in enumerate(pairs):
        n = base + (1 if idx in extra_set else 0)
        if n <= 0:
            continue
        children = env.candidate_children_for_source(h, source)
        mean = sample_column_rows(env, h, source, children, n, amp, rng)
        # Without enough samples, threshold remains intentionally conservative.
        # Use same beta-min midpoint as the structured methods.
        for child, val0 in zip(children, mean):
            val = float(val0)
            if abs(val) > env.beta_min / 2.0:
                edge_maps[h][(int(child), source)] = val
    model = LearnedCausalModel(env.H, env.D, env.q_task, env.targets.copy(), edge_maps)
    return LearnerResult(
        name="Random-SameBudget",
        interactions=budget,
        model=model,
        direct_G=None,
        diagnostics={"mean_samples_per_layer_source": budget / max(1, len(pairs))},
    )


def oracle_ancestor(
    env: LayeredCausalEnv,
    rng: Generator,
    delta: float,
    amp: float,
    coef_tol: float,
) -> LearnerResult:
    """Privileged lower bound: knows exact task ancestry, not a fair competitor."""
    logt = confidence_logterm(env, delta)
    n_est = required_n_for_radius(env.noise_sigma, amp, coef_tol, logt)
    edge_maps: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(env.H)]
    interactions = 0
    queried = 0
    for h in range(env.H - 1, -1, -1):
        next_front = env.true_frontiers[h + 1]
        sources = np.unique(env.true_parent[h][next_front])
        for source in sources:
            vals = sample_column_rows(env, h, int(source), next_front, n_est, amp, rng)
            interactions += n_est
            queried += 1
            for child, val0 in zip(next_front, vals):
                if env.coefficient(h, int(child), int(source)) != 0.0:
                    edge_maps[h][(int(child), int(source))] = float(val0)
    model = LearnedCausalModel(env.H, env.D, env.q_task, env.targets.copy(), edge_maps)
    return LearnerResult(
        name="Oracle-Ancestor",
        interactions=interactions,
        model=model,
        direct_G=None,
        diagnostics={"n_est": n_est, "queried_layer_sources": queried},
    )


def direct_source(
    env: LayeredCausalEnv,
    rng: Generator,
    delta: float,
    direct_tol: float,
) -> LearnerResult:
    """Task-aware nonrecursive baseline: estimate terminal consequence of every source.

    One direct sample is a full H-step rollout, therefore interaction cost is H
    primitive transitions per sample.
    """
    Sigma = env.terminal_noise_cov()
    # Conservative scalar variance proxy across task coordinates.
    var = float(max(np.max(np.diag(Sigma)), EPS))
    logt = math.log(max(4.0, 2.0 * env.D * env.q_task / max(delta, 1e-12)))
    n = int(math.ceil(2.0 * var * logt / max(direct_tol ** 2, 1e-16)))
    Ghat = np.empty_like(env.true_G)
    cov_mean = Sigma / max(1, n)
    # Numerical PSD cleanup.
    cov_mean = 0.5 * (cov_mean + cov_mean.T) + 1e-12 * np.eye(env.q_task)
    for i in range(env.D):
        Ghat[:, i] = env.true_G[:, i] + rng.multivariate_normal(np.zeros(env.q_task), cov_mean)
    return LearnerResult(
        name="Direct-Source",
        interactions=int(env.D * n * env.H),
        model=None,
        direct_G=Ghat,
        diagnostics={"rollouts_per_source": n, "primitive_steps_per_rollout": env.H},
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@dataclass
class EvalResult:
    success_rate: float
    mean_regret: float
    median_regret: float
    influence_mse: float


def evaluate_influence(env: LayeredCausalEnv, Ghat: np.ndarray, weights: np.ndarray, epsilon: float) -> EvalResult:
    G = env.true_G
    regrets: List[float] = []
    for w in weights:
        true_scores = np.asarray(w @ G, dtype=float)
        pred_scores = np.asarray(w @ Ghat, dtype=float)
        best = float(np.max(np.abs(true_scores)))
        if best <= 1e-10:
            regrets.append(0.0)
            continue
        i = int(np.argmax(np.abs(pred_scores)))
        sgn = 1.0 if pred_scores[i] >= 0 else -1.0
        achieved = sgn * float(true_scores[i])
        # Normalize to [0,1] where choosing the opposite sign of the best source is 1.
        reg = (best - achieved) / (2.0 * best)
        regrets.append(float(np.clip(reg, 0.0, 1.0)))
    return EvalResult(
        success_rate=float(np.mean(np.asarray(regrets) <= epsilon)),
        mean_regret=float(np.mean(regrets)),
        median_regret=float(np.median(regrets)),
        influence_mse=float(np.mean((Ghat - G) ** 2)),
    )


# -----------------------------------------------------------------------------
# Trial / suites
# -----------------------------------------------------------------------------


def run_trial(
    *,
    base_seed: int,
    tag: str,
    trial: int,
    D: int,
    H: int,
    q_task: int,
    candidate_degree: int,
    beta_min: float,
    beta_max: float,
    noise_sigma: float,
    locality: str,
    delta: float,
    amp: float,
    coef_tol: float,
    direct_tol: float,
    epsilon_regret: float,
    n_task_weights: int,
) -> List[Dict[str, Any]]:
    env = LayeredCausalEnv(
        D=D,
        H=H,
        q_task=q_task,
        candidate_degree=candidate_degree,
        beta_min=beta_min,
        beta_max=beta_max,
        noise_sigma=noise_sigma,
        locality=locality,
        seed=stable_seed(base_seed, tag, "env", trial),
    )
    weights = unit_weights(rng_for(base_seed, tag, "weights", trial), n_task_weights, q_task)

    methods: List[LearnerResult] = []
    cap = capability_frontier(env, rng_for(base_seed, tag, "cap", trial), delta, amp, coef_tol)
    methods.append(cap)
    methods.append(full_id(env, rng_for(base_seed, tag, "full", trial), delta, amp, coef_tol))
    methods.append(direct_source(env, rng_for(base_seed, tag, "direct", trial), delta, direct_tol))
    methods.append(random_same_budget(env, rng_for(base_seed, tag, "random", trial), cap.interactions, delta, amp))
    methods.append(oracle_ancestor(env, rng_for(base_seed, tag, "oracle", trial), delta, amp, coef_tol))

    recs: List[Dict[str, Any]] = []
    for res in methods:
        try:
            Ghat = res.influence()
            ev = evaluate_influence(env, Ghat, weights, epsilon_regret)
            recs.append({
                "suite": tag,
                "trial": trial,
                "method": res.name,
                "D": D,
                "H": H,
                "q_task": q_task,
                "candidate_degree": candidate_degree,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "noise_sigma": noise_sigma,
                "locality": locality,
                "interactions": int(res.interactions),
                "success_rate": ev.success_rate,
                "mean_regret": ev.mean_regret,
                "median_regret": ev.median_regret,
                "influence_mse": ev.influence_mse,
                "true_frontier_sizes": [int(len(x)) for x in env.true_frontiers],
                "diagnostics": res.diagnostics,
                "rss_mb": rss_mb(),
            })
        except Exception as exc:
            recs.append({
                "suite": tag, "trial": trial, "method": res.name, "D": D,
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
    return recs


def run_dimension_suite(base: int, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for locality in ["local", "global"]:
        grid = cfg["D_grid"] if locality == "local" else cfg["D_grid_global"]
        for D in grid:
            tag = f"dimension:{locality}:D={D}"
            print(f"[Stage2B] {tag}", flush=True)
            for tr in range(cfg["trials"]):
                recs.extend(run_trial(
                    base_seed=base, tag=tag, trial=tr, D=int(D), H=cfg["H"], q_task=cfg["q_task"],
                    candidate_degree=cfg["candidate_degree"], beta_min=cfg["beta_min"], beta_max=cfg["beta_max"],
                    noise_sigma=cfg["noise_sigma"], locality=locality, delta=cfg["delta"], amp=cfg["amp"],
                    coef_tol=cfg["coef_tol"], direct_tol=cfg["direct_tol"], epsilon_regret=cfg["epsilon_regret"],
                    n_task_weights=cfg["task_weights"],
                ))
            print(f"[Stage2B] {tag} complete", flush=True)
    return recs


def run_gap_suite(base: int, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    D = cfg["gap_D"]
    for beta in cfg["beta_grid"]:
        tag = f"gap:beta={beta}"
        print(f"[Stage2B] {tag}", flush=True)
        for tr in range(cfg["trials_gap"]):
            recs.extend(run_trial(
                base_seed=base, tag=tag, trial=tr, D=D, H=cfg["H"], q_task=cfg["q_task"],
                candidate_degree=cfg["candidate_degree"], beta_min=float(beta), beta_max=max(float(beta) * 1.25, float(beta) + 0.05),
                noise_sigma=cfg["noise_sigma"], locality="local", delta=cfg["delta"], amp=cfg["amp"],
                coef_tol=min(cfg["coef_tol"], float(beta) / 5.0), direct_tol=cfg["direct_tol"],
                epsilon_regret=cfg["epsilon_regret"], n_task_weights=cfg["task_weights"],
            ))
        print(f"[Stage2B] {tag} complete", flush=True)
    return recs


def run_task_suite(base: int, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    D = cfg["task_D"]
    for q in cfg["q_grid"]:
        if q > D:
            continue
        tag = f"tasks:q={q}"
        print(f"[Stage2B] {tag}", flush=True)
        for tr in range(cfg["trials_task"]):
            recs.extend(run_trial(
                base_seed=base, tag=tag, trial=tr, D=D, H=cfg["H"], q_task=int(q),
                candidate_degree=cfg["candidate_degree"], beta_min=cfg["beta_min"], beta_max=cfg["beta_max"],
                noise_sigma=cfg["noise_sigma"], locality="local", delta=cfg["delta"], amp=cfg["amp"],
                coef_tol=cfg["coef_tol"], direct_tol=cfg["direct_tol"], epsilon_regret=cfg["epsilon_regret"],
                n_task_weights=cfg["task_weights"],
            ))
        print(f"[Stage2B] {tag} complete", flush=True)
    return recs


# -----------------------------------------------------------------------------
# Aggregation / automatic evidence
# -----------------------------------------------------------------------------


def valid(recs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [r for r in recs if "error" not in r]


def group(recs: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for r in recs:
        k = tuple(r.get(x) for x in keys)
        out.setdefault(k, []).append(r)
    return out


def summarize_dimension(recs: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    good = valid(recs)
    g = group(good, ["locality", "D", "method"])
    lines = ["Dimension / locality suite", "locality\tD\tmethod\tinteractions\tsuccess\tregret\tinfluence_mse\tfrontier0"]
    metrics: Dict[str, Any] = {"means": {}, "slopes": {}}
    for key in sorted(g, key=lambda x: (str(x[0]), int(x[1]), str(x[2]))):
        loc, D, method = key
        rs = g[key]
        inter = float(np.mean([r["interactions"] for r in rs]))
        succ = float(np.mean([r["success_rate"] for r in rs]))
        reg = float(np.mean([r["mean_regret"] for r in rs]))
        mse = float(np.mean([r["influence_mse"] for r in rs]))
        f0 = float(np.mean([r["true_frontier_sizes"][0] for r in rs]))
        lines.append(f"{loc}\t{D}\t{method}\t{inter:.2f}\t{succ:.4f}\t{reg:.5f}\t{mse:.6g}\t{f0:.2f}")
        metrics["means"][str(key)] = {"interactions": inter, "success": succ, "regret": reg, "mse": mse, "frontier0": f0}
    local_Ds = sorted({int(D) for (loc, D, method) in g if loc == "local"})
    if local_Ds:
        Dmax = local_Ds[-1]
        metrics["largest_local_D"] = Dmax
        metrics["largest_local"] = {}
        for method in ["Capability-Frontier", "Full-ID", "Direct-Source", "Random-SameBudget", "Oracle-Ancestor"]:
            rs = g.get(("local", Dmax, method), [])
            if rs:
                metrics["largest_local"][method] = {
                    "success": float(np.mean([r["success_rate"] for r in rs])),
                    "interactions": float(np.mean([r["interactions"] for r in rs])),
                }
    for loc in ["local", "global"]:
        for method in ["Capability-Frontier", "Full-ID", "Direct-Source", "Random-SameBudget", "Oracle-Ancestor"]:
            xs, ys = [], []
            for (l, D, m), rs in g.items():
                if l == loc and m == method:
                    xs.append(float(D)); ys.append(float(np.mean([r["interactions"] for r in rs])))
            order = np.argsort(xs) if xs else []
            xs2 = [xs[i] for i in order] if len(xs) else []
            ys2 = [ys[i] for i in order] if len(xs) else []
            s, r2 = loglog_slope(xs2, ys2)
            metrics["slopes"][f"{loc}:{method}"] = {"slope": s, "r2": r2, "points": len(xs2)}
    lines.append("")
    lines.append("Log-log interaction slopes vs D")
    for k, v in sorted(metrics["slopes"].items()):
        lines.append(f"{k}\tslope={fmt(v['slope'])}\tR2={fmt(v['r2'])}\tpoints={v['points']}")
    return "\n".join(lines), metrics


def summarize_gap(recs: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    good = valid(recs)
    g = group(good, ["beta_min", "method"])
    lines = ["Weak-edge / identifiability suite", "beta\tmethod\tinteractions\tsuccess\tregret"]
    metrics: Dict[str, Any] = {"means": {}, "slopes": {}}
    for key in sorted(g, key=lambda x: (float(x[0]), str(x[1]))):
        beta, method = key; rs = g[key]
        inter = float(np.mean([r["interactions"] for r in rs]))
        succ = float(np.mean([r["success_rate"] for r in rs]))
        reg = float(np.mean([r["mean_regret"] for r in rs]))
        lines.append(f"{beta:.4f}\t{method}\t{inter:.2f}\t{succ:.4f}\t{reg:.5f}")
        metrics["means"][str(key)] = {"interactions": inter, "success": succ, "regret": reg}
    for method in ["Capability-Frontier", "Full-ID", "Oracle-Ancestor"]:
        xs, ys = [], []
        for (beta, m), rs in g.items():
            if m == method:
                xs.append(float(beta)); ys.append(float(np.mean([r["interactions"] for r in rs])))
        s, r2 = loglog_slope(xs, ys)
        metrics["slopes"][method] = {"slope": s, "r2": r2, "points": len(xs)}
    lines.append("")
    lines.append("Log-log interaction slopes vs beta (information prediction is near -2 when classification dominates)")
    for k, v in sorted(metrics["slopes"].items()):
        lines.append(f"{k}\tslope={fmt(v['slope'])}\tR2={fmt(v['r2'])}\tpoints={v['points']}")
    return "\n".join(lines), metrics


def summarize_tasks(recs: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    good = valid(recs)
    g = group(good, ["q_task", "method"])
    lines = ["Task-family richness suite", "q\tmethod\tinteractions\tsuccess\tregret\tfrontier0"]
    metrics: Dict[str, Any] = {"means": {}, "slopes": {}}
    for key in sorted(g, key=lambda x: (int(x[0]), str(x[1]))):
        q, method = key; rs = g[key]
        inter = float(np.mean([r["interactions"] for r in rs]))
        succ = float(np.mean([r["success_rate"] for r in rs]))
        reg = float(np.mean([r["mean_regret"] for r in rs]))
        f0 = float(np.mean([r["true_frontier_sizes"][0] for r in rs]))
        lines.append(f"{q}\t{method}\t{inter:.2f}\t{succ:.4f}\t{reg:.5f}\t{f0:.2f}")
        metrics["means"][str(key)] = {"interactions": inter, "success": succ, "regret": reg, "frontier0": f0}
    for method in ["Capability-Frontier", "Full-ID", "Direct-Source", "Oracle-Ancestor"]:
        xs, ys = [], []
        for (q, m), rs in g.items():
            if m == method:
                xs.append(float(q)); ys.append(float(np.mean([r["interactions"] for r in rs])))
        s, r2 = loglog_slope(xs, ys)
        metrics["slopes"][method] = {"slope": s, "r2": r2, "points": len(xs)}
    lines.append("")
    lines.append("Log-log interaction slopes vs q")
    for k, v in sorted(metrics["slopes"].items()):
        lines.append(f"{k}\tslope={fmt(v['slope'])}\tR2={fmt(v['r2'])}\tpoints={v['points']}")
    return "\n".join(lines), metrics


def automatic_evidence(dim: Mapping[str, Any], gap: Mapping[str, Any], task: Mapping[str, Any]) -> Tuple[int, List[str]]:
    notes: List[str] = []
    means = dim.get("means", {})
    # Local capability competence across largest D.
    local_ds = []
    for k, v in means.items():
        if "'local'" in k and "'Capability-Frontier'" in k:
            local_ds.append((k, v))
    if not local_ds:
        return 0, ["FAIL: no local Capability records"]
    # Parse D from tuple string conservatively via stored values scan instead of eval.
    min_succ = min(float(v["success"]) for _, v in local_ds)
    if min_succ < 0.90:
        return 0, [f"FAIL E0: Capability local-regime minimum success={min_succ:.3f} < 0.90"]
    level = 1
    notes.append(f"PASS E1 competence: Capability local minimum success={min_succ:.3f}")

    slopes = dim.get("slopes", {})
    cap_local = float(slopes.get("local:Capability-Frontier", {}).get("slope", float("nan")))
    full_local = float(slopes.get("local:Full-ID", {}).get("slope", float("nan")))
    direct_local = float(slopes.get("local:Direct-Source", {}).get("slope", float("nan")))
    cap_global = float(slopes.get("global:Capability-Frontier", {}).get("slope", float("nan")))
    if math.isfinite(cap_local) and math.isfinite(full_local) and math.isfinite(direct_local):
        if cap_local < 0.30 and full_local > 0.70 and direct_local > 0.70:
            # No-locality control should force Capability toward D scaling.
            largest = dim.get("largest_local", {})
            cap_s = float(largest.get("Capability-Frontier", {}).get("success", float("nan")))
            rnd_s = float(largest.get("Random-SameBudget", {}).get("success", float("nan")))
            active_gap_ok = math.isfinite(cap_s) and math.isfinite(rnd_s) and (cap_s - rnd_s >= 0.10 or rnd_s < 0.80)
            if math.isfinite(cap_global) and cap_global > 0.60 and active_gap_ok:
                level = 2
                notes.append(
                    f"PASS E2: local slopes cap={cap_local:.3f}, full={full_local:.3f}, direct={direct_local:.3f}; "
                    f"global-control cap={cap_global:.3f}; largest-D same-budget success cap={cap_s:.3f}, random={rnd_s:.3f}"
                )
            elif not active_gap_ok:
                notes.append(f"NO E2: same-budget uniform exploration remained too competitive at largest local D (cap={cap_s:.3f}, random={rnd_s:.3f})")
            else:
                notes.append(f"NO E2: no-locality control did not restore D scaling (cap_global={cap_global:.3f})")
        else:
            notes.append(f"NO E2: local scaling gap insufficient (cap={cap_local:.3f}, full={full_local:.3f}, direct={direct_local:.3f})")

    if level >= 2:
        bg = float(gap.get("slopes", {}).get("Capability-Frontier", {}).get("slope", float("nan")))
        tq = float(task.get("slopes", {}).get("Capability-Frontier", {}).get("slope", float("nan")))
        if math.isfinite(bg) and -2.6 <= bg <= -1.3 and math.isfinite(tq) and 0.4 <= tq <= 1.6:
            level = 3
            notes.append(f"PASS E3: beta slope={bg:.3f}, task-richness slope={tq:.3f}")
        else:
            notes.append(f"NO E3: beta slope={bg:.3f}, task-richness slope={tq:.3f}")
    return level, notes


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def metadata() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "psutil": getattr(psutil, "__version__", None) if psutil is not None else None,
        "rss_mb": rss_mb(),
    }


def profiles() -> Dict[str, Dict[str, Any]]:
    common = dict(
        H=4,
        q_task=2,
        candidate_degree=6,
        beta_min=0.55,
        beta_max=0.85,
        noise_sigma=0.35,
        delta=0.05,
        amp=1.0,
        coef_tol=0.10,
        direct_tol=0.10,
        epsilon_regret=0.05,
        task_weights=128,
        gap_D=128,
        task_D=512,
    )
    return {
        "smoke": dict(common, D_grid=[16, 32, 64], D_grid_global=[16, 32], trials=5,
                      beta_grid=[0.35, 0.55, 0.75], trials_gap=5, q_grid=[1, 2, 4], trials_task=5),
        "standard": dict(common, D_grid=[16, 32, 64, 128, 256, 512], D_grid_global=[16, 32, 64, 128], trials=30,
                         beta_grid=[0.30, 0.40, 0.55, 0.75], trials_gap=30,
                         q_grid=[1, 2, 4, 8, 16], trials_task=30),
        "deep": dict(common, D_grid=[16, 32, 64, 128, 256, 512, 1024], D_grid_global=[16, 32, 64, 128, 256], trials=80,
                     beta_grid=[0.25, 0.30, 0.40, 0.55, 0.75, 0.90], trials_gap=80,
                     q_grid=[1, 2, 4, 8, 16, 32], trials_task=80),
    }


def write_report(path: Path, script_path: Path, profile: str, cfg: Mapping[str, Any], base_seed: int,
                 elapsed: float, dim_recs: Sequence[Mapping[str, Any]], gap_recs: Sequence[Mapping[str, Any]],
                 task_recs: Sequence[Mapping[str, Any]], dim_txt: str, dim_metrics: Mapping[str, Any],
                 gap_txt: str, gap_metrics: Mapping[str, Any], task_txt: str, task_metrics: Mapping[str, Any],
                 level: int, notes: Sequence[str]) -> None:
    all_recs = list(dim_recs) + list(gap_recs) + list(task_recs)
    errors = [r for r in all_recs if "error" in r]
    with path.open("w", encoding="utf-8") as f:
        f.write("CAPABILITY-TRACK STAGE 2B ACTIVE CAUSAL REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"script_version: {SCRIPT_VERSION}\n")
        f.write(f"script_sha256: {sha256_file(script_path)}\n")
        f.write(f"profile: {profile}\nbase_seed: {base_seed}\nelapsed_seconds: {elapsed:.3f}\n")
        f.write(f"command: {' '.join(sys.argv)}\n\n")
        f.write("MACHINE / DEPENDENCY METADATA\n")
        f.write(json.dumps(metadata(), indent=2, sort_keys=True) + "\n\n")
        f.write("FROZEN CONFIG\n")
        f.write(json.dumps(dict(cfg), indent=2, sort_keys=True) + "\n\n")

        f.write("WHY STAGE 2A WAS NOT SIMPLY RETUNED\n")
        f.write(
            "For multivariate ridge with a common design matrix X, the ridge solution is linear in the target Y. "
            "If M_hat fits full next observation Y, then fitting a fixed task projection W Y directly gives exactly "
            "W M_hat (same regularization/intercept conventions). Recursively, an untruncated Capability basis is an "
            "exact factorization of the full ridge task prediction. Therefore passive same-data linear regression is "
            "not a valid place to demand a task-relative sample advantage. Stage 2B instead tests selective acquisition.\n\n"
        )

        f.write("SCIENTIFIC SCOPE\n")
        f.write(
            "This is a resettable/generative intervention benchmark with a known candidate causal skeleton and unknown true ancestry. "
            "Capability knows the terminal task-family targets but not their upstream causal ancestors. It chooses interventions only on "
            "candidate parents of the current task-relevant frontier. Full-ID uses the same intervention primitive but estimates the whole system. "
            "A positive result supports decision-directed acquisition under explicit sparsity/locality assumptions. It does not prove superiority "
            "over task-family-aware reward-agnostic RL or causal RL, which may exploit the same structure.\n\n"
        )

        f.write("FROZEN KILL CRITERIA\n")
        f.write("K0: Capability must reach >=90% epsilon-adequate deployment decisions in every tested local-D condition.\n")
        f.write("K1: In local sparse skeletons, Capability interaction slope vs D should be <0.30 while Full-ID and Direct-Source exceed 0.70.\n")
        f.write("K2: Capability must beat uniform Random-SameBudget at the largest local D, and in the global/no-locality control it should lose the flat-D advantage (slope >0.60).\n")
        f.write("K3: Weak-edge interaction scaling should be broadly consistent with inverse-square identifiability.\n")
        f.write("K4: Task-family cost should grow with task ancestry; no claim of free universal transfer.\n")
        f.write("K5: Oracle-Ancestor is privileged. Random-SameBudget is diagnostic, not a strong RL baseline.\n")
        f.write("K6: Even Evidence 3 only licenses a later same-assumption comparison against strong reward-agnostic/causal RL.\n\n")

        f.write("AUTOMATIC EVIDENCE LEVEL\n")
        f.write(f"level: {level}\n")
        for n in notes:
            f.write(f"- {n}\n")
        f.write("\n")
        f.write(dim_txt + "\n\n")
        f.write(gap_txt + "\n\n")
        f.write(task_txt + "\n\n")

        f.write("THEOREM-MATCHING SAMPLE RADII\n")
        f.write(
            "For a one-step intervention mean with Gaussian noise sigma, familywise confidence radius is "
            "r(n)=sigma/amp*sqrt(2 logterm/n). Under the beta-min assumption (edge is 0 or |a|>=beta_min), "
            "r<=beta_min/4 makes threshold beta_min/2 classify every tested coefficient correctly on the simultaneous-confidence event. "
            "This yields the predicted O(1/beta_min^2) classification burden.\n\n"
        )

        f.write("INTERPRETATION BOUNDARIES\n")
        f.write("1. Candidate skeleton/locality is supplied to every causal learner. Unknown arbitrary relevance among D unrelated variables has an Omega(D)-type search cost.\n")
        f.write("2. The intervention oracle can reset at intermediate layers. This is stronger than ordinary online episodic RL.\n")
        f.write("3. The declared terminal task family is known before exploration, while individual scalar weights are post-hoc.\n")
        f.write("4. A task-family-aware reward-agnostic RL algorithm may realize the same structural benefit. If so, credit the principle, not a new algorithm.\n")
        f.write("5. This stage tests acquisition complexity, not learned visual/latent representation discovery.\n\n")

        f.write("LITERATURE ANCHORS FOR INTERPRETATION\n")
        f.write("- Jin et al. 2020, Reward-Free Exploration for Reinforcement Learning, PMLR 119.\n")
        f.write("- Li et al. 2024, Minimax-optimal reward-agnostic exploration in reinforcement learning, PMLR 247.\n")
        f.write("- Weichwald et al. 2022, Learning by Doing: Controlling a Dynamical System using Causality, Control, and RL, PMLR 176.\n")
        f.write("- Ahuja et al. 2023, Interventional Causal Representation Learning, PMLR 202.\n")
        f.write("- Khorasani et al. 2025, Hierarchical Reinforcement Learning with Targeted Causal Interventions, PMLR 267.\n\n")

        f.write("ERROR SUMMARY\n")
        f.write(f"records_total: {len(all_recs)}\nrecords_with_error: {len(errors)}\n")
        for e in errors[:20]:
            f.write(json.dumps(e, sort_keys=True) + "\n")
        f.write("\nRAW RECORDS JSONL\n")
        f.write("-" * 80 + "\n")
        for r in all_recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability-Track Stage 2B active causal falsification harness")
    p.add_argument("--profile", choices=["smoke", "standard", "deep"], default="standard")
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = profiles()[args.profile]
    t0 = time.perf_counter()
    dim = run_dimension_suite(args.seed, cfg)
    gap = run_gap_suite(args.seed, cfg)
    task = run_task_suite(args.seed, cfg)
    dim_txt, dim_m = summarize_dimension(dim)
    gap_txt, gap_m = summarize_gap(gap)
    task_txt, task_m = summarize_tasks(task)
    level, notes = automatic_evidence(dim_m, gap_m, task_m)
    elapsed = time.perf_counter() - t0
    script_path = Path(__file__).resolve()
    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path.cwd() / f"capability_track_stage2b_report_{args.profile}_{stamp}.txt"
    write_report(out, script_path, args.profile, cfg, args.seed, elapsed, dim, gap, task,
                 dim_txt, dim_m, gap_txt, gap_m, task_txt, task_m, level, notes)
    print(f"[Stage2B] report: {out}")
    print(f"[Stage2B] automatic evidence level: {level}")
    for n in notes:
        print(f"[Stage2B] {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
