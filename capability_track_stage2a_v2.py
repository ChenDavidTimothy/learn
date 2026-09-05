#!/usr/bin/env python3
"""
Capability-Track Stage 2A: Unknown Recursive Representation Falsification Harness
===============================================================================

Research question
-----------------
Stage 1 established a clean *information* separation when the relevant structure
was supplied to the learner. Stage 2A removes that crutch.

Every non-oracle method receives the SAME opaque/mixed observations and the SAME
reward-free exploration trajectories. The task family is declared only through
its terminal outcome feature map. The learner is NOT told which latent state
coordinates are relevant, which observed coordinates contain them, or which
nuisance dynamics may be ignored.

The narrow hypothesis tested here is:

    A recursively task-relative predictive representation can sometimes be
    discovered from interaction with fewer samples than are needed to learn a
    reward-general/full predictive model, when the observation/dynamics class
    has enough structure to make that smaller representation learnable.

This is deliberately weaker than "Capability-Track beats RL". Existing methods
such as HOMER, BRIEE, value-equivalent model learning, reward-free RL, and
structured PSR learning already exploit important forms of representation and
reward-independent exploration. A positive Stage-2A result only justifies moving
on to official/faithful implementations of those stronger baselines.

What this script tests
----------------------
1. Unknown task-relative recursive subspace discovery.
   - Latent state = relevant controlled subsystem + independent nuisance system.
   - Observations are an unknown orthogonal mixing of both subsystems.
   - Only a low-dimensional terminal task feature map is declared.
   - Backward task-relative prediction must discover which current directions
     matter after all future action branches.

2. Same-data comparison against:
   - Capability-Backward-Auto: learns only residual task predictions and
     recursively compresses their row space.
   - Capability-Backward-Ridge: same target, dense regression only.
   - Full-Model-Ridge: learns complete observation dynamics.
   - Full-Reduced-Rank: learns complete observation dynamics and then truncates
     each action model by SVD; this is a stronger reward-general predictive
     baseline than raw full-rank fitting.
   - Direct-Sequence-Auto: learns task consequences directly for every complete
     open-loop action sequence; task-aware but non-recursive, so it should pay
     exponentially with horizon.
   - Oracle-Relevant-Fit: privileged lower-bound baseline that sees the relevant
     latent coordinates. It is NOT a fair competitor and is never used as
     evidence of superiority.

3. Observation geometries:
   - aligned: signed permutation of latent coordinates.
   - sparse: product of a modest number of Givens rotations.
   - dense: Haar-like dense orthogonal mixing.

The dense case is intentionally adversarial. If the relevant subspace is embedded
by an arbitrary dense rotation, learning it from raw D-dimensional observations
can itself require D-dependent samples. Failure of nuisance-invariance there is
NOT automatically a failure of the foundation; it demonstrates the required
representation-learnability assumption.

4. Frozen empirical predictions:
   A. In aligned/sparse regimes, task-relative subspace rank should converge to
      the true residual-closed rank and nuisance leakage should become small.
   B. Samples-to-epsilon-regret should grow more slowly with nuisance dimension
      for Capability than for full predictive modeling, if the structure is
      discoverable.
   C. In dense mixing, this gap may shrink or disappear.
   D. Direct sequence regression should deteriorate with horizon relative to the
      recursive method because the number of complete action sequences is A^H.
   E. If Full-Reduced-Rank matches Capability, the advantage belongs to generic
      predictive compression rather than task-relative compression.

5. Report discipline:
   - one text report;
   - raw trial records as JSON Lines;
   - exact seeds/config/dependency metadata;
   - predeclared kill/caution criteria;
   - no plots required.

What this script does NOT establish
-----------------------------------
- It does not implement BRIEE, HOMER, MusIK, B-stable PSR algorithms, or a
  production reward-free RL algorithm. Naming a toy baseline after those methods
  would be scientifically misleading.
- It does not test online regret. Data are collected reward-free, then models are
  evaluated for post-hoc task competence.
- Deployment planning in this Stage-2A harness is over finite OPEN-LOOP action
  sequences. This isolates recursive predictive representation discovery. It does
  not yet test observation-contingent closed-loop policy trees or the full causal
  strategic-measure object from the mathematical foundation.
- It does not test image-scale nonlinear representation learning.
- It does not prove superiority over RL.

If this Stage 2A harness passes, the next serious step is Stage 2B: integrate
faithful external implementations or reimplementations of strong representation-
learning/reward-free baselines and add active decision-invalidating exploration.

Recommended command
-------------------
    python capability_track_stage2a.py --profile deep

Dependencies
------------
    pip install numpy scipy scikit-learn psutil

`psutil` is optional. `scikit-learn` is required for the adaptive sparse/dense
linear regression backend.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid accidental BLAS oversubscription on user machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.random import Generator

try:
    import scipy
except Exception:
    scipy = None

try:
    import sklearn
    from sklearn.linear_model import Lasso, Ridge
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install with: pip install numpy scipy scikit-learn psutil"
    ) from exc

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


SCRIPT_VERSION = "2026-09-05-stage2a-v2"
EPS = 1e-12


# -----------------------------------------------------------------------------
# Profiles
# -----------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "trials": 3,
        "eval_states": 80,
        "task_weights": 12,
        "sample_grid": [64, 128, 256, 512],
        "nuisance_grid": [0, 4, 12],
        "mix_modes": ["aligned", "sparse", "dense"],
        "horizon_grid": [2, 3, 4],
        "d_rel": 5,
        "q_task": 2,
        "n_actions": 2,
        "process_noise": 0.05,
        "epsilon_regret": 0.08,
        "ridge_alpha": 1e-3,
        "svd_rel_tol": 0.04,
        "rrr_energy": 0.995,
        "direct_min_per_sequence": 4,
        "raw_records": True,
    },
    "standard": {
        "trials": 8,
        "eval_states": 200,
        "task_weights": 24,
        "sample_grid": [64, 128, 256, 512, 1024, 2048],
        "nuisance_grid": [0, 4, 8, 16, 32],
        "mix_modes": ["aligned", "sparse", "dense"],
        "horizon_grid": [2, 3, 4, 5],
        "d_rel": 6,
        "q_task": 2,
        "n_actions": 2,
        "process_noise": 0.05,
        "epsilon_regret": 0.06,
        "ridge_alpha": 1e-3,
        "svd_rel_tol": 0.035,
        "rrr_energy": 0.997,
        "direct_min_per_sequence": 6,
        "raw_records": True,
    },
    "deep": {
        "trials": 16,
        "eval_states": 500,
        "task_weights": 48,
        "sample_grid": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "nuisance_grid": [0, 4, 8, 16, 32, 64],
        "mix_modes": ["aligned", "sparse", "dense"],
        "horizon_grid": [2, 3, 4, 5, 6],
        "d_rel": 6,
        "q_task": 2,
        "n_actions": 2,
        "process_noise": 0.05,
        "epsilon_regret": 0.05,
        "ridge_alpha": 1e-3,
        "svd_rel_tol": 0.03,
        "rrr_energy": 0.998,
        "direct_min_per_sequence": 8,
        "raw_records": True,
    },
}


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def rng_for(base_seed: int, *parts: Any) -> Generator:
    return np.random.default_rng(stable_seed(base_seed, *parts))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_ci95(xs: Sequence[float]) -> Tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(xs, dtype=float)
    m = float(np.mean(a))
    if len(a) < 2:
        return m, m, m
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    half = 1.96 * se
    return m, m - half, m + half


def fmt(x: Any, digits: int = 5) -> str:
    if x is None:
        return "NA"
    try:
        y = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(y):
        return str(y)
    if abs(y) >= 1e4 or (abs(y) > 0 and abs(y) < 1e-4):
        return f"{y:.{digits}e}"
    return f"{y:.{digits}f}"


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
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r2)


def row_basis(M: np.ndarray, rel_tol: float = 1e-8, max_rank: Optional[int] = None) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return np.zeros((0, M.shape[1] if M.ndim == 2 else 0), dtype=float)
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        return np.zeros((0, M.shape[1]), dtype=float)
    r = int(np.sum(s >= max(rel_tol * s[0], 1e-12)))
    r = max(1, r)
    if max_rank is not None:
        r = min(r, int(max_rank))
    return vh[:r, :]


def projection_matrix(B: np.ndarray) -> np.ndarray:
    if B.shape[0] == 0:
        return np.zeros((B.shape[1], B.shape[1]))
    Q = row_basis(B, rel_tol=1e-10)
    return Q.T @ Q


def subspace_distance(B_hat: np.ndarray, B_true: np.ndarray) -> float:
    """Normalized chordal-like projection distance in [0, ~1] for equal ranks."""
    P = projection_matrix(B_hat)
    Q = projection_matrix(B_true)
    denom = math.sqrt(max(1.0, float(np.linalg.matrix_rank(Q))))
    return float(np.linalg.norm(P - Q, ord="fro") / (math.sqrt(2.0) * denom))


def nuisance_leakage(B_obs: np.ndarray, C: np.ndarray, d_rel: int) -> float:
    """Energy of learned row functionals on nuisance latent coordinates."""
    if B_obs.shape[0] == 0:
        return 1.0
    B_lat = B_obs @ C  # because o=Cx, row b_obs corresponds to b_obs C on x
    total = float(np.sum(B_lat**2))
    if total <= EPS:
        return 1.0
    nuis = float(np.sum(B_lat[:, d_rel:] ** 2))
    return nuis / total


def all_action_sequences(n_actions: int, horizon: int) -> List[Tuple[int, ...]]:
    return list(itertools.product(range(n_actions), repeat=horizon))


def unit_random_vectors(rng: Generator, n: int, d: int) -> np.ndarray:
    X = rng.normal(size=(n, d))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
    return X


def rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Environment construction
# -----------------------------------------------------------------------------


def random_orthogonal(rng: Generator, d: int) -> np.ndarray:
    if d == 0:
        return np.zeros((0, 0))
    Q, R = np.linalg.qr(rng.normal(size=(d, d)))
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def signed_permutation(rng: Generator, d: int) -> np.ndarray:
    p = rng.permutation(d)
    s = rng.choice([-1.0, 1.0], size=d)
    C = np.zeros((d, d))
    C[np.arange(d), p] = s
    return C


def sparse_orthogonal(rng: Generator, d: int, steps_factor: float = 1.5) -> np.ndarray:
    C = signed_permutation(rng, d)
    steps = max(2, int(steps_factor * d))
    for _ in range(steps):
        i, j = rng.choice(d, size=2, replace=False)
        theta = float(rng.uniform(-0.45, 0.45))
        c, s = math.cos(theta), math.sin(theta)
        G = np.eye(d)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        C = G @ C
    return C


def make_mixing(rng: Generator, d: int, mode: str) -> np.ndarray:
    if mode == "aligned":
        return signed_permutation(rng, d)
    if mode == "sparse":
        return sparse_orthogonal(rng, d)
    if mode == "dense":
        return random_orthogonal(rng, d)
    raise ValueError(f"unknown mixing mode: {mode}")


def make_action_dynamics(rng: Generator, d: int, n_actions: int, radius: float = 0.94) -> List[np.ndarray]:
    """Stable but action-distinct dense dynamics within one latent subsystem."""
    if d == 0:
        return [np.zeros((0, 0)) for _ in range(n_actions)]
    mats: List[np.ndarray] = []
    # Shared non-normal component makes backward task closure nontrivial.
    base_q = random_orthogonal(rng, d)
    scales = np.linspace(0.72, radius, d)
    base = base_q @ np.diag(scales) @ base_q.T
    for a in range(n_actions):
        rot = random_orthogonal(rng, d)
        # Convex combination retains stability while making actions distinct.
        M = 0.55 * base + 0.40 * (rot @ np.diag(scales[::-1]) @ rot.T)
        # Add small action-specific skew interaction.
        skew = rng.normal(scale=0.08, size=(d, d))
        skew = skew - skew.T
        M = M + (a + 1) * 0.025 * skew / max(1, n_actions)
        # Normalize spectral radius below target.
        eig = np.linalg.eigvals(M)
        sr = float(np.max(np.abs(eig))) if len(eig) else 0.0
        if sr > radius:
            M = M * (radius / (sr + EPS))
        mats.append(M)
    return mats


@dataclass
class LinearProductEnv:
    d_rel: int
    k_nuis: int
    n_actions: int
    horizon: int
    q_task: int
    process_noise: float
    mix_mode: str
    seed: int

    def __post_init__(self) -> None:
        if self.q_task > self.d_rel:
            raise ValueError("q_task must be <= d_rel")
        self.D = self.d_rel + self.k_nuis
        rng = np.random.default_rng(self.seed)
        # Time-inhomogeneous dynamics. This prevents a full-model baseline from
        # being artificially disadvantaged by our stage-specific regressions: there
        # is no hidden stationarity for it to pool that Capability is forbidden to use.
        self.A_rel: List[List[np.ndarray]] = []
        self.A_nuis: List[List[np.ndarray]] = []
        self.A_lat: List[List[np.ndarray]] = []
        for h in range(self.horizon):
            rel_h = make_action_dynamics(rng, self.d_rel, self.n_actions, radius=0.95)
            nuis_h = make_action_dynamics(rng, self.k_nuis, self.n_actions, radius=0.97)
            self.A_rel.append(rel_h)
            self.A_nuis.append(nuis_h)
            lat_h: List[np.ndarray] = []
            for a in range(self.n_actions):
                M = np.zeros((self.D, self.D))
                M[: self.d_rel, : self.d_rel] = rel_h[a]
                if self.k_nuis:
                    M[self.d_rel :, self.d_rel :] = nuis_h[a]
                lat_h.append(M)
            self.A_lat.append(lat_h)
        self.C = make_mixing(rng, self.D, self.mix_mode)
        self.T_obs: List[List[np.ndarray]] = [
            [self.C @ self.A_lat[h][a] @ self.C.T for a in range(self.n_actions)]
            for h in range(self.horizon)
        ]
        self.W_lat = np.zeros((self.q_task, self.D))
        self.W_lat[:, : self.q_task] = np.eye(self.q_task)
        self.W_obs = self.W_lat @ self.C.T
        self._true_bases = self._compute_true_task_bases()

    def _compute_true_task_bases(self) -> List[np.ndarray]:
        bases: List[np.ndarray] = [np.zeros((0, self.D)) for _ in range(self.horizon + 1)]
        bases[self.horizon] = row_basis(self.W_obs, rel_tol=1e-12)
        for h in range(self.horizon - 1, -1, -1):
            rows = []
            for a in range(self.n_actions):
                rows.append(bases[h + 1] @ self.T_obs[h][a])
            max_rank = min(self.D, self.q_task * (self.n_actions ** (self.horizon - h)))
            bases[h] = row_basis(np.vstack(rows), rel_tol=1e-10, max_rank=max_rank)
        return bases

    @property
    def true_bases(self) -> List[np.ndarray]:
        return self._true_bases

    def sample_episode(self, rng: Generator, actions: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        if actions is None:
            act = rng.integers(0, self.n_actions, size=self.horizon, endpoint=False)
        else:
            act = np.asarray(actions, dtype=int)
            if len(act) != self.horizon:
                raise ValueError("actions length mismatch")
        x = rng.normal(size=self.D)
        X = [x.copy()]
        O = [self.C @ x]
        for h, a in enumerate(act):
            noise = rng.normal(scale=self.process_noise, size=self.D)
            x = self.A_lat[h][int(a)] @ x + noise
            X.append(x.copy())
            O.append(self.C @ x)
        return {
            "X": np.asarray(X, dtype=float),
            "O": np.asarray(O, dtype=float),
            "A": np.asarray(act, dtype=int),
        }

    def expected_terminal_features(self, o0: np.ndarray, seq: Sequence[int]) -> np.ndarray:
        o = np.asarray(o0, dtype=float)
        for h, a in enumerate(seq):
            o = self.T_obs[h][int(a)] @ o
        return self.W_obs @ o

    def expected_terminal_features_rel(self, r0: np.ndarray, seq: Sequence[int]) -> np.ndarray:
        r = np.asarray(r0, dtype=float)
        for h, a in enumerate(seq):
            r = self.A_rel[h][int(a)] @ r
        return r[: self.q_task]


@dataclass
class EpisodeDataset:
    O: np.ndarray  # [N, H+1, D]
    A: np.ndarray  # [N, H]
    X: np.ndarray  # [N, H+1, D], only oracle may use

    @property
    def n(self) -> int:
        return int(self.O.shape[0])

    def prefix(self, n: int) -> "EpisodeDataset":
        return EpisodeDataset(O=self.O[:n], A=self.A[:n], X=self.X[:n])


def collect_dataset(env: LinearProductEnv, n: int, rng: Generator) -> EpisodeDataset:
    O = np.empty((n, env.horizon + 1, env.D), dtype=float)
    A = np.empty((n, env.horizon), dtype=int)
    X = np.empty((n, env.horizon + 1, env.D), dtype=float)
    # Random action exploration. Same trajectories are passed to every learner.
    for i in range(n):
        ep = env.sample_episode(rng)
        O[i] = ep["O"]
        A[i] = ep["A"]
        X[i] = ep["X"]
    return EpisodeDataset(O=O, A=A, X=X)


# -----------------------------------------------------------------------------
# Linear regression backends
# -----------------------------------------------------------------------------


def _split_train_val(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 10:
        idx = np.arange(n)
        return idx, idx
    cut = max(6, int(0.8 * n))
    return np.arange(cut), np.arange(cut, n)


def _scale_design(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-scale a near-zero-mean design; return scaled X and original-unit scales."""
    scale = np.std(X, axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return X / scale[None, :], scale


def fit_ridge_map(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Return C with Y_hat = X @ C.T, using scale-stable ridge regularization."""
    if len(X) == 0:
        return np.zeros((Y.shape[1], X.shape[1]))
    Xs, scale = _scale_design(X)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Xs, Y)
    Cscaled = np.asarray(model.coef_, dtype=float)
    if Cscaled.ndim == 1:
        Cscaled = Cscaled[None, :]
    return Cscaled / scale[None, :]


def fit_lasso_map(X: np.ndarray, Y: np.ndarray, alpha_scale: float = 0.55) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((Y.shape[1], X.shape[1]))
    Xs, scale = _scale_design(X)
    d = X.shape[1]
    n = max(1, X.shape[0])
    cols = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        yscale = max(float(np.std(y)), 1e-3)
        alpha = alpha_scale * yscale * math.sqrt(math.log(d + 2.0) / n)
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=8000, tol=1e-5)
        model.fit(Xs, y)
        cols.append(np.asarray(model.coef_, dtype=float) / scale)
    return np.vstack(cols)


def map_mse(C: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0:
        return float("inf")
    pred = X @ C.T
    return float(np.mean((pred - Y) ** 2))


def fit_auto_map(X: np.ndarray, Y: np.ndarray, ridge_alpha: float) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Choose ridge vs sparse Lasso by a deterministic holdout from the supplied data."""
    if len(X) < 4:
        C = fit_ridge_map(X, Y, ridge_alpha)
        return C, "ridge", {"ridge_val": float("nan"), "lasso_val": float("nan")}
    tr, va = _split_train_val(len(X))
    Cr = fit_ridge_map(X[tr], Y[tr], ridge_alpha)
    Cl = fit_lasso_map(X[tr], Y[tr])
    mr = map_mse(Cr, X[va], Y[va])
    ml = map_mse(Cl, X[va], Y[va])
    # Small tolerance toward simpler sparse model only when validation supports it.
    choice = "lasso" if ml <= mr * 1.02 else "ridge"
    if choice == "lasso":
        C = fit_lasso_map(X, Y)
    else:
        C = fit_ridge_map(X, Y, ridge_alpha)
    return C, choice, {"ridge_val": mr, "lasso_val": ml}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class CapabilityBackwardModel:
    def __init__(self, backend: str, ridge_alpha: float, svd_rel_tol: float):
        self.backend = backend
        self.ridge_alpha = ridge_alpha
        self.svd_rel_tol = svd_rel_tol
        self.B: List[np.ndarray] = []
        self.L: List[List[np.ndarray]] = []
        self.G: Optional[np.ndarray] = None
        self.backend_counts: Dict[str, int] = {"ridge": 0, "lasso": 0}

    def _fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.backend == "ridge":
            self.backend_counts["ridge"] += 1
            return fit_ridge_map(X, Y, self.ridge_alpha)
        if self.backend == "lasso":
            self.backend_counts["lasso"] += 1
            return fit_lasso_map(X, Y)
        C, choice, _ = fit_auto_map(X, Y, self.ridge_alpha)
        self.backend_counts[choice] += 1
        return C

    def fit(self, data: EpisodeDataset, W_obs: np.ndarray, n_actions: int) -> "CapabilityBackwardModel":
        N, Hp1, D = data.O.shape
        H = Hp1 - 1
        self.B = [np.zeros((0, D)) for _ in range(H + 1)]
        self.L = [[np.zeros((0, 0)) for _ in range(n_actions)] for _ in range(H)]
        self.B[H] = row_basis(W_obs, rel_tol=1e-10)
        self.G = W_obs @ self.B[H].T

        for h in range(H - 1, -1, -1):
            Ks: List[np.ndarray] = []
            for a in range(n_actions):
                mask = data.A[:, h] == a
                X = data.O[mask, h, :]
                Ynext = data.O[mask, h + 1, :] @ self.B[h + 1].T
                if len(X) < 2:
                    K = np.zeros((self.B[h + 1].shape[0], D))
                else:
                    K = self._fit(X, Ynext)
                Ks.append(K)
            max_rank = min(D, W_obs.shape[0] * (n_actions ** (H - h)))
            stack = np.vstack(Ks) if Ks else np.zeros((0, D))
            self.B[h] = row_basis(stack, rel_tol=self.svd_rel_tol, max_rank=max_rank)
            if self.B[h].shape[0] == 0:
                self.B[h] = row_basis(stack, rel_tol=1e-10, max_rank=1)
            for a in range(n_actions):
                # K maps o_h -> expected z_{h+1}; project K through z_h=B_h o_h.
                self.L[h][a] = Ks[a] @ self.B[h].T
        return self

    def predict_features(self, o0: np.ndarray, seq: Sequence[int]) -> np.ndarray:
        if self.G is None:
            raise RuntimeError("model not fitted")
        z = self.B[0] @ o0
        for h, a in enumerate(seq):
            z = self.L[h][int(a)] @ z
        return self.G @ z


class FullLinearModel:
    def __init__(self, backend: str, ridge_alpha: float, reduced_rank_energy: Optional[float] = None):
        self.backend = backend
        self.ridge_alpha = ridge_alpha
        self.reduced_rank_energy = reduced_rank_energy
        self.T: List[List[np.ndarray]] = []
        self.backend_counts: Dict[str, int] = {"ridge": 0, "lasso": 0}
        self.ranks: List[int] = []

    def _fit_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.backend == "ridge":
            self.backend_counts["ridge"] += 1
            C = fit_ridge_map(X, Y, self.ridge_alpha)
        elif self.backend == "lasso":
            self.backend_counts["lasso"] += 1
            C = fit_lasso_map(X, Y)
        else:
            C, choice, _ = fit_auto_map(X, Y, self.ridge_alpha)
            self.backend_counts[choice] += 1
        if self.reduced_rank_energy is not None and C.size:
            u, s, vh = np.linalg.svd(C, full_matrices=False)
            power = s**2
            if float(np.sum(power)) > EPS:
                csum = np.cumsum(power) / np.sum(power)
                r = int(np.searchsorted(csum, self.reduced_rank_energy) + 1)
            else:
                r = 1
            C = (u[:, :r] * s[:r]) @ vh[:r, :]
            self.ranks.append(r)
        return C

    def fit(self, data: EpisodeDataset, n_actions: int) -> "FullLinearModel":
        N, Hp1, D = data.O.shape
        H = Hp1 - 1
        self.T = [[np.zeros((D, D)) for _ in range(n_actions)] for _ in range(H)]
        for h in range(H):
            for a in range(n_actions):
                mask = data.A[:, h] == a
                X = data.O[mask, h, :]
                Y = data.O[mask, h + 1, :]
                if len(X) < 2:
                    self.T[h][a] = np.zeros((D, D))
                else:
                    self.T[h][a] = self._fit_map(X, Y)
        return self

    def predict_features(self, o0: np.ndarray, seq: Sequence[int], W_obs: np.ndarray) -> np.ndarray:
        o = np.asarray(o0, dtype=float)
        for h, a in enumerate(seq):
            o = self.T[h][int(a)] @ o
        return W_obs @ o


class DirectSequenceModel:
    def __init__(self, backend: str, ridge_alpha: float, min_per_sequence: int):
        self.backend = backend
        self.ridge_alpha = ridge_alpha
        self.min_per_sequence = min_per_sequence
        self.maps: Dict[Tuple[int, ...], np.ndarray] = {}
        self.fallback: Optional[np.ndarray] = None
        self.backend_counts: Dict[str, int] = {"ridge": 0, "lasso": 0}
        self.counts: Dict[Tuple[int, ...], int] = {}

    def _fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.backend == "ridge":
            self.backend_counts["ridge"] += 1
            return fit_ridge_map(X, Y, self.ridge_alpha)
        if self.backend == "lasso":
            self.backend_counts["lasso"] += 1
            return fit_lasso_map(X, Y)
        C, choice, _ = fit_auto_map(X, Y, self.ridge_alpha)
        self.backend_counts[choice] += 1
        return C

    def fit(self, data: EpisodeDataset, W_obs: np.ndarray, n_actions: int) -> "DirectSequenceModel":
        X0 = data.O[:, 0, :]
        Yt = data.O[:, -1, :] @ W_obs.T
        self.fallback = self._fit(X0, Yt)
        seqs = all_action_sequences(n_actions, data.A.shape[1])
        for seq in seqs:
            mask = np.all(data.A == np.asarray(seq)[None, :], axis=1)
            self.counts[seq] = int(np.sum(mask))
            if self.counts[seq] >= self.min_per_sequence:
                self.maps[seq] = self._fit(X0[mask], Yt[mask])
        return self

    def predict_features(self, o0: np.ndarray, seq: Sequence[int]) -> np.ndarray:
        key = tuple(int(a) for a in seq)
        C = self.maps.get(key, self.fallback)
        if C is None:
            raise RuntimeError("model not fitted")
        return C @ o0


class OracleRelevantFit:
    """Privileged lower bound. Uses hidden relevant coordinates from the dataset."""

    def __init__(self, ridge_alpha: float):
        self.ridge_alpha = ridge_alpha
        self.A: List[List[np.ndarray]] = []
        self.d_rel = 0
        self.q_task = 0

    def fit(self, data: EpisodeDataset, d_rel: int, q_task: int, n_actions: int) -> "OracleRelevantFit":
        self.d_rel = d_rel
        self.q_task = q_task
        H = data.A.shape[1]
        self.A = [[np.zeros((d_rel, d_rel)) for _ in range(n_actions)] for _ in range(H)]
        for h in range(H):
            for a in range(n_actions):
                mask = data.A[:, h] == a
                X = data.X[mask, h, :d_rel]
                Y = data.X[mask, h + 1, :d_rel]
                if len(X) >= 2:
                    self.A[h][a] = fit_ridge_map(X, Y, self.ridge_alpha)
        return self

    def predict_features(self, r0: np.ndarray, seq: Sequence[int]) -> np.ndarray:
        r = np.asarray(r0, dtype=float)
        for h, a in enumerate(seq):
            r = self.A[h][int(a)] @ r
        return r[: self.q_task]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@dataclass
class EvalSummary:
    mean_rel_regret: float
    median_rel_regret: float
    success_rate: float
    feature_mse: float


def choose_sequence(
    seqs: Sequence[Tuple[int, ...]],
    weight: np.ndarray,
    predictor,
) -> Tuple[int, ...]:
    best_seq = seqs[0]
    best = -float("inf")
    for seq in seqs:
        f = np.asarray(predictor(seq), dtype=float)
        score = float(weight @ f)
        if score > best:
            best = score
            best_seq = seq
    return best_seq


def evaluate_model(
    env: LinearProductEnv,
    model_name: str,
    model: Any,
    eval_eps: Sequence[Dict[str, np.ndarray]],
    weights: np.ndarray,
    epsilon_regret: float,
) -> EvalSummary:
    """Evaluate one fitted model without recomputing predictions per task weight.

    V1 repeatedly propagated the same model once for every task weight. That was
    mathematically redundant and made each condition far slower than intended.
    Here each action-sequence prediction is computed exactly once per evaluation
    state, then all task weights are scored by matrix multiplication.
    """
    seqs = all_action_sequences(env.n_actions, env.horizon)
    nseq = len(seqs)
    regrets: List[float] = []
    mse_sum = 0.0
    mse_count = 0
    successes = 0
    total_tasks = 0

    for ep in eval_eps:
        o0 = ep["O"][0]
        r0 = ep["X"][0, : env.d_rel]

        true_mat = np.vstack([env.expected_terminal_features(o0, seq) for seq in seqs])

        if model_name == "Oracle-Relevant-Fit":
            pred_mat = np.vstack([model.predict_features(r0, seq) for seq in seqs])
        elif model_name.startswith("Full-"):
            pred_mat = np.vstack([model.predict_features(o0, seq, env.W_obs) for seq in seqs])
        else:
            pred_mat = np.vstack([model.predict_features(o0, seq) for seq in seqs])

        diff = pred_mat - true_mat
        mse_sum += float(np.sum(diff * diff))
        mse_count += int(diff.size)

        # Shapes: [num_weights, num_sequences]. One model propagation per sequence,
        # then all downstream tasks are cheap linear queries on the predicted features.
        true_scores = weights @ true_mat.T
        pred_scores = weights @ pred_mat.T
        chosen_idx = np.argmax(pred_scores, axis=1)
        best = np.max(true_scores, axis=1)
        worst = np.min(true_scores, axis=1)
        scale = np.maximum(best - worst, 1e-6)
        chosen_score = true_scores[np.arange(len(weights)), chosen_idx]
        rr = np.maximum(0.0, (best - chosen_score) / scale)

        regrets.extend(rr.tolist())
        successes += int(np.sum(rr <= epsilon_regret))
        total_tasks += int(len(rr))

    return EvalSummary(
        mean_rel_regret=float(np.mean(regrets)) if regrets else float("nan"),
        median_rel_regret=float(np.median(regrets)) if regrets else float("nan"),
        success_rate=float(successes / total_tasks) if total_tasks else float("nan"),
        feature_mse=float(mse_sum / mse_count) if mse_count else float("nan"),
    )

def model_diagnostics(env: LinearProductEnv, name: str, model: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(model, CapabilityBackwardModel):
        dists = []
        leaks = []
        ranks = []
        true_ranks = []
        for h, B in enumerate(model.B):
            Bt = env.true_bases[h]
            dists.append(subspace_distance(B, Bt))
            leaks.append(nuisance_leakage(B, env.C, env.d_rel))
            ranks.append(int(B.shape[0]))
            true_ranks.append(int(Bt.shape[0]))
        out.update(
            mean_subspace_distance=float(np.mean(dists)),
            max_subspace_distance=float(np.max(dists)),
            mean_nuisance_leakage=float(np.mean(leaks)),
            ranks=ranks,
            true_ranks=true_ranks,
            backend_counts=dict(model.backend_counts),
        )
    elif isinstance(model, FullLinearModel):
        pred_errs = []
        for h in range(env.horizon):
            for a in range(env.n_actions):
                pred_errs.append(float(np.linalg.norm(model.T[h][a] - env.T_obs[h][a], ord="fro") ** 2 / env.D))
        out.update(
            mean_transition_fro_sq_per_dim=float(np.mean(pred_errs)),
            backend_counts=dict(model.backend_counts),
            reduced_ranks=list(model.ranks),
        )
    elif isinstance(model, DirectSequenceModel):
        out.update(
            learned_sequences=len(model.maps),
            total_sequences=env.n_actions ** env.horizon,
            min_sequence_count=min(model.counts.values()) if model.counts else 0,
            backend_counts=dict(model.backend_counts),
        )
    return out


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------


def make_models(cfg: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    return [
        (
            "Capability-Backward-Auto",
            CapabilityBackwardModel("auto", cfg["ridge_alpha"], cfg["svd_rel_tol"]),
        ),
        (
            "Capability-Backward-Ridge",
            CapabilityBackwardModel("ridge", cfg["ridge_alpha"], cfg["svd_rel_tol"]),
        ),
        (
            "Full-Model-Ridge",
            FullLinearModel("ridge", cfg["ridge_alpha"], None),
        ),
        (
            "Full-Reduced-Rank",
            FullLinearModel("ridge", cfg["ridge_alpha"], cfg["rrr_energy"]),
        ),
        (
            "Direct-Sequence-Auto",
            DirectSequenceModel("auto", cfg["ridge_alpha"], cfg["direct_min_per_sequence"]),
        ),
        (
            "Oracle-Relevant-Fit",
            OracleRelevantFit(cfg["ridge_alpha"]),
        ),
    ]


def fit_model(name: str, model: Any, data: EpisodeDataset, env: LinearProductEnv) -> Any:
    if isinstance(model, CapabilityBackwardModel):
        return model.fit(data, env.W_obs, env.n_actions)
    if isinstance(model, FullLinearModel):
        return model.fit(data, env.n_actions)
    if isinstance(model, DirectSequenceModel):
        return model.fit(data, env.W_obs, env.n_actions)
    if isinstance(model, OracleRelevantFit):
        return model.fit(data, env.d_rel, env.q_task, env.n_actions)
    raise TypeError(type(model))


def run_condition(
    *,
    base_seed: int,
    condition_tag: str,
    d_rel: int,
    k_nuis: int,
    n_actions: int,
    horizon: int,
    q_task: int,
    process_noise: float,
    mix_mode: str,
    sample_grid: Sequence[int],
    trials: int,
    eval_states: int,
    task_weights: int,
    cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    max_n = max(sample_grid)

    for trial in range(trials):
        env_seed = stable_seed(base_seed, condition_tag, "env", trial)
        env = LinearProductEnv(
            d_rel=d_rel,
            k_nuis=k_nuis,
            n_actions=n_actions,
            horizon=horizon,
            q_task=q_task,
            process_noise=process_noise,
            mix_mode=mix_mode,
            seed=env_seed,
        )
        train_rng = rng_for(base_seed, condition_tag, "train", trial)
        data = collect_dataset(env, max_n, train_rng)
        eval_rng = rng_for(base_seed, condition_tag, "eval", trial)
        eval_eps = [env.sample_episode(eval_rng, actions=[0] * horizon) for _ in range(eval_states)]
        # Only initial states are used from eval episodes; actions above are irrelevant.
        w_rng = rng_for(base_seed, condition_tag, "weights", trial)
        weights = unit_random_vectors(w_rng, task_weights, q_task)

        for n in sample_grid:
            prefix = data.prefix(int(n))
            for name, model in make_models(cfg):
                t0 = time.perf_counter()
                try:
                    fitted = fit_model(name, model, prefix, env)
                    train_sec = time.perf_counter() - t0
                    e0 = time.perf_counter()
                    summary = evaluate_model(env, name, fitted, eval_eps, weights, cfg["epsilon_regret"])
                    eval_sec = time.perf_counter() - e0
                    diag = model_diagnostics(env, name, fitted)
                    rec = {
                        "suite": condition_tag,
                        "trial": trial,
                        "model": name,
                        "n": int(n),
                        "d_rel": d_rel,
                        "k_nuis": k_nuis,
                        "D": env.D,
                        "horizon": horizon,
                        "n_actions": n_actions,
                        "q_task": q_task,
                        "mix_mode": mix_mode,
                        "true_rank_h0": int(env.true_bases[0].shape[0]),
                        "mean_rel_regret": summary.mean_rel_regret,
                        "median_rel_regret": summary.median_rel_regret,
                        "success_rate": summary.success_rate,
                        "feature_mse": summary.feature_mse,
                        "train_sec": train_sec,
                        "eval_sec": eval_sec,
                        "rss_mb": rss_mb(),
                        "diagnostics": diag,
                    }
                except Exception as exc:
                    rec = {
                        "suite": condition_tag,
                        "trial": trial,
                        "model": name,
                        "n": int(n),
                        "d_rel": d_rel,
                        "k_nuis": k_nuis,
                        "D": env.D,
                        "horizon": horizon,
                        "n_actions": n_actions,
                        "q_task": q_task,
                        "mix_mode": mix_mode,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                records.append(rec)
            print(
                f"[Stage2A] {condition_tag} trial={trial + 1}/{trials} n={int(n)} complete",
                flush=True,
            )
    return records


def run_nuisance_suite(base_seed: int, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for mode in cfg["mix_modes"]:
        for k in cfg["nuisance_grid"]:
            tag = f"nuisance:{mode}:K={k}"
            print(f"[Stage2A] {tag}", flush=True)
            records.extend(
                run_condition(
                    base_seed=base_seed,
                    condition_tag=tag,
                    d_rel=cfg["d_rel"],
                    k_nuis=int(k),
                    n_actions=cfg["n_actions"],
                    horizon=4,
                    q_task=cfg["q_task"],
                    process_noise=cfg["process_noise"],
                    mix_mode=mode,
                    sample_grid=cfg["sample_grid"],
                    trials=cfg["trials"],
                    eval_states=cfg["eval_states"],
                    task_weights=cfg["task_weights"],
                    cfg=cfg,
                )
            )
    return records


def run_horizon_suite(base_seed: int, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    # Sparse mixing is the main regime: enough opacity to require learning, but
    # still structured enough that task-relative discovery has a plausible edge.
    k = max(8, int(cfg["nuisance_grid"][min(len(cfg["nuisance_grid"]) - 1, 2)]))
    # Cap sample grid for high H because direct sequence regression can become expensive.
    sample_grid = cfg["sample_grid"]
    for H in cfg["horizon_grid"]:
        tag = f"horizon:sparse:H={H}"
        print(f"[Stage2A] {tag}", flush=True)
        records.extend(
            run_condition(
                base_seed=base_seed,
                condition_tag=tag,
                d_rel=cfg["d_rel"],
                k_nuis=k,
                n_actions=cfg["n_actions"],
                horizon=int(H),
                q_task=cfg["q_task"],
                process_noise=cfg["process_noise"],
                mix_mode="sparse",
                sample_grid=sample_grid,
                trials=cfg["trials"],
                eval_states=max(60, cfg["eval_states"] // 2),
                task_weights=max(10, cfg["task_weights"] // 2),
                cfg=cfg,
            )
        )
    return records


# -----------------------------------------------------------------------------
# Aggregation and verdicts
# -----------------------------------------------------------------------------


def valid_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [r for r in records if "error" not in r]


def group_records(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for r in records:
        k = tuple(r.get(x) for x in keys)
        out.setdefault(k, []).append(r)
    return out


def samples_to_threshold(
    records: Sequence[Mapping[str, Any]],
    success_target: float = 0.90,
) -> Dict[Tuple[str, str, int], float]:
    """Key=(mix_mode, model, k_nuis), value=min N with mean success >= target."""
    out: Dict[Tuple[str, str, int], float] = {}
    groups = group_records(valid_records(records), ["mix_mode", "model", "k_nuis", "n"])
    grouped2: Dict[Tuple[str, str, int], List[Tuple[int, float]]] = {}
    for (mode, model, k, n), rs in groups.items():
        m = float(np.mean([float(r["success_rate"]) for r in rs]))
        grouped2.setdefault((str(mode), str(model), int(k)), []).append((int(n), m))
    for key, vals in grouped2.items():
        vals.sort()
        hit = [n for n, s in vals if s >= success_target]
        out[key] = float(min(hit)) if hit else float("inf")
    return out


def summarize_nuisance(records: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    lines: List[str] = []
    metrics: Dict[str, Any] = {}
    good = valid_records(records)
    groups = group_records(good, ["mix_mode", "k_nuis", "model", "n"])
    lines.append("Nuisance scaling aggregate (mean success / mean regret / mean feature MSE)")
    lines.append("mode\tK\tmodel\tN\tsuccess\tregret\tfeat_mse\tsubspace_dist\tleakage")
    for key in sorted(groups, key=lambda x: (str(x[0]), int(x[1]), str(x[2]), int(x[3]))):
        mode, k, model, n = key
        rs = groups[key]
        success = float(np.mean([r["success_rate"] for r in rs]))
        regret = float(np.mean([r["mean_rel_regret"] for r in rs]))
        mse = float(np.mean([r["feature_mse"] for r in rs]))
        sdist = []
        leak = []
        for r in rs:
            d = r.get("diagnostics", {})
            if "mean_subspace_distance" in d:
                sdist.append(float(d["mean_subspace_distance"]))
            if "mean_nuisance_leakage" in d:
                leak.append(float(d["mean_nuisance_leakage"]))
        lines.append(
            f"{mode}\t{k}\t{model}\t{n}\t{success:.4f}\t{regret:.5f}\t{mse:.6g}\t"
            f"{(np.mean(sdist) if sdist else float('nan')):.5f}\t{(np.mean(leak) if leak else float('nan')):.5f}"
        )

    threshold = samples_to_threshold(good, 0.90)
    lines.append("")
    lines.append("Samples to >=90% epsilon-adequate decisions (INF = not reached in grid)")
    lines.append("mode\tK\tmodel\tN90")
    for key in sorted(threshold, key=lambda x: (x[0], x[2], x[1])):
        mode, model, k = key
        n = threshold[key]
        lines.append(f"{mode}\t{k}\t{model}\t{('INF' if not math.isfinite(n) else int(n))}")

    # Scaling slopes for methods that reached threshold at >=2 positive K points.
    slopes: Dict[str, Dict[str, float]] = {}
    for mode in cfg["mix_modes"]:
        for model in [
            "Capability-Backward-Auto",
            "Capability-Backward-Ridge",
            "Full-Model-Ridge",
            "Full-Reduced-Rank",
            "Direct-Sequence-Auto",
        ]:
            xs, ys = [], []
            for k in cfg["nuisance_grid"]:
                if int(k) <= 0:
                    continue
                y = threshold.get((mode, model, int(k)), float("inf"))
                if math.isfinite(y):
                    xs.append(float(k))
                    ys.append(float(y))
            slope, r2 = loglog_slope(xs, ys)
            slopes[f"{mode}:{model}"] = {"slope": slope, "r2": r2, "points": len(xs)}
    metrics["thresholds"] = {str(k): v for k, v in threshold.items()}
    metrics["slopes"] = slopes
    lines.append("")
    lines.append("Log-log N90 nuisance slopes")
    for key, d in sorted(slopes.items()):
        lines.append(f"{key}\tslope={fmt(d['slope'])}\tR2={fmt(d['r2'])}\tpoints={d['points']}")

    # Deepest sample subspace recovery summary.
    max_n = max(cfg["sample_grid"])
    cap = [
        r for r in good
        if r["model"] == "Capability-Backward-Auto" and int(r["n"]) == max_n
    ]
    recovery: Dict[str, Any] = {}
    for mode in cfg["mix_modes"]:
        rs = [r for r in cap if r["mix_mode"] == mode]
        if rs:
            dists = [float(r["diagnostics"]["mean_subspace_distance"]) for r in rs]
            leaks = [float(r["diagnostics"]["mean_nuisance_leakage"]) for r in rs]
            recovery[mode] = {
                "mean_subspace_distance": float(np.mean(dists)),
                "mean_nuisance_leakage": float(np.mean(leaks)),
            }
    metrics["recovery_at_max_n"] = recovery
    return "\n".join(lines), metrics


def summarize_horizon(records: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    lines: List[str] = []
    metrics: Dict[str, Any] = {}
    good = valid_records(records)
    groups = group_records(good, ["horizon", "model", "n"])
    lines.append("Horizon scaling aggregate")
    lines.append("H\tmodel\tN\tsuccess\tregret\tfeature_mse\tlearned_seq")
    for key in sorted(groups, key=lambda x: (int(x[0]), str(x[1]), int(x[2]))):
        H, model, n = key
        rs = groups[key]
        succ = float(np.mean([r["success_rate"] for r in rs]))
        reg = float(np.mean([r["mean_rel_regret"] for r in rs]))
        mse = float(np.mean([r["feature_mse"] for r in rs]))
        learned = []
        for r in rs:
            d = r.get("diagnostics", {})
            if "learned_sequences" in d:
                learned.append(float(d["learned_sequences"]))
        lines.append(
            f"{H}\t{model}\t{n}\t{succ:.4f}\t{reg:.5f}\t{mse:.6g}\t"
            f"{(np.mean(learned) if learned else float('nan')):.2f}"
        )

    # N90 by horizon.
    threshold: Dict[Tuple[int, str], float] = {}
    groups2 = group_records(good, ["horizon", "model", "n"])
    temp: Dict[Tuple[int, str], List[Tuple[int, float]]] = {}
    for (H, model, n), rs in groups2.items():
        temp.setdefault((int(H), str(model)), []).append(
            (int(n), float(np.mean([r["success_rate"] for r in rs])))
        )
    for key, vals in temp.items():
        vals.sort()
        hit = [n for n, s in vals if s >= 0.90]
        threshold[key] = float(min(hit)) if hit else float("inf")
    lines.append("")
    lines.append("Horizon N90")
    lines.append("H\tmodel\tN90")
    for (H, model), n in sorted(threshold.items()):
        lines.append(f"{H}\t{model}\t{('INF' if not math.isfinite(n) else int(n))}")
    metrics["thresholds"] = {str(k): v for k, v in threshold.items()}
    return "\n".join(lines), metrics


def automatic_verdict(nuis_metrics: Mapping[str, Any], horizon_metrics: Mapping[str, Any]) -> Tuple[int, List[str]]:
    """
    Evidence levels are deliberately conservative.

    0: implementation/sanity failed or capability cannot discover structure even
       in aligned/sparse regimes.
    1: representation discovery works in at least aligned/sparse regime.
    2: same-data nuisance sample scaling is materially better than full model in
       a structured regime.
    3: recursive capability learner also beats direct-sequence consequence
       regression as horizon grows.

    Level 3 still does NOT mean superiority over strong RL literature methods.
    """
    notes: List[str] = []
    recovery = nuis_metrics.get("recovery_at_max_n", {})
    aligned = recovery.get("aligned", {})
    sparse = recovery.get("sparse", {})
    rec_ok = False
    for mode, d in [("aligned", aligned), ("sparse", sparse)]:
        if d:
            dist = float(d.get("mean_subspace_distance", 1.0))
            leak = float(d.get("mean_nuisance_leakage", 1.0))
            if dist < 0.35 and leak < 0.25:
                rec_ok = True
                notes.append(f"PASS representation recovery in {mode}: distance={dist:.3f}, leakage={leak:.3f}")
            else:
                notes.append(f"CAUTION weak representation recovery in {mode}: distance={dist:.3f}, leakage={leak:.3f}")
    if not rec_ok:
        return 0, notes + ["FAIL: task-relative recursive representation was not reliably discovered."]

    level = 1
    slopes = nuis_metrics.get("slopes", {})
    scaling_ok = False
    for mode in ["aligned", "sparse"]:
        cap = slopes.get(f"{mode}:Capability-Backward-Auto", {})
        full = slopes.get(f"{mode}:Full-Model-Ridge", {})
        sc = float(cap.get("slope", float("nan")))
        sf = float(full.get("slope", float("nan")))
        pc = int(cap.get("points", 0))
        pf = int(full.get("points", 0))
        if pc >= 2 and pf >= 2 and math.isfinite(sc) and math.isfinite(sf) and sc + 0.20 < sf:
            scaling_ok = True
            notes.append(f"PASS nuisance scaling gap in {mode}: capability slope={sc:.3f}, full slope={sf:.3f}")
        elif pc >= 2 and pf >= 2:
            notes.append(f"NO clear nuisance scaling gap in {mode}: capability slope={sc:.3f}, full slope={sf:.3f}")
    if scaling_ok:
        level = 2

    # Horizon comparison: require direct sequence to have larger N90 at the two
    # largest horizons for which both are finite, or direct to fail while cap succeeds.
    ht = horizon_metrics.get("thresholds", {})
    parsed: Dict[Tuple[int, str], float] = {}
    for k, v in ht.items():
        # keys stringify tuples like "(4, 'Capability-Backward-Auto')"
        try:
            # safe small parser, no eval
            left = k.strip()[1:-1]
            hpart, mpart = left.split(",", 1)
            H = int(hpart.strip())
            model = mpart.strip().strip("'").strip('"')
            parsed[(H, model)] = float(v)
        except Exception:
            pass
    hs = sorted({h for h, m in parsed if m == "Capability-Backward-Auto"})
    comparisons = []
    for H in hs:
        c = parsed.get((H, "Capability-Backward-Auto"), float("inf"))
        d = parsed.get((H, "Direct-Sequence-Auto"), float("inf"))
        if math.isfinite(c):
            comparisons.append((H, c, d))
    horizon_ok = False
    if comparisons:
        tail = comparisons[-2:]
        horizon_ok = all((not math.isfinite(d)) or d >= 1.5 * c for _, c, d in tail)
        if horizon_ok:
            notes.append("PASS recursive-vs-direct horizon criterion on largest evaluated horizons.")
        else:
            notes.append("NO clear recursive-vs-direct horizon advantage on largest evaluated horizons.")
    if level >= 2 and horizon_ok:
        level = 3

    dense = recovery.get("dense", {})
    if dense:
        notes.append(
            "Dense-mixing diagnostic: distance="
            f"{float(dense.get('mean_subspace_distance', float('nan'))):.3f}, leakage="
            f"{float(dense.get('mean_nuisance_leakage', float('nan'))):.3f}. "
            "Dense degradation is expected unless additional representation structure is available."
        )
    return level, notes


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def dependency_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": getattr(scipy, "__version__", None),
        "sklearn": sklearn.__version__,
        "psutil": getattr(psutil, "__version__", None) if psutil else None,
        "cpu_count": os.cpu_count(),
        "rss_mb_at_report": rss_mb(),
    }


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    nuisance_records: Sequence[Mapping[str, Any]],
    horizon_records: Sequence[Mapping[str, Any]],
    nuisance_text: str,
    nuisance_metrics: Mapping[str, Any],
    horizon_text: str,
    horizon_metrics: Mapping[str, Any],
    level: int,
    verdict_notes: Sequence[str],
    elapsed: float,
) -> None:
    script_path = Path(__file__).resolve()
    lines: List[str] = []
    lines.append("CAPABILITY-TRACK STAGE 2A REPORT")
    lines.append("=" * 80)
    lines.append(f"script_version: {SCRIPT_VERSION}")
    lines.append(f"script_sha256: {sha256_file(script_path)}")
    lines.append(f"profile: {args.profile}")
    lines.append(f"base_seed: {args.seed}")
    lines.append(f"elapsed_seconds: {elapsed:.3f}")
    lines.append(f"command: {' '.join(sys.argv)}")
    lines.append("")
    lines.append("DEPENDENCY / MACHINE METADATA")
    lines.append(json.dumps(dependency_metadata(), indent=2, sort_keys=True))
    lines.append("")
    lines.append("FROZEN CONFIG")
    lines.append(json.dumps(dict(cfg), indent=2, sort_keys=True))
    lines.append("")
    lines.append("SCIENTIFIC SCOPE")
    lines.append(
        "This is a same-data reward-free learn-then-deploy representation-discovery test with open-loop deployment sequence planning. "
        "Non-oracle methods receive identical mixed observations and trajectories. Capability "
        "receives the declared terminal task-feature map, not hidden relevance labels."
    )
    lines.append(
        "A positive result does NOT establish superiority over RL, HOMER, BRIEE, MusIK, or "
        "B-stable PSR algorithms. It establishes only that the task-relative recursive target "
        "is empirically learnable in this synthetic class and can reduce sample cost relative "
        "to reward-general/full predictive fitting."
    )
    lines.append("")
    lines.append("PREREGISTERED CAUTIONS / KILL CRITERIA")
    lines.extend(
        [
            "K0: If Capability cannot recover the task-relative subspace in aligned/sparse mixing, stop and debug/revise before stronger claims.",
            "K1: If same-data Capability N90 does not scale materially better than full predictive fitting in any structured regime, the hoped-for statistical advantage is unsupported here.",
            "K2: If Full-Reduced-Rank matches Capability, credit generic predictive compression rather than task-relative compression.",
            "K3: If Direct-Sequence matches recursive Capability as H grows, recursive closure has not shown empirical value in this class.",
            "K4: Dense-mixing degradation is expected and diagnoses representation-learning cost; do not hide it.",
            "K5: Oracle-Relevant-Fit is privileged and must never be counted as a fair win.",
            "K6: Even Evidence Level 3 only licenses Stage 2B against faithful strong RL/representation-learning baselines.",
            "K7: Stage 2A uses open-loop deployment sequence planning; do not generalize its result to closed-loop partially observed agency.",
        ]
    )
    lines.append("")
    lines.append("AUTOMATIC EVIDENCE LEVEL")
    lines.append(f"level: {level}")
    for n in verdict_notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("NUISANCE / REPRESENTATION DISCOVERY SUITE")
    lines.append("-" * 80)
    lines.append(nuisance_text)
    lines.append("")
    lines.append("HORIZON / RECURSIVE CLOSURE SUITE")
    lines.append("-" * 80)
    lines.append(horizon_text)
    lines.append("")
    lines.append("ERROR SUMMARY")
    all_records = list(nuisance_records) + list(horizon_records)
    errors = [r for r in all_records if "error" in r]
    lines.append(f"records_total: {len(all_records)}")
    lines.append(f"records_with_error: {len(errors)}")
    for r in errors[:20]:
        lines.append(json.dumps(r, sort_keys=True))
    if len(errors) > 20:
        lines.append(f"... {len(errors) - 20} additional errors omitted here; present in RAW RECORDS.")
    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.extend(
        [
            "Evidence 0: representation discovery itself failed; do not advance.",
            "Evidence 1: task-relative recursive representation was discovered in a structured observation regime.",
            "Evidence 2: same-data nuisance sample scaling also favored task-relative learning over full predictive fitting.",
            "Evidence 3: recursive task-relative learning additionally beat non-recursive direct consequence regression as horizon grew.",
            "No level here means 'better than RL'. Strong literature baselines remain Stage 2B.",
        ]
    )
    lines.append("")
    lines.append("LITERATURE ANCHORS TO USE WHEN INTERPRETING")
    lines.extend(
        [
            "HOMER: Misra et al., ICML 2020, Kinematic State Abstraction and Provably Efficient Rich-Observation RL.",
            "BRIEE: Zhang et al., ICML 2022, Efficient Reinforcement Learning in Block MDPs: A Model-free Representation Learning approach.",
            "Value Equivalence: Grimm et al., NeurIPS 2020, The Value Equivalence Principle for Model-Based RL.",
            "B-Stable PSRs: Chen, Bai, Mei, arXiv:2209.14990 / ICLR line of work on sample-efficient partially observable RL.",
            "Reward-free RL: Zhang, Du, Ji, ICML 2021, Near Optimal Reward-Free Reinforcement Learning.",
        ]
    )
    lines.append("")
    lines.append("RAW RECORDS JSONL")
    lines.append("-" * 80)
    if cfg.get("raw_records", True):
        for r in all_records:
            lines.append(json.dumps(r, sort_keys=True, default=str))
    else:
        lines.append("raw_records disabled")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability-Track Stage 2A falsification harness")
    p.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--output", type=str, default=None, help="optional explicit report path")
    p.add_argument("--skip-horizon", action="store_true", help="run only nuisance/representation suite")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = dict(PROFILES[args.profile])
    start = time.perf_counter()
    print(f"Capability-Track Stage 2A | profile={args.profile} | seed={args.seed}", flush=True)
    print("This can be computationally heavy. Progress is printed per condition.", flush=True)

    nuisance_records = run_nuisance_suite(args.seed, cfg)
    horizon_records: List[Dict[str, Any]] = []
    if not args.skip_horizon:
        horizon_records = run_horizon_suite(args.seed, cfg)

    nuisance_text, nuisance_metrics = summarize_nuisance(nuisance_records, cfg)
    horizon_text, horizon_metrics = summarize_horizon(horizon_records, cfg) if horizon_records else ("SKIPPED", {})
    level, notes = automatic_verdict(nuisance_metrics, horizon_metrics)
    elapsed = time.perf_counter() - start

    if args.output:
        out = Path(args.output)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path.cwd() / f"capability_track_stage2a_report_{args.profile}_{stamp}.txt"
    write_report(
        out,
        args,
        cfg,
        nuisance_records,
        horizon_records,
        nuisance_text,
        nuisance_metrics,
        horizon_text,
        horizon_metrics,
        level,
        notes,
        elapsed,
    )
    print(f"REPORT: {out.resolve()}", flush=True)
    print(f"AUTOMATIC EVIDENCE LEVEL: {level}", flush=True)
    for n in notes:
        print(f"  {n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
