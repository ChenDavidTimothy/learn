"""Ordinary retained local regression for standardized camera/action records.

Inputs are already scaled using the shared training-only scales. A fitted
coefficient matrix maps [1, context - query_context, scaled_command] to four
normalized observation corrections. This module has no physical-model or
external-evaluator interface.
"""

from __future__ import annotations

import numpy as np


class LocalPredictor:
    """Fit one local affine correction from a retained, appendable corpus."""

    CONTEXT_DIM = 66
    COMMAND_DIM = 2
    TARGET_DIM = 4
    DESIGN_DIM = 69

    def __init__(
        self,
        context_train: np.ndarray,
        command_train: np.ndarray,
        target_train: np.ndarray,
        neighbors: int,
        ridge: float,
    ) -> None:
        context = np.asarray(context_train, dtype=np.float64)
        command = np.asarray(command_train, dtype=np.float64)
        target = np.asarray(target_train, dtype=np.float64)
        if context.ndim != 2 or context.shape[1] != self.CONTEXT_DIM:
            raise ValueError("context_train must have shape (N, 66)")
        n = context.shape[0]
        if n == 0:
            raise ValueError("the retained corpus must contain at least one row")
        if command.shape != (n, self.COMMAND_DIM):
            raise ValueError("command_train must have shape (N, 2)")
        if target.shape != (n, self.TARGET_DIM):
            raise ValueError("target_train must have shape (N, 4)")
        if not all(np.all(np.isfinite(a)) for a in (context, command, target)):
            raise ValueError("training arrays must be finite")
        if isinstance(neighbors, (bool, np.bool_)) or int(neighbors) != neighbors:
            raise ValueError("neighbors must be a positive integer")
        if int(neighbors) <= 0:
            raise ValueError("neighbors must be a positive integer")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")

        # Own copies: appending an actual observation must not mutate another
        # bootstrap member or the shared ordinary acquisition arrays.
        self.context_train = context.copy()
        self.command_train = command.copy()
        self.target_train = target.copy()
        self.neighbors = int(neighbors)
        self.ridge = float(ridge)

    @property
    def n_rows(self) -> int:
        return int(self.context_train.shape[0])

    def fit(self, query_context: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return frozen local coefficients and reproducible fit diagnostics.

        Commands passed to the constructor/append are already divided by .5.
        Candidate actions therefore enter a returned design as command/.5;
        they do not influence which retained contexts are selected.
        """
        query = np.asarray(query_context, dtype=np.float64)
        if query.shape != (self.CONTEXT_DIM,) or not np.all(np.isfinite(query)):
            raise ValueError("query_context must be a finite vector of shape (66,)")
        centered = self.context_train - query
        d2 = (
            np.mean(centered[:, :2] ** 2, axis=1)
            + np.mean(centered[:, 2:34] ** 2, axis=1)
            + np.mean(centered[:, 34:66] ** 2, axis=1)
        )
        k = min(self.neighbors, self.n_rows)
        # Stable ordering makes exact ties (including bootstrap duplicates)
        # reproducible without using commands or target values as tie breakers.
        indices = np.argsort(d2, kind="stable")[:k]
        bandwidth_d2 = max(float(d2[indices[-1]]), 1e-12)
        weights = np.exp(-d2[indices] / bandwidth_d2)
        normalized_weights = weights / np.sum(weights)
        design = np.column_stack(
            (np.ones(k), centered[indices], self.command_train[indices])
        )
        targets = self.target_train[indices]
        gram = design.T @ (normalized_weights[:, None] * design)
        rhs = design.T @ (normalized_weights[:, None] * targets)
        penalty = np.ones(self.DESIGN_DIM)
        penalty[0] = 0.0
        normal = gram + self.ridge * np.diag(penalty)
        coefficients = np.linalg.solve(normal, rhs)
        if not np.all(np.isfinite(coefficients)):
            raise FloatingPointError("nonfinite local regression coefficients")

        residual = design @ coefficients - targets
        mse_per_output = np.sum(normalized_weights[:, None] * residual**2, axis=0)
        regularizer_per_output = self.ridge * np.sum(coefficients[1:] ** 2, axis=0)
        stationarity = normal @ coefficients - rhs
        diagnostics = {
            "n_rows": self.n_rows,
            "neighbors_requested": self.neighbors,
            "neighbors_used": k,
            "ridge": self.ridge,
            "bandwidth_d2": bandwidth_d2,
            "nearest_d2": float(d2[indices[0]]),
            "farthest_selected_d2": float(d2[indices[-1]]),
            "weighted_mean_d2": float(normalized_weights @ d2[indices]),
            "effective_neighbors": float(1.0 / np.sum(normalized_weights**2)),
            "neighbor_indices": indices.tolist(),
            "normalized_weights": normalized_weights.tolist(),
            "weighted_mse_per_output": mse_per_output.tolist(),
            "weighted_mse": float(np.mean(mse_per_output)),
            "penalized_objective_per_output": (
                mse_per_output + regularizer_per_output
            ).tolist(),
            "normal_equation_max_residual": float(np.max(np.abs(stationarity))),
        }
        return coefficients, diagnostics

    def append(
        self, context: np.ndarray, command: np.ndarray, target: np.ndarray
    ) -> None:
        """Retain one newly completed ordinary row in the shared scale system."""
        arrays = [np.asarray(a, dtype=np.float64) for a in (context, command, target)]
        expected = [(self.CONTEXT_DIM,), (self.COMMAND_DIM,), (self.TARGET_DIM,)]
        if any(a.shape != shape for a, shape in zip(arrays, expected)):
            raise ValueError("append requires vectors with shapes (66,), (2,), (4,)")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("appended row must be finite")
        self.context_train = np.vstack((self.context_train, arrays[0]))
        self.command_train = np.vstack((self.command_train, arrays[1]))
        self.target_train = np.vstack((self.target_train, arrays[2]))
