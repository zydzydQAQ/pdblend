"""Small deterministic forecasters and one-sided uncertainty bounds."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np


def _matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("features must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError("features must be finite")
    return result


def _vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("targets must be non-empty and finite")
    return result


class RidgeRegressor:
    """Minimal ridge regression with an unregularized intercept."""

    def __init__(self, alpha: float = 1e-6):
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.intercept_ = 0.0
        self.coef_ = np.empty(0, dtype=float)
        self.fitted_ = False

    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
    ) -> "RidgeRegressor":
        x = _matrix(features)
        y = _vector(targets)
        if x.shape[0] != y.size:
            raise ValueError("features and targets have different lengths")
        design = np.column_stack((np.ones(x.shape[0]), x))
        penalty = np.eye(design.shape[1], dtype=float) * self.alpha
        penalty[0, 0] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ y
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
        self.intercept_ = float(weights[0])
        self.coef_ = np.asarray(weights[1:], dtype=float)
        self.fitted_ = True
        return self

    def predict(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ridge model is not fitted")
        x = _matrix(features)
        if x.shape[1] != self.coef_.size:
            raise ValueError("unexpected feature width")
        return self.intercept_ + x @ self.coef_


class ConformalUpperBound:
    """Finite-sample one-sided empirical residual bound."""

    def __init__(
        self,
        alpha: float = 0.1,
        max_samples: int = 512,
        default_margin: float = 0.0,
    ):
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive")
        if float(default_margin) < 0.0:
            raise ValueError("default_margin must be non-negative")
        self.alpha = float(alpha)
        self.default_margin = float(default_margin)
        self._residuals: Deque[float] = deque(maxlen=int(max_samples))

    @property
    def sample_count(self) -> int:
        return len(self._residuals)

    def clear(self) -> None:
        self._residuals.clear()

    def update(self, actual: float, predicted: float) -> None:
        residual = float(actual) - float(predicted)
        if math.isfinite(residual):
            self._residuals.append(residual)

    def fit(
        self,
        actual: Iterable[float],
        predicted: Iterable[float],
    ) -> "ConformalUpperBound":
        self.clear()
        actual_values = list(actual)
        predicted_values = list(predicted)
        if len(actual_values) != len(predicted_values):
            raise ValueError("actual and predicted have different lengths")
        for observed, estimate in zip(actual_values, predicted_values):
            self.update(observed, estimate)
        return self

    def margin(self) -> float:
        if not self._residuals:
            return self.default_margin
        ordered = sorted(self._residuals)
        rank = int(math.ceil((len(ordered) + 1) * (1.0 - self.alpha)))
        index = min(max(rank - 1, 0), len(ordered) - 1)
        return max(float(ordered[index]), 0.0, self.default_margin)

    def upper(self, predicted: float) -> float:
        return float(predicted) + self.margin()


class ResidualRegressor:
    """Ridge correction plus a conformal bound on remaining residuals."""

    def __init__(
        self,
        ridge_alpha: float = 1e-6,
        conformal_alpha: float = 0.1,
        default_margin: float = 0.0,
    ):
        self.model = RidgeRegressor(alpha=ridge_alpha)
        self.bound = ConformalUpperBound(
            alpha=conformal_alpha, default_margin=default_margin
        )

    @property
    def fitted(self) -> bool:
        return self.model.fitted_

    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        actual: Sequence[float] | np.ndarray,
        baseline: Sequence[float] | np.ndarray,
    ) -> "ResidualRegressor":
        y = _vector(actual)
        base = _vector(baseline)
        if y.size != base.size:
            raise ValueError("actual and baseline have different lengths")
        residual = y - base
        self.model.fit(features, residual)
        correction = self.model.predict(features)
        self.bound.fit(residual, correction)
        return self

    def predict(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        baseline: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        base = _vector(baseline)
        x = _matrix(features)
        if x.shape[0] != base.size:
            raise ValueError("features and baseline have different lengths")
        correction = self.model.predict(x) if self.fitted else 0.0
        return base + correction

    def predict_upper(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        baseline: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.predict(features, baseline) + self.bound.margin()


@dataclass(frozen=True)
class Forecast:
    mean: Tuple[float, ...]
    upper: Tuple[float, ...]
    model_version: str

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.upper):
            raise ValueError("forecast vectors must be non-empty and aligned")


class LoadForecaster:
    """Autoregressive ridge forecast with passive conformal calibration."""

    def __init__(
        self,
        order: int = 3,
        window: int = 256,
        ridge_alpha: float = 1e-3,
        conformal_alpha: float = 0.1,
        default_margin: float = 0.0,
        model_version: str = "load-ridge-v1",
    ):
        if int(order) <= 0:
            raise ValueError("order must be positive")
        if int(window) <= int(order):
            raise ValueError("window must exceed order")
        self.order = int(order)
        self.window = int(window)
        self.model_version = str(model_version)
        self.model = RidgeRegressor(alpha=ridge_alpha)
        self.bound = ConformalUpperBound(
            alpha=conformal_alpha,
            max_samples=window,
            default_margin=default_margin,
        )
        self._history: Deque[float] = deque(maxlen=self.window)

    @property
    def history(self) -> Tuple[float, ...]:
        return tuple(self._history)

    def _training_rows(
        self, history: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        for index in range(self.order, len(history)):
            rows.append(
                [float(history[index - lag - 1]) for lag in range(self.order)]
            )
            targets.append(float(history[index]))
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def fit(self, loads: Sequence[float]) -> "LoadForecaster":
        parsed = [max(float(value), 0.0) for value in loads]
        if not parsed or not all(math.isfinite(value) for value in parsed):
            raise ValueError("loads must be non-empty, finite, and non-negative")
        self._history.clear()
        self._history.extend(parsed[-self.window :])
        if len(self._history) > self.order:
            x, y = self._training_rows(tuple(self._history))
            self.model.fit(x, y)
            self.bound.fit(y, self.model.predict(x))
        return self

    def observe(self, load_rps: float) -> None:
        value = float(load_rps)
        if not math.isfinite(value) or value < 0.0:
            return
        self._history.append(value)
        if len(self._history) > self.order:
            x, y = self._training_rows(tuple(self._history))
            self.model.fit(x, y)
            self.bound.fit(y, self.model.predict(x))

    def forecast(self, horizon: int = 1) -> Forecast:
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        work = list(self._history)
        if not work:
            work = [0.0]
        means = []
        uppers = []
        for _ in range(int(horizon)):
            if self.model.fitted_ and len(work) >= self.order:
                features = np.asarray(
                    [[work[-lag - 1] for lag in range(self.order)]],
                    dtype=float,
                )
                estimate = float(self.model.predict(features)[0])
            else:
                estimate = float(work[-1])
            estimate = max(estimate, 0.0)
            means.append(estimate)
            uppers.append(max(self.bound.upper(estimate), estimate, 0.0))
            work.append(estimate)
        return Forecast(
            mean=tuple(means),
            upper=tuple(uppers),
            model_version=self.model_version,
        )

