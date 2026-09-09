"""Ridge residual learning and empirical one-sided uncertainty."""
from __future__ import annotations

import numpy as np
import pytest

from ecopadg.forecast import (
    ConformalUpperBound,
    LoadForecaster,
    ResidualRegressor,
    RidgeRegressor,
)


def test_ridge_recovers_small_linear_model():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * x[:, 0] + 1.0
    model = RidgeRegressor(alpha=1e-12).fit(x, y)
    assert model.predict([[4.0]])[0] == pytest.approx(9.0, rel=1e-6)


def test_conformal_upper_bound_uses_finite_sample_rank():
    bound = ConformalUpperBound(alpha=0.2)
    predicted = [1.0] * 5
    actual = [1.0, 1.1, 1.2, 1.3, 1.4]
    bound.fit(actual, predicted)
    # ceil((5+1)*0.8)=5: maximum residual for this small sample.
    assert bound.margin() == pytest.approx(0.4)
    assert bound.upper(2.0) == pytest.approx(2.4)


def test_residual_regressor_corrects_baseline_and_bounds_above_mean():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    baseline = np.asarray([10.0, 10.0, 10.0, 10.0])
    actual = baseline + 2.0 * x[:, 0]
    model = ResidualRegressor(ridge_alpha=1e-12)
    model.fit(x, actual, baseline)
    mean = model.predict([[4.0]], [10.0])[0]
    upper = model.predict_upper([[4.0]], [10.0])[0]
    assert mean == pytest.approx(18.0, rel=1e-6)
    assert upper >= mean


def test_load_forecast_tracks_trend_and_is_nonnegative():
    model = LoadForecaster(order=2, ridge_alpha=1e-8)
    model.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    forecast = model.forecast(3)
    assert len(forecast.mean) == 3
    assert forecast.mean[0] > 5.5
    assert all(value >= 0.0 for value in forecast.mean)
    assert all(
        upper >= mean
        for mean, upper in zip(forecast.mean, forecast.upper)
    )


def test_load_forecast_persists_with_too_little_history():
    model = LoadForecaster(order=3)
    model.observe(2.5)
    forecast = model.forecast(2)
    assert forecast.mean == pytest.approx((2.5, 2.5))

