import numpy as np
import pandas as pd
import pytest

from ml_training.investing_allocation_optimizer.utils.model import (
    BootstrapPortfolioOptimizer,
    infer_periods_per_year,
)


def test_predict_returns_structure():
    rng = np.random.default_rng(1)
    n = 120
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "total-world": rng.normal(0.0004, 0.01, n),
            "growth": rng.normal(0.0005, 0.012, n),
        },
        index=dates,
    )
    mdl = BootstrapPortfolioOptimizer().fit(df)
    out = mdl.predict(
        metric="mean_return",
        n_trials=5,
        n_bootstrap_paths=50,
        horizon_years=1.0,
        random_seed=0,
    )
    assert "weights" in out
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-6
    assert out["benchmark_stats"] is not None


def test_predict_raises_runtime_error_when_no_trial_is_feasible():
    rng = np.random.default_rng(1)
    n = 120
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "total-world": rng.normal(0.0004, 0.01, n),
            "growth": rng.normal(0.0005, 0.012, n),
        },
        index=dates,
    )
    mdl = BootstrapPortfolioOptimizer().fit(df)
    with pytest.raises(RuntimeError):
        mdl.predict(
            metric="mean_return",
            p_5_constraint=10.0,
            n_trials=3,
            n_bootstrap_paths=20,
            horizon_years=1.0,
            random_seed=0,
        )


def _monthly_returns(n: int = 36) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    return {
        "total-world": rng.normal(0.005, 0.04, n),
        "growth": rng.normal(0.006, 0.05, n),
    }


def test_fit_infers_monthly_frequency_from_string_date_column():
    n = 36
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), **_monthly_returns(n)}
    )
    mdl = BootstrapPortfolioOptimizer().fit(df)
    assert mdl.periods_per_year_ == 12.0


def test_infer_periods_per_year_parses_string_date_index():
    n = 36
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        _monthly_returns(n),
        index=pd.Index(dates.strftime("%Y-%m-%d"), name="Date"),
    )
    assert infer_periods_per_year(df) == 12.0
