import numpy as np
import pandas as pd

from ml_training.investing_allocation_optimizer.utils.model import (
    BootstrapPortfolioOptimizer,
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
