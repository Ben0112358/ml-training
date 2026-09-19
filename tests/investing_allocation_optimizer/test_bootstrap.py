import numpy as np
import pandas as pd

from ml_training.investing_allocation_optimizer.utils.monte_carlo import (
    stationary_bootstrap,
)


def test_joint_bootstrap_preserves_cross_section():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 0.01, n)
    y = x + rng.normal(0, 0.001, n)
    df = pd.DataFrame({"a": x, "b": y})
    tensor, assets = stationary_bootstrap(
        df,
        path_length=50,
        n_bootstrap_paths=500,
        random_seed=42,
    )
    assert tensor.shape == (50, 500, 2)
    assert assets == ["a", "b"]
    corr_orig = np.corrcoef(df["a"], df["b"])[0, 1]
    corrs = []
    for p in range(500):
        corrs.append(np.corrcoef(tensor[:, p, 0], tensor[:, p, 1])[0, 1])
    assert np.mean(corrs) > corr_orig * 0.5
