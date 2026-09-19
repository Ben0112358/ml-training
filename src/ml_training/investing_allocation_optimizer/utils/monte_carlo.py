from typing import Literal, Union

import numpy as np
import pandas as pd


def _block_length(
    block_size: Union[int, Literal["cube root"]], n_obs: int
) -> int:
    if block_size == "cube root":
        return max(1, int(round(n_obs ** (1 / 3))))
    if isinstance(block_size, int) and block_size >= 1:
        return block_size
    raise ValueError(f"Invalid block_size: {block_size}")


def stationary_bootstrap(
    df: pd.DataFrame,
    path_length: int,
    block_size: Union[int, Literal["cube root"]] = "cube root",
    n_bootstrap_paths: int = 9999,
    random_seed: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Joint stationary bootstrap: one index path per bootstrap draw, all
    assets share the same resampled time indices (preserves cross-asset
    correlation within each path).

    Returns
    -------
    returns : ndarray, shape (path_length, n_bootstrap_paths, n_assets)
    assets : list of column names (asset labels)
    """
    if path_length < 1:
        raise ValueError("path_length must be >= 1")
    if n_bootstrap_paths < 1:
        raise ValueError("n_bootstrap_paths must be >= 1")

    assets = list(df.columns)
    values = df[assets].values.astype(float)
    n_obs, n_assets = values.shape
    if n_obs < 2:
        raise ValueError("Need at least 2 observations for bootstrap")

    L = _block_length(block_size, n_obs)
    p = 1.0 / L

    rng = np.random.default_rng(random_seed)

    idx = np.empty((path_length, n_bootstrap_paths), dtype=np.int64)
    idx[0] = rng.integers(0, n_obs, size=n_bootstrap_paths)
    for t in range(1, path_length):
        cont = (idx[t - 1] + 1) % n_obs
        fresh = rng.integers(0, n_obs, size=n_bootstrap_paths)
        renew = rng.random(n_bootstrap_paths) < p
        idx[t] = np.where(renew, fresh, cont)

    # values[idx] -> (path_length, n_paths, n_assets)
    returns = values[idx]
    return returns, assets
