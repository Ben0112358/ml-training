import pandas as pd
import numpy as np
from typing import Literal, Union


def stationary_bootstrap(
    df: pd.DataFrame,
    path_length: int,
    block_size: Union[int, Literal["cube root"]] = "cube root",
    n_bootstrap_paths: int = 9999,
    random_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generates stationary bootstrap resampled from a time series DataFrame.

    Args:
        df (pd.DataFrame):
            DataFrame with a datetime-like index and one time series per
            column.
        path_length (int):
            Number of observations in each bootstrap path (unit of index
            doesn't matter).
        block_size (Union[int, Literal["cube root"]]):
            Block size for the stationary bootstrap. Default: cube root
            of len(df).
        n_bootstrap_paths (int):
            Number of bootstrap paths to generate per column.
            Default: 9999.
        random_seed (int | None):
            Random seed for reproducibility. Default: None.

    Returns:
        dict[str, pd.DataFrame]:
            Dictionary mapping column names to DataFrames of shape
            (path_length, n_bootstrap_paths) containing bootstrap paths.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    n_obs = len(df)

    if block_size == "cube root":
        L = int(round(n_obs ** (1 / 3)))
    else:
        raise NotImplementedError(
            f"block_size method {block_size} is not implemented."
        )

    p = 1 / L

    bootstrap_dict = {
        col: pd.DataFrame(
            index=range(path_length), columns=range(n_bootstrap_paths)
        )
        for col in df.columns
    }

    for col in df.columns:
        series = df[col].values
        for path in range(n_bootstrap_paths):
            sample = []
            t = 0
            while t < path_length:
                start_idx = np.random.randint(0, n_obs)
                block_len = np.random.geometric(p)
                block = series[start_idx : start_idx + block_len]  # noqa: E203

                # Extends beyond end => wrap around
                if start_idx + block_len > n_obs:
                    overflow = start_idx + block_len - n_obs
                    block = np.concatenate(
                        [series[start_idx:], series[:overflow]]
                    )

                sample.extend(block)
                t += len(block)

            bootstrap_dict[col].iloc[:, path] = sample[:path_length]

    return bootstrap_dict
