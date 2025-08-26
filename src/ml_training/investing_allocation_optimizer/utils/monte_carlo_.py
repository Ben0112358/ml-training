import pandas as pd
from typing import Literal, Union


def stationary_bootstrap(
    df: pd.DataFrame,
    bootstrap_sample_path_length_year: float,
    block_size: Union[
        int, Literal["cube root", "autocorrelation based"]
    ] = "cube root",
    n_bootstrap_paths: int = 9999,
    random_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generates stationary bootstrap resampled from given time series.

    Args:
        df (pd.DataFrame):
            Dataframe with date index, and one time series per column.
            Frequency is inferred from the index.
        bootstrap_sample_path_length_year (float):
            length in years of all generated bootstrap sample paths.
        block_size (Union[int, Literal["cube root", "autocorrelation based"]]):
            Strategy for block size; can be set to a fixed integer as well.
            Default is cube root of len(df).
        n_bootstrap_paths (int):
            Number of bootstrap sample paths to generate for each
            column / time series. Default is 9999.
        random_seed (int | None):
            The random seed of the bootstrap sample path generation.
            Default is None.
    Returns:
        dict[str, pd.DataFrame]:
            A dict whose keys are the column names of df, and whose
              value is a dataframe of size
              bootstrap_sample_path_length_year x n_bootstrap_paths.
    """
    raise NotImplementedError
