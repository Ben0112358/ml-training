import numpy as np
import pandas as pd
from typing import Dict, Callable
import optuna


def bootstrap_portfolio_optuna_objective(
    trial: optuna.trial.Trial,
    bootstrap_paths: Dict[str, pd.DataFrame],
    metric: Callable[[np.ndarray], float],
    p_1_constraint: float | None = None,
    p_5_constraint: float | None = None,
    max_std: float | None = None,
) -> float:
    """
    Optuna objective function for portfolio optimization using stationary
    bootstrap paths of arithmetic returns.

    Args:
        trial (optuna.trial.Trial):
            Optuna trial object used to suggest portfolio weights.
        bootstrap_paths (Dict[str, pd.DataFrame]):
            Dictionary of bootstrap sample paths from the stationary
            bootstrap function. Keys are asset names, values are
            DataFrames of shape (n_periods x n_bootstrap_paths)
            containing returns.
        metric (Callable[[np.ndarray], float]):
            User-defined function that takes an array of final
            portfolio values across bootstrap paths and returns
            a scalar metric to maximize.
        p_1_constraint (float | None):
            Optional lower bound on the 1st percentile of final
            portfolio values.
        p_5_constraint (float | None):
            Optional lower bound on the 5th percentile of final
            portfolio values.
        max_std (float | None):
            Optional maximum standard deviation allowed for final
            portfolio values.

    Returns:
        float: Value to be maximized by Optuna. Returns -np.inf if any
        constraint is violated.
    """
    assets = list(bootstrap_paths.keys())

    raw_weights = [trial.suggest_float(f"w_{a}", 0.0, 1.0) for a in assets]
    weights = np.array(raw_weights)
    weights /= weights.sum()
    trial.set_user_attr("normalized_weights", weights)

    returns = np.stack(
        [bootstrap_paths[a].values.astype(float) for a in assets], axis=2
    )

    portfolio_returns = (returns * weights).sum(axis=2)

    final_values = np.prod(1 + portfolio_returns, axis=0) - 1

    if (
        p_1_constraint is not None
        and np.percentile(final_values, 1) < p_1_constraint
    ):
        return -np.inf
    if (
        p_5_constraint is not None
        and np.percentile(final_values, 5) < p_5_constraint
    ):
        return -np.inf
    if max_std is not None and final_values.std() > max_std:
        return -np.inf

    return metric(final_values)
