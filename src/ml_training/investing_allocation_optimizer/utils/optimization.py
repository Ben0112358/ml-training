import optuna
from typing import Dict, Callable
import pandas as pd

def bootstrap_portfolio_optuna_objective(
    trial: optuna.trial.Trial,
    bootstrap_paths: Dict[str, pd.DataFrame],
    custom_objective: Callable[[pd.DataFrame], float],
    p_1_constraint: float | None = None,
    p_5_constraint: float | None = None,
    max_std: float | None = None,
) -> float:
    """
    Optuna objective function for portfolio optimization using stationary bootstrap paths.

    Args:
        trial (optuna.trial.Trial): 
            Optuna trial object used to suggest weights.
        bootstrap_paths (Dict[str, pd.DataFrame]): 
            Dictionary of bootstrap sample paths from the stationary bootstrap function.
            Keys are column names, values are DataFrames of size (n_periods x n_bootstrap_paths).
        custom_objective (Callable[[pd.DataFrame], float]): 
            User-defined function that takes a weighted outcomes DataFrame and returns a scalar metric to maximize (e.g., expected return).
        p_1_constraint (float | None): 
            Optional worst 1% return constraint.
        p_5_constraint (float | None): 
            Optional worst 5% return constraint.
        max_std (float | None): 
            Optional maximum standard deviation constraint.

    Returns:
        float: Value to be maximized by Optuna.
    """
    raise NotImplementedError
