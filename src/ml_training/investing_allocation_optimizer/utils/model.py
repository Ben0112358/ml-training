from typing import Callable, Dict, Union
import pandas as pd
import numpy as np
import optuna
from .optimization import bootstrap_portfolio_optuna_objective
from .monte_carlo import stationary_bootstrap


class BootstrapPortfolioOptimizer:
    """
    Portfolio optimizer using stationary bootstrap and user-defined objective.

    Parameters
    ----------
    metric : Callable[[np.ndarray], float]
        Function that takes an array of final portfolio outcomes and returns
        a scalar metric to maximize.
    p_1_constraint, p_5_constraint, max_std : float | None
        Optional constraints on final portfolio outcomes.
    n_trials : int
        Number of Optuna trials during fit().
    random_seed : int | None
        Random seed for reproducibility.
    bootstrap_block_size : int | "cube root"
        Block size for stationary bootstrap.
    bootstrap_path_length : int
        Number of observations per bootstrap path.
    n_bootstrap_paths : int
        Number of bootstrap paths to generate per asset.
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray], float],
        p_1_constraint: float | None = None,
        p_5_constraint: float | None = None,
        max_std: float | None = None,
        n_trials: int = 100,
        random_seed: int | None = None,
        bootstrap_block_size: Union[int, str] = "cube root",
        bootstrap_path_length: int = 100,
        n_bootstrap_paths: int = 1000,
    ):
        self.metric = metric
        self.p_1_constraint = p_1_constraint
        self.p_5_constraint = p_5_constraint
        self.max_std = max_std
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.bootstrap_block_size = bootstrap_block_size
        self.bootstrap_path_length = bootstrap_path_length
        self.n_bootstrap_paths = n_bootstrap_paths

        self.optimal_weights_ = None
        self.assets_ = None

    def fit(self, df: pd.DataFrame):
        """
        Fit the portfolio optimizer using the original return series dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of returns with datetime-like index and one
            column per asset.
        """
        self.assets_ = list(df.columns)

        bootstrap_paths = stationary_bootstrap(
            df=df,
            path_length=self.bootstrap_path_length,
            block_size=self.bootstrap_block_size,
            n_bootstrap_paths=self.n_bootstrap_paths,
            random_seed=self.random_seed,
        )

        def objective(trial):
            return bootstrap_portfolio_optuna_objective(
                trial,
                bootstrap_paths,
                self.metric,
                p_1_constraint=self.p_1_constraint,
                p_5_constraint=self.p_5_constraint,
                max_std=self.max_std,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.optimal_weights_ = pd.Series(
            [study.best_trial.params[f"w_{a}"] for a in self.assets_],
            index=self.assets_,
        )
        self.optimal_weights_ = study.best_trial.user_attrs[
            "normalized_weights"
        ]
        return self

    def predict(self) -> Dict[str, float]:
        """
        Return optimized portfolio allocation as a dictionary.
        """
        if self.optimal_weights_ is None:
            raise ValueError(
                "Model has not been fitted yet. Call `.fit()` first."
            )
        return self.optimal_weights_.to_dict()
