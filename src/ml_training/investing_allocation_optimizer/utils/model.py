from typing import Callable, Dict, Union
import pandas as pd
import numpy as np
import optuna
from .optimization import bootstrap_portfolio_optuna_objective
from .monte_carlo import stationary_bootstrap


class BootstrapPortfolioOptimizer:
    """
    An investment allocation optimizer model. Takes
    a dataframe of assets along with user preferences
    regarding risk and optimizes a custom metric.

    Optimization of allocation weights is done via
    Optuna. The optimal weights are computed not
    based on the single provided realization(s)
    but rather based on stationary bootstrap 
    samples.

    Due to the stochastic nature of the problem,
    as well due to the fact that change in user
    preferences/settings (ie input) requires
    a new Optuna optimization to be done.

    The .fit() method merely saved the data as 
    an attribute. The .predict() method uses this 
    data along with user input to optimize 
    allocation.
    """

    def fit(self, df):
        """
        Fits the optimizer. Merely saves the
        dataframe so that it can be referenced
        during .predict()
        """
        self.df = df

    def predict(self, 
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
        """
        Predicts the optimal asset allocation weights.

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
        
        assets = list(self.df.columns)

        bootstrap_paths = stationary_bootstrap(
            df=self.df,
            path_length=bootstrap_path_length,
            block_size=bootstrap_block_size,
            n_bootstrap_paths=n_bootstrap_paths,
            random_seed=random_seed,
        )

        def objective(trial):
            return bootstrap_portfolio_optuna_objective(
                trial,
                bootstrap_paths,
                metric,
                p_1_constraint=p_1_constraint,
                p_5_constraint=p_5_constraint,
                max_std=max_std,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        normalized_optimal_weights = pd.Series(
        study.best_trial.user_attrs["normalized_weights"],
        index=assets
        )
        return normalized_optimal_weights.to_dict()
