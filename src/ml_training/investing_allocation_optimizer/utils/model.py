from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import BENCHMARK_LABEL, METRICS
from .monte_carlo import stationary_bootstrap
from .optimization import portfolio_period_returns, run_optuna_study
from .statistics import portfolio_stats


def infer_periods_per_year(df: pd.DataFrame) -> float:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"])
        else:
            return 252.0
    else:
        dates = df.index
    if len(dates) < 2:
        return 252.0
    deltas = pd.Series(dates).diff().dropna().dt.days
    median_days = float(deltas.median())
    if median_days <= 0:
        return 252.0
    if median_days <= 2:
        return 252.0
    if median_days <= 8:
        return 52.0
    if median_days <= 35:
        return 12.0
    return 365.0 / median_days


class BootstrapPortfolioOptimizer:
    """
    Stores historical asset returns; optimizes allocation via stationary
    bootstrap + Optuna on a chosen metric.
    """

    def fit(self, df: pd.DataFrame) -> "BootstrapPortfolioOptimizer":
        if "Date" in df.columns:
            work = df.set_index("Date")
        else:
            work = df.copy()
        self.df = work.sort_index()
        self.assets_ = list(self.df.columns)
        self.periods_per_year_ = infer_periods_per_year(self.df)
        idx = self.df.index
        if hasattr(idx, "min"):
            self.sample_window_ = {
                "start": str(idx.min()),
                "end": str(idx.max()),
                "rows": int(len(self.df)),
            }
        else:
            self.sample_window_ = {
                "start": None,
                "end": None,
                "rows": len(self.df),
            }
        return self

    def predict(
        self,
        metric: str = "mean_return",
        metric_params: dict[str, Any] | None = None,
        p_1_constraint: float | None = None,
        p_5_constraint: float | None = None,
        max_std: float | None = None,
        n_trials: int = 100,
        random_seed: int | None = None,
        bootstrap_block_size: int | str = "cube root",
        horizon_years: float = 10.0,
        n_bootstrap_paths: int = 1000,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
        progress_callback: Callable[[int, float, float], None] | None = None,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric: {metric}. Choose from {list(METRICS)}"
            )

        path_length = max(
            1,
            int(round(horizon_years * self.periods_per_year_)),
        )

        returns_tensor, assets = stationary_bootstrap(
            df=self.df,
            path_length=path_length,
            block_size=bootstrap_block_size,
            n_bootstrap_paths=n_bootstrap_paths,
            random_seed=random_seed,
        )

        benchmark_returns = None
        if BENCHMARK_LABEL in assets:
            b_idx = assets.index(BENCHMARK_LABEL)
            bench_w = np.zeros(len(assets))
            bench_w[b_idx] = 1.0
            benchmark_returns = portfolio_period_returns(
                returns_tensor,
                bench_w,
            )

        study, history = run_optuna_study(
            returns_tensor=returns_tensor,
            assets=assets,
            metric_key=metric,
            periods_per_year=self.periods_per_year_,
            metric_params=metric_params,
            benchmark_returns=benchmark_returns,
            weight_bounds=weight_bounds,
            p_1_constraint=p_1_constraint,
            p_5_constraint=p_5_constraint,
            max_std=max_std,
            n_trials=n_trials,
            random_seed=random_seed,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        if study.best_trial is None or study.best_trial.value is None:
            raise RuntimeError("Optimization produced no feasible trials")

        weights_list = study.best_trial.user_attrs["normalized_weights"]
        weights_arr = np.array(weights_list)
        weights = {a: float(w) for a, w in zip(assets, weights_arr)}

        optimal_port = portfolio_period_returns(returns_tensor, weights_arr)
        stats = portfolio_stats(optimal_port, self.periods_per_year_)

        benchmark_stats = None
        if benchmark_returns is not None:
            benchmark_stats = portfolio_stats(
                benchmark_returns,
                self.periods_per_year_,
            )

        return {
            "weights": weights,
            "metric": metric,
            "metric_value": float(study.best_value),
            "stats": stats,
            "benchmark_stats": benchmark_stats,
            "benchmark_label": BENCHMARK_LABEL,
            "history": history,
            "sample_window": self.sample_window_,
            "periods_per_year": self.periods_per_year_,
            "horizon_years": horizon_years,
            "n_trials": n_trials,
            "trials_completed": len(study.trials),
        }
