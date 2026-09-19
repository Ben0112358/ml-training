from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .metrics import apply_metric
from .weights import project_box_simplex


def portfolio_period_returns(
    returns_tensor: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """returns_tensor (T, P, A), weights (A,) -> (T, P)."""
    return np.tensordot(returns_tensor, weights, axes=([2], [0]))


def bootstrap_portfolio_optuna_objective(
    trial: optuna.trial.Trial,
    returns_tensor: np.ndarray,
    assets: list[str],
    metric_key: str,
    periods_per_year: float,
    metric_params: dict[str, Any] | None,
    benchmark_returns: np.ndarray | None,
    weight_lower: np.ndarray,
    weight_upper: np.ndarray,
    p_1_constraint: float | None,
    p_5_constraint: float | None,
    max_std: float | None,
) -> float:
    raw = [trial.suggest_float(f"w_{a}", 0.0, 1.0) for a in assets]
    weights = project_box_simplex(np.array(raw), weight_lower, weight_upper)
    trial.set_user_attr("normalized_weights", weights.tolist())

    port = portfolio_period_returns(returns_tensor, weights)
    terminal = np.prod(1.0 + port, axis=0) - 1.0

    if p_1_constraint is not None:
        trial.set_user_attr("p1_terminal", float(np.percentile(terminal, 1)))
    if p_5_constraint is not None:
        trial.set_user_attr("p5_terminal", float(np.percentile(terminal, 5)))
    if max_std is not None:
        trial.set_user_attr("terminal_std", float(terminal.std()))

    value = apply_metric(
        metric_key,
        port,
        periods_per_year,
        metric_params=metric_params,
        benchmark_returns=benchmark_returns,
    )
    return value


def _constraints_func(
    trial: optuna.trial.FrozenTrial,
    p_1_constraint: float | None,
    p_5_constraint: float | None,
    max_std: float | None,
) -> list[float]:
    out: list[float] = []
    if p_1_constraint is not None:
        p1 = trial.user_attrs.get("p1_terminal", float("-inf"))
        out.append(p_1_constraint - p1)
    if p_5_constraint is not None:
        p5 = trial.user_attrs.get("p5_terminal", float("-inf"))
        out.append(p_5_constraint - p5)
    if max_std is not None:
        std = trial.user_attrs.get("terminal_std", float("inf"))
        out.append(std - max_std)
    return out


def run_optuna_study(
    returns_tensor: np.ndarray,
    assets: list[str],
    metric_key: str,
    periods_per_year: float,
    metric_params: dict[str, Any] | None,
    benchmark_returns: np.ndarray | None,
    weight_bounds: dict[str, tuple[float, float]] | None,
    p_1_constraint: float | None,
    p_5_constraint: float | None,
    max_std: float | None,
    n_trials: int,
    random_seed: int | None,
    progress_callback: Callable[[int, float, float], None] | None,
    cancel_event: Any | None,
) -> tuple[optuna.Study, list[dict[str, float]]]:
    n_assets = len(assets)
    lower = np.zeros(n_assets)
    upper = np.ones(n_assets)
    if weight_bounds:
        for i, a in enumerate(assets):
            if a in weight_bounds:
                lo, hi = weight_bounds[a]
                lower[i] = lo
                upper[i] = hi

    def objective(trial: optuna.trial.Trial) -> float:
        if cancel_event is not None and cancel_event.is_set():
            trial.study.stop()
            return float("-inf")
        return bootstrap_portfolio_optuna_objective(
            trial,
            returns_tensor,
            assets,
            metric_key,
            periods_per_year,
            metric_params,
            benchmark_returns,
            lower,
            upper,
            p_1_constraint,
            p_5_constraint,
            max_std,
        )

    def constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
        return _constraints_func(
            trial,
            p_1_constraint,
            p_5_constraint,
            max_std,
        )

    sampler = TPESampler(
        seed=random_seed,
        constraints_func=(
            constraints
            if any(
                x is not None
                for x in (p_1_constraint, p_5_constraint, max_std)
            )
            else None
        ),
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: list[dict[str, float]] = []
    best = float("-inf")

    def callback(
        study_: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        nonlocal best
        if cancel_event is not None and cancel_event.is_set():
            study_.stop()
            return
        if trial.value is None:
            return
        if trial.value > best:
            best = trial.value
        history.append(
            {
                "trial": float(trial.number),
                "value": float(trial.value),
                "best_value": float(best),
            }
        )
        if progress_callback is not None:
            progress_callback(trial.number, float(trial.value), float(best))

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    return study, history
