from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

BENCHMARK_LABEL = "total-world"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    description: str
    fn: Callable[..., float]
    param_schema: dict[str, Any] | None = None


def _terminal_wealth(period_returns: np.ndarray) -> np.ndarray:
    """period_returns shape (T, P) -> terminal wealth (P,)"""
    return np.prod(1.0 + period_returns, axis=0)


def _cagr(period_returns: np.ndarray, periods_per_year: float) -> float:
    wealth = _terminal_wealth(period_returns)
    T = period_returns.shape[0]
    years = T / periods_per_year
    if years <= 0:
        return float("-inf")
    cagr_paths = wealth ** (1.0 / years) - 1.0
    return float(np.mean(cagr_paths))


def _annualized_mean(
    period_returns: np.ndarray, periods_per_year: float
) -> float:
    per_path = period_returns.mean(axis=0)
    return float(np.mean(per_path) * periods_per_year)


def _annualized_vol(
    period_returns: np.ndarray, periods_per_year: float
) -> float:
    per_path = period_returns.std(axis=0, ddof=1)
    return float(np.mean(per_path) * np.sqrt(periods_per_year))


def _sharpe(period_returns: np.ndarray, periods_per_year: float) -> float:
    vol = _annualized_vol(period_returns, periods_per_year)
    if vol <= 1e-12:
        return float("-inf")
    return _annualized_mean(period_returns, periods_per_year) / vol


def _downside_std(
    period_returns: np.ndarray, periods_per_year: float
) -> float:
    neg = np.minimum(period_returns, 0.0)
    per_path = neg.std(axis=0, ddof=1)
    return float(np.mean(per_path) * np.sqrt(periods_per_year))


def _sortino(period_returns: np.ndarray, periods_per_year: float) -> float:
    ds = _downside_std(period_returns, periods_per_year)
    if ds <= 1e-12:
        return float("-inf")
    return _annualized_mean(period_returns, periods_per_year) / ds


def _p5_terminal(period_returns: np.ndarray, periods_per_year: float) -> float:
    del periods_per_year
    terminal = _terminal_wealth(period_returns) - 1.0
    return float(np.percentile(terminal, 5))


def _cvar_5(period_returns: np.ndarray, periods_per_year: float) -> float:
    del periods_per_year
    terminal = _terminal_wealth(period_returns) - 1.0
    p5 = np.percentile(terminal, 5)
    tail = terminal[terminal <= p5]
    if len(tail) == 0:
        return float(p5)
    return float(np.mean(tail))


def _prob_beat_benchmark(
    period_returns: np.ndarray,
    periods_per_year: float,
    benchmark_returns: np.ndarray,
) -> float:
    del periods_per_year
    w = _terminal_wealth(period_returns)
    b = _terminal_wealth(benchmark_returns)
    return float(np.mean(w > b))


def apply_metric(
    key: str,
    portfolio_returns: np.ndarray,
    periods_per_year: float,
    metric_params: dict[str, Any] | None = None,
    benchmark_returns: np.ndarray | None = None,
) -> float:
    spec = METRICS[key]
    if key == "prob_beat_benchmark":
        if benchmark_returns is None:
            raise ValueError(
                "benchmark_returns required for prob_beat_benchmark"
            )
        return spec.fn(
            portfolio_returns,
            periods_per_year,
            benchmark_returns,
        )
    return spec.fn(portfolio_returns, periods_per_year)


METRICS: dict[str, MetricSpec] = {
    "mean_return": MetricSpec(
        key="mean_return",
        label="Mean annualized return",
        description="Average per-path mean return, scaled to a year",
        fn=_annualized_mean,
    ),
    "median_return": MetricSpec(
        key="median_return",
        label="Median terminal return",
        description="Median cumulative return across bootstrap paths",
        fn=lambda pr, ppy: float(
            np.median(_terminal_wealth(pr) - 1.0),
        ),
    ),
    "cagr": MetricSpec(
        key="cagr",
        label="Mean CAGR",
        description="Mean compound annual growth rate across paths",
        fn=_cagr,
    ),
    "sharpe": MetricSpec(
        key="sharpe",
        label="Sharpe ratio",
        description="Mean annualized return divided by annualized volatility",
        fn=_sharpe,
    ),
    "sortino": MetricSpec(
        key="sortino",
        label="Sortino ratio",
        description="Mean return divided by downside volatility",
        fn=_sortino,
    ),
    "p5_terminal": MetricSpec(
        key="p5_terminal",
        label="5th percentile terminal return",
        description=(
            "Terminal return at the 5% quantile (a single bad-scenario floor)"
        ),
        fn=_p5_terminal,
    ),
    "cvar_5": MetricSpec(
        key="cvar_5",
        label="CVaR 5% (terminal)",
        description=(
            "Average terminal return in the worst 5% of paths (tail mean, "
            "usually below the 5th percentile)"
        ),
        fn=_cvar_5,
    ),
    "prob_beat_benchmark": MetricSpec(
        key="prob_beat_benchmark",
        label="Prob. beat 100% VT",
        description=(
            "Share of paths beating 100% total-world on terminal wealth"
        ),
        fn=_prob_beat_benchmark,
    ),
}


def metrics_for_api() -> list[dict[str, Any]]:
    out = []
    for spec in METRICS.values():
        entry = {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
        }
        out.append(entry)
    return out
