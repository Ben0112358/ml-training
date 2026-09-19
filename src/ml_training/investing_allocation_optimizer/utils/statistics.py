from typing import Any

import numpy as np


def _max_drawdown_path(period_returns: np.ndarray) -> float:
    """period_returns 1d -> max drawdown (negative number)."""
    wealth = np.cumprod(1.0 + period_returns)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(np.min(dd))


def portfolio_stats(
    period_returns: np.ndarray,
    periods_per_year: float,
) -> dict[str, float]:
    """
    period_returns shape (T, P) — bootstrap paths in columns.
    """
    T, P = period_returns.shape
    years = T / periods_per_year if periods_per_year > 0 else 0.0
    terminal = np.prod(1.0 + period_returns, axis=0) - 1.0

    per_path_mean = period_returns.mean(axis=0) * periods_per_year
    per_path_vol = period_returns.std(axis=0, ddof=1) * np.sqrt(
        periods_per_year
    )

    if years > 0:
        cagr_paths = (1.0 + terminal) ** (1.0 / years) - 1.0
        mean_cagr = float(np.mean(cagr_paths))
    else:
        mean_cagr = float("nan")

    mdd = [_max_drawdown_path(period_returns[:, p]) for p in range(P)]
    neg = np.minimum(period_returns, 0.0)
    ds = neg.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
    mean_ret = float(np.mean(per_path_mean))
    mean_vol = float(np.mean(per_path_vol))
    sharpe = mean_ret / mean_vol if mean_vol > 1e-12 else float("nan")
    mean_ds = float(np.mean(ds))
    sortino = mean_ret / mean_ds if mean_ds > 1e-12 else float("nan")

    return {
        "mean_cagr": mean_cagr,
        "mean_terminal_return": float(np.mean(terminal)),
        "median_terminal_return": float(np.median(terminal)),
        "annualized_volatility": mean_vol,
        "p1_terminal_return": float(np.percentile(terminal, 1)),
        "p5_terminal_return": float(np.percentile(terminal, 5)),
        "p95_terminal_return": float(np.percentile(terminal, 95)),
        "prob_loss": float(np.mean(terminal < 0)),
        "median_max_drawdown": float(np.median(mdd)),
        "sharpe": sharpe,
        "sortino": sortino,
    }


def stats_table_row(
    name: str,
    stats: dict[str, float],
) -> dict[str, Any]:
    return {"portfolio": name, **stats}
