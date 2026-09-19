import numpy as np


def project_box_simplex(
    raw: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Project raw weights onto {w | lower <= w <= upper, sum(w) == 1}.
    Falls back to uniform on the feasible set if projection fails.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    w = np.clip(np.asarray(raw, dtype=float), lower, upper)

    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("Infeasible weight bounds: cannot sum to 1")

    for _ in range(max_iter):
        total = w.sum()
        if abs(total - 1.0) < 1e-9:
            return w
        if total > 1.0:
            excess = total - 1.0
            room = w - lower
            denom = room.sum()
            if denom <= 1e-12:
                break
            w -= excess * (room / denom)
        else:
            deficit = 1.0 - total
            room = upper - w
            denom = room.sum()
            if denom <= 1e-12:
                break
            w += deficit * (room / denom)
        w = np.clip(w, lower, upper)

    # Uniform fallback within bounds
    w = (lower + upper) / 2.0
    w /= w.sum()
    return np.clip(w, lower, upper)
