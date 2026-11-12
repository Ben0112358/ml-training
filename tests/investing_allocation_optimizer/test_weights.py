import numpy as np
import pytest

from ml_training.investing_allocation_optimizer.utils.weights import (
    project_box_simplex,
)


def test_raises_value_error_when_lower_bounds_sum_above_one():
    with pytest.raises(ValueError):
        project_box_simplex(
            np.array([0.5, 0.5]),
            lower=np.array([0.6, 0.6]),
            upper=np.array([1.0, 1.0]),
        )


def test_raises_value_error_when_upper_bounds_sum_below_one():
    with pytest.raises(ValueError):
        project_box_simplex(
            np.array([0.5, 0.5]),
            lower=np.array([0.0, 0.0]),
            upper=np.array([0.4, 0.4]),
        )


def test_scales_down_weights_above_one_to_sum_one_within_bounds():
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([0.5, 0.5, 0.5])
    w = project_box_simplex(np.array([0.9, 0.9, 0.9]), lower, upper)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= lower)
    assert np.all(w <= upper)


def test_scales_up_weights_below_one_to_sum_one_within_bounds():
    lower = np.array([0.1, 0.1, 0.1])
    upper = np.array([1.0, 1.0, 1.0])
    w = project_box_simplex(np.array([0.0, 0.0, 0.0]), lower, upper)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= lower)
    assert np.all(w <= upper)
