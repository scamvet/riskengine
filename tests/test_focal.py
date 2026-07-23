"""The focal derivatives must match the loss they claim to differentiate.

A wrong gradient does not raise. XGBoost accepts whatever it is handed, trains
to completion, and produces a model that is merely worse - so an analytic
derivative that nobody checked is an unmeasured weakness of exactly the kind
this project refuses to ship. Both derivatives are therefore compared against
central differences of the function one level up.
"""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.models.focal import (
    focal_grad_hess,
    focal_loss,
    looks_like_labels,
    sigmoid,
    xgboost_focal_objective,
)

GAMMAS = (0.0, 0.5, 1.0, 2.0)
MARGINS = np.array([-6.0, -2.5, -0.75, -0.1, 0.0, 0.1, 0.75, 2.5, 6.0])


def _central_difference(function, margin: np.ndarray, step: float = 1e-5) -> np.ndarray:
    return (function(margin + step) - function(margin - step)) / (2.0 * step)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("label", [0, 1])
def test_gradient_matches_finite_difference(gamma: float, label: int) -> None:
    y = np.full(MARGINS.shape, label)
    analytic, _ = focal_grad_hess(y, MARGINS, gamma)
    numeric = _central_difference(lambda z: focal_loss(y, z, gamma), MARGINS)
    assert np.allclose(analytic, numeric, atol=1e-5), f"gamma={gamma} label={label}"


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("label", [0, 1])
def test_hessian_matches_finite_difference(gamma: float, label: int) -> None:
    y = np.full(MARGINS.shape, label)
    _, analytic = focal_grad_hess(y, MARGINS, gamma)
    numeric = _central_difference(lambda z: focal_grad_hess(y, z, gamma)[0], MARGINS)
    # The clip to MIN_HESSIAN is deliberate, so only compare where it is inactive.
    comparable = analytic > 1e-5
    assert np.allclose(
        analytic[comparable], numeric[comparable], atol=1e-4
    ), f"gamma={gamma} label={label}"


def test_gamma_zero_reduces_to_log_loss() -> None:
    """At gamma=0 focal loss is ordinary log loss, so the gradient is p - y."""
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
    grad, _ = focal_grad_hess(y, MARGINS, 0.0)
    assert np.allclose(grad, sigmoid(MARGINS) - y, atol=1e-9)


def test_hessian_is_strictly_positive() -> None:
    for gamma in GAMMAS:
        for label in (0, 1):
            _, hess = focal_grad_hess(np.full(MARGINS.shape, label), MARGINS, gamma)
            assert np.all(hess > 0.0), f"non-positive curvature at gamma={gamma}"


def test_objective_detects_argument_order() -> None:
    """Either calling convention must produce the same derivatives."""
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
    objective = xgboost_focal_objective(2.0)
    forward = objective(y, MARGINS)
    reversed_order = objective(MARGINS, y)
    assert np.allclose(forward[0], reversed_order[0])
    assert np.allclose(forward[1], reversed_order[1])


def test_objective_rejects_two_non_label_arrays() -> None:
    with pytest.raises(ValueError, match="binary labels"):
        xgboost_focal_objective(1.0)(MARGINS, MARGINS)


def test_negative_gamma_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        xgboost_focal_objective(-1.0)


def test_uniform_margins_are_not_mistaken_for_labels() -> None:
    """At the first boosting round every margin is identical."""
    assert not looks_like_labels(np.zeros(10))
    assert not looks_like_labels(np.full(10, 0.5))
    assert looks_like_labels(np.array([0, 1, 1, 0]))


def test_sigmoid_is_stable_at_extremes() -> None:
    extreme = np.array([-1e4, -800.0, 0.0, 800.0, 1e4])
    values = sigmoid(extreme)
    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
