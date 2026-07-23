"""Focal loss for gradient boosting.

Focal loss down-weights the examples a model already classifies confidently, so
each boosting round concentrates on the ones it gets wrong. It came from dense
object detection, where background boxes outnumber objects by orders of
magnitude, and it is a reflexive recommendation whenever a classification
problem is described as imbalanced.

It is implemented here to be measured, not adopted. The product corpus is
18.63% positive on the phishing view - mild by the standards this technique was
built for - and it carries label errors in both directions. Focal loss
up-weights hard examples by construction, and a mislabelled row is maximally
hard, so the mechanism that helps under clean labels is the same one that
chases noise here. Which effect dominates is an empirical question, and
``scripts/imbalance_study.py`` is where it gets answered.

XGBoost has no built-in focal objective. Gradient and hessian are therefore
supplied analytically below and verified against finite differences of the loss
in ``tests/test_focal.py``, because a wrong derivative trains without raising
and produces a model that is merely worse - the kind of failure no exception
reports.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: Probabilities are clipped away from 0 and 1 before any logarithm.
EPS = 1e-7

#: XGBoost requires strictly positive curvature to form a split.
MIN_HESSIAN = 1e-6


def sigmoid(margin: np.ndarray) -> np.ndarray:
    """Logistic function, computed through tanh to avoid overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(margin, dtype=float)))


def focal_loss(y_true: np.ndarray, margin: np.ndarray, gamma: float) -> np.ndarray:
    """Per-example focal loss. Present so the derivatives can be tested."""
    p = np.clip(sigmoid(margin), EPS, 1.0 - EPS)
    positive = -((1.0 - p) ** gamma) * np.log(p)
    negative = -(p**gamma) * np.log(1.0 - p)
    return np.where(np.asarray(y_true) == 1, positive, negative)


def _positive_terms(p: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and hessian of the y=1 focal loss with respect to the margin.

    With ``p = sigmoid(z)`` and ``L = -(1 - p)^g * log(p)``::

        dL/dz = g * (1-p)^g * p * log(p) - (1-p)^(g+1)

    The y=0 case is the same function under ``p -> 1 - p`` with the gradient
    negated, which is why only one derivation appears here.
    """
    q = 1.0 - p
    log_p = np.log(p)
    grad = gamma * q**gamma * p * log_p - q ** (gamma + 1.0)
    hess = gamma * p * q * (-gamma * q ** (gamma - 1.0) * p * log_p + q**gamma * (log_p + 1.0)) + (
        gamma + 1.0
    ) * p * q ** (gamma + 1.0)
    return grad, hess


def focal_grad_hess(
    y_true: np.ndarray, margin: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and hessian of the focal loss with respect to the margin."""
    p = np.clip(sigmoid(margin), EPS, 1.0 - EPS)
    grad_positive, hess_positive = _positive_terms(p, gamma)
    grad_negative, hess_negative = _positive_terms(1.0 - p, gamma)
    is_positive = np.asarray(y_true) == 1
    grad = np.where(is_positive, grad_positive, -grad_negative)
    hess = np.where(is_positive, hess_positive, hess_negative)
    return grad, np.maximum(hess, MIN_HESSIAN)


def looks_like_labels(values: np.ndarray) -> bool:
    """Whether an array is plausibly a binary label vector.

    Requires exactly two distinct values, both in {0, 1}. Exactly two matters:
    at the first boosting round every margin is the same number, so a one-value
    array is a margin vector rather than labels.
    """
    unique = np.unique(np.asarray(values, dtype=float))
    return unique.size == 2 and bool(np.all(np.isin(unique, (0.0, 1.0))))


def xgboost_focal_objective(gamma: float) -> Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Build an objective callable for XGBoost's scikit-learn API.

    Argument order is detected rather than assumed. XGBoost hands sklearn-API
    objectives ``(y_true, y_pred)`` and native-API objectives ``(y_pred,
    dtrain)``, the convention has moved between major versions, and getting it
    backwards trains happily while optimising nonsense. Labels are binary and
    margins are not, so the two are separable at runtime.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    def objective(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
        left = np.asarray(first, dtype=float).ravel()
        right = np.asarray(second, dtype=float).ravel()
        if looks_like_labels(left):
            return focal_grad_hess(left, right, gamma)
        if looks_like_labels(right):
            return focal_grad_hess(right, left, gamma)
        raise ValueError(
            "neither argument to the focal objective looks like binary labels; "
            "XGBoost's calling convention may have changed"
        )

    return objective
