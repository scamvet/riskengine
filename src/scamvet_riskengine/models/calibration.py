"""Probability calibration.

The registry maps verdict bands from probabilities, so a model whose 0.9 does
not mean 0.9 produces bands that lie to users. Calibration is therefore part of
the model, not a post-hoc nicety.

Implemented explicitly rather than via ``CalibratedClassifierCV`` for two
reasons. First, the prefit-estimator API has churned across recent scikit-learn
releases, and this is load-bearing code that should not break on a minor
upgrade. Second, the fitting data matters enormously here and an explicit
signature makes it impossible to get wrong by accident: a calibrator fitted on
the training split learns the model's training-set overconfidence and cancels
it, producing a calibration curve that looks excellent and is meaningless. The
calibration split exists for this and nothing else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibrationError(ValueError):
    """Raised on unfitted use or malformed calibration input."""


def _check(scores: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise CalibrationError("empty scores")
    if not np.isfinite(scores).all():
        raise CalibrationError("scores contain NaN or infinity")
    if y is not None:
        y = np.asarray(y).ravel()
        if y.shape != scores.shape:
            raise CalibrationError(f"shape mismatch: {y.shape} vs {scores.shape}")
        if len(set(np.unique(y).tolist())) < 2:
            raise CalibrationError("calibration data has only one class")
    return scores


class PlattCalibrator:
    """Sigmoid calibration: a one-dimensional logistic fit on model scores.

    Parametric, so it needs little data and cannot overfit badly. It assumes
    the miscalibration is sigmoid-shaped, which is a real assumption and the
    reason isotonic is offered alongside it.
    """

    method = "platt"

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        scores = _check(scores, y)
        self._model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._model.fit(scores.reshape(-1, 1), np.asarray(y).ravel())
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise CalibrationError("calibrator is not fitted")
        scores = _check(scores)
        return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]

    def to_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise CalibrationError("calibrator is not fitted")
        return {
            "method": self.method,
            "coef": float(self._model.coef_.ravel()[0]),
            "intercept": float(self._model.intercept_.ravel()[0]),
        }


class IsotonicCalibrator:
    """Non-parametric calibration: a monotone step function on model scores.

    Makes no shape assumption, so it fixes miscalibration a sigmoid cannot.
    The cost is that it can overfit a small calibration split, and it is
    piecewise-constant, which coarsens the score resolution near the tails.
    """

    method = "isotonic"

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        scores = _check(scores, y)
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(scores, np.asarray(y).ravel().astype(float))
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise CalibrationError("calibrator is not fitted")
        scores = _check(scores)
        return np.clip(self._model.predict(scores), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise CalibrationError("calibrator is not fitted")
        return {
            "method": self.method,
            "n_thresholds": int(len(self._model.X_thresholds_)),
        }


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def build_calibrator(method: str) -> PlattCalibrator | IsotonicCalibrator:
    """Construct a calibrator by name."""
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise CalibrationError(
        f"unknown calibration method {method!r}; expected one of {sorted(CALIBRATORS)}"
    )
