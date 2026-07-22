"""The baseline model.

A calibrated logistic regression. Its job is not to win; its job to be the bar
that every more complex model has to clear before it earns its place in the
registry. A gradient-boosted ensemble that beats a linear model by half a point
of PR-AUC has not justified its training cost, its inference cost, or its
opacity, and the benchmark should say so plainly.

Linear also means the coefficients are directly readable, which is a useful
sanity check on the feature set before any SHAP work: if the sign of a
coefficient contradicts the reason the feature was engineered, either the
feature or the reasoning is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ImbalanceStrategy = Literal["none", "balanced"]


@dataclass(frozen=True)
class BaselineConfig:
    """Everything that determines the fitted model, recorded in the manifest."""

    imbalance: ImbalanceStrategy = "none"
    C: float = 1.0
    max_iter: int = 1000
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "logistic_regression",
            "imbalance": self.imbalance,
            "C": self.C,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }


def build_baseline(config: BaselineConfig | None = None) -> Pipeline:
    """Standardise then fit a logistic regression.

    Scaling is not optional here: the feature set mixes character counts in the
    hundreds with ratios in [0, 1] and one-hot columns, and an unscaled
    L2-penalised fit would effectively regularise the small-magnitude features
    out of existence regardless of their predictive value.
    """
    config = config or BaselineConfig()
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    class_weight=None if config.imbalance == "none" else "balanced",
                    random_state=config.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def decision_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Uncalibrated scores, the input to a calibrator.

    ``decision_function`` rather than ``predict_proba``: logistic regression
    already emits probabilities, and calibrating a probability with another
    logistic squashes the tails twice. The raw margin is the right input.
    """
    return np.asarray(pipeline.decision_function(X)).ravel()


@dataclass(frozen=True)
class Coefficient:
    """One fitted weight, named."""

    feature: str
    coefficient: float

    def as_dict(self) -> dict[str, float | str]:
        return {"feature": self.feature, "coefficient": self.coefficient}


def coefficient_table(pipeline: Pipeline, feature_names: list[str]) -> list[Coefficient]:
    """Coefficients sorted by magnitude, for the readability sanity check."""
    coefficients = np.asarray(pipeline.named_steps["clf"].coef_).ravel()
    if len(coefficients) != len(feature_names):
        raise ValueError(f"{len(coefficients)} coefficients but {len(feature_names)} feature names")
    rows = [
        Coefficient(feature=name, coefficient=float(value))
        for name, value in zip(feature_names, coefficients, strict=True)
    ]
    rows.sort(key=lambda row: abs(row.coefficient), reverse=True)
    return rows
