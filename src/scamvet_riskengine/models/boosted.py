"""Gradient-boosted model families.

These exist to be measured against the linear baseline, not assumed better.
The question the benchmark answers is narrow and worth stating plainly: does
tree boosting buy enough on *phishing* to justify a model that is slower to
train, larger to ship, and harder to explain than a logistic regression?

Phishing is where interactions should matter. A domain with a brand token is
unremarkable; a brand token plus a suspicious TLD plus several hyphens plus no
HTTPS is not. A linear model cannot represent that conjunction; a tree can. If
the boosted models do not separate from the baseline on phishing recall, the
honest conclusion is that the lexical feature set is the ceiling, not the
model class — and that conclusion is worth more than a tuned leaderboard entry.

No scaling here: trees are invariant to monotone feature transforms, so the
StandardScaler the baseline needs would be pure overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ModelName = Literal["logistic", "xgboost", "lightgbm"]


@dataclass(frozen=True)
class BoostedConfig:
    """Hyperparameters, frozen per benchmark run and recorded in the manifest.

    Deliberately modest and identical in spirit across both libraries, so the
    benchmark compares model families rather than tuning effort. Optuna search
    comes later, on top of a protocol that is already fixed.

    ``n_estimators`` is 100 rather than 400 **for size, not quality**. At 400
    the ONNX export is 877 KB, over the 500 KB committed-artifact cap, forcing
    a separate smaller model for the offline path; at 100 it is 217 KB and one
    model serves both paths. Measured across four seeds the two configurations
    do not separate - the gap in mean phishing recall is roughly 0.0005 while
    the seed spread is roughly 0.013 - so the smaller model is free, not
    better. An earlier reading of a single seed suggested 400 was overfitting;
    it was not, and that claim should not be revived without multi-seed
    evidence (ADR-009, amendment of 2026-07-23).
    """

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 8
    subsample: float = 0.9
    colsample: float = 0.9
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    imbalance: Literal["none", "balanced"] = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample": self.colsample,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "imbalance": self.imbalance,
            "seed": self.seed,
            **self.extra,
        }


def _scale_pos_weight(y: np.ndarray, imbalance: str) -> float:
    if imbalance != "balanced":
        return 1.0
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    return negatives / positives if positives else 1.0


def build_xgboost(config: BoostedConfig, y_train: np.ndarray) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=_scale_pos_weight(y_train, config.imbalance),
        random_state=config.seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="aucpr",
    )


def build_lightgbm(config: BoostedConfig, y_train: np.ndarray) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        is_unbalance=config.imbalance == "balanced",
        random_state=config.seed,
        n_jobs=-1,
        verbose=-1,
    )


def boosted_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Uncalibrated positive-class scores.

    Boosted classifiers expose ``predict_proba``, but those outputs are not
    calibrated - notably, subsampling and early stopping both distort them -
    so they are treated as scores and passed through a calibrator like any
    other decision function.
    """
    return np.asarray(model.predict_proba(X))[:, 1]
