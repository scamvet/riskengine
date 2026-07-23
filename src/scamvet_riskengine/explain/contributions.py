"""Per-prediction and global feature attributions.

Uses each library's native exact-TreeSHAP path: LightGBM's
``predict(pred_contrib=True)`` and XGBoost's ``Booster.predict(pred_contribs=True)``.
Both are exact; they differ in arithmetic precision. LightGBM reconciles to the
raw margin within about 5e-15, XGBoost within about 2e-06 because its
contribution path is float32 internally. The default tolerance accommodates the
looser of the two rather than silently passing a real divergence in the tighter
one.

This deliberately avoids the ``shap`` package. The numbers are identical - both
implement TreeSHAP - but ``shap`` pulls in numba and llvmlite, which is a heavy
transitive burden for a library whose whole point is being cheap to install and
whose scoring path must stay small enough to run on a phone. The package
remains an optional extra for plotting; the attribution numbers do not need it.

Attributions here are the raw material for :mod:`scamvet_riskengine.explain.reasons`,
which turns them into the user-facing ``reasons[]`` contract that Core's
Explainer agent consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


#: Reconciliation tolerance. LightGBM achieves ~5e-15; XGBoost ~2e-06, because
#: its contribution path is float32 internally. Set for the looser of the two,
#: which is still four orders of magnitude tighter than any error that could
#: change a reason.
ADDITIVITY_TOLERANCE = 1e-4

#: Minimum active rows before a feature is ranked by conditional importance.
#: A mean over a handful of observations is noise, and this ranking exists to
#: surface real rare-but-sharp features, not to promote accidents.
MIN_ACTIVE_FOR_RANKING = 30


class ExplainError(ValueError):
    """Raised when attributions cannot be computed or do not reconcile."""


@dataclass(frozen=True)
class GlobalImportance:
    """Mean absolute attribution per feature, over a sample."""

    feature_names: tuple[str, ...]
    mean_abs: tuple[float, ...]
    n_samples: int
    #: Mean absolute attribution restricted to rows where the feature is
    #: non-zero, and how many such rows there were. Frequency-weighted
    #: importance buries rare features: brand_domain_mismatch fires on 0.44%
    #: of rows and ranks 56th overall while being one of the sharpest signals
    #: when it does fire. Conditional importance is what says whether a rare
    #: feature works.
    mean_abs_when_active: tuple[float, ...] = ()
    n_active: tuple[int, ...] = ()

    def ranked(self, top: int | None = None) -> list[tuple[str, float]]:
        pairs = sorted(
            zip(self.feature_names, self.mean_abs, strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return pairs[:top] if top else pairs

    def ranked_when_active(
        self, top: int | None = None, min_active: int = MIN_ACTIVE_FOR_RANKING
    ) -> list[tuple[str, float, int]]:
        """Features ranked by attribution among the rows where they are active.

        The ranking that matters for rare features. Frequency-weighted
        importance divides by every row, so a feature firing on 0.44% of the
        corpus looks negligible even when it dominates the rows it touches.

        Features active in fewer than ``min_active`` rows are excluded. Without
        that guard this ranking does the thing it was built to prevent: on a
        20,000-row sample ``fragment_length`` placed second at 2.686 while
        being active in two rows, which is a coin flip presented as a finding.
        """
        triples = sorted(
            (
                item
                for item in zip(
                    self.feature_names, self.mean_abs_when_active, self.n_active, strict=True
                )
                if item[2] >= min_active
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return triples[:top] if top else triples

    def as_dict(self) -> dict[str, Any]:
        active = dict(
            zip(
                self.feature_names,
                zip(self.mean_abs_when_active, self.n_active, strict=True),
                strict=True,
            )
        )
        return {
            "n_samples": self.n_samples,
            "features": [
                {
                    "feature": name,
                    "mean_abs_contribution": value,
                    "mean_abs_when_active": active[name][0],
                    "n_active": active[name][1],
                }
                for name, value in self.ranked()
            ],
        }


def _raw_contributions(model: Any, X: np.ndarray) -> np.ndarray:
    """Dispatch to the fitted library's native exact-TreeSHAP path."""
    if hasattr(model, "get_booster"):  # XGBoost sklearn wrapper
        import xgboost as xgb

        booster = model.get_booster()
        return np.asarray(booster.predict(xgb.DMatrix(X), pred_contribs=True), dtype=float)
    if hasattr(model, "booster_"):  # LightGBM sklearn wrapper
        return np.asarray(model.predict(X, pred_contrib=True), dtype=float)
    raise ExplainError(
        f"no known TreeSHAP path for {type(model).__name__}; expected a fitted "
        "LightGBM or XGBoost classifier"
    )


def raw_margin(model: Any, X: np.ndarray) -> np.ndarray:
    """The model's uncalibrated margin, for reconciling attributions."""
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        return np.asarray(
            model.get_booster().predict(xgb.DMatrix(X), output_margin=True), dtype=float
        ).ravel()
    return np.asarray(model.predict(X, raw_score=True), dtype=float).ravel()


def contributions(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP contributions for a fitted LightGBM or XGBoost classifier.

    Returns:
        (values, base) where ``values`` has shape (n_samples, n_features) and
        ``base`` has shape (n_samples,). For every row,
        ``values[i].sum() + base[i]`` equals the model's raw margin.

    Raises:
        ExplainError: if the returned array is not the expected shape, which
            would mean the library changed its contract underneath us.
    """
    raw = _raw_contributions(model, X)
    if raw.ndim != 2 or raw.shape[0] != X.shape[0]:
        raise ExplainError(f"unexpected contribution shape {raw.shape} for input {X.shape}")
    if raw.shape[1] != X.shape[1] + 1:
        raise ExplainError(
            f"expected {X.shape[1] + 1} columns (features plus base), got {raw.shape[1]}"
        )
    return raw[:, :-1], raw[:, -1]


def verify_additivity(model: Any, X: np.ndarray, tolerance: float = ADDITIVITY_TOLERANCE) -> float:
    """Check that contributions reconcile to the model's own output.

    TreeSHAP's defining property is additivity. If it does not hold, the
    attributions are not explanations of this model, and a reason string built
    from them would be a confident fabrication - the exact failure mode the
    guardrail layer exists to prevent.

    Returns:
        The maximum absolute reconciliation error.
    """
    values, base = contributions(model, X)
    margin = raw_margin(model, X)
    error = float(np.max(np.abs(values.sum(axis=1) + base - margin)))
    if error > tolerance:
        raise ExplainError(
            f"contributions do not reconcile to the model output: max error {error:.3e} "
            f"exceeds tolerance {tolerance:.1e}"
        )
    return error


def global_importance(
    model: Any, X: np.ndarray, feature_names: list[str], max_samples: int = 20_000
) -> GlobalImportance:
    """Mean absolute attribution per feature over a sample of rows.

    Sampled because attribution over a full corpus is expensive and the ranking
    stabilises long before the sample is exhausted. The sample size is recorded
    so the figure is reproducible.
    """
    if len(feature_names) != X.shape[1]:
        raise ExplainError(f"{len(feature_names)} feature names but {X.shape[1]} columns")
    sample = X[:max_samples]
    values, _ = contributions(model, sample)
    absolute = np.abs(values)

    active_means: list[float] = []
    active_counts: list[int] = []
    for column in range(sample.shape[1]):
        mask = sample[:, column] != 0
        count = int(mask.sum())
        active_counts.append(count)
        active_means.append(float(absolute[mask, column].mean()) if count else 0.0)

    return GlobalImportance(
        feature_names=tuple(feature_names),
        mean_abs=tuple(float(v) for v in absolute.mean(axis=0)),
        n_samples=int(sample.shape[0]),
        mean_abs_when_active=tuple(active_means),
        n_active=tuple(active_counts),
    )
