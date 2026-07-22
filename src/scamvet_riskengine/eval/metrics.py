"""The frozen metric protocol for RiskEngine (ADR-009).

Fixed before any model is trained, so that the choice of metric cannot be made
after seeing which one flatters a result. Every benchmark table, model card and
published figure reports exactly these numbers, computed by this module.

Design notes worth keeping in one place:

**PR-AUC is primary, ROC-AUC is reported but never headlined.** At the positive
rates in this corpus, ROC-AUC is dominated by the enormous true-negative pool
and stays high for models with poor precision. Two models a full order of
magnitude apart in usable precision can differ by a couple of points of
ROC-AUC. Precision-recall does not have that blind spot.

**Recall at a fixed false-positive rate is the operating-point metric.** An
aggregate curve summary does not tell you what a deployment gets. ScamVet's
consumer verdict is precision-leaning because a false red destroys trust in a
way a false amber does not, so recall at FPR 1% and 5% is what actually
constrains the product.

**Calibration is scored, not assumed.** The verdict bands in the registry map
from a calibrated probability. A model whose 0.9 does not mean 0.9 produces
bands that lie, so Brier and expected calibration error are first-class here
rather than diagnostics.

**Cost is stated explicitly.** Any threshold choice implies a ratio between
the cost of missing a scam and the cost of alarming a user wrongly. Making the
ratio an argument forces it to be written down and argued about instead of
being smuggled in as a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

#: Operating points reported for every model.
FPR_TARGETS: tuple[float, ...] = (0.01, 0.05)

#: Bin count for expected calibration error. Fixed so ECE is comparable
#: across models and releases; changing it changes every published number.
ECE_BINS: int = 15

#: Default cost ratio: missing a scam is treated as ten times worse than a
#: false alarm. Stated, not hidden. Overridden per use case in the registry.
DEFAULT_FN_FP_COST_RATIO: float = 10.0


class MetricError(ValueError):
    """Raised when inputs cannot support the requested metric."""


@dataclass(frozen=True)
class EvalResult:
    """Every number the protocol requires, for one model on one split."""

    n: int
    n_positive: int
    positive_rate: float
    pr_auc: float
    roc_auc: float
    brier: float
    ece: float
    recall_at_fpr: dict[str, float]
    threshold_at_fpr: dict[str, float]
    precision_at_fpr: dict[str, float]
    expected_cost: float
    cost_ratio: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_line(self) -> str:
        """One-line form used in logs and benchmark tables."""
        r1 = self.recall_at_fpr.get("0.01", float("nan"))
        r5 = self.recall_at_fpr.get("0.05", float("nan"))
        return (
            f"PR-AUC {self.pr_auc:.4f} | R@FPR1% {r1:.4f} | R@FPR5% {r5:.4f} | "
            f"Brier {self.brier:.4f} | ECE {self.ece:.4f} | ROC-AUC {self.roc_auc:.4f}"
        )


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise MetricError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.size == 0:
        raise MetricError("empty input")
    if not np.isfinite(y_score).all():
        raise MetricError("y_score contains NaN or infinity")
    unique = set(np.unique(y_true).tolist())
    if not unique <= {0, 1}:
        raise MetricError(f"y_true must be binary 0/1, found {sorted(unique)}")
    if len(unique) < 2:
        raise MetricError("y_true has only one class; ranking metrics are undefined")
    return y_true, y_score


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Recall achievable at or below a target false-positive rate.

    Returns:
        (recall, threshold). The threshold is the score cut that realises it,
        which is the number a deployment actually configures.
    """
    y_true, y_score = _validate(y_true, y_score)
    if not 0.0 < target_fpr < 1.0:
        raise MetricError(f"target_fpr must be in (0, 1), got {target_fpr}")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    admissible = fpr <= target_fpr
    if not admissible.any():
        return 0.0, float("inf")
    index = int(np.max(np.flatnonzero(admissible)))
    return float(tpr[index]), float(thresholds[index])


def precision_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Precision when everything scoring at or above ``threshold`` is flagged."""
    y_true, y_score = _validate(y_true, y_score)
    flagged = y_score >= threshold
    if not flagged.any():
        return 0.0
    return float(y_true[flagged].mean())


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Weighted mean gap between confidence and accuracy across equal-width bins.

    Requires probabilities, not arbitrary scores. An uncalibrated decision
    function scored here produces a meaningless number, so the range is
    checked rather than clipped.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise MetricError(
            "expected_calibration_error requires probabilities in [0, 1]; "
            "calibrate the model's scores first"
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; the closed right edge keeps 1.0 in the last bin.
    indices = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if not count:
            continue
        gap = abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
        total += (count / y_true.size) * gap
    return float(total)


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    fn_fp_ratio: float = DEFAULT_FN_FP_COST_RATIO,
) -> float:
    """Mean cost per sample at a threshold, with false negatives weighted.

    A false positive costs 1. A false negative costs ``fn_fp_ratio``. The
    ratio is an argument because it is a product decision, not a statistical
    one, and it must appear in the model card next to the threshold it justifies.
    """
    y_true, y_score = _validate(y_true, y_score)
    if fn_fp_ratio <= 0:
        raise MetricError("fn_fp_ratio must be positive")
    predicted = (y_score >= threshold).astype(int)
    false_pos = int(((predicted == 1) & (y_true == 0)).sum())
    false_neg = int(((predicted == 0) & (y_true == 1)).sum())
    return float((false_pos + fn_fp_ratio * false_neg) / y_true.size)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    fpr_targets: tuple[float, ...] = FPR_TARGETS,
    fn_fp_ratio: float = DEFAULT_FN_FP_COST_RATIO,
    cost_threshold: float | None = None,
) -> EvalResult:
    """Compute the full protocol for one model on one split.

    Args:
        y_true: binary ground truth.
        y_prob: calibrated probabilities. Brier and ECE are only meaningful
            on probabilities; passing raw decision-function output raises.
        fpr_targets: operating points to report.
        fn_fp_ratio: cost of a false negative relative to a false positive.
        cost_threshold: threshold at which expected cost is reported. Defaults
            to the threshold realising the tightest FPR target, so the cost
            figure corresponds to an operating point rather than to 0.5.
    """
    y_true, y_prob = _validate(y_true, y_prob)

    recalls: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    precisions: dict[str, float] = {}
    for target in fpr_targets:
        recall, threshold = recall_at_fpr(y_true, y_prob, target)
        key = str(target)
        recalls[key] = recall
        thresholds[key] = threshold
        precisions[key] = (
            precision_at_threshold(y_true, y_prob, threshold) if np.isfinite(threshold) else 0.0
        )

    if cost_threshold is None:
        finite = [t for t in thresholds.values() if np.isfinite(t)]
        cost_threshold = min(finite) if finite else 0.5

    return EvalResult(
        n=int(y_true.size),
        n_positive=int(y_true.sum()),
        positive_rate=float(y_true.mean()),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob),
        recall_at_fpr=recalls,
        threshold_at_fpr=thresholds,
        precision_at_fpr=precisions,
        expected_cost=expected_cost(y_true, y_prob, cost_threshold, fn_fp_ratio),
        cost_ratio=float(fn_fp_ratio),
    )


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS
) -> list[dict[str, float]]:
    """Per-bin confidence and observed frequency, for reliability diagrams.

    Returned as data rather than plotted here: the model card commits the
    numbers, and the plot is regenerated from them.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": float(count),
                "mean_predicted": float(y_prob[mask].mean()) if count else 0.0,
                "observed_rate": float(y_true[mask].mean()) if count else 0.0,
            }
        )
    return bins
