"""Evaluation: the frozen metric protocol (ADR-009)."""

from scamvet_riskengine.eval.metrics import (
    DEFAULT_FN_FP_COST_RATIO,
    ECE_BINS,
    FPR_TARGETS,
    EvalResult,
    MetricError,
    evaluate,
    expected_calibration_error,
    expected_cost,
    precision_at_threshold,
    recall_at_fpr,
    reliability_bins,
)

__all__ = [
    "DEFAULT_FN_FP_COST_RATIO",
    "ECE_BINS",
    "FPR_TARGETS",
    "EvalResult",
    "MetricError",
    "evaluate",
    "expected_calibration_error",
    "expected_cost",
    "precision_at_threshold",
    "recall_at_fpr",
    "reliability_bins",
]
