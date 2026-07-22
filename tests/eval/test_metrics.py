"""Tests for the frozen metric protocol.

Metrics are the instrument every claim in this project is measured with, so
they get known-answer tests rather than only smoke tests. A quietly wrong
metric does not fail loudly; it publishes a wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scamvet_riskengine.eval import metrics as m

RNG = np.random.default_rng(0)


def _separable(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A perfectly separable problem: every positive outranks every negative."""
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = np.concatenate([np.linspace(0.01, 0.40, n // 2), np.linspace(0.60, 0.99, n // 2)])
    return y, scores


def _random(n: int = 2000, rate: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    y = (RNG.random(n) < rate).astype(int)
    return y, RNG.random(n)


# --------------------------------------------------------------------------
# Known answers
# --------------------------------------------------------------------------


def test_perfect_separation_scores_perfectly() -> None:
    y, s = _separable()
    result = m.evaluate(y, s)
    assert result.pr_auc == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.recall_at_fpr["0.01"] == pytest.approx(1.0)


def test_random_scores_have_pr_auc_near_base_rate() -> None:
    y, s = _random(20_000, rate=0.05)
    result = m.evaluate(y, s)
    assert result.pr_auc == pytest.approx(0.05, abs=0.02)
    assert result.roc_auc == pytest.approx(0.5, abs=0.03)


def test_roc_auc_stays_high_where_pr_auc_collapses() -> None:
    """The reason PR-AUC is primary, demonstrated rather than asserted.

    A ranker that is decent but not excellent on a 1% positive class reports a
    respectable ROC-AUC while its precision-recall performance is less than
    half that. Headlining the ROC number would misrepresent the model by a
    wide margin, which is why the protocol reports it but never leads with it.
    """
    rng = np.random.default_rng(0)
    n, rate, boost = 20_000, 0.01, 0.25
    y = (rng.random(n) < rate).astype(int)
    scores = rng.random(n) * (1 - boost) + y * boost
    result = m.evaluate(y, scores)
    assert result.roc_auc > 0.70, "ROC-AUC looks respectable"
    assert result.pr_auc < 0.40, "PR-AUC tells the real story"
    assert result.roc_auc - result.pr_auc > 0.30


def test_constant_predictions_are_uninformative_but_do_not_crash() -> None:
    y = np.array([0, 1] * 50)
    result = m.evaluate(y, np.full(100, 0.5))
    assert result.roc_auc == pytest.approx(0.5)
    assert result.ece == pytest.approx(0.0, abs=1e-9)


def test_perfectly_calibrated_predictions_have_near_zero_ece() -> None:
    probs = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 4000)
    y = (RNG.random(probs.size) < probs).astype(int)
    assert m.expected_calibration_error(y, probs) < 0.02


def test_overconfident_predictions_have_large_ece() -> None:
    probs = np.repeat([0.99], 4000)
    y = (RNG.random(4000) < 0.5).astype(int)
    assert m.expected_calibration_error(y, probs) > 0.4


def test_brier_matches_mean_squared_error() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7])
    assert m.evaluate(y, p).brier == pytest.approx(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------
# Operating points
# --------------------------------------------------------------------------


def test_recall_at_fpr_respects_the_budget() -> None:
    y, s = _random(20_000, rate=0.05)
    for target in (0.01, 0.05):
        _, threshold = m.recall_at_fpr(y, s, target)
        realised_fpr = float((s[y == 0] >= threshold).mean())
        assert realised_fpr <= target + 1e-9, f"budget {target} exceeded"


def test_recall_is_monotone_in_the_fpr_budget() -> None:
    y, s = _random(20_000, rate=0.05)
    r1, _ = m.recall_at_fpr(y, s, 0.01)
    r5, _ = m.recall_at_fpr(y, s, 0.05)
    assert r5 >= r1


def test_expected_cost_penalises_misses_more_than_false_alarms() -> None:
    y = np.array([0] * 90 + [1] * 10)
    miss_everything = np.zeros(100)
    flag_everything = np.ones(100)
    cost_missing = m.expected_cost(y, miss_everything, 0.5, fn_fp_ratio=10.0)
    cost_flagging = m.expected_cost(y, flag_everything, 0.5, fn_fp_ratio=10.0)
    assert cost_missing == pytest.approx(10 * 10 / 100)
    assert cost_flagging == pytest.approx(90 / 100)
    assert cost_missing > cost_flagging


def test_cost_ratio_of_one_is_symmetric() -> None:
    y = np.array([0] * 50 + [1] * 50)
    assert m.expected_cost(y, np.zeros(100), 0.5, 1.0) == pytest.approx(0.5)
    assert m.expected_cost(y, np.ones(100), 0.5, 1.0) == pytest.approx(0.5)


def test_reliability_bins_cover_every_sample() -> None:
    y, s = _random(5000, rate=0.2)
    bins = m.reliability_bins(y, s)
    assert len(bins) == m.ECE_BINS
    assert sum(b["count"] for b in bins) == pytest.approx(float(y.size))


# --------------------------------------------------------------------------
# Input validation — these guard against publishing a meaningless number
# --------------------------------------------------------------------------


def test_ece_rejects_uncalibrated_scores() -> None:
    y = np.array([0, 1, 0, 1])
    with pytest.raises(m.MetricError, match="probabilities"):
        m.expected_calibration_error(y, np.array([-2.0, 3.5, 0.1, 8.0]))


def test_single_class_input_raises() -> None:
    with pytest.raises(m.MetricError, match="one class"):
        m.evaluate(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_nan_scores_raise() -> None:
    with pytest.raises(m.MetricError, match="NaN"):
        m.evaluate(np.array([0, 1]), np.array([0.5, np.nan]))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(m.MetricError, match="shape mismatch"):
        m.evaluate(np.array([0, 1, 0]), np.array([0.1, 0.9]))


def test_non_binary_labels_raise() -> None:
    with pytest.raises(m.MetricError, match="binary"):
        m.evaluate(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_invalid_fpr_target_raises() -> None:
    y, s = _separable()
    with pytest.raises(m.MetricError, match="target_fpr"):
        m.recall_at_fpr(y, s, 1.5)


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=40, max_value=400), st.integers(0, 10_000))
def test_all_reported_values_are_finite_and_in_range(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(max(2, n // 10), dtype=int)])
    p = rng.random(y.size)
    result = m.evaluate(y, p)
    for value in (result.pr_auc, result.roc_auc, result.brier, result.ece):
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
    for recall in result.recall_at_fpr.values():
        assert 0.0 <= recall <= 1.0
    assert result.expected_cost >= 0.0


def test_summary_line_is_stable_text() -> None:
    y, s = _separable()
    line = m.evaluate(y, s).summary_line()
    for token in ("PR-AUC", "R@FPR1%", "R@FPR5%", "Brier", "ECE", "ROC-AUC"):
        assert token in line
