"""Tests for multi-seed aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.eval import metrics as m
from scamvet_riskengine.eval.aggregate import (
    AggregateError,
    MetricSpread,
    as_dict,
    separates,
    summarise,
    summary_table,
)


def _result(seed: int) -> m.EvalResult:
    rng = np.random.default_rng(seed)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    p = np.clip(rng.random(n) * 0.6 + y * 0.3, 0.0, 1.0)
    return m.evaluate(y, p)


def _spread(name: str, values: tuple[float, ...]) -> MetricSpread:
    return MetricSpread(
        name=name,
        values=values,
        mean=sum(values) / len(values),
        minimum=min(values),
        maximum=max(values),
    )


def test_summarise_covers_every_reported_metric() -> None:
    summary = summarise([_result(s) for s in (1, 2, 3)])
    for expected in ("pr_auc", "roc_auc", "brier", "ece", "recall_at_fpr_0.05"):
        assert expected in summary


def test_mean_and_range_are_consistent() -> None:
    summary = summarise([_result(s) for s in (1, 2, 3, 4)])
    for spread in summary.values():
        assert spread.minimum <= spread.mean <= spread.maximum
        assert spread.spread == pytest.approx(spread.maximum - spread.minimum)
        assert len(spread.values) == 4


def test_single_seed_is_refused() -> None:
    with pytest.raises(AggregateError, match="at least 2 seeds"):
        summarise([_result(1)])


def test_mismatched_fpr_targets_are_refused() -> None:
    a = m.evaluate(*_inputs(1), fpr_targets=(0.01, 0.05))
    b = m.evaluate(*_inputs(2), fpr_targets=(0.02,))
    with pytest.raises(AggregateError, match="different FPR targets"):
        summarise([a, b])


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    y = (rng.random(3000) < 0.2).astype(int)
    return y, np.clip(rng.random(3000) * 0.6 + y * 0.3, 0.0, 1.0)


# --------------------------------------------------------------------------
# The separation rule
# --------------------------------------------------------------------------


def test_the_real_case_that_motivated_this_does_not_separate() -> None:
    """Measured phishing recall at FPR 5%, four seeds, on the real corpus.

    400 trees looked worse than 100 on seed 42 alone. Across four seeds the
    gap is smaller than the seed spread, so it is not a finding. This test
    pins the numbers that disproved the claim.
    """
    trees_400 = _spread("recall_at_fpr_0.05", (0.8511, 0.8648, 0.8597, 0.8606))
    trees_100 = _spread("recall_at_fpr_0.05", (0.8658, 0.8689, 0.8641, 0.8564))
    assert not separates(trees_400, trees_100)
    assert trees_400.spread > abs(trees_400.mean - trees_100.mean)


def test_a_genuine_difference_does_separate() -> None:
    """The logistic-vs-boosted gap, which is real by any measure."""
    logistic = _spread("recall_at_fpr_0.05", (0.4099, 0.4120, 0.4080))
    boosted = _spread("recall_at_fpr_0.05", (0.8511, 0.8489, 0.8534))
    assert separates(logistic, boosted)


def test_identical_configurations_never_separate() -> None:
    values = (0.80, 0.81, 0.82)
    assert not separates(_spread("x", values), _spread("x", values))


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def test_format_shows_mean_and_spread() -> None:
    text = _spread("x", (0.8511, 0.8648, 0.8597)).format()
    assert "+/-" in text
    assert text.startswith("0.85")


def test_summary_table_is_markdown() -> None:
    table = summary_table(summarise([_result(s) for s in (1, 2, 3)]))
    assert table.splitlines()[0].startswith("| metric |")
    assert "|---|" in table


def test_as_dict_is_json_shaped() -> None:
    payload = as_dict(summarise([_result(s) for s in (1, 2, 3)]))
    entry = payload["pr_auc"]
    assert set(entry) == {"mean", "min", "max", "spread", "values"}
    assert len(entry["values"]) == 3
