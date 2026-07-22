"""Aggregation across seeds.

A single-seed metric is a point estimate presented as a fact. On this corpus,
phishing recall at FPR 5% varies by roughly 0.013 across seeds for a fixed
configuration - which is larger than the gap between several configurations we
were about to choose between. Reporting one number invites exactly the mistake
of reading a favourable draw as a real improvement.

So: every published figure is a mean with its observed range, and any claim
that one configuration beats another must clear the seed spread before it
counts as a finding. See ADR-009, amendment of 2026-07-23.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence

from scamvet_riskengine.eval.metrics import EvalResult

#: Metrics summarised across seeds. Nested FPR-keyed values are flattened to
#: ``recall_at_fpr_0.05`` style names so the summary is a flat table.
SUMMARISED = ("pr_auc", "roc_auc", "brier", "ece")


class AggregateError(ValueError):
    """Raised when a summary cannot be computed from the inputs given."""


@dataclass(frozen=True)
class MetricSpread:
    """One metric observed across several seeds."""

    name: str
    values: tuple[float, ...]
    mean: float
    minimum: float
    maximum: float

    @property
    def spread(self) -> float:
        """Observed range. Used as the bar a real difference must clear."""
        return self.maximum - self.minimum

    def format(self, places: int = 4) -> str:
        """Human form, e.g. ``0.8643 +/- 0.0027``."""
        return f"{self.mean:.{places}f} +/- {self.spread:.{places}f}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "spread": self.spread,
            "values": list(self.values),
        }


def _spread(name: str, values: Sequence[float]) -> MetricSpread:
    numbers = tuple(float(v) for v in values)
    return MetricSpread(
        name=name,
        values=numbers,
        mean=fmean(numbers),
        minimum=min(numbers),
        maximum=max(numbers),
    )


def summarise(results: Iterable[EvalResult]) -> dict[str, MetricSpread]:
    """Summarise a set of per-seed results into per-metric spreads.

    Args:
        results: one :class:`EvalResult` per seed, all from the same
            configuration on the same split.

    Raises:
        AggregateError: if fewer than two results are given, since a spread
            over one observation is not a spread and reporting it as one would
            be worse than reporting the bare number.
    """
    collected = list(results)
    if len(collected) < 2:
        raise AggregateError(f"need at least 2 seeds to report a spread, got {len(collected)}")

    summary: dict[str, MetricSpread] = {}
    for metric in SUMMARISED:
        summary[metric] = _spread(metric, [getattr(r, metric) for r in collected])

    fpr_keys = set(collected[0].recall_at_fpr)
    for other in collected[1:]:
        if set(other.recall_at_fpr) != fpr_keys:
            raise AggregateError("results report different FPR targets")
    for key in sorted(fpr_keys):
        summary[f"recall_at_fpr_{key}"] = _spread(
            f"recall_at_fpr_{key}", [r.recall_at_fpr[key] for r in collected]
        )
    return summary


def separates(a: MetricSpread, b: MetricSpread) -> bool:
    """Whether two configurations differ by more than their own seed noise.

    The rule from ADR-009's amendment: a difference in means only counts as a
    finding if it exceeds the larger of the two observed spreads. Anything
    smaller is a favourable draw, and calling it an improvement is how a
    fluctuation becomes a published claim.
    """
    return abs(a.mean - b.mean) > max(a.spread, b.spread)


def summary_table(summary: dict[str, MetricSpread], places: int = 4) -> str:
    """Markdown table of one configuration's summarised metrics."""
    lines = ["| metric | mean | min | max | spread |", "|---|---|---|---|---|"]
    for name, spread in summary.items():
        lines.append(
            f"| {name} | {spread.mean:.{places}f} | {spread.minimum:.{places}f} | "
            f"{spread.maximum:.{places}f} | {spread.spread:.{places}f} |"
        )
    return "\n".join(lines)


def as_dict(summary: dict[str, MetricSpread]) -> dict[str, Any]:
    """JSON-serialisable form for the model card."""
    return {name: spread.as_dict() for name, spread in summary.items()}
