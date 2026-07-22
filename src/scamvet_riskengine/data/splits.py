"""Deterministic, group-aware dataset splitting.

Why grouped: URLs from one registered domain are near-duplicates of each
other. A random row split puts ``evil.com/login`` in train and
``evil.com/verify`` in test, the model memorises the domain, and every
reported metric is inflated by an amount nobody can measure after the fact.
Splitting by registered domain removes that path.

Why hash-based rather than shuffle-based: assignment is a pure function of
the group name and the seed. This gives three properties a shuffled split
does not have.

* Reproducible without storing index lists. The split is recomputable from
  the seed alone, which is what makes ``scamvet-riskengine eval`` able to
  regenerate a published figure.
* Stable under growth. When the corpus gains rows later, every existing
  group keeps its assignment. A shuffled split reshuffles, which quietly
  moves yesterday's test domains into today's training set and makes a
  retrained model's metrics incomparable with the ones already published.
* Stable under row order. Reading the corpus in a different order, or in
  Spark partitions, cannot change the result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

_U64 = 1 << 64

DEFAULT_RATIOS = {"train": 0.70, "calibration": 0.15, "test": 0.15}


class SplitError(ValueError):
    """Raised when a split is misconfigured or fails its own invariants."""


@dataclass(frozen=True)
class SplitReport:
    """What a split actually produced, for logging and for the model card."""

    n_rows: dict[str, int]
    n_groups: dict[str, int]
    positive_rate: dict[str, float]
    n_ungrouped_rows: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
            "positive_rate": self.positive_rate,
            "n_ungrouped_rows": self.n_ungrouped_rows,
            "seed": self.seed,
        }


def group_fraction(group: str, seed: int) -> float:
    """Map a group name to a stable float in [0, 1).

    SHA-256 rather than Python's ``hash``: the builtin is salted per process
    unless PYTHONHASHSEED is fixed, which would make splits silently
    irreproducible across runs.
    """
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _U64


def assign_split(
    group: str,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> str:
    """Assign one group to a split name."""
    ratios = ratios or DEFAULT_RATIOS
    position = group_fraction(group, seed)
    cumulative = 0.0
    last = ""
    for name, share in ratios.items():
        cumulative += share
        last = name
        if position < cumulative:
            return name
    return last


def grouped_split(
    frame: pd.DataFrame,
    group_col: str = "group",
    label_col: str = "label",
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, SplitReport]:
    """Add a ``split`` column assigning every row to train/calibration/test.

    Rows whose group is empty (IP hosts, unparseable input) cannot be grouped
    safely. They are assigned by hashing the row's own URL so they are still
    usable, and counted separately in the report so the caller can see how
    many there were rather than discovering it later.

    Args:
        frame: must contain ``group_col`` and ``label_col``.
        group_col: column holding the grouping key, normally registered domain.
        label_col: binary 0/1 column, used only for the report.
        seed: split seed. Changing it produces a different, equally valid
            split; the value used is recorded in the model card.
        ratios: split name to share. Must sum to 1.0.

    Returns:
        (frame with a ``split`` column, SplitReport)
    """
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"ratios must sum to 1.0, got {total}")
    if any(share <= 0 for share in ratios.values()):
        raise SplitError("every ratio must be positive")
    for column in (group_col, label_col):
        if column not in frame.columns:
            raise SplitError(f"frame is missing required column {column!r}")

    out = frame.copy()
    groups = out[group_col].fillna("").astype(str)
    ungrouped = groups == ""
    # Ungrouped rows fall back to their own URL so that each is independent
    # rather than all landing in one split together.
    keys = groups.where(
        ~ungrouped, "url:" + out.get("url", pd.Series("", index=out.index)).astype(str)
    )

    unique_keys = keys.unique()
    mapping = {key: assign_split(key, seed, ratios) for key in unique_keys}
    out["split"] = keys.map(mapping)

    report = SplitReport(
        n_rows={name: int((out["split"] == name).sum()) for name in ratios},
        n_groups={name: int(keys[out["split"] == name].nunique()) for name in ratios},
        positive_rate={
            name: float(out.loc[out["split"] == name, label_col].mean())
            if int((out["split"] == name).sum())
            else 0.0
            for name in ratios
        },
        n_ungrouped_rows=int(ungrouped.sum()),
        seed=seed,
    )
    return out, report


def assert_no_group_leakage(
    frame: pd.DataFrame,
    group_col: str = "group",
    split_col: str = "split",
) -> None:
    """Raise if any non-empty group appears in more than one split.

    This is the invariant the whole module exists to hold. It is asserted
    rather than assumed, because a leak here is invisible in every downstream
    metric except as unexplained optimism.
    """
    grouped = frame[frame[group_col].fillna("").astype(str) != ""]
    counts = grouped.groupby(group_col, observed=True)[split_col].nunique()
    offenders = counts[counts > 1]
    if len(offenders):
        sample = list(offenders.index[:5])
        raise SplitError(
            f"{len(offenders)} group(s) appear in more than one split; examples: {sample}"
        )
