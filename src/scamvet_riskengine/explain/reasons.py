"""The ``reasons[]`` contract.

RiskEngine produces reason *tokens*; Core's Explainer agent produces the prose,
in English or Hindi, from the same versioned template table. The split matters:
if RiskEngine emitted sentences, every wording change would be a library
release, and translation would live in the wrong repository.

Three properties this module enforces, each because the alternative reaches a
user as something worse than no explanation at all.

**No raw column names escape.** A feature with no entry in the template table
cannot become a reason. Without that rule, adding a feature silently ships
``num_hyphens_host`` into somebody's verdict screen. Adding a feature therefore
requires adding its template in the same change; a test enforces it.

**One-hot columns collapse.** The encoder expands ``tld`` into 52 columns. Left
alone, a single "the domain extension is unusual" signal would arrive as
dozens of near-identical fragments, and the top reasons would be crowded out by
one feature wearing many hats. Grouped columns have their attributions summed
and surface once.

**Direction is explicit.** An attribution can argue *for* safety, and a verdict
that lists only incriminating evidence is a prosecutor, not an explanation.
``direction`` distinguishes the two so Core can say "this looked risky because
X, though Y is reassuring".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from scamvet_riskengine.explain.contributions import contributions

TEMPLATE_VERSION = "reason_templates_v1"
_DATA_DIR = Path(__file__).parent / "data"

#: Reasons weaker than this share of total attribution are dropped. A reason
#: accounting for under two percent of a decision is noise dressed as insight.
MIN_MAGNITUDE = 0.02

DEFAULT_TOP_K = 4


class ReasonError(ValueError):
    """Raised when a reason cannot be built from the template table."""


@dataclass(frozen=True)
class Reason:
    """One element of the ``reasons[]`` array Core consumes.

    Attributes:
        feature_id: source feature or feature group. Diagnostic, not shown.
        template_key: stable key into the phrase table. This is the contract.
        direction: ``increases`` or ``decreases`` risk.
        magnitude: share of the prediction's total absolute attribution, 0-1.
        evidence: human-readable observed value, or None where there is none.
        severity: template severity hint, for ordering and emphasis.
    """

    feature_id: str
    template_key: str
    direction: str
    magnitude: float
    evidence: str | None
    severity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "template_key": self.template_key,
            "direction": self.direction,
            "magnitude": round(self.magnitude, 4),
            "evidence": self.evidence,
            "severity": self.severity,
        }


@lru_cache(maxsize=1)
def load_templates() -> dict[str, Any]:
    """Load the versioned template table."""
    path = _DATA_DIR / f"{TEMPLATE_VERSION}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _render_evidence(kind: str, value: float, feature: str) -> str | None:
    """Turn a raw feature value into something a person can read."""
    if kind == "flag":
        return None
    if kind == "count":
        return f"{int(round(value))}"
    if kind == "length":
        return f"{int(round(value))} characters"
    if kind == "ratio":
        return f"{value:.0%}"
    if kind == "score":
        return f"{value:.2f}"
    if kind == "distance":
        return None if value >= 64 else f"{int(round(value))} character difference"
    if kind == "category":
        return feature.removeprefix("tld_is_")
    return None


def _resolve(feature: str, templates: dict[str, Any]) -> tuple[str, str, str] | None:
    """Map a column to (group_id, template_key, evidence_kind), or None."""
    for group_id, group in templates["groups"].items():
        if feature.startswith(group["prefix"]):
            return group_id, group["template_key"], group["evidence_kind"]
    entry = templates["features"].get(feature)
    if entry is None:
        return None
    return feature, entry["template_key"], entry["evidence_kind"]


def unmapped_features(feature_names: list[str]) -> list[str]:
    """Columns that cannot become reasons because no template covers them."""
    templates = load_templates()
    return [name for name in feature_names if _resolve(name, templates) is None]


def build_reasons(
    values: np.ndarray,
    row: np.ndarray,
    feature_names: list[str],
    top_k: int = DEFAULT_TOP_K,
    min_magnitude: float = MIN_MAGNITUDE,
) -> list[Reason]:
    """Build the reason list for one prediction.

    Args:
        values: per-feature attributions for this row.
        row: the row's feature values, for evidence rendering.
        feature_names: column names, same order as both arrays.
        top_k: maximum reasons returned.
        min_magnitude: drop reasons below this share of total attribution.
    """
    if not (len(values) == len(row) == len(feature_names)):
        raise ReasonError(
            f"length mismatch: values {len(values)}, row {len(row)}, names {len(feature_names)}"
        )
    templates = load_templates()

    grouped: dict[str, dict[str, Any]] = {}
    for value, observed, name in zip(values, row, feature_names, strict=True):
        resolved = _resolve(name, templates)
        if resolved is None:
            continue
        group_id, template_key, evidence_kind = resolved
        bucket = grouped.setdefault(
            group_id,
            {
                "total": 0.0,
                "template_key": template_key,
                "kind": evidence_kind,
                "best_abs": -1.0,
                "evidence_feature": name,
                "evidence_value": float(observed),
            },
        )
        bucket["total"] += float(value)
        # Evidence comes from whichever member contributed most, so a collapsed
        # group reports the value that actually drove it.
        if abs(float(value)) > bucket["best_abs"]:
            bucket["best_abs"] = abs(float(value))
            bucket["evidence_feature"] = name
            bucket["evidence_value"] = float(observed)

    total_abs = sum(abs(b["total"]) for b in grouped.values())
    if total_abs == 0.0:
        return []

    reasons: list[Reason] = []
    for group_id, bucket in grouped.items():
        magnitude = abs(bucket["total"]) / total_abs
        if magnitude < min_magnitude:
            continue
        template = templates["templates"].get(bucket["template_key"])
        if template is None:
            raise ReasonError(
                f"feature {group_id!r} maps to template {bucket['template_key']!r} "
                f"which is not defined in {TEMPLATE_VERSION}"
            )
        reasons.append(
            Reason(
                feature_id=group_id,
                template_key=bucket["template_key"],
                direction="increases" if bucket["total"] > 0 else "decreases",
                magnitude=magnitude,
                evidence=_render_evidence(
                    bucket["kind"], bucket["evidence_value"], bucket["evidence_feature"]
                ),
                severity=template.get("severity", "low"),
            )
        )

    reasons.sort(key=lambda r: r.magnitude, reverse=True)
    return reasons[:top_k]


def explain_rows(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[list[Reason]]:
    """Reasons for each row of ``X``."""
    values, _ = contributions(model, X)
    return [build_reasons(values[i], X[i], feature_names, top_k=top_k) for i in range(X.shape[0])]


def render_english(reason: Reason) -> str:
    """Reference English rendering, for tests and standalone library use.

    Core owns production copy in English and Hindi; this exists so the contract
    is demonstrably renderable without depending on Core.
    """
    templates = load_templates()
    text = templates["templates"][reason.template_key]["en"]
    return f"{text} ({reason.evidence})" if reason.evidence else text
