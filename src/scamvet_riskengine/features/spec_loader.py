"""Loader and validator for versioned feature specs.

The spec JSON is the contract between RiskEngine, Core's risk-scorer agent and
ScamPulse's dbt models. This module is the only supported way to read it, so
that all three consumers agree on field names, dtypes and ranges without
copying the list into three codebases.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_SPEC_DIR = Path(__file__).parent / "spec"

_INT_DTYPES = {"int8", "int16", "int32", "int64"}
_FLOAT_DTYPES = {"float32", "float64"}


class SpecViolation(ValueError):
    """Raised when a feature dict does not satisfy its declared spec."""


@lru_cache(maxsize=8)
def load_spec(spec_version: str) -> dict[str, Any]:
    """Load a feature spec by version, e.g. ``url_lexical_v1``."""
    path = _SPEC_DIR / f"{spec_version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No feature spec named {spec_version!r} in {_SPEC_DIR}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def spec_feature_names(spec_version: str) -> list[str]:
    """Ordered field names declared by a spec."""
    return [f["name"] for f in load_spec(spec_version)["features"]]


def validate(features: dict[str, Any], spec_version: str) -> None:
    """Check a feature dict against its spec. Raises :class:`SpecViolation`.

    Checks names and order, dtype family, nullability, and declared ranges.
    Used by the contract test and available to downstream consumers as a
    runtime guard at pipeline boundaries.
    """
    spec = load_spec(spec_version)
    declared = [f["name"] for f in spec["features"]]
    produced = list(features.keys())

    if produced != declared:
        missing = [n for n in declared if n not in produced]
        extra = [n for n in produced if n not in declared]
        raise SpecViolation(
            f"field mismatch for {spec_version}: "
            f"missing={missing} extra={extra} order_matches={produced == declared}"
        )

    for field in spec["features"]:
        name = field["name"]
        value = features[name]
        dtype = field["dtype"]

        if value is None:
            if not field["nullable"]:
                raise SpecViolation(f"{name} is null but spec declares non-nullable")
            continue

        if dtype == "string":
            if not isinstance(value, str):
                raise SpecViolation(f"{name} expected string, got {type(value).__name__}")
            continue

        if dtype in _INT_DTYPES:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SpecViolation(f"{name} expected int, got {type(value).__name__}")
        elif dtype in _FLOAT_DTYPES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecViolation(f"{name} expected float, got {type(value).__name__}")
            if value != value:  # NaN
                raise SpecViolation(f"{name} is NaN but spec declares non-nullable")
        else:
            raise SpecViolation(f"{name} declares unknown dtype {dtype!r}")

        low = field.get("min")
        high = field.get("max")
        if low is not None and value < low:
            raise SpecViolation(f"{name}={value} below declared min {low}")
        if high is not None and value > high:
            raise SpecViolation(f"{name}={value} above declared max {high}")
