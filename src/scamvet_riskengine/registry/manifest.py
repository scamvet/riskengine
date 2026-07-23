"""Model manifests and version resolution.

A manifest is the complete description of a shipped scorer: which corpus and
licence row it came from, which feature spec it was built against, what it
scored, how it is calibrated, and where its verdict bands cut. It travels with
the artifact, because a model file without one cannot be reproduced, audited or
safely consumed.

Two fields exist to make a whole class of mistake impossible rather than merely
discouraged. ``track`` records whether a model came from the research side
(non-commercially licensed corpora, per ADR-007) or the product side; the
registry refuses to hand a research model to a product caller. And
``feature_spec_version`` is checked at load, because four spec versions have
already come and gone in this chapter and a v2-trained model fed v4 features
would produce confident nonsense with nothing in any metric to reveal it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Track = Literal["product", "research"]

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SPEC = re.compile(r"^(>=|<=|==|~=|\^)?\s*(\d+)\.(\d+)\.(\d+)$")

REQUIRED_FIELDS = (
    "name",
    "version",
    "track",
    "feature_spec_version",
    "model",
    "calibration",
    "thresholds",
    "metrics",
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed, or is not the one asked for."""


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse a strict ``major.minor.patch`` string."""
    match = _SEMVER.match(text.strip())
    if not match:
        raise ManifestError(f"not a semantic version: {text!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def satisfies(version: str, spec: str | None) -> bool:
    """Whether ``version`` satisfies a version specifier.

    Supports ``==``, ``>=``, ``<=``, ``~=`` (compatible within the minor) and
    ``^`` (compatible within the major). A bare version means ``==``, and
    ``None`` matches anything. Deliberately small: a registry that needs a
    dependency resolver to answer "which model do I load" has a design problem.
    """
    if spec is None:
        return True
    match = _SPEC.match(spec.strip())
    if not match:
        raise ManifestError(f"not a version specifier: {spec!r}")
    operator = match.group(1) or "=="
    target = tuple(int(part) for part in match.groups()[1:])
    current = parse_version(version)

    if operator == "==":
        return current == target
    if operator == ">=":
        return current >= target
    if operator == "<=":
        return current <= target
    if operator == "~=":
        return current >= target and current[:2] == target[:2]
    if operator == "^":
        return current >= target and current[0] == target[0]
    raise ManifestError(f"unsupported operator {operator!r}")


@dataclass(frozen=True)
class Thresholds:
    """Where the verdict bands cut, for one use case.

    Thresholds are a property of the use case, not of the model: the consumer
    verdict is precision-leaning because a false red destroys trust in a way a
    false amber does not, while a pre-payment QR check can afford to warn more
    freely. Both read the same calibrated probability.
    """

    red: float
    amber: float

    def band(self, probability: float) -> str:
        if probability >= self.red:
            return "red"
        if probability >= self.amber:
            return "amber"
        return "green"

    def as_dict(self) -> dict[str, float]:
        return {"red": self.red, "amber": self.amber}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Thresholds:
        try:
            red, amber = float(payload["red"]), float(payload["amber"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"malformed thresholds: {payload!r}") from exc
        if not 0.0 <= amber <= red <= 1.0:
            raise ManifestError(
                f"thresholds must satisfy 0 <= amber <= red <= 1, got amber={amber} red={red}"
            )
        return cls(red=red, amber=amber)


@dataclass(frozen=True)
class Manifest:
    """The complete description of a shipped scorer."""

    name: str
    version: str
    track: Track
    feature_spec_version: str
    model: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Thresholds]
    metrics: dict[str, Any]
    featurizer: dict[str, Any] = field(default_factory=dict)
    corpus: str = ""
    corpus_licence_row: str = ""
    trained_at: str = ""
    git_sha: str = ""
    requires_network: bool = False
    model_card: str = ""
    notes: str = ""

    @property
    def artifact_name(self) -> str:
        return str(self.model.get("artifact", "model.onnx"))

    @property
    def native_artifact(self) -> str:
        """The training-format model, needed for TreeSHAP reasons.

        Optional: a deployment that only needs bands should not be forced to
        ship a second artifact, and a verdict without reasons is degraded
        rather than broken (ADR-008).
        """
        return str(self.model.get("native_artifact", "model.json"))

    def threshold_for(self, use_case: str) -> Thresholds:
        try:
            return self.thresholds[use_case]
        except KeyError:
            raise ManifestError(
                f"{self.name} {self.version} declares no thresholds for use case "
                f"{use_case!r}; available: {sorted(self.thresholds)}"
            ) from None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "track": self.track,
            "corpus": self.corpus,
            "corpus_licence_row": self.corpus_licence_row,
            "trained_at": self.trained_at,
            "git_sha": self.git_sha,
            "feature_spec_version": self.feature_spec_version,
            "featurizer": self.featurizer,
            "model": self.model,
            "calibration": self.calibration,
            "thresholds": {k: v.as_dict() for k, v in self.thresholds.items()},
            "metrics": self.metrics,
            "requires_network": self.requires_network,
            "model_card": self.model_card,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        missing = [key for key in REQUIRED_FIELDS if key not in payload]
        if missing:
            raise ManifestError(f"manifest is missing required field(s): {missing}")
        if payload["track"] not in ("product", "research"):
            raise ManifestError(f"track must be 'product' or 'research', got {payload['track']!r}")
        parse_version(payload["version"])
        thresholds = {
            use_case: Thresholds.from_dict(value)
            for use_case, value in payload["thresholds"].items()
        }
        if not thresholds:
            raise ManifestError("a manifest must declare at least one use case's thresholds")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in payload.items() if k in known and k != "thresholds"}
        return cls(thresholds=thresholds, **kwargs)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"no manifest at {path}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest at {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
