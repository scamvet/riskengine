"""The scoring contract Core consumes.

``ScoreResult`` is frozen. Core's risk-scorer agent imports it as a tool
result, the PWA renders it, the Telegram bot formats it, and ScamPulse warehouses
it. Any field change is a breaking semver bump and triggers the cross-surface
audit, because a field that exists on one surface and not another is exactly the
drift the parity rule exists to catch.

Inference runs on ONNX rather than the training library. That keeps the runtime
dependency small - onnxruntime is an 18 MB wheel against xgboost's 132 MB - and
it means the Python path scores through the same artifact the browser and the
Android app do. A verdict that differs by client is worse than a slow one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scamvet_riskengine.explain.reasons import Reason, build_reasons
from scamvet_riskengine.features.featurize import LexicalFeaturizer, extract_frame
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION
from scamvet_riskengine.registry.manifest import Manifest, ManifestError

DEFAULT_USE_CASE = "vet"

#: Below this distance from a band boundary, a verdict is reported as uncertain.
#: Bands are a discretisation of a continuous score, so a probability sitting
#: one thousandth from a cut is not meaningfully different from one on the other
#: side, and presenting it as a clean verdict would overstate what was measured.
BOUNDARY_MARGIN = 0.05


class ScorerError(RuntimeError):
    """Raised when a scorer cannot be built or run."""


@dataclass(frozen=True)
class ScoreResult:
    """One verdict. Frozen contract - see module docstring before changing.

    Attributes:
        url: the input, as given.
        probability: calibrated probability that the URL is malicious.
        band: ``green``, ``amber`` or ``red``.
        reasons: ordered attributions, strongest first.
        uncertainty: 0 when the score sits far from any band boundary, rising
            to 1 at a boundary. Not a confidence interval.
        flags: machine-readable caveats, e.g. ``unparseable_input``.
        use_case: which threshold set produced the band.
        model_name, model_version, feature_spec_version: provenance.
        requires_network: whether this scorer needed enrichment. False for the
            lexical scorer, which is what makes the offline path possible.
    """

    url: str
    probability: float
    band: str
    reasons: tuple[Reason, ...]
    uncertainty: float
    flags: tuple[str, ...]
    use_case: str
    model_name: str
    model_version: str
    feature_spec_version: str
    requires_network: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "probability": round(self.probability, 6),
            "band": self.band,
            "reasons": [r.as_dict() for r in self.reasons],
            "uncertainty": round(self.uncertainty, 4),
            "flags": list(self.flags),
            "use_case": self.use_case,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_spec_version": self.feature_spec_version,
            "requires_network": self.requires_network,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Scorer:
    """A loaded scorer: featurizer, ONNX model, calibrator and thresholds.

    Constructed by :func:`scamvet_riskengine.registry.load`, not directly.
    """

    def __init__(self, manifest: Manifest, directory: Path) -> None:
        if manifest.feature_spec_version != FEATURE_SPEC_VERSION:
            raise ScorerError(
                f"{manifest.name} {manifest.version} was built against feature spec "
                f"{manifest.feature_spec_version!r}, but this build produces "
                f"{FEATURE_SPEC_VERSION!r}. Loading it would feed the model features "
                f"that do not mean what it learned."
            )
        self.manifest = manifest
        self.directory = Path(directory)
        self.featurizer = LexicalFeaturizer.from_dict(manifest.featurizer)
        self._session = self._open_session()
        self._booster: Any = None

    def _open_session(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional at build time
            raise ScorerError(
                "scoring requires onnxruntime; install with pip install 'scamvet-riskengine[onnx]'"
            ) from exc
        artifact = self.directory / self.manifest.artifact_name
        if not artifact.is_file():
            raise ScorerError(f"model artifact missing: {artifact}")
        return ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])

    # -- calibration -----------------------------------------------------

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        method = self.manifest.calibration.get("method")
        if method == "platt":
            coef = float(self.manifest.calibration["coef"])
            intercept = float(self.manifest.calibration["intercept"])
            return _sigmoid(coef * raw + intercept)
        if method == "identity":
            return raw
        raise ScorerError(
            f"manifest declares calibration method {method!r}, which this build cannot "
            "apply. Platt state is two floats and travels in the manifest; isotonic "
            "would need its threshold arrays exported alongside."
        )

    # -- scoring ---------------------------------------------------------

    def score_many(self, urls: list[str], use_case: str = DEFAULT_USE_CASE) -> list[ScoreResult]:
        """Score several URLs. One ONNX call, so prefer this for batches."""
        thresholds = self.manifest.threshold_for(use_case)
        raw_features = extract_frame(urls)
        matrix = self.featurizer.transform(raw_features)
        X = matrix.to_numpy(dtype=np.float32)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: np.ascontiguousarray(X)})
        probabilities = np.asarray(outputs[1])
        scores = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        calibrated = np.clip(self._calibrate(np.asarray(scores, dtype=float)), 0.0, 1.0)

        names = list(matrix.columns)
        attributions = self._attributions(X)
        parse_ok = (
            raw_features["parse_ok"].to_numpy()
            if "parse_ok" in raw_features
            else np.ones(len(urls))
        )

        results = []
        for index, url in enumerate(urls):
            probability = float(calibrated[index])
            band = thresholds.band(probability)
            distance = min(abs(probability - thresholds.red), abs(probability - thresholds.amber))
            flags: list[str] = []
            if not parse_ok[index]:
                flags.append("unparseable_input")
            if self.manifest.requires_network:
                flags.append("enrichment_used")
            results.append(
                ScoreResult(
                    url=url,
                    probability=probability,
                    band=band,
                    reasons=tuple(
                        build_reasons(attributions[index], X[index], names)
                        if attributions is not None
                        else ()
                    ),
                    uncertainty=float(max(0.0, 1.0 - distance / BOUNDARY_MARGIN)),
                    flags=tuple(flags),
                    use_case=use_case,
                    model_name=self.manifest.name,
                    model_version=self.manifest.version,
                    feature_spec_version=self.manifest.feature_spec_version,
                    requires_network=self.manifest.requires_network,
                )
            )
        return results

    def score(self, url: str, use_case: str = DEFAULT_USE_CASE) -> ScoreResult:
        """Score one URL."""
        return self.score_many([url], use_case=use_case)[0]

    def _attributions(self, X: np.ndarray) -> np.ndarray | None:
        """Attributions from the training-format artifact, when it is present.

        ONNX carries no TreeSHAP path, so reasons need the native model. It is
        optional on purpose: a deployment that only needs bands should not be
        forced to ship a second artifact, and a verdict without reasons is
        degraded rather than broken.
        """
        if self._booster is None:
            self._booster = self._load_booster()
        if self._booster is None:
            return None
        try:
            import xgboost as xgb

            raw = np.asarray(self._booster.predict(xgb.DMatrix(X), pred_contribs=True), dtype=float)
            return raw[:, :-1]
        except Exception:  # pragma: no cover - reasons are best-effort
            return None

    def _load_booster(self) -> Any:
        """Load the native model once, from gzip or plain JSON.

        Stored gzipped: XGBoost's JSON dump is several megabytes uncompressed
        and compresses by roughly an order of magnitude. It is committed once
        per model version and lives in git forever, so the uncompressed form is
        waste rather than a constraint worth respecting.
        """
        import gzip

        for name, opener in (
            (self.manifest.native_artifact + ".gz", lambda p: gzip.decompress(p.read_bytes())),
            (self.manifest.native_artifact, lambda p: p.read_bytes()),
        ):
            path = self.directory / name
            if not path.is_file():
                continue
            try:
                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(bytearray(opener(path)))
                return booster
            except Exception:  # pragma: no cover - reasons are best-effort
                return None
        return None

    def __repr__(self) -> str:
        return (
            f"Scorer({self.manifest.name!r}, {self.manifest.version!r}, "
            f"track={self.manifest.track!r}, spec={self.manifest.feature_spec_version!r})"
        )


def build_scorer(directory: Path) -> Scorer:
    """Load a scorer from a registry directory containing ``manifest.json``."""
    directory = Path(directory)
    manifest = Manifest.load(directory / "manifest.json")
    if manifest.track != "product":
        raise ManifestError(
            f"{manifest.name} {manifest.version} is a research-track model and must not be "
            "served to users. Research models train on non-commercially licensed corpora "
            "(ADR-007) and are published as analysis, not product."
        )
    return Scorer(manifest, directory)
