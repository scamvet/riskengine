"""ONNX export for the offline scorer.

ADR-004 requires a verdict even when every network-dependent tier is
unavailable, and Chapter G runs a scorer on-device. Both consume an ONNX
artifact, so export is not a nicety here - it is the mechanism by which the
always-live promise is kept.

Two hard constraints shape this module.

**Size.** The repository caps committed files at 500 KB, and an on-device model
downloaded at first use would break the offline guarantee on exactly the
connection where it matters. Measured on this feature set, a 400-tree
depth-8 LightGBM exports to roughly 3.8 MB - almost eight times the cap. The
shipped offline model is therefore a deliberately smaller sibling of the
production model, and the recall it gives up is measured and published rather
than discovered later.

**Parity.** A silent divergence between the trained model and its export is
precisely the class of unmeasured failure the second promise exists to prevent:
it would produce different verdicts online and offline with nothing in any
metric to reveal it. Export is therefore always verified against the native
model on a fixture, and the maximum deviation is recorded in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Committed-artifact ceiling, matching the pre-commit large-file hook.
#:
#: Raised from 500 KB to 1 MB, then to 2 MB, once it became clear the limit was shaping the
#: product rather than the other way round. The hook exists to stop accidental
#: data commits, not to constrain a runtime dependency the deployed app loads;
#: GitHub's own per-file hard limit is 100 MB, and a PWA shipping a sub-megabyte
#: model is unremarkable. At 500 KB the offline scorer cost 0.122 phishing
#: recall against the online model - a 14.4% relative loss borne by users on
#: poor connections, who in this market are disproportionately the people the
#: product exists to protect. One model at 936 KB serving both paths is both
#: simpler and fairer than two models split by a self-imposed number.
#:
#: 2 MB admits the 200 x 8 configuration (1,819 KB), which is the last size
#: where more trees buy measurable recall: 100 to 200 separates at five seeds,
#: 200 to 400 does not. The artifact is loaded by onnxruntime-web and bundled
#: on-device, where a tree ensemble expands roughly three to five times in
#: memory - so this is not unbounded. It is affordable because the offline
#: scorer is only reachable after a successful online load, and can therefore
#: be lazy-fetched and cached without ever blocking the three-second test.
SIZE_CAP_BYTES = 2 * 1024 * 1024

#: Maximum tolerated difference between native and ONNX probabilities.
#: Measured deviation is around 2e-07, so this is a wide margin that still
#: catches any structural conversion error.
PARITY_TOLERANCE = 1e-5


class ExportError(RuntimeError):
    """Raised when conversion fails or the export does not match the model."""


@dataclass(frozen=True)
class ExportResult:
    """What an export produced, for the model manifest."""

    n_features: int
    size_bytes: int
    fits_size_cap: bool
    max_abs_parity_diff: float
    parity_ok: bool
    n_parity_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "size_bytes": self.size_bytes,
            "size_kb": round(self.size_bytes / 1024, 1),
            "fits_size_cap": self.fits_size_cap,
            "max_abs_parity_diff": self.max_abs_parity_diff,
            "parity_ok": self.parity_ok,
            "n_parity_samples": self.n_parity_samples,
        }


def to_onnx(model: Any, n_features: int) -> bytes:
    """Convert a fitted LightGBM or XGBoost classifier to serialised ONNX.

    ``zipmap=False`` keeps the probability output as a plain tensor. With
    ZipMap enabled the output is a sequence of dictionaries, which is awkward
    in every non-Python runtime and is the last thing a browser or mobile
    client needs to unpack on the offline path.
    """
    try:
        from onnxmltools import convert_lightgbm, convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ExportError(
            "ONNX export requires onnxmltools; install with pip install 'scamvet-riskengine[train]'"
        ) from exc

    initial_types = [("input", FloatTensorType([None, n_features]))]
    # zipmap is a LightGBM-converter option; convert_xgboost rejects the keyword
    # outright. XGBoost's converter already emits a plain probability tensor.
    if hasattr(model, "get_booster"):
        converter, family, kwargs = convert_xgboost, "XGBoost", {}
    elif hasattr(model, "booster_"):
        converter, family, kwargs = convert_lightgbm, "LightGBM", {"zipmap": False}
    else:
        raise ExportError(
            f"no ONNX converter for {type(model).__name__}; expected a fitted "
            "LightGBM or XGBoost classifier"
        )

    try:
        onnx_model = converter(model, initial_types=initial_types, **kwargs)
    except Exception as exc:
        raise ExportError(f"{family} to ONNX conversion failed: {exc}") from exc
    return onnx_model.SerializeToString()


def onnx_probabilities(blob: bytes, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from a serialised ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ExportError(
            "ONNX verification requires onnxruntime; install with "
            "pip install 'scamvet-riskengine'"
        ) from exc

    session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.ascontiguousarray(X, dtype=np.float32)})
    probabilities = np.asarray(outputs[1])
    if probabilities.ndim == 1:  # some converters emit a flat positive-class vector
        return probabilities
    # Output 0 is the predicted label, output 1 the probability tensor. The
    # runtime emits a benign shape warning about the label output; it does not
    # affect the probabilities, which are the only thing read here.
    return probabilities[:, 1]


def export_and_verify(
    model: Any,
    X_sample: np.ndarray,
    path: Path | None = None,
    tolerance: float = PARITY_TOLERANCE,
) -> tuple[bytes, ExportResult]:
    """Convert, verify against the native model, and optionally write to disk.

    Args:
        model: fitted LightGBM or XGBoost classifier.
        X_sample: rows to compare on. Use real feature rows, not random noise -
            a conversion bug in a rarely-taken tree branch will not show up on
            data that never reaches it.
        path: if given, the verified bytes are written here.
        tolerance: maximum permitted absolute probability difference.

    Raises:
        ExportError: if the export does not match the native model.
    """
    X_sample = np.ascontiguousarray(X_sample, dtype=np.float32)
    n_features = int(X_sample.shape[1])

    blob = to_onnx(model, n_features)
    native = np.asarray(model.predict_proba(X_sample))[:, 1]
    exported = onnx_probabilities(blob, X_sample)
    max_diff = float(np.max(np.abs(native - exported)))
    parity_ok = bool(max_diff <= tolerance)

    result = ExportResult(
        n_features=n_features,
        size_bytes=len(blob),
        fits_size_cap=len(blob) <= SIZE_CAP_BYTES,
        max_abs_parity_diff=max_diff,
        parity_ok=parity_ok,
        n_parity_samples=int(X_sample.shape[0]),
    )

    if not parity_ok:
        raise ExportError(
            f"ONNX export diverges from the native model: max |diff| {max_diff:.3e} "
            f"exceeds tolerance {tolerance:.1e}. Shipping this would produce "
            f"different verdicts online and offline with nothing to reveal it."
        )

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)

    return blob, result
