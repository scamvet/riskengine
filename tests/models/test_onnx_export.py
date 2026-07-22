"""Tests for ONNX export.

Parity is the point. A conversion that silently disagrees with the trained
model would produce different verdicts online and offline, and nothing in any
metric would show it.
"""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.models.boosted import BoostedConfig, build_lightgbm

onnx = pytest.importorskip("onnxmltools", reason="ONNX extra not installed")
pytest.importorskip("onnxruntime", reason="ONNX extra not installed")

from scamvet_riskengine.models.onnx_export import (  # noqa: E402
    PARITY_TOLERANCE,
    SIZE_CAP_BYTES,
    ExportError,
    export_and_verify,
    onnx_probabilities,
    to_onnx,
)


@pytest.fixture(scope="module")
def fitted() -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d = 4000, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.7 * X[:, 1] * X[:, 2] + rng.normal(0, 0.4, n) > 0.4).astype(int)
    model = build_lightgbm(BoostedConfig(n_estimators=30, max_depth=5), y)
    model.fit(X, y)
    return model, X


def test_export_matches_native_probabilities(fitted) -> None:
    model, X = fitted
    _, result = export_and_verify(model, X[:800])
    assert result.parity_ok
    assert result.max_abs_parity_diff < PARITY_TOLERANCE
    assert result.n_parity_samples == 800


def test_export_deviation_is_tiny_in_practice(fitted) -> None:
    """Documents the observed headroom, so a regression is obvious."""
    model, X = fitted
    _, result = export_and_verify(model, X[:800])
    assert result.max_abs_parity_diff < 1e-5


def test_probabilities_are_valid(fitted) -> None:
    model, X = fitted
    blob = to_onnx(model, X.shape[1])
    probs = onnx_probabilities(blob, X[:500])
    assert probs.shape == (500,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert np.isfinite(probs).all()


def test_small_model_fits_the_commit_cap(fitted) -> None:
    model, X = fitted
    _, result = export_and_verify(model, X[:200])
    assert result.size_bytes < SIZE_CAP_BYTES
    assert result.fits_size_cap


def test_large_model_is_reported_as_not_fitting() -> None:
    """The whole reason the offline scorer is a separate, smaller model."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6000, 60)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.5, 6000) > 0).astype(int)
    model = build_lightgbm(BoostedConfig(n_estimators=400, max_depth=8), y)
    model.fit(X, y)
    _, result = export_and_verify(model, X[:200])
    assert not result.fits_size_cap, "a production-scale model should exceed the cap"


def test_export_writes_to_disk(fitted, tmp_path) -> None:
    model, X = fitted
    target = tmp_path / "nested" / "model.onnx"
    blob, _ = export_and_verify(model, X[:200], path=target)
    assert target.is_file()
    assert target.read_bytes() == blob


def test_tolerance_violation_raises(fitted) -> None:
    model, X = fitted
    with pytest.raises(ExportError, match="diverges"):
        export_and_verify(model, X[:200], tolerance=0.0)


def test_result_serialises(fitted) -> None:
    model, X = fitted
    _, result = export_and_verify(model, X[:200])
    payload = result.as_dict()
    assert payload["parity_ok"] is True
    assert payload["size_kb"] > 0
