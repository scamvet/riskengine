"""Tests for ONNX export.

Parity is the point. A conversion that silently disagrees with the trained
model would produce different verdicts online and offline, and nothing in any
metric would show it.
"""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.models.boosted import BoostedConfig, build_lightgbm, build_xgboost

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
    """The cap is enforced, and enforcement is reported rather than raised.

    ScamVet ships one model on both paths, so the sweep picks the best config
    that fits rather than falling back to a weaker sibling. That only works if
    over-cap candidates are flagged instead of failing the export.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8000, 60)).astype(np.float32)
    # Noisy labels and light pruning, so the trees are genuinely bushy. With the
    # default min_child_weight the same nominal config prunes down under the cap.
    y = (X[:, 0] + rng.normal(0, 0.9, 8000) > 0).astype(int)
    model = build_xgboost(BoostedConfig(n_estimators=400, max_depth=8, min_child_weight=1.0), y)
    model.fit(X, y)
    _, result = export_and_verify(model, X[:200])
    assert not result.fits_size_cap, "a production-scale model should exceed the cap"
    assert result.parity_ok, "over-cap is a size verdict, not a correctness failure"


def test_size_cap_is_one_megabyte() -> None:
    """Pinned deliberately.

    The cap is our own pre-commit setting, not a platform limit - GitHub's
    per-file hard limit is 100 MB. At 500 KB it was costing 0.122 phishing
    recall on the offline path, which is a product decision being made by a
    lint rule. Changing it should be a decision, so it is a test.
    """
    assert SIZE_CAP_BYTES == 1024 * 1024


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


# --------------------------------------------------------------------------
# XGBoost is the registry candidate, so its export path is tested too
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_xgb() -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(2)
    n, d = 4000, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.7 * X[:, 1] * X[:, 2] + rng.normal(0, 0.4, n) > 0.4).astype(int)
    model = build_xgboost(BoostedConfig(n_estimators=30, max_depth=5), y)
    model.fit(X, y)
    return model, X


def test_xgboost_export_matches_native(fitted_xgb) -> None:
    model, X = fitted_xgb
    _, result = export_and_verify(model, X[:800])
    assert result.parity_ok
    assert result.max_abs_parity_diff < PARITY_TOLERANCE


def test_xgboost_probabilities_are_valid(fitted_xgb) -> None:
    model, X = fitted_xgb
    probs = onnx_probabilities(to_onnx(model, X.shape[1]), X[:500])
    assert probs.shape == (500,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_unsupported_model_is_rejected() -> None:
    class NotATree:
        pass

    with pytest.raises(ExportError, match="no ONNX converter"):
        to_onnx(NotATree(), 5)
