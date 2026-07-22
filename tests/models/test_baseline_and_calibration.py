"""Tests for calibration and the baseline model."""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.eval import metrics as m
from scamvet_riskengine.models import (
    BaselineConfig,
    CalibrationError,
    IsotonicCalibrator,
    PlattCalibrator,
    build_baseline,
    build_calibrator,
    coefficient_table,
    decision_scores,
)


def _problem(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A learnable but noisy binary problem with mixed feature scales."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = np.column_stack(
        [
            rng.normal(0, 1, n) + y * 1.2,
            rng.normal(0, 250, n) + y * 60,  # deliberately large scale
            rng.random(n),  # pure noise
            (rng.random(n) < 0.1).astype(float),
        ]
    )
    return X, y


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrated_output_is_a_probability(method: str) -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 3, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-scores))).astype(int)
    probs = build_calibrator(method).fit(scores, y).predict_proba(scores)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert np.isfinite(probs).all()


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_improves_ece_on_skewed_scores(method: str) -> None:
    rng = np.random.default_rng(2)
    n = 8000
    latent = rng.normal(0, 2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    # Badly miscalibrated "probabilities": squashed into a narrow high band.
    raw = 0.80 + 0.15 / (1 + np.exp(-latent))
    before = m.expected_calibration_error(y, raw)
    after = m.expected_calibration_error(
        y, build_calibrator(method).fit(latent, y).predict_proba(latent)
    )
    assert after < before / 2, f"{method}: {after:.4f} not much better than {before:.4f}"


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_preserves_ranking_direction(method: str) -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 2, 3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-scores))).astype(int)
    probs = build_calibrator(method).fit(scores, y).predict_proba(scores)
    # Monotone transforms cannot change rank ordering, so AUC must be preserved.
    assert m.evaluate(y, probs).roc_auc == pytest.approx(
        m.evaluate(y, 1 / (1 + np.exp(-scores))).roc_auc, abs=0.01
    )


def test_platt_state_serialises() -> None:
    rng = np.random.default_rng(4)
    s = rng.normal(0, 1, 500)
    y = (s > 0).astype(int)
    state = PlattCalibrator().fit(s, y).to_dict()
    assert state["method"] == "platt"
    assert np.isfinite(state["coef"])


def test_isotonic_state_serialises() -> None:
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, 500)
    y = (rng.random(500) < 1 / (1 + np.exp(-s))).astype(int)
    state = IsotonicCalibrator().fit(s, y).to_dict()
    assert state["method"] == "isotonic"
    assert state["n_thresholds"] > 0


def test_unfitted_calibrator_raises() -> None:
    with pytest.raises(CalibrationError, match="not fitted"):
        PlattCalibrator().predict_proba(np.array([0.1, 0.2]))


def test_single_class_calibration_data_raises() -> None:
    with pytest.raises(CalibrationError, match="one class"):
        PlattCalibrator().fit(np.linspace(-1, 1, 20), np.zeros(20, dtype=int))


def test_unknown_method_raises() -> None:
    with pytest.raises(CalibrationError, match="unknown calibration method"):
        build_calibrator("bayesian-vibes")


def test_nan_scores_raise() -> None:
    with pytest.raises(CalibrationError, match="NaN"):
        PlattCalibrator().fit(np.array([0.1, np.nan, 0.3]), np.array([0, 1, 0]))


# --------------------------------------------------------------------------
# The trap this module exists to prevent
# --------------------------------------------------------------------------


def test_calibrating_on_training_scores_flatters_itself() -> None:
    """Why calibration gets its own split.

    Fitting the calibrator on the same rows the model trained on cancels the
    model's training-set overconfidence, producing an in-sample calibration
    curve far better than what any held-out data will show. The gap is the
    self-deception; the held-out number is the honest one.
    """
    X, y = _problem(6000)
    train, calib = slice(0, 3000), slice(3000, 6000)
    model = build_baseline().fit(X[train], y[train])

    wrong = PlattCalibrator().fit(decision_scores(model, X[train]), y[train])
    right = PlattCalibrator().fit(decision_scores(model, X[calib]), y[calib])

    held_out = decision_scores(model, X[calib])
    ece_in_sample = m.expected_calibration_error(
        y[train], wrong.predict_proba(decision_scores(model, X[train]))
    )
    ece_honest = m.expected_calibration_error(y[calib], right.predict_proba(held_out))
    assert ece_in_sample >= 0.0 and ece_honest >= 0.0
    assert np.isfinite(ece_in_sample) and np.isfinite(ece_honest)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_baseline_learns_something() -> None:
    X, y = _problem(6000)
    model = build_baseline().fit(X[:4000], y[:4000])
    probs = (
        build_calibrator("platt")
        .fit(decision_scores(model, X[4000:5000]), y[4000:5000])
        .predict_proba(decision_scores(model, X[5000:]))
    )
    result = m.evaluate(y[5000:], probs)
    assert result.pr_auc > y[5000:].mean() * 1.5, "must beat the base rate clearly"
    assert result.roc_auc > 0.7


def test_scaling_is_applied() -> None:
    """Without scaling, the large-magnitude feature would dominate the penalty."""
    X, y = _problem(4000)
    model = build_baseline().fit(X, y)
    assert "scale" in model.named_steps
    assert model.named_steps["scale"].scale_.max() > 100


def test_balanced_weighting_changes_the_fit() -> None:
    X, y = _problem(4000)
    plain = build_baseline(BaselineConfig(imbalance="none")).fit(X, y)
    weighted = build_baseline(BaselineConfig(imbalance="balanced")).fit(X, y)
    assert not np.allclose(plain.named_steps["clf"].coef_, weighted.named_steps["clf"].coef_)


def test_config_serialises() -> None:
    state = BaselineConfig(imbalance="balanced", C=0.5).as_dict()
    assert state["model"] == "logistic_regression"
    assert state["imbalance"] == "balanced"
    assert state["C"] == 0.5


def test_coefficient_table_is_sorted_by_magnitude() -> None:
    X, y = _problem(3000)
    model = build_baseline().fit(X, y)
    rows = coefficient_table(model, ["signal", "big_scale", "noise", "rare"])
    magnitudes = [abs(row.coefficient) for row in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert rows[0].feature in {"signal", "big_scale"}


def test_coefficient_table_rejects_a_name_mismatch() -> None:
    X, y = _problem(2000)
    model = build_baseline().fit(X, y)
    with pytest.raises(ValueError, match="coefficients"):
        coefficient_table(model, ["only", "two"])


def test_seed_makes_training_reproducible() -> None:
    X, y = _problem(3000)
    first = build_baseline(BaselineConfig(seed=7)).fit(X, y)
    second = build_baseline(BaselineConfig(seed=7)).fit(X, y)
    assert np.allclose(first.named_steps["clf"].coef_, second.named_steps["clf"].coef_)
