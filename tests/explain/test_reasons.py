"""Tests for attributions and the reasons[] contract.

The contract tests matter most. A reason that reaches a user as a raw column
name, or a reason list crowded out by one one-hot feature wearing fifty hats,
is worse than no explanation: it looks authoritative and says nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from scamvet_riskengine.explain import (
    ExplainError,
    Reason,
    ReasonError,
    build_reasons,
    contributions,
    explain_rows,
    global_importance,
    load_templates,
    render_english,
    unmapped_features,
    verify_additivity,
)
from scamvet_riskengine.features.featurize import LexicalFeaturizer, extract_frame
from scamvet_riskengine.models.boosted import BoostedConfig, build_lightgbm, build_xgboost

BENIGN = [f"https://newsdaily{i}.com/articles/section/{i}/story" for i in range(120)]
PHISH = [f"https://sbi-kyc-verify{i}.xyz/login/update/otp" for i in range(120)]


@pytest.fixture(scope="module")
def fitted():
    urls = BENIGN + PHISH
    raw = extract_frame(urls)
    featurizer = LexicalFeaturizer(max_tld_vocab=10).fit(raw)
    X = featurizer.transform(raw).to_numpy(dtype=np.float32)
    y = np.array([0] * len(BENIGN) + [1] * len(PHISH))
    model = build_lightgbm(BoostedConfig(n_estimators=40, max_depth=5), y)
    model.fit(X, y)
    return model, X, featurizer.feature_columns, y


# --------------------------------------------------------------------------
# Attributions
# --------------------------------------------------------------------------


def test_contributions_have_the_right_shape(fitted) -> None:
    model, X, names, _ = fitted
    values, base = contributions(model, X[:20])
    assert values.shape == (20, len(names))
    assert base.shape == (20,)


def test_attributions_reconcile_to_the_model_output(fitted) -> None:
    """TreeSHAP's defining property. Without it these are not explanations."""
    model, X, _, _ = fitted
    assert verify_additivity(model, X[:100]) < 1e-6


def test_additivity_violation_would_raise(fitted) -> None:
    model, X, _, _ = fitted
    with pytest.raises(ExplainError, match="reconcile"):
        verify_additivity(model, X[:20], tolerance=-1.0)


def test_global_importance_ranks_features(fitted) -> None:
    model, X, names, _ = fitted
    importance = global_importance(model, X, names, max_samples=200)
    ranked = importance.ranked(top=5)
    assert len(ranked) == 5
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:], strict=False))
    assert importance.n_samples == 200


def test_global_importance_rejects_a_name_mismatch(fitted) -> None:
    model, X, _, _ = fitted
    with pytest.raises(ExplainError, match="feature names"):
        global_importance(model, X, ["only", "two"], max_samples=50)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_every_feature_column_has_a_template(fitted) -> None:
    """The guarantee that no raw column name can reach a user.

    If this fails, a feature was added without its template. Add the template
    in the same change rather than relaxing the test.
    """
    _, _, names, _ = fitted
    assert unmapped_features(names) == []


def test_every_template_key_used_is_defined() -> None:
    templates = load_templates()
    defined = set(templates["templates"])
    used = {entry["template_key"] for entry in templates["features"].values()}
    used |= {group["template_key"] for group in templates["groups"].values()}
    assert used <= defined, f"undefined template keys: {sorted(used - defined)}"


def test_one_hot_columns_collapse_to_one_reason(fitted) -> None:
    model, X, names, _ = fitted
    reasons = explain_rows(model, X[:40], names, top_k=20)
    for row in reasons:
        tld_reasons = [r for r in row if r.template_key == "tld_reputation"]
        assert len(tld_reasons) <= 1, "one-hot TLD columns must surface once"


def test_reasons_are_ordered_and_bounded(fitted) -> None:
    model, X, names, _ = fitted
    for row in explain_rows(model, X[:50], names, top_k=4):
        assert len(row) <= 4
        magnitudes = [r.magnitude for r in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(0.0 <= value <= 1.0 for value in magnitudes)


def test_direction_is_present_and_valid(fitted) -> None:
    model, X, names, _ = fitted
    for row in explain_rows(model, X[:50], names):
        for reason in row:
            assert reason.direction in {"increases", "decreases"}


def test_both_directions_occur_across_a_corpus(fitted) -> None:
    """A verdict that only ever incriminates is a prosecutor, not an explainer."""
    model, X, names, _ = fitted
    directions = {r.direction for row in explain_rows(model, X, names) for r in row}
    assert directions == {"increases", "decreases"}


def test_a_dominant_feature_becomes_its_template(fitted) -> None:
    """The mapping this module owns, tested without a model in the way.

    An earlier version of this test trained on a synthetic fixture and asserted
    that brand features would surface. They did not - the model separated the
    classes on address length and path depth instead, and the explainer
    faithfully reported that. The explainer was right and the test was wrong:
    it was asserting something about model behaviour in a test of reason
    construction. Attribution is supplied directly here so the assertion is
    about the thing under test.
    """
    _, X, names, _ = fitted
    values = np.zeros(len(names))
    values[names.index("brand_domain_mismatch")] = 3.0
    values[names.index("keyword_hits_auth")] = 1.0
    reasons = build_reasons(values, X[0], names, top_k=4)
    keys = [r.template_key for r in reasons]
    assert keys[0] == "brand_impersonation"
    assert "login_keywords" in keys
    assert reasons[0].direction == "increases"
    assert reasons[0].severity == "high"


def test_negative_attribution_reads_as_reassuring(fitted) -> None:
    _, X, names, _ = fitted
    values = np.zeros(len(names))
    values[names.index("brand_domain_mismatch")] = -2.0
    reason = build_reasons(values, X[0], names)[0]
    assert reason.direction == "decreases"


def test_model_reasons_are_all_known_templates(fitted) -> None:
    """Whatever the model happens to use, every reason must be renderable."""
    model, X, names, _ = fitted
    known = set(load_templates()["templates"])
    produced = {r.template_key for row in explain_rows(model, X, names) for r in row}
    assert produced
    assert produced <= known


def test_weak_reasons_are_dropped(fitted) -> None:
    model, X, names, _ = fitted
    values, _ = contributions(model, X[:1])
    reasons = build_reasons(values[0], X[0], names, top_k=50, min_magnitude=0.30)
    assert all(r.magnitude >= 0.30 for r in reasons)


def test_length_mismatch_raises(fitted) -> None:
    _, X, names, _ = fitted
    with pytest.raises(ReasonError, match="length mismatch"):
        build_reasons(np.zeros(3), X[0], names)


def test_zero_attribution_gives_no_reasons(fitted) -> None:
    _, X, names, _ = fitted
    assert build_reasons(np.zeros(len(names)), X[0], names) == []


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_reference_rendering_is_a_sentence(fitted) -> None:
    model, X, names, _ = fitted
    for row in explain_rows(model, X[:20], names):
        for reason in row:
            text = render_english(reason)
            assert text and text[0].isupper()
            assert "_" not in text.split("(")[0], "no raw column name in user text"


def test_reason_serialises() -> None:
    payload = Reason(
        feature_id="brand_domain_mismatch",
        template_key="brand_impersonation",
        direction="increases",
        magnitude=0.4123,
        evidence=None,
        severity="high",
        observed=1.0,
        evidence_kind="flag",
    ).as_dict()
    assert payload["magnitude"] == 0.4123
    assert set(payload) == {
        "feature_id",
        "template_key",
        "direction",
        "magnitude",
        "evidence",
        "severity",
        "observed",
        "evidence_kind",
        "asserted_when",
    }


# --------------------------------------------------------------------------
# Absent-form rendering
# --------------------------------------------------------------------------


def test_an_unset_flag_renders_its_absent_form(fitted) -> None:
    """A flag contributing negatively means its absence lowered risk.

    Rendering the positive phrasing would state something untrue about the
    address. Found by reading real output, not by any metric.
    """
    _, X, names, _ = fitted
    row = X[0].copy()
    row[names.index("host_is_ip")] = 0.0
    values = np.zeros(len(names))
    values[names.index("host_is_ip")] = -2.0
    reason = build_reasons(values, row, names)[0]
    assert reason.template_key == "ip_address_host"
    assert reason.direction == "decreases"
    assert render_english(reason) == "The address uses a domain name rather than a raw IP number."


def test_a_set_flag_renders_its_positive_form(fitted) -> None:
    _, X, names, _ = fitted
    row = X[0].copy()
    row[names.index("host_is_ip")] = 1.0
    values = np.zeros(len(names))
    values[names.index("host_is_ip")] = 2.0
    reason = build_reasons(values, row, names)[0]
    assert (
        render_english(reason) == "The address points at a raw IP number instead of a domain name."
    )


def test_continuous_templates_are_neutral() -> None:
    """A positive attribution does not mean a high value.

    Continuous templates once asserted direction - "the path is unusually
    long" - and were rendered over a zero-length path. They are now neutral
    noun phrases; Core composes the directional sentence from the direction
    and magnitude that travel on the Reason.
    """
    templates = load_templates()
    continuous = {
        entry["template_key"]
        for entry in templates["features"].values()
        if entry["evidence_kind"] != "flag"
    }
    banned = ("unusually", "not encrypted", "does not")
    for key in continuous:
        text = templates["templates"][key]["en"].lower()
        for word in banned:
            assert word not in text, f"{key} still asserts a direction: {text!r}"


def test_every_flag_template_has_an_absent_form() -> None:
    """Any template reachable from a flag must be renderable both ways."""
    templates = load_templates()
    flag_keys = {
        entry["template_key"]
        for entry in templates["features"].values()
        if entry["evidence_kind"] == "flag"
    }
    missing = [k for k in flag_keys if "en_absent" not in templates["templates"][k]]
    assert missing == [], f"flag templates without an absent reading: {missing}"


def test_attributions_work_for_xgboost_too(fitted) -> None:
    """XGBoost is the registry candidate; its contribution path differs.

    LightGBM exposes pred_contrib on the sklearn wrapper, XGBoost only on the
    underlying Booster, and XGBoost reconciles to about 2e-06 rather than
    5e-15 because its contribution path is float32 internally.
    """
    _, X, names, y = fitted
    model = build_xgboost(BoostedConfig(n_estimators=30, max_depth=4), y)
    model.fit(X, y)
    assert verify_additivity(model, X[:200]) < 1e-4
    rows = explain_rows(model, X[:20], names)
    assert any(rows)
    known = set(load_templates()["templates"])
    assert {r.template_key for row in rows for r in row} <= known
