"""Tests for the model registry.

Three of these guard against shipping something incoherent rather than merely
broken: a research-track model reaching users, a model loaded against a feature
spec it was not trained on, and a manifest whose thresholds do not order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scamvet_riskengine.features.featurize import LexicalFeaturizer, extract_frame
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION
from scamvet_riskengine.models.boosted import BoostedConfig, build_xgboost
from scamvet_riskengine.registry import (
    Manifest,
    ManifestError,
    ScorerError,
    Thresholds,
    list_models,
    load,
    resolve,
)
from scamvet_riskengine.registry.manifest import parse_version, satisfies

pytest.importorskip("onnxruntime", reason="ONNX extra not installed")

from scamvet_riskengine.models.onnx_export import export_and_verify  # noqa: E402

BENIGN = [f"https://newsdaily{i}.com/articles/section/{i}" for i in range(150)]
PHISH = [f"https://sbi-kyc-verify{i}.xyz/login/update/otp" for i in range(150)]


def _publish(root: Path, version: str = "0.1.0", track: str = "product") -> Path:
    """Build a tiny but genuine registry entry: real model, real ONNX, real manifest."""
    urls = BENIGN + PHISH
    y = np.array([0] * len(BENIGN) + [1] * len(PHISH))
    raw = extract_frame(urls)
    featurizer = LexicalFeaturizer().fit(raw)
    X = featurizer.transform(raw).to_numpy(dtype=np.float32)

    model = build_xgboost(BoostedConfig(n_estimators=20, max_depth=4), y)
    model.fit(X, y)

    target = root / "url-scorer-lexical" / version
    target.mkdir(parents=True, exist_ok=True)
    blob, export = export_and_verify(model, X[:100], path=target / "model.onnx")
    model.get_booster().save_model(str(target / "model.json"))

    Manifest(
        name="url-scorer-lexical",
        version=version,
        track=track,  # type: ignore[arg-type]
        feature_spec_version=FEATURE_SPEC_VERSION,
        featurizer=featurizer.to_dict(),
        model={"family": "xgboost", "artifact": "model.onnx", "size_bytes": export.size_bytes},
        calibration={"method": "platt", "coef": 1.0, "intercept": 0.0},
        thresholds={"vet": Thresholds(red=0.8, amber=0.4)},
        metrics={"phishing": {}},
        corpus="test fixture",
        requires_network=False,
    ).write(target / "manifest.json")
    return target


@pytest.fixture(scope="module")
def registry(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("registry")
    _publish(root, "0.1.0")
    _publish(root, "0.2.0")
    return root


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------


def test_parse_version() -> None:
    assert parse_version("1.2.3") == (1, 2, 3)
    with pytest.raises(ManifestError):
        parse_version("1.2")


@pytest.mark.parametrize(
    ("version", "spec", "expected"),
    [
        ("0.2.0", None, True),
        ("0.2.0", "0.2.0", True),
        ("0.2.0", "==0.1.0", False),
        ("0.2.0", ">=0.1.0", True),
        ("0.1.0", ">=0.2.0", False),
        ("0.2.5", "~=0.2.0", True),
        ("0.3.0", "~=0.2.0", False),
        ("0.9.0", "^0.2.0", True),
        ("1.0.0", "^0.2.0", False),
    ],
)
def test_satisfies(version: str, spec: str | None, expected: bool) -> None:
    assert satisfies(version, spec) is expected


def test_bad_specifier_raises() -> None:
    with pytest.raises(ManifestError):
        satisfies("1.0.0", "roughly 1")


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_newest_version_wins(registry: Path) -> None:
    manifest, _ = resolve("url-scorer-lexical", root=registry)
    assert manifest.version == "0.2.0"


def test_specifier_selects_an_older_version(registry: Path) -> None:
    manifest, _ = resolve("url-scorer-lexical", "==0.1.0", root=registry)
    assert manifest.version == "0.1.0"


def test_unknown_name_lists_what_exists(registry: Path) -> None:
    with pytest.raises(ManifestError, match="Available"):
        resolve("no-such-model", root=registry)


def test_unsatisfiable_specifier_lists_present_versions(registry: Path) -> None:
    with pytest.raises(ManifestError, match="present"):
        resolve("url-scorer-lexical", ">=9.0.0", root=registry)


def test_list_models(registry: Path) -> None:
    manifests = list_models(registry)
    assert {m.version for m in manifests} == {"0.1.0", "0.2.0"}


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------


def test_research_track_models_are_refused(tmp_path: Path) -> None:
    """Research models train on NC-licensed corpora and must not reach users."""
    _publish(tmp_path, "0.1.0", track="research")
    with pytest.raises(ManifestError, match="research-track"):
        load("url-scorer-lexical", root=tmp_path)


def test_feature_spec_mismatch_is_refused(tmp_path: Path) -> None:
    """Four spec versions have come and gone; a mismatch is silent nonsense."""
    target = _publish(tmp_path, "0.1.0")
    payload = json.loads((target / "manifest.json").read_text())
    payload["feature_spec_version"] = "url_lexical_v1"
    (target / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ScorerError, match="feature spec"):
        load("url-scorer-lexical", root=tmp_path)


def test_missing_artifact_is_refused(tmp_path: Path) -> None:
    target = _publish(tmp_path, "0.1.0")
    (target / "model.onnx").unlink()
    with pytest.raises(ScorerError, match="artifact missing"):
        load("url-scorer-lexical", root=tmp_path)


def test_thresholds_must_order() -> None:
    with pytest.raises(ManifestError, match="amber <= red"):
        Thresholds.from_dict({"red": 0.2, "amber": 0.8})


def test_manifest_requires_its_fields() -> None:
    with pytest.raises(ManifestError, match="missing required field"):
        Manifest.from_dict({"name": "x", "version": "0.1.0"})


def test_unknown_use_case_lists_available(registry: Path) -> None:
    scorer = load("url-scorer-lexical", root=registry)
    with pytest.raises(ManifestError, match="available"):
        scorer.score("https://example.com/", use_case="nonexistent")


# --------------------------------------------------------------------------
# Scoring end to end
# --------------------------------------------------------------------------


def test_score_returns_the_contract(registry: Path) -> None:
    result = load("url-scorer-lexical", root=registry).score("https://sbi-kyc-verify9.xyz/login")
    assert 0.0 <= result.probability <= 1.0
    assert result.band in {"green", "amber", "red"}
    assert 0.0 <= result.uncertainty <= 1.0
    assert result.feature_spec_version == FEATURE_SPEC_VERSION
    assert result.requires_network is False
    assert result.model_version == "0.2.0"


def test_score_serialises(registry: Path) -> None:
    payload = load("url-scorer-lexical", root=registry).score("https://example.com/").as_dict()
    assert set(payload) == {
        "url",
        "probability",
        "band",
        "reasons",
        "uncertainty",
        "flags",
        "use_case",
        "model_name",
        "model_version",
        "feature_spec_version",
        "requires_network",
    }
    json.dumps(payload)  # must be JSON-safe for Core and the warehouse


def test_batch_matches_single(registry: Path) -> None:
    scorer = load("url-scorer-lexical", root=registry)
    urls = ["https://sbi-kyc-verify1.xyz/login", "https://newsdaily1.com/articles/section/1"]
    batch = scorer.score_many(urls)
    for url, batched in zip(urls, batch, strict=True):
        assert scorer.score(url).probability == pytest.approx(batched.probability)


def test_unparseable_input_is_flagged_not_raised(registry: Path) -> None:
    result = load("url-scorer-lexical", root=registry).score("not a url at all")
    assert "unparseable_input" in result.flags


def test_reasons_are_produced_when_the_native_model_is_present(registry: Path) -> None:
    result = load("url-scorer-lexical", root=registry).score("https://sbi-kyc-verify3.xyz/login")
    assert result.reasons, "reasons require model.json alongside the ONNX artifact"
    assert all(r.template_key for r in result.reasons)


def test_scoring_survives_without_the_native_model(tmp_path: Path) -> None:
    """Reasons are best-effort; bands are not."""
    target = _publish(tmp_path, "0.1.0")
    (target / "model.json").unlink()
    (target / "model.json.gz").unlink(missing_ok=True)
    result = load("url-scorer-lexical", root=tmp_path).score("https://example.com/")
    assert result.band in {"green", "amber", "red"}
    assert result.reasons == ()


def test_use_case_changes_the_band(registry: Path) -> None:
    target = registry / "url-scorer-lexical" / "0.2.0"
    payload = json.loads((target / "manifest.json").read_text())
    payload["thresholds"]["qr"] = {"red": 0.05, "amber": 0.01}
    (target / "manifest.json").write_text(json.dumps(payload))
    scorer = load("url-scorer-lexical", root=registry)
    url = "https://newsdaily2.com/articles/section/2"
    strict = scorer.score(url, use_case="vet")
    lenient = scorer.score(url, use_case="qr")
    assert strict.probability == pytest.approx(lenient.probability)
    order = {"green": 0, "amber": 1, "red": 2}
    assert order[lenient.band] >= order[strict.band]


def test_native_model_loads_from_gzip(tmp_path: Path) -> None:
    """Published entries store the native model gzipped.

    XGBoost's JSON dump is several megabytes uncompressed and compresses by
    roughly an order of magnitude; it is committed once per version and lives
    in git forever, so the uncompressed form is waste. Reasons must survive
    the change.
    """
    import gzip

    target = _publish(tmp_path, "0.1.0")
    raw = (target / "model.json").read_bytes()
    (target / "model.json.gz").write_bytes(gzip.compress(raw, 9))
    (target / "model.json").unlink()

    result = load("url-scorer-lexical", root=tmp_path).score("https://sbi-kyc-verify5.xyz/login")
    assert result.reasons, "gzipped native model must still produce reasons"
    assert len((target / "model.json.gz").read_bytes()) < len(raw)
