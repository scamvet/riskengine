"""The extractor/spec contract.

``spec/url_lexical_v1.json`` is consumed by Core and by ScamPulse's dbt models.
If the extractor and the spec ever disagree, those consumers break silently at
a repo boundary no single test suite covers. These tests are the guarantee, and
they are the reason a spec version must be bumped rather than edited in place.
"""

from __future__ import annotations

import pytest

from scamvet_riskengine.features import spec_loader
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION, extract, feature_names

SAMPLES = [
    "https://www.sbi.co.in/web/personal-banking",
    "http://sbi-kyc-verify.xyz/login/update.php",
    "http://182.127.6.238:37368/i",
    "https://bit.ly/3xYz1Ab",
    "",
    "not a url",
]


def test_spec_loads() -> None:
    spec = spec_loader.load_spec(FEATURE_SPEC_VERSION)
    assert spec["spec_version"] == FEATURE_SPEC_VERSION
    assert spec["requires_network"] is False


def test_extractor_field_order_matches_spec_exactly() -> None:
    assert feature_names() == spec_loader.spec_feature_names(FEATURE_SPEC_VERSION)


@pytest.mark.parametrize("url", SAMPLES)
def test_samples_validate_against_spec(url: str) -> None:
    spec_loader.validate(extract(url), FEATURE_SPEC_VERSION)


def test_every_declared_field_has_required_metadata() -> None:
    for field in spec_loader.load_spec(FEATURE_SPEC_VERSION)["features"]:
        for key in ("name", "dtype", "nullable", "description"):
            assert key in field, f"{field.get('name')} missing {key}"
        assert field["description"].strip(), f"{field['name']} has empty description"


def test_missing_field_is_rejected() -> None:
    broken = extract("https://example.com/")
    broken.pop("tld")
    with pytest.raises(spec_loader.SpecViolation):
        spec_loader.validate(broken, FEATURE_SPEC_VERSION)


def test_out_of_range_value_is_rejected() -> None:
    broken = extract("https://example.com/")
    broken["host_digit_ratio"] = 1.5
    with pytest.raises(spec_loader.SpecViolation):
        spec_loader.validate(broken, FEATURE_SPEC_VERSION)


def test_unknown_spec_version_raises() -> None:
    with pytest.raises(FileNotFoundError):
        spec_loader.load_spec("url_lexical_v99")
