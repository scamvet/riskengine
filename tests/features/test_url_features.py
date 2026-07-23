"""Tests for the lexical URL feature extractor.

Two layers:

* property tests, which assert the invariants that must hold for *any* input,
  because scam URLs are adversarial and the extractor sits on the offline path
  where an exception means no verdict at all;
* golden cases, which pin the specific signals the model is supposed to learn.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scamvet_riskengine.features import spec_loader
from scamvet_riskengine.features.url import (
    BRAND_DISTANCE_CAP,
    FEATURE_SPEC_VERSION,
    _levenshtein,
    _shannon_entropy,
    extract,
    feature_names,
)

SETTINGS = settings(
    max_examples=250,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


# --------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------


@SETTINGS
@given(st.text())
def test_never_raises_on_arbitrary_text(text: str) -> None:
    extract(text)


@SETTINGS
@given(st.text())
def test_always_returns_full_declared_field_set(text: str) -> None:
    assert list(extract(text).keys()) == feature_names()


@SETTINGS
@given(st.text())
def test_output_always_satisfies_the_spec(text: str) -> None:
    spec_loader.validate(extract(text), FEATURE_SPEC_VERSION)


@SETTINGS
@given(st.text())
def test_is_deterministic(text: str) -> None:
    assert extract(text) == extract(text)


@SETTINGS
@given(st.text())
def test_no_nan_floats(text: str) -> None:
    for name, value in extract(text).items():
        if isinstance(value, float):
            assert not math.isnan(value), f"{name} is NaN"
            assert math.isfinite(value), f"{name} is not finite"


@SETTINGS
@given(st.text(), st.sampled_from(["", " ", "\t", "\n", "  "]))
def test_surrounding_whitespace_is_ignored(text: str, pad: str) -> None:
    assert extract(text) == extract(pad + text + pad)


@SETTINGS
@given(st.text(min_size=1), st.text(min_size=1))
def test_levenshtein_is_symmetric_and_bounded(a: str, b: str) -> None:
    forward = _levenshtein(a, b)
    assert forward == _levenshtein(b, a)
    assert 0 <= forward <= BRAND_DISTANCE_CAP
    assert (forward == 0) == (a == b)


@SETTINGS
@given(st.text())
def test_entropy_is_bounded(text: str) -> None:
    assert 0.0 <= _shannon_entropy(text) <= 8.0


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["", "   ", "\n", "http://", "https://", "://", "not a url at all", "...", "@@@"],
)
def test_unparseable_input_is_flagged_not_raised(value: str) -> None:
    result = extract(value)
    assert result["parse_ok"] == 0
    assert result["brand_min_distance"] == BRAND_DISTANCE_CAP


def test_non_string_input_is_tolerated() -> None:
    assert extract(None)["parse_ok"] == 0  # type: ignore[arg-type]
    assert extract(12345)["parse_ok"] == 0  # type: ignore[arg-type]


def test_malformed_port_does_not_raise() -> None:
    assert extract("http://example.com:notaport/x")["parse_ok"] == 0


# --------------------------------------------------------------------------
# Golden cases
# --------------------------------------------------------------------------


def test_plain_legitimate_url() -> None:
    f = extract("https://www.sbi.co.in/web/personal-banking")
    assert f["parse_ok"] == 1
    assert f["tld"] == "in"
    assert f["host_is_ip"] == 0
    assert f["brand_in_host"] == 1
    assert f["brand_domain_mismatch"] == 0, "official SBI domain must not flag"


def test_brand_impersonation_is_flagged() -> None:
    f = extract("http://sbi-kyc-verify.xyz/login/update.php")
    assert f["brand_in_host"] == 1
    assert f["brand_domain_mismatch"] == 1
    assert f["suspicious_tld"] == 1
    assert f["num_hyphens_host"] == 2
    assert f["keyword_hits_auth"] >= 3


def test_subdomain_impersonation_is_flagged() -> None:
    f = extract("https://hdfcbank.com.secure-login.top/otp")
    assert f["brand_in_host"] == 1
    assert f["brand_domain_mismatch"] == 1, "brand in subdomain, not registered domain"
    assert f["num_subdomains"] >= 2


def test_typosquat_has_low_nonzero_brand_distance() -> None:
    f = extract("https://paytn.com/login")
    assert 0 < f["brand_min_distance"] <= 3


def test_ip_host() -> None:
    f = extract("http://182.127.6.238:37368/i")
    assert f["host_is_ip"] == 1
    assert f["host_is_ipv6"] == 0
    assert f["has_port"] == 1
    assert f["num_subdomains"] == 0
    assert f["tld"] == ""
    assert f["brand_min_distance"] == BRAND_DISTANCE_CAP


def test_ipv6_host() -> None:
    f = extract("http://[2001:db8::1]/path")
    assert f["host_is_ip"] == 1
    assert f["host_is_ipv6"] == 1


def test_punycode_homograph() -> None:
    assert extract("http://xn--pytm-2ta.com/pay")["has_punycode"] == 1


def test_shortener_detected() -> None:
    f = extract("https://bit.ly/3xYz1Ab")
    assert f["is_shortener"] == 1
    assert extract("https://sbi.co.in/x")["is_shortener"] == 0


def test_at_symbol_obfuscation() -> None:
    f = extract("http://sbi.co.in@evil.xyz/login")
    assert f["has_at_symbol"] == 1
    assert f["tld"] == "xyz", "true host is after the @, not before"


def test_scheme_is_optional() -> None:
    """A missing scheme must not change what the URL is understood to be.

    ``had_scheme`` was itself a feature until v3, when it turned out to encode
    which upstream file a corpus row came from rather than anything about the
    address. What remains is the requirement that parsing works either way.
    """
    with_scheme = extract("http://example.com/a")
    without = extract("example.com/a")
    assert without["parse_ok"] == 1
    assert without["tld"] == with_scheme["tld"]
    assert without["host_length"] == with_scheme["host_length"]
    assert "had_scheme" not in without


def test_indian_second_level_suffix_resolves() -> None:
    f = extract("https://portal.incometax.gov.in/iec/foservices/")
    assert f["tld"] == "in"
    assert f["num_subdomains"] == 1, "gov.in is a public suffix, so only 'portal' is sub"


def test_keyword_categories_are_separated() -> None:
    f = extract("http://x.com/urgent/refund/claim/verify")
    assert f["keyword_hits_auth"] >= 1
    assert f["keyword_hits_payment"] >= 1
    assert f["keyword_hits_urgency"] >= 1
    assert f["keyword_hits_reward"] >= 1
    assert f["keyword_hits_total"] == (
        f["keyword_hits_auth"]
        + f["keyword_hits_payment"]
        + f["keyword_hits_urgency"]
        + f["keyword_hits_reward"]
    )


def test_percent_encoding_counted() -> None:
    assert extract("http://x.com/a%20b%2Fc")["num_encoded_chars"] == 2


def test_high_entropy_dga_style_host() -> None:
    low = extract("https://google.com/")["host_entropy"]
    high = extract("https://x7k2p9qz3mv8bwn4.com/")["host_entropy"]
    assert high > low


# --------------------------------------------------------------------------
# Scheme invariance
# --------------------------------------------------------------------------


@SETTINGS
@given(
    st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789.-/?=&_%#:"),
        min_size=3,
        max_size=60,
    )
)
def test_features_do_not_depend_on_the_scheme(body: str) -> None:
    """The invariant that v5 exists to hold.

    Whole-URL features were measured on the raw input, which includes the
    scheme. 72% of corpus rows carry no scheme while every real user pastes
    one, so ``url_length`` - the top-ranked feature by attribution - shifted by
    eight characters between training and inference. The same site scored 0.018
    bare and 0.985 with https, either side of the red threshold, and 36 of 55
    domains the project itself asserts are legitimate were flagged.

    Any feature computed over the raw string inherits how that string happened
    to be stored, so the property is asserted over generated input rather than
    a fixture. The alphabet is URL-shaped deliberately: over arbitrary bytes
    the property is false for a reason unrelated to what it protects, since
    leading whitespace is stripped from a bare string but becomes an invalid
    interior character once a scheme is prepended.
    """
    bare = extract(body)
    if not bare["parse_ok"]:
        return
    assert extract("https://" + body) == bare
    assert extract("http://" + body) == bare


@pytest.mark.parametrize(
    "url",
    [
        "bankofbaroda.com/",
        "sbi.co.in/web/personal-banking",
        "example.com:8080/a#frag",
        "x.com/p?q=1&r=2",
        "192.168.1.1/admin",
    ],
)
def test_known_addresses_are_scheme_invariant(url: str) -> None:
    assert extract(url) == extract("https://" + url) == extract("http://" + url)


def test_canonical_length_excludes_the_scheme() -> None:
    f = extract("https://example.com/a")
    assert f["url_length"] == len("example.com/a")
