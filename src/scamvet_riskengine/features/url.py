"""Lexical feature extraction for URLs.

Pure and offline by construction: no network calls, no filesystem access
beyond loading the versioned data files at import, no clocks, no randomness.
The same URL always yields the same features for a given
(FEATURE_SPEC_VERSION, tldextract version) pair.

This module backs ``url-scorer-lexical``, the model that must keep working
when every network-dependent enrichment source is unavailable. See ADR-004
(always-live hot path) and ADR-010 (feeds are enrichment, never training).

``extract`` never raises. Unparseable input yields the full feature dict with
``parse_ok = 0`` and neutral values, because a scorer that throws on a weird
string is a scorer that fails on exactly the inputs scams are made of.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import tldextract

FEATURE_SPEC_VERSION = "url_lexical_v1"

_DATA_DIR = Path(__file__).parent / "data"

# Distance sentinel: returned when no brand comparison is possible (e.g. the
# host is a bare IP). Chosen above any realistic label length so that "no
# comparison" sorts as maximally dissimilar rather than as a near match.
BRAND_DISTANCE_CAP = 64

_MAX_ENTROPY = 8.0

# Frozen public-suffix resolution. suffix_list_urls=() disables the network
# fetch; fallback_to_snapshot=True uses the snapshot bundled with the pinned
# tldextract version. Both are required for reproducibility: a live PSL fetch
# would make feature values drift silently between training and serving.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")

# Characters that cannot appear in a hostname. Unicode is deliberately allowed
# through, because users paste internationalised domains in their display form
# and rejecting them would blind the extractor to homograph attacks.
_INVALID_HOST_CHARS = frozenset(" \t\n\r\v\f\"'<>\\^`{}|,;()!$&*+=?")


def _load(name: str) -> dict[str, Any]:
    with (_DATA_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _keywords() -> dict[str, Any]:
    return _load("url_keywords_v1.json")


@lru_cache(maxsize=1)
def _brands() -> dict[str, Any]:
    return _load("brands_v1.json")


@lru_cache(maxsize=1)
def _shorteners() -> frozenset[str]:
    return frozenset(_load("shorteners_v1.json")["domains"])


def _shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character. Empty string is 0.0."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return min(entropy, _MAX_ENTROPY)


def _levenshtein(a: str, b: str, cap: int = BRAND_DISTANCE_CAP) -> int:
    """Edit distance with an early cutoff.

    Implemented here rather than pulled from a dependency: the comparison set
    is tiny (one host label against a fixed brand list) and keeping the
    lexical path dependency-light matters for the on-device ONNX story.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= cap:
        return cap
    if not a:
        return min(len(b), cap)
    if not b:
        return min(len(a), cap)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        if min(current) >= cap:
            return cap
        previous = current
    return min(previous[-1], cap)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def _is_ip_host(host: str) -> tuple[int, int]:
    """Return (is_ip, is_ipv6) as 0/1 ints.

    Note that ``urlsplit(...).hostname`` already strips the brackets from an
    IPv6 literal, so the bracketed form is handled defensively rather than as
    the expected input.
    """
    candidate = host
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return 0, 0
    return 1, (1 if address.version == 6 else 0)


def _host_is_plausible(host: str) -> bool:
    """Reject strings that parsed into a host slot but cannot be a host.

    ``urlsplit`` is permissive: prefixing an arbitrary sentence with a scheme
    yields a "hostname" containing spaces. Without this guard the extractor
    reports ``parse_ok = 1`` for plain prose, and every downstream feature
    becomes noise measured on a non-URL.
    """
    if not host:
        return False
    if any(ch in _INVALID_HOST_CHARS for ch in host):
        return False
    return any(ch.isalnum() for ch in host)


def _empty_features() -> dict[str, Any]:
    """Neutral feature dict used when the input cannot be parsed at all."""
    return {
        "parse_ok": 0,
        "had_scheme": 0,
        "scheme_is_https": 0,
        "url_length": 0,
        "host_length": 0,
        "path_length": 0,
        "query_length": 0,
        "fragment_length": 0,
        "num_path_segments": 0,
        "num_query_params": 0,
        "num_dots_host": 0,
        "num_hyphens_host": 0,
        "num_subdomains": 0,
        "longest_host_label_len": 0,
        "host_is_ip": 0,
        "host_is_ipv6": 0,
        "has_port": 0,
        "has_punycode": 0,
        "has_at_symbol": 0,
        "has_double_slash_in_path": 0,
        "num_encoded_chars": 0,
        "host_digit_ratio": 0.0,
        "url_digit_ratio": 0.0,
        "host_entropy": 0.0,
        "path_entropy": 0.0,
        "tld": "",
        "tld_length": 0,
        "suspicious_tld": 0,
        "is_shortener": 0,
        "keyword_hits_total": 0,
        "keyword_hits_auth": 0,
        "keyword_hits_payment": 0,
        "keyword_hits_urgency": 0,
        "keyword_hits_reward": 0,
        "brand_in_host": 0,
        "brand_domain_mismatch": 0,
        "brand_min_distance": BRAND_DISTANCE_CAP,
    }


def extract(url: str) -> dict[str, Any]:
    """Extract lexical features from a URL. Never raises.

    Args:
        url: any string. Malformed, empty, unicode and non-URL input is
            handled; the caller does not pre-validate.

    Returns:
        A dict whose keys exactly match the fields declared in
        ``spec/url_lexical_v1.json``, in that order.
    """
    features = _empty_features()

    if not isinstance(url, str):
        return features

    raw = url.strip()
    if not raw:
        return features

    had_scheme = 1 if _SCHEME_RE.match(raw) else 0
    parseable = raw if had_scheme else "http://" + raw

    try:
        parts = urlsplit(parseable)
        host = (parts.hostname or "").lower()
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return features

    if not _host_is_plausible(host):
        return features

    features["parse_ok"] = 1
    features["had_scheme"] = had_scheme
    features["scheme_is_https"] = 1 if parts.scheme == "https" else 0

    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""

    features["url_length"] = len(raw)
    features["host_length"] = len(host)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["fragment_length"] = len(fragment)

    features["num_path_segments"] = len([s for s in path.split("/") if s])
    features["num_query_params"] = len([p for p in query.split("&") if p])

    features["num_dots_host"] = host.count(".")
    features["num_hyphens_host"] = host.count("-")

    is_ip, is_ipv6 = _is_ip_host(host)
    features["host_is_ip"] = is_ip
    features["host_is_ipv6"] = is_ipv6
    features["has_port"] = 1 if port is not None else 0
    features["has_punycode"] = 1 if "xn--" in host else 0
    features["has_at_symbol"] = 1 if "@" in parts.netloc else 0
    features["has_double_slash_in_path"] = 1 if "//" in path else 0
    features["num_encoded_chars"] = len(_PCT_RE.findall(raw))

    features["host_digit_ratio"] = _digit_ratio(host)
    features["url_digit_ratio"] = _digit_ratio(raw)
    features["host_entropy"] = _shannon_entropy(host)
    features["path_entropy"] = _shannon_entropy(path)

    labels = [label for label in host.split(".") if label]
    features["longest_host_label_len"] = max((len(x) for x in labels), default=0)

    if is_ip:
        registered = ""
        subdomain = ""
        suffix = ""
        domain_label = ""
    else:
        ext = _EXTRACT(host)
        subdomain = ext.subdomain
        suffix = ext.suffix
        domain_label = ext.domain
        registered = f"{domain_label}.{suffix}" if domain_label and suffix else ""

    features["num_subdomains"] = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    features["tld"] = suffix.rsplit(".", 1)[-1] if suffix else ""
    features["tld_length"] = len(features["tld"])
    features["suspicious_tld"] = 1 if features["tld"] in set(_keywords()["suspicious_tlds"]) else 0
    features["is_shortener"] = 1 if registered in _shorteners() else 0

    haystack = f"{host}{path}{query}".lower()
    total = 0
    for category, tokens in _keywords()["categories"].items():
        hits = sum(1 for token in tokens if token in haystack)
        features[f"keyword_hits_{category}"] = hits
        total += hits
    features["keyword_hits_total"] = total

    brand_in_host = 0
    mismatch = 0
    min_distance = BRAND_DISTANCE_CAP
    if not is_ip:
        for brand in _brands()["brands"]:
            official = set(brand["official_domains"])
            for token in brand["tokens"]:
                if token in host:
                    brand_in_host = 1
                    if registered and registered not in official:
                        mismatch = 1
                if domain_label:
                    min_distance = min(min_distance, _levenshtein(domain_label, token))
    features["brand_in_host"] = brand_in_host
    features["brand_domain_mismatch"] = mismatch
    features["brand_min_distance"] = min_distance

    return features


def feature_names() -> list[str]:
    """Ordered feature names produced by :func:`extract`."""
    return list(_empty_features().keys())
