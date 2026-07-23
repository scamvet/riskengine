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

FEATURE_SPEC_VERSION = "url_lexical_v5"

_DATA_DIR = Path(__file__).parent / "data"

# Distance sentinel: returned when no brand comparison is possible (e.g. the
# host is a bare IP). Chosen above any realistic label length so that "no
# comparison" sorts as maximally dissimilar rather than as a near match.
BRAND_DISTANCE_CAP = 64

# Edit distances at or above this are reported as BRAND_DISTANCE_CAP. A domain
# four or more edits away from a brand token is not a typosquat of it, so the
# exact value carries no signal - and computing it dominated extraction cost
# (93% of runtime, ~37 full distance computations per URL). Pruning against
# this bound instead of the cap made a full-corpus build roughly ten times
# faster. The change in reported values is why the spec moved to v2.
BRAND_DISTANCE_USEFUL_MAX = 4

# Changed in v5: url_length and url_digit_ratio are measured on a canonical
# form with the scheme stripped, not on the raw input.
#
# This was the same corpus-provenance bug as v3 and v4, hiding where feature
# removal could not reach it. 72% of corpus rows carry no scheme, so the model
# learned url_length - its top feature by attribution - on scheme-less strings,
# while every real user pastes a URL with one. Eight extra characters on a
# median length of 45 moved the score enormously: bankofbaroda.com/ scored
# 0.018 and https://bankofbaroda.com/ scored 0.985, the same site either side
# of the red threshold. 36 of 55 domains our own brand list asserts are
# legitimate were flagged, against 0.5% of comparable corpus rows.
#
# Any feature computed over the raw input string inherits how that string
# happened to be stored. Canonicalising first is the general fix; the invariant
# is that extract(u) == extract("https://" + u), and it is a property test.
#
# Removed in v4: scheme_is_https. Only 2.4% of the corpus carries an explicit
# https prefix, and those rows are 87.5% malicious - benign rows are 0.46%
# https, defacement 0.00%. That is a storage convention of the upstream files,
# not a property of the sites. Worse than a merely useless feature: in
# production nearly every legitimate site is https, so a model taught
# "https implies malicious" would flag the legitimate web. This corpus cannot
# teach the correct relationship, so the feature is removed rather than kept
# and hoped over.
#
# Removed in v3: had_scheme. Whether the stored string carried "http://" turned
# out to identify which upstream file a corpus row came from - defacement rows
# were 100% scheme-prefixed, benign rows 8% - so the model learned provenance
# rather than phishing. It was the single strongest feature by attribution,
# 3.3x the next. At inference the same field means only "how the user happened
# to copy the link", which carries no information about the destination. A
# feature that cannot mean the same thing at inference as in training does not
# belong in the spec, so it is gone rather than zeroed.

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


@lru_cache(maxsize=1)
def _suspicious_tlds() -> frozenset[str]:
    return frozenset(_keywords()["suspicious_tlds"])


@lru_cache(maxsize=1)
def _brand_index() -> tuple[tuple[str, frozenset[str]], ...]:
    """Flatten brands to (token, official_domains) pairs once.

    The per-URL path walked a nested structure and rebuilt the same sets on
    every call. Flattening at import turns that into a tight loop over a
    tuple.
    """
    flat: list[tuple[str, frozenset[str]]] = []
    for brand in _brands()["brands"]:
        official = frozenset(brand["official_domains"])
        for token in brand["tokens"]:
            flat.append((token, official))
    return tuple(flat)


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


@lru_cache(maxsize=262_144)
def _host_analysis(host: str) -> tuple[int, int, int, str, str, int, int, int]:
    """Everything derivable from the host alone, cached by host.

    Returns:
        (is_ip, is_ipv6, n_subdomain_labels, suffix, registered_domain,
         brand_in_host, brand_domain_mismatch, brand_min_distance)

    Cached because a corpus of URLs contains far fewer distinct hosts than
    URLs, and public-suffix resolution plus the brand scan are by a wide
    margin the most expensive part of extraction. Recomputing them per path
    on the same host is pure waste.
    """
    is_ip, is_ipv6 = _is_ip_host(host)
    if is_ip:
        return (1, is_ipv6, 0, "", "", 0, 0, BRAND_DISTANCE_CAP)

    ext = _EXTRACT(host)
    domain_label = ext.domain
    suffix = ext.suffix
    registered = f"{domain_label}.{suffix}" if domain_label and suffix else ""
    n_subdomains = len([s for s in ext.subdomain.split(".") if s])

    brand_in_host = 0
    mismatch = 0
    best = BRAND_DISTANCE_USEFUL_MAX
    label_len = len(domain_label)
    for token, official in _brand_index():
        if token in host:
            brand_in_host = 1
            if registered and registered not in official:
                mismatch = 1
        if domain_label:
            # Edit distance is at least the length difference, so a token that
            # far from the label cannot beat the current best. Exact pruning.
            if abs(label_len - len(token)) >= best:
                continue
            distance = _levenshtein(domain_label, token, cap=best)
            if distance < best:
                best = distance

    reported = best if best < BRAND_DISTANCE_USEFUL_MAX else BRAND_DISTANCE_CAP
    return (0, 0, n_subdomains, suffix, registered, brand_in_host, mismatch, reported)


def _empty_features() -> dict[str, Any]:
    """Neutral feature dict used when the input cannot be parsed at all."""
    return {
        "parse_ok": 0,
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

    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""

    # Canonical form: what the address *is*, independent of how it was written
    # down. Everything measured over the whole URL uses this rather than `raw`.
    canonical = f"{host}{path}"
    if query:
        canonical += f"?{query}"
    if fragment:
        canonical += f"#{fragment}"
    if parts.port is not None:
        canonical = f"{host}:{parts.port}{canonical[len(host) :]}"

    features["url_length"] = len(canonical)
    features["host_length"] = len(host)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["fragment_length"] = len(fragment)

    features["num_path_segments"] = len([s for s in path.split("/") if s])
    features["num_query_params"] = len([p for p in query.split("&") if p])

    features["num_dots_host"] = host.count(".")
    features["num_hyphens_host"] = host.count("-")

    analysis = _host_analysis(host)
    features["host_is_ip"] = analysis[0]
    features["host_is_ipv6"] = analysis[1]
    features["has_port"] = 1 if port is not None else 0
    features["has_punycode"] = 1 if "xn--" in host else 0
    features["has_at_symbol"] = 1 if "@" in parts.netloc else 0
    features["has_double_slash_in_path"] = 1 if "//" in path else 0
    features["num_encoded_chars"] = len(_PCT_RE.findall(canonical))

    features["host_digit_ratio"] = _digit_ratio(host)
    features["url_digit_ratio"] = _digit_ratio(canonical)
    features["host_entropy"] = _shannon_entropy(host)
    features["path_entropy"] = _shannon_entropy(path)

    labels = [label for label in host.split(".") if label]
    features["longest_host_label_len"] = max((len(x) for x in labels), default=0)

    (_, _, n_subdomains, suffix, registered, brand_in_host, mismatch, min_distance) = analysis

    features["num_subdomains"] = n_subdomains
    features["tld"] = suffix.rsplit(".", 1)[-1] if suffix else ""
    features["tld_length"] = len(features["tld"])
    features["suspicious_tld"] = 1 if features["tld"] in _suspicious_tlds() else 0
    features["is_shortener"] = 1 if registered in _shorteners() else 0

    haystack = f"{host}{path}{query}".lower()
    total = 0
    for category, tokens in _keywords()["categories"].items():
        hits = sum(1 for token in tokens if token in haystack)
        features[f"keyword_hits_{category}"] = hits
        total += hits
    features["keyword_hits_total"] = total

    features["brand_in_host"] = brand_in_host
    features["brand_domain_mismatch"] = mismatch
    features["brand_min_distance"] = min_distance

    return features


def feature_names() -> list[str]:
    """Ordered feature names produced by :func:`extract`."""
    return list(_empty_features().keys())


def registered_domain(url: str) -> str:
    """Return the registered domain (eTLD+1) of a URL, or "" if there is none.

    Public because it is the grouping key for dataset splitting. Random
    splitting of URLs puts ``evil.com/a`` in train and ``evil.com/b`` in test,
    which inflates every reported metric; grouping by registered domain is the
    fix, and the grouping must use exactly the same public-suffix resolution
    the features use, so it lives here rather than being reimplemented.

    Returns "" for IP hosts and for anything unparseable, so callers must
    handle the empty group explicitly rather than silently pooling them.
    """
    if not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    parseable = raw if _SCHEME_RE.match(raw) else "http://" + raw
    try:
        host = (urlsplit(parseable).hostname or "").lower()
    except ValueError:
        return ""
    if not _host_is_plausible(host):
        return ""
    if _is_ip_host(host)[0]:
        return ""
    ext = _EXTRACT(host)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}"
