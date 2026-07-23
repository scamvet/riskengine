"""ScamVet RiskEngine: calibrated fraud and phishing scoring.

    from scamvet_riskengine import load

    scorer = load("url-scorer-lexical")
    result = scorer.score("http://sbi-kyc-verify.xyz/login/update")
    print(result.band, result.probability)
    for reason in result.reasons:
        print(reason.template_key, reason.direction, reason.magnitude)

The registry is the supported entry point. Everything under ``features``,
``models``, ``eval`` and ``explain`` is the machinery that builds what the
registry serves, and is public so the benchmarks are reproducible rather than
because consumers are expected to assemble a scorer by hand.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from scamvet_riskengine.registry import (
    Manifest,
    ManifestError,
    ScoreResult,
    Scorer,
    ScorerError,
    Thresholds,
    list_models,
    load,
    resolve,
)

try:
    __version__ = _version("scamvet-riskengine")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Manifest",
    "ManifestError",
    "ScoreResult",
    "Scorer",
    "ScorerError",
    "Thresholds",
    "__version__",
    "list_models",
    "load",
    "resolve",
]
