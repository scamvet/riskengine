"""Feature extraction for RiskEngine.

Extractors here are pure functions over their inputs: no network, no clocks,
no randomness. Anything requiring enrichment lives in a separate module so
that the offline path can never silently depend on it.
"""

from scamvet_riskengine.features.featurize import (
    FeaturizerError,
    FeaturizeReport,
    LexicalFeaturizer,
    extract_frame,
)
from scamvet_riskengine.features.spec_loader import (
    SpecViolation,
    load_spec,
    spec_feature_names,
    validate,
)
from scamvet_riskengine.features.url import (
    BRAND_DISTANCE_CAP,
    FEATURE_SPEC_VERSION,
    extract,
    feature_names,
)

__all__ = [
    "BRAND_DISTANCE_CAP",
    "FeaturizeReport",
    "FeaturizerError",
    "LexicalFeaturizer",
    "extract_frame",
    "FEATURE_SPEC_VERSION",
    "SpecViolation",
    "extract",
    "feature_names",
    "load_spec",
    "spec_feature_names",
    "validate",
]
