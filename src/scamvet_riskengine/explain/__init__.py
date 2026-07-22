"""Explainability: exact TreeSHAP attributions and the reasons[] contract."""

from scamvet_riskengine.explain.contributions import (
    ExplainError,
    GlobalImportance,
    contributions,
    global_importance,
    verify_additivity,
)
from scamvet_riskengine.explain.reasons import (
    DEFAULT_TOP_K,
    MIN_MAGNITUDE,
    TEMPLATE_VERSION,
    Reason,
    ReasonError,
    build_reasons,
    explain_rows,
    load_templates,
    render_english,
    unmapped_features,
)

__all__ = [
    "DEFAULT_TOP_K",
    "MIN_MAGNITUDE",
    "TEMPLATE_VERSION",
    "ExplainError",
    "GlobalImportance",
    "Reason",
    "ReasonError",
    "build_reasons",
    "contributions",
    "explain_rows",
    "global_importance",
    "load_templates",
    "render_english",
    "unmapped_features",
    "verify_additivity",
]
