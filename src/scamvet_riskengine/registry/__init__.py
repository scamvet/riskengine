"""The model registry: the public surface Core imports.

    from scamvet_riskengine import load
    scorer = load("url-scorer-lexical")
    result = scorer.score("http://sbi-kyc-verify.xyz/login")

Layout on disk, one directory per version::

    models/
      url-scorer-lexical/
        0.1.0/
          manifest.json
          model.onnx
          MODEL_CARD.md

The registry root defaults to the ``models/`` directory shipped inside the
package, and can be pointed elsewhere with ``SCAMVET_MODEL_REGISTRY`` for
deployments that mount models separately from code.
"""

from __future__ import annotations

import os
from pathlib import Path

from scamvet_riskengine.registry.manifest import (
    Manifest,
    ManifestError,
    Thresholds,
    parse_version,
    satisfies,
)
from scamvet_riskengine.registry.scorer import (
    BOUNDARY_MARGIN,
    DEFAULT_USE_CASE,
    ScoreResult,
    Scorer,
    ScorerError,
    build_scorer,
)

ENV_REGISTRY_ROOT = "SCAMVET_MODEL_REGISTRY"

__all__ = [
    "BOUNDARY_MARGIN",
    "DEFAULT_USE_CASE",
    "ENV_REGISTRY_ROOT",
    "Manifest",
    "ManifestError",
    "ScoreResult",
    "Scorer",
    "ScorerError",
    "Thresholds",
    "build_scorer",
    "list_models",
    "load",
    "parse_version",
    "registry_root",
    "resolve",
    "satisfies",
]


def registry_root(root: Path | str | None = None) -> Path:
    """Where models live: explicit argument, then env var, then packaged default."""
    if root is not None:
        return Path(root)
    from_env = os.environ.get(ENV_REGISTRY_ROOT)
    if from_env:
        return Path(from_env)
    return Path(__file__).parent.parent / "models"


def list_models(root: Path | str | None = None) -> list[Manifest]:
    """Every manifest in the registry, newest version first within each name."""
    base = registry_root(root)
    if not base.is_dir():
        return []
    manifests = []
    for manifest_path in sorted(base.glob("*/*/manifest.json")):
        try:
            manifests.append(Manifest.load(manifest_path))
        except ManifestError:
            # A malformed directory should not make the whole registry
            # unreadable; resolve() reports precisely when a name is asked for.
            continue
    manifests.sort(key=lambda m: (m.name, parse_version(m.version)), reverse=True)
    return manifests


def resolve(
    name: str, version: str | None = None, root: Path | str | None = None
) -> tuple[Manifest, Path]:
    """Find the newest model matching ``name`` and an optional version specifier."""
    base = registry_root(root)
    directory = base / name
    if not directory.is_dir():
        available = sorted({m.name for m in list_models(root)})
        raise ManifestError(f"no model named {name!r} in {base}. Available: {available or 'none'}")

    candidates = []
    for manifest_path in directory.glob("*/manifest.json"):
        manifest = Manifest.load(manifest_path)
        if satisfies(manifest.version, version):
            candidates.append((manifest, manifest_path.parent))
    if not candidates:
        present = sorted(p.name for p in directory.iterdir() if p.is_dir())
        raise ManifestError(f"no version of {name!r} satisfies {version!r}; present: {present}")
    candidates.sort(key=lambda item: parse_version(item[0].version), reverse=True)
    return candidates[0]


def load(name: str, version: str | None = None, root: Path | str | None = None) -> Scorer:
    """Load a product-track scorer by name and optional version specifier.

    Args:
        name: e.g. ``url-scorer-lexical``.
        version: ``"0.1.0"``, ``">=0.1.0"``, ``"~=0.1.0"``, ``"^0.1.0"``, or
            None for the newest available.
        root: registry root override.

    Raises:
        ManifestError: unknown name, no matching version, or a research-track
            model, which must never be served to users.
        ScorerError: artifact missing, or built against a different feature
            spec than this package produces.
    """
    _, directory = resolve(name, version, root)
    return build_scorer(directory)
