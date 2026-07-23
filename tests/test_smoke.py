"""Package-level invariants.

``__version__`` reads from installed metadata rather than a literal, so this
asserts shape and resolution, never a specific number. A test pinned to the
current version fails on every release and tests nothing: the real failure it
guards against is ``__version__`` disappearing from the package root, which is
what a wholesale ``__init__`` rewrite once did.
"""

import re

import scamvet_riskengine
from scamvet_riskengine import __version__

_SEMVER = re.compile(r"^\d+\.\d+\.\d+")


def test_version_resolves_from_metadata():
    assert _SEMVER.match(__version__), f"not a semantic version: {__version__!r}"
    assert __version__ != "0.0.0", (
        "version fell back to the source-tree default; the package is not installed "
        "or its metadata is missing"
    )


def test_public_api_is_importable():
    for name in ("load", "list_models", "resolve", "ScoreResult", "Manifest"):
        assert hasattr(scamvet_riskengine, name), f"missing from package root: {name}"
