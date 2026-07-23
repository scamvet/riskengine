"""Turn URLs into a model-ready numeric matrix.

:func:`~scamvet_riskengine.features.url.extract` produces a dict per URL with
one categorical field, ``tld``. Models need a fixed-width numeric matrix with a
stable column order, so this module owns the encoding step.

The encoder is *fitted*, which makes it a leakage surface. The TLD vocabulary
is learned from the training split only; validation, calibration and test data
are transformed with that vocabulary and never contribute to it. Fitting on the
full corpus would let the test split influence which TLDs get their own column,
which is a small leak that inflates results and is invisible afterwards.

The fitted state is a plain dict so it serialises into the model manifest
alongside the weights. A model artifact without its encoder state cannot be
reproduced, so they travel together.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scamvet_riskengine.features.spec_loader import load_spec
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION, extract

#: Column holding the categorical TLD in the raw extract output.
CATEGORICAL_COLUMN = "tld"

#: One-hot columns for TLDs outside the learned vocabulary, and for URLs with
#: no resolvable public suffix. Kept distinct: "a TLD we have not seen" and
#: "no TLD at all" are different signals and collapsing them loses the second.
OTHER_TLD = "__other__"
NO_TLD = "__none__"

DEFAULT_MAX_TLD_VOCAB = 50


class FeaturizerError(ValueError):
    """Raised on unfitted use or on state that does not round-trip."""


@dataclass(frozen=True)
class FeaturizeReport:
    """What featurisation produced, for the model card."""

    n_rows: int
    n_columns: int
    n_tld_vocab: int
    tld_coverage: float
    spec_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_tld_vocab": self.n_tld_vocab,
            "tld_coverage": self.tld_coverage,
            "spec_version": self.spec_version,
        }


def extract_frame(urls: Iterable[str]) -> pd.DataFrame:
    """Run the extractor over many URLs into a raw feature frame.

    Columns are exactly the spec fields, ``tld`` still a string. This is the
    expensive step; callers that need the matrix more than once should keep
    the frame rather than re-extracting.
    """
    rows = [extract(u) for u in urls]
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[f["name"] for f in load_spec(FEATURE_SPEC_VERSION)["features"]]
        )
    return frame


class LexicalFeaturizer:
    """Encode raw lexical features into a numeric matrix.

    Args:
        max_tld_vocab: how many distinct TLDs get their own column when
            ``use_tld_onehot`` is on. Bounded because the tail is enormous and
            mostly single-occurrence, and an unbounded vocabulary would make
            the matrix width depend on the training sample.
        use_tld_onehot: whether to emit per-TLD columns at all. **Off by
            default.** A provenance audit measured their contribution at 0.0075
            phishing recall against a seed spread of 0.0176 - within noise -
            while costing 52 of 86 matrix columns. They also carried a
            collection artifact: ``.ca`` rows are 10.6% malicious against a
            33.2% corpus rate, because the benign upstream (ISCX-URL-2016,
            University of New Brunswick) is Canada-skewed, so the column
            encoded which crawl a row came from. Dropping them is smaller,
            faster and removes a contamination vector for no measurable loss.
            ``suspicious_tld`` and ``tld_length`` remain and carry the semantic
            part of the signal.
    """

    def __init__(
        self,
        max_tld_vocab: int = DEFAULT_MAX_TLD_VOCAB,
        use_tld_onehot: bool = False,
    ) -> None:
        self.max_tld_vocab = int(max_tld_vocab)
        self.use_tld_onehot = bool(use_tld_onehot)
        self.spec_version: str = FEATURE_SPEC_VERSION
        self.tld_vocabulary: list[str] = []
        self._numeric_columns: list[str] = []
        self._fitted = False

    # -- fitting ---------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> LexicalFeaturizer:
        """Learn the TLD vocabulary. Pass the TRAINING split only."""
        self._check_frame(frame)
        spec = load_spec(self.spec_version)
        self._numeric_columns = [
            f["name"] for f in spec["features"] if f["name"] != CATEGORICAL_COLUMN
        ]
        if not self.use_tld_onehot:
            self.tld_vocabulary = []
            self._fitted = True
            return self
        counts = Counter(value for value in frame[CATEGORICAL_COLUMN].astype(str) if value)
        # Sort by frequency then alphabetically, so ties break deterministically
        # and the same training data always yields the same column order.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.tld_vocabulary = [tld for tld, _ in ranked[: self.max_tld_vocab]]
        self._fitted = True
        return self

    # -- transforming ----------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        """Ordered output columns. Stable across calls once fitted."""
        self._require_fitted()
        if not self.use_tld_onehot:
            return list(self._numeric_columns)
        one_hot = [f"tld_is_{t}" for t in self.tld_vocabulary]
        return [*self._numeric_columns, *one_hot, f"tld_is_{OTHER_TLD}", f"tld_is_{NO_TLD}"]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode a raw feature frame into the numeric matrix."""
        self._require_fitted()
        self._check_frame(frame)

        numeric = frame[self._numeric_columns].astype("float32")
        if not self.use_tld_onehot:
            return numeric.reset_index(drop=True)[self.feature_columns]

        tlds = frame[CATEGORICAL_COLUMN].astype(str)
        known = set(self.tld_vocabulary)
        encoded: dict[str, np.ndarray] = {}
        for tld in self.tld_vocabulary:
            encoded[f"tld_is_{tld}"] = (tlds == tld).to_numpy(dtype="float32")
        encoded[f"tld_is_{OTHER_TLD}"] = ((tlds != "") & (~tlds.isin(known))).to_numpy(
            dtype="float32"
        )
        encoded[f"tld_is_{NO_TLD}"] = (tlds == "").to_numpy(dtype="float32")

        out = pd.concat([numeric.reset_index(drop=True), pd.DataFrame(encoded)], axis=1)
        return out[self.feature_columns]

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def report(self, frame: pd.DataFrame) -> FeaturizeReport:
        """Coverage statistics for a frame under the fitted vocabulary."""
        self._require_fitted()
        tlds = frame[CATEGORICAL_COLUMN].astype(str)
        covered = (
            float(tlds.isin(set(self.tld_vocabulary)).mean())
            if len(tlds) and self.use_tld_onehot
            else 0.0
        )
        return FeaturizeReport(
            n_rows=len(frame),
            n_columns=len(self.feature_columns),
            n_tld_vocab=len(self.tld_vocabulary),
            tld_coverage=float(covered),
            spec_version=self.spec_version,
        )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Fitted state, for embedding in a model manifest."""
        self._require_fitted()
        return {
            "class": "LexicalFeaturizer",
            "spec_version": self.spec_version,
            "max_tld_vocab": self.max_tld_vocab,
            "use_tld_onehot": self.use_tld_onehot,
            "tld_vocabulary": list(self.tld_vocabulary),
            "numeric_columns": list(self._numeric_columns),
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> LexicalFeaturizer:
        """Rebuild a fitted featurizer from manifest state."""
        if state.get("class") != "LexicalFeaturizer":
            raise FeaturizerError(f"unexpected state class {state.get('class')!r}")
        if state.get("spec_version") != FEATURE_SPEC_VERSION:
            raise FeaturizerError(
                f"state was fitted under spec {state.get('spec_version')!r} but this "
                f"build produces {FEATURE_SPEC_VERSION!r}; feature values would not match"
            )
        instance = cls(
            max_tld_vocab=int(state["max_tld_vocab"]),
            use_tld_onehot=bool(state.get("use_tld_onehot", True)),
        )
        instance.tld_vocabulary = list(state["tld_vocabulary"])
        instance._numeric_columns = list(state["numeric_columns"])
        instance._fitted = True
        if instance.feature_columns != list(state["feature_columns"]):
            raise FeaturizerError("rebuilt column order does not match stored state")
        return instance

    # -- internals -------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise FeaturizerError("featurizer is not fitted; call fit() on the training split")

    @staticmethod
    def _check_frame(frame: pd.DataFrame) -> None:
        if CATEGORICAL_COLUMN not in frame.columns:
            raise FeaturizerError(f"frame is missing the {CATEGORICAL_COLUMN!r} column")
