"""Tests for the feature matrix builder.

The encoder is the only fitted object between raw URLs and the model, which
makes it the only place a train/test leak can hide on this path. Most of these
tests exist to close that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scamvet_riskengine.features.featurize import (
    NO_TLD,
    OTHER_TLD,
    FeaturizerError,
    LexicalFeaturizer,
    extract_frame,
)
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION

TRAIN_URLS = [
    "https://a.com/1",
    "https://b.com/2",
    "https://c.com/3",
    "https://d.in/4",
    "https://e.in/5",
    "https://f.org/6",
    "http://182.127.6.238/x",
    "not a url",
]
UNSEEN_URLS = ["https://g.xyz/7", "https://h.top/8"]


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    return extract_frame(TRAIN_URLS)


@pytest.fixture(scope="module")
def unseen_frame() -> pd.DataFrame:
    return extract_frame(UNSEEN_URLS)


def test_extract_frame_has_spec_columns(train_frame: pd.DataFrame) -> None:
    assert "tld" in train_frame.columns
    assert len(train_frame) == len(TRAIN_URLS)


def test_extract_frame_handles_empty_input() -> None:
    assert len(extract_frame([])) == 0


# --------------------------------------------------------------------------
# Leakage surface
# --------------------------------------------------------------------------


def test_vocabulary_comes_only_from_fit_data(train_frame: pd.DataFrame) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    assert "xyz" not in f.tld_vocabulary
    assert "top" not in f.tld_vocabulary
    assert set(f.tld_vocabulary) <= {"com", "in", "org"}


def test_unseen_tld_goes_to_other_not_its_own_column(
    train_frame: pd.DataFrame, unseen_frame: pd.DataFrame
) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    matrix = f.transform(unseen_frame)
    assert f"tld_is_{OTHER_TLD}" in matrix.columns
    assert not any(c.endswith("_xyz") for c in matrix.columns)
    assert (matrix[f"tld_is_{OTHER_TLD}"] == 1.0).all()


def test_no_tld_is_distinct_from_unknown_tld(train_frame: pd.DataFrame) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    matrix = f.transform(train_frame)
    ip_row = matrix.iloc[TRAIN_URLS.index("http://182.127.6.238/x")]
    assert ip_row[f"tld_is_{NO_TLD}"] == 1.0
    assert ip_row[f"tld_is_{OTHER_TLD}"] == 0.0


def test_transform_column_set_is_identical_across_frames(
    train_frame: pd.DataFrame, unseen_frame: pd.DataFrame
) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    assert list(f.transform(train_frame).columns) == list(f.transform(unseen_frame).columns)


# --------------------------------------------------------------------------
# Matrix shape and content
# --------------------------------------------------------------------------


def test_matrix_is_all_float32_and_finite(train_frame: pd.DataFrame) -> None:
    matrix = LexicalFeaturizer().fit_transform(train_frame)
    assert (matrix.dtypes == np.float32).all()
    assert np.isfinite(matrix.to_numpy()).all()


def test_no_categorical_column_survives(train_frame: pd.DataFrame) -> None:
    matrix = LexicalFeaturizer().fit_transform(train_frame)
    assert "tld" not in matrix.columns


def test_one_hot_rows_sum_to_exactly_one(train_frame: pd.DataFrame) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    matrix = f.transform(train_frame)
    one_hot = [c for c in matrix.columns if c.startswith("tld_is_")]
    assert (matrix[one_hot].sum(axis=1) == 1.0).all(), "TLD encoding must be exhaustive"


def test_vocabulary_is_bounded(train_frame: pd.DataFrame) -> None:
    f = LexicalFeaturizer(max_tld_vocab=2).fit(train_frame)
    assert len(f.tld_vocabulary) == 2
    assert len(f.feature_columns) == 34 + 2 + 2  # numeric + vocab + other + none


def test_column_order_is_deterministic(train_frame: pd.DataFrame) -> None:
    first = LexicalFeaturizer().fit(train_frame).feature_columns
    shuffled = train_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    second = LexicalFeaturizer().fit(shuffled).feature_columns
    assert first == second, "row order must not change the matrix layout"


# --------------------------------------------------------------------------
# Serialisation — a model without its encoder cannot be reproduced
# --------------------------------------------------------------------------


def test_state_round_trips(train_frame: pd.DataFrame, unseen_frame: pd.DataFrame) -> None:
    original = LexicalFeaturizer().fit(train_frame)
    restored = LexicalFeaturizer.from_dict(original.to_dict())
    assert restored.feature_columns == original.feature_columns
    pd.testing.assert_frame_equal(
        restored.transform(unseen_frame), original.transform(unseen_frame)
    )


def test_state_from_a_different_spec_version_is_rejected(train_frame: pd.DataFrame) -> None:
    state = LexicalFeaturizer().fit(train_frame).to_dict()
    state["spec_version"] = "url_lexical_v1"
    with pytest.raises(FeaturizerError, match="spec"):
        LexicalFeaturizer.from_dict(state)


def test_tampered_column_order_is_rejected(train_frame: pd.DataFrame) -> None:
    state = LexicalFeaturizer().fit(train_frame).to_dict()
    state["feature_columns"] = list(reversed(state["feature_columns"]))
    with pytest.raises(FeaturizerError, match="column order"):
        LexicalFeaturizer.from_dict(state)


def test_state_records_the_spec_version(train_frame: pd.DataFrame) -> None:
    assert LexicalFeaturizer().fit(train_frame).to_dict()["spec_version"] == FEATURE_SPEC_VERSION


# --------------------------------------------------------------------------
# Misuse
# --------------------------------------------------------------------------


def test_transform_before_fit_raises(train_frame: pd.DataFrame) -> None:
    with pytest.raises(FeaturizerError, match="not fitted"):
        LexicalFeaturizer().transform(train_frame)


def test_frame_without_tld_column_raises() -> None:
    with pytest.raises(FeaturizerError, match="tld"):
        LexicalFeaturizer().fit(pd.DataFrame({"url_length": [1, 2]}))


def test_report_gives_coverage(train_frame: pd.DataFrame, unseen_frame: pd.DataFrame) -> None:
    f = LexicalFeaturizer().fit(train_frame)
    assert f.report(train_frame).tld_coverage > 0.5
    assert f.report(unseen_frame).tld_coverage == 0.0
