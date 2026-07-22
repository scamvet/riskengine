"""Tests for corpus loading and leakage-free splitting.

The leakage tests here are the ones that matter. Everything downstream —
benchmark tables, calibration curves, the fairness report, the published
figures — is only trustworthy if no registered domain spans two splits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scamvet_riskengine.data import (
    DEFAULT_RATIOS,
    CorpusError,
    SplitError,
    assert_no_group_leakage,
    assign_split,
    group_fraction,
    grouped_split,
    load_url_corpus,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "url_corpus_sample.csv"


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    frame, _ = load_url_corpus(FIXTURE)
    return frame


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_loads_expected_columns(corpus: pd.DataFrame) -> None:
    assert set(corpus.columns) == {"url", "type", "label", "group"}


def test_exact_duplicate_urls_are_dropped() -> None:
    frame, report = load_url_corpus(FIXTURE)
    assert report.n_duplicates_dropped >= 1
    assert frame["url"].is_unique, "duplicates would straddle splits"


def test_binary_label_matches_type(corpus: pd.DataFrame) -> None:
    assert (corpus.loc[corpus["type"] == "benign", "label"] == 0).all()
    assert (corpus.loc[corpus["type"] != "benign", "label"] == 1).all()


def test_defacement_counts_as_positive(corpus: pd.DataFrame) -> None:
    defaced = corpus[corpus["type"] == "defacement"]
    assert len(defaced) > 0
    assert (defaced["label"] == 1).all()


def test_group_is_registered_domain(corpus: pd.DataFrame) -> None:
    lookup = dict(zip(corpus["url"], corpus["group"], strict=False))
    assert lookup["https://www.sbi.co.in/web/personal-banking"] == "sbi.co.in"
    assert lookup["https://retail.sbi.co.in/login"] == "sbi.co.in"
    assert lookup["http://hdfcbank.com.secure-login.top/otp"] == "secure-login.top"


def test_ip_hosts_get_empty_group(corpus: pd.DataFrame) -> None:
    ip_rows = corpus[corpus["url"].str.contains("182.127.6.238", regex=False)]
    assert len(ip_rows) == 1
    assert ip_rows.iloc[0]["group"] == ""


def test_missing_file_raises() -> None:
    with pytest.raises(CorpusError):
        load_url_corpus("/nonexistent/malicious_phish.csv")


def test_strict_row_count_rejects_a_truncated_file() -> None:
    with pytest.raises(CorpusError, match="expected"):
        load_url_corpus(FIXTURE, strict_row_count=True)


def test_unexpected_type_value_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("url,type\nhttps://a.com/,spamalot\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="unexpected type"):
        load_url_corpus(bad)


def test_report_numbers_are_consistent() -> None:
    frame, report = load_url_corpus(FIXTURE)
    assert report.n_after_dedup == len(frame)
    assert report.n_raw - report.n_duplicates_dropped >= report.n_after_dedup
    assert sum(report.class_counts.values()) == len(frame)
    assert 0.0 <= report.positive_rate <= 1.0


# --------------------------------------------------------------------------
# Splitting — the invariant
# --------------------------------------------------------------------------


def test_no_group_spans_two_splits(corpus: pd.DataFrame) -> None:
    split, _ = grouped_split(corpus)
    assert_no_group_leakage(split)


def test_leakage_check_actually_catches_a_leak(corpus: pd.DataFrame) -> None:
    split, _ = grouped_split(corpus)
    tampered = split.copy()
    target = tampered[tampered["group"] == "sbi.co.in"].index
    assert len(target) >= 2
    tampered.loc[target[0], "split"] = "train"
    tampered.loc[target[1], "split"] = "test"
    with pytest.raises(SplitError, match="more than one split"):
        assert_no_group_leakage(tampered)


def test_every_row_is_assigned(corpus: pd.DataFrame) -> None:
    split, report = grouped_split(corpus)
    assert split["split"].notna().all()
    assert set(split["split"].unique()) <= set(DEFAULT_RATIOS)
    assert sum(report.n_rows.values()) == len(split)


def test_split_is_reproducible(corpus: pd.DataFrame) -> None:
    first, _ = grouped_split(corpus, seed=42)
    second, _ = grouped_split(corpus, seed=42)
    assert first["split"].tolist() == second["split"].tolist()


def test_different_seed_gives_a_different_split(corpus: pd.DataFrame) -> None:
    first, _ = grouped_split(corpus, seed=42)
    second, _ = grouped_split(corpus, seed=7)
    assert first["split"].tolist() != second["split"].tolist()


def test_split_is_independent_of_row_order(corpus: pd.DataFrame) -> None:
    shuffled = corpus.sample(frac=1.0, random_state=3).reset_index(drop=True)
    ordered, _ = grouped_split(corpus)
    reordered, _ = grouped_split(shuffled)
    a = dict(zip(ordered["url"], ordered["split"], strict=False))
    b = dict(zip(reordered["url"], reordered["split"], strict=False))
    assert a == b


def test_existing_groups_keep_assignment_when_corpus_grows(
    corpus: pd.DataFrame,
) -> None:
    """The property a shuffled split cannot offer.

    Without it, retraining on a grown corpus silently moves last release's
    test domains into training, and the two releases' metrics stop being
    comparable.
    """
    before, _ = grouped_split(corpus)
    extra = pd.DataFrame(
        {
            "url": [f"https://brandnew{i}.example.net/x" for i in range(25)],
            "type": ["phishing"] * 25,
            "label": [1] * 25,
            "group": [f"brandnew{i}.example.net" for i in range(25)],
        }
    )
    grown = pd.concat([corpus, extra], ignore_index=True)
    after, _ = grouped_split(grown)
    original = after.iloc[: len(corpus)]
    assert original["split"].tolist() == before["split"].tolist()


def test_ungrouped_rows_are_counted_not_hidden(corpus: pd.DataFrame) -> None:
    _, report = grouped_split(corpus)
    assert report.n_ungrouped_rows >= 2, "IP-host rows exist in the fixture"


def test_ratios_must_sum_to_one(corpus: pd.DataFrame) -> None:
    with pytest.raises(SplitError, match="sum to 1.0"):
        grouped_split(corpus, ratios={"train": 0.5, "test": 0.2})


def test_missing_column_raises() -> None:
    with pytest.raises(SplitError, match="missing required column"):
        grouped_split(pd.DataFrame({"url": ["https://a.com/"]}))


def test_custom_ratios_are_honoured() -> None:
    groups = [f"domain{i}.com" for i in range(4000)]
    frame = pd.DataFrame({"url": [f"https://{g}/x" for g in groups], "group": groups, "label": 0})
    _, report = grouped_split(frame, ratios={"train": 0.8, "test": 0.2})
    share = report.n_rows["train"] / sum(report.n_rows.values())
    assert 0.77 < share < 0.83, f"train share {share} far from 0.8"


# --------------------------------------------------------------------------
# Hash assignment properties
# --------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(st.text(min_size=1, max_size=60), st.integers(0, 10_000))
def test_group_fraction_is_in_unit_interval(group: str, seed: int) -> None:
    assert 0.0 <= group_fraction(group, seed) < 1.0


@settings(max_examples=300, deadline=None)
@given(st.text(min_size=1, max_size=60))
def test_assignment_is_stable_for_a_fixed_seed(group: str) -> None:
    assert assign_split(group, 42) == assign_split(group, 42)
    assert assign_split(group, 42) in DEFAULT_RATIOS
