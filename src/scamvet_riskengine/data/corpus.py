"""Loader for the malicious-URL corpus.

Source: "Malicious URLs dataset" (651,191 rows), CC0 1.0 Public Domain.
See ``LEGAL.md`` C4 for the licence, the five upstream sources, and the
attribution obligations. This is the only corpus permitted in the product
track (ADR-007).

The file is not vendored: it is ~46 MB and derived data is gitignored, with
this module as the source of record. Fetch it once to ``data/raw/``:

    kaggle datasets download -d sid321axn/malicious-urls-dataset -p data/raw --unzip

or download ``malicious_phish.csv`` manually from the dataset page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scamvet_riskengine.features.url import registered_domain

#: Raw ``type`` values in the source file.
BENIGN_TYPE = "benign"
MALICIOUS_TYPES = ("phishing", "malware", "defacement")

#: Expected row count of the pristine source file, used as a provenance check.
EXPECTED_RAW_ROWS = 651_191


class CorpusError(ValueError):
    """Raised when the corpus file is not shaped the way this loader expects."""


@dataclass(frozen=True)
class LoadReport:
    """What loading actually did, so the numbers in a model card are earned."""

    n_raw: int
    n_after_dedup: int
    n_duplicates_dropped: int
    n_empty_group: int
    class_counts: dict[str, int]
    positive_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_raw": self.n_raw,
            "n_after_dedup": self.n_after_dedup,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "n_empty_group": self.n_empty_group,
            "class_counts": self.class_counts,
            "positive_rate": self.positive_rate,
        }


def load_url_corpus(
    path: str | Path,
    *,
    strict_row_count: bool = False,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load, normalise, deduplicate and group the URL corpus.

    Args:
        path: path to ``malicious_phish.csv``.
        strict_row_count: if True, raise unless the file has exactly
            :data:`EXPECTED_RAW_ROWS` rows. Off by default so that fixtures
            and samples work; turn it on in the corpus build script, where a
            silently truncated download is a real hazard.

    Returns:
        (frame, report). The frame has columns:

        ``url``     normalised URL string
        ``type``    original multiclass label, kept for per-class reporting
        ``label``   binary target: 0 benign, 1 phishing/malware/defacement
        ``group``   registered domain (eTLD+1), "" for IP hosts and unparseables

    Binary target rationale: ScamVet's consumer question is "is it safe to
    click", so defacement joins phishing and malware on the positive side.
    ``type`` is retained because per-class recall must be reported separately;
    a model that finds every defacement and no phishing would be useless here
    and a single aggregate number would hide that.
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"corpus file not found: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = {"url", "type"} - set(frame.columns)
    if missing:
        raise CorpusError(f"corpus is missing column(s): {sorted(missing)}")

    n_raw = len(frame)
    if strict_row_count and n_raw != EXPECTED_RAW_ROWS:
        raise CorpusError(
            f"expected {EXPECTED_RAW_ROWS} rows, found {n_raw}. "
            "A truncated or re-versioned download invalidates published figures."
        )

    frame["url"] = frame["url"].astype(str).str.strip()
    frame["type"] = frame["type"].astype(str).str.strip().str.lower()

    frame = frame[frame["url"] != ""]

    unknown = set(frame["type"].unique()) - {BENIGN_TYPE, *MALICIOUS_TYPES}
    if unknown:
        raise CorpusError(f"unexpected type value(s): {sorted(unknown)}")

    before = len(frame)
    frame = frame.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    duplicates_dropped = before - len(frame)

    frame["label"] = (frame["type"] != BENIGN_TYPE).astype("int8")
    frame["group"] = frame["url"].map(registered_domain)

    report = LoadReport(
        n_raw=n_raw,
        n_after_dedup=len(frame),
        n_duplicates_dropped=duplicates_dropped,
        n_empty_group=int((frame["group"] == "").sum()),
        class_counts={str(k): int(v) for k, v in frame["type"].value_counts().items()},
        positive_rate=float(frame["label"].mean()) if len(frame) else 0.0,
    )
    return frame, report
