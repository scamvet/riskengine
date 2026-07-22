#!/usr/bin/env python
"""Build the URL feature matrix from the raw corpus.

The source of record for everything under ``data/derived/``. That directory is
gitignored; this script is how it is regenerated, so any figure computed from
it is reproducible from the repository plus the licensed corpus download.

    python scripts/build_url_features.py

Outputs, into ``data/derived/``:

    url_features.parquet    encoded matrix plus url, label, type, group, split
    featurizer.json         fitted encoder state, for the model manifest
    build_report.json       every count and rate this build produced

Runtime is a few minutes, dominated by feature extraction. See LEGAL.md C4 for
the corpus licence and the attribution obligations that come with it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from scamvet_riskengine.data import assert_no_group_leakage, grouped_split, load_url_corpus
from scamvet_riskengine.features.featurize import LexicalFeaturizer, extract_frame
from scamvet_riskengine.features.url import FEATURE_SPEC_VERSION

DEFAULT_RAW = Path("data/raw/malicious_phish.csv")
DEFAULT_OUT = Path("data/derived")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tld-vocab", type=int, default=50)
    parser.add_argument(
        "--allow-any-row-count",
        action="store_true",
        help="Skip the exact row-count check. Only for sampled inputs; a silently "
        "truncated download would otherwise invalidate every published figure.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading {args.raw}")
    frame, load_report = load_url_corpus(args.raw, strict_row_count=not args.allow_any_row_count)
    print(
        f"      {load_report.n_after_dedup:,} rows after dedup "
        f"({load_report.n_duplicates_dropped:,} duplicates dropped), "
        f"positive rate {load_report.positive_rate:.4f}"
    )

    print(f"[2/5] splitting by registered domain (seed {args.seed})")
    frame, split_report = grouped_split(frame, seed=args.seed)
    assert_no_group_leakage(frame)
    for name in split_report.n_rows:
        print(
            f"      {name:<12} {split_report.n_rows[name]:>8,} rows  "
            f"{split_report.n_groups[name]:>7,} groups  "
            f"pos rate {split_report.positive_rate[name]:.4f}"
        )

    print(
        f"[3/5] extracting {FEATURE_SPEC_VERSION} features for "
        f"{len(frame):,} urls (several minutes)"
    )
    started = time.perf_counter()
    raw_features = extract_frame(frame["url"])
    elapsed = time.perf_counter() - started
    print(f"      done in {elapsed:.0f}s ({elapsed / len(frame) * 1e6:.0f} us/url)")

    unparsed = int((raw_features["parse_ok"] == 0).sum())
    print(f"      {unparsed:,} rows failed to parse ({unparsed / len(frame):.4%})")

    print("[4/5] fitting encoder on the training split only")
    is_train = (frame["split"] == "train").to_numpy()
    featurizer = LexicalFeaturizer(max_tld_vocab=args.max_tld_vocab)
    featurizer.fit(raw_features[is_train])
    matrix = featurizer.transform(raw_features)
    coverage = featurizer.report(raw_features).tld_coverage
    print(
        f"      {len(featurizer.feature_columns)} columns, "
        f"tld vocabulary {len(featurizer.tld_vocabulary)}, "
        f"coverage {coverage:.4f}"
    )

    print(f"[5/5] writing to {args.out}")
    combined = pd.concat(
        [
            frame[["url", "type", "label", "group", "split"]].reset_index(drop=True),
            matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    combined.to_parquet(args.out / "url_features.parquet", index=False)
    (args.out / "featurizer.json").write_text(
        json.dumps(featurizer.to_dict(), indent=2), encoding="utf-8"
    )
    report = {
        "spec_version": FEATURE_SPEC_VERSION,
        "seed": args.seed,
        "load": load_report.as_dict(),
        "split": split_report.as_dict(),
        "n_unparsed": unparsed,
        "n_feature_columns": len(featurizer.feature_columns),
        "tld_vocabulary_size": len(featurizer.tld_vocabulary),
        "tld_coverage": coverage,
        "extract_seconds": round(elapsed, 1),
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("      ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
