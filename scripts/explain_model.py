#!/usr/bin/env python
"""Global attributions and worked reason examples for the model card.

    python scripts/explain_model.py

Trains the registry-candidate LightGBM on train, verifies that its TreeSHAP
attributions reconcile to its own output, ranks features by mean absolute
attribution over a test sample, and prints reasons for real corpus URLs -
including a false positive and a false negative, because a model card that
only shows successes is advertising.

Outputs ``data/derived/explain_report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scamvet_riskengine.explain import (
    build_reasons,
    contributions,
    global_importance,
    render_english,
    unmapped_features,
    verify_additivity,
)
from scamvet_riskengine.models import build_calibrator
from scamvet_riskengine.models.boosted import BoostedConfig, boosted_scores, build_lightgbm

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]


def _show(
    title: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    X: np.ndarray,
    names: list[str],
    probs: np.ndarray,
    indices: list[int],
) -> list[dict[str, Any]]:
    print(f"\n{title}")
    captured = []
    for index in indices:
        url = frame["url"].iloc[index]
        display = url if len(url) <= 88 else url[:85] + "..."
        print(f"  {display}")
        print(f"    type={frame['type'].iloc[index]}  p={probs[index]:.4f}")
        reasons = build_reasons(values[index], X[index], names)
        for reason in reasons:
            arrow = "^" if reason.direction == "increases" else "v"
            print(f"    {arrow} {reason.magnitude:>4.0%}  {render_english(reason)}")
        captured.append(
            {
                "url": url,
                "type": str(frame["type"].iloc[index]),
                "probability": float(probs[index]),
                "reasons": [r.as_dict() for r in reasons],
            }
        )
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=20_000)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    names = [c for c in data.columns if c not in META_COLUMNS]
    parts = {n: data[data["split"] == n] for n in ("train", "calibration", "test")}
    X = {n: p[names].to_numpy(dtype=np.float32) for n, p in parts.items()}
    y = {n: p["label"].to_numpy() for n, p in parts.items()}
    test = parts["test"].reset_index(drop=True)

    missing = unmapped_features(names)
    if missing:
        raise SystemExit(
            f"{len(missing)} feature(s) have no reason template and could reach a user "
            f"as a raw column name: {missing[:10]}"
        )
    print(f"template coverage: all {len(names)} features mapped")

    config = BoostedConfig(seed=args.seed)
    model = build_lightgbm(config, y["train"])
    model.fit(X["train"], y["train"])

    error = verify_additivity(model, X["test"][:2000])
    print(f"attribution additivity: max reconciliation error {error:.2e}")

    calibrator = build_calibrator("platt")
    calibrator.fit(boosted_scores(model, X["calibration"]), y["calibration"])
    probs = calibrator.predict_proba(boosted_scores(model, X["test"]))

    importance = global_importance(model, X["test"], names, max_samples=args.sample)
    ranked = importance.ranked()
    print(f"\nglobal importance, mean |contribution| over {importance.n_samples:,} rows")
    for rank, (name, value) in enumerate(ranked[:25], start=1):
        print(f"  {rank:>2}. {name:<28} {value:.5f}")

    active = importance.ranked_when_active()
    print("\nimportance among rows where the feature is active (rare-but-sharp check)")
    for rank, (name, value, count) in enumerate(active[:15], start=1):
        share = count / importance.n_samples
        print(f"  {rank:>2}. {name:<28} {value:.5f}  active in {share:>6.2%} of rows")

    lookup = {name: rank for rank, (name, _) in enumerate(ranked, start=1)}
    active_lookup = {
        name: (rank, value, count) for rank, (name, value, count) in enumerate(active, start=1)
    }
    print("\nfeatures worth checking specifically:")
    for name in (
        "is_shortener",
        "brand_domain_mismatch",
        "brand_in_host",
        "brand_min_distance",
        "suspicious_tld",
        "host_is_ip",
    ):
        if name in lookup:
            a_rank, a_value, a_count = active_lookup[name]
            print(
                f"  {name:<24} overall rank {lookup[name]:>3}  {dict(ranked)[name]:.5f}"
                f"   | active rank {a_rank:>3}  {a_value:.5f}  n={a_count:,}"
            )

    values, _ = contributions(model, X["test"])
    is_phishing = (test["type"] == "phishing").to_numpy()
    is_benign = (test["type"] == "benign").to_numpy()

    examples: dict[str, Any] = {}
    examples["confident_phishing"] = _show(
        "most confidently flagged phishing",
        test,
        values,
        X["test"],
        names,
        probs,
        list(np.argsort(-np.where(is_phishing, probs, -1))[:2]),
    )
    examples["missed_phishing"] = _show(
        "worst misses - phishing scored lowest",
        test,
        values,
        X["test"],
        names,
        probs,
        list(np.argsort(np.where(is_phishing, probs, 2))[:2]),
    )
    examples["false_positives"] = _show(
        "worst false positives - benign scored highest",
        test,
        values,
        X["test"],
        names,
        probs,
        list(np.argsort(-np.where(is_benign, probs, -1))[:2]),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "explain_report.json").write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "seed": args.seed,
                "additivity_max_error": error,
                "global_importance": importance.as_dict(),
                "examples": examples,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / 'explain_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
