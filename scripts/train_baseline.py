#!/usr/bin/env python
"""Train and evaluate the calibrated logistic baseline.

Reads ``data/derived/url_features.parquet`` (produced by
``scripts/build_url_features.py``) and writes the numbers that go into the
model card.

    python scripts/train_baseline.py

The split discipline this script enforces:

* the model fits on ``train`` only;
* the calibrator fits on ``calibration`` only, never on training scores;
* every reported figure comes from ``test``, which neither has touched.

It also reports recall on the feature slices this corpus barely contains
(``brand_domain_mismatch``, ``.in`` domains, suspicious TLDs). Those slices are
a fraction of a percent of rows, so an aggregate metric cannot move whether
they work or not — and they are the patterns most likely to matter in
production. Reporting them separately is the only way the gap stays visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scamvet_riskengine.eval import metrics as m
from scamvet_riskengine.models import (
    BaselineConfig,
    build_baseline,
    build_calibrator,
    coefficient_table,
    decision_scores,
)

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]

#: Slices reported separately because they are rare here and common in production.
SLICES = {
    "brand_domain_mismatch": lambda d: d["brand_domain_mismatch"] == 1.0,
    "suspicious_tld": lambda d: d["suspicious_tld"] == 1.0,
    "tld_in": lambda d: d.get("tld_is_in", pd.Series(0.0, index=d.index)) == 1.0,
    "is_shortener": lambda d: d["is_shortener"] == 1.0,
    "host_is_ip": lambda d: d["host_is_ip"] == 1.0,
}


def _slice_recall(frame: pd.DataFrame, y: np.ndarray, probs: np.ndarray, threshold: float):
    out = {}
    for name, predicate in SLICES.items():
        try:
            mask = predicate(frame).to_numpy()
        except KeyError:
            continue
        n = int(mask.sum())
        positives = int(y[mask].sum()) if n else 0
        out[name] = {
            "n_rows": n,
            "n_positive": positives,
            "recall": float((probs[mask] >= threshold)[y[mask] == 1].mean()) if positives else None,
            "positive_rate": float(y[mask].mean()) if n else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imbalance", choices=["none", "balanced"], default="none")
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/4] loading {args.features}")
    data = pd.read_parquet(args.features)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS]
    parts = {name: data[data["split"] == name] for name in ("train", "calibration", "test")}
    for name, part in parts.items():
        print(f"      {name:<12} {len(part):>8,} rows  pos rate {part['label'].mean():.4f}")

    config = BaselineConfig(imbalance=args.imbalance, seed=args.seed)
    print(f"[2/4] fitting baseline on train ({config.as_dict()})")
    model = build_baseline(config)
    model.fit(parts["train"][feature_columns].to_numpy(), parts["train"]["label"].to_numpy())

    print(f"[3/4] fitting {args.calibration} calibrator on the calibration split")
    calibrator = build_calibrator(args.calibration)
    calibrator.fit(
        decision_scores(model, parts["calibration"][feature_columns].to_numpy()),
        parts["calibration"]["label"].to_numpy(),
    )

    print("[4/4] evaluating on test")
    test = parts["test"]
    y_test = test["label"].to_numpy()
    raw = decision_scores(model, test[feature_columns].to_numpy())
    probs = calibrator.predict_proba(raw)
    result = m.evaluate(y_test, probs)
    print(f"      {result.summary_line()}")

    uncalibrated = 1.0 / (1.0 + np.exp(-raw))
    ece_before = m.expected_calibration_error(y_test, uncalibrated)
    print(f"      ECE before calibration {ece_before:.4f} -> after {result.ece:.4f}")

    threshold = result.threshold_at_fpr.get("0.05", 0.5)
    slices = _slice_recall(test, y_test, probs, threshold)
    print(f"      per-slice recall at FPR 5% threshold {threshold:.4f}:")
    for name, stats in slices.items():
        recall = "n/a" if stats["recall"] is None else f"{stats['recall']:.4f}"
        print(f"        {name:<22} n={stats['n_rows']:>7,}  recall={recall}")

    per_type = {}
    for label_type in sorted(test["type"].unique()):
        if label_type == "benign":
            continue
        mask = (test["type"] == label_type).to_numpy()
        per_type[label_type] = {
            "n_rows": int(mask.sum()),
            "recall": float((probs[mask] >= threshold).mean()),
        }
    print("      per-class recall:")
    for label_type, stats in per_type.items():
        print(f"        {label_type:<22} n={stats['n_rows']:>7,}  recall={stats['recall']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.as_dict(),
        "calibration_method": args.calibration,
        "calibrator_state": calibrator.to_dict(),
        "n_features": len(feature_columns),
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
        "test": result.as_dict(),
        "ece_before_calibration": ece_before,
        "slice_recall_at_fpr_05": slices,
        "per_class_recall_at_fpr_05": per_type,
        "top_coefficients": [c.as_dict() for c in coefficient_table(model, feature_columns)[:25]],
    }
    (args.out / "baseline_report.json").write_text(json.dumps(payload, indent=2))
    print(f"      wrote {args.out / 'baseline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
