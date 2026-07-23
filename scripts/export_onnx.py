#!/usr/bin/env python
"""Find the shipping scorer: the best model that fits the committed-artifact cap.

    python scripts/export_onnx.py

ScamVet ships **one** model, serving both the online and offline paths, so this
script's job is to find the best configuration that fits the committed-artifact
cap rather than to pick a weaker sibling for on-device use.

That decision was made on evidence. At a 500 KB cap the best fitting model gave
up 0.122 phishing recall - 14.4% relative - against the unconstrained one, and
that loss would have fallen on users with poor connectivity, who in this market
are disproportionately the people the product exists to protect. Splitting into
online and offline models would also have meant two sets of thresholds, two
model cards, two red-team runs, and - worst - divergent ``reasons[]`` for the
same URL depending on connectivity, which undermines the one thing ScamVet
claims to do better than anyone. The cap was a pre-commit setting we chose, not
a platform limit, so it was raised to 1 MB instead.

Every candidate is trained on train, calibrated on the calibration split, and
scored on test - the same discipline as the benchmark. Reporting is phishing-
first for the same reason: the aggregate target is dominated by classes that
are easy for structural reasons.

Outputs into ``data/derived/``:

    onnx_sweep.json     every candidate with size, recall and parity
    url_scorer.onnx     the chosen model, used on both paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scamvet_riskengine.eval import metrics as m
from scamvet_riskengine.models import build_calibrator
from scamvet_riskengine.models.boosted import BoostedConfig, boosted_scores, build_xgboost
from scamvet_riskengine.models.onnx_export import SIZE_CAP_BYTES, ExportError, export_and_verify

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]

#: Candidate sizes. Sampled finely just under the cap, because that is where
#: the decision actually lives: XGBoost trees are roughly four times denser
#: than LightGBM's at the same nominal config - depth-wise growth against
#: leaf-wise capping - so small changes in depth move the artifact by hundreds
#: of kilobytes.
CANDIDATES = [
    (400, 8),
    (200, 8),
    (150, 8),
    (120, 8),
    (100, 8),
    (200, 7),
    (150, 7),
    (100, 7),
    (200, 6),
    (150, 6),
    (100, 6),
    (60, 6),
    (40, 6),
    (20, 4),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity-samples", type=int, default=5000)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS]
    parts = {n: data[data["split"] == n] for n in ("train", "calibration", "test")}
    X = {n: p[feature_columns].to_numpy(dtype=np.float32) for n, p in parts.items()}
    y = {n: p["label"].to_numpy() for n, p in parts.items()}
    test = parts["test"].reset_index(drop=True)

    phishing_mask = test["type"].isin(["benign", "phishing"]).to_numpy()
    y_phishing = (test["type"].to_numpy()[phishing_mask] == "phishing").astype(int)

    # Real feature rows, not noise: a conversion bug in a rarely-taken branch
    # will not surface on data that never reaches it.
    parity_rows = X["test"][: args.parity_samples]

    print(
        f"features {len(feature_columns)} | train {len(X['train']):,} | cap {SIZE_CAP_BYTES // 1024} KB"
    )
    print(f"calibration={args.calibration}\n")
    print(
        f"{'trees':>6} {'depth':>6} {'onnx KB':>9} {'fits':>5} {'ph PR-AUC':>10} "
        f"{'ph R@FPR5%':>11} {'parity':>10}"
    )

    rows: list[dict[str, Any]] = []
    for n_estimators, max_depth in CANDIDATES:
        config = BoostedConfig(n_estimators=n_estimators, max_depth=max_depth, seed=args.seed)
        model = build_xgboost(config, y["train"])
        model.fit(X["train"], y["train"])

        calibrator = build_calibrator(args.calibration)
        calibrator.fit(boosted_scores(model, X["calibration"]), y["calibration"])
        probs = calibrator.predict_proba(boosted_scores(model, X["test"]))
        result = m.evaluate(y_phishing, probs[phishing_mask])

        try:
            blob, export = export_and_verify(model, parity_rows)
        except ExportError as exc:
            print(f"{n_estimators:>6} {max_depth:>6}  export failed: {exc}")
            continue

        print(
            f"{n_estimators:>6} {max_depth:>6} {export.size_bytes / 1024:>9.1f} "
            f"{'yes' if export.fits_size_cap else 'NO':>5} {result.pr_auc:>10.4f} "
            f"{result.recall_at_fpr['0.05']:>11.4f} {export.max_abs_parity_diff:>10.2e}"
        )

        rows.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "config": config.as_dict(),
                "export": export.as_dict(),
                "phishing": result.as_dict(),
                "calibrator_state": calibrator.to_dict(),
                "_blob": blob,
            }
        )

    fitting = [r for r in rows if r["export"]["fits_size_cap"]]
    if not fitting:
        print("\nNo candidate fits the cap. Raise it or reduce the feature set.")
        return 1

    best = max(fitting, key=lambda r: r["phishing"]["recall_at_fpr"]["0.05"])
    largest = max(rows, key=lambda r: r["phishing"]["recall_at_fpr"]["0.05"])
    cost = largest["phishing"]["recall_at_fpr"]["0.05"] - best["phishing"]["recall_at_fpr"]["0.05"]

    print(
        f"\nchosen model: {best['n_estimators']} trees, depth {best['max_depth']}, "
        f"{best['export']['size_kb']} KB"
    )
    print(
        f"phishing recall {best['phishing']['recall_at_fpr']['0.05']:.4f} vs "
        f"{largest['phishing']['recall_at_fpr']['0.05']:.4f} for the best unconstrained model"
    )
    print(
        f"cost of fitting under the cap: {cost:.4f} recall "
        f"({cost / largest['phishing']['recall_at_fpr']['0.05']:.1%} relative)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "url_scorer.onnx").write_bytes(best["_blob"])
    for row in rows:
        row.pop("_blob")
    (args.out / "onnx_sweep.json").write_text(
        json.dumps(
            {
                "cap_bytes": SIZE_CAP_BYTES,
                "calibration": args.calibration,
                "seed": args.seed,
                "chosen": {k: v for k, v in best.items()},
                "candidates": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out / 'url_scorer.onnx'} and {args.out / 'onnx_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
