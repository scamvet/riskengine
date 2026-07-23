#!/usr/bin/env python
"""Find the shipping scorer: the best model that fits the committed-artifact cap.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --seeds 42 1 7 --candidates 200x8 150x7

ScamVet ships **one** model, serving both the online and offline paths, so this
script picks the best configuration that fits rather than choosing a weaker
sibling for on-device use.

That decision was made on evidence. Under a 500 KB cap the best fitting model
gave up 0.122 phishing recall - 14.4% relative - and that loss would have landed
on users with poor connectivity, who in this market are disproportionately the
people the product exists to protect. Splitting online and offline models would
also have meant two sets of thresholds, two model cards, two red-team runs, and
divergent ``reasons[]`` for the same URL depending on connectivity. The cap was
a pre-commit setting we chose, not a platform limit, so it was raised.

Reporting is multi-seed, for the same reason the benchmark is: a single-seed
sweep once picked between two indistinguishable configurations and presented it
as a decision. Recall figures are means with their observed range, and a
configuration only counts as better if the gap exceeds the seed spread
(ADR-009, amendment of 2026-07-23).

Outputs into ``data/derived/``:

    onnx_sweep.json     every candidate with size, multi-seed recall and parity
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
from scamvet_riskengine.eval.aggregate import as_dict, separates, summarise
from scamvet_riskengine.models import build_calibrator
from scamvet_riskengine.models.boosted import BoostedConfig, boosted_scores, build_xgboost
from scamvet_riskengine.models.onnx_export import SIZE_CAP_BYTES, ExportError, export_and_verify

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]
HEADLINE = "recall_at_fpr_0.05"

#: Candidate sizes, sampled finely just under the cap because that is where the
#: decision lives. XGBoost trees are roughly four times denser than LightGBM's
#: at the same nominal config - depth-wise growth against leaf-wise capping - so
#: a single step in depth moves the artifact by hundreds of kilobytes.
CANDIDATES = [
    (400, 8),
    (300, 8),
    (200, 8),
    (150, 8),
    (120, 8),
    (100, 8),
    (200, 7),
    (150, 7),
    (200, 6),
    (100, 6),
    (60, 6),
    (20, 4),
]


def _parse_candidates(raw: list[str] | None) -> list[tuple[int, int]]:
    if not raw:
        return CANDIDATES
    parsed = []
    for item in raw:
        try:
            trees, depth = item.lower().split("x")
            parsed.append((int(trees), int(depth)))
        except ValueError:
            raise SystemExit(f"bad candidate {item!r}; expected forms like 200x8") from None
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1, 7],
        help="At least 2. The exported artifact is trained on the first.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Override the sweep, e.g. --candidates 200x8 150x7",
    )
    parser.add_argument("--parity-samples", type=int, default=5000)
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise SystemExit("--seeds needs at least 2 values; a spread over one seed is not a spread")

    candidates = _parse_candidates(args.candidates)
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
        f"features {len(feature_columns)} | train {len(X['train']):,} | "
        f"cap {SIZE_CAP_BYTES // 1024} KB | seeds {args.seeds}"
    )
    print(f"calibration={args.calibration}\n")
    print(
        f"{'trees':>6} {'depth':>6} {'onnx KB':>9} {'fits':>5} {'phishing R@FPR5%':>22} {'parity':>10}"
    )

    rows: list[dict[str, Any]] = []
    for n_estimators, max_depth in candidates:
        runs = []
        blob = b""
        export = None
        for seed in args.seeds:
            config = BoostedConfig(n_estimators=n_estimators, max_depth=max_depth, seed=seed)
            model = build_xgboost(config, y["train"])
            model.fit(X["train"], y["train"])

            calibrator = build_calibrator(args.calibration)
            calibrator.fit(boosted_scores(model, X["calibration"]), y["calibration"])
            probs = calibrator.predict_proba(boosted_scores(model, X["test"]))
            runs.append(m.evaluate(y_phishing, probs[phishing_mask]))

            if export is None:  # the exported artifact comes from the first seed
                try:
                    blob, export = export_and_verify(model, parity_rows)
                except ExportError as exc:
                    print(f"{n_estimators:>6} {max_depth:>6}  export failed: {exc}")
                    break
        if export is None:
            continue

        summary = summarise(runs)
        print(
            f"{n_estimators:>6} {max_depth:>6} {export.size_bytes / 1024:>9.1f} "
            f"{'yes' if export.fits_size_cap else 'NO':>5} {summary[HEADLINE].format():>22} "
            f"{export.max_abs_parity_diff:>10.2e}"
        )
        rows.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "export": export.as_dict(),
                "phishing": as_dict(summary),
                "_summary": summary,
                "_blob": blob,
            }
        )

    fitting = [r for r in rows if r["export"]["fits_size_cap"]]
    if not fitting:
        print("\nNo candidate fits the cap. Raise it or reduce the feature set.")
        return 1

    best = max(fitting, key=lambda r: r["_summary"][HEADLINE].mean)
    largest = max(rows, key=lambda r: r["_summary"][HEADLINE].mean)

    print(
        f"\nchosen model: {best['n_estimators']} trees, depth {best['max_depth']}, "
        f"{best['export']['size_kb']} KB, {best['_summary'][HEADLINE].format()}"
    )
    if best is largest:
        print("the best fitting model is also the best overall; the cap is not binding")
    else:
        gap = largest["_summary"][HEADLINE].mean - best["_summary"][HEADLINE].mean
        verdict = (
            "SEPARATES - the cap is costing real recall"
            if separates(largest["_summary"][HEADLINE], best["_summary"][HEADLINE])
            else "within seed noise - the cap costs nothing measurable"
        )
        print(
            f"best over cap: {largest['n_estimators']}x{largest['max_depth']} at "
            f"{largest['export']['size_kb']} KB, {largest['_summary'][HEADLINE].format()}"
        )
        print(f"gap {gap:.4f} -> {verdict}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "url_scorer.onnx").write_bytes(best["_blob"])
    for row in rows:
        row.pop("_blob")
        row.pop("_summary")
    (args.out / "onnx_sweep.json").write_text(
        json.dumps(
            {
                "cap_bytes": SIZE_CAP_BYTES,
                "calibration": args.calibration,
                "seeds": args.seeds,
                "exported_from_seed": args.seeds[0],
                "chosen": {"n_estimators": best["n_estimators"], "max_depth": best["max_depth"]},
                "candidates": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out / 'url_scorer.onnx'} and {args.out / 'onnx_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
