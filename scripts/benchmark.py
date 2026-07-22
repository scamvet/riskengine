#!/usr/bin/env python
"""Benchmark model families under the frozen metric protocol (ADR-009).

    python scripts/benchmark.py
    python scripts/benchmark.py --seeds 42 1 7 --models logistic lightgbm

Reporting is phishing-first and multi-seed, both deliberately.

Phishing-first because the aggregate binary target mixes three problems and the
two easy ones - defacement and malware URLs, which look structurally wrong -
dominate it. ScamVet's actual question is whether a link someone was sent is a
phishing page, and phishing is engineered to look legitimate.

Multi-seed because a single-seed metric is a point estimate presented as a
fact. Phishing recall at FPR 5% moves by roughly 0.013 across seeds for a fixed
configuration on this corpus, which is larger than the gap between several
configurations we were about to choose between. Every figure here is therefore
a mean with its observed range, and a difference only counts as a finding if it
exceeds the seed spread.

Every model fits on train, calibrates on the calibration split, and is scored
only on test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scamvet_riskengine.eval import metrics as m
from scamvet_riskengine.eval.aggregate import as_dict, separates, summarise
from scamvet_riskengine.models import BaselineConfig, build_baseline, build_calibrator
from scamvet_riskengine.models.baseline import decision_scores
from scamvet_riskengine.models.boosted import (
    BoostedConfig,
    boosted_scores,
    build_lightgbm,
    build_xgboost,
)

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]
ALL_MODELS = ("logistic", "xgboost", "lightgbm")
HEADLINE = "recall_at_fpr_0.05"


def _fit(
    name: str, X_train: np.ndarray, y_train: np.ndarray, imbalance: str, seed: int
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any], float]:
    started = time.perf_counter()
    if name == "logistic":
        config: Any = BaselineConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
        model = build_baseline(config)
        model.fit(X_train, y_train)

        def score(X: np.ndarray) -> np.ndarray:
            return decision_scores(model, X)
    elif name in ("xgboost", "lightgbm"):
        config = BoostedConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
        builder = build_xgboost if name == "xgboost" else build_lightgbm
        booster = builder(config, y_train)
        booster.fit(X_train, y_train)

        def score(X: np.ndarray) -> np.ndarray:
            return boosted_scores(booster, X)
    else:
        raise SystemExit(f"unknown model {name!r}")
    return score, config.as_dict(), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    parser.add_argument("--imbalance", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1, 7],
        help="At least 2. Reported figures are the mean and observed range.",
    )
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise SystemExit("--seeds needs at least 2 values; a spread over one seed is not a spread")

    data = pd.read_parquet(args.features)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS]
    parts = {n: data[data["split"] == n] for n in ("train", "calibration", "test")}
    X = {n: p[feature_columns].to_numpy(dtype=np.float32) for n, p in parts.items()}
    y = {n: p["label"].to_numpy() for n, p in parts.items()}
    test = parts["test"].reset_index(drop=True)

    phishing_mask = test["type"].isin(["benign", "phishing"]).to_numpy()
    y_phishing = (test["type"].to_numpy()[phishing_mask] == "phishing").astype(int)

    print(f"features {len(feature_columns)} | train {len(X['train']):,} | test {len(X['test']):,}")
    print(f"phishing view: {int(phishing_mask.sum()):,} rows, base rate {y_phishing.mean():.4f}")
    print(f"calibration={args.calibration} imbalance={args.imbalance} seeds={args.seeds}\n")

    results: dict[str, Any] = {}
    for name in args.models:
        overall_runs, phishing_runs, per_class_runs, fit_times = [], [], [], []
        config: dict[str, Any] = {}
        for seed in args.seeds:
            score, config, fit_seconds = _fit(name, X["train"], y["train"], args.imbalance, seed)
            fit_times.append(fit_seconds)
            calibrator = build_calibrator(args.calibration)
            calibrator.fit(score(X["calibration"]), y["calibration"])
            probs = calibrator.predict_proba(score(X["test"]))

            overall = m.evaluate(y["test"], probs)
            overall_runs.append(overall)
            phishing_runs.append(m.evaluate(y_phishing, probs[phishing_mask]))

            threshold = overall.threshold_at_fpr.get("0.05", 0.5)
            per_class_runs.append(
                {
                    label: float((probs[(test["type"] == label).to_numpy()] >= threshold).mean())
                    for label in ("phishing", "malware", "defacement")
                }
            )

        overall_summary = summarise(overall_runs)
        phishing_summary = summarise(phishing_runs)
        per_class = {
            label: float(np.mean([run[label] for run in per_class_runs]))
            for label in per_class_runs[0]
        }

        print(f"--- {name} ---  fit {np.mean(fit_times):.1f}s mean over {len(args.seeds)} seeds")
        print(f"  all      PR-AUC       {overall_summary['pr_auc'].format()}")
        print(f"  phishing PR-AUC       {phishing_summary['pr_auc'].format()}")
        print(f"  phishing R@FPR5%      {phishing_summary[HEADLINE].format()}")
        print(f"  phishing ECE          {phishing_summary['ece'].format()}")
        print("  mean recall@FPR5%  " + "  ".join(f"{k}={v:.4f}" for k, v in per_class.items()))
        print()

        results[name] = {
            "config": config,
            "seeds": args.seeds,
            "mean_fit_seconds": round(float(np.mean(fit_times)), 2),
            "overall": as_dict(overall_summary),
            "phishing_only": as_dict(phishing_summary),
            "mean_per_class_recall_at_fpr_05": per_class,
            "_summary": phishing_summary,
        }

    print("| model | phishing PR-AUC | phishing R@FPR5% | phishing ECE | fit s |")
    print("|---|---|---|---|---|")
    for name, payload in results.items():
        summary = payload["_summary"]
        print(
            f"| {name} | {summary['pr_auc'].format()} | {summary[HEADLINE].format()} | "
            f"{summary['ece'].format()} | {payload['mean_fit_seconds']:.1f} |"
        )

    if len(results) > 1:
        print("\nPairwise separation on phishing R@FPR5% (gap must exceed seed spread):")
        names = list(results)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                a = results[first]["_summary"][HEADLINE]
                b = results[second]["_summary"][HEADLINE]
                verdict = "SEPARATES" if separates(a, b) else "within noise"
                print(f"  {first} vs {second}: gap {abs(a.mean - b.mean):.4f} -> {verdict}")

    for payload in results.values():
        payload.pop("_summary")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(
        json.dumps(
            {
                "calibration": args.calibration,
                "imbalance": args.imbalance,
                "seeds": args.seeds,
                "n_features": len(feature_columns),
                "split_sizes": {k: int(len(v)) for k, v in parts.items()},
                "phishing_view_rows": int(phishing_mask.sum()),
                "phishing_view_base_rate": float(y_phishing.mean()),
                "models": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
