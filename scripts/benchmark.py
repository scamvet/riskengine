#!/usr/bin/env python
"""Benchmark model families under the frozen metric protocol (ADR-009).

    python scripts/benchmark.py
    python scripts/benchmark.py --models logistic xgboost --calibration isotonic

Reporting is phishing-first, deliberately. The aggregate binary target mixes
three problems, and the two easy ones - defacement and malware URLs, which look
structurally wrong - dominate it. ScamVet's actual question is whether a link
someone was sent is a phishing page, and phishing is engineered to look
legitimate. An aggregate number is therefore not a proxy for the thing the
product does, so this script reports:

* aggregate metrics on the full binary target, for comparability;
* per-class recall, so the easy classes cannot hide the hard one;
* a benign-vs-phishing view, which is the product-relevant problem;
* the same view with IP-host rows removed, because those are ~99.8 percent
  malicious in this corpus and belong in the deterministic layer of the verdict
  stack rather than being counted as a model win.

Every model fits on train, calibrates on the calibration split, and is scored
only on test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scamvet_riskengine.eval import metrics as m
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


def _fit_and_score(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: dict[str, np.ndarray],
    imbalance: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    started = time.perf_counter()
    if name == "logistic":
        config: Any = BaselineConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
        model = build_baseline(config)
        model.fit(X_train, y_train)
        scorer = lambda X: decision_scores(model, X)  # noqa: E731
    elif name == "xgboost":
        config = BoostedConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
        model = build_xgboost(config, y_train)
        model.fit(X_train, y_train)
        scorer = lambda X: boosted_scores(model, X)  # noqa: E731
    elif name == "lightgbm":
        config = BoostedConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
        model = build_lightgbm(config, y_train)
        model.fit(X_train, y_train)
        scorer = lambda X: boosted_scores(model, X)  # noqa: E731
    else:
        raise SystemExit(f"unknown model {name!r}")
    elapsed = time.perf_counter() - started
    return {key: scorer(value) for key, value in X_other.items()}, config.as_dict(), elapsed


def _phishing_view(test: pd.DataFrame, probs: np.ndarray, mask: np.ndarray | None = None):
    """Benign vs phishing only, optionally excluding rows."""
    keep = test["type"].isin(["benign", "phishing"]).to_numpy()
    if mask is not None:
        keep = keep & mask
    y = (test["type"].to_numpy()[keep] == "phishing").astype(int)
    if len(set(y.tolist())) < 2:
        return None
    return m.evaluate(y, probs[keep])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="isotonic")
    parser.add_argument("--imbalance", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS]
    parts = {n: data[data["split"] == n] for n in ("train", "calibration", "test")}
    X = {n: p[feature_columns].to_numpy() for n, p in parts.items()}
    y = {n: p["label"].to_numpy() for n, p in parts.items()}
    test = parts["test"].reset_index(drop=True)
    not_ip = (test["host_is_ip"] == 0.0).to_numpy()

    print(f"features {len(feature_columns)} | train {len(X['train']):,} | test {len(X['test']):,}")
    print(f"calibration={args.calibration} imbalance={args.imbalance} seed={args.seed}\n")

    results: dict[str, Any] = {}
    for name in args.models:
        print(f"--- {name} ---")
        scores, config, fit_seconds = _fit_and_score(
            name,
            X["train"],
            y["train"],
            {"calibration": X["calibration"], "test": X["test"]},
            args.imbalance,
            args.seed,
        )
        calibrator = build_calibrator(args.calibration)
        calibrator.fit(scores["calibration"], y["calibration"])
        probs = calibrator.predict_proba(scores["test"])

        overall = m.evaluate(y["test"], probs)
        phishing = _phishing_view(test, probs)
        phishing_no_ip = _phishing_view(test, probs, not_ip)

        threshold = overall.threshold_at_fpr.get("0.05", 0.5)
        per_class = {}
        for label_type in ("phishing", "malware", "defacement"):
            mask = (test["type"] == label_type).to_numpy()
            if mask.sum():
                per_class[label_type] = float((probs[mask] >= threshold).mean())

        print(f"  fit {fit_seconds:.1f}s")
        print(f"  all      {overall.summary_line()}")
        if phishing:
            print(f"  phishing {phishing.summary_line()}")
        if phishing_no_ip:
            print(f"  ph no-ip {phishing_no_ip.summary_line()}")
        print("  recall@FPR5%  " + "  ".join(f"{k}={v:.4f}" for k, v in per_class.items()))
        print()

        results[name] = {
            "config": config,
            "fit_seconds": round(fit_seconds, 2),
            "overall": overall.as_dict(),
            "phishing_only": phishing.as_dict() if phishing else None,
            "phishing_only_excluding_ip_hosts": phishing_no_ip.as_dict()
            if phishing_no_ip
            else None,
            "per_class_recall_at_fpr_05": per_class,
        }

    print("| model | all PR-AUC | phishing PR-AUC | phishing R@FPR5% | phishing ECE | fit s |")
    print("|---|---|---|---|---|---|")
    for name, payload in results.items():
        ph = payload["phishing_only"]
        print(
            f"| {name} | {payload['overall']['pr_auc']:.4f} | "
            f"{ph['pr_auc']:.4f} | {ph['recall_at_fpr']['0.05']:.4f} | "
            f"{ph['ece']:.4f} | {payload['fit_seconds']:.1f} |"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration": args.calibration,
        "imbalance": args.imbalance,
        "seed": args.seed,
        "n_features": len(feature_columns),
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
        "models": results,
    }
    (args.out / "benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
