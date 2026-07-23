#!/usr/bin/env python
"""Measure whether standard imbalance interventions move phishing recall.

    python scripts/imbalance_study.py
    python scripts/imbalance_study.py --arms baseline smote --seeds 42 1 7
    python scripts/imbalance_study.py --ratios 20 100

Five arms, one model family, everything else held fixed: the shipped XGBoost
configuration, the same grouped splits, the same calibrator, and the frozen
metric protocol of ADR-009. Only the treatment of class balance changes.

The question is not whether the corpus is imbalanced. At 18.63% positive on the
phishing view it barely is, by the standards these techniques were built for.
The question is whether the toolkit that gets recommended reflexively does
anything measurable here - and if not, at what ratio it would start to, which
is what ``--ratios`` is for.

Two predictions are recorded before the run, because a study that only reports
what it found afterwards cannot be wrong:

1. SMOTE should hurt. ``brand_min_distance`` returns a sentinel 64 when no
   brand is near, so interpolating an active 2 against an inactive 64 yields
   33 - a synthetic row asserting a distance no real URL can have. The run
   quantifies how many such rows it manufactures.
2. Focal loss should hurt. It up-weights hard examples by construction, and
   this corpus carries label errors in both directions, so its hardest
   examples include its wrong ones.

Class weighting is expected to do nothing measurable: the headline metrics are
rank-based, and at this ratio ``scale_pos_weight`` is close to a monotone
rescaling of the score.

Every arm is calibrated identically before evaluation. Focal loss deliberately
distorts probability scale, so scoring a raw focal arm against a calibrated
baseline would manufacture a difference that belongs to the measurement rather
than to the model.

Runtime: roughly 20-40 minutes for the default five arms over three seeds. The
SMOTE arm spends most of its time in a nearest-neighbour search over the
minority class and prints nothing while it does. Do not interrupt it.
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
from scamvet_riskengine.models import build_calibrator
from scamvet_riskengine.models.boosted import BoostedConfig, boosted_scores, build_xgboost
from scamvet_riskengine.models.focal import sigmoid, xgboost_focal_objective

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]
HEADLINE = "recall_at_fpr_0.05"

#: Metrics compared against baseline, with whether higher is better. ECE is a
#: calibration error, so its improvement direction is downward: reading a lower
#: ECE as "worse" because the gap is negative would invert the finding.
COMPARED = ((HEADLINE, True), ("pr_auc", True), ("ece", False))
ARMS = ("baseline", "scale_pos_weight", "smote", "focal_g1", "focal_g2")

#: ``brand_min_distance`` reports this when no brand is within edit range. It is
#: a categorical "not applicable" wearing a numeric costume, which is precisely
#: what makes it hostile to interpolation.
BRAND_SENTINEL = 64.0
BRAND_COLUMN = "brand_min_distance"

ScoreFn = Callable[[np.ndarray], np.ndarray]


def _fit_arm(
    arm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[ScoreFn, dict[str, Any], float]:
    """Fit one arm and return a scoring function on the same probability scale."""
    started = time.perf_counter()
    imbalance = "balanced" if arm == "scale_pos_weight" else "none"
    config = BoostedConfig(imbalance=imbalance, seed=seed)  # type: ignore[arg-type]
    model = build_xgboost(config, y_train)

    if arm.startswith("focal_"):
        gamma = float(arm.split("_g")[1])
        model.set_params(objective=xgboost_focal_objective(gamma))
        # base_score stays at its default. XGBoost reads it as a probability for
        # a classifier, so forcing 0.0 starts every example at logit(0) - deep in
        # the saturated tail, where focal curvature collapses. Measured on an
        # identical fit: hessian sum 0.1 against min_child_weight 5.0, therefore
        # 0 splits and a constant score, against 2,497 splits at the default.
        model.fit(X_train, y_train)

        def score(X: np.ndarray) -> np.ndarray:
            # A custom objective leaves predict_proba without a link function,
            # so the margin is converted explicitly. Every arm therefore reaches
            # the calibrator on one scale.
            return sigmoid(model.predict(X, output_margin=True))
    else:
        model.fit(X_train, y_train)

        def score(X: np.ndarray) -> np.ndarray:
            return boosted_scores(model, X)

    return score, config.as_dict(), time.perf_counter() - started


def _smote_resample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Oversample the minority class, and report what interpolation invented.

    SMOTE is applied to the training fold only, after the grouped split, so no
    synthetic row can carry information across an eTLD+1 boundary into the
    calibration or test folds.
    """
    from imblearn.over_sampling import SMOTE

    original = len(X_train)
    X_resampled, y_resampled = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    X_resampled = np.asarray(X_resampled, dtype=np.float32)
    y_resampled = np.asarray(y_resampled)

    report: dict[str, Any] = {
        "rows_before": int(original),
        "rows_after": int(len(X_resampled)),
        "synthetic_rows": int(len(X_resampled) - original),
    }

    if BRAND_COLUMN in feature_columns:
        index = feature_columns.index(BRAND_COLUMN)
        real = X_train[:, index]
        synthetic = X_resampled[original:, index]
        active = real[real < BRAND_SENTINEL]
        max_active = float(active.max()) if active.size else 0.0
        impossible = (synthetic > max_active) & (synthetic < BRAND_SENTINEL)
        non_integer = np.abs(synthetic - np.round(synthetic)) > 1e-6
        report[BRAND_COLUMN] = {
            "sentinel": BRAND_SENTINEL,
            "max_real_active_value": max_active,
            "sentinel_share_of_real_rows": float((real >= BRAND_SENTINEL).mean()),
            "synthetic_in_impossible_gap": int(impossible.sum()),
            "synthetic_in_impossible_gap_share": float(impossible.mean()),
            "synthetic_non_integer": int(non_integer.sum()),
            "synthetic_non_integer_share": float(non_integer.mean()),
        }
    return X_resampled, y_resampled, report


def _subsample_positives(
    X: np.ndarray, y: np.ndarray, ratio: int, seed: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Thin the positive class until negatives outnumber it by ``ratio``."""
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    keep = max(1, int(len(negatives) / ratio))
    if keep >= len(positives):
        return X, y, False
    chosen = rng.choice(positives, size=keep, replace=False)
    order = np.sort(np.concatenate([negatives, chosen]))
    return X[order], y[order], True


def _run_arm(
    arm: str,
    data: dict[str, Any],
    seeds: list[int],
    calibration: str,
    ratio: int | None,
) -> dict[str, Any]:
    phishing_runs, overall_runs, fit_times = [], [], []
    resample_report: dict[str, Any] = {}
    config: dict[str, Any] = {}

    for seed in seeds:
        X_train, y_train = data["X"]["train"], data["y"]["train"]
        if ratio is not None:
            X_train, y_train, applied = _subsample_positives(X_train, y_train, ratio, seed)
            if not applied:
                print(f"    ratio 1:{ratio} is already exceeded naturally; using the full fold")
        if arm == "smote":
            X_train, y_train, resample_report = _smote_resample(
                X_train, y_train, data["feature_columns"], seed
            )

        score, config, fit_seconds = _fit_arm(arm, X_train, y_train, seed)
        fit_times.append(fit_seconds)

        raw_test = score(data["X"]["test"])
        if float(np.std(raw_test)) < 1e-9:
            raise SystemExit(
                f"arm {arm!r} at seed {seed} gave every test row the same score "
                f"(std {float(np.std(raw_test)):.2e}); the model did not train. That is a "
                "harness fault, not a result. A degenerate arm must never reach the table "
                "as evidence about the technique it was supposed to test."
            )
        calibrator = build_calibrator(calibration)
        calibrator.fit(score(data["X"]["calibration"]), data["y"]["calibration"])
        probabilities = calibrator.predict_proba(raw_test)

        overall_runs.append(m.evaluate(data["y"]["test"], probabilities))
        phishing_runs.append(m.evaluate(data["y_phishing"], probabilities[data["phishing_mask"]]))

    phishing_summary = summarise(phishing_runs)
    print(f"--- {arm} ---  fit {np.mean(fit_times):.1f}s mean over {len(seeds)} seeds")
    print(f"  phishing R@FPR5%      {phishing_summary[HEADLINE].format()}")
    print(f"  phishing PR-AUC       {phishing_summary['pr_auc'].format()}")
    print(f"  phishing ECE          {phishing_summary['ece'].format()}")
    if resample_report.get(BRAND_COLUMN):
        brand = resample_report[BRAND_COLUMN]
        print(
            f"  {BRAND_COLUMN}: {brand['synthetic_in_impossible_gap_share']:.1%} of synthetic "
            f"rows fall in the impossible gap "
            f"({brand['max_real_active_value']:.0f} < value < {BRAND_SENTINEL:.0f}); "
            f"{brand['synthetic_non_integer_share']:.1%} are non-integer"
        )
    print()

    return {
        "config": config,
        "mean_fit_seconds": round(float(np.mean(fit_times)), 2),
        "overall": as_dict(summarise(overall_runs)),
        "phishing_only": as_dict(phishing_summary),
        "resampling": resample_report,
        "_summary": phishing_summary,
    }


def _load(features: Path) -> dict[str, Any]:
    frame = pd.read_parquet(features)
    feature_columns = [c for c in frame.columns if c not in META_COLUMNS]
    parts = {name: frame[frame["split"] == name] for name in ("train", "calibration", "test")}
    test = parts["test"].reset_index(drop=True)
    phishing_mask = test["type"].isin(["benign", "phishing"]).to_numpy()
    return {
        "feature_columns": feature_columns,
        "X": {
            name: part[feature_columns].to_numpy(dtype=np.float32) for name, part in parts.items()
        },
        "y": {name: part["label"].to_numpy() for name, part in parts.items()},
        "phishing_mask": phishing_mask,
        "y_phishing": (test["type"].to_numpy()[phishing_mask] == "phishing").astype(int),
        "split_sizes": {name: int(len(part)) for name, part in parts.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1, 7],
        help="At least 2. A difference smaller than the seed spread is not a finding.",
    )
    parser.add_argument(
        "--ratios",
        nargs="*",
        type=int,
        default=[],
        help="Negative-to-positive ratios to additionally test by thinning positives, "
        "e.g. --ratios 20 100. Each ratio re-runs every arm.",
    )
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise SystemExit("--seeds needs at least 2 values; a spread over one seed is not a spread")

    data = _load(args.features)
    train_rate = float(data["y"]["train"].mean())
    print(f"features {len(data['feature_columns'])} | train {len(data['X']['train']):,}")
    per_positive = (1.0 - train_rate) / train_rate
    print(f"train positive rate {train_rate:.4f} " f"({per_positive:.2f} negatives per positive)")
    print(
        f"phishing view {int(data['phishing_mask'].sum()):,} rows, "
        f"base rate {data['y_phishing'].mean():.4f}"
    )
    print(f"calibration={args.calibration} seeds={args.seeds} arms={args.arms}")
    print(f"ratios={args.ratios or 'natural only'}\n")

    conditions: list[tuple[str, int | None]] = [("natural", None)]
    conditions += [(f"1:{ratio}", ratio) for ratio in args.ratios]

    payload: dict[str, Any] = {
        "calibration": args.calibration,
        "seeds": args.seeds,
        "n_features": len(data["feature_columns"]),
        "split_sizes": data["split_sizes"],
        "train_positive_rate": train_rate,
        "phishing_view_rows": int(data["phishing_mask"].sum()),
        "phishing_view_base_rate": float(data["y_phishing"].mean()),
        "conditions": {},
    }

    for label, ratio in conditions:
        print(f"===== condition: {label} =====\n")
        results = {
            arm: _run_arm(arm, data, args.seeds, args.calibration, ratio) for arm in args.arms
        }

        print("| arm | phishing R@FPR5% | phishing PR-AUC | phishing ECE | fit s |")
        print("|---|---|---|---|---|")
        for arm, result in results.items():
            summary = result["_summary"]
            print(
                f"| {arm} | {summary[HEADLINE].format()} | {summary['pr_auc'].format()} | "
                f"{summary['ece'].format()} | {result['mean_fit_seconds']:.1f} |"
            )

        separation: dict[str, Any] = {}
        if "baseline" in results and len(results) > 1:
            for metric, higher_is_better in COMPARED:
                print(f"\nSeparation against baseline on {metric}:")
                reference = results["baseline"]["_summary"][metric]
                separation[metric] = {}
                for arm, result in results.items():
                    if arm == "baseline":
                        continue
                    spread = result["_summary"][metric]
                    gap = spread.mean - reference.mean
                    is_separated = separates(reference, spread)
                    improved = gap > 0 if higher_is_better else gap < 0
                    direction = "better" if improved else "worse"
                    verdict = "SEPARATES" if is_separated else "within noise"
                    print(f"  {arm}: {gap:+.4f} ({direction}) -> {verdict}")
                    separation[metric][arm] = {
                        "gap": float(gap),
                        "direction": direction,
                        "separates": bool(is_separated),
                        "bar": float(max(reference.spread, spread.spread)),
                    }
        print()

        payload["conditions"][label] = {
            "arms": {
                arm: {k: v for k, v in result.items() if k != "_summary"}
                for arm, result in results.items()
            },
            "separation": separation,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "imbalance_study.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
