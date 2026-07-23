#!/usr/bin/env python
"""Bound how much of the model's performance rests on corpus provenance.

    python scripts/audit_provenance.py

Three features have now been removed for encoding *where a row came from*
rather than anything about the address:

* ``had_scheme`` - defacement rows were 100% scheme-prefixed, benign 8%;
* ``scheme_is_https`` - https rows were 87.5% malicious, and only 2.4% of the
  corpus is https at all;
* and the ``.ca`` and ``.nl`` one-hots look like the same thing, since the
  benign upstream (ISCX-URL-2016, University of New Brunswick) is Canada-
  skewed.

Three findings with one root cause: the corpus is a compilation of five sources
with different collection methods, so any feature correlated with source is
contaminated. Removing them one at a time is whack-a-mole with no defined end.

This script bounds the exposure instead. It trains the same model on nested
feature subsets, from everything down to only those features that cannot
plausibly encode storage convention or collection geography, and reports
phishing recall for each. The narrowest subset's score is the honest floor:
the performance that would survive a differently-sourced corpus.

It is a bound, not a proof. Without source labels no subset can be certified
clean; what this shows is how much of the headline number depends on features
whose meaning is suspect.
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

DEFAULT_FEATURES = Path("data/derived/url_features.parquet")
DEFAULT_OUT = Path("data/derived")
META_COLUMNS = ["url", "type", "label", "group", "split"]
HEADLINE = "recall_at_fpr_0.05"

#: Features that describe the address in ways a person would recognise, and
#: that cannot encode which file a row was stored in. Curated by hand, because
#: the judgement is semantic and cannot be derived from the data.
SEMANTIC_CORE = {
    "brand_in_host",
    "brand_domain_mismatch",
    "brand_min_distance",
    "suspicious_tld",
    "is_shortener",
    "host_is_ip",
    "host_is_ipv6",
    "has_punycode",
    "has_at_symbol",
    "has_port",
    "has_double_slash_in_path",
    "num_hyphens_host",
    "num_encoded_chars",
    "keyword_hits_auth",
    "keyword_hits_payment",
    "keyword_hits_urgency",
    "keyword_hits_reward",
    "keyword_hits_total",
    "host_entropy",
    "path_entropy",
    "host_digit_ratio",
    "url_digit_ratio",
}


def _subsets(columns: list[str]) -> dict[str, list[str]]:
    one_hot = [c for c in columns if c.startswith("tld_is_")]
    shape = [c for c in columns if c not in SEMANTIC_CORE and c not in one_hot and c != "parse_ok"]
    return {
        "all": list(columns),
        "no_tld_onehot": [c for c in columns if c not in one_hot],
        "semantic_plus_shape": [c for c in columns if c in SEMANTIC_CORE or c in shape],
        "semantic_only": [c for c in columns if c in SEMANTIC_CORE],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 7])
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    columns = [c for c in data.columns if c not in META_COLUMNS]
    parts = {n: data[data["split"] == n] for n in ("train", "calibration", "test")}
    y = {n: p["label"].to_numpy() for n, p in parts.items()}
    test = parts["test"].reset_index(drop=True)

    phishing_mask = test["type"].isin(["benign", "phishing"]).to_numpy()
    y_phishing = (test["type"].to_numpy()[phishing_mask] == "phishing").astype(int)

    subsets = _subsets(columns)
    print(f"phishing view: {int(phishing_mask.sum()):,} rows, base rate {y_phishing.mean():.4f}")
    print(f"model xgboost | seeds {args.seeds}\n")
    print(f"{'subset':<22} {'cols':>5} {'phishing PR-AUC':>22} {'phishing R@FPR5%':>22}")

    results: dict[str, Any] = {}
    for name, chosen in subsets.items():
        X = {n: p[chosen].to_numpy(dtype=np.float32) for n, p in parts.items()}
        runs = []
        for seed in args.seeds:
            model = build_xgboost(BoostedConfig(seed=seed), y["train"])
            model.fit(X["train"], y["train"])
            calibrator = build_calibrator("platt")
            calibrator.fit(boosted_scores(model, X["calibration"]), y["calibration"])
            probs = calibrator.predict_proba(boosted_scores(model, X["test"]))
            runs.append(m.evaluate(y_phishing, probs[phishing_mask]))
        summary = summarise(runs)
        print(
            f"{name:<22} {len(chosen):>5} {summary['pr_auc'].format():>22} "
            f"{summary[HEADLINE].format():>22}"
        )
        results[name] = {
            "n_columns": len(chosen),
            "columns": chosen,
            "phishing": as_dict(summary),
            "_summary": summary,
        }

    everything = results["all"]["_summary"][HEADLINE]
    floor = results["semantic_only"]["_summary"][HEADLINE]
    gap = everything.mean - floor.mean
    print(f"\nheadline (all features):      {everything.format()}")
    print(f"honest floor (semantic only): {floor.format()}")
    print(f"attributable to features of suspect provenance: {gap:.4f}")
    print(f"verdict: {'SEPARATES' if separates(everything, floor) else 'within seed noise'}")

    for payload in results.values():
        payload.pop("_summary")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance_audit.json").write_text(
        json.dumps({"seeds": args.seeds, "model": "xgboost", "subsets": results}, indent=2)
    )
    print(f"\nwrote {args.out / 'provenance_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
