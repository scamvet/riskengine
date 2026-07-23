# RiskEngine

> Calibrated phishing and fraud risk scoring with TreeSHAP reasons and a
> versioned model registry — the scoring library behind
> [ScamVet](https://github.com/scamvet).

```bash
pip install scamvet-riskengine
```

> **Install string note:** the PyPI distribution is **`scamvet-riskengine`**
> (`import scamvet_riskengine`). PyPI normalizes the bare name `riskengine` onto
> the pre-existing `risk-engine` project (PEP 503), which hard-blocks it — so the
> library is brand-scoped. The repo and product name stay "RiskEngine." See
> [ADR-003](https://github.com/scamvet/riskengine/tree/main/docs/adr).

## Quick start

```python
from scamvet_riskengine import load

scorer = load("url-scorer-lexical")
result = scorer.score("http://sbi-kyc-verify.xyz/login/update")

print(result.band, result.probability)     # amber 0.9021
for reason in result.reasons:
    print(reason.template_key, reason.direction, reason.magnitude)
```

## Install profiles

| Command | Adds | Gives you |
|---|---|---|
| `pip install scamvet-riskengine` | onnxruntime | Scoring, bands, calibrated probabilities |
| `pip install "scamvet-riskengine[explain]"` | xgboost | TreeSHAP reasons on every verdict |
| `pip install "scamvet-riskengine[train]"` | lightgbm, scikit-learn, imbalanced-learn, ONNX exporters | Reproducing the benchmarks |

Inference runs on ONNX rather than the training library, so the default install
stays small and the Python path scores through the same artifact the browser and
the Android app do. Without `[explain]` a verdict carries no reasons and sets the
`explanations_unavailable` flag — an empty `reasons` list is never silently
ambiguous.

## What ships

`url-scorer-lexical 0.1.0` — XGBoost (200 trees, depth 8), exported to a 1,408 KB
ONNX artifact with verified parity, Platt-calibrated, with per-use-case verdict
bands and exact TreeSHAP attributions mapped to versioned reason templates.

| Figure | Value |
|---|---|
| Phishing recall @ 5% FPR | 0.706 ± 0.011 (3 seeds) |
| Phishing PR-AUC | 0.814 ± 0.000 |
| Intrinsic-feature floor | 0.230 ± 0.016 |
| False positives on known-legitimate domains | 3 of 55, all amber |
| Training set | 651,191 rows, CC0 |
| Feature spec | `url_lexical_v5`, 34 columns, scheme-invariant |

Every model carries a manifest recording its corpus, licence row, feature spec,
calibration state, thresholds and the library versions its features were
extracted with — because a model file without one cannot be reproduced, audited
or safely consumed.

## The intrinsic-feature floor

Three features in earlier spec versions encoded which source file a row came
from rather than anything about the address. The worst was invisible to feature
review: `url_length` counted the `https://` prefix, and since most corpus rows
carry no scheme while every real user pastes one, the same bank domain scored
0.018 or 0.985 depending on how it was written. No metric on 86,115 held-out
rows could catch it — train and test shared the convention. It was found by
scoring 55 domains already known to be legitimate.

Every headline number here is therefore published beside an intrinsic-feature
floor measured on the semantic-only subset, and a false-positive check against
known-legitimate domains runs at publish time.

## Corpus and licensing

Two tracks, per
[ADR-007](https://github.com/scamvet/riskengine/tree/main/docs/adr). The
**product** track — everything distributed in this package — trains on the CC0
malicious-URL corpus alone. The **research** track (Bank Account Fraud, IEEE-CIS,
PaySim; ~13.6M rows, largely synthetic) is used for benchmarking and fairness
work published as notebooks, and **is never shipped in a model artifact**. Every
dataset is credited with source, author, licence and link in
[LEGAL.md](https://github.com/scamvet/riskengine/blob/main/LEGAL.md).

## Status

0.1.0. The product scorer, registry, calibration, TreeSHAP reasons and provenance
audit are shipped. Imbalance study, the BAF fairness report across six
bias-controlled variants, and published Kaggle notebooks are in progress.

## Contributing

See
[CONTRIBUTING.md](https://github.com/scamvet/riskengine/blob/main/CONTRIBUTING.md).
CLA required (one click, first PR).

## License

[AGPL-3.0-or-later](https://github.com/scamvet/riskengine/blob/main/LICENSE).

## Built by

[Ayush Gupta](https://github.com/ayushgupta07xx).
