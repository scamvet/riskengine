# RiskEngine

> Calibrated fraud & phishing risk scoring with SHAP explanations and a versioned
> model registry — the scoring library behind [ScamVet](https://github.com/scamvet).

```bash
pip install scamvet-riskengine
```

> **Install string note:** the PyPI distribution is **`scamvet-riskengine`**
> (`import scamvet_riskengine`). PyPI normalizes the bare name `riskengine` onto
> the pre-existing `risk-engine` project (PEP 503), which hard-blocks it — so the
> library is brand-scoped. The repo and product name stay "RiskEngine." See
> [ADR-003](docs/adr/).

## What it does

Classical-ML scorers (calibrated logistic, RF, XGBoost, LightGBM) trained on a
~13M-row public corpus — Bank Account Fraud suite (NeurIPS 2022), IEEE-CIS,
PaySim, and a 650K+ malicious-URL corpus with engineered lexical/host features.
Every score ships with:

- **Calibration** (Platt / isotonic + reliability diagrams) — the probability
  means what it says.
- **SHAP explanations** — global and per-prediction.
- A **published fairness analysis** (the BAF suite's six bias-controlled variants).
- A **versioned model registry** consumed by ScamVet Core as tools.

## Status

🚧 Early development. Corpus, benchmark, and calibration work is Chapter A. This
is a `0.0.0` placeholder release until then.

## Compute & data

Trained on Kaggle free GPUs (tabular ML at this scale is minutes per model).
Corpus and notebooks are published on Kaggle. Every dataset used is credited with
source/author/licence/link in [LEGAL.md](LEGAL.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). CLA required (one click, first PR).

## License

[AGPL-3.0-or-later](LICENSE).

## Built by

[Ayush Gupta](https://github.com/ayushgupta07xx).
