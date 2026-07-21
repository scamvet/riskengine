# Architecture — RiskEngine

> Skeleton. Full spec produced at the start of Chapter A.

## Pipeline

Public corpus (~13M rows) → Spark preprocessing (contract with ScamPulse) →
Postgres/dbt feature store → imbalance study (class weights vs SMOTE vs focal) →
model benchmark (calibrated logistic, RF, XGBoost, LightGBM; PR-AUC primary, cost
curves) → Platt/isotonic calibration + reliability diagrams → threshold tuning per
use-case → drift monitoring → SHAP (global + per-prediction) → fairness report →
model cards → versioned registry.

## Consumed by

ScamVet Core's risk-scorer agent calls registry models as tools. ShieldOps
canary-deploys new versions weekly.

## Compute

Kaggle free GPUs (tabular ML = minutes per model at this scale). Corpus +
notebooks published on Kaggle.
