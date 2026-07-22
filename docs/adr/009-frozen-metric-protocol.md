# ADR-009 — Frozen metric protocol

- **Status:** Accepted
- **Date:** 2026-07-22
- **Deciders:** Ayush Gupta
- **Applies to:** `riskengine`, and every figure published from it
- **Related:** ADR-004 (always-live hot path), ADR-007 (corpus licence posture)

## Context

RiskEngine will produce a benchmark across several model families, an imbalance
study, a calibration study and a fairness report. Each of those produces many
numbers, and the corpus has a heavily imbalanced positive class.

Under those conditions, the choice of headline metric is itself a result. If the
metric is chosen after the experiments run, the natural and almost unconscious
move is to report whichever one looks best, and neither the author nor the
reader can tell afterwards that this happened. The second of the five promises
is "it never ships a failure we didn't measure first"; a metric selected in
hindsight measures nothing.

Two concrete hazards on this data:

**ROC-AUC flatters imbalanced problems.** Measured on a synthetic 1%-positive
problem with a deliberately mediocre ranker, ROC-AUC reads 0.76 while PR-AUC
reads 0.32. Both describe the same model. Leading with the first would
overstate it by a wide margin. This is pinned as a test, not left as an
assertion in prose.

**Aggregate curve summaries hide the operating point.** A deployment does not
run at "the whole curve"; it runs at a threshold. The consumer verdict is
precision-leaning, because a false red destroys trust in a way a false amber
does not, so what constrains the product is recall achievable inside a fixed
false-positive budget.

## Decision

The protocol is frozen before any model is trained and is implemented once, in
`scamvet_riskengine.eval.metrics`. Every benchmark table, model card, README
figure and external claim reports these numbers, computed by that module.

**Primary metric:** PR-AUC (average precision).

**Operating points:** recall at FPR 1% and FPR 5%, each reported with the
threshold that realises it and the precision at that threshold. The threshold
is included because it is the number a deployment configures.

**Calibration:** Brier score and expected calibration error over 15 equal-width
bins, plus committed reliability-bin data. Calibration is a first-class metric
here, not a diagnostic, because the registry maps verdict bands from calibrated
probabilities — a model whose 0.9 does not mean 0.9 produces bands that lie.

**Cost:** expected cost per sample at a stated false-negative to false-positive
ratio, default 10:1, reported alongside the ratio. The ratio is an explicit
argument rather than a hidden default so that it must be argued about.

**Reported but never headlined:** ROC-AUC. It appears in every table for
comparability with outside work and is never the lead number.

**Enforcement decisions:**

1. `expected_calibration_error` raises on inputs outside [0, 1] rather than
   clipping them. Silently clipping a decision function into probability range
   produces a number that looks like a calibration score and is meaningless.
2. `evaluate` raises on single-class input, NaN scores, shape mismatch and
   non-binary labels, rather than returning a degraded value.
3. `ECE_BINS = 15` is a module constant. Changing it changes every published
   ECE, so it is a versioned decision, not a call-site option.
4. Splits are grouped by registered domain and assigned by hashing the group
   name, so any metric is computed on data no part of whose registered domain
   was seen in training. See `data/splits.py`.

## Consequences

- Benchmark tables are wider than a single-number leaderboard, which is the
  intent: model selection here is a trade-off between ranking quality,
  usable recall at a tolerable alarm rate, and honest probabilities.
- A model that wins on PR-AUC but calibrates badly does not automatically win,
  because band mapping depends on calibration. That trade-off is visible in the
  table instead of being resolved silently.
- The imbalance study can report its real finding — that resampling often
  improves ranking while degrading calibration — because both are measured. A
  PR-AUC-only protocol would have shown resampling as a free win.
- Adding a metric later is allowed; removing or replacing the primary metric
  requires superseding this ADR, so it cannot happen quietly mid-chapter.

## Alternatives considered

**F1 as primary.** Rejected. F1 fixes an operating point implicitly at
threshold 0.5 and weights precision and recall equally, which contradicts the
precision-leaning consumer verdict. It also hides calibration entirely.

**ROC-AUC as primary, for comparability with published work.** Rejected for the
reason measured above. It is reported for exactly that comparability and never
led with.

**Accuracy.** Rejected outright. At this positive rate, predicting "benign"
for everything scores extremely well and detects nothing.

**Choose the metric per experiment.** Rejected. That is the failure this ADR
exists to prevent.

---

## Amendment — 2026-07-23: seed variance and the separation rule

**Status:** Accepted. Extends the original decision; does not supersede it.

### What happened

A single-seed sweep on the real corpus showed a 400-tree LightGBM reaching
0.8511 phishing recall at FPR 5% and a 100-tree model reaching 0.8658. This was
read as evidence that the larger configuration overfits and the smaller one is
strictly better, and it was nearly written into the model card and the default
config on that basis.

Repeating across four seeds:

| config | s42 | s1 | s7 | s13 | mean | spread |
|---|---|---|---|---|---|---|
| 400 x 8 | 0.8511 | 0.8648 | 0.8597 | 0.8606 | 0.8591 | 0.0137 |
| 200 x 8 | 0.8656 | 0.8629 | 0.8633 | 0.8654 | 0.8643 | 0.0027 |
| 100 x 8 | 0.8658 | 0.8689 | 0.8641 | 0.8564 | 0.8638 | 0.0125 |

The gap between configurations is roughly 0.005. The seed spread within a
single configuration is roughly 0.013 — nearly three times larger. Seed 42
happened to be 400 x 8's worst draw. There is no overfitting result; there is
a fluctuation that was about to be published as a finding.

This is the failure the original ADR exists to prevent, arriving by a route it
did not anticipate. The ADR fixed *which* metric is reported so the choice
could not be made after seeing results. It said nothing about how many times a
metric is measured, which left the same freedom open in a different dimension.

### Decision

**1. Reported metrics are means with their observed range over at least three
seeds.** A bare point estimate is not publishable. This applies to every
benchmark table, model card, README figure and Canonical Numbers entry.
`scamvet_riskengine.eval.aggregate` computes it; `scripts/benchmark.py`
refuses fewer than two seeds.

**2. A difference is a finding only if it exceeds the seed spread.**
Formally, `separates(a, b)` holds when `|mean(a) - mean(b)|` is greater than
`max(spread(a), spread(b))`. Configuration and model-family choices must cite
it. The benchmark prints the pairwise verdict so the claim cannot be made
casually.

**3. Choices that do not separate are decided on other grounds, stated as
such.** `BoostedConfig.n_estimators` is 100 rather than 400 because at 400 the
ONNX export is 877 KB — over the 500 KB committed-artifact cap, which would
force a separate smaller model for the offline path — while at 100 it is
217 KB and one model serves both paths. That is a size decision. The docstring
says so, so that nobody later reconstructs the phantom quality argument.

### Consequences

- Benchmarks cost three times the compute. On this corpus that is a few
  minutes, which is not a reason to prefer a number we cannot support.
- Some previously interesting-looking differences will dissolve. That is the
  point.
- The imbalance study inherits this: "resampling improved recall" needs
  multi-seed evidence or it is not a result.
- `MetricSpread.spread` is the observed range rather than a standard
  deviation. Chosen for legibility at three to five seeds, where a standard
  deviation over so few observations is itself noisy and invites false
  precision. If seed counts grow, revisit.

### What this does not claim

Seed spread is not a confidence interval and no significance test is implied.
It is a floor: differences below the observed variation are not evidence.
Differences above it are worth investigating, not proven.
