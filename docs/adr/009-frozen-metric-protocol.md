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
