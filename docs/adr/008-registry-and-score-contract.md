# ADR-008 — Registry manifest and the ScoreResult contract

- **Status:** Accepted
- **Date:** 2026-07-23
- **Deciders:** Ayush Gupta
- **Applies to:** `riskengine`, consumed by `core`, `scampulse`, and the clients
- **Related:** ADR-004 (always-live), ADR-006 (scorer architecture), ADR-007 (corpus licence posture), ADR-009 (metric protocol)

## Context

Chapter A produced a scorer that is correct but not consumable: training
scripts, a parquet, an ONNX blob and a set of JSON reports. Core cannot import
that. It needs one call that takes a URL and returns a verdict, and it needs the
verdict shape to be stable enough that the PWA, the Telegram bot, the Android
app and the warehouse can all render the same object.

Three failure modes shaped the design, all of them observed during this chapter
rather than imagined.

**Feature spec versions churn.** Four came and went in a fortnight, each
removing something that turned out to encode corpus provenance. A model trained
under v2 and fed v4 features would produce confident, wrong verdicts with
nothing in any metric to reveal it.

**Some models must never be served.** ADR-007 splits research-track models
(non-commercially licensed corpora) from product-track ones. That split is only
worth anything if it is enforced at load rather than remembered by a person.

**Thresholds are not a model property.** The consumer verdict is
precision-leaning because a false red destroys trust in a way a false amber does
not; a pre-payment QR check can afford to warn more freely. The same calibrated
probability serves both.

## Decision

**1. Every shipped model carries a manifest.** `manifest.json` records name,
version, track, corpus and its LEGAL.md licence row, feature spec version,
featurizer state, model family and config, artifact hash and ONNX parity,
calibration state, per-use-case thresholds, multi-seed metrics, and whether the
scorer needs the network. A model file without one cannot be reproduced,
audited or safely consumed, so the loader refuses it.

**2. `ScoreResult` is frozen.** Its fields are `url`, `probability`, `band`,
`reasons`, `uncertainty`, `flags`, `use_case`, `model_name`, `model_version`,
`feature_spec_version`, `requires_network`. Any change is a breaking semver bump
and triggers the cross-surface audit, because a field present on one surface and
absent on another is precisely the drift the parity rule exists to catch.

**3. Two guards run at load, not by convention.** A research-track manifest
raises. A `feature_spec_version` that differs from the one this build produces
raises, and says why.

**4. Inference runs on ONNX, not the training library.** The Python path scores
through the same artifact the browser and the Android app use, so a verdict
cannot differ by client. It is also lighter: onnxruntime is an 18 MB wheel
against xgboost's 132 MB, which means the training libraries can move to a
`[train]` extra and `pip install scamvet-riskengine` stays cheap.

**5. Reasons are best-effort, bands are not.** TreeSHAP needs the native model,
so a registry entry ships `model.json` alongside `model.onnx`. If it is absent,
scoring still returns a band and `reasons` is empty. A verdict without an
explanation is degraded; a deployment that cannot produce a verdict is broken,
and ADR-004 does not permit the second.

**6. `uncertainty` is boundary distance, not confidence.** Bands discretise a
continuous score, so a probability a thousandth from a cut is not meaningfully
different from one just across it. The field says how close the call was and
nothing more; it is explicitly not a confidence interval, and the docstring says
so to stop it being read as one.

**7. Model cards are generated, never hand-written.** `scripts/publish_model.py`
writes the card from the same run that produced the artifact, so the numbers
cannot drift from the model. The card states the corpus limitations - provenance
contamination, the intrinsic-feature floor, label noise, the sparsity of the
India-tuned features - as prominently as the headline metric, because that is
the most important thing a reader needs in order to use the number correctly.

## Consequences

- Core imports `load()` and `ScoreResult` and needs nothing else from this
  package. Everything under `features`, `models`, `eval` and `explain` remains
  public for reproducibility, not because consumers should assemble a scorer.
- Registry entries are directories, so a model, its manifest and its card are
  impossible to separate by accident.
- The registry root is overridable via `SCAMVET_MODEL_REGISTRY`, so deployments
  can mount models separately from code without a repackage.
- Version resolution supports `==`, `>=`, `<=`, `~=` and `^` and nothing more. A
  registry needing a dependency resolver to answer "which model do I load" has a
  design problem.
- Publishing is a script, not a manual step, which is what makes the card's
  claims traceable to a commit.

## Alternatives considered

**Pickle or joblib artifacts.** Rejected. They tie the runtime to the training
library and its version, cannot be loaded in a browser, and are an arbitrary
code-execution surface for a file the product downloads.

**Thresholds baked into the model.** Rejected. It would require retraining to
change a product decision, and would make one model per use case.

**A model server instead of a library.** Rejected for this chapter: it
contradicts the offline path in ADR-004 and adds a network hop to the hot path.
A hosted API remains viable later as a *consumer* of this registry.

**Emitting rendered sentences instead of reason tokens.** Rejected. Every
wording change would become a library release, and Hindi rendering would live in
the wrong repository.
