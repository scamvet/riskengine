# ADR-011: Inference and training carry different dependencies

Date: 2026-07-23
Status: Accepted

## Context

Until 0.1.0 the package declared xgboost, lightgbm and scikit-learn as core
dependencies and put onnxruntime in an `[onnx]` extra. That inverted the real
requirement. `Scorer.__init__` opens an ONNX session eagerly, so a default
install could not run the five lines printed in the package's own docstring - it
raised `ScorerError` on `load()`. Meanwhile it pulled xgboost's 132 MB wheel,
which the scoring path does not import at all.

The defect was invisible to the test suite because the development environment
always had every library installed. It surfaced only when the wheel was
installed into a venv holding nothing else.

ADR-004 (recorded in the `core` repo) commits the product to a degradation path: a local ONNX scorer must
return an instant verdict when richer tiers are unavailable. That contract is
meaningless if obtaining the ONNX scorer requires the training stack.

## Decision

Four core dependencies: tldextract, pandas, numpy, onnxruntime.

- `[explain]` adds xgboost. ONNX carries no TreeSHAP path, so exact
  attributions need the native booster. This is a *serving* dependency of the
  explanation path, not a training one - `registry/scorer.py` imports xgboost
  inside `_attributions` and `_load_booster`.
- `[train]` adds lightgbm, scikit-learn, imbalanced-learn, pyarrow and the ONNX
  exporters. Nothing here is imported by the scoring chain.
- `[dev]` is unchanged.

A verdict produced without `[explain]` sets the `explanations_unavailable` flag.
An empty `reasons` tuple would otherwise mean either "no notable attributions"
or "the library was absent", and every surface that renders a verdict reads
that tuple.

## Consequences

- `pip install scamvet-riskengine` runs the documented example. A CI job
  (`default-install` in `test.yml`) installs the built wheel with core
  dependencies only and executes it, so the regression cannot recur silently.
- Consumers wanting reasons must ask for them explicitly. This is the intended
  trade: the offline and edge paths get a small install, and the flag makes the
  reduced capability legible rather than ambiguous.
- `[explain]` currently pins only xgboost because the sole shipped model is
  XGBoost. A LightGBM model entering the registry requires revisiting this -
  `explain/contributions.py` already dispatches on both.
- `pyarrow` is uncapped in `[train]`. The previous `<22` cap was already
  violated by the installed 25.x, and pandas 3 requires pyarrow regardless. A
  research-track extra can absorb that risk; core could not.
