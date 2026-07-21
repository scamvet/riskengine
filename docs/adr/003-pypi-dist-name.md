# ADR-003: RiskEngine PyPI distribution name

- **Status:** Accepted
- **Date:** 2026-07-20
- **Deciders:** Ayush Gupta

## Context

RiskEngine is a `pip install`-able library. During the week-1 name claim, the
JSON API reported `pypi.org/pypi/riskengine/json` as 404 (free), but the upload
was rejected: **"The name 'riskengine' is too similar to an existing project"**
(HTTP 400). PyPI normalizes names under PEP 503 — `riskengine` folds onto the
pre-existing `risk-engine` / `risk_engine` project. The similarity guard only
fires at *upload*, not on the JSON API, so the 404 check couldn't see it. This is
a hard, permanent block.

## Decision

- **PyPI distribution name: `scamvet-riskengine`.** Install: `pip install
  scamvet-riskengine`. Import: `import scamvet_riskengine`.
- **GitHub repo stays `riskengine`** (a separate namespace with no similarity
  rule) and the **product/brand name stays "RiskEngine"** in all prose.
- The `0.0.0` placeholder is published to hold the name; the first real release
  (Chapter A) will be `0.1.0`+ (a version can never be reused or downgraded).

## Consequences

- Every `pip install` reference, the resume bullet's install string, and any
  import example must read `scamvet-riskengine` / `scamvet_riskengine`.
- The brand name `scamvet` was also claimed on PyPI as a defensive placeholder.

## Alternatives considered

- `riskengine` — blocked (above).
- `scamvet-scoring`, `scamvet-risk` — available, but `scamvet-riskengine` keeps
  the recognizable product name in the install string.
