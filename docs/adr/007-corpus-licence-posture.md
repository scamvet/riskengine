# ADR-007 — Corpus licence posture: research track and product track

- **Status:** Accepted
- **Date:** 2026-07-22
- **Deciders:** Ayush Gupta
- **Applies to:** `riskengine`, and by consumption `core`
- **Related:** ADR-001 (licensing & IP), ADR-010 (feed licence posture)

## Context

RiskEngine was planned around four public corpora: Bank Account Fraud (BAF,
~6M rows), IEEE-CIS Fraud Detection (590,540), PaySim (6,362,620) and a malicious
URL corpus (651,191). ScamVet is AGPL-3.0 with documented commercial intent
(ADR-001: open-core, possible future hosted API, BSL fallback).

Licence verification on 2026-07-22 established:

| Corpus | Licence | Commercial use | Redistribution |
|---|---|---|---|
| BAF | CC BY-NC-SA 4.0 | Prohibited | Under NC + SA |
| IEEE-CIS | Kaggle competition rules §7 | Prohibited | **Expressly prohibited** |
| PaySim | CC BY-SA 4.0 | Permitted | Under SA |
| Malicious URLs 651K | CC0 1.0 | Permitted | Permitted |

Two of the four are non-commercial. One expressly forbids providing the data to
any party not participating in the competition.

A second, independent fact pushes the same way. The four corpora describe four
different problems — bank account-opening fraud, card-not-present transaction
fraud, mobile-money transfer fraud, and malicious URL classification. None of the
first three has a consumer input surface: nothing a ScamVet user can paste maps to
their feature spaces. Only the URL corpus produces a model the product can call.

So the licence constraint and the architecture constraint agree. The corpora that
cannot be used commercially are the same corpora that cannot serve users anyway.

## Decision

RiskEngine maintains two strictly separated tracks.

**Research track.** May use NC and competition-licensed sources. Produces
published benchmarks, the fairness analysis across BAF's six bias-controlled
variants, the imbalance-vs-calibration study, and Kaggle notebooks. Models trained
here carry `track: research` in their manifest and are **never** referenced by
Core or shipped to users.

**Product track.** Trains only on permissively licensed sources. Currently this
means the CC0 URL corpus alone. Models carry `track: product`, are the only models
the registry exposes to Core, and are the only models exported to ONNX for the
always-live degradation path.

Specific consequences:

1. IEEE-CIS is never re-hosted by us in any form — not as derived parquet, not as
   feature extracts, not as a Kaggle dataset upload. Notebooks may attach it
   natively on Kaggle, where the data does not leave the platform.
2. BAF and PaySim derivatives, where published, respect share-alike.
3. The Tranco default list is not used in product models, because its Cloudflare
   Radar component is CC BY-NC 4.0. The product scorer pins a custom Tranco list
   configured to exclude that provider.
4. Every source, its licence, its read date and its track are recorded in
   `LEGAL.md`. A source absent from `LEGAL.md` may not be used in a committed
   artifact.
5. Public claims distinguish three figures that must never be conflated: the
   ~13.6M-row research corpus, the 651,191-row product training set, and the fact
   that ~12.4M of the research corpus is synthetic by construction.

## Consequences

**Accepted costs.**

- The product scorer trains on 651K rows, not 13.6M. Every external claim must be
  precise about this. The larger figure remains true of the corpus that was
  ingested, engineered and benchmarked, and false of any single model.
- The "400+ engineered features" depth from IEEE-CIS is a research claim only.
- The product track has a single training source, so a defect or bias in that
  source has no counterweight. Mitigated by the golden eval set and red-team suite
  in Chapter B, which are independent of it.

**Gains.**

- The commercial path in ADR-001 stays open with no licence entanglement, because
  nothing NC-licensed is ever shipped.
- The research work is not merely permitted but is what these datasets exist for.
  BAF's fairness variants are its stated purpose; publishing that analysis is
  aligned with the licence, not tolerated by it.
- The split is auditable in one file and one manifest field.

**Enforcement.** The `track` field is required in every model manifest. Registry
resolution refuses to return a `track: research` model to a product caller. This
is a test, not a convention.

## Alternatives considered

**One blended model over all four corpora.** Rejected on both grounds. It is
technically incoherent — the corpora share no feature space — and it would make
every shipped model a derivative of NC data.

**Ship NC-trained models while the project is free, revisit on monetisation.**
Rejected. It requires retraining and re-validating the entire registry under
deadline pressure at exactly the moment the project is least able to absorb it,
and it puts an NC dependency into artifacts already distributed to users.

**Drop the NC corpora entirely.** Rejected. It discards the fairness analysis,
which is the most distinctive research output available to this project, for no
gain — research use is explicitly permitted.

**Substitute a permissive transaction-fraud corpus for IEEE-CIS.** Deferred, not
rejected. Unnecessary because IEEE-CIS is research-track and permitted there.
Revisit only if a product-track transaction scorer is ever needed.
