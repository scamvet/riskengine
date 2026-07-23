# Architecture Decision Records

Significant, hard-to-reverse decisions are recorded here as ADRs — one file per
decision, numbered sequentially, committed inline with the code that implements
them. Use `adr-template.md` as the starting point.

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Licensing & IP (AGPL + CLA + brand claims) | Accepted |
| 002 | Product centering — Rescue wing is the hero | Accepted |
| 003 | RiskEngine PyPI distribution name | Accepted |
| 004 | Always-live hot-path architecture | Accepted |
| 005 | Kavach motif is a lens, not a shield | Accepted |
| 006 | Scorer architecture and corpus limits | Accepted |
| 007 | Corpus licence posture — research vs product track | Accepted |
| 008 | Registry manifest and ScoreResult contract | Accepted |
| 009 | Frozen metric protocol | Accepted |
| 010 | Live threat-intelligence feed posture | Accepted |
| 011 | Inference and training carry different dependencies | Accepted |
| 012 | Spark, feature store and drift monitoring move to Chapter F | Accepted |

ADRs 001-005 are org-wide decisions whose files live in the `core` repo;
they are listed here so the numbering has no unexplained gaps. The files
present in this directory are the ones that govern the library directly.
