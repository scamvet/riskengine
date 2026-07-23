# ADR-012: Spark preprocessing, the feature store and drift monitoring move to Chapter F

Date: 2026-07-23
Status: Accepted

## Context

ScamVet.md's Chapter A work-item list names ten items. Three were never built
and appear in neither the chapter's definition of done nor the handoff: Spark
preprocessing (contract with ScamPulse), a Postgres/dbt feature store, and
drift monitoring. Closing the chapter without addressing them would drop them
silently.

## Decision

All three move to Chapter F (ScamPulse), where Spark and dbt are already in the
stack.

## Rationale

The product corpus is 651,191 rows. Pandas handles it in seconds; the research
corpora are read once per benchmark. Building a Spark surface inside
`riskengine` would duplicate infrastructure Chapter F builds properly, and the
preprocessing contract it is supposed to define has ScamPulse on the other end
- a contract with an endpoint that does not exist yet is a guess.

The same applies to the feature store: a store serving one consumer with one
feature spec is a file, and `data/derived/` already is that file.

Drift monitoring is the sharpest case. Drift is measured against something. At
Chapter A there is no traffic and no feed history, so a drift module could only
compare a static corpus to itself. In Chapter F it compares the corpus against
accumulating URLhaus snapshots and live scoring distributions, which is the
measurement the product actually needs.

## Consequences

- Chapter A's definition of done is unchanged; these were never in it.
- Chapter F's scope grows by three items. ScamVet.md is updated in the same
  batch as this ADR so the roadmap and the repo agree.
- The Analyst resume bullet already claims PySpark over the corpus through
  ScamPulse, not through RiskEngine, so no bullet changes.
