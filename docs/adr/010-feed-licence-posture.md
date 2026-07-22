# ADR-010 — Live threat-intelligence feed posture

- **Status:** Accepted
- **Date:** 2026-07-22
- **Deciders:** Ayush Gupta
- **Applies to:** `riskengine` (enrichment features), `core` (Evidence agent MCP tools), `scampulse` (ingestion DAGs)
- **Related:** ADR-004 (always-live hot path), ADR-007 (corpus licence posture)

## Context

ScamVet's plan assumed three live phishing/malware feeds — PhishTank, OpenPhish
and URLhaus — as enrichment sources for the Evidence agent, as blocklist inputs to
the degradation path, and as streaming inputs to ScamPulse.

All three are free to fetch. Reading their terms on 2026-07-22 showed that free to
fetch and free to use are unrelated properties, and that each was written for an
enterprise consuming intelligence internally — the opposite shape to a consumer
product that shows verdicts to members of the public.

**OpenPhish** Terms of Use: the Services are provided solely for personal use;
no commercial use without prior written consent; and, except as expressly
permitted in writing, no licensing, distribution, disclosure, creation of
derivative works, or making any portion of the information obtained through the
Services available to any third party.

**URLhaus** (abuse.ch website Terms of Use, effective 2025-11-04): free access is
granted to authenticated users for not-for-profit purposes under fair-use
principles; commercial or for-profit needs may require a paid subscription managed
by Spamhaus; §7.3 bars derivative works based on the Platforms or abuse.ch/
Spamhaus IP without express consent. No data licence is stated. An Auth-Key has
been mandatory since 2025-06-30.

**PhishTank**: the Terms of Use link in the site footer resolves to Cisco's
General Terms, a commercial framework governing Cisco Offers acquired under an
Order — a document that never mentions PhishTank or feeds, and under which we hold
no Order. Meanwhile the PhishTank FAQ and homepage describe an open API provided
at no charge for developers to integrate anti-phishing data into their
applications, and state that Cisco considers phishing data something to share
freely rather than compete on.

## Decision

**1. No feed is ever used as training data.** Feeds are enrichment only, queried
live through documented interfaces. This follows from the terms above, and it was
independently the right call: feed membership is the label in disguise for
feed-sourced positives, and ScamVet's layered verdict architecture already places
deterministic facts (blocklist hits) ahead of the ML score rather than inside it.

**2. OpenPhish is excluded.** Three independent clauses fail: ScamVet is a public
service rather than personal use; a derived scorer or blocklist is a derivative
work; and showing a user that a URL appears in the feed discloses information
obtained through the Services to a third party — which is the core Vet verdict,
not an incidental feature. Removed from the Evidence agent, the corpus and the
roadmap. Revisit only on express written permission.

**3. URLhaus is permitted, under recorded conditions.** ScamVet is free,
open-source and not-for-profit, so §3.1 fair use applies today. Conditions:
documented endpoints only, no scraping, no training, no republication, and no
fetching of dumps more often than every five minutes. A **commercial tripwire** is
recorded: the day ScamVet takes revenue in any form, URLhaus use must be
re-evaluated against §4 before the next release. Note that §4.1 makes data
contribution a post-hoc rebate after six months of paid subscription, not an
alternative to paying.

**4. PhishTank is pending written confirmation.** Not wired into any shipped
surface until Talos replies. Clarification requested 2026-07-22 via the site
contact form. The published intent reads as permitting our use, but intent is not
a grant. The reply is archived in the repository and `LEGAL.md` is updated in the
same commit as activation.

**5. Coverage gap is acknowledged, not hidden.** URLhaus is malware-distribution
URLs only; its own submission policy directs phishing reports elsewhere. With
OpenPhish excluded and PhishTank pending, ScamVet currently has **no permitted
live phishing feed**, and phishing is its primary case. Google Safe Browsing
Lookup API is to be evaluated in Chapter B as a candidate, on the grounds that it
is the only such service designed for a client application checking URLs on behalf
of its own users rather than for internal SOC consumption.

**6. Feed terms are re-read before every release** that changes ingestion, per
`LEGAL.md` §7. A changed term blocks the release.

## Consequences

- The Chapter B Evidence agent ships with one live feed (URLhaus, malware only)
  unless PhishTank clears or Safe Browsing is adopted. Chapter B's DoD is adjusted
  accordingly rather than being quietly missed.
- The degradation-path local blocklist cannot be seeded from feed data. It is
  built from the CC0 corpus plus our own verified detections.
- ScamPulse's `fact_feed_events` will initially carry a single source. The star
  schema is source-dimensioned already, so adding sources later is data, not
  schema change.
- The commercial conversation with Spamhaus becomes a known, dated, budgeted item
  rather than a surprise discovered after monetisation.

## Alternatives considered

**Use the feeds anyway while small and unnoticed.** Rejected outright. ScamVet's
entire trust position is that it is the honest alternative to an industry of
scam-adjacent operators. A product that tells users which links to trust cannot be
built on terms it is knowingly breaching, and "too small to notice" is not a
licence.

**Route feed lookups client-side so the user, not ScamVet, queries the feed.**
Rejected. It is a transparent circumvention of the same terms, it leaks user URLs
to third parties from the user's own IP, and it breaks the always-live promise on
poor connections.

**Treat the CC0 corpus's embedded PhishTank-derived rows as equivalent to live
feed access.** Rejected as a substitute for live access — a 2021-era snapshot has
no value against phishing domains whose median lifetime is measured in hours — but
retained as training data, since the compilation carries its own CC0 grant.

**Delay Chapter B until a phishing feed clears.** Rejected. The lexical URL scorer
requires no feed at all, and shipping it establishes the degradation path that
ADR-004 requires. Enrichment sources are additive.
