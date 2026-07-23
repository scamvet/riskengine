# ADR-006 — Scorer architecture and what the corpus can support

- **Status:** Accepted
- **Date:** 2026-07-23
- **Deciders:** Ayush Gupta
- **Applies to:** `riskengine`, and by consumption `core`
- **Related:** ADR-004 (always-live hot path), ADR-007 (corpus licence posture), ADR-009 (metric protocol), ADR-010 (feed posture)

## Context

Chapter A was planned around "a 13M-row corpus producing calibrated fraud and
phishing scorers". Building it established three things that were not visible
from the plan.

**The four corpora are four different problems.** Bank account-opening fraud,
card transaction fraud, mobile-money transfer fraud and malicious URL
classification share no feature space, and only the last has a consumer input
surface. Nothing a ScamVet user can paste maps to the first three. There is no
single 13M-row model, and there never could have been.

**The aggregate metric was measuring the wrong thing.** On the URL corpus, the
binary target mixes phishing, malware and defacement. The latter two look
structurally wrong - IP hosts, executable extensions, odd paths - and dominate
the aggregate. Baseline logistic regression scored PR-AUC 0.9124 overall while
catching 0.41 of phishing. ScamVet's actual question is phishing.

**Three features encoded corpus provenance rather than the address.**
`had_scheme` (defacement 100% scheme-prefixed, benign 8%), `scheme_is_https`
(https rows 87.5% malicious, and only 2.4% of the corpus is https at all - the
inverse of production reality), and the per-TLD one-hots (`.ca` rows 10.6%
malicious against 33.2% corpus-wide, because the benign upstream ISCX-URL-2016
is a University of New Brunswick crawl). All three were found by reading
attributions, not by any metric: each removal *lowered* the headline number.

A provenance audit was then run rather than continuing feature-by-feature
surgery, training the same model on nested subsets:

| subset | columns | phishing R@FPR5% |
|---|---|---|
| all | 86 | 0.7973 +/- 0.0089 |
| minus TLD one-hots | 34 | 0.7898 +/- 0.0176 |
| semantic + shape | 33 | 0.7891 +/- 0.0169 |
| semantic only | 22 | 0.2295 +/- 0.0158 |

## Decision

**1. RiskEngine is a set of independently trained scorers, not one model.**
Each targets one problem, carries its own calibration and thresholds, and
declares its track. Only product-track scorers are exposed to Core.

**2. The product scorer is `url-scorer-lexical`, XGBoost.** XGBoost separates
from LightGBM on phishing recall at both 3 and 5 seeds (0.7973 vs 0.7490, gap
0.0483 against a spread of 0.0089) and calibrates better (ECE 0.0178 vs
0.0294). Both remain supported: attribution and ONNX export dispatch on the
fitted model, because a single-library assumption is a hostage to one library's
release notes.

**3. Per-TLD one-hot columns are off by default.** They contribute 0.0075
recall against a seed spread of 0.0176 - within noise - while costing 52 of 86
matrix columns and carrying the geography artifact above. `suspicious_tld` and
`tld_length` remain and carry the semantic part. Opt-in is preserved for the
research track.

**4. The reported figure is phishing recall, quoted twice.** The in-corpus
number (0.797) and the intrinsic-feature floor (0.230) are published together
with the audit JSON. Neither alone is honest: the first is inflated by a benign
population collected as deep-crawled article URLs, the second is depressed by
sparsity that inverts in the Indian market this product serves.

**5. The layered verdict stack is load-bearing, not stylistic.** Promise 2
specifies deterministic facts, then calibrated ML, then LLM reasoning. The
audit shows the ML layer cannot be the primary evidence for a phishing verdict
on lexical features alone. Deterministic facts and the LLM layer carry more
than the plan assumed, and the Chapter B golden set - real Indian scam
artifacts, scored by the full stack - becomes the product metric rather than an
eval fixture.

## Consequences

- No README, dashboard, UI string or bullet may state or imply that a shipped
  model was trained on 13M rows. The corpus figure describes what was ingested
  and benchmarked; the product scorer trains on 651,191 CC0 URLs.
- Chapter A's headline moves from PR-AUC 0.98 to phishing recall 0.797
  in-corpus / 0.230 intrinsic. Smaller and defensible.
- Four feature-spec versions exist, each recording why its predecessor died.
  The versioning discipline is the integrity record, not bookkeeping.
- The ADR-010 feed gap is now quantified rather than noted: intrinsic lexical
  signal tops out near 0.23, so enrichment is the mechanism for phishing
  coverage, not an enhancement.
- Model cards state the corpus limitation explicitly, including that some
  phishing-labelled rows are ordinary articles - the worst "misses" included
  real Ars Technica URLs - which caps achievable recall for reasons unrelated
  to the model.

## Alternatives considered

**Keep the contaminated features and report the higher number.** Rejected. It
would be measurably wrong in production, and `scheme_is_https` would have
flagged the legitimate web.

**Remove contaminated features one at a time until none are found.** Rejected
as the primary strategy: with five upstream sources and no source labels, there
is no point at which the search is provably complete. The audit bounds the
exposure instead.

**Abandon the corpus.** Rejected. It is the only permissively licensed source
available (ADR-007), it supports the benchmark, imbalance study and published
methodology honestly, and the real fix - Indian-sourced artifacts and live
enrichment - is Chapter B and Chapter F work, not a reason to stop here.

**Report only the intrinsic floor.** Rejected as over-correction. 0.230 is
depressed by feature sparsity specific to this corpus's geography; presenting
it alone would understate the model as badly as 0.797 overstates it.
