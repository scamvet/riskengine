# LEGAL.md — data sources, licences and attribution

Every dataset, feed and reference list used by RiskEngine is recorded here with its
source, author, licence, link, the date its terms were read, whether we may
redistribute it, and which side of the research/product split it sits on.

**A source that is not listed here may not be used in a committed artifact.**

Terms were read on the dates shown. Terms change — see "Review cadence" at the end.

Not legal advice. This file records our reading of published terms and the
conservative posture we adopted; it is not a legal opinion.

---

## 1. How to read this file

RiskEngine keeps two strictly separated tracks. The split is required by licence
terms, not by preference, and is recorded as ADR-007.

| Track | What it is | Licence requirement |
|---|---|---|
| **Research** | Published benchmarks, fairness analysis, imbalance study, Kaggle notebooks. Outputs are reports and notebooks. | May use non-commercial (NC) sources. Models trained here are **never** shipped in the registry Core consumes. |
| **Product** | Scorers in the model registry, consumed by ScamVet Core and shipped to users. | **Permissively licensed sources only.** No NC, no competition data, no feed-derived training data. |

A model card states which track its model belongs to. A model trained on any
research-track source must carry `track: research` in its manifest and must not
be referenced by Core.

---

## 2. Row counts and what they mean

| Figure | Value | Meaning |
|---|---|---|
| Research corpus | ~13.6M rows | BAF + IEEE-CIS + PaySim + URL corpus, combined, as ingested and benchmarked |
| Product training set | 651,191 rows | The CC0 URL corpus only |
| Predominantly synthetic | ~12.4M of ~13.6M | BAF (CTGAN + differential privacy) and PaySim (simulator) are synthetic by construction |

These three figures must never be conflated. In particular: no README, dashboard,
UI string or external claim may state or imply that a single model was trained on
13M rows, or that 13M rows of real-world fraud were used. The corpus figure
describes what was ingested, engineered and benchmarked across the registry.

---

## 3. Training corpora

| # | Source | Author / publisher | Licence | Rows | Redistribute? | Track |
|---|---|---|---|---|---|---|
| C1 | Bank Account Fraud (BAF) Suite, NeurIPS 2022 | Feedzai Research | CC BY-NC-SA 4.0 | ~6,000,000 | No | Research |
| C2 | IEEE-CIS Fraud Detection | IEEE Computational Intelligence Society; data by Vesta Corporation; hosted by Kaggle | Kaggle competition rules (non-commercial) | 590,540 | **No — expressly prohibited** | Research |
| C3 | Synthetic Financial Datasets For Fraud Detection (PaySim) | Edgar Lopez-Rojas | CC BY-SA 4.0 | 6,362,620 | Yes, with attribution + share-alike | Research |
| C4 | Malicious URLs dataset (651,191 URLs) | Manu Siddhartha (Kaggle: sid321axn) | CC0 1.0 Public Domain | 651,191 | Yes | **Product** |

### C1 — Bank Account Fraud (BAF)

- Link: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- Supporting repo: https://github.com/feedzai/bank-account-fraud
- Paper: https://arxiv.org/abs/2211.13358
- Licence read: 2026-07-22, from the Kaggle data card licence field.
- **NC**: commercial use not permitted. **SA**: derivatives share alike.
- Synthetic: generated with differential-privacy noise addition, feature encoding
  and a CTGAN generative model over an anonymised real-world dataset.
- Protected attributes (age group, employment status, income percentile) support
  the fairness analysis across all six bias-controlled variants.
- Use: research track only. Fairness report, imbalance study, benchmark.

Citation:

```bibtex
@article{jesusTurningTablesBiased2022,
  title   = {Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author  = {Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e}
             and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```

### C2 — IEEE-CIS Fraud Detection

- Link: https://www.kaggle.com/competitions/ieee-fraud-detection
- Rules read: 2026-07-22, Competition Rules section 7 (Competition Data).
- §7.A Data Access and Use: the Competition Data may be accessed and used for
  non-commercial purposes only, including participation in the Competition, use on
  Kaggle.com forums, and academic research and education.
- §7.B Data Security: participants agree not to transmit, duplicate, publish,
  redistribute or otherwise provide or make available the Competition Data to any
  party not participating in the Competition.
- Consequences for us:
  - Research track only. Never in the product registry.
  - **Never re-hosted by us in any form**, including derived parquet, feature
    extracts, or a Kaggle dataset upload.
  - Notebooks that attach the competition data natively on Kaggle are acceptable,
    because the data never leaves Kaggle's platform.
- This is the source of the "400+ engineered features" line in project documents.
  That claim is a research-track claim and is only valid in that context.

### C3 — PaySim

- Link: https://www.kaggle.com/datasets/ealaxi/paysim1
- Licence read: 2026-07-22, from the Kaggle data card licence field.
- Synthetic output of the PaySim mobile-money simulator, scaled to 1/4 of the
  original, derived from one month of logs from a mobile money service in an
  African country.
- **Known leakage hazard, documented by the dataset itself:** fraudulent
  transactions are annulled, so `oldbalanceOrg`, `newbalanceOrig`,
  `oldbalanceDest` and `newbalanceDest` act as label shortcuts. These four columns
  are dropped in the feature spec. Any future re-addition must cite this line.

Citation:

```bibtex
@inproceedings{lopezrojas2016paysim,
  title     = {PaySim: A financial mobile money simulator for fraud detection},
  author    = {Lopez-Rojas, Edgar Alonso and Elmir, Ahmad and Axelsson, Stefan},
  booktitle = {28th European Modeling and Simulation Symposium (EMSS)},
  address   = {Larnaca, Cyprus},
  year      = {2016}
}
```

### C4 — Malicious URLs dataset (product training set)

- Link: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
- Licence read: 2026-07-22, data card licence field: CC0 1.0 Public Domain.
- Composition: 651,191 URLs — 428,103 benign, 96,457 defacement, 94,111 phishing,
  32,520 malware. Two columns: `url`, `type`.
- **This is the only source permitted in the product registry.**

**Upstream provenance — read this before relying on the CC0 grant.** The uploader's
data card states the set was curated from five sources. A CC0 dedication by a
compiler does not by itself clear rights in compiled upstream material, and at
least one upstream carries its own citation expectation. We therefore credit all
five, and treat the CC0 declaration as the uploader's grant over their compilation:

| Upstream | Note |
|---|---|
| ISCX-URL-2016 | University of New Brunswick, Canadian Institute for Cybersecurity. Cite per CIC's dataset citation policy. |
| Malware domain blacklist dataset | As linked from the data card. |
| Benign URL set from a public GitHub repository ("faizan git repo") | As linked from the data card. |
| PhishTank dataset | Historical snapshot embedded in the compilation. Distinct from our live PhishTank feed question (see F2). |
| PhishStorm dataset | As linked from the data card. |

Open item: verify each upstream's own terms before the first public Kaggle
publication that includes derived URL features. Tracked as a Chapter A issue.

---

## 4. Live feeds

Feeds are consumed at query time for enrichment. **No feed is used as training
data.** Rationale in ADR-010.

| # | Feed | Operator | Terms | Status |
|---|---|---|---|---|
| F1 | URLhaus | abuse.ch AG (CH), with Spamhaus Technology Ltd as primary licensee | Website Terms of Use, effective 2025-11-04 | **Permitted** — not-for-profit use, live lookups only |
| F2 | PhishTank | Cisco Talos Intelligence Group | Unclear; footer link resolves to Cisco General Terms | **Pending written confirmation** — requested 2026-07-22 |
| F3 | OpenPhish Community Feed | OpenPhish | Terms of Use | **Excluded** — incompatible |

### F1 — URLhaus

- API docs: https://urlhaus.abuse.ch/api/
- Terms: https://abuse.ch/ (Website Terms of Use, effective and last updated 2025-11-04)
- Terms read: 2026-07-22.
- Auth-Key required since 2025-06-30; obtained free at https://auth.abuse.ch/.
  Account has two auth providers linked, per abuse.ch guidance.
- §3.1 Fair Use: authenticated users may access the platforms **for
  not-for-profit purposes**, subject to query volume limits.
- §4 Commercial Use: use by companies, networks or individuals with commercial or
  for-profit needs may require a paid subscription managed by Spamhaus, under
  separate terms. §4.1: a commercial user qualifying for free use must contribute
  consistently over six months **and** subscribe and pay via Spamhaus during that
  period, with a review after six months. Contribution is a post-hoc rebate, not
  an alternative to paying.
- §7.3 bars copying, adapting, modifying or making derivative works based on the
  Platforms or abuse.ch/Spamhaus intellectual property without express consent,
  and bars high-volume automated harvesting outside the documented interfaces.
- **No data licence is stated.** These terms grant access under fair use; they do
  not grant redistribution or derivative rights over the data.
- Our posture: live API lookups through documented endpoints only; never train on
  it; never republish it; respect the 5-minute minimum interval on dump fetches.
- **Coverage limit, important:** URLhaus is malware-distribution URLs only. Their
  submission policy states that phishing sites and phishing kits should not be
  submitted because phishing is not malware, and directs reporters to APWG,
  PhishTank or Netcraft. URLhaus therefore provides no coverage of ScamVet's
  primary case.
- **Commercial tripwire:** the day ScamVet takes revenue in any form, F1 requires
  re-evaluation against §4 before the next release.

### F2 — PhishTank

- Site: https://phishtank.org/
- Developer info: https://phishtank.org/developer_info.php
- Read: 2026-07-22.
- The Terms of Use link in the site footer resolves to Cisco's General Terms
  (Controlled Doc. EDCS-24218913 v6.0), which govern "Cisco Offers" acquired
  through an Order from an Approved Source. That document does not mention
  PhishTank, phishing data, or feeds, and we hold no Order.
- The PhishTank FAQ and homepage state that Cisco believes phishing data is not a
  place to be competitive, that sharing it freely — including with those who do
  not contribute — benefits everyone, and that PhishTank provides an open API for
  developers and researchers to integrate anti-phishing data into their
  applications at no charge.
- Assessment: the mismatch reads as a stale footer link rather than a prohibition.
  Intent appears to permit our use. Intent is not a grant, so we asked.
- Clarification requested via https://phishtank.org/contact.php on 2026-07-22.
  Reply, when received, is archived at `docs/legal/phishtank-response.md` and this
  row updated in the same commit.
- Until then: not wired into any shipped surface.
- Operational notes for when it is enabled: descriptive `User-Agent` is required
  (`phishtank/<username>`); HTTP 509 on over-limit; with an application key,
  unlimited HEAD requests may be used to check the `ETag` on the hourly export, so
  our polling pattern is hourly conditional fetch, not per-request lookup.

### F3 — OpenPhish Community Feed — EXCLUDED

- Terms: https://openphish.com/terms.html
- Read: 2026-07-22.
- Rules of Conduct: the Services are provided solely for personal use; users agree
  not to use any part of the Services for any commercial purpose without prior
  written consent; and, except as expressly permitted in writing, not to license,
  sell, rent, lease, transfer, assign, distribute, display, disclose, create
  derivative works, or otherwise make any portion of the information obtained
  through the Services available to any third party.
- Three independent incompatibilities with ScamVet:
  1. ScamVet is a public service, not personal use, even while free.
  2. A scorer or blocklist derived from the feed is a derivative work.
  3. Showing a user that a URL appears in the feed discloses information obtained
     through the Services to a third party — that is the core Vet verdict.
- Excluded from the Evidence agent, the corpus, and the roadmap. Revisit only on
  express written permission from OpenPhish.

---

## 5. Reference data

### R1 — Tranco

- Link: https://tranco-list.eu/
- Read: 2026-07-22.
- Used for: domain popularity rank as an enrichment feature, and as a source of
  hard negatives.
- **Constraint on the default list.** Tranco's default list aggregates five
  providers: Cisco Umbrella (free of charge), Majestic (CC BY 3.0), Farsight
  (default list only), Chrome User Experience Report (CC BY-SA 4.0), and
  **Cloudflare Radar (CC BY-NC 4.0)**. The Cloudflare Radar component imports a
  non-commercial constraint into the default list.
- Our posture: the product scorer uses a **custom Tranco list configured to
  exclude Cloudflare Radar**, and the permanent list ID of that configuration is
  pinned in the model manifest for reproducibility. The default list may be used
  on the research track.
- Access: `tranco` Python package, the permanent-URL download, or BigQuery tables
  `tranco.daily.daily` and `tranco.list_ids.list_ids`.
- Precedent worth noting: URLhaus excludes hostnames belonging to Tranco Top 1M
  domains from its RPZ and hostfile exports to reduce false positives.

Citation:

```bibtex
@inproceedings{lepochat2019tranco,
  title     = {Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation},
  author    = {Le Pochat, Victor and Van Goethem, Tom and Tajalizadehkhoob, Samaneh
               and Korczy{\'n}ski, Maciej and Joosen, Wouter},
  booktitle = {Proceedings of the 26th Annual Network and Distributed System Security Symposium (NDSS)},
  year      = {2019}
}
```

---

## 6. Open items

| Item | Owner | Blocking |
|---|---|---|
| PhishTank clarification reply | Ayush | F2 activation |
| Verify the five C4 upstream terms | Chapter A | First public Kaggle publication of derived URL features |
| Evaluate Google Safe Browsing Lookup API as a phishing enrichment source | Chapter B | Phishing coverage on the live path |
| `scampulse/LEGAL.md` mirrors the feed rows | Chapter F | ScamPulse first feed DAG |

---

## 7. Review cadence

- Every source row is re-read **before each release** that changes what is ingested.
- Any row whose terms have changed blocks the release until the row and any
  affected code are updated in the same commit.
- Terms pages are versioned by "read on" date above, not assumed stable. The
  abuse.ch terms were last revised 2025-11-04; the URLhaus Auth-Key requirement
  landed 2025-06-30. Both post-date material we had previously assumed current.

**Standing lesson recorded here because it cost us a chapter's first week:**
free-to-fetch is not free-to-use. Threat-intelligence feeds are licensed for
internal security operations, not for redistribution to consumers, and a public
URL reachable without a key carries no implied grant.
