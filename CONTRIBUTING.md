# Contributing to ScamVet

Thanks for helping vet scams and get people's money back. ScamVet is a genuine
open product — contributions of any size are welcome.

## Before you start

- **CLA required.** This repository is licensed **AGPL-3.0-or-later**. Before your
  first pull request can be merged, the CLA Assistant bot will ask you to sign a
  Contributor License Agreement. This is a one-time step (a single click on your
  first PR). It keeps future relicensing/dual-licensing possible for the project.
- Look for issues labelled **`good first issue`** to get started.
- Open a **Discussion** first for anything larger than a bug fix, so we can agree
  on the approach before you spend time.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you push

```bash
ruff check --fix && ruff format
pre-commit run --all-files
pytest
```

All of these must be green. CI runs the same checks plus the quality gate
(golden-set precision/recall bars + zero red-team false-positive regressions);
releases block on it.

## Ground rules

- No fake metrics. Simulated/dogfooded numbers are labelled as such.
- Verdicts state uncertainty honestly. Rescue artifacts are informational, not
  legal advice, and are user-reviewed before filing.
- Every dataset/feed used gets a source/author/licence/link credit in `LEGAL.md`.
- Respect bot protection on any source: official APIs and open feeds only, never
  proxies or challenge-solving.

## Code of Conduct

Be decent. Harassment of any kind isn't tolerated.
