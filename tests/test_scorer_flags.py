"""Flags must distinguish 'nothing found' from 'could not compute'.

An empty ``reasons`` tuple is ambiguous on its own: it means either the model
found no notable attributions, or the native model needed for TreeSHAP was not
installed. Every surface that renders a verdict reads this tuple, so the
difference has to travel in the contract rather than in the caller's guess.
"""

from scamvet_riskengine import load


def test_absent_attributions_are_flagged(monkeypatch):
    scorer = load("url-scorer-lexical")
    monkeypatch.setattr(scorer, "_attributions", lambda X: None)
    result = scorer.score("https://example.com/")
    assert result.reasons == ()
    assert "explanations_unavailable" in result.flags


def test_present_attributions_are_not_flagged():
    scorer = load("url-scorer-lexical")
    result = scorer.score("http://sbi-kyc-verify.xyz/login/update")
    assert "explanations_unavailable" not in result.flags
    assert result.reasons, "native model is installed, so reasons should be present"


def test_manifest_records_featurizer_env():
    scorer = load("url-scorer-lexical")
    env = scorer.manifest.featurizer_env
    assert env.get("tldextract"), f"no tldextract version recorded: {env}"
    assert env.get("recorded") in ("captured-at-publish", "backfilled-2026-07-23")
