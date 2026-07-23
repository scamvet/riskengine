"""The five lines in the package docstring, run against a default install.

This exists because `pip install scamvet-riskengine` once shipped without
onnxruntime while `Scorer.__init__` opened an ONNX session eagerly, so the
documented entry point raised on the documented install command. No unit test
could catch it: the dev environment always had the training extras present.

Assertions are deliberately loose on band boundaries and strict on direction.
A known-legitimate bank domain must never come back red, and an obvious
credential-harvest URL must never come back green - that is the failure class
that the scheme-prefix defect produced, and it stays gated here.
"""

import scamvet_riskengine as r

scorer = r.load("url-scorer-lexical")
legit = scorer.score("https://bankofbaroda.com/")
phish = scorer.score("http://sbi-kyc-verify.xyz/login/update")

assert legit.band != "red", f"known-legitimate domain scored red: {legit}"
assert phish.band != "green", f"credential-harvest URL scored green: {phish}"
assert "explanations_unavailable" in legit.flags, (
    "a default install carries no native model, so an empty reasons list must be "
    f"flagged rather than left indistinguishable from no findings: {legit.flags}"
)

print("version:", r.__version__)
print("legit :", legit.band, round(legit.probability, 4), "| flags", legit.flags)
print("phish :", phish.band, round(phish.probability, 4), "| reasons", len(phish.reasons))
