# Imbalance study

Whether the standard class-imbalance toolkit does anything measurable to
`url-scorer-lexical`, and if not, at what imbalance it would start to.

Reproduce: `python -u scripts/imbalance_study.py --ratios 100 --seeds 42 1 7 13
21 3 99 5 17 8`. Raw output in `data/derived/imbalance_study.json`.

## Summary

At the ratio the model actually trains on — 31.97% positive in the training
fold, one positive per 2.13 negatives — **no intervention separates from the
untreated baseline on any reported metric.** Class weighting, SMOTE and focal
loss at two gammas are all indistinguishable from doing nothing.

Thinning the positive class to 1:100 makes two effects appear. SMOTE damages
ranking. Focal loss improves calibration and leaves ranking alone. Neither
affects the shipped model, which trains at the natural ratio, but both matter
for the far sparser positive rates that live threat feeds will bring.

**Decision: the shipped configuration is unchanged.** That is now a measured
choice rather than a default.

## Method

One model family (the shipped XGBoost, 200 trees, depth 8), the same grouped
splits, the same Platt calibrator, the frozen metric protocol of ADR-009. Only
the treatment of class balance varies across arms.

| Arm | Treatment |
|---|---|
| `baseline` | none |
| `scale_pos_weight` | XGBoost class weighting (`imbalance="balanced"`) |
| `smote` | SMOTE oversampling, training fold only, after the grouped split |
| `focal_g1`, `focal_g2` | focal loss, custom objective, gamma 1 and 2 |

Every arm is calibrated identically before evaluation. Focal loss deliberately
distorts probability scale, so scoring a raw focal arm against a calibrated
baseline would manufacture a difference belonging to the measurement rather
than to the model.

SMOTE runs on the training fold only, after grouped splitting, so no synthetic
row can carry information across an eTLD+1 boundary into calibration or test.

Ten seeds per arm per condition. Reported as mean +/- observed range. A
difference counts only if it exceeds the larger of the two arms' ranges.

## Results

Phishing view, 68,180 rows, 18.63% positive.

### Natural training ratio (1:2.13)

| Arm | R@FPR5% | PR-AUC | ECE |
|---|---|---|---|
| baseline | 0.6996 +/- 0.0234 | 0.8123 +/- 0.0044 | 0.0658 +/- 0.0164 |
| scale_pos_weight | 0.6949 +/- 0.0209 | 0.8129 +/- 0.0080 | 0.0659 +/- 0.0113 |
| smote | 0.6921 +/- 0.0241 | 0.8096 +/- 0.0051 | 0.0653 +/- 0.0070 |
| focal_g1 | 0.6980 +/- 0.0190 | 0.8137 +/- 0.0059 | 0.0662 +/- 0.0103 |
| focal_g2 | 0.7018 +/- 0.0173 | 0.8138 +/- 0.0052 | 0.0645 +/- 0.0097 |

Nothing separates, on any metric. The largest recall gap is 0.0075 against a
bar of 0.0241.

### Positives thinned to 1:100

| Arm | R@FPR5% | PR-AUC | ECE |
|---|---|---|---|
| baseline | 0.6379 +/- 0.0354 | 0.7686 +/- 0.0192 | 0.0981 +/- 0.0179 |
| scale_pos_weight | 0.6359 +/- 0.0486 | 0.7598 +/- 0.0257 | 0.0821 +/- 0.0306 |
| smote | 0.6067 +/- 0.0417 | 0.7421 +/- 0.0168 | 0.0905 +/- 0.0238 |
| focal_g1 | 0.6508 +/- 0.0399 | 0.7763 +/- 0.0191 | 0.0703 +/- 0.0209 |
| focal_g2 | 0.6536 +/- 0.0305 | 0.7762 +/- 0.0181 | 0.0695 +/- 0.0258 |

Three differences clear their bar:

| Comparison | Gap | Bar | Verdict |
|---|---|---|---|
| smote PR-AUC | -0.0265 | 0.0192 | separates, worse |
| focal_g1 ECE | -0.0277 | 0.0209 | separates, better |
| focal_g2 ECE | -0.0286 | 0.0258 | separates, better (narrowly) |

Recall does not separate for any arm at any ratio. Focal's +0.0129 and +0.0156
are directionally consistent across both gammas but stay inside the noise.

## Findings

**1. The toolkit is inert at the ratio it gets applied to.** At 1:2.13 nothing
moves. This is the result most worth stating plainly, because reaching for
SMOTE on a 32%-positive problem is a reflex rather than a diagnosis.

**2. SMOTE damages ranking once positives are sparse.** PR-AUC falls 0.0265 at
1:100, and the mechanism is visible in the resampling report: the share of
synthetic rows carrying non-integer `brand_min_distance` values rises with
sparsity — 1.2% at natural, 2.4% at 1:20, 3.3% at 1:100. Edit distance is
integer-valued, so those rows sit off the data manifold. Fewer minority points
mean sparser neighbours and longer interpolation segments, so the technique
manufactures more impossible rows exactly where it is supposed to help most.

**3. Focal loss improves calibration under severe imbalance, not ranking.**
Baseline ECE degrades as positives thin (0.0658 -> 0.0981); focal's does not
(0.0645 -> 0.0695). Both gammas separate. The mechanism is plausible: focal
down-weights the confident majority, which is what distorts probability
estimates as the majority grows. **This was not predicted.** It was noticed in a
pilot table and then confirmed at higher seed count, which is weaker evidence
than a pre-registered test, and is recorded here as such.

## The prediction that was wrong

The study pre-registered, in the script docstring, that SMOTE would hurt
because `brand_min_distance` returns a sentinel 64 when no brand is near:
interpolating an active 2 against an inactive 64 should yield 33, a distance no
real URL can have.

**It does not happen. 0.0% of synthetic rows land in that gap, at every ratio.**
SMOTE interpolates between a point and its nearest *minority* neighbours, and a
61-unit gap in one dimension dominates the distance metric — so sentinel rows
neighbour sentinel rows and active rows neighbour active rows, and the pairing
that would produce a 33 is essentially never selected. The prediction also
assumed the largest real active value was around 20; it is 3.

The damage SMOTE does is real but arrives by the smaller mechanism in finding
2, not the one predicted.

## A limitation of the separation rule

ADR-009's rule uses the observed range across seeds as the bar a difference
must clear. Observed range grows with the number of seeds — more draws, more
chance of an extreme — so **the rule becomes stricter as evidence accumulates**,
which inverts the usual relationship between sample size and power. Baseline
recall reads 0.7055 +/- 0.0112 over three seeds and 0.6996 +/- 0.0234 over ten:
the same configuration, the same data, a range twice as wide.

Two consequences:

- **`+/-` values are not comparable across runs with different seed counts.**
  Within a single run every arm shares the same n, so the comparison is fair,
  which is why the findings above stand. Across runs it is a category error.
- The three findings here cleared a bar that grew when the seed count grew,
  which makes them stronger than the same result at three seeds would have
  been.

A bar that does not grow with n — standard error of the mean, or a bootstrap
interval — would behave better. Changing it is not a free edit: the
`n_estimators` 200-versus-400 decision recorded in `BoostedConfig` was made
under the current rule and would need revisiting. Recorded here as a known
limitation pending that decision.

## What this means downstream

The shipped model trains at the natural ratio, where nothing helps, so nothing
changes.

Live threat feeds are a different regime. A stream where confirmed phishing is
rare against ordinary traffic sits far closer to 1:100 than to 1:2.13, and at
that ratio two things from this study apply: do not reach for SMOTE, and if
calibrated probabilities matter — they do, because verdict bands are thresholds
on a probability — focal loss at gamma 2 is the intervention with measured
support.

## Note on the first run

The initial execution of both focal arms produced PR-AUC exactly equal to the
base rate, recall 0.0000, and zero variance across seeds. That was a harness
fault, not a result: `base_score` had been set to 0.0, which XGBoost reads as a
probability for a classifier, starting every example deep in the saturated tail
where focal curvature collapses. Measured on an identical fit, the hessian sum
was 0.1 against a `min_child_weight` of 5.0 — no node could form, every tree
was a bare leaf. Removing the override restored 2,497 splits.

The script now aborts when an arm gives every test row the same score, because
a degenerate arm that reaches the results table looks exactly like a finding
about the technique it was supposed to test.
