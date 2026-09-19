# Confirmatory results, 19 September 2026

Pre-registration frozen the same day (commit `fa8f718`) **before** any test-window number was computed. Analysis: `results/analysis/CONFIRMATORY/`, produced by

```
python scripts/08_analyze.py --windows test --n-boot 1000 --name CONFIRMATORY \
    --shortcut results/shortcut/20260918-234156_S_train11-14 \
    --runs results/track_a/20260919-113607_S_train11-14 \
           results/track_a/20260919-131820_S_train24-27 \
           results/track_a/20260919-143354_S_train37-40
```

**Data:** 18 test windows (9 + 6 + 3 across start dates 11, 24, 37), 200k flows each, test unknown services, 1,000 day × service cluster bootstrap resamples, Holm over 10 hypotheses.

## Outcome

| # | Hypothesis | Estimate (95% CI) | Holm p | Supported |
|---|---|---|---|---|
| H1 | inheritance is teacher-specific | kdA4 +0.0071 (0.0048–0.0090); kdB4 +0.0255 (0.0235–0.0279) | 0.009 | **yes** |
| H2[energy] | KD beats `ls` and `directTS` | AUROC vs `ls` +0.021, vs `directTS` −0.000; NLL vs `ls` −0.040 | 1.0 | no |
| H2[msp] | as above, MSP | AUROC vs `ls` +0.006, vs `directTS` −0.002 | 1.0 | no |
| H3[energy] | gap grows over time | slope **−0.0004/week** (−0.0007 to −0.0001) | 1.0 | no (reversed) |
| H3[msp] | as above, MSP | slope −0.0003/week | 1.0 | no (reversed) |
| H4a | KD student relies on the shortcut more | −0.032 | 1.0 | no |
| H4b1 | shortcut-reliant teacher passes on over-confidence | ECE +0.0086 | 0.005 | **yes** |
| H4b2 | …and worse unknown detection | −0.020 (detection improves) | 1.0 | no (reversed) |
| H5[energy] | EnDD keeps more than KD | −0.092 | 1.0 | no |
| H5[msp] | as above, MSP | −0.008 | 1.0 | no |

**Two of ten supported.** Both sensitivity analyses (duplicates removed; ≥5-packet flows) give the same decision for every hypothesis, so nothing rests on duplicate flows or on very short flows.

## What the numbers say

**1. Inheritance is real but small on the test windows.** The T = 4 students move toward their own teacher by +0.007 (A) and +0.026 (B), against +0.046 / +0.042 on the validation weeks. The raw own-minus-other numbers (+0.046 / −0.045) are dominated by "everyone follows the ensemble", which is exactly the artefact H1's baseline was designed to remove. The tuned T = 1 students inherit nothing (+0.001, −0.000).

**2. Distillation does not preserve unknown detection better than cheap alternatives.** Against `directTS` (the same student, temperature-scaled) the difference is zero: energy −0.000, MSP −0.002. Against label smoothing, KD wins on detection (+0.021 energy) but loses on calibration (NLL −0.040). No comparison survives the intersection-union requirement.

**3. The gap *shrinks* over time, the opposite of H3.** Teacher A's energy AUROC falls faster than the student's, so after about three months the student is the better unknown detector:

| Weeks after training (start 11) | Teacher A | kdA | gap |
|---|---|---|---|
| 3.5 | 0.869 | 0.854 | +0.015 |
| 11.5 | 0.841 | 0.846 | −0.005 |
| 23.5 | 0.813 | 0.814 | −0.001 |
| 35.3 | 0.784 | 0.793 | −0.008 |

With MSP the teacher stays ahead by a roughly constant 0.025, so which model degrades faster depends on the score. Accuracy decays steeply for everyone (kdA macro-F1 0.90 → 0.69 over 35 weeks), which is the drift the dataset was chosen for.

**4. Shortcuts: capacity matters, distillation does not.** The KD student does not rely on the planted shortcut more than the direct student (−0.032); both 101k students rely about twice as much as the 2.3M teacher. A shortcut-reliant teacher does pass on over-confidence to a student that never sees the feature (ECE +0.009, the second supported hypothesis), while that student's unknown detection gets *better*, not worse.

**5. EnDD loses decisively on the energy score** (−0.092) and slightly on MSP (−0.008), so the proxy-Dirichlet student is not the uncertainty-preserving method the plan hoped for.

## Reading for the paper
The teacher's *score pattern* transfers, teacher-specifically and measurably, but none of the properties that would make distillation attractive for deployment do: not detection quality, not calibration, not shortcut robustness. Temperature-scaling a directly trained student matches distillation on every unknown-detection outcome, at a fraction of the cost. And over a year of real traffic the compressed student is at least as durable as its teacher.

## Headline numbers, averaged over all test windows

| Model | macro-F1 | AUROC energy | AUROC MSP | NLL (scaled) |
|---|---|---|---|---|
| Teacher A (5 × 2.3M) | 0.860 | 0.836 | 0.868 | 0.447 |
| Teacher B (10.3M) | 0.855 | 0.840 | 0.851 | 0.480 |
| direct (101k) | 0.806 | 0.835 | 0.844 | 0.609 |
| directTS | 0.806 | 0.835 | 0.845 | — |
| kdA (T = 1) | 0.807 | 0.835 | 0.843 | 0.601 |
| kdA4 (T = 4) | 0.790 | 0.809 | 0.824 | 0.565 |
| ls (0.05) | 0.797 | 0.814 | 0.836 | 0.561 |
| enddA | 0.801 | 0.743 | 0.835 | 0.563 |
