# Results (draft, 20 Sep 2026)

All numbers in this section come from the pre-registered test windows (`results/analysis/CONFIRMATORY/`, OSF registration osf.io/rts6n). Validation-week numbers appear only where marked, and only as design context. Conventions: unknown-scores are oriented so that higher means "more likely known"; AUROC treats known flows as the positive class; "gap" means teacher minus student. Confidence intervals are 95% percentile intervals from 1,000 day × service cluster bootstrap resamples that also resample the three student seeds; p-values are one-sided and Holm-corrected across the ten pre-registered hypotheses.

## 1. Setting and headline numbers

Three training start dates (weeks 11–14, 24–27, 37–40 of 2022) give 18 test windows of 200,000 flows each, spanning 3.5 to 35 weeks after the end of training. Table 1 averages over all windows.

**Table 1. Test-window averages, all flows, all test-unknown services.**

| Model | Params | Macro-F1 | AUROC energy | AUROC MSP | FPR@95 energy | OSCR | NLL (post-hoc) | AURC |
|---|---|---|---|---|---|---|---|---|
| Teacher A (5 × mm_cesnet_v2) | 11.3M | 0.860 | 0.836 | 0.868 | 0.626 | 0.789 | 0.447 | 0.012 |
| Teacher B (wide) | 10.3M | 0.855 | 0.840 | 0.851 | 0.614 | 0.791 | 0.480 | 0.014 |
| direct | 101k | 0.806 | 0.835 | 0.844 | 0.674 | 0.768 | 0.609 | 0.020 |
| directTS | 101k | 0.806 | 0.835 | 0.845 | 0.676 | 0.767 | — | 0.020 |
| ls (ε = 0.05) | 101k | 0.797 | 0.814 | 0.836 | 0.683 | 0.747 | 0.561 | 0.023 |
| kdA (T = 1) | 101k | 0.807 | 0.835 | 0.843 | 0.667 | 0.767 | 0.601 | 0.020 |
| kdB (T = 1) | 101k | 0.806 | 0.832 | 0.840 | 0.674 | 0.764 | 0.602 | 0.020 |
| kdA4 (T = 4) | 101k | 0.790 | 0.809 | 0.824 | 0.678 | 0.739 | 0.565 | 0.023 |
| kdB4 (T = 4) | 101k | 0.789 | 0.810 | 0.822 | 0.669 | 0.741 | 0.582 | 0.023 |
| enddA | 101k | 0.801 | 0.743 | 0.835 | 0.780 | 0.679 | 0.563 | 0.022 |

Three things frame everything that follows. First, the teachers are 5–6 macro-F1 points more accurate than any student, but their advantage in unknown detection is small: 0.001 energy AUROC over `kdA` (CI −0.003 to 0.005) and 0.025 with MSP (CI 0.023 to 0.026). Second, the accuracy-tuned distillation arm (`kdA`, `kdB`, T = 1) is indistinguishable from the directly trained student on every column. Third, the two teachers are well matched: Teacher B trails Teacher A by 0.55 macro-F1 points on average across windows (maximum 1.6), inside the pre-registered 2-point tolerance, so the teacher-swap comparison is between equals.

**Table 2. The ten pre-registered hypotheses.**

| # | Hypothesis | Estimate (95% CI) | Holm p | Outcome |
|---|---|---|---|---|
| H1 | inheritance is teacher-specific | kdA4 +0.007 (0.005, 0.009); kdB4 +0.026 (0.024, 0.028) | 0.009 | **supported** |
| H2[energy] | KD beats `ls` and `directTS` at matched accuracy | AUROC vs `ls` +0.021, vs `directTS` 0.000; NLL vs `ls` −0.040, vs `directTS` +0.008 | 1.0 | not supported |
| H2[msp] | as H2, MSP | AUROC vs `ls` +0.006, vs `directTS` −0.002 | 1.0 | not supported |
| H3[energy] | gap grows with time | −0.0004 per week (−0.0007, −0.0001) | 1.0 | reversed |
| H3[msp] | as H3, MSP | −0.0003 per week (−0.0004, −0.0001) | 1.0 | reversed |
| H4a | KD student relies on a visible shortcut more | −0.032 (lower bound −0.071) | 1.0 | not supported |
| H4b1 | shortcut-reliant teacher transfers over-confidence | ECE +0.0086 (lower bound 0.0061) | 0.005 | **supported** |
| H4b2 | …and degrades unknown detection | −0.020 (detection improves) | 1.0 | reversed |
| H5[energy] | EnDD keeps more than Hinton KD | −0.092 (−0.099, −0.084) | 1.0 | not supported |
| H5[msp] | as H5, MSP | −0.008 (−0.009, −0.006) | 1.0 | not supported |

The two sensitivity analyses (exact duplicates of training flows removed, 1.8% of test flows; flows with at least five packets only, 94.0% of test flows) reach the same decision on every hypothesis, with every p-value within 0.1 of the primary analysis.

## 2. RQ1: what the student inherits

### 2.1 The score pattern transfers, and it is teacher-specific (H1)

Every student, including the directly trained one, correlates more with Teacher A than with Teacher B (Spearman ρ of per-flow energy scores: 0.746 vs 0.705 for `direct`). Teacher A is an ensemble and therefore produces the more "average" score, so a raw own-minus-other comparison would credit that artefact to distillation. H1 therefore measures the shift *relative to the direct student*.

On that measure both conventional-KD students (T = 4) move toward their own teacher: `kdA4` by +0.007 and `kdB4` by +0.026 (Fig. 2). The effect is small in absolute terms, but it is estimated from 3.6 million test flows and it survives Holm correction (p = 0.009). It is also asymmetric. The `kdB4` shift is positive in all 18 windows (0.017 to 0.033). The `kdA4` shift is positive in the nine windows of start dates 24 and 37 (0.011 to 0.023) and indistinguishable from zero in the nine windows of start date 11 (−0.007 to +0.005). We have no pre-registered explanation for the start-11 asymmetry and report it as a limitation of H1's support.

The accuracy-tuned arm inherits nothing: the `kdA` and `kdB` shifts are +0.001 (CI −0.001, 0.003) and −0.000 (CI −0.001, 0.001). This is not a null result about distillation in general but about distillation at T = 1 from a 96%-accurate teacher, whose soft targets are almost one-hot. The amount of teacher-specific signal a student receives is set by the temperature, and the temperature the tuning procedure selected for accuracy is the one that transmits none of it.

### 2.2 Detection quality and calibration do not transfer (H2, H5)

Table 1 already shows the result: `kdA` and `directTS` have the same energy AUROC to three decimals (0.835), the same MSP AUROC within 0.002, the same OSCR, AURC and macro-F1. Temperature-scaling a directly trained student, which costs one scalar fitted on validation data, matches distillation on every unknown-detection outcome we measured.

Against label smoothing the picture is mixed and does not favour distillation overall: `kdA` detects unknowns better (+0.021 energy AUROC, +0.006 MSP) but is worse calibrated (post-hoc NLL 0.601 vs 0.561). Because H2 requires distillation to win on both outcomes against both controls, it is not supported for either score.

The ensemble-distribution-distillation student (`enddA`) is the worst unknown detector in the grid with the energy score (0.743, a 0.092 deficit to `kdA`) and slightly worse with MSP (−0.008). Its energy scores are also the least stable across seeds. H5 is not supported for either score. We return to the mechanism in the discussion; the short version is that a proxy-Dirichlet target with a precision cap of 10⁴ trains the student's logit scale rather than its ranking of unknowns.

### 2.3 Near and far unknowns

**Table 3. Unknown detection by distance of the unknown services (test windows, energy AUROC / MSP AUROC).**

| Model | Near unknowns | Far unknowns |
|---|---|---|
| Teacher A | 0.805 / 0.836 | 0.840 / 0.872 |
| Teacher B | 0.788 / 0.794 | 0.848 / 0.859 |
| direct, directTS, kdA | 0.846 / 0.824 | 0.833 / 0.847 |
| ls | 0.795 / 0.809 | 0.816 / 0.840 |
| kdA4 | 0.788 / 0.794 | 0.812 / 0.829 |
| enddA | 0.695 / 0.816 | 0.749 / 0.838 |

The small direct student is a *better* detector of near unknowns (services from a known category) than either teacher with the energy score, and the teachers' advantage is confined to far unknowns and to the MSP score. Distillation at T = 4 moves the student toward the teacher's profile: `kdA4` loses 0.058 energy AUROC on near unknowns relative to `direct` and lands exactly on Teacher B's value. Inheriting the teacher's score pattern therefore includes inheriting the teacher's weakness.

## 3. RQ2: the gap over time (H3)

We pre-registered that the teacher–student gap would grow with weeks since training. It shrinks (Fig. 1, left). The fixed-effects slope is −0.0004 energy AUROC per week (CI −0.0007 to −0.0001) and −0.0003 per week with MSP; both are opposite in sign to H3.

**Table 4. Start date 11, energy AUROC against all test-unknown services.**

| Weeks after training | Teacher A | kdA | Gap | kdA macro-F1 |
|---|---|---|---|---|
| 3.5 | 0.869 | 0.854 | +0.015 | 0.902 |
| 7.5 | 0.863 | 0.858 | +0.005 | 0.880 |
| 11.5 | 0.841 | 0.846 | −0.005 | 0.848 |
| 19.5 | 0.815 | 0.818 | −0.003 | 0.747 |
| 27.5 | 0.803 | 0.806 | −0.003 | 0.724 |
| 35.3 | 0.784 | 0.793 | −0.008 | 0.687 |

Both models degrade over the year, and steeply: closed-set macro-F1 for every student falls from about 0.90 to about 0.69 over 35 weeks (Fig. 1, right). But the 11.3M-parameter ensemble loses energy AUROC faster than the 101k student, and from roughly the third month onwards the student is the better unknown detector. With MSP the teacher keeps a roughly constant 0.02–0.03 lead throughout, so the direction of the drift in the gap depends on the score; with neither score does the gap widen. The label-smoothing student is the exception in the other direction: its gap to the teacher grows from 0.011 to 0.027 over start date 11.

## 4. RQ3: shortcuts (H4a, H4b1, H4b2)

The shortcut experiment (start date 11, validation week, 3 seeds; pre-registered as such because it needs the training window) plants a one-hot feature that agrees with the class with probability ρ during training and measures reliance as the macro-F1 lost when the feature is flipped to a wrong value.

**Table 5. Flip-test reliance (macro-F1 aligned − flipped), mean of 3 seeds.**

| ρ | Teacher (2.3M) | direct student | KD student (T = 4) |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.037 | 0.069 | 0.068 |
| 0.9 | 0.103 | 0.185 | 0.196 |
| 1.0 | 0.800 | 0.945 | 0.870 |

Distillation does not increase shortcut reliance: at ρ ∈ {0.9, 1.0} the KD student relies on the feature 0.032 *less* than the direct student (H4a not supported), the difference being driven by ρ = 1, where the direct student collapses almost completely. What the table does show is a capacity effect. At every ρ below 1, both 101k students rely on the shortcut about twice as much as the 2.3M-parameter teacher (0.185–0.196 vs 0.103 at ρ = 0.9), whichever way they were trained.

In the teacher-only setting, the student never sees the feature and can be influenced only through the soft targets. A teacher trained with a reliable shortcut (ρ ∈ {0.9, 1.0}) produces a student with higher ECE than a shortcut-free teacher does (+0.0086, Holm p = 0.005; H4b1 supported): over-confidence transfers even when the feature that caused it does not. The same student's energy AUROC is *higher*, not lower (+0.020; H4b2 reversed), and its macro-F1 is unchanged. Whatever the shortcut-reliant teacher's soft targets carry, it sharpens the student's confidence without damaging its ranking of unknown flows.

## 5. Baselines and cost (validation week; exploratory)

XGBoost on the 44 flow statistics reaches macro-F1 0.883 and energy-margin AUROC 0.855 on the validation week, against 0.964 and 0.837 for Teacher A: the gradient-boosted model is 8 points less accurate but a slightly better detector of unknown services, and clearly better on near unknowns (0.875 vs 0.828). k-NN on packet sequences reaches 0.712 / 0.745. CPU latency and memory measurements are pending.

## Figures

- **Fig. 1** (`fig_gap.png`): left, teacher-minus-student energy AUROC against weeks since training, per student condition and start date; right, student macro-F1 over the same axis.
- **Fig. 2** (`fig_specificity.png`): Spearman correlation of per-flow energy scores with Teacher A and Teacher B, per student condition, averaged over test windows.
- **Fig. 3** (`fig_reliance.png`): flip-test reliance against ρ for the teacher and the two students that see the feature.
