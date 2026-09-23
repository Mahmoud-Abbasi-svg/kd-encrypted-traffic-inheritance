# Response to the reviewer

Manuscript: *What Does the Student Inherit? A Pre-Registered Study of Knowledge Distillation for
Encrypted-Traffic Classification*

We thank the reviewer for a reading that identified the weakest link in the paper — that a negative
result about inheriting detection quality is uninformative if there was no detection quality to
inherit — and for eight further points that have made the work considerably more solid. Every
concern is answered below with new evidence, or with a claim narrowed to what our evidence supports.

**Two commitments shape this revision.** First, the ten pre-registered hypotheses and their numbers
are unchanged: no confirmatory result was recomputed, rerun or reinterpreted. Second, everything new
is explicitly exploratory — added after the confirmatory results were known, reported under its own
heading, carrying its own intervals, and sharing no multiple-comparison family with the registered
ten.

To make the first commitment checkable rather than asserted:

- `kdtraffic/analysis.py`, which computes every statistic, is byte-identical to the frozen version
  (`git diff fa8f718 -- kdtraffic/analysis.py` is empty). `scripts/08_analyze.py` carries one change,
  commit `1d79a0c`, confined to its `figures()` function (the specificity plot gained the T=4 arm, and
  its bar order, size, axis limits and legend changed). It computes nothing and writes no table.
- Before writing any new code we re-ran the stored confirmatory command in the current environment.
  `hypotheses.csv` is byte-identical to the original, and restricted to the primary analysis so are
  all 27 rows of `components.csv` and all 18 of `units.csv`. This also demonstrates empirically that
  the figure change moves no number: the check ran the changed code against results produced by the
  original.
- The new analyses live in `kdtraffic/exploratory.py` and `scripts/12_exploratory.py`, which *import*
  the frozen primitives instead of altering them. As a standing check, the exploratory code recomputes
  the H1 statistic and reproduces the confirmatory estimates (+0.0071335537 and +0.0255090221) to
  thirteen decimal places.
- Deviations are recorded as a dated entry in `docs/preregistration.md` §11, listing every added
  analysis (OSF registration https://osf.io/rts6n).

---

## 1. "With an energy AUROC of 0.836 for the teacher and 0.835 for the direct student, there is no teacher advantage to inherit, so H2 and H5 failing tells us little."

This is the central objection and we accept its logic. Two additions address it.

**A dedicated open-set detector.** The original submission scored unknown traffic only with logit-based
scores (energy, MSP). We added Mahalanobis and feature-space k-nearest-neighbour detectors computed on
each model's penultimate features, for every teacher and student already trained — no model was
retrained, and the scores are *not* added to the frozen `SCORES` constant, because that constant sets
the size of the Holm family and would have changed every corrected p-value in the confirmatory table.

**The reviewer's premise turns out to be a property of the scoring rule, not of the models.** Averaged
over the 18 test windows, Teacher A detects unknown traffic 0.073 AUROC better than the directly
trained student under the Mahalanobis score and 0.034 better under the feature k-NN score, against
0.001 under the energy score. A teacher advantage of the size the objection presumes does exist; the
logit-based scores cannot see it. The pattern replicates across all three start dates.

This makes H2 and H5 interpretable, and it sharpens rather than softens the paper's reading of them:

- The conventional-temperature student **kdA4 recovers nearly all of the advantage** (+0.076 over
  `direct`, of the +0.073 available), while the accuracy-optimal **kdA recovers none** (−0.003). This
  is the same temperature split the teacher-swap test finds, now in an outcome an operator cares
  about. H2's tuned arm transfers nothing because its targets are nearly one-hot, not because there
  was nothing to transfer.
- **Label smoothing transfers more than distillation does** (+0.088), so the conclusion that a cheaper,
  teacher-free control matches or beats distillation holds in the regime where a real advantage exists.
- **EnDD, the worst detector under the energy score (−0.092 against kdA), is among the best under both
  feature-space scores** (+0.077, +0.043). This supports the mechanism already proposed in the
  discussion: the proxy-Dirichlet precision cap distorts the logit scale that the energy score reads,
  while leaving the representation intact. We now state H5's negative result as specific to
  logit-based scores.

One caveat we state in the paper: teacher-versus-student rows compare 600- or 1,200-dimensional
features against 128-dimensional ones, and a Mahalanobis distance is not scale-free across
dimensionality, so those rows are indicative. The student-versus-student rows, which carry the
argument, compare identical architectures. We also note that `directTS` is by construction identical
to `direct` under these scores, since temperature scaling does not change the penultimate features.

Section V-A (*A Dedicated Open-Set Detector*) and Fig. 5 report this in full, with 95% day × service
cluster bootstrap intervals throughout: Teacher A over `direct` is +0.073 (0.064, 0.082) under
Mahalanobis, +0.034 (0.028, 0.040) under feature k-NN and +0.000 (−0.004, 0.004) under energy;
`kdA4` over `direct` is +0.076 (0.069, 0.083) under Mahalanobis against `kdA`'s −0.003 (−0.010,
0.006); `ls` exceeds `kdA4` by +0.011 (0.008, 0.015) under Mahalanobis and +0.017 (0.013, 0.020)
under feature k-NN, which we test as a paired difference rather than by comparing two intervals.

**A student-capacity gradient.** We add students of width 16 and 96 beside the registered width of 48
(start week 11). No teacher advantage appears as the student shrinks. Against width 48, the width-16
student is a clearly worse unknown detector (−0.024 energy AUROC, CI −0.033 to −0.018) and the
width-96 student is no better (+0.002, interval spanning zero): detection has already saturated by
width 48. Distillation does not close the gap at any width — `kdA4` is worse than the equally sized
directly trained student at width 16 (−0.054) and at width 96 (−0.016) — so the property the paper
relies on, a small student detecting unknowns at least as well as distillation makes it, holds across
a sixfold range of student width.

## 2. "The teacher swap varies ensemble-ness and width, not model family."

Correct, and the original text did not say so plainly. Teachers A and B differ in ensemble-ness (5
members vs 1) and width, not in architecture family. We now state this in Section III and add three
conditions that separate the factors:

| Condition | Teacher | What it isolates |
|---|---|---|
| `kdM0`, `kdM1` | single members of Teacher A | swap between two single models: removes ensemble-ness |
| `kdM0` vs `kdM1` | two members of the same ensemble | teachers identical in family, size and data, differing only in seed — teacher identity alone |
| `kdC` | a transformer teacher | swap across model families (start week 11 only, labelled a single-start probe) |

The effect survives every factor being removed, and grows as the teachers become more different. All
four conditions are analysed with the H1 statistic, so they are read on the scale of Table III:

- **Ensemble-ness removed.** `kdM0`, distilled from a single ensemble member, follows that member
  over Teacher B by +0.0171 (0.0150, 0.0193). Both teachers are now single models.
- **Identity alone.** Between two members of the same ensemble — identical architecture, width and
  training data, differing only in random seed — the shifts are +0.0224 (0.0207, 0.0242) for `kdM0`
  and +0.0193 (0.0170, 0.0217) for `kdM1`. Nothing but the teacher's identity distinguishes them, and
  both exceed the registered `kdA4` arm's +0.007, so the original swap understated the effect rather
  than manufacturing it.
- **Across families.** A student distilled from the transformer Teacher C follows it over Teacher A by
  +0.0432 (0.0297, 0.0573), the largest shift we measure. This rests on nine units from one start date
  and is reported as a probe.

Section V-B reports these, and the ordering they produce — 0.000 for label copying, 0.007–0.026
between an ensemble and a wide single model, 0.019–0.022 between two seeds of one architecture, 0.043
across architectures — is what one would want a real effect to look like.

## 3. "H1's effect sizes (0.007–0.026) have no anchor."

Two anchors are added. An **upper anchor**, `hardA`: a student trained by cross-entropy on Teacher A's
top-1 labels, which copies the teacher's decisions while receiving none of the soft-target information
distillation is meant to convey. Whatever shift it shows is the part explained by label agreement
alone. And a **null scale**: the spread of an undistilled student's rank correlation with two equally
good, same-family teachers — how much a teacher preference varies for no reason at all.

Both are now measured. `hardA`'s shift toward Teacher A is −0.0000 (−0.0021, 0.0019), one-sided
p = 0.57: copying a teacher's decisions produces no measurable preference for that teacher's per-flow
scores, and since `hardA`'s closed-set accuracy matches the directly trained student's (0.913 against
0.913) this is not a control that failed to learn. The null scale, over the ten member pairs and all
units and seeds, has a median of 0.006, a 95th percentile of 0.016 and a maximum of 0.025.

Read against that scale, the results divide. The identity-only shifts (+0.019, +0.022), the `kdB4`
arm (+0.026) and the family swap (+0.043) all exceed the 95th percentile of variation between two
arbitrary same-family teachers. The `kdA4` arm's +0.007 does not; it sits close to the median. H1 is
still supported, because that arm's interval excludes zero, but we now say plainly in Section V-C that
the effect is small relative to that spread. We also state there that the two quantities are not
strictly commensurable, since H1's difference-in-differences already removes the baseline preference
the null measures.

## 4. "The drift claim is stated from the energy score while the co-primary MSP score points the other way, and the mechanism is speculation."

We agree the discussion over-reached. The abstract and discussion now state that the teacher–student
gap does not widen under either score, and that under the energy score it reverses. The speculative
explanation — that the ensemble ages faster because of aggregation — is now tested: we compute the
unknown-traffic AUROC of each Teacher A member separately against the ensemble's, over all test
windows, under both scores, with the H3-style slope of each.

The explanation is wrong. The ensemble ages at −0.00285 energy AUROC per week against a member mean of
−0.00285, with individual members between −0.00261 and −0.00306, and the same holds under every
scoring rule. Aggregation explains nothing. What remains is the other reading we offered: the teacher
and its members age at one rate and the 101k student ages more slowly. The deployment implication is
unchanged but now rests on a tested mechanism rather than a guess, and Section V-D reports the test.

## 5. "The 18 units come from three calendar-overlapping series and are not independent."

We now report the H3 trend fitted separately for each start date, and a table of exactly which
calendar weeks each unit covers and how many are shared with another start date. The overlap is
substantial: 15 of the 18 units contain at least one week that another start date also tests, 55
unit-weeks in total. Fig. 1(a) now draws the three series on one calendar so that the overlap is
visible, and Table XI gives the per-start counts. The reversal is not driven by one series: fitting
H3's trend separately gives −0.00044, −0.00055 and −0.00112 AUROC per week for the `kdA4` arm at start
dates 11, 24 and 37, all negative, and −0.00050, +0.00014 and −0.00152 for the tuned `kdA` arm, where
the middle series is flat — which is why the pooled estimate is small. No start date shows the
widening gap H3 predicted. The confirmatory analysis already clusters
the bootstrap by day × service, which handles dependence within a window but not between series; we
say so explicitly rather than implying the 18 units are independent replicates.

## 6. "The capacity claim rests on one student size."

The shortcut experiment's models were not saved and retraining its teachers is expensive, so we do not
re-run it at other sizes. We narrow the claim accordingly: from "capacity, not distillation" to "both
101k-parameter students rely about twice as much on the injected shortcut as the 2.3M-parameter
teacher, and distillation does not add to that reliance", with the width sweep of concern 1 as context
and the single-size limitation stated in Section VII. We have added model saving to the shortcut script
so a second round is cheap.

## 7. "H1's arm was added after seeing validation data."

True, and it should have been stated. The T=4 arm was added before the pre-registration was frozen but
after the validation weeks showed that distillation at the tuned temperature (T=1) transferred almost
nothing (ρ 0.503 vs 0.507 for the direct student). The freeze is public and time-stamped, and the OSF
form records that validation data had been observed. We now say this in three places: Section III-C,
which describes the tuning pass and why the arm was added; the paragraph introducing Table III, which
repeats it where the H1 result is read; and §11 of the pre-registration. The limitations paragraph of
Section VI records more generally that the freeze followed validation data that had already informed
eight design decisions.

## 8. "Only logit distillation is studied."

We have added a feature-distillation arm rather than leaving this as a limitation, though it answers the
concern only partly and we say so in the paper.

`kdF` is trained with similarity-preserving distillation (Tung and Mori, ICCV 2019) from ensemble member
0, at three seeds on start date 11. We chose that teacher because `kdM0` already distils the *logits* of
the same model, so the two arms differ in what they match and in nothing else, and the pair can be
compared directly.

The result runs against the expectation the reviewer's concern sets up. Matching the teacher's
representation transferred *less* of the teacher's representation-space advantage than matching its
logits did. Paired over the nine windows both arms share, `kdF` is 0.064 Mahalanobis AUROC below `kdM0`
(95% CI −0.071 to −0.053) and 0.009 below it under the feature k-NN score. Against the directly trained
student it gains +0.014 Mahalanobis AUROC with an interval spanning zero, where the teacher has +0.073
available. It does avoid the cost logit distillation pays under the logit-based scores, sitting +0.033
above `kdM0` under energy, so across all four scoring rules it behaves close to a student trained with no
teacher at all. Section V-G reports this, and offers one explanation: the objective fixes a
row-normalised Gram matrix, which leaves invariant exactly the rotations and rescalings of the feature
space that a Mahalanobis distance to class means is not invariant to.

Where this falls short of a full answer: the loss weight was fixed at β = 100 and never tuned, because
the tuning budget was spent before the pre-registration was frozen and we did not reopen it. A single
setting cannot separate "feature distillation does not transfer this" from "β was wrong". The training
loss shows the penalty was active (0.257 against the direct student's 0.182) but not that it was near a
value that would matter. We therefore report the arm as a probe, keep the contribution and conclusion
scoped to logit distillation, and state in Section VII and in the conclusion what a proper test would
need: a tuned weight, more than one start date, and objectives that constrain the absolute arrangement
of the representation rather than only its pairwise similarities.

## 9. "No dedicated open-set baseline; the XGBoost latency comparison is unfair."

The open-set baselines are answered in concern 1. On latency: the reviewer is right that 300 rounds ×
102 classes (30,600 trees, 244 MB, 41.6 ms per flow) is not a deployment configuration and its timing
flattered the neural students. We re-tune XGBoost to a deployable size (150 rounds, depth 6) and report
both configurations. The smaller model costs 2.9 macro-F1 points (0.854 against 0.883) and 0.029
energy-margin AUROC (0.826 against 0.855), and is 3.3× smaller and 2.9× faster at 73 MB and 14.4 ms per
flow. Even so it remains 180× the student's per-flow latency and 180× its weight footprint, so the
comparison's direction is unchanged while its magnitude is fairer. Both configurations are in
Section IV-E and Table VII.

---

## Smaller points

- The introduction's foundation-model framing is aligned with the 2.3M-parameter teacher actually used.
- The 20-point accuracy decline over the year is related to the drift reported in the dataset paper.
- Figure 2's left panel is split so the curves are legible. It is now Fig. 3, with the conditions whose
  teacher gap closes drawn apart from those whose gap does not, each panel on its own vertical scale.

**Presentation.** Four further changes were made to help a reader follow a design with several moving
parts. A new Fig. 1 draws the study: the three replicates on one calendar, and the swap control. A new
Table I lists every training condition with its teacher and objective, so the condition names used
throughout are defined in one place. The detection-advantage results, which previously needed a
four-column table, are now a forest plot (Fig. 5) that puts the logit-based and feature-space scores in
separate panels on one scale — the contrast the section argues from. The shortcut figure gained a
second panel for ρ ≤ 0.9, where the capacity effect lives, since the collapse at ρ = 1 had compressed
it. The results are split into a *Confirmatory Results* section and an *Exploratory Analyses* section,
so which evidence carries which status is visible from the section headings.
- Table II's H4 rows use the same one-sided bound convention as the other rows, and the p-value floor
  of 1/1001 imposed by 1,000 bootstrap resamples is stated.
- The repository URL is filled in: https://github.com/Mahmoud-Abbasi-svg/kd-encrypted-traffic-inheritance,
  with the pre-registration freeze commit tagged `prereg-frozen` so the registered protocol and the
  analysis code that implements it can be checked against each other.
- All 50 bibliography entries were verified against the arXiv API and Crossref; twelve titles were
  corrected and two journal references gained DOIs.
