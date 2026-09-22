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

Section VI-A and Table VII report this in full. [PENDING — bootstrap intervals, running.]

**A student-capacity gradient.** We add students of width 16 and 96 beside the registered width of 48
(start week 11). [PENDING — whether a teacher advantage appears as the student shrinks.]

## 2. "The teacher swap varies ensemble-ness and width, not model family."

Correct, and the original text did not say so plainly. Teachers A and B differ in ensemble-ness (5
members vs 1) and width, not in architecture family. We now state this in Section III and add three
conditions that separate the factors:

| Condition | Teacher | What it isolates |
|---|---|---|
| `kdM0`, `kdM1` | single members of Teacher A | swap between two single models: removes ensemble-ness |
| `kdM0` vs `kdM1` | two members of the same ensemble | teachers identical in family, size and data, differing only in seed — teacher identity alone |
| `kdC` | a transformer teacher | swap across model families (start week 11 only, labelled a single-start probe) |

[PENDING — results.]

## 3. "H1's effect sizes (0.007–0.026) have no anchor."

Two anchors are added. An **upper anchor**, `hardA`: a student trained by cross-entropy on Teacher A's
top-1 labels, which copies the teacher's decisions while receiving none of the soft-target information
distillation is meant to convey. Whatever shift it shows is the part explained by label agreement
alone. And a **null scale**: the spread of an undistilled student's rank correlation with two equally
good, same-family teachers — how much a teacher preference varies for no reason at all. [PENDING —
both numbers.]

## 4. "The drift claim is stated from the energy score while the co-primary MSP score points the other way, and the mechanism is speculation."

We agree the discussion over-reached. The abstract and discussion now state that the teacher–student
gap does not widen under either score, and that under the energy score it reverses. The speculative
explanation — that the ensemble ages faster because of aggregation — is now tested: we compute the
unknown-traffic AUROC of each Teacher A member separately against the ensemble's, over all test
windows, under both scores, with the H3-style slope of each. [PENDING — result.]

## 5. "The 18 units come from three calendar-overlapping series and are not independent."

We now report the H3 trend fitted separately for each start date, and a table of exactly which
calendar weeks each unit covers and how many are shared with another start date. [PENDING — the count
of overlapping unit-weeks and the three per-start slopes.] The confirmatory analysis already clusters
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
form records that validation data had been observed. We now say this in the abstract, in a footnote to
Table II, and in §11 of the pre-registration.

## 8. "Only logit distillation is studied."

Accepted as a limitation rather than answered. Feature-based distillation (FitNets and successors) may
well transfer what logit matching does not — that is a real possibility our design cannot exclude. The
contribution and conclusion are scoped to logit-based distillation throughout, and Section VII states
this as the most important open question. We are willing to add a feature-distillation arm on one start
date if the editor considers it necessary.

## 9. "No dedicated open-set baseline; the XGBoost latency comparison is unfair."

The open-set baselines are answered in concern 1. On latency: the reviewer is right that 300 rounds ×
102 classes (30,600 trees, 244 MB, 41.6 ms per flow) is not a deployment configuration and its timing
flattered the neural students. We re-tune XGBoost to a deployable size and report both configurations.
[PENDING — re-tuned accuracy and latency.]

---

## Smaller points

- The introduction's foundation-model framing is aligned with the 2.3M-parameter teacher actually used.
- The 20-point accuracy decline over the year is related to the drift reported in the dataset paper.
- Figure 2's left panel is split so the curves are legible.
- Table II's H4 rows use the same one-sided bound convention as the other rows, and the p-value floor
  of 1/1001 imposed by 1,000 bootstrap resamples is stated.
- The repository URL is filled in: https://github.com/Mahmoud-Abbasi-svg/kd-encrypted-traffic-inheritance,
  with the pre-registration freeze commit tagged `prereg-frozen` so the registered protocol and the
  analysis code that implements it can be checked against each other.
- All 50 bibliography entries were verified against the arXiv API and Crossref; twelve titles were
  corrected and two journal references gained DOIs.
