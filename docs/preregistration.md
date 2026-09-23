# Pre-registration: What Does the Student Inherit?

**Status:** FROZEN

> **How this file is used.** `scripts/06_track_a.py --with-test` and `scripts/08_analyze.py --windows test` refuse to touch the test windows unless the status line reads `**Status:** FROZEN`. This file was frozen on 19 September 2026, after the week-3 pilot passed and decisions D1–D8 were settled on validation data alone. **From now on it changes only by appending dated entries to section 11 (Deviations).** Nothing above section 11 may be edited.

| Item | Value |
|---|---|
| Frozen on | 19 September 2026 |
| Git commit (frozen code and design) | `fa8f718` |
| OSF registration | https://osf.io/rts6n (public, registered 20 September 2026); project https://osf.io/xcgpb |
| Decisions settled | D1–D8, all recorded below with their reasons |
| Evidence used to settle them | validation weeks only (weeks 15, 28, 41); no test-window result has been computed |
| Split file | `configs/splits.json` (digest `305ac8d0ea2a`) |

## 1. Question
When a large traffic classifier (teacher) is distilled into a ~100k-parameter student, does the student inherit the teacher's own unknown-traffic detection, calibration and shortcuts? How does that change over the months after training? The alternative is that the student only gets the generic regularisation effect of soft labels.

## 2. Data
- **Dataset:** CESNET-TLS-Year22, size S, loaded with cesnet-datazoo 0.2.0.
- **Inputs:** PPI-30 (inter-packet time, direction, size) plus 12 flow statistics and 32 packet-histogram features. No payload, SNI, IP, port, ASN or JA3.
- **Services:** 102 known; unknown services fixed in `configs/splits.json` (see §3).
- **Start dates:** training weeks 11–14, 24–27 and 37–40. Validation is the following single week (15, 28, 41).
- **Test windows:** consecutive 4-week blocks after the validation week, up to week 52, with low-coverage weeks 50 and 52 removed (e.g. 16–19, 20–23, …, 48–51 without 50).
- **Sizes:** all training flows of the window; validation 200k known flows plus all validation-unknown flows; each test window 100k known plus 100k test-unknown flows.
- **Excluded weeks:** weeks 1–10, because of the flow-exporter change in week 10.

## 3. Known and unknown services
Fixed by `scripts/03_make_splits.py` (seed 2022) before any model was trained.
- **Validation unknowns:**
  - far = Music, Videoconferencing;
  - near = booking-com, king-games, microsoft-weather, opera-sitecheck, youtube.
- **Test unknowns:**
  - far = Search, Social;
  - near = alza-logapi, ctu-matrix, google-drive, mozilla-push, mozilla-token, rozhlas-api, seznam-ssp, unpkg.
- **Supplementary natural-emergence set** (start date 11 only, exploratory): cesnet-login, gitlab, cesnet-kalendar, font-awesome, amazon-alexa, cesnet-perun, redmine.

## 4. Models and training conditions
- **Teacher A:** 5 × `mm_cesnet_v2` (2.26M parameters each), seeds 0–4. The ensemble prediction is the mean of member softmaxes. The ensemble energy score is the mean of member energies.
- **Teacher B:** one wider multimodal CESNET network (the swap control). **Decision D1, settled 19 Sep 2026:** the teacher swap is interpretable only if Teacher B's validation macro-F1 is within **2 points** of Teacher A's, checked per start date. If a start date fails, its teacher-swap results (H1) are reported but excluded from the pooled test, and the exclusion is recorded in §11. Validation results (A − B, macro-F1 points): start 11 +0.19, start 24 +0.04, start 37 +0.06, so all three are well inside the tolerance.
- **Student:** 1D-CNN, 101k parameters. Conditions:
  - `direct`: cross-entropy;
  - `ls`: label smoothing ε (tuned, D2; planned 0.1);
  - `directTS`: `direct` after temperature scaling;
  - `kdA`, `kdB`: Hinton KD with temperature T and weight α (tuned on `kdA`, D2; planned T = 4, α = 0.9). `kdB` uses the same values;
  - `kdA4`, `kdB4`: the same, at the conventional T = 4 (the **inheritance arm**, see D8);
  - `enddA`: ensemble distribution distillation with proxy Dirichlet targets and a reverse-KL loss (after Ryabinin et al., NeurIPS 2021). The target mean is Teacher A's mean prediction, and the precision is estimated from member disagreement (capped at 10⁴). Plain maximum-likelihood EnDD was replaced because it is unstable with ~100 classes: in the laptop smoke run its energy score was anti-correlated with both teachers.
- **Training (all models):** AdamW, lr 1e-3, weight decay 1e-4, one-cycle schedule, batch 1024, bf16 autocast; 10 epochs for teachers, 20 epochs for students (see the epoch check below). The kept epoch is the one with the lowest validation cross-entropy on known flows.
- **Seeds:** 3 student seeds per condition and start date.
- **Tuning (decision D2, settled 16 Sep 2026):** a small grid with the same kind of budget for each tunable method, run once by `scripts/09_tune_students.py` on start date 11 (validation week 15, student seed 0):
  - KD: T ∈ {1, 2, 4} × α ∈ {0.5, 0.9};
  - label smoothing: ε ∈ {0.05, 0.1, 0.2}.

  The selection criterion is validation macro-F1 on known flows only. Unknown detection and calibration are not computed during tuning. A tie (within 0.001) goes to the planned value. The chosen values are stored in `configs/student_hparams.json` and used for all start dates and for the shortcut experiment. `direct` and `enddA` have no tuned settings.

  **Reason:** in the first validation-only grid run (start 11, planned values), the KD students lost about 2 macro-F1 points against `direct` (0.881 vs 0.901, 3 seeds). With untuned settings, a comparison at matched accuracy (H2) would not be fair to KD.
- **Tuning result (17 Sep 2026)** — run `results/tuning/20260917-085200_S_train11-14`, validation macro-F1 after 10 epochs:

  | Setting | Macro-F1 |
  |---|---|
  | direct | 0.9000 |
  | KD, T = 1, α = 0.5 | 0.9012 |
  | **KD, T = 1, α = 0.9 (selected)** | **0.9034** |
  | KD, T = 2, α = 0.5 | 0.8993 |
  | KD, T = 2, α = 0.9 | 0.9005 |
  | KD, T = 4, α = 0.5 | 0.8879 |
  | KD, T = 4, α = 0.9 (planned) | 0.8821 |
  | **LS 0.05 (selected)** | **0.8929** |
  | LS 0.1 (planned) | 0.8876 |
  | LS 0.2 | 0.8782 |

  The selected values are stored in `configs/student_hparams.json`.
- **Epoch check and change (17 Sep 2026, decided before freezing):**
  - With 20 epochs, `direct` reached 0.9150 and the selected KD student 0.9154, against 0.9000 and 0.9034 at 10 epochs.
  - Their best epoch was the last or second-to-last one, so 10 epochs left the students unconverged.
  - **Change:** students are trained for 20 epochs (`student_epochs` in `configs/student_hparams.json`). Teachers stay at 10 epochs; their validation loss had flattened by epoch 9–10 in the pilot.
  - The validation-only start-11 grid run of 16 Sep (10-epoch students, planned KD values) is kept only as the record behind the H1 and D6 decisions.

## 5. Outcomes
- **Unknown-scores** (higher = more likely known): energy (logsumexp of logits) and maximum softmax probability (MSP).
- **Primary outcome:** the teacher-minus-student gap in unknown-detection AUROC (`kdA` vs Teacher A), with energy and MSP as co-primary scores (D6). All test flows, all test-unknown services, as a function of weeks since the end of training.
- **Secondary outcomes:**
  - macro-F1 on known flows;
  - AUROC and FPR@95TPR (MSP and energy);
  - OSCR;
  - calibration: NLL after temperature scaling (fitted on validation known flows) is the main calibration outcome (D7); ECE (15 bins) before and after temperature scaling, NLL before scaling and Brier are also reported;
  - AURC;
  - per-flow inheritance: Spearman correlation of energy scores between student and teacher, measured relative to the direct student (H1), plus top-1 agreement and error Jaccard on known flows.
- **Reporting groups:** all flows and flows with ≥5 packets; all, near and far unknown services.

## 6. Hypotheses and confirmatory tests
All tests are one-sided at α = 0.05, Holm-corrected across the family of 10 hypotheses:
- H1;
- H2, H3 and H5, each once per co-primary score: `[energy]` and `[msp]` (D6) — 6 in total;
- H4a, H4b1 and H4b2 (D4). The analysis code is `scripts/08_analyze.py` with `kdtraffic/analysis.py`, and it is frozen together with this file.

**Unit and pooling.** The unit is one start date × one test window. Pooled estimates are unweighted means over units of the seed-averaged statistic.

**Bootstrap.** 1,000 resamples. Each resample does two things:
- It redraws day × service clusters with replacement, **globally**: a cluster that appears in windows of several start dates gets the same multiplicity everywhere.
- It redraws the 3 student seeds with replacement, separately per start date.

Resampled clusters enter the metrics as flow weights. The reported interval is the 95% percentile interval.

**p-values.** The one-sided p-value of a component is (1 + #{resampled estimate ≤ 0}) / (B + 1).

**Several components.** A hypothesis with several components is supported only if every component is (intersection–union test). Its p-value is therefore the largest component p-value.

**Sensitivity analyses** (same code, not in the Holm family):
- (a) test flows with an exact packet-sequence duplicate in the training window removed (see D5);
- (b) flows with ≥5 packets only.

| # | Hypothesis | Test |
|---|---|---|
| H1 | Teacher-specific inheritance: distillation moves a student's energy scores toward its own teacher, beyond what a directly trained student shares with each teacher. Tested on the inheritance arm `kdA4` / `kdB4` (D8); the tuned arm is reported | Difference in differences of pooled Spearman correlations over all test flows (2 components, both > 0): `kdA4`: [ρ(kdA4, A) − ρ(kdA4, B)] − [ρ(direct, A) − ρ(direct, B)]; `kdB4`: [ρ(kdB4, B) − ρ(kdB4, A)] − [ρ(direct, B) − ρ(direct, A)]. The raw own − other differences are reported. **Reason for the baseline:** in the validation-only run (start 11), every student, including `direct`, correlated more with Teacher A (the ensemble) than with B (+0.044 for `direct`). A raw test would therefore confuse "follows its own teacher" with "follows the more central teacher" (raw kdB: −0.009; relative to `direct`: +0.034) |
| H2 | Beyond regularisation: `kdA` beats `ls` and `directTS` on unknown-detection AUROC and on post-hoc NLL at matched macro-F1 | AUROC of `kdA` minus each control > 0 (tested separately for energy and MSP), and post-hoc NLL of each control minus `kdA` > 0 (4 components). Post-hoc NLL is the mean negative log-likelihood of the true class on known flows after temperature scaling; for `directTS` the scaled model itself. Post-hoc ECE is reported with intervals but not tested (decision D7). **Matched accuracy (decision D3, settled 19 Sep 2026):** H2 is interpreted as a matched-accuracy comparison only while \|Δ macro-F1\| ≤ 1 point between `kdA` and the control, pooled over units; the difference is always reported with its interval. If a control exceeds 1 point, that comparison is reported as unmatched and excluded from H2, and the exclusion is recorded in §11. Validation values with the tuned settings: `kdA` − `ls` = +0.9 points, `kdA` − `directTS` = +0.1 points, so both are matched |
| H3 | Decay: the Teacher A − `kdA` AUROC gap (energy; MSP) grows with weeks since training | Slope > 0 in an OLS regression of the per-unit gap (seed-averaged) on weeks since the end of training, with one intercept per start date. Three start dates are too few to estimate a random-effect variance, so start date enters as a fixed effect; seed variation enters through the bootstrap |
| H4a, H4b1, H4b2 | Shortcut transfer | See D4 below |
| H5 | EnDD keeps more of A's unknown detection than Hinton KD | AUROC of `enddA` minus `kdA` > 0 (energy; MSP) |

**Decision D4 (H4 wording), settled 19 Sep 2026.** The study plan's H4 ("with a feature visible only to the teacher, the KD student relies on it more") cannot be measured directly: a student that never sees the feature cannot rely on it. H4 is therefore replaced by three hypotheses:
- **H4a (both see the feature):** at ρ ∈ {0.9, 1.0}, the flip-test reliance (macro-F1 aligned − flipped) of the `kd` student exceeds that of the `direct` student.
- **H4b1 (teacher only, confidence):** a KD student distilled from a shortcut-reliant teacher (ρ ∈ {0.9, 1.0}) has higher ECE than one distilled from the ρ = 0 teacher.
- **H4b2 (teacher only, detection):** the same student has lower energy AUROC than one distilled from the ρ = 0 teacher.

H4b1 and H4b2 were one hypothesis with two components until 19 Sep. They are separate because they are independent predictions, and the validation run showed them pointing in opposite directions (ECE +0.009 while AUROC was +0.020, i.e. detection improved). Bundling them would report only "not supported" and hide which half holds.

Implemented in `scripts/07_shortcut.py` (KD at the conventional T = 4, D8), validation week of start date 11, 3 seeds. Test: a paired one-sided t-test over the 6 (ρ, seed) pairs, one component each.

**Decision D6 (primary unknown-score), settled 16 Sep 2026: energy and MSP are co-primary.** The week-3 pilot (validation week only, §8) found:

| Model | Energy AUROC | MSP AUROC |
|---|---|---|
| Teacher A | 0.837 | 0.916 |
| Direct student | 0.819 | 0.851 |

So the energy score compresses the teacher–student gap (0.018 vs 0.065 with MSP), and for the teacher itself energy is the weaker score. The options are:
- keep energy as primary;
- switch to MSP;
- make energy and MSP co-primary, with every AUROC component of H2, H3 and H5 tested for both scores and Holm applied over the doubled family.

Co-primary was chosen instead of switching to MSP. Choosing the score that looked better on validation data would be a selection on outcomes, and in the validation-only grid run (start 11) the two scores ranked the student methods differently: `ls` was best with energy, `enddA` with MSP. H2, H3 and H5 are tested once per score, inside the Holm family.

**Decision D8 (KD temperature arms), settled 17 Sep 2026.** The accuracy-tuned KD setting (T = 1, α = 0.9) makes distillation behave like ordinary training, so there is nothing to inherit:

| Start date | Student | ρ(student, Teacher A) | Shift toward own teacher vs `direct` |
|---|---|---|---|
| 11 | `direct` | 0.507 | — |
| 11 | `kdA`, T = 1 | 0.503 | +0.004 |
| 11 | `kdA`, T = 4 (16 Sep run) | 0.876 | +0.040 |
| 24 | `direct` | 0.440 | — |
| 24 | `kdA`, T = 1 | 0.446 | +0.006 |

A teacher with 96% accuracy produces nearly one-hot targets at T = 1, so the KD loss is close to cross-entropy. The grid therefore keeps both arms:
- **tuned arm** (`kdA`, `kdB`): the accuracy-optimal setting, used for H2 and H5, where matched accuracy matters;
- **inheritance arm** (`kdA4`, `kdB4`, T = 4): conventional KD, used for H1 and for the shortcut experiment (`scripts/07_shortcut.py`), which asks the same inheritance question.

The difference between the arms is itself reported: how much a student inherits is governed by the distillation temperature, and the most accurate setting inherits nothing beyond the labels.

**Decision D7 (calibration outcome), settled 16 Sep 2026.** H2 tests post-hoc NLL instead of post-hoc ECE. Reason: in the pilot, all neural models already had ECE below 1% (Teacher A 0.14%, direct student 0.54%), so ECE differences would be close to zero and dominated by binning noise. NLL separated the models clearly (0.086 vs 0.239). ECE is still reported.

## 7. Exclusions and data rules
- **Deduplication:** test flows whose exact PPI-30 sequence (timing, direction, size) occurs in the training window are reported separately (sensitivity analysis). They are flagged by a 64-bit hash of the scaled sequence (`duplicate` in `scores_<split>.npz`). **Decision D5, settled 19 Sep 2026:** the primary analysis **keeps** them, because they are part of the traffic a deployed classifier really sees, and the `drop_duplicates` sensitivity analysis reports every hypothesis without them. A hypothesis whose decision differs between the two is flagged in the paper. Validation duplicate rate: 5.0% of week-15 flows.
- **Minimum training flows:** a known service with fewer than 100 training flows in a window stops the run (none expected under the split rules).
- **No data from after the end of a training window** is used for training, early stopping or temperature fitting.

## 8. Stopping and pivot rules
- **Week-3 pilot gate** (`scripts/05_pilot.py`, validation only): continue if Teacher A beats the direct student by ≥2 macro-F1 points or ≥0.02 energy AUROC. Otherwise shrink the student (width 16), then use PPI-10. Every change is recorded here before freezing.
  - **Result (16 Sep 2026)** — size S, laptop RTX 3050 Ti, run `results/pilot/20260916-122627_S_train11-14`: **passed**.

    | Model | Macro-F1 | Energy AUROC |
    |---|---|---|
    | Teacher A | 0.964 | 0.837 |
    | Direct student | 0.902 | 0.819 |
    | Gap | 6.17 points | 0.018 |

  - **Baselines** (`results/baselines/20260916-134400_S_train11-14`):
    - XGBoost on flow statistics: macro-F1 0.883, energy AUROC 0.855 (0.875 on near unknowns);
    - k-NN on PPI: macro-F1 0.712, AUROC 0.745 with the k-th-neighbour distance.
  - No change to the student or the task was needed.
- **Track B** (netFound): continue only if preprocessing works within one week and netFound-small beats XGBoost by ≥2 macro-F1 points; otherwise it becomes an appendix.

## 9. Exploratory analyses (labelled as such in the paper)
- MSP and kNN-embedding scores beyond the primary energy score.
- Per-category breakdowns.
- The natural-emergence services.
- Conformal recalibration with delayed labels.
- Natural shortcut fields (DST_IP, DST_ASN, TLS_JA3).
- XGBoost and k-NN baselines.
- CPU deployment measurements.

## 10. Code and data
- Code: this repository.
- Splits: `configs/splits.json`.
- Per-flow scores: `scores_<split>.npz` from `scripts/06_track_a.py`.
- Confirmatory analysis: `scripts/08_analyze.py --windows test`, which refuses to run until this file is frozen.

Everything is released with the paper.

## 11. Deviations after freezing
*(append dated entries)*

**20 Sep 2026 — registered on OSF.** Registration https://osf.io/rts6n, public with no embargo; project https://osf.io/xcgpb, which archives this file and `configs/splits.json`. The only change to this document since the freeze is the OSF link in the header table, which was a placeholder. The registration form answers "Authors have observed the data, but have not performed the proposed analyses", with the explanation that the validation weeks settled decisions D1–D8 and that no test-window result existed at the freeze.

**20 Sep 2026 — note on timing.** The confirmatory test-window runs and their analysis (`results/analysis/CONFIRMATORY/`, commit `7ef3b5d`) were computed on 19 September, after the freeze (commit `fa8f718`) but before the OSF registration was posted. The freeze commit and its timestamp are in the public repository; the OSF registration certifies the same document one day later. No design decision was changed between the two.

**22 Sep 2026 — analyses added at peer review, all exploratory.** A reviewer asked for evidence on eight points. Everything listed below was added in response, *after* the confirmatory results were known, and is therefore exploratory in the strict sense: it is reported under its own heading in the paper, it carries its own uncertainty intervals, and **it shares no multiple-comparison family with the ten pre-registered hypotheses**, whose Holm correction is over those ten alone and is unchanged.

*The confirmatory results are unchanged.* Section 6 freezes the analysis code as well as the protocol. `kdtraffic/analysis.py`, which computes every statistic, is untouched since the freeze: `git diff fa8f718 -- kdtraffic/analysis.py` is empty. `scripts/08_analyze.py` carries one change, commit `1d79a0c` of 20 September, made while drafting the results and confined to the `figures()` function: it adds the T=4 arm to the specificity plot and adjusts bar order, figure size, axis limits and legend placement. It touches no statistic and writes no CSV, and the reproduction below — run with the changed code against results produced by the original — demonstrates that it moves no number. The new analyses live in `kdtraffic/exploratory.py` and `scripts/12_exploratory.py`, which *import* the frozen primitives rather than altering them, and read their own run directories under `results/track_a_exploratory/` and `results/feature_scores/`. Before any of this was written, the stored confirmatory command was re-run in the current environment (`results/analysis/CHECK_primary/`, 22 Sep): `hypotheses.csv` is byte-identical to the archived original and, restricted to the primary analysis, so are all 27 rows of `components.csv` and all 18 of `units.csv`. As a standing check, `12_exploratory.py` recomputes the H1 statistic with the new code and reproduces the confirmatory estimates (+0.0071335537 and +0.0255090221) to thirteen decimals.

Added:
1. **Feature-space open-set scores** — Mahalanobis and k-nearest-neighbour distances on each model's penultimate features (`kdtraffic/openset.py`, `scripts/13_feature_scores.py`), for every teacher and student already trained. Section 9 already lists a kNN-embedding score among the planned exploratory analyses; the Mahalanobis score is new. No model was retrained. These scores are **not** added to `SCORES` in the frozen code, because that constant sets the size of the Holm family.
2. **Drift mechanism** — unknown-traffic AUROC of each Teacher A member against the ensemble over time, testing the aggregation explanation the discussion had offered as a guess.
3. **H3 per start date**, and a table of how much the three start dates' test windows overlap in calendar time.
4. **Chance-level teacher preference** — the spread of an undistilled student's rank correlation with two equally good same-family teachers, which gives the scale on which H1's effect size should be read.
5. **Further teacher-swap conditions** (`kdM0`, `kdM1`, `hardA`, `kdC`), separating teacher identity from ensemble-ness, width and model family, and anchoring the swap against a student trained on the teacher's labels alone. Teacher C is a transformer, trained for this purpose on start week 11 only.
6. **Student width sweep** (16 and 96 beside the registered 48) on start week 11.
7. **XGBoost re-tuned** to a deployable size, reported beside the original configuration.

Exploratory intervals are built by the same day × service cluster bootstrap as the confirmatory ones, resampling training seeds in the same order, so that the two can be compared directly.

**23 Sep 2026 — feature distillation added, also exploratory.** An eighth analysis joins the seven above, under the same terms: it was added after the confirmatory results were known, it shares no multiple-comparison family with the ten hypotheses, and it changes nothing already reported.

8. **Feature distillation** (`kdF`) — a student trained with similarity-preserving distillation (Tung and Mori, ICCV 2019) from one member of Teacher A, on start week 11, with three seeds. The pre-registered conditions all distil from the output layer; this one matches the pairwise similarity structure of the penultimate features instead, which is the natural follow-up to analysis 1, since that analysis locates the teacher's advantage in the representation. The loss weight (β = 100) was **not** tuned — no budget was spent searching it — so a null result from this arm is weak evidence of absence and is reported as such. The training code adds `SimilarityPreservingKD` to `kdtraffic/distill.py` and a feature-returning path to `kdtraffic/train.py`, taken only by objectives that declare `needs_features`; every pre-registered condition keeps the original path. The frozen analysis code is untouched.

The feature-space scores of analysis 1 were extended to `kdF`, `hardA`, `kdM0` and `kdM1` on 23 September, with no model retrained, and the exploratory analysis was re-run over the enlarged set (`results/exploratory/REVISION_FULL2/`). The statistics that both passes share are unchanged.
