# Pre-registration: What Does the Student Inherit?

**Status:** DRAFT (not frozen)

> **How this file is used.** `scripts/06_track_a.py --with-test` refuses to load the test windows unless the status line reads `**Status:** FROZEN`. Freeze this file only after the week-3 pilot has passed and the open decisions below are settled. Then post it on OSF and record the OSF link and the git commit here. After freezing, change it only by appending dated deviations (section 11).

| Item | Value |
|---|---|
| Frozen on | *(date)* |
| Git commit | *(hash)* |
| OSF registration | *(link)* |
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
- **Teacher B:** one wider multimodal CESNET network (the swap control). *Open decision D1:* its accuracy must be similar to A's; the tolerance is to be fixed here (proposal: validation macro-F1 within 2 points).
- **Student:** 1D-CNN, 101k parameters. Conditions:
  - `direct`: cross-entropy;
  - `ls`: label smoothing 0.1;
  - `directTS`: `direct` after temperature scaling;
  - `kdA`, `kdB`: Hinton KD with T = 4, α = 0.9;
  - `enddA`: ensemble distribution distillation with proxy Dirichlet targets and a reverse-KL loss (after Ryabinin et al., NeurIPS 2021). The target mean is Teacher A's mean prediction, and the precision is estimated from member disagreement (capped at 10⁴). Plain maximum-likelihood EnDD was replaced because it is unstable with ~100 classes: in the laptop smoke run its energy score was anti-correlated with both teachers.
- **Training (all models):** AdamW, lr 1e-3, weight decay 1e-4, one-cycle schedule, batch 1024, 10 epochs, bf16 autocast. The kept epoch is the one with the lowest validation cross-entropy on known flows.
- **Seeds:** 3 student seeds per condition and start date.
- **Tuning:** no tuning beyond these fixed values. *Open decision D2:* whether a small tuning budget per condition, on validation data only, is allowed (proposal: none).

## 5. Outcomes
- **Unknown-scores** (higher = more likely known): energy (logsumexp of logits) and maximum softmax probability (MSP).
- **Primary outcome:** the teacher-minus-student gap in unknown-detection AUROC with the energy score (`kdA` vs Teacher A). All test flows, all test-unknown services, as a function of weeks since the end of training.
- **Secondary outcomes:**
  - macro-F1 on known flows;
  - AUROC and FPR@95TPR (MSP and energy);
  - OSCR;
  - ECE (15 bins) before and after temperature scaling (fitted on validation known flows);
  - NLL, Brier, AURC;
  - per-flow inheritance: Spearman correlation of energy scores between student and teacher, top-1 agreement and error Jaccard on known flows.
- **Reporting groups:** all flows and flows with ≥5 packets; all, near and far unknown services.

## 6. Hypotheses and confirmatory tests
All tests are one-sided at α = 0.05, Holm-corrected across H1–H5. Confidence intervals come from bootstraps that resample day × service clusters (1,000 resamples), pooled over test windows and start dates.

| # | Hypothesis | Test |
|---|---|---|
| H1 | Teacher-specific inheritance: the `kdA` student's energy scores correlate more with Teacher A than with Teacher B, and the `kdB` student's more with B than with A | Mean over seeds and start dates of (own − other) Spearman correlation > 0, for both students |
| H2 | Beyond regularisation: `kdA` beats `ls` and `directTS` on energy AUROC and on post-hoc ECE at matched macro-F1 | Difference in AUROC (and ECE) between `kdA` and each control > 0 (< 0 for ECE); matched accuracy is checked by reporting the macro-F1 difference. *Open decision D3:* the matching method (proposal: report only if \|Δ macro-F1\| ≤ 1 point, otherwise compare within macro-F1 bins across seeds) |
| H3 | Decay: the Teacher A − `kdA` energy-AUROC gap grows with weeks since training | Slope > 0 in a mixed-effects model: gap ~ weeks_since + (1 \| start date) + (1 \| seed) |
| H4 | Shortcut transfer (see D4) | See D4 |
| H5 | EnDD keeps more of A's unknown detection than Hinton KD | Energy AUROC of `enddA` minus `kdA` > 0 |

**Open decision D4 (H4 wording).** The study plan's H4 ("with a feature visible only to the teacher, the KD student relies on it more") cannot be measured directly: a student that never sees the feature cannot rely on it. The proposal is to split H4:
- **H4a (both see the feature):** at ρ ∈ {0.9, 1.0}, the flip-test reliance (macro-F1 aligned − flipped) of the `kd` student exceeds that of the `direct` student.
- **H4b (teacher only):** a KD student distilled from a shortcut-reliant teacher (ρ ∈ {0.9, 1.0}) has higher ECE and lower energy AUROC than one distilled from the ρ = 0 teacher.

Implemented in `scripts/07_shortcut.py`, validation week of start date 11, 3 seeds.

## 7. Exclusions and data rules
- **Deduplication:** test flows whose exact PPI-30 sequence (timing, direction, size) occurs in the training window are reported separately (sensitivity analysis). The primary analysis keeps them. *Open decision D5:* keep (proposal) or drop in the primary analysis.
- **Minimum training flows:** a known service with fewer than 100 training flows in a window stops the run (none expected under the split rules).
- **No data from after the end of a training window** is used for training, early stopping or temperature fitting.

## 8. Stopping and pivot rules
- **Week-3 pilot gate** (`scripts/05_pilot.py`, validation only): continue if Teacher A beats the direct student by ≥2 macro-F1 points or ≥0.02 energy AUROC. Otherwise shrink the student (width 16), then use PPI-10. Every change is recorded here before freezing.
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

Everything is released with the paper.

## 11. Deviations after freezing
*(append dated entries)*
