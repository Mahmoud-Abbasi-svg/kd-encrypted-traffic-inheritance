# Study plan: What Does the Student Inherit?

**Working title:** *What Does the Student Inherit? Unknown-Traffic Detection, Calibration and Shortcuts in Distilled Traffic Classifiers over Time*

**Target:** IEEE TNSM (fallback TIFS / ToN) · **Duration:** 26 weeks · **Compute:** one 24–48 GB GPU, server CPU for deployment measurements
**Status:** plan approved 15 Sep 2026; pre-registration to be frozen at the end of week 3

---

## 1. Summary

Knowledge distillation (KD) is the standard way to turn a large traffic classifier into one that can be deployed. Papers judge the student almost entirely by closed-set accuracy on random splits. Nobody has tested whether the student also keeps the properties that matter in operation:
- detecting traffic from services it has never seen;
- calibrated confidence;
- robustness as traffic changes over months;
- freedom from shortcuts the teacher learned.

This study measures that on CESNET-TLS-Year22, a year of backbone TLS traffic, using time-ordered splits. It adds controls that separate real inheritance from the generic regularisation effect of soft labels: label smoothing, temperature scaling, and a teacher swap. A secondary, time-boxed track repeats the core measurement with a network foundation model (netFound) as teacher, on IoT device identification. The method contribution is deliberately light: ensemble distribution distillation plus conformal recalibration using delayed labels.

Either outcome is publishable:
- **Inheritance is real and teacher-specific:** we characterise how it decays over time and whether shortcuts carry over.
- **Only the regularisation effect transfers:** this shows that a common assumption behind distilled traffic classifiers is unfounded.

## 2. Positioning

### 2.1 Gap (novelty check, 15 Sep 2026)
No work found tests whether a distilled traffic classifier keeps its teacher's unknown detection and calibration, how that changes over time, or whether shortcuts carry over. Three works overlap partly:

| Work | Overlap | What it does not do |
|---|---|---|
| NetClus (arXiv 2508.02282) | Distils TrafficFormer/YaTC into a small feed-forward network; flags new traffic types | Random 8:1:1 splits; one held-out malware class; ISCX/USTC data; no calibration, drift or shortcut analysis |
| ResAware (arXiv 2606.17462) | Distilled student gains open-world TPR and calibration under 150-day drift | Website fingerprinting only; privileged-information teacher; no application classification, CESNET data or foundation models |
| PreDyn-IDS (Information Fusion 137, 2027) | 2.1M-parameter pretrained open-set IDS | No KD. Random splits; pretraining pool includes its test datasets; unknown detection shown only as confusion matrices |

General ML already suggests a student's out-of-distribution (OOD) detection follows the teacher's soft labels (arXiv 2007.03212). **A plain "the student inherits OOD ability" finding is therefore not enough.** The contribution rests on three things:
1. separating teacher-specific inheritance from regularisation (teacher swap, label-smoothing and temperature controls);
2. the time dimension, measured over a real year of traffic;
3. controlled shortcut-transfer tests.

### 2.2 Framing
- The classifier sees no SNI. This stands in for ECH (RFC 9849) and QUIC deployments, where the server name is hidden from passive observers.
- CESNET labels come from SNI. In operation, SNI or other ground truth arrives with a delay on part of the traffic, which makes periodic conformal recalibration realistic.

### 2.3 Evaluation bar this study must meet
Recent critiques set the standard reviewers will apply:
- **Sweet Danger of Sugar (SIGCOMM 2025):** packet-level splits and shortcuts inflate pretrained-model results.
- **SoK on encrypted traffic classifiers (S&P 2025):** "encrypted" datasets often contain unencrypted payload.
- **Is Network Traffic Classification in Crisis? (arXiv 2506.08655):** over 50% of samples are duplicated across train and test; k-NN matches complex models.
- **LiM (WWW 2025):** XGBoost on header fields matches ET-BERT.
- **Arp et al. (USENIX Security 2022):** common ML-in-security pitfalls.

Section 8 checks this plan against that bar.

## 3. Research questions and hypotheses

| RQ | Question | Hypothesis |
|---|---|---|
| RQ1 | Does KD pass on the *teacher's own* unknown detection and calibration, or only the generic effects of soft targets? | **H1:** the student's per-flow unknown-scores agree more with its own teacher than with an equally accurate other teacher (teacher swap).<br>**H2:** at matched accuracy, the KD student beats direct training with label smoothing and direct training with temperature scaling on unknown-detection AUROC and ECE. |
| RQ2 | How does the teacher–student gap change with weeks since training? | **H3:** the teacher-minus-student gaps in AUROC and ECE grow with weeks since training. |
| RQ3 | Do students inherit teacher shortcuts? | **H4:** when a shortcut feature is visible to the teacher only, the KD student relies on it more than a directly trained student (flip test). |
| RQ4 | Can a light method keep what matters? | **H5:** an ensemble distribution distillation (EnDD) student keeps more of the teacher's unknown detection than a Hinton-KD student. |

**Primary outcome:** the teacher-minus-student gap in unknown-detection AUROC (energy score), plotted against weeks since training. All other metrics are secondary. Holm correction applies across the confirmatory tests (H1–H5).

## 4. Track A: core study on CESNET-TLS-Year22

### 4.1 Data
- **Source:** CESNET-TLS-Year22, 1 Jan–31 Dec 2022, 180 web-service classes in 24 categories, about 508M flows. Loaded with [`cesnet-datazoo`](https://cesnet.github.io/cesnet-datazoo/).
- **Features:** sizes, directions and inter-packet times of the first 30 packets (PPI-30), plus flow statistics and packet histograms. There is no payload and no SNI in the inputs.
- **Size:** S or M, with stratified sampling per week. Size L will not fit the compute budget.
- **Period:** weeks 11–52 only. In week 10 (7–13 Mar 2022) the flow exporter was updated, which changed the distribution of packet-sequence features, so the dataset authors warn against training before week 10 and testing after it.

### 4.2 Time-ordered splits
- Three training start dates: **weeks 11–14, 24–27 and 37–40.** Each is followed by one validation week and monthly test windows up to week 52.
- Results are plotted against *weeks since training*, pooled over the three start dates. A single start date would confuse drift with calendar effects.
- **Mark holiday and academic-calendar weeks.** CESNET serves Czech universities, so term breaks shift the traffic mix. Weeks 15–16 of 2022 contain the Easter holidays; if week 15 is too unusual, move the first validation week.
- **Measured coverage (week 1):** weekly totals fall from about 11–14M flows to about 7M in weeks 27–35 (summer break). Weeks 50 and 52 have less than half the median volume (missing dates 12–13 and 29–31 Dec) and are excluded from test windows. Week 15 (Easter) has 9.9M flows, within the normal range.
- Configuration: `train_dates`, `val_dates` and `test_dates` in `DatasetConfig`.

### 4.3 Known and unknown services
- `apps_selection=FIXED` with explicit lists (`apps_selection_fixed_known`, `apps_selection_fixed_unknown`). DataZoo serves unknown flows only through its test loader, so the validation week is loaded as the test period of a second configuration that reuses the scalers fitted on the training window (`kdtraffic/data.py`).
- **Frozen split** (`configs/splits.json`, generated by `scripts/03_make_splits.py`, seed 2022; rules in the script docstring):
  - **Known:** 102 services in 18 categories, 71% of study-period flows. A known service must be usable in all three training windows and in ≥90% of valid study weeks.
  - **Validation unknowns:** far = Music and Videoconferencing (8 services); near = booking-com, king-games, microsoft-weather, opera-sitecheck, youtube.
  - **Test unknowns:** far = Search and Social (11 services); near = alza-logapi, ctu-matrix, google-drive, mozilla-push, mozilla-token, rozhlas-api, seznam-ssp, unpkg.
  - **Disabled:** 46 services (35 not usable enough, 7 emerging, 4 never usable).
  - The file is treated as frozen: the script refuses to overwrite it without `--force`.
- **Validation unknowns and test unknowns are different services.** Unknown-scores and thresholds are chosen on validation unknowns only.
- **Test unknowns are split in two:**
  - *near*: the unknown service belongs to the same one of the 24 categories as some known service;
  - *far*: it belongs to a different category.
- Distillation data contains no unknown services and nothing from later weeks.
- **Week-1 result (full-dataset weekly counts, `results/week1/summary.txt`):** no service first appears partway through the year; 177 of the 180 have flows in every study week. **Novelty is therefore simulated by holding services out, and the paper says so.**
  - *Emerging services* (below ~5,000 flows/week during training, usable later): 7 after weeks 11–14 (cesnet-login, gitlab, cesnet-kalendar, font-awesome, amazon-alexa, cesnet-perun, redmine), 2 after weeks 24–27, 1 after weeks 37–40. They form a small supplementary "natural emergence" test for the first start date only.
  - *Seasonal dips* (17–20 services, mainly university and CESNET services in the summer break) and *fading services* (mcafee-ccs, docker-authentication, eset-esa, opera-weather) are real drift events for known classes and feed RQ2.
  - Never usable in the study period, so excluded: uzis-ocko, ctu-kosapi, adobe-search, vscode-update.

### 4.4 Models and training conditions

| Role | Model | Notes |
|---|---|---|
| Teacher A | Ensemble of 5 × `mm_cesnet_v2` ([cesnet-models](https://cesnet.github.io/cesnet-models/)) | PPI + flow statistics; ensemble members use different seeds |
| Teacher B | Single wider PPI + flow-statistics network | Tuned to roughly match Teacher A's macro-F1; this is the teacher-swap control |
| Student | 1D-CNN, ~100k parameters | Same inputs as the teachers |
| ~~Optional teacher~~ | `model_30pktTCNET_256` (pretrained PPI embedding) | **Dropped:** its only weights (`CESNET_QUIC22_Week46_Domains`) were trained on CESNET-QUIC22 traffic from week 46 of 2022, the same network and a period after the training windows |

Student training conditions (same architecture, same tuning budget):
1. **Direct:** cross-entropy on hard labels.
2. **Direct + label smoothing.**
3. **Direct + temperature scaling:** post-hoc, applied to run 1, so no extra training.
4. **Hinton KD from Teacher A**, and **Hinton KD from Teacher B**.
5. **EnDD from Teacher A:** ensemble distribution distillation (Malinin et al., ICLR 2020).

Baselines: **XGBoost** on flow statistics and **k-NN** on PPI-30.

### 4.5 Measurements

| Group | Metrics |
|---|---|
| Closed-set accuracy | Macro-F1 (primary accuracy metric), flow-weighted F1 |
| Unknown detection | AUROC and FPR@95TPR with MSP, energy and kNN-embedding scores; OSCR. All reported separately for near and far unknowns |
| Calibration | ECE after temperature scaling, NLL, Brier score |
| Selective prediction | AURC |
| Inheritance per flow | Spearman correlation of teacher vs student unknown-scores; teacher–student error overlap (Jaccard); teacher-swap specificity: correlation with own teacher minus correlation with the other teacher |
| Fair comparison | All comparisons also made at matched macro-F1 (Vaze et al., ICLR 2022, show AUROC tracks closed-set accuracy) |

### 4.6 Shortcut tests (RQ3)
1. **Hidden handshake shortcut.** Labels come from SNI, and ClientHello size depends on SNI length. Server handshake sizes also reflect server configuration. Mask the sizes of the ClientHello, ServerHello and Certificate packets, then report the accuracy and AUROC drop for teacher and student.
2. **Synthetic shortcut.** Add a feature correlated with the label at ρ ∈ {0.5, 0.9, 1.0} during training, and remove the correlation at test time. Two settings:
   - (a) the feature is visible to both teacher and student;
   - (b) it is visible to the teacher only, so any transfer happens through soft labels.

   Reliance is measured with a flip test: the change in prediction when the feature is set to a value implying another class.
3. **Error floor.** Report the share of test flows whose exact PPI-30 sequence appears in training under a different label. No model can beat this floor. *Week-1 measurement (XS; 200k training flows from weeks 11–14 vs 100k from weeks 15–52; lower bounds):* 0.02% with timing included, 0.18% on directions and sizes only, 0.21% on the first 10 packets. The floor is negligible except for flows of 1–4 packets (1.7%).
4. **Natural shortcut candidates.** `return_other_fields` provides `DST_IP`, `DST_ASN`, `DST_PORT` and `TLS_JA3`, and server addresses appear to be real (e.g. Google 74.125.x in a week-11 sample). These can serve as natural shortcuts alongside the synthetic test; confirm whether client addresses (`SRC_IP`) are anonymised.

### 4.7 Statistics
- **Deduplication first:** identical flows are removed from test sets when they also appear in training.
  - *Week-1 measurement* (`results/week1/duplicates_w11_14.txt`, lower bounds from a 200k training sample):
    - exact PPI-30 duplicates (timing, direction and size): 1.7% of test flows;
    - ignoring timing: 12.1% (PPI-10: 14.7%);
    - flows of 1–4 packets (5% of test flows): about 50% duplicated.
  - **Rule:** deduplicate on exact PPI-30. Report every headline metric both for all flows and for flows with at least 5 packets.
  - **Drift signal:** duplication on directions and sizes falls from 23.8% in week 15 to 6.0% in week 52, a direct measure of drift worth a motivating figure.
  - **Memory:** loading DataZoo dataframes peaked at 9 GB RAM for 300k flows, so S-size experiments run on the server with streaming dataloaders.
- **Uncertainty:** bootstrap resampling by *day × service*, because flows are not independent and per-flow confidence intervals would be misleadingly narrow. Mixed-effects models are the alternative for the time analysis.
- **Seeds:** 3 per condition.

### 4.8 Conformal recalibration (RQ4, light)
- Split conformal prediction and adaptive conformal inference (Gibbs & Candès, NeurIPS 2021), updated with delayed SNI labels from recent windows.
- Report empirical coverage and mean prediction-set size per test window.
- **No coverage guarantee is claimed under drift.** Coverage is measured, not promised.

### 4.9 CPU deployment
- **Runtime:** ONNX Runtime on the server CPU, single-thread and batched.
- **Reported numbers:**
  - p50 and p99 latency; flows per second;
  - parameter count; peak RAM;
  - feature-extraction cost;
  - **time to decision:** PPI-30 needs the first 30 packets, so waiting time dominates inference time.
- XGBoost is included. If it wins on CPU, that is reported as a deployment finding.

### 4.10 Experiment grid (Track A)
| Block | Runs |
|---|---|
| Teachers: (5 ensemble members + Teacher B) × 3 start dates | 18 |
| Students: 5 trained conditions × 3 seeds × 3 start dates | 45 |
| Synthetic shortcut: 3 ρ × 2 visibility settings × (teacher + student) × 3 seeds, one start date | 36 (small) |
| XGBoost and k-NN | cheap, not counted |

About 100 training runs in total, which fits one GPU within weeks 4–10.

## 5. Track B: foundation-model teacher (time-boxed, gated)

### 5.1 Design
- **Data:** [UNSW-IoTraffic](https://doi.org/10.5061/dryad.w0vt4b94b): 27 consumer IoT devices, about 203 days, per-device raw captures (95.5M packets, about 27 GB).
- **Task:** IoT device identification. This extends the TMLCN 2024 paper to foundation-model teachers, time-ordered evaluation and unknown devices.
- **Teacher:** netFound-small (53.4M parameters, [SNL-UCSB](https://huggingface.co/snlucsb)), headers only, no payload.
  - *Check first:* netFound's pretraining data (UCSB campus traces) must not overlap UNSW-IoTraffic.
- **Swap-control teacher:** a supervised CNN on packet size and direction sequences, with accuracy similar to netFound-small.
- **Student:** a tiny CNN on packet size and direction sequences.
- **Removed fields:** MAC addresses, IP addresses, DNS names, payload. Device identity must not leak through identifiers.
- **Splits:** train on the first 30 days and test in monthly windows. Unknown devices come from rotating leave-3-devices-out folds.
- **Measurements:** as in Track A, covering the primary metric, teacher swap and flip test.
- **Not used:** NetMamba, because it reads payload bytes, which breaks the no-payload rule.

### 5.2 Gate (week 12)
Track B continues only if **both** hold:
1. capture preprocessing into netFound's input format works within one week;
2. netFound-small beats XGBoost by at least 2 macro-F1 points on the first test window.

Otherwise, Track B shrinks to an appendix or becomes the follow-up paper.

## 6. Mapping: question to experiment to output

| Hypothesis | Experiment | Metric | Planned output |
|---|---|---|---|
| H1 | Teacher swap (KD from A vs from B) | Swap specificity (Spearman difference) | Fig. 2: 2×2 specificity matrix with CIs |
| H2 | KD vs direct, + label smoothing, + temperature scaling | AUROC, ECE at matched macro-F1 | Fig. 3: AUROC and ECE vs macro-F1 scatter |
| H3 | Three start dates, monthly windows | AUROC gap and ECE gap vs weeks since training | **Fig. 1 (headline):** gap curves with day × service bootstrap CIs |
| H4 | Synthetic shortcut, both-visible vs teacher-only | Flip-test reliance vs ρ | Fig. 4: reliance curves |
| H5 | EnDD vs Hinton KD | AUROC (near/far), inheritance correlation | Table 2: main results per window |
| (secondary) | Handshake masking, error floor | Accuracy and AUROC drop; floor share | Table 3 |
| (secondary) | Conformal recalibration | Coverage, set size per window | Fig. 5 |
| (secondary) | CPU deployment | Latency, throughput, RAM, time to decision | Table 4 |
| (secondary) | Track B | Same as H1–H4 | Table 5 or appendix |

Table 1 summarises the datasets, splits and known/unknown service lists.

## 7. Pre-registration (frozen at end of week 3, posted on OSF)

To be fixed **before any test-window unknown results are computed**:
1. **Hypotheses H1–H5** as worded in Section 3, with the direction of each effect.
2. **Primary outcome:** the teacher-minus-student energy-AUROC gap against weeks since training. The analysis is the slope of that gap from a mixed-effects model with start date as a random effect.
3. **Confirmatory tests and correction:** one test per hypothesis, Holm-corrected, α = 0.05.
4. **Splits:** exact week ranges, known/unknown/near/far service lists, and the deduplication rule.
5. **Tuning budget:** the number of hyperparameter trials per model, chosen on validation data only.
6. **Exclusion rules:** services with fewer than 100 training flows are disabled (`min_train_samples_per_app`); holiday weeks are reported but not dropped.
7. **Stopping and pivot rules:** the week-3 pilot gate and the week-12 Track B gate (Section 9).
8. **Exploratory analyses**, labelled as such in the paper: kNN and MSP scores, per-category breakdowns, and the optional 30pktTCNET teacher.

## 8. Evaluation-bar checklist

| Requirement | Where addressed |
|---|---|
| Flow- or time-based splits | §4.2 three start dates, monthly windows; §5.1 time splits |
| Shortcut fields removed, with occlusion tests | No SNI/IP/port inputs; handshake masking and synthetic flip test (§4.6); MAC/IP/DNS/payload removed (§5.1) |
| Teacher beats XGBoost and a directly trained small model | XGBoost and k-NN baselines; direct student; week-3 pilot gate |
| Modern data | CESNET-TLS-Year22 (2022, backbone); UNSW-IoTraffic (2025 release) |
| Deployment measurements on real hardware | §4.9 CPU latency, RAM, time to decision |
| Robustness over time | §4.2, H3 |
| Duplicates removed | §4.7 |
| Pretraining data disjoint from test data | 30pktTCNET and netFound provenance checks |
| Released code and splits | Week 20–26 artifact release, together with the pre-registration |

## 9. Timeline and gates

| Weeks | Work | Gate |
|---|---|---|
| 1–3 | DataZoo setup; deduplication; check which services appear when and mark holiday weeks; check 30pktTCNET provenance and `return_other_fields`; XGBoost and k-NN baselines; **pilot:** Teacher A vs direct 100k CNN on one start date, evaluated on the validation week only so test unknowns stay unseen (`scripts/05_pilot.py`, `SERVER_SETUP.md`); write and freeze the pre-registration | **Week 3 pilot gate:** Teacher A must beat the direct student by ≥ 2 macro-F1 points **or** ≥ 0.02 AUROC. If not: shrink the student (~10k parameters) or make the task harder (PPI-10, fine-grained classes), then re-run the pilot. |
| 4–10 | Track A full grid (RQ1–RQ3) | **Week 10:** decide the paper's framing: teacher-specific inheritance, or regularisation only |
| 11–16 | Track B | **Week 12 gate** (§5.2) |
| 13–18 | EnDD and conformal recalibration (RQ4) | None |
| 18–20 | CPU deployment table; ablations | None |
| 20–26 | Writing; release code, splits and pre-registration; submit to TNSM | None |

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Teacher–student gap too small to measure (metadata models often match k-NN and XGBoost) | Week-3 pilot gate with explicit fallbacks |
| RQ1 alone reads as known (ResAware, general ML) | Teacher swap, time dimension and shortcut tests carry the contribution |
| No services genuinely appear mid-year | State that novelty is simulated; near/far split keeps the unknown test meaningful |
| Pretrained teachers saw test-period or test-set data | Provenance checks; drop contaminated teachers |
| XGBoost wins on CPU | Report as a deployment finding |
| netFound preprocessing slower than expected | Week-12 gate; Track B becomes an appendix or the follow-up paper |
| Compute overrun | Grid is about 100 runs; drop feature-level KD variants (already excluded) before cutting seeds |

## 11. Open checks for weeks 1–3
- [x] Subset sizes: XS 10M flows (2.5 GB), S 25M (6.3 GB), M 50M; per-week coverage in `results/week1/weekly_totals.csv`
- [x] First-appearance week of each of the 180 services: none appear mid-year (§4.3)
- [ ] Official 2022 Czech holiday calendar (seasonal and low-coverage weeks already identified from weekly totals, §4.2)
- [ ] Whether client addresses (`SRC_IP`) are anonymised (server addresses appear real, §4.6)
- [x] Train/test duplicate rate and error floor on XS for weeks 11–14 (§4.6, §4.7); repeat for the other two start dates on the server with the full training sets
- [x] Pretraining data of `model_30pktTCNET_256`: CESNET-QUIC22, week 46 of 2022, same network, so the teacher is dropped (§4.4)
- [ ] netFound pretraining data vs UNSW-IoTraffic overlap; whether UNSW captures retain MAC and IP addresses
- [ ] Final manual novelty search on Google Scholar and IEEE Xplore (KD + unknown detection + CESNET-TLS-Year22)
- [ ] Skim PreDyn-IDS citing papers and any new work from the ResAware group before freezing the pre-registration

## 12. Key references

**Closest work**
- NetClus: https://arxiv.org/abs/2508.02282
- ResAware: https://arxiv.org/abs/2606.17462
- PreDyn-IDS: https://doi.org/10.1016/j.inffus.2026.104611
- Soft labels and OOD detection: https://arxiv.org/abs/2007.03212

**Evaluation critiques**
- Sweet Danger of Sugar (SIGCOMM 2025): https://arxiv.org/abs/2507.16438
- SoK: Encrypted Network Traffic Classifiers (S&P 2025): https://arxiv.org/abs/2503.20093
- Is Network Traffic Classification in Crisis?: https://arxiv.org/abs/2506.08655
- Dos and Don'ts of ML in Computer Security (USENIX Security 2022): https://arxiv.org/abs/2010.09470
- Colossus with Feet of Clay (PQC drift): https://arxiv.org/abs/2608.22683

**Methods**
- Hinton et al., Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Malinin et al., Ensemble Distribution Distillation (ICLR 2020): https://arxiv.org/abs/1905.00076
- Müller et al., When Does Label Smoothing Help?: https://arxiv.org/abs/1906.02629
- Vaze et al., Open-Set Recognition: a Good Closed-Set Classifier Is All You Need? (ICLR 2022): https://arxiv.org/abs/2110.06207
- Liu et al., Energy-based OOD Detection (NeurIPS 2020): https://arxiv.org/abs/2010.03759
- Sun et al., OOD Detection with Deep Nearest Neighbors (ICML 2022): https://arxiv.org/abs/2204.06507
- Dhamija et al., Reducing Network Agnostophobia (OSCR): https://arxiv.org/abs/1811.04110
- Geifman et al., Bias-Reduced Uncertainty Estimation (AURC): https://arxiv.org/abs/1805.08206
- DeVries & Taylor, Learning Confidence for OOD Detection: https://arxiv.org/abs/1802.04865
- Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift (NeurIPS 2021): https://arxiv.org/abs/2106.00170

**Data and models**
- CESNET-TLS-Year22 (Scientific Data 2024): https://doi.org/10.1038/s41597-024-03927-4
- CESNET DataZoo: https://cesnet.github.io/cesnet-datazoo/
- CESNET Models: https://cesnet.github.io/cesnet-models/
- netFound: https://arxiv.org/abs/2310.17025 · weights: https://huggingface.co/snlucsb
- UNSW-IoTraffic: https://doi.org/10.5061/dryad.w0vt4b94b
- RFC 9849 (TLS Encrypted Client Hello): https://www.rfc-editor.org/info/rfc9849/
