# Methods (draft, 20 Sep 2026)

The study was pre-registered (OSF osf.io/rts6n, frozen 19 September 2026 at repository commit `fa8f718`) after a pilot and a validation-only design phase and before any test-window result was computed. This section describes the design as registered; deviations are listed in the pre-registration's §11, and there are none that affect the analysis.

## 1. Data

**Dataset.** CESNET-TLS-Year22 [Luxemburk et al., Sci. Data 2024] contains one year (2022) of TLS flows from the Czech national research network, labelled with 180 web services from the server name. We use the S subset (25M flows) through `cesnet-datazoo` 0.2.0. Weeks 1–10 are excluded because the flow exporter was updated in week 10, which changed the packet-sequence distribution; weeks 50 and 52 are excluded because they contain less than half the median weekly volume. No service first appears mid-year (177 of 180 are present in every week), so unknown services are simulated by holding them out rather than observed emerging.

**Inputs.** Each flow is represented by its first 30 packets as three sequences (inter-packet time, direction, size), plus 12 flow statistics and a 32-bin packet-size histogram (44 tabular features). Server name, IP addresses, ports, autonomous system and JA3 fingerprint are not used, so the classifier sees what it would see under Encrypted Client Hello. Sequences and statistics are scaled with transforms fitted on each training window only.

**Known and unknown services.** A fixed split, generated once with seed 2022 before any model was trained (`configs/splits.json`), designates 102 services as known. Unknown services are held out and differ between validation and test, so no threshold or design choice made on validation unknowns can be tuned to the test unknowns. Each unknown set is divided into *far* unknowns, whole categories held out (validation: Music and Videoconferencing, 8 services; test: Search and Social, 11 services), and *near* unknowns, individual services whose category remains known (validation: 5; test: 8). The remaining 46 services are excluded, mostly for insufficient or inconsistent volume; seven that grow into usable volume after week 14 were reserved for an exploratory natural-emergence analysis.

**Time structure.** Three training start dates give three replicates of the whole design: training on weeks 11–14, 24–27 and 37–40 (1.89M, 1.37M and 1.35M known flows respectively), validation on the following week (15, 28, 41; 200k known flows plus every flow of the validation unknown services, 37–54k), and testing on consecutive 4-week windows from the week after validation to week 52, skipping the excluded weeks. This yields 9, 6 and 3 test windows, 18 in total, each a stratified sample of 100k known and 100k test-unknown flows, at 3.5 to 35.3 weeks after the end of training. Each test flow carries its day, its packet count, and a flag for whether its exact 30-packet sequence occurs in its training window (a 64-bit hash of the scaled sequence; 1.8% of test flows).

## 2. Models

**Teacher A** is an ensemble of five `mm_cesnet_v2` networks [cesnet-models 0.4.1], the dataset authors' multimodal CNN (2.26M parameters each, seeds 0–4). Its prediction is the mean of the members' softmax outputs; its energy score is the mean of the members' energies.

**Teacher B**, the teacher-swap control, is a single wider network of the same family (3 convolutional blocks of 400/600/600 channels, two 450-unit flow-statistics layers, a 1,200-unit shared layer; 10.3M parameters). It must be within 2 macro-F1 points of Teacher A on the validation week of each start date; it was within 0.2 points at all three.

**Student.** A multimodal 1D-CNN of 101,190 parameters: three convolutions (48, 96, 96 channels; kernels 5, 5, 3) over the packet sequences, pooled by mean and max; a 64-unit layer over the flow statistics; a 128-unit shared layer with dropout 0.1; and a linear classifier. It is about 22× smaller than one teacher member and 112× smaller than the ensemble.

**Baselines** (validation week only): XGBoost on the 44 flow statistics (500k training flows, 300 trees, depth 8) with the margin between the top two classes as unknown-score, and k-nearest-neighbours on the packet sequences (300k reference flows, k = 10) with the negative k-th distance.

## 3. Training conditions

Every network is trained with AdamW (learning rate 10⁻³, weight decay 10⁻⁴), a one-cycle schedule with 10% warm-up, batch size 1,024 and bf16 autocast; teachers for 10 epochs and students for 20. The kept checkpoint is the epoch with the lowest cross-entropy on the validation week's known flows. Students are trained under seven conditions, three seeds each:

- `direct`: cross-entropy on the labels.
- `ls`: cross-entropy with label smoothing ε = 0.05.
- `directTS`: the `direct` student after post-hoc temperature scaling (no extra training).
- `kdA`, `kdB`: Hinton distillation from Teacher A or B, loss α·T²·KL(teacher_T ‖ student_T) + (1 − α)·CE, with T = 1 and α = 0.9.
- `kdA4`, `kdB4`: the same at T = 4, the conventional temperature.
- `enddA`: ensemble distribution distillation from Teacher A's five members, with proxy-Dirichlet targets [Ryabinin et al., 2021] whose mean is the members' mean prediction and whose precision is estimated from member disagreement (capped at 10⁴), and a reverse-KL loss. Maximum-likelihood distribution distillation was unstable with 102 classes in the pilot and was replaced before freezing.

**Tuning.** One tuning pass on start date 11 (validation week, one seed) chose the KD temperature and weight from {1, 2, 4} × {0.5, 0.9} and the smoothing strength from {0.05, 0.1, 0.2}, by validation macro-F1 alone; no unknown-detection or calibration metric was computed during tuning. The accuracy-optimal temperature was T = 1. Because a 96%-accurate teacher's targets at T = 1 are nearly one-hot, the validation grid showed that `kdA` and `kdB` at T = 1 were indistinguishable from `direct` in their per-flow scores, leaving nothing for a teacher-swap test to detect. The T = 4 arm was therefore added, before freezing, as the arm on which inheritance (H1) and the shortcut experiment are tested, while the tuned arm carries the matched-accuracy comparisons (H2, H5). Both arms are reported throughout. The same tuning pass showed that 10 epochs left students unconverged (macro-F1 rose by 1.5 points at 20), which fixed the student budget at 20 epochs.

Post-hoc temperatures for every model are fitted by minimising NLL on the validation week's known flows.

## 4. Outcomes

For each model, window and flow group we report closed-set macro-F1 over the known services; unknown-detection AUROC and FPR at 95% TPR with two unknown-scores, the energy score (log-sum-exp of the logits) and the maximum softmax probability (MSP), treating known flows as positives; the open-set classification rate (OSCR); calibration as ECE (15 bins), NLL and Brier score, before and after temperature scaling; and the area under the risk–coverage curve. Per-flow inheritance is measured by the Spearman correlation of a student's energy scores with each teacher's, plus top-1 agreement and the Jaccard overlap of misclassified known flows. Everything is computed for all flows and for flows with at least five packets, against all, near and far unknown services.

## 5. Hypotheses and analysis

The unit of analysis is one start date × one test window (18 units). A per-unit statistic is the mean over the three student seeds; a pooled estimate is the unweighted mean over units. Uncertainty comes from 1,000 bootstrap resamples that redraw day × service clusters with replacement, *globally*, so that a cluster appearing in the windows of several start dates receives the same multiplicity everywhere, and independently redraw the three seeds per start date; resampled clusters enter every metric as integer flow weights. Intervals are 95% percentile intervals; p-values are one-sided, (1 + #{resampled estimate ≤ 0}) / 1,001, and Holm-corrected across ten hypotheses. A hypothesis with several components is supported only if every component is, so its p-value is its largest component p-value (intersection–union).

- **H1 (teacher-specific inheritance).** For the T = 4 students, the own-minus-other difference in Spearman correlation with the two teachers, minus the same difference for the `direct` student; both `kdA4` (toward A) and `kdB4` (toward B) must exceed zero. The baseline is necessary because every student correlates more with the ensemble teacher regardless of training.
- **H2 (beyond regularisation)**, once per score: `kdA` exceeds each of `ls` and `directTS` in unknown-detection AUROC, and has lower post-hoc NLL than each; interpreted as a matched-accuracy comparison only while the macro-F1 difference is at most 1 point (it was 0.99 and 0.08).
- **H3 (decay)**, once per score: positive slope of the per-unit Teacher A − `kdA` AUROC gap on weeks since training, in an ordinary-least-squares fit with one intercept per start date.
- **H4a, H4b1, H4b2 (shortcuts).** A one-hot feature with 8 values, equal to the class index modulo 8 with probability ρ and random otherwise, is appended to the flow statistics during training (ρ ∈ {0, 0.5, 0.9, 1.0}, three seeds, start date 11, evaluated on its validation week). Reliance is macro-F1 with the feature aligned to the true class minus macro-F1 with it shifted to a wrong value. H4a: with the feature visible to both teacher and student, the KD student (T = 4) relies on it more than the `direct` student at ρ ∈ {0.9, 1.0}. H4b1 and H4b2: a student that never sees the feature, distilled from a teacher trained at ρ ∈ {0.9, 1.0}, has higher ECE (H4b1) and lower energy AUROC (H4b2) than one distilled from the ρ = 0 teacher of the same seed. Paired one-sided t-tests over the six (ρ, seed) pairs.
- **H5 (distribution distillation)**, once per score: `enddA` exceeds `kdA` in unknown-detection AUROC.

Two sensitivity analyses repeat everything with exact duplicates of training flows removed and with flows of fewer than five packets removed; they are reported but are outside the Holm family. The pilot gate (Teacher A must beat the `direct` student by ≥ 2 macro-F1 points or ≥ 0.02 energy AUROC on the validation week) was passed with a 6.2-point accuracy gap.

## 6. Implementation

All code is a single Python package (PyTorch 2.14, `cesnet-datazoo` 0.2.0, `cesnet-models` 0.4.1) with 45 unit tests, including checks that every weighted bootstrap metric equals the standard metric at unit weights and that the pooled bootstrap recovers planted effects on synthetic data. The analysis script refuses to load test windows unless the pre-registration file carries the frozen status line. Experiments ran on one consumer GPU (RTX 3050 Ti, 4 GB); the full grid takes about 30 GPU-hours. Code, the split file, per-flow scores and the frozen pre-registration are released with the paper.
