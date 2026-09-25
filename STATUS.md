# Project status: Inheritance in Distilled Encrypted-Traffic Classifiers

**Last updated:** 16 Sep 2026

## Where things stand
| Step | Status | Where |
|---|---|---|
| Research-gap scan | Done | `docs/research-gap-scan.md` |
| Novelty check (incl. full reading of PreDyn-IDS) | Done: gap open | `docs/novelty-check-2026-09-15.md` |
| Novelty re-check before freezing (19 Sep) | Gap still open; new must-cite work found; 2 findings need a defence against contrary literature | `docs/novelty-check-2026-09-19.md` |
| Study plan | Approved, updated with week-1 findings | `study-plan.md` |
| Week-1 data checks (service appearance, coverage, duplicates) | Done on laptop | `scripts/00–02`, `results/week1/` |
| Known/unknown split | Generated and frozen | `configs/splits.json`, `scripts/03_make_splits.py` |
| Pilot + baselines code | Written, smoke-tested on laptop | `scripts/04–05` |
| Track A grid code (teachers A/B, 5 student conditions, test windows, inheritance, per-flow scores) | Written, smoke-tested on laptop (validation unknowns only) | `scripts/06_track_a.py`, `kdtraffic/distill.py`, `kdtraffic/inheritance.py` |
| Shortcut experiment code (RQ3) | Written, smoke-tested on laptop | `scripts/07_shortcut.py` |
| Cluster bootstrap (day × service) | Written and tested | `kdtraffic/stats.py` |
| Confirmatory analysis (pooled cluster + seed bootstrap, Holm, sensitivity analyses, figures, report) | Written, tested, smoke-run on laptop | `scripts/08_analyze.py`, `kdtraffic/analysis.py` |
| **Pre-registration** | **FROZEN 19 Sep 2026** at commit `fa8f718`, decisions D1–D8 settled on validation data only. Changes now only as dated entries in §11. Registered on OSF 20 Sep 2026: https://osf.io/rts6n (public), project https://osf.io/xcgpb | `docs/preregistration.md` |
| Pre-registration (history) | **was draft.** Settled: D2 (small equal tuning grid, chosen by validation macro-F1), D6 (energy and MSP co-primary), D7 (NLL), and H1 as a shift relative to the direct student. Open: D1, D3, D4, D5. Analysis section matches the code | `docs/preregistration.md` |
| Student tuning (D2) | **Done (17 Sep).** Selected KD T=1, α=0.9 (0.9034 vs direct 0.9000; planned T=4 gave 0.8821) and LS 0.05 (0.8929). Epoch check: students gain 1.5 points at 20 epochs → students now train for 20 epochs, teachers 10 | `results/tuning/20260917-085200_S_train11-14`, `configs/student_hparams.json` |
| Remaining validation-only work (tuning, 3 start dates, shortcut) | Ready to run on the laptop, about 11–12 h | `run_validation_grid_laptop.cmd`, `logs/validation_grid_laptop.log` |
| Unit tests | 44 pass | `tests/` |
| **Pilot on size S (week-3 gate)** | **PASSED** on the laptop GPU (16 Sep): macro-F1 gap 6.2 points; energy-AUROC gap 0.018 (MSP gap 0.065) | `results/pilot/20260916-122627_S_train11-14` |
| Baselines on size S | Done: XGBoost macro-F1 0.883, energy AUROC 0.855 (beats the teacher's 0.837); k-NN 0.712 / 0.745 | `results/baselines/20260916-134400_S_train11-14` |
| Track A validation-only runs with tuned settings (17 Sep) | start 11 `results/track_a/20260917-110225_S_train11-14` and start 24 `results/track_a/20260917-121305_S_train24-27` done; **start 37 interrupted, must be rerun**. With the tuned T=1, KD equals direct training in accuracy *and* in per-flow scores (ρ 0.50 vs 0.51) → the T=4 inheritance arm was added (D8) | `logs/validation_grid_laptop.log` |
| Conventional-KD arm (kdA4, kdB4) | Code added and smoke-tested; supplementary pass still to run for starts 11 and 24 (start 37 includes it) | `run_kd4_laptop.cmd` |
| Track A, start 11 (10-epoch students, 16 Sep) | Done on the laptop (16 Sep, 14:00–16:05). Teacher B macro-F1 0.962 vs A 0.964 (D1 met). With T=4, α=0.9 the KD students lose ~2 macro-F1 points and ~0.05 energy AUROC vs direct, but follow the teachers' per-flow scores much more closely (ρ 0.88 vs 0.50). Teacher-specific shift vs direct is +0.04 for kdA and kdB alike, while raw own−other is negative for kdB (→ H1 wording) | `results/track_a/20260916-140053_S_train11-14` |
| BISITE cluster access (`hpc-bisite.usal.es`, 8 × H100, SLURM) | Blocked: port 22 times out through eduVPN (VPN address 10.52.64.6); follow-up sent to Juanan | `slurm/*.sbatch` ready |

## CONFIRMATORY RESULTS (test windows, 19 Sep 2026) — `docs/results-confirmatory-2026-09-19.md`
18 test windows, 200k flows each, 1,000 cluster bootstraps, Holm over 10 hypotheses. **Supported: H1** (inheritance is teacher-specific: kdA4 +0.007, kdB4 +0.026, Holm p = 0.009) and **H4b1** (a shortcut-reliant teacher passes on over-confidence, ECE +0.009, Holm p = 0.005). The other eight are not supported, and two are reversed: the teacher–student gap **shrinks** over time (H3, slope −0.0004/week; after ~3 months the student detects unknowns better than its teacher), and a shortcut-reliant teacher's student detects unknowns *better*, not worse (H4b2). Both sensitivity analyses agree with the primary analysis everywhere.

## Validation-week findings (all 3 start dates, 745k flows, plus the shortcut experiment; NOT confirmatory)
From `results/analysis/val_all_with_shortcut/` (`scripts/08_analyze.py --windows val`). Numbers below are from starts 11 + 24; the three-start values are within 0.003 of them.

**Shortcut experiment** (`results/shortcut/20260918-234156_S_train11-14`, flip-test reliance, mean of 3 seeds):

| ρ | Teacher | direct student | KD student |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.037 | 0.069 | 0.068 |
| 0.9 | 0.103 | 0.185 | 0.196 |
| 1.0 | 0.800 | 0.945 | 0.870 |

- **H4a not supported** (−0.032, p = 0.92): the KD student does not rely on the shortcut more than the direct student. The real pattern is capacity: both 101k students rely about twice as much as the 2.3M-parameter teacher.
- **H4b splits:** a shortcut-reliant teacher does pass on over-confidence (ECE +0.009 vs the ρ = 0 teacher, p = 0.0005), but its student's unknown detection gets *better*, not worse (energy AUROC +0.020), so the hypothesis as written (both parts) fails. See D4.


| Hypothesis | Result | Numbers |
|---|---|---|
| **H1 inheritance** | **supported** | kdA4 shifts +0.048 toward A (CI 0.040–0.057), kdB4 +0.041 toward B (0.033–0.049), Holm p = 0.017. The tuned T=1 students: +0.005 and −0.004 |
| **H2 beyond regularisation** | not supported | kdA vs ls −0.011 energy AUROC, vs directTS −0.003; NLL better than ls (+0.024) but only +0.002 vs directTS. Accuracy matched within 0.9 points (D3 rule satisfied) |
| **H5 EnDD** | score-dependent | MSP +0.012 (supported), energy −0.048 (clearly worse) — the case for keeping both scores co-primary |
| Teacher − kdA gap | — | 0.024 energy AUROC, 0.045 MSP |

Reading: distillation transfers the teacher's per-flow score pattern, and it is teacher-specific, but it does not transfer the teacher's unknown-detection quality; label smoothing does as well or better. How much transfers is set by the temperature (D8).

## Next steps
1. **All validation-week runs are done** (19 Sep). Run directories: start 11 `20260917-110225` + `20260918-121700`, start 24 `20260917-121305` + `20260918-124618`, start 37 `20260918-132337`, shortcut `20260918-234156`. Folders ending `_stopped` are interrupted runs and must not be used. Rerun the analysis with:
   `python scripts\08_analyze.py --windows val --shortcut results\shortcut\20260918-234156_S_train11-14 --runs <the five above>`
   Laptop rules: one job at a time, lid open.
2. **Settle the open pre-registration decisions** (`docs/preregistration.md`):
   - **D1:** how close Teacher B's accuracy must be to Teacher A's (proposal: within 2 macro-F1 points).
   - **D2:** hyperparameter tuning budget (proposal: none).
   - **D3:** how to compare at matched accuracy for H2.
   - **D4:** split H4 into H4a (flip-test reliance) and H4b (over-confidence transfer). **Proposal after the validation run:** keep H4a, and split H4b into two separate hypotheses — H4b1 (ECE higher, supported on validation) and H4b2 (energy AUROC lower, contradicted on validation) — instead of bundling two independent predictions into one test.
   - **D5:** keep or drop exact-duplicate test flows in the primary analysis (proposal: keep, drop in a sensitivity analysis).
   - **D6:** primary unknown-score: energy, MSP, or both co-primary (proposal: co-primary). This came from the pilot: the energy score shrinks the teacher–student gap.
   - **D7:** settled. H2 tests post-hoc NLL.
3. **Before freezing:** the automated re-check is done (19 Sep). Still by hand: IEEE Xplore KD-for-traffic papers, a dblp listing of the Luxemburk / Hynek / Čejka 2026 output, and **NTC-R 2026 at CoNEXT (programme published 15 Oct 2026)** — the highest scoop risk, topics include shortcut learning and negative results.
4. **Done (19 Sep):** test-window runs (`results/track_a/20260919-*`) and the confirmatory analysis (`results/analysis/CONFIRMATORY/`). Still to do: post the frozen pre-registration on OSF and record the link as a §11 entry.
5. **Writing.** The paper's result is mostly negative and that is the contribution: the teacher's score pattern transfers teacher-specifically, but detection quality, calibration and shortcut robustness do not; temperature scaling of a directly trained student matches distillation on every unknown-detection outcome; the compressed student ages at least as well as its teacher. Defend the two findings that run against the literature (see `docs/novelty-check-2026-09-19.md`).
5. **Still to write:** Track B (netFound on UNSW-IoTraffic); CPU deployment measurements (ONNX Runtime); conformal recalibration (exploratory).

## Watch items
- **`enddA` training:** its loss is large because proxy-Dirichlet precisions go up to 10⁴. If the student trains poorly on S, lower `--endd-max-precision` and record why.
- **Teacher B accuracy:** it must land close to Teacher A's (D1); in the one-epoch smoke run it was slightly higher. Check this on S before interpreting the teacher swap.
- **H3 analysis changed before any results:** the slope uses start-date fixed effects instead of a mixed model (3 start dates cannot support a random-effect variance). Recorded in `study-plan.md` and the pre-registration draft.
- **Validation-only grid runs (§10) lack test windows;** the analysis needs the `--with-test` runs of §12.
- **Laptop smoke results** (`results/*/…_smoke`) are meaningless and are not tracked by git.

## This laptop
| Item | Value |
|---|---|
| Python environment | `C:\venvs\kd-traffic` (Python 3.10, torch 2.14.0+cu130, cesnet-datazoo 0.2.0, cesnet-models 0.4.1) |
| Data | `C:\datasets\CESNET-TLS-Year22\` (XS and S, full-year stats JSON, servicemap) |
| Array cache | `C:\datasets\cache\` (version 3) |
| Limits | RTX 3050 Ti 4 GB, 15 GB RAM: the size-S pilot fits (teacher ~23k flows/s, ~1 GB of arrays); the full grid belongs on the cluster. Close Chrome, VS Code, PostgreSQL and Notion before long runs |
| Run a script | `C:\venvs\kd-traffic\Scripts\python.exe scripts\<name>.py` |
| Run the tests | `C:\venvs\kd-traffic\Scripts\python.exe -m pytest tests -q` |

## Key findings so far
- No service first appears mid-year (177/180 present every week), so unknown services are simulated by holding them out. 7 services grow into usable volume later (a small natural-emergence set).
- Traffic halves in the summer break (weeks 27–35); weeks 50 and 52 have low coverage and are excluded.
- The week-10 change was a flow-exporter update.
- Duplicates are small with exact PPI-30 (1.7%) and larger ignoring timing (12.1%). About half of 1–4-packet flows are duplicates, so metrics are also reported for ≥5 packets. Duplication falls from 23.8% (week 15) to 6.0% (week 52).
- Server IPs, ASN and JA3 appear real: natural shortcut candidates.
- 30pktTCNET dropped as a teacher (weights trained on CESNET-QUIC22 week 46 of 2022).
- **DataZoo quirks:** FIXED selection forbids `disabled_apps` and ignores the min-train-samples check; the validation loader serves known flows only, so evaluation weeks are loaded through the test loader with pre-fitted scalers.

## File map
```
study-plan.md              the study design
STATUS.md                  this file
SERVER_SETUP.md            how to run everything on the GPU server
docs/                      gap scan, novelty check, pre-registration draft
configs/splits.json        frozen known/unknown services
kdtraffic/                 shared code: splits, data, models, train, distill, metrics, evaluation,
                           inheritance, stats, cli
scripts/00-02              week-1 data checks
scripts/03                 split generation
scripts/04                 XGBoost and k-NN baselines
scripts/05                 week-3 pilot (Teacher A vs direct student)
scripts/06                 Track A grid for one start date
scripts/07                 synthetic-shortcut experiment (RQ3)
scripts/08                 confirmatory analysis (H1–H5), report and figures
tests/                     unit tests (pytest)
results/week1/             week-1 outputs
```
