# Project status: What Does the Student Inherit?

**Last updated:** 16 Sep 2026

## Where things stand
| Step | Status | Where |
|---|---|---|
| Research-gap scan | Done | `docs/research-gap-scan.md` |
| Novelty check (incl. full reading of PreDyn-IDS) | Done: gap open | `docs/novelty-check-2026-09-15.md` |
| Study plan | Approved, updated with week-1 findings | `study-plan.md` |
| Week-1 data checks (service appearance, coverage, duplicates) | Done on laptop | `scripts/00–02`, `results/week1/` |
| Known/unknown split | Generated and frozen | `configs/splits.json`, `scripts/03_make_splits.py` |
| Pilot + baselines code | Written, smoke-tested on laptop | `scripts/04–05` |
| Track A grid code (teachers A/B, 5 student conditions, test windows, inheritance, per-flow scores) | Written, smoke-tested on laptop (validation unknowns only) | `scripts/06_track_a.py`, `kdtraffic/distill.py`, `kdtraffic/inheritance.py` |
| Shortcut experiment code (RQ3) | Written, smoke-tested on laptop | `scripts/07_shortcut.py` |
| Cluster bootstrap (day × service) | Written and tested | `kdtraffic/stats.py` |
| Confirmatory analysis (pooled cluster + seed bootstrap, Holm, sensitivity analyses, figures, report) | Written, tested, smoke-run on laptop | `scripts/08_analyze.py`, `kdtraffic/analysis.py` |
| Pre-registration | **Draft.** Settled: D2 (small equal tuning grid, chosen by validation macro-F1), D6 (energy and MSP co-primary), D7 (NLL), and H1 as a shift relative to the direct student. Open: D1, D3, D4, D5. Analysis section matches the code | `docs/preregistration.md` |
| Student tuning (D2) | **Done (17 Sep).** Selected KD T=1, α=0.9 (0.9034 vs direct 0.9000; planned T=4 gave 0.8821) and LS 0.05 (0.8929). Epoch check: students gain 1.5 points at 20 epochs → students now train for 20 epochs, teachers 10 | `results/tuning/20260917-085200_S_train11-14`, `configs/student_hparams.json` |
| Remaining validation-only work (tuning, 3 start dates, shortcut) | Ready to run on the laptop, about 11–12 h | `run_validation_grid_laptop.cmd`, `logs/validation_grid_laptop.log` |
| Unit tests | 44 pass | `tests/` |
| **Pilot on size S (week-3 gate)** | **PASSED** on the laptop GPU (16 Sep): macro-F1 gap 6.2 points; energy-AUROC gap 0.018 (MSP gap 0.065) | `results/pilot/20260916-122627_S_train11-14` |
| Baselines on size S | Done: XGBoost macro-F1 0.883, energy AUROC 0.855 (beats the teacher's 0.837); k-NN 0.712 / 0.745 | `results/baselines/20260916-134400_S_train11-14` |
| Track A, start 11, validation only | Done on the laptop (16 Sep, 14:00–16:05). Teacher B macro-F1 0.962 vs A 0.964 (D1 met). With T=4, α=0.9 the KD students lose ~2 macro-F1 points and ~0.05 energy AUROC vs direct, but follow the teachers' per-flow scores much more closely (ρ 0.88 vs 0.50). Teacher-specific shift vs direct is +0.04 for kdA and kdB alike, while raw own−other is negative for kdB (→ H1 wording) | `results/track_a/20260916-140053_S_train11-14` |
| BISITE cluster access (`hpc-bisite.usal.es`, 8 × H100, SLURM) | Blocked: port 22 times out through eduVPN (VPN address 10.52.64.6); follow-up sent to Juanan | `slurm/*.sbatch` ready |

## Next steps
1. **Validation-only runs** (on the laptop while the cluster is blocked; about 8–10 h in total):
   - Track A for the three start dates (§10). Start 11 reuses the pilot teachers: `--teachers-from results\pilot\20260916-122627_S_train11-14 --workers 0`;
   - the shortcut experiment (§11).
   - Score files written before 16 Sep 14:00 lack post-hoc NLL; re-run them with the current code.
2. **Settle the open pre-registration decisions** (`docs/preregistration.md`):
   - **D1:** how close Teacher B's accuracy must be to Teacher A's (proposal: within 2 macro-F1 points).
   - **D2:** hyperparameter tuning budget (proposal: none).
   - **D3:** how to compare at matched accuracy for H2.
   - **D4:** split H4 into H4a (flip-test reliance) and H4b (over-confidence transfer).
   - **D5:** keep or drop exact-duplicate test flows in the primary analysis (proposal: keep, drop in a sensitivity analysis).
   - **D6:** primary unknown-score: energy, MSP, or both co-primary (proposal: co-primary). This came from the pilot: the energy score shrinks the teacher–student gap.
   - **D7:** settled. H2 tests post-hoc NLL.
3. **Before freezing:**
   - run the manual Google Scholar / IEEE Xplore novelty search;
   - check for new work from the ResAware group.
4. **Freeze the pre-registration** together with the analysis code: set the status line to `**Status:** FROZEN`, commit, and post it on OSF. Only then run `--with-test` and `08_analyze.py --windows test` (§12).
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
