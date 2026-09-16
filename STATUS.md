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
| Pre-registration | **Draft with 5 open decisions** | `docs/preregistration.md` |
| Unit tests | 36 pass | `tests/` |
| **Pilot on size S (week-3 gate)** | **Not run yet: needs the GPU server** | `SERVER_SETUP.md` §7 |

## Next steps
1. **On the GPU server:**
   - pilot and baselines (`SERVER_SETUP.md` §5–9);
   - if the gate passes: Track A for the three start dates (§10) and the shortcut experiment (§11). All of these use the validation week only.
2. **Settle the open pre-registration decisions** (`docs/preregistration.md`):
   - **D1:** how close Teacher B's accuracy must be to Teacher A's (proposal: within 2 macro-F1 points).
   - **D2:** hyperparameter tuning budget (proposal: none).
   - **D3:** how to compare at matched accuracy for H2.
   - **D4:** split H4 into H4a (flip-test reliance) and H4b (over-confidence transfer).
   - **D5:** keep or drop exact-duplicate test flows in the primary analysis (proposal: keep, drop in a sensitivity analysis).
3. **Before freezing:**
   - run the manual Google Scholar / IEEE Xplore novelty search;
   - check for new work from the ResAware group.
4. **Freeze the pre-registration:** set the status line to `**Status:** FROZEN`, commit, and post it on OSF. Only then run `--with-test` (§12).
5. **Still to write:** the analysis script (mixed-effects slope for H3, bootstrap CIs for H1–H5, figures); Track B (netFound on UNSW-IoTraffic); CPU deployment measurements.

## Watch items
- **`enddA` training:** its loss is large because proxy-Dirichlet precisions go up to 10⁴. If the student trains poorly on S, lower `--endd-max-precision` and record why.
- **Teacher B accuracy:** it must land close to Teacher A's (D1); in the one-epoch smoke run it was slightly higher. Check this on S before interpreting the teacher swap.
- **Laptop smoke results** (`results/*/…_smoke`) are meaningless and are not tracked by git.

## This laptop
| Item | Value |
|---|---|
| Python environment | `C:\venvs\kd-traffic` (Python 3.10, torch CPU, cesnet-datazoo 0.2.0, cesnet-models 0.4.1) |
| Data | `C:\datasets\CESNET-TLS-Year22\` (XS subset, full-year stats JSON, servicemap) |
| Array cache | `C:\datasets\cache\` (version 3) |
| Limits | 4 GB GPU, 15 GB RAM: CPU checks and smoke runs only |
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
tests/                     unit tests (pytest)
results/week1/             week-1 outputs
```
