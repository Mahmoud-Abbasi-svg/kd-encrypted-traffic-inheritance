# Project status: What Does the Student Inherit?

**Last updated:** 15 Sep 2026

## Where things stand
| Step | Status | Where |
|---|---|---|
| Research-gap scan | Done | `docs/research-gap-scan.md` |
| Novelty check (incl. full reading of PreDyn-IDS) | Done: gap open | `docs/novelty-check-2026-09-15.md` |
| Study plan | Approved, updated with week-1 findings | `study-plan.md` |
| Week-1 data checks (service appearance, coverage, duplicates) | Done on laptop (XS + full-year stats) | `scripts/00–02`, `results/week1/` |
| Known/unknown split | Generated and frozen | `configs/splits.json`, `scripts/03_make_splits.py` |
| Pilot + baselines code | Written, 18 unit tests pass, smoke-tested on laptop | `kdtraffic/`, `scripts/04–05`, `tests/` |
| **Pilot on size S (week-3 gate)** | **Not run yet: needs the GPU server** | `SERVER_SETUP.md` |
| Pre-registration | Not written; freeze after the pilot, before any `--with-test` run | `study-plan.md` §7 |

## Start here tonight
1. **Decide on the split rule** (optional, only before the pilot). Known services must be usable in ≥90% of study weeks, leaving 102 known. To loosen it: `python scripts/03_make_splits.py --min-usable-share 0.8 --force`, then record why in `study-plan.md`.
2. **On the GPU server**, follow `SERVER_SETUP.md`:
   ```bash
   python -m pytest tests -q
   python scripts/05_pilot.py --smoke --size S
   python scripts/05_pilot.py --size S --workers 8
   python scripts/04_baselines.py --size S --workers 8
   ```
3. **Read the gate** (`results/pilot/<run>/gate.json`): passed if Teacher A beats the direct student by ≥2 macro-F1 points or ≥0.02 energy AUROC on the validation week. If not, try `--student-width 16`, then the PPI-10 fallback.
4. **Then:**
   - write the pre-registration (§7 of the plan);
   - run the manual Google Scholar / IEEE Xplore novelty search;
   - start the Track A grid (distillation conditions: label smoothing, temperature scaling, Hinton KD from A and B, EnDD).

## This laptop
| Item | Value |
|---|---|
| Python environment | `C:\venvs\kd-traffic` (Python 3.10, torch CPU, cesnet-datazoo 0.2.0, cesnet-models 0.4.1) |
| Data | `C:\datasets\CESNET-TLS-Year22\` (XS subset, full-year stats JSON, servicemap) |
| Array cache | `C:\datasets\cache\` |
| Limits | 4 GB GPU, 15 GB RAM: CPU checks and smoke runs only |
| Run a script | `C:\venvs\kd-traffic\Scripts\python.exe scripts\<name>.py` |

## Key findings so far
- No service first appears mid-year (177/180 present every week), so unknown services are simulated by holding them out. 7 services grow into usable volume later (a small natural-emergence test).
- Traffic halves in the summer break (weeks 27–35); weeks 50 and 52 have low coverage and are excluded.
- The week-10 change was a flow-exporter update.
- Duplicates are small with exact PPI-30 (1.7%) and larger ignoring timing (12.1%). About half of 1–4-packet flows are duplicates, so metrics are also reported for ≥5 packets. Duplication falls from 23.8% (week 15) to 6.0% (week 52).
- Server IPs, ASN and JA3 appear real: natural shortcut candidates.
- 30pktTCNET dropped as a teacher (weights trained on CESNET-QUIC22 week 46 of 2022).

## File map
```
study-plan.md              the study design (RQs, hypotheses, splits, timeline, pre-registration outline)
STATUS.md                  this file
SERVER_SETUP.md            how to run the pilot on the GPU server
docs/                      gap scan and novelty check
configs/splits.json        frozen known/unknown services
kdtraffic/                 shared code: splits, data cache, models, training, metrics, evaluation, CLI
scripts/00-02              week-1 data checks
scripts/03                 split generation
scripts/04                 XGBoost and k-NN baselines
scripts/05                 week-3 pilot (Teacher A vs direct student)
tests/                     unit tests (pytest)
results/week1/             week-1 outputs
results/pilot|baselines/   laptop smoke runs (meaningless numbers; safe to delete)
```
