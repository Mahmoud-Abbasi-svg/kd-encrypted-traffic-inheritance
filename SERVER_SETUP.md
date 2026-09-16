# Running the pilot on the GPU server

This guide takes the project from the laptop to the GPU server and runs the week-3 pilot and baselines.
Scripts `00`–`03` have already run on the laptop. Their outputs (`results/week1/`, `configs/splits.json`) are copied with the project.

## 1. Requirements
- Linux with an NVIDIA GPU (24–48 GB) and a recent driver (`nvidia-smi` works)
- Python 3.10 or newer
- **Disk:** about 15 GB for CESNET-TLS-Year22 size S (6.3 GB) and the array cache
- **RAM:** 32 GB or more. The array cache for size S weeks 11–15 is a few GB; DataZoo also loads a sample of the training set to fit scalers.

## 2. Copy the project
Copy the whole project folder except the laptop's virtual environment. It is small:
```
kdtraffic/  scripts/  tests/  configs/splits.json  results/week1/  requirements.txt  study-plan.md  SERVER_SETUP.md
```
**Do not regenerate `configs/splits.json` on the server.** It is the frozen known/unknown split, and `scripts/03_make_splits.py` refuses to overwrite it.

## 3. Environment
```bash
python3 -m venv ~/venvs/kd-traffic
source ~/venvs/kd-traffic/bin/activate
pip install --upgrade pip
# 1) torch 2.14.0 with CUDA: take the exact command for your CUDA version from https://pytorch.org/get-started/
# 2) everything else (the torch==2.14.0 pin is satisfied by the CUDA build)
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Data location
```bash
export KD_DATA_ROOT=/path/to/datasets   # e.g. /data/datasets; add to ~/.bashrc
```
DataZoo downloads `CESNET-TLS-Year22-S.h5` (6.3 GB) into `$KD_DATA_ROOT/CESNET-TLS-Year22/S/` on first use. Arrays are cached in `$KD_DATA_ROOT/cache/`.

## 5. Check the code (about 1 minute)
```bash
python -m pytest tests -q
```

## 6. Smoke run (a few minutes, tiny sizes)
This checks the whole pipeline: DataZoo loading, caching, training, evaluation and outputs. It uses size S so nothing extra is downloaded. The gate result of a smoke run is meaningless.
```bash
python scripts/05_pilot.py --smoke --size S
python scripts/04_baselines.py --smoke --size S
```

## 7. The pilot
```bash
python scripts/05_pilot.py --size S --workers 8
```
- **Training window:** weeks 11–14. **Evaluation:** validation week 15, with known services plus the validation unknown services.
- **Models:** Teacher A is 5 × `mm_cesnet_v2` (2.3M parameters each); the student is a 1D-CNN (~100k parameters) trained directly. Each trains for 10 epochs.
- **First run:** builds the array cache. The log prints the time of each step and each epoch.
- **Outputs** (`results/pilot/<run>/`):
  - `gate.json`: the go/no-go decision and the gaps;
  - `metrics.csv`: all metrics, for all flows and flows with ≥5 packets, against all, near and far unknowns;
  - `log.txt`, `training.json`, `config.json`;
  - `models/*.pt`;
  - `logits_val.npz`, for the later inheritance analysis.

## 8. Baselines (reuse the same cache)
```bash
python scripts/04_baselines.py --size S --workers 8
```
XGBoost on flow statistics (500k training flows, GPU `hist`) and k-NN on packet sequences (300k reference flows, on the GPU). Output: `results/baselines/<run>/metrics.csv`.

## 9. Reading the gate
- **Passed** when Teacher A beats the direct student by **at least 2 macro-F1 points or at least 0.02 energy AUROC** on all validation flows. Then the full Track A grid can start.
- **Not passed:** re-run with a smaller student (`--student-width 16`, about 15k parameters). If it still fails, make the task harder (PPI-10), as the plan says.
- **Baselines:** compare the teacher with `xgboost_flowstats` and `knn_ppi`. If a baseline matches the teacher, report it; that result is itself important.

## 10. Track A grid (after the gate passes; validation week only)
Run once per start date. For start 11, reuse the pilot's Teacher A: the data settings must match the pilot, which the script checks.
```bash
python scripts/06_track_a.py --smoke --smoke-windows --size S          # quick check
python scripts/06_track_a.py --size S --start 11 --teachers-from results/pilot/<run> --workers 8
python scripts/06_track_a.py --size S --start 24 --workers 8
python scripts/06_track_a.py --size S --start 37 --workers 8
```
- **Per start date:** Teacher A (5 members), Teacher B (wide, 10.3M parameters) and 3 seeds × 5 student conditions (`direct`, `ls`, `kdA`, `kdB`, `enddA`); `directTS` is reported too.
- **Outputs** (`results/track_a/<run>/`):
  - `metrics.csv`;
  - `inheritance.csv` (student–teacher rank correlation, error overlap);
  - `scores_val.npz` (per-flow scores, for bootstraps);
  - `models/`.
- **If the `enddA` student trains poorly** (its loss is large by design), try `--endd-max-precision 1000` and record the change in `docs/preregistration.md`.

## 11. Shortcut experiment (RQ3; validation week only)
```bash
python scripts/07_shortcut.py --smoke --size S
python scripts/07_shortcut.py --size S --workers 8
```
- **Runs:** ρ ∈ {0, 0.5, 0.9, 1.0} × 3 seeds.
- **Outputs:**
  - `reliance.csv`: flip-test macro-F1 drop for the teacher and the two students that see the feature;
  - `shortcut.csv`: all metrics, including the teacher-only setting.

## 12. Test windows (only after the pre-registration is frozen)
Settle the open decisions in `docs/preregistration.md`, set its status line to `**Status:** FROZEN`, commit, and register it on OSF. Only then:
```bash
python scripts/06_track_a.py --size S --start 11 --teachers-from results/pilot/<run> --with-test --workers 8
```
`--with-test` refuses to run while the file is not frozen. Test windows are 4-week blocks after the validation week up to week 52, without weeks 50 and 52, and are cached separately.

## 13. What to send back
- From the pilot: `results/pilot/<run>/gate.json`, `metrics.csv` and `log.txt`.
- From the other runs: `results/baselines/<run>/metrics.csv`, `results/track_a/<run>/{metrics,inheritance}.csv` and `results/shortcut/<run>/{reliance,shortcut}.csv`, with their `log.txt`.

Models, logits and `scores_*.npz` can stay on the server.

## Rules that protect the pre-registration
- `--with-test` loads the test window (weeks 16–19, test unknown services). **Do not use it until the pre-registration is frozen.**
- Do not change `configs/splits.json`, the week ranges or the gate thresholds after seeing pilot results without recording why in `study-plan.md`.
