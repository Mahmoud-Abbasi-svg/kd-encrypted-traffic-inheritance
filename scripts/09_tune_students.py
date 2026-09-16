"""Equal, small tuning budget for the student training methods (pre-registration decision D2).

On start date 11 (training weeks 11-14, validation week 15), with Teacher A reused from the pilot:
    Hinton KD         T in --temperatures x alpha in --alphas   (tuned on kdA; kdB uses the same values)
    label smoothing   epsilon in --smoothings
One student (seed 0 of the grid) per setting. The setting with the highest validation macro-F1 on known
flows is selected; ties within --tie of the best go to the planned value (T=4, alpha=0.9, epsilon=0.1).
Unknown-detection and calibration metrics are deliberately not computed here, so the selection
cannot favour the hypotheses.

With --epoch-check, the direct student and the selected KD student are also trained for twice the
epochs, to show whether 10 epochs leave the KD student unconverged (reported only).

Outputs: results/tuning/<run>/tuning.csv, log.txt, and configs/student_hparams.json (read by
scripts/06_track_a.py). The config is not overwritten without --force.

Example:
    python scripts/09_tune_students.py --size S --teachers-from results/pilot/<run> --workers 0 --epoch-check
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle  # noqa: E402
from kdtraffic.distill import HintonKD, softmax_chunked  # noqa: E402
from kdtraffic.models import build_model  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import TrainConfig, predict_logits, resolve_device, train_classifier  # noqa: E402

HPARAMS_FILE = PROJECT_ROOT / "configs" / "student_hparams.json"
PLANNED = {"kd_temperature": 4.0, "kd_alpha": 0.9, "label_smoothing": 0.1}
STUDENT_SEED = 1000  # seed of student seed 0 in scripts/06_track_a.py


def floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def select(rows: pd.DataFrame, keys: list[str], tie: float) -> dict:
    """Best macro-F1; within `tie` of the best, the setting closest to the planned one wins."""
    best = rows.macro_f1.max()
    near = rows[rows.macro_f1 >= best - tie].copy()
    near["distance"] = sum((near[k] - PLANNED[k]).abs() for k in keys)
    chosen = near.sort_values(["distance", "macro_f1"], ascending=[True, False]).iloc[0]
    return {k: float(chosen[k]) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--teachers-from", type=Path, required=True, help="pilot run with models/teacher_<i>.pt")
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--temperatures", default="1,2,4")
    parser.add_argument("--alphas", default="0.5,0.9")
    parser.add_argument("--smoothings", default="0.05,0.1,0.2")
    parser.add_argument("--tie", type=float, default=0.001, help="macro-F1 difference treated as a tie")
    parser.add_argument("--epoch-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "tuning")
    parser.add_argument("--force", action="store_true", help=f"overwrite {HPARAMS_FILE.name}")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.with_test:
        raise SystemExit("Tuning uses the validation week only")
    if HPARAMS_FILE.exists() and not args.force and not args.smoke:
        raise SystemExit(f"{HPARAMS_FILE} exists; use --force to replace it (and record why in the pre-registration)")
    if args.smoke:
        args.temperatures, args.alphas, args.smoothings, args.epochs, args.ensemble = "2,4", "0.9", "0.1", 1, 2

    spec = spec_from_args(args)
    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    amp = not args.no_amp
    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    pilot_config = json.loads((args.teachers_from / "config.json").read_text(encoding="utf-8"))
    if str(pilot_config.get("cache_dir")) != str(bundle.cache_dir):
        raise SystemExit(f"{args.teachers_from} used different data settings ({pilot_config.get('cache_dir')})")
    write_json(run_dir / "config.json", {"args": vars(args), "cache_dir": bundle.cache_dir})
    num_classes, flow_dim = bundle.num_classes, bundle.flowstats_dim
    base = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                       seed=STUDENT_SEED, device=device, amp=amp)
    temperatures, alphas, smoothings = floats(args.temperatures), floats(args.alphas), floats(args.smoothings)

    log(f"Teacher A targets at T in {temperatures}")
    targets = {t: np.zeros((len(bundle.train), num_classes), dtype=np.float32) for t in temperatures}
    for i in range(args.ensemble):
        member = build_model("mm_cesnet_v2", num_classes, flow_dim)
        member.load_state_dict(torch.load(args.teachers_from / "models" / f"teacher_{i}.pt", map_location="cpu"))
        logits = predict_logits(member.to(device), bundle.train, device, amp)
        for t in temperatures:
            targets[t] += softmax_chunked(logits, t) / args.ensemble
        del member, logits

    rows = []

    def run(name: str, cfg: TrainConfig, objective=None, **setting) -> dict:
        started = time.time()
        model = build_model("student", num_classes, flow_dim, student_width=args.student_width)
        result = train_classifier(model, bundle.train, bundle.val, cfg, log, name, objective)
        best = result["history"][result["best_epoch"] - 1]
        row = {"student": name, "epochs": cfg.epochs, **setting, "best_epoch": result["best_epoch"],
               "macro_f1": best["val_macro_f1"], "val_loss": best["val_loss"],
               "minutes": round((time.time() - started) / 60, 1)}
        rows.append(row)
        log(f"{name}: macro-F1 {row['macro_f1']:.4f} (epoch {row['best_epoch']})")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return row

    run("direct", base, method="direct")
    for t in temperatures:
        for a in alphas:
            run(f"kdA_T{t:g}_a{a:g}", base, HintonKD(targets[t], t, a, device), method="kd",
                kd_temperature=t, kd_alpha=a)
    for eps in smoothings:
        run(f"ls_{eps:g}", replace(base, label_smoothing=eps), method="ls", label_smoothing=eps)

    table = pd.DataFrame(rows)
    selected = {**select(table[table.method == "kd"], ["kd_temperature", "kd_alpha"], args.tie),
                **select(table[table.method == "ls"], ["label_smoothing"], args.tie)}
    log(f"Selected: {selected}")

    if args.epoch_check:
        long = replace(base, epochs=2 * args.epochs)
        t, a = selected["kd_temperature"], selected["kd_alpha"]
        run("direct_long", long, method="direct")
        run(f"kdA_T{t:g}_a{a:g}_long", long, HintonKD(targets[t], t, a, device), method="kd",
            kd_temperature=t, kd_alpha=a)
        table = pd.DataFrame(rows)

    table.to_csv(run_dir / "tuning.csv", index=False)
    log("Validation macro-F1 (known flows):\n" + table.drop(columns=["student"]).to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    if args.smoke:
        log("Smoke run: the selection is meaningless and configs/student_hparams.json is not written.")
        return
    write_json(HPARAMS_FILE, {
        "selected": selected,
        "planned": PLANNED,
        "criterion": "highest validation macro-F1 on known flows (start 11, week 15, student seed 0); "
                     f"ties within {args.tie} go to the planned value",
        "grid": {"kd_temperature": temperatures, "kd_alpha": alphas, "label_smoothing": smoothings},
        "tuning_run": str(run_dir.relative_to(PROJECT_ROOT)),
        "date": time.strftime("%Y-%m-%d"),
    })
    log(f"Wrote {HPARAMS_FILE}")


if __name__ == "__main__":
    main()
