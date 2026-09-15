"""Week-3 pilot: is the teacher-student gap large enough to study?

Trains Teacher A (an ensemble of mm_cesnet_v2 networks) and the ~100k-parameter student trained
directly, without distillation, on one training window. Both are compared on the validation week
(known services plus the validation unknown services). The test window is not loaded unless
--with-test is given, so no test result is seen before the pre-registration is frozen.

Gate (study plan, section 9): continue if Teacher A beats the direct student by >= 2 macro-F1 points
OR by >= 0.02 unknown-detection AUROC (energy score), on all validation flows.

Outputs (results/pilot/<run>/): config.json, log.txt, metrics.csv, gate.json, models/*.pt,
logits_<split>.npz (float16, in the order of the cached arrays).

Examples:
    python scripts/05_pilot.py --smoke --size XS        # end-to-end check on a laptop (minutes)
    python scripts/05_pilot.py --size S --workers 8     # the real pilot on the GPU server
"""

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle  # noqa: E402
from kdtraffic.evaluation import ensemble_log_probs, from_ensemble, from_logits, report_rows  # noqa: E402
from kdtraffic.metrics import fit_temperature  # noqa: E402
from kdtraffic.models import build_model, count_parameters  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import TrainConfig, predict_logits, resolve_device, train_classifier  # noqa: E402

GATE_F1_POINTS = 2.0
GATE_AUROC = 0.02
ENSEMBLE = "teacherA_ensemble"
STUDENT = "student_direct"
SUMMARY_COLUMNS = ["model", "flows", "unknown", "n_known", "n_unknown", "macro_f1", "auroc_energy", "auroc_msp",
                   "fpr95_energy", "oscr_energy", "ece", "ece_ts", "aurc"]


def pilot_gate(df: pd.DataFrame, split: str = "val") -> dict:
    def pick(model: str, flows: str) -> pd.Series:
        rows = df[(df.model == model) & (df.split == split) & (df.flows == flows) & (df.unknown == "all")]
        return rows.iloc[0]

    result: dict = {"split": split, "thresholds": {"macro_f1_points": GATE_F1_POINTS, "auroc_energy": GATE_AUROC}}
    for flows in ("all", "ge5"):
        teacher, student = pick(ENSEMBLE, flows), pick(STUDENT, flows)
        result[flows] = {
            "teacher_macro_f1": teacher.macro_f1,
            "student_macro_f1": student.macro_f1,
            "macro_f1_gap_points": 100 * (teacher.macro_f1 - student.macro_f1),
            "teacher_auroc_energy": teacher.auroc_energy,
            "student_auroc_energy": student.auroc_energy,
            "auroc_energy_gap": teacher.auroc_energy - student.auroc_energy,
        }
    members = df[df.model.str.match(r"teacher_\d+$") & (df.split == split) & (df.flows == "all") & (df.unknown == "all")]
    result["teacher_members_mean"] = {"macro_f1": members.macro_f1.mean(), "auroc_energy": members.auroc_energy.mean()}
    primary = result["all"]
    result["passed"] = bool(primary["macro_f1_gap_points"] >= GATE_F1_POINTS or primary["auroc_energy_gap"] >= GATE_AUROC)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "pilot")
    parser.add_argument("--ensemble", type=int, default=5, help="number of mm_cesnet_v2 members in Teacher A")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save-logits", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.ensemble, args.epochs, args.batch_size = min(args.ensemble, 2), 1, min(args.batch_size, 512)

    spec = spec_from_args(args)
    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    log(f"Run directory: {run_dir}")
    log(f"torch {torch.__version__}, device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""))

    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    log(f"Classes {bundle.num_classes}, flow-statistics features {bundle.flowstats_dim}; "
        f"train {len(bundle.train):,}, val {len(bundle.val):,} flows")
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cache_dir": bundle.cache_dir,
                                          "data_meta": bundle.meta, "splits_digest": splits.digest})

    eval_sets = {"val": bundle.val}
    unknown_groups = {"val": (splits.val_unknown_near, splits.val_unknown_far)}
    if bundle.test is not None:
        eval_sets["test"] = bundle.test
        unknown_groups["test"] = (splits.test_unknown_near, splits.test_unknown_far)

    base = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                       seed=args.seed, device=device, amp=not args.no_amp)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    logits: dict[str, dict[str, np.ndarray]] = {split: {} for split in eval_sets}
    training: dict[str, dict] = {}

    members = [f"teacher_{i}" for i in range(args.ensemble)]
    plan = [(name, "mm_cesnet_v2", args.seed + i) for i, name in enumerate(members)] + [(STUDENT, "student", args.seed + 1000)]
    for name, architecture, seed in plan:
        model = build_model(architecture, bundle.num_classes, bundle.flowstats_dim, student_width=args.student_width)
        log(f"Training {name} ({architecture}, {count_parameters(model):,} parameters, seed {seed})")
        info = train_classifier(model, bundle.train, bundle.val, replace(base, seed=seed), log, name)
        info["parameters"] = count_parameters(model)
        training[name] = info
        for split, arrays in eval_sets.items():
            logits[split][name] = predict_logits(model, arrays, device, not args.no_amp)
        torch.save(model.state_dict(), models_dir / f"{name}.pt")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(run_dir / "training.json", training)

    log("Evaluating")
    val_known = bundle.val.y >= 0
    y_val_known = bundle.val.y[val_known]
    rows: list[dict] = []
    for name in members + [STUDENT]:
        temperature = fit_temperature(logits["val"][name][val_known], y_val_known)
        for split, arrays in eval_sets.items():
            rows += report_rows(name, split, arrays, *unknown_groups[split], from_logits(logits[split][name], temperature))
    temperature = fit_temperature(ensemble_log_probs([logits["val"][m][val_known] for m in members]), y_val_known)
    for split, arrays in eval_sets.items():
        outputs = from_ensemble([logits[split][m] for m in members], temperature)
        rows += report_rows(ENSEMBLE, split, arrays, *unknown_groups[split], outputs)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    gate = pilot_gate(df)
    write_json(run_dir / "gate.json", gate)
    if not args.no_save_logits:
        for split in eval_sets:
            np.savez(run_dir / f"logits_{split}.npz", **{n: l.astype(np.float16) for n, l in logits[split].items()})

    view = df[(df.split == "val") & (df.unknown == "all")].reindex(columns=SUMMARY_COLUMNS)
    log("Validation results (unknown = all validation unknown services):\n"
        + view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    primary = gate["all"]
    log(f"GATE {'PASSED' if gate['passed'] else 'NOT PASSED'}: macro-F1 gap {primary['macro_f1_gap_points']:.2f} points "
        f"(threshold {GATE_F1_POINTS}), energy-AUROC gap {primary['auroc_energy_gap']:.4f} (threshold {GATE_AUROC})")
    if args.smoke:
        log("Smoke run: sizes are tiny, so the gate result is meaningless; this only checks the pipeline.")


if __name__ == "__main__":
    main()
