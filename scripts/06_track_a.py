"""Track A grid for one training start date (study plan, section 4).

For start week s (training weeks s..s+3, validation week s+4) the script trains or reuses
    Teacher A   an ensemble of --ensemble mm_cesnet_v2 networks (reusable from a pilot run)
    Teacher B   one wider multimodal network (the teacher-swap control)
and, for each seed, the ~100k-parameter student under five training conditions:
    direct   cross-entropy                       ls      cross-entropy with label smoothing
    kdA      Hinton KD from Teacher A            kdB     Hinton KD from Teacher B
    enddA    ensemble distribution distillation from Teacher A's members (proxy Dirichlet targets,
             reverse KL; maximum-likelihood EnDD is unstable with ~100 classes)
The direct student is also reported after temperature scaling (condition directTS).

Models are evaluated on the validation week (known + validation unknown services). Test windows
(4 weeks each up to week 52, low-coverage weeks excluded; known + test unknown services) are used
only with --with-test, which requires docs/preregistration.md to say "**Status:** FROZEN".
--smoke-windows (with --smoke) checks the window machinery with the *validation* unknown services
and never touches the test unknowns.

Outputs (results/track_a/<run>/):
    metrics.csv         metric rows per model, split, flow group (all / >=5 packets) and unknown group
    inheritance.csv     student-teacher agreement: rank correlation of unknown-scores, error overlap
    scores_<split>.npz  per-flow energy, msp (and temperature-scaled msp_ts), post-hoc NLL of the true class
                        (nll_ts) and prediction of every model,
                        with labels, services, days, packet counts and an exact-duplicate-of-training flag
                        (input of scripts/08_analyze.py)
    training.json, config.json, log.txt, models/*.pt

Examples:
    python scripts/06_track_a.py --smoke --smoke-windows --size XS
    python scripts/06_track_a.py --size S --start 11 --teachers-from results/pilot/<run> --workers 8
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle, evaluation_windows, load_window, sequence_keys  # noqa: E402
from kdtraffic.distill import DirichletProxyAccumulator, HintonKD, ProxyDirichletKL, softmax_chunked  # noqa: E402
from kdtraffic.evaluation import Outputs, ensemble_log_probs, from_ensemble, from_logits, report_rows  # noqa: E402
from kdtraffic.inheritance import inheritance_row  # noqa: E402
from kdtraffic.metrics import fit_temperature  # noqa: E402
from kdtraffic.models import build_model, count_parameters  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import TrainConfig, predict_logits, resolve_device, train_classifier  # noqa: E402

CONDITIONS = ("direct", "ls", "kdA", "kdB", "enddA", "kdA4", "kdB4")
HPARAMS_FILE = PROJECT_ROOT / "configs" / "student_hparams.json"  # written by scripts/09_tune_students.py


def student_hparams() -> dict:
    """Tuned student settings (decision D2), or the planned defaults if tuning has not been run."""
    planned = {"kd_temperature": 4.0, "kd_alpha": 0.9, "label_smoothing": 0.1, "student_epochs": 10,
               "kd_temperature_alt": 4.0}
    if HPARAMS_FILE.exists():
        stored = json.loads(HPARAMS_FILE.read_text(encoding="utf-8"))
        planned.update(stored["selected"])
        for key in ("student_epochs", "kd_temperature_alt"):
            planned[key] = stored.get(key, planned[key])
    return planned
SUMMARY_METRICS = ["macro_f1", "auroc_energy", "auroc_msp", "fpr95_energy", "ece", "ece_ts", "aurc"]


def preregistration_frozen(path: Path) -> bool:
    return path.exists() and re.search(r"^\*\*Status:\*\*\s*FROZEN", path.read_text(encoding="utf-8"),
                                       re.MULTILINE) is not None


def reusable_teachers(source: Path | None, cache_dir: Path, log) -> bool:
    if source is None:
        return False
    config = source / "config.json"
    if not config.exists():
        log(f"WARNING: {config} not found; Teacher A will be retrained")
        return False
    cached = json.loads(config.read_text(encoding="utf-8")).get("cache_dir")
    if str(cached) != str(cache_dir):
        log(f"WARNING: pilot run used cache {cached}, this run uses {cache_dir}; Teacher A will be retrained")
        return False
    return True


def specificity_table(inheritance: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """Mean energy-score rank correlation of each condition with teachers A and B, and own minus other."""
    table = inheritance[inheritance.split == split].groupby(["condition", "teacher"])["spearman_energy"].mean().unstack()
    if {"A", "B"} <= set(table.columns):
        is_b = table.index == "kdB"
        table["own_minus_other"] = np.where(is_b, table["B"] - table["A"], table["A"] - table["B"])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--start", type=int, default=11, choices=[11, 24, 37], help="first training week")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "track_a")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--teachers-from", type=Path, default=None, help="pilot run directory with models/teacher_<i>.pt")
    hp = student_hparams()
    parser.add_argument("--epochs", type=int, default=10, help="teacher epochs")
    parser.add_argument("--student-epochs", type=int, default=hp["student_epochs"],
                        help=f"default from {HPARAMS_FILE.name} if present")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--label-smoothing", type=float, default=hp["label_smoothing"],
                        help=f"default from {HPARAMS_FILE.name} if present")
    parser.add_argument("--kd-temperature", type=float, default=hp["kd_temperature"])
    parser.add_argument("--kd-alpha", type=float, default=hp["kd_alpha"])
    parser.add_argument("--kd-temperature-alt", type=float, default=hp["kd_temperature_alt"],
                        help="temperature of the kdA4 / kdB4 conditions (conventional KD)")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="student conditions to train in this run (the rest are left to other runs)")
    parser.add_argument("--teacher-b-from", type=Path, default=None,
                        help="teacherB_wide.pt from an earlier run with the same data settings")
    parser.add_argument("--endd-max-precision", type=float, default=1e4,
                        help="upper bound on the proxy Dirichlet precision")
    parser.add_argument("--window-known-size", default="100000")
    parser.add_argument("--window-unknown-size", default="100000")
    parser.add_argument("--preregistration", type=Path, default=PROJECT_ROOT / "docs" / "preregistration.md")
    parser.add_argument("--smoke-windows", action="store_true",
                        help="with --smoke: load two windows using the validation unknown services")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.with_test and not preregistration_frozen(args.preregistration):
        raise SystemExit(f"--with-test needs a frozen pre-registration: {args.preregistration} must contain "
                         "'**Status:** FROZEN'")
    if args.smoke_windows and not args.smoke:
        raise SystemExit("--smoke-windows is only allowed together with --smoke")
    if args.smoke:
        args.seeds, args.ensemble, args.epochs, args.batch_size = 1, min(args.ensemble, 2), 1, min(args.batch_size, 512)
        args.student_epochs = 1
        args.window_known_size = args.window_unknown_size = "2000"
    args.train_weeks, args.val_weeks = f"{args.start}-{args.start + 3}", f"{args.start + 4}"
    use_test = args.with_test
    args.with_test = False  # the bundle holds train + validation; test windows are loaded separately
    spec = spec_from_args(args)
    args.with_test = use_test

    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    amp = not args.no_amp
    log(f"Run directory: {run_dir}; device {device}; start week {args.start}")
    log(f"Student settings: KD T={args.kd_temperature:g} (alt T={args.kd_temperature_alt:g}), alpha={args.kd_alpha:g}; "
        f"label smoothing {args.label_smoothing:g}; {args.student_epochs} epochs (teachers {args.epochs}); "
        f"conditions {args.conditions}" + (f" (from {HPARAMS_FILE.name})" if HPARAMS_FILE.exists() else ""))
    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    num_classes, flow_dim = bundle.num_classes, bundle.flowstats_dim
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cache_dir": bundle.cache_dir,
                                          "splits_digest": splits.digest})

    # Evaluation sets --------------------------------------------------------------------------
    train_end = spec.train_weeks[1]
    eval_sets = {"val": bundle.val}
    groups = {"val": (splits.val_unknown_near, splits.val_unknown_far)}
    weeks_since = {"val": float(spec.val_weeks[0] - train_end)}
    if use_test or args.smoke_windows:
        windows = evaluation_windows(spec.val_weeks)
        if args.smoke_windows:
            windows, unknown, near_far, kind = windows[:2], splits.val_unknown, groups["val"], "smokewin"
        else:
            unknown, near_far, kind = splits.test_unknown, (splits.test_unknown_near, splits.test_unknown_far), "test"
        for weeks in windows:
            name = f"{kind}_w{weeks[0]}-{weeks[-1]}"
            eval_sets[name] = load_window(bundle, splits, weeks, unknown, kind, int(args.window_known_size),
                                          int(args.window_unknown_size), args.data_root, args.workers, log)
            groups[name] = near_far
            weeks_since[name] = float(np.mean(weeks) - train_end)
    log("Evaluation sets: " + ", ".join(f"{k} ({len(v):,})" for k, v in eval_sets.items()))

    base = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                       seed=args.seed, device=device, amp=amp)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    training: dict[str, dict] = {}
    rows: list[dict] = []
    inheritance: list[dict] = []
    scores: dict[str, dict[str, np.ndarray]] = {split: {} for split in eval_sets}
    val_known = bundle.val.y >= 0
    y_val_known = bundle.val.y[val_known]

    def record(model: str, condition: str, seed, split: str, outputs: Outputs) -> None:
        extra = {"start": args.start, "condition": condition, "seed": seed, "weeks_since": weeks_since[split]}
        rows.extend({**r, **extra} for r in report_rows(model, split, eval_sets[split], *groups[split], outputs))
        scores[split][f"{model}__energy"] = outputs.scores["energy"].astype(np.float16)
        scores[split][f"{model}__msp"] = outputs.scores["msp"].astype(np.float16)
        if outputs.probs_ts is not None:
            scores[split][f"{model}__msp_ts"] = outputs.probs_ts.max(axis=1).astype(np.float16)
        # post-hoc (temperature-scaled) negative log-likelihood of the true class; 0 for unknown flows
        y = eval_sets[split].y
        post_hoc = outputs.probs if outputs.probs_ts is None else outputs.probs_ts
        p_true = np.where(y >= 0, post_hoc[np.arange(len(y)), np.maximum(y, 0)], 1.0)
        scores[split][f"{model}__nll_ts"] = -np.log(np.clip(p_true, 1e-12, 1.0)).astype(np.float32)
        scores[split][f"{model}__pred"] = outputs.probs.argmax(axis=1).astype(np.int16)

    def train(name: str, architecture: str, seed: int, cfg: TrainConfig, objective=None):
        model = build_model(architecture, num_classes, flow_dim, student_width=args.student_width)
        log(f"Training {name} ({architecture}, {count_parameters(model):,} parameters, seed {seed})")
        training[name] = train_classifier(model, bundle.train, bundle.val, replace(cfg, seed=seed), log, name, objective)
        training[name]["parameters"] = count_parameters(model)
        torch.save(model.state_dict(), models_dir / f"{name}.pt")
        return model

    # Teachers ---------------------------------------------------------------------------------
    reuse = reusable_teachers(args.teachers_from, bundle.cache_dir, log)
    members = []
    for i in range(args.ensemble):
        name = f"teacherA_{i}"
        candidates = [args.teachers_from / "models" / f"{stem}_{i}.pt" for stem in ("teacher", "teacherA")] if reuse else []
        source = next((c for c in candidates if c.exists()), None)
        if source is not None:
            model = build_model("mm_cesnet_v2", num_classes, flow_dim)
            model.load_state_dict(torch.load(source, map_location="cpu"))
            model.to(device)
            training[name] = {"loaded_from": str(source)}
            log(f"Loaded {name} from {source}")
        else:
            model = train(name, "mm_cesnet_v2", args.seed + i, base)
        members.append(model)
    b_source = args.teacher_b_from
    if b_source is not None and reusable_teachers(b_source.parent.parent, bundle.cache_dir, log) and b_source.exists():
        teacher_b = build_model("wide_teacher", num_classes, flow_dim)
        teacher_b.load_state_dict(torch.load(b_source, map_location="cpu"))
        teacher_b.to(device)
        training["teacherB_wide"] = {"loaded_from": str(b_source)}
        log(f"Loaded teacherB_wide from {b_source}")
    else:
        if b_source is not None:
            log(f"WARNING: cannot reuse {b_source}; Teacher B will be retrained")
        teacher_b = train("teacherB_wide", "wide_teacher", args.seed + 500, base)

    conditions = [c for c in args.conditions.split(",") if c]
    unknown_conditions = sorted(set(conditions) - set(CONDITIONS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions {unknown_conditions}; choose from {list(CONDITIONS)}")
    temperature_of = {"kdA": args.kd_temperature, "kdB": args.kd_temperature,
                      "kdA4": args.kd_temperature_alt, "kdB4": args.kd_temperature_alt}
    needed = {t for c, t in temperature_of.items() if c in conditions and c.startswith("kdA")}
    needed_b = {t for c, t in temperature_of.items() if c in conditions and c.startswith("kdB")}

    log(f"Teacher targets on the training set (A at T {sorted(needed)}, B at T {sorted(needed_b)})")
    targets_a = {t: np.zeros((len(bundle.train), num_classes), dtype=np.float32) for t in needed}
    proxy = DirichletProxyAccumulator(len(bundle.train), num_classes) if "enddA" in conditions else None
    for model in members:
        logits = predict_logits(model, bundle.train, device, amp)
        for t in needed:
            targets_a[t] += softmax_chunked(logits, t) / len(members)
        if proxy is not None:
            proxy.add(logits)
        del logits
    dirichlet_targets = proxy.concentrations(args.endd_max_precision) if proxy is not None else None
    del proxy
    b_logits_train = predict_logits(teacher_b, bundle.train, device, amp) if needed_b else None
    targets_b = {t: softmax_chunked(b_logits_train, t) for t in needed_b}
    del b_logits_train

    log("Evaluating teachers")
    val_member_logits = [predict_logits(m, bundle.val, device, amp) for m in members]
    member_t = [fit_temperature(l[val_known], y_val_known) for l in val_member_logits]
    ensemble_t = fit_temperature(ensemble_log_probs([l[val_known] for l in val_member_logits]), y_val_known)
    val_b_logits = predict_logits(teacher_b, bundle.val, device, amp)
    b_t = fit_temperature(val_b_logits[val_known], y_val_known)
    teacher_outputs: dict[str, dict[str, Outputs]] = {}
    for split, arrays in eval_sets.items():
        member_logits = val_member_logits if split == "val" else [predict_logits(m, arrays, device, amp) for m in members]
        for i, logits in enumerate(member_logits):
            record(f"teacherA_{i}", "teacherA_member", i, split, from_logits(logits, member_t[i]))
        out_a = from_ensemble(member_logits, ensemble_t)
        record("teacherA", "teacherA", "", split, out_a)
        b_logits = val_b_logits if split == "val" else predict_logits(teacher_b, arrays, device, amp)
        out_b = from_logits(b_logits, b_t)
        record("teacherB", "teacherB", "", split, out_b)
        # keep only what the inheritance comparison needs
        teacher_outputs[split] = {"A": Outputs(out_a.probs, out_a.scores), "B": Outputs(out_b.probs, out_b.scores)}
        del member_logits, b_logits, out_a, out_b
    del val_member_logits, val_b_logits

    # Students ---------------------------------------------------------------------------------
    student_base = replace(base, epochs=args.student_epochs)
    kd = lambda targets, t: HintonKD(targets[t], t, args.kd_alpha, device)  # noqa: E731
    objectives = {
        "direct": (student_base, None),
        "ls": (replace(student_base, label_smoothing=args.label_smoothing), None),
        "kdA": (student_base, kd(targets_a, args.kd_temperature) if "kdA" in conditions else None),
        "kdB": (student_base, kd(targets_b, args.kd_temperature) if "kdB" in conditions else None),
        "kdA4": (student_base, kd(targets_a, args.kd_temperature_alt) if "kdA4" in conditions else None),
        "kdB4": (student_base, kd(targets_b, args.kd_temperature_alt) if "kdB4" in conditions else None),
        "enddA": (student_base, ProxyDirichletKL(dirichlet_targets, device) if "enddA" in conditions else None),
    }
    for seed in range(args.seeds):
        for condition in conditions:
            cfg, objective = objectives[condition]
            name = f"student_{condition}_s{seed}"
            model = train(name, "student", args.seed + 1000 + 10 * seed, cfg, objective)
            val_logits = predict_logits(model, bundle.val, device, amp)
            temperature = fit_temperature(val_logits[val_known], y_val_known)
            for split, arrays in eval_sets.items():
                logits = val_logits if split == "val" else predict_logits(model, arrays, device, amp)
                variants = [(name, condition, from_logits(logits, temperature))]
                if condition == "direct":
                    variants.append((f"student_directTS_s{seed}", "directTS", from_logits(logits / temperature)))
                for variant, variant_condition, outputs in variants:
                    record(variant, variant_condition, seed, split, outputs)
                    for teacher, teacher_out in teacher_outputs[split].items():
                        row = inheritance_row(variant, teacher, split, arrays.y, outputs, teacher_out)
                        inheritance.append({**row, "start": args.start, "condition": variant_condition,
                                            "seed": seed, "weeks_since": weeks_since[split]})
            del model, val_logits
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Outputs ----------------------------------------------------------------------------------
    write_json(run_dir / "training.json", training)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    inh = pd.DataFrame(inheritance)
    inh.to_csv(run_dir / "inheritance.csv", index=False)
    train_keys = sequence_keys(bundle.train.ppi)
    for split, arrays in eval_sets.items():
        extra = {} if arrays.day is None else {"day": arrays.day}
        duplicate = np.isin(sequence_keys(arrays.ppi), train_keys)
        log(f"{split}: {duplicate.mean():.1%} of flows have an exact packet-sequence duplicate in the training window")
        np.savez(run_dir / f"scores_{split}.npz", y=arrays.y, app=arrays.app, ppi_len=arrays.ppi_len.astype(np.int8),
                 duplicate=duplicate, **extra, **scores[split])

    val = df[(df.split == "val") & (df.flows == "all") & (df.unknown == "all")]
    log("Validation week, mean over seeds / members:\n"
        + val.groupby("condition")[SUMMARY_METRICS].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    log("Energy-score rank correlation with each teacher on the validation week "
        "(own_minus_other = A - B, except B - A for kdB):\n"
        + specificity_table(inh).to_string(float_format=lambda v: f"{v:.4f}"))
    if args.smoke:
        log("Smoke run: sizes are tiny, so the numbers are meaningless; this only checks the pipeline.")


if __name__ == "__main__":
    main()
