"""RQ3: does a distilled student inherit a teacher's shortcut? (study plan, section 4.6)

A synthetic shortcut feature (one-hot group = class index mod --groups) is added to the flow
statistics. During training it agrees with the class with probability rho; otherwise it is random.

Two settings, for each rho and seed:
    both          teacher and students see the feature. Reliance on it is measured with a flip test
                  on the validation week: macro-F1 with the feature aligned to the true class minus
                  macro-F1 with the feature shifted to a wrong group (plus the share of changed
                  predictions). Students: direct (cross-entropy) and kd (Hinton KD from the teacher).
    teacher_only  only the teacher sees the feature; the KD student (no feature) is compared with
                  a directly trained student and with KD from the rho = 0 teacher. Transfer can only
                  happen through soft labels, e.g. as over-confidence (ECE) or weaker unknown detection.

Validation only; test windows are not used.

Outputs (results/shortcut/<run>/): shortcut.csv (metric rows), reliance.csv (flip test), log.txt, config.json

Examples:
    python scripts/07_shortcut.py --smoke --size XS
    python scripts/07_shortcut.py --size S --workers 8
"""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle  # noqa: E402
from kdtraffic.distill import (HintonKD, add_feature, aligned_and_flipped_codes, one_hot,  # noqa: E402
                               shortcut_codes, softmax_chunked)
from kdtraffic.evaluation import from_logits, report_rows  # noqa: E402
from kdtraffic.metrics import closed_set_metrics, fit_temperature  # noqa: E402
from kdtraffic.models import build_model  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import TrainConfig, predict_logits, resolve_device, train_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "shortcut")
    parser.add_argument("--rhos", default="0,0.5,0.9,1.0")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    hparams_file = PROJECT_ROOT / "configs" / "student_hparams.json"  # tuned settings (decision D2)
    stored = json.loads(hparams_file.read_text(encoding="utf-8")) if hparams_file.exists() else {}
    hp = stored.get("selected", {})
    parser.add_argument("--epochs", type=int, default=10, help="teacher epochs")
    parser.add_argument("--student-epochs", type=int, default=stored.get("student_epochs", 10))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--student-width", type=int, default=48)
    # the conventional-KD temperature (kdA4 arm): at the accuracy-tuned T = 1 the teacher's targets are
    # nearly one-hot and nothing teacher-specific transfers, so nothing could be inherited here either
    parser.add_argument("--kd-temperature", type=float, default=stored.get("kd_temperature_alt", 4.0))
    parser.add_argument("--kd-alpha", type=float, default=hp.get("kd_alpha", 0.9))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.with_test:
        raise SystemExit("The shortcut experiment uses the validation week only")
    if args.smoke:
        args.rhos, args.seeds, args.epochs, args.batch_size = "0,1", 1, 1, min(args.batch_size, 512)
        args.student_epochs = 1
    rhos = [float(r) for r in args.rhos.split(",")]

    spec = spec_from_args(args)
    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    amp = not args.no_amp
    log(f"KD settings: T={args.kd_temperature:g}, alpha={args.kd_alpha:g}; "
        f"student epochs {args.student_epochs}, teacher epochs {args.epochs}")
    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cache_dir": bundle.cache_dir})
    num_classes, flow_dim, groups = bundle.num_classes, bundle.flowstats_dim, args.groups
    near_far = (splits.val_unknown_near, splits.val_unknown_far)
    val, val_known = bundle.val, bundle.val.y >= 0
    base = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                       device=device, amp=amp)

    rows, reliance = [], []

    def train(name, architecture, dim, train_arrays, val_arrays, seed, objective=None):
        model = build_model(architecture, num_classes, dim, student_width=args.student_width)
        epochs = args.student_epochs if architecture == "student" else args.epochs
        train_classifier(model, train_arrays, val_arrays, replace(base, seed=seed, epochs=epochs), log, name, objective)
        return model

    def report(name, rho, seed, setting, arrays, logits):
        temperature = fit_temperature(logits[val_known], arrays.y[val_known])
        extra = {"rho": rho, "seed": seed, "setting": setting}
        rows.extend({**r, **extra} for r in report_rows(name, "val", arrays, *near_far, from_logits(logits, temperature)))

    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed + seed)
        aligned, flipped = aligned_and_flipped_codes(val.y, groups, rng)
        random_codes = shortcut_codes(val.y, groups, 0.0, rng)
        views = {name: add_feature(val, one_hot(codes, groups))
                 for name, codes in (("aligned", aligned), ("flipped", flipped), ("random", random_codes))}
        direct_plain = train(f"plain_direct_s{seed}", "student", flow_dim, bundle.train, val, args.seed + 100 + seed)
        report("student_plain_direct", "", seed, "teacher_only", val, predict_logits(direct_plain, val, device, amp))

        for rho in rhos:
            train_codes = shortcut_codes(bundle.train.y, groups, rho, rng)
            train_sc = add_feature(bundle.train, one_hot(train_codes, groups))
            val_sc = add_feature(val, one_hot(shortcut_codes(val.y, groups, rho, rng), groups))  # same rho as training
            tag = f"rho{rho:g}_s{seed}"
            teacher = train(f"teacher_{tag}", "mm_cesnet_v2", flow_dim + groups, train_sc, val_sc, args.seed + seed)
            targets = softmax_chunked(predict_logits(teacher, train_sc, device, amp), args.kd_temperature)
            kd = HintonKD(targets, args.kd_temperature, args.kd_alpha, device)
            both_direct = train(f"both_direct_{tag}", "student", flow_dim + groups, train_sc, val_sc, args.seed + 200 + seed)
            both_kd = train(f"both_kd_{tag}", "student", flow_dim + groups, train_sc, val_sc, args.seed + 300 + seed, kd)
            teacher_only_kd = train(f"teacheronly_kd_{tag}", "student", flow_dim, bundle.train, val,
                                    args.seed + 400 + seed, kd)

            for name, model in (("teacher", teacher), ("student_both_direct", both_direct), ("student_both_kd", both_kd)):
                preds = {}
                for view_name, view in views.items():
                    logits = predict_logits(model, view, device, amp)
                    preds[view_name] = logits.argmax(axis=1)
                    if view_name == "random":
                        report(name, rho, seed, "both", view, logits)
                f1 = {v: closed_set_metrics(val.y[val_known], p[val_known])["macro_f1"] for v, p in preds.items()}
                reliance.append({
                    "model": name, "rho": rho, "seed": seed,
                    "macro_f1_aligned": f1["aligned"], "macro_f1_flipped": f1["flipped"], "macro_f1_random": f1["random"],
                    "reliance_f1_drop": f1["aligned"] - f1["flipped"],
                    "prediction_change_rate": float(np.mean(preds["aligned"][val_known] != preds["flipped"][val_known])),
                })
            report("student_teacheronly_kd", rho, seed, "teacher_only", val, predict_logits(teacher_only_kd, val, device, amp))
            log(f"rho={rho:g} seed={seed}: " + ", ".join(
                f"{r['model']} drop={r['reliance_f1_drop']:.3f}" for r in reliance[-3:]))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "shortcut.csv", index=False)
    rel = pd.DataFrame(reliance)
    rel.to_csv(run_dir / "reliance.csv", index=False)
    log("Flip-test reliance (macro-F1 aligned - flipped), mean over seeds:\n"
        + rel.groupby(["rho", "model"])["reliance_f1_drop"].mean().unstack().to_string(float_format=lambda v: f"{v:.4f}"))
    view = df[(df.flows == "all") & (df.unknown == "all") & (df.setting == "teacher_only")]
    view = view.assign(rho=view["rho"].astype(str))  # the plain direct student has no rho
    log("Teacher-only setting (student never sees the feature), mean over seeds:\n"
        + view.groupby(["model", "rho"])[["macro_f1", "auroc_energy", "ece", "ece_ts"]].mean()
        .to_string(float_format=lambda v: f"{v:.4f}"))
    if args.smoke:
        log("Smoke run: sizes are tiny, so the numbers are meaningless; this only checks the pipeline.")


if __name__ == "__main__":
    main()
