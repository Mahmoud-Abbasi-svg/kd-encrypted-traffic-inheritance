"""Feature-space unknown-traffic scores for models that are already trained (exploratory, revision).

A reviewer objected that the study evaluates no dedicated open-set detector, and that with the logit
scores the teacher has almost no detection advantage over the student - so "the student does not
inherit detection quality" cannot be told apart from "there was no detection quality to inherit".
This script answers both by scoring every existing model with a Mahalanobis and a k-nearest-neighbour
detector built on its penultimate features (`kdtraffic/openset.py`), using the checkpoints and cached
windows already on disk. Nothing is retrained and no confirmatory output is touched.

For each model it fits on the training window (class means, a tied shrunk covariance, and a small
reference sample), then scores the validation week and every test window. The ensemble's score is the
mean of its members', matching how the energy score is aggregated in the frozen code.

Outputs (results/feature_scores/<run>/): scores_<split>.npz with `{model}__maha` / `{model}__knnfeat`
(float32; distances overflow float16), metrics.csv, log.txt, config.json.

Examples:
    python scripts/13_feature_scores.py --smoke --size XS --students-from results/track_a/<smoke run>
    python scripts/13_feature_scores.py --size S --start 11 --with-test \
        --students-from results/track_a/20260919-113607_S_train11-14 \
        --teachers-from results/pilot/20260916-122627_S_train11-14 \
        --teacher-b-from results/track_a/20260916-140053_S_train11-14/models/teacherB_wide.pt
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle, evaluation_windows, load_window, sequence_keys  # noqa: E402
from kdtraffic.evaluation import Outputs, report_rows  # noqa: E402
from kdtraffic.metrics import detection_metrics  # noqa: E402
from kdtraffic.models import build_model  # noqa: E402
from kdtraffic.openset import FeatureScorer, accumulate_feature_stats, extract_features, score_arrays  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import predict_logits, resolve_device  # noqa: E402

STUDENT_CONDITIONS = ("direct", "ls", "kdA", "kdB", "kdA4", "kdB4", "enddA")


def detection_rows(model: str, split: str, arrays, near, far, scores: dict[str, np.ndarray]) -> list[dict]:
    """Detection-only rows, for a model whose class probabilities are not kept (the ensemble).

    `report_rows` needs a probability matrix to report macro-F1 and calibration. The ensemble's mean
    probabilities are not retained here - only its members' scores are - so those columns are left
    out rather than filled with a placeholder that would look like a real measurement.
    """
    known = arrays.y >= 0
    unknown_groups = {"all": ~known, "near": np.isin(arrays.app, np.array(near, dtype=str)),
                      "far": np.isin(arrays.app, np.array(far, dtype=str))}
    flow_groups = {"all": np.ones(len(arrays), dtype=bool), "ge5": arrays.ppi_len >= 5}
    rows = []
    for flows, flow_mask in flow_groups.items():
        for unknown, unknown_mask in unknown_groups.items():
            mask = flow_mask & (known | unknown_mask)
            inside = arrays.y[mask] >= 0
            row = {"model": model, "split": split, "flows": flows, "unknown": unknown,
                   "n_known": int(inside.sum()), "n_unknown": int((~inside).sum())}
            for name, value in scores.items():
                selected = value[mask]
                found = detection_metrics(selected[inside], selected[~inside])
                row[f"auroc_{name}"], row[f"fpr95_{name}"] = found["auroc"], found["fpr95"]
            rows.append(row)
    return rows


def student_checkpoints(models_dir: Path, conditions, seeds) -> list[tuple[str, str, Path]]:
    found = []
    for condition in conditions:
        for seed in seeds:
            path = models_dir / f"student_{condition}_s{seed}.pt"
            if path.exists():
                found.append((f"student_{condition}_s{seed}", "student", path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--start", type=int, default=11, choices=[11, 24, 37])
    parser.add_argument("--students-from", type=Path, required=True, help="run directory with models/student_*.pt")
    parser.add_argument("--teachers-from", type=Path, default=None, help="run with models/teacher[A]_<i>.pt")
    parser.add_argument("--teacher-b-from", type=Path, default=None, help="path to teacherB_wide.pt")
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--conditions", default=",".join(STUDENT_CONDITIONS))
    parser.add_argument("--train-stats-size", type=int, default=300_000, help="training flows used to fit each scorer")
    parser.add_argument("--knn-reference", type=int, default=50_000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--window-known-size", default="100000")
    parser.add_argument("--window-unknown-size", default="100000")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "feature_scores")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.smoke:
        args.train_stats_size, args.knn_reference, args.seeds = 4_000, 1_000, 1
        args.window_known_size = args.window_unknown_size = "2000"

    use_test = args.with_test
    args.train_weeks, args.val_weeks = f"{args.start}-{args.start + 3}", f"{args.start + 4}"
    args.with_test = False
    spec = spec_from_args(args)
    args.with_test = use_test

    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    log(f"Feature-space open-set scores; device {device}; start week {args.start}")
    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cache_dir": bundle.cache_dir})
    num_classes, flow_dim = bundle.num_classes, bundle.flowstats_dim

    train_end = spec.train_weeks[1]
    eval_sets = {"val": bundle.val}
    groups = {"val": (splits.val_unknown_near, splits.val_unknown_far)}
    weeks_since = {"val": float(spec.val_weeks[0] - train_end)}
    if use_test:
        for weeks in evaluation_windows(spec.val_weeks):
            name = f"test_w{weeks[0]}-{weeks[-1]}"
            eval_sets[name] = load_window(bundle, splits, weeks, splits.test_unknown, "test",
                                          int(args.window_known_size), int(args.window_unknown_size),
                                          args.data_root, args.workers, log)
            groups[name] = (splits.test_unknown_near, splits.test_unknown_far)
            weeks_since[name] = float(np.mean(weeks) - train_end)
    log("Evaluation sets: " + ", ".join(f"{k} ({len(v):,})" for k, v in eval_sets.items()))

    # Which models to score ---------------------------------------------------------------------
    wanted: list[tuple[str, str, Path]] = []
    if args.teachers_from is not None:
        for i in range(args.ensemble):
            for stem in ("teacher", "teacherA"):
                path = args.teachers_from / "models" / f"{stem}_{i}.pt"
                if path.exists():
                    wanted.append((f"teacherA_{i}", "mm_cesnet_v2", path))
                    break
    if args.teacher_b_from is not None and args.teacher_b_from.exists():
        wanted.append(("teacherB", "wide_teacher", args.teacher_b_from))
    wanted += student_checkpoints(args.students_from / "models", args.conditions.split(","), range(args.seeds))
    if not wanted:
        raise SystemExit("No checkpoints found to score")
    log(f"Scoring {len(wanted)} models: " + ", ".join(name for name, _, _ in wanted))

    rng = np.random.default_rng(args.seed)
    known_train = np.flatnonzero(bundle.train.y >= 0)
    reference_index = np.sort(rng.choice(known_train, size=min(args.knn_reference, len(known_train)), replace=False))

    scores: dict[str, dict[str, np.ndarray]] = {split: {} for split in eval_sets}
    rows: list[dict] = []
    member_scores: dict[str, dict[str, list[np.ndarray]]] = {split: {} for split in eval_sets}

    for name, architecture, path in wanted:
        started = time.time()
        model = build_model(architecture, num_classes, flow_dim, student_width=args.student_width)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        stats = accumulate_feature_stats(model, bundle.train, device, amp=False, batch_size=args.batch_size,
                                         max_flows=args.train_stats_size, seed=args.seed, num_classes=num_classes)
        reference = extract_features(model, bundle.train, reference_index, device, amp=False,
                                     batch_size=args.batch_size)
        scorer = FeatureScorer.fit(stats, reference, shrinkage=args.shrinkage, k=args.knn_k)
        for split, arrays in eval_sets.items():
            values = score_arrays(model, arrays, scorer, device, amp=False, batch_size=args.batch_size)
            for score_name, value in values.items():
                scores[split][f"{name}__{score_name}"] = value.astype(np.float32)
                if name.startswith("teacherA_"):
                    member_scores[split].setdefault(score_name, []).append(value)
            logits = predict_logits(model, arrays, device, amp=False, batch_size=args.batch_size)
            outputs = Outputs(softmax(logits, axis=1), values)
            rows.extend({**r, "start": args.start, "weeks_since": weeks_since[split]}
                        for r in report_rows(name, split, arrays, *groups[split], outputs))
            del logits, outputs, values
        log(f"  {name}: {stats.dim}-d features, {time.time() - started:.0f}s")
        del model, stats, reference, scorer

    # The ensemble's score is the mean of its members', as the energy score already is.
    for split, arrays in eval_sets.items():
        if not member_scores[split]:
            continue
        combined = {n: np.mean(v, axis=0).astype(np.float32) for n, v in member_scores[split].items()}
        for score_name, value in combined.items():
            scores[split][f"teacherA__{score_name}"] = value
        rows.extend({**r, "start": args.start, "weeks_since": weeks_since[split]}
                    for r in detection_rows("teacherA", split, arrays, *groups[split], combined))

    for split, arrays in eval_sets.items():
        extra = {} if arrays.day is None else {"day": arrays.day}
        np.savez(run_dir / f"scores_{split}.npz", y=arrays.y, app=arrays.app,
                 ppi_len=arrays.ppi_len.astype(np.int8), **extra, **scores[split])
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "metrics.csv", index=False)

    view = frame[(frame.flows == "all") & (frame.unknown == "all")]
    summary = view.groupby(["split", "model"])[["auroc_maha", "auroc_knnfeat"]].mean()
    log("AUROC by feature-space score (all flows, all unknowns):\n"
        + summary.to_string(float_format=lambda v: f"{v:.4f}"))
    if args.smoke:
        log("Smoke run: tiny sizes, the numbers are meaningless; this only checks the pipeline.")


if __name__ == "__main__":
    main()
