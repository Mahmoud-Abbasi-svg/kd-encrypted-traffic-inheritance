"""Baselines the teacher must beat: XGBoost on flow statistics and k-NN on packet sequences.

Uses the same cached arrays as the pilot (same --size / weeks / sizes / splits), so run either first.
Evaluated on the validation week only unless --with-test is given.

    XGBoost  unknown-scores: max softmax probability (msp) and energy of the margins
    k-NN     class votes of the k nearest training flows (scaled PPI, Euclidean); unknown-score is
             the negative distance to the k-th neighbour (neg_kth_distance) and the vote share (msp)

Outputs (results/baselines/<run>/): config.json, log.txt, metrics.csv

Examples:
    python scripts/04_baselines.py --smoke --size XS
    python scripts/04_baselines.py --size S --workers 8
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import Arrays, build_bundle  # noqa: E402
from kdtraffic.evaluation import Outputs, report_rows  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402
from kdtraffic.train import resolve_device  # noqa: E402


def sample(arrays: Arrays, n: int, rng: np.random.Generator) -> Arrays:
    if n >= len(arrays):
        return arrays
    return arrays.subset(np.sort(rng.choice(len(arrays), size=n, replace=False)))


def xgboost_baseline(train: Arrays, val: Arrays, num_classes: int, args, device: str, log):
    from xgboost import XGBClassifier

    classes = np.unique(train.y)
    remap = np.full(num_classes, -1, dtype=np.int64)
    remap[classes] = np.arange(len(classes))
    val_known = val.subset(np.flatnonzero((val.y >= 0) & (remap[np.clip(val.y, 0, None)] >= 0)))
    model = XGBClassifier(
        n_estimators=args.xgb_estimators, max_depth=args.xgb_depth, learning_rate=args.xgb_lr,
        tree_method="hist", device="cuda" if device.startswith("cuda") else "cpu",
        early_stopping_rounds=20, eval_metric="mlogloss", n_jobs=-1, random_state=args.seed,
    )
    started = time.time()
    model.fit(train.flowstats, remap[train.y], eval_set=[(val_known.flowstats, remap[val_known.y])], verbose=False)
    log(f"XGBoost trained on {len(train):,} flows ({len(classes)} classes) in {time.time() - started:.0f}s; "
        f"best iteration {model.best_iteration}")

    def outputs(arrays: Arrays) -> Outputs:
        margin = model.predict(arrays.flowstats, output_margin=True).astype(np.float32)
        probs = np.zeros((len(arrays), num_classes), dtype=np.float32)
        probs[:, classes] = softmax(margin, axis=1)
        return Outputs(probs, {"msp": probs.max(axis=1), "energy": logsumexp(margin, axis=1)})

    return outputs


def knn_baseline(reference: Arrays, num_classes: int, k: int, device: str, log):
    ref = torch.from_numpy(reference.ppi.reshape(len(reference), -1)).to(device)
    ref_y = torch.from_numpy(reference.y).to(device)
    k = min(k, len(reference))
    chunk = max(1, int(5e7 // len(reference)))
    log(f"k-NN reference set: {len(reference):,} flows, k={k}, query chunk {chunk}")

    def outputs(arrays: Arrays) -> Outputs:
        probs = np.zeros((len(arrays), num_classes), dtype=np.float32)
        kth = np.zeros(len(arrays), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(arrays), chunk):
                query = torch.from_numpy(arrays.ppi[start:start + chunk].reshape(-1, ref.shape[1])).to(device)
                dist, idx = torch.topk(torch.cdist(query, ref), k, largest=False)
                votes = torch.zeros(len(query), num_classes, device=device)
                votes.scatter_add_(1, ref_y[idx], torch.ones_like(dist))
                probs[start:start + len(query)] = (votes / k).cpu().numpy()
                kth[start:start + len(query)] = dist[:, -1].cpu().numpy()
        return Outputs(probs, {"msp": probs.max(axis=1), "neg_kth_distance": -kth})

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "baselines")
    parser.add_argument("--xgb-train-size", type=int, default=500_000)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-depth", type=int, default=8)
    parser.add_argument("--xgb-lr", type=float, default=0.1)
    parser.add_argument("--knn-ref-size", type=int, default=300_000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.smoke:
        args.xgb_train_size, args.xgb_estimators, args.knn_ref_size = 5_000, 20, 5_000

    spec = spec_from_args(args)
    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    device = resolve_device(args.device)
    log(f"Run directory: {run_dir}; device {device}")

    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cache_dir": bundle.cache_dir,
                                          "splits_digest": splits.digest})
    eval_sets = {"val": (bundle.val, splits.val_unknown_near, splits.val_unknown_far)}
    if bundle.test is not None:
        eval_sets["test"] = (bundle.test, splits.test_unknown_near, splits.test_unknown_far)

    rng = np.random.default_rng(args.seed)
    baselines = {}
    if not args.skip_xgb:
        baselines["xgboost_flowstats"] = xgboost_baseline(sample(bundle.train, args.xgb_train_size, rng), bundle.val,
                                                          bundle.num_classes, args, device, log)
    if not args.skip_knn:
        baselines["knn_ppi"] = knn_baseline(sample(bundle.train, args.knn_ref_size, rng), bundle.num_classes,
                                            args.knn_k, device, log)

    rows = []
    for name, predict in baselines.items():
        for split, (arrays, near, far) in eval_sets.items():
            started = time.time()
            rows += report_rows(name, split, arrays, near, far, predict(arrays))
            log(f"{name} evaluated on {split} ({len(arrays):,} flows) in {time.time() - started:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    columns = ["model", "flows", "n_known", "n_unknown", "macro_f1"] + [c for c in df.columns if c.startswith("auroc_")] + ["ece", "aurc"]
    view = df[(df.split == "val") & (df.unknown == "all")].reindex(columns=columns)
    log("Validation results (unknown = all validation unknown services):\n"
        + view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
