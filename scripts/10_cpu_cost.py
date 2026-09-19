"""CPU deployment cost of the models (study plan, section 4.8; exploratory, not pre-registered).

For the student, one Teacher A member, the 5-member ensemble, Teacher B and XGBoost on flow statistics:
    latency      p50 / p99 per flow at batch size 1, single thread
    throughput   flows per second at --batch-size, single thread and all threads
    size         parameters, state-dict bytes, and peak resident memory during inference
in PyTorch (CPU) and, when onnxruntime is installed and the export succeeds, in ONNX Runtime.
XGBoost is retrained here (the baselines script does not save its model) with the baseline settings,
on the GPU if available, and timed on the CPU.

Inputs come from the cached arrays of the given start date; models from existing run directories.

Outputs (results/cpu_cost/<run>/): cpu_cost.csv, log.txt, config.json

Example:
    python scripts/10_cpu_cost.py --student results/track_a/<run>/models/student_direct_s0.pt \
        --teachers-from results/pilot/<run> --teacher-b results/track_a/<run>/models/teacherB_wide.pt
"""

import argparse
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.cli import RunLogger, add_data_args, make_run_dir, spec_from_args, write_json  # noqa: E402
from kdtraffic.data import build_bundle  # noqa: E402
from kdtraffic.models import build_model, count_parameters  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402


def peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().peak_wset / 1e6 if platform.system() == "Windows" \
            else psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return float("nan")


def time_calls(fn, n_warm: int, n: int) -> np.ndarray:
    for _ in range(n_warm):
        fn()
    times = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t
    return times


class TorchRunner:
    def __init__(self, models: list[torch.nn.Module]):
        self.models = [m.eval().cpu() for m in models]

    def __call__(self, ppi: torch.Tensor, stats: torch.Tensor) -> None:
        with torch.no_grad():
            for m in self.models:
                m(ppi, stats)


def export_onnx(model: torch.nn.Module, ppi: torch.Tensor, stats: torch.Tensor, path: Path, log):
    try:
        import onnxruntime as ort
        common = dict(input_names=["ppi", "flowstats"], output_names=["logits"],
                      dynamic_axes={"ppi": {0: "n"}, "flowstats": {0: "n"}, "logits": {0: "n"}})
        try:  # the legacy exporter first (smallest graphs); the cesnet-models teachers need the dynamo exporter
            torch.onnx.export(model.eval().cpu(), (ppi[:1], stats[:1]), str(path), opset_version=17, dynamo=False, **common)
        except Exception:  # noqa: BLE001
            torch.onnx.export(model.eval().cpu(), (ppi[:1], stats[:1]), str(path), opset_version=18, dynamo=True,
                              dynamic_shapes={"ppi": {0: "n"}, "flowstats": {0: "n"}}, input_names=common["input_names"],
                              output_names=common["output_names"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        log(f"  ONNX export/session failed ({type(e).__name__}: {str(e)[:120]}); PyTorch numbers only")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--student", type=Path, required=True, help="student_*.pt from a Track A run")
    parser.add_argument("--teachers-from", type=Path, required=True, help="run with models/teacher_<i>.pt")
    parser.add_argument("--teacher-b", type=Path, required=True, help="teacherB_wide.pt")
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-latency", type=int, default=2000)
    parser.add_argument("--n-throughput", type=int, default=20)
    parser.add_argument("--xgb-train-size", type=int, default=500_000)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-depth", type=int, default=8)
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "cpu_cost")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.smoke:
        args.n_latency, args.n_throughput, args.xgb_train_size, args.xgb_estimators = 50, 3, 20_000, 20

    spec = spec_from_args(args)
    run_dir = make_run_dir(args.out, spec, args.smoke)
    log = RunLogger(run_dir / "log.txt")
    splits = load_splits(args.splits)
    bundle = build_bundle(spec, splits, args.data_root, args.cache_dir, args.workers, log)
    num_classes, flow_dim = bundle.num_classes, bundle.flowstats_dim
    cpu = platform.processor() or platform.machine()
    log(f"CPU: {cpu}; {os.cpu_count()} logical cores; torch {torch.__version__}")
    write_json(run_dir / "config.json", {"args": vars(args), "spec": asdict(spec), "cpu": cpu, "cores": os.cpu_count()})

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(bundle.val), size=args.batch_size, replace=False)
    ppi = torch.from_numpy(bundle.val.ppi[idx])
    stats = torch.from_numpy(bundle.val.flowstats[idx])

    def load(name: str, path: Path) -> torch.nn.Module:
        model = build_model(name, num_classes, flow_dim, student_width=args.student_width)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()

    members = []
    for i in range(args.ensemble):
        candidates = [args.teachers_from / "models" / f"{stem}_{i}.pt" for stem in ("teacher", "teacherA")]
        members.append(load("mm_cesnet_v2", next(c for c in candidates if c.exists())))
    systems = {
        "student": [load("student", args.student)],
        "teacherA_member": [members[0]],
        "teacherA_ensemble": members,
        "teacherB_wide": [load("wide_teacher", args.teacher_b)],
    }

    rows = []
    for name, models in systems.items():
        params = sum(count_parameters(m) for m in models)
        nbytes = sum(sum(v.numel() * v.element_size() for v in m.state_dict().values()) for m in models)
        log(f"{name}: {params:,} parameters, {nbytes / 1e6:.1f} MB of weights")
        runner = TorchRunner(models)
        for backend in ("torch", "onnx"):
            if backend == "onnx":
                sessions = [export_onnx(m, ppi, stats, run_dir / f"{name}_{i}.onnx", log) for i, m in enumerate(models)]
                if any(s is None for s in sessions):
                    continue
                p1, s1 = ppi[:1].numpy(), stats[:1].numpy()
                pb, sb = ppi.numpy(), stats.numpy()
                one = lambda: [s.run(None, {"ppi": p1, "flowstats": s1}) for s in sessions]  # noqa: E731
                batch = lambda: [s.run(None, {"ppi": pb, "flowstats": sb}) for s in sessions]  # noqa: E731
                threads_all = None  # ORT sessions are created single-threaded; all-thread run skipped
            else:
                one = lambda: runner(ppi[:1], stats[:1])  # noqa: E731
                batch = lambda: runner(ppi, stats)  # noqa: E731
                threads_all = os.cpu_count()
            torch.set_num_threads(1)
            lat = time_calls(one, 50, args.n_latency) * 1e3
            thr = time_calls(batch, 3, args.n_throughput)
            row = {"system": name, "backend": backend, "parameters": params, "weights_mb": nbytes / 1e6,
                   "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p99": float(np.percentile(lat, 99)),
                   "flows_per_s_1thread": args.batch_size / float(np.median(thr))}
            if threads_all:
                torch.set_num_threads(threads_all)
                thr_all = time_calls(batch, 3, args.n_throughput)
                row["flows_per_s_all_threads"] = args.batch_size / float(np.median(thr_all))
                torch.set_num_threads(1)
            row["peak_rss_mb"] = peak_rss_mb()
            rows.append(row)
            log(f"  {backend}: p50 {row['latency_ms_p50']:.3f} ms, p99 {row['latency_ms_p99']:.3f} ms, "
                f"{row['flows_per_s_1thread']:,.0f} flows/s (1 thread)"
                + (f", {row['flows_per_s_all_threads']:,.0f} flows/s (all threads)" if threads_all else ""))

    if not args.skip_xgb:
        import xgboost as xgb
        log("Training XGBoost on flow statistics for the CPU timing")
        tr = rng.choice(len(bundle.train), size=min(args.xgb_train_size, len(bundle.train)), replace=False)
        y = bundle.train.y[tr]
        classes, y_remap = np.unique(y, return_inverse=True)
        booster = xgb.XGBClassifier(n_estimators=args.xgb_estimators, max_depth=args.xgb_depth, learning_rate=0.1,
                                    tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
                                    n_jobs=os.cpu_count(), random_state=args.seed)
        started = time.time()
        booster.fit(bundle.train.flowstats[tr], y_remap)
        log(f"  trained in {time.time() - started:.0f}s")
        booster.save_model(str(run_dir / "xgb.json"))
        cpu_model = xgb.XGBClassifier(device="cpu", n_jobs=1)
        cpu_model.load_model(str(run_dir / "xgb.json"))
        x1, xb = bundle.val.flowstats[idx[:1]], bundle.val.flowstats[idx]
        lat = time_calls(lambda: cpu_model.predict_proba(x1), 20, min(args.n_latency, 500)) * 1e3
        thr = time_calls(lambda: cpu_model.predict_proba(xb), 2, args.n_throughput)
        all_model = xgb.XGBClassifier(device="cpu", n_jobs=os.cpu_count())
        all_model.load_model(str(run_dir / "xgb.json"))
        thr_all = time_calls(lambda: all_model.predict_proba(xb), 2, args.n_throughput)
        size = (run_dir / "xgb.json").stat().st_size / 1e6
        rows.append({"system": "xgboost_flowstats", "backend": "xgboost", "parameters": np.nan, "weights_mb": size,
                     "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p99": float(np.percentile(lat, 99)),
                     "flows_per_s_1thread": args.batch_size / float(np.median(thr)),
                     "flows_per_s_all_threads": args.batch_size / float(np.median(thr_all)), "peak_rss_mb": peak_rss_mb()})
        log(f"  xgboost: p50 {rows[-1]['latency_ms_p50']:.3f} ms, {rows[-1]['flows_per_s_1thread']:,.0f} flows/s (1 thread), "
            f"model file {size:.1f} MB")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "cpu_cost.csv", index=False)
    log("\n" + df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if args.smoke:
        log("Smoke run: few repetitions; timings are indicative only.")


if __name__ == "__main__":
    main()
