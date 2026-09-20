"""Turn results/cpu_cost/<run>/cpu_cost.csv into paper/cpu_table.tex (a LaTeX table)."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    ("student", "onnx"): r"Student, ONNX Runtime",
    ("student", "torch"): r"Student, PyTorch",
    ("teacherA_member", "torch"): r"Teacher A, one member",
    ("teacherA_ensemble", "torch"): r"Teacher A, 5-member ensemble",
    ("teacherB_wide", "torch"): r"Teacher B",
    ("xgboost_flowstats", "xgboost"): r"XGBoost (flow statistics)",
}


def main(run: Path, out: Path = PROJECT_ROOT / "paper" / "cpu_table.tex") -> None:
    df = pd.read_csv(run / "cpu_cost.csv")
    rows = []
    for (system, backend), label in LABELS.items():
        r = df[(df.system == system) & (df.backend == backend)]
        if r.empty:
            continue
        r = r.iloc[0]
        params = "---" if pd.isna(r.parameters) else f"{r.parameters / 1e6:.1f}M" if r.parameters >= 1e6 else f"{r.parameters / 1e3:.0f}k"
        all_thr = "---" if pd.isna(r.get("flows_per_s_all_threads")) else f"{r.flows_per_s_all_threads:,.0f}"
        rows.append(f"{label} & {params} & {r.weights_mb:.1f} & {r.latency_ms_p50:.3f} & {r.latency_ms_p99:.3f} & "
                    f"{r.flows_per_s_1thread:,.0f} & {all_thr} \\\\")
    body = "\n".join(rows)
    out.write_text(rf"""\begin{{table*}}[t]
\centering
\caption{{Inference cost on a laptop CPU (AMD Ryzen 7 5800H, 8 cores), used as a stand-in for an edge device. Latency is per flow at batch size 1 on one thread; throughput is at batch size 1,024. Weights are the serialised model size. The teachers could not be exported to ONNX Runtime.}}
\label{{tab:cpu}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Parameters & Weights (MB) & Latency p50 (ms) & Latency p99 (ms) & Flows/s, 1 thread & Flows/s, all cores \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""", encoding="utf-8")
    print(f"wrote {out} from {run}")


if __name__ == "__main__":
    runs = sorted(p for p in (PROJECT_ROOT / "results" / "cpu_cost").glob("*") if p.is_dir() and not p.name.endswith("smoke"))
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else runs[-1])
