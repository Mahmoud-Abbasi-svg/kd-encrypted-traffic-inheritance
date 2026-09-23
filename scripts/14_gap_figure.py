"""Redraw the teacher-minus-student gap figure with its crowded left panel split (revision).

The reviewer found the original left panel unreadable: four student conditions times three start
dates, twelve curves in one column-width axis. This script splits them by behaviour - the conditions
whose gap closes, and those whose gap does not - and keeps the macro-F1 panel beside them, across the
full text width.

It reads `per_window.csv` from a finished analysis directory and writes `fig_gap_split.png` into
`paper/figures/`. It computes nothing: every value comes from the analysis output, following the same
file-to-figure rule as `scripts/11_cpu_table.py`.

    python scripts/14_gap_figure.py --analysis results/analysis/CONFIRMATORY.archive
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Conditions whose teacher gap closes over the year, and those whose does not.
CLOSING = ("kdA", "directTS")
PERSISTING = ("ls", "enddA")
LABEL = {"kdA": "kdA ($T{=}1$)", "directTS": "direct + TS", "ls": "label smoothing", "enddA": "EnDD"}
COLOUR = {"kdA": "#1f77b4", "directTS": "#2ca02c", "ls": "#d62728", "enddA": "#9467bd"}
STYLE = {11: "-", 24: "--", 37: ":"}


def gap_frame(per_window: pd.DataFrame) -> pd.DataFrame:
    """Teacher A minus student energy AUROC, per start date, window and condition."""
    view = per_window[(per_window.flows == "all") & (per_window.unknown == "all")]
    teacher = view[view.condition == "teacherA"][["start", "split", "auroc_energy_mean"]]
    teacher = teacher.rename(columns={"auroc_energy_mean": "teacher"})
    students = view[view.condition.isin(CLOSING + PERSISTING)]
    merged = students.merge(teacher, on=["start", "split"], how="inner")
    merged["gap"] = merged.teacher - merged.auroc_energy_mean
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=PROJECT_ROOT / "results" / "analysis" / "CONFIRMATORY.archive")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper" / "figures" / "fig_gap_split.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    per_window = pd.read_csv(args.analysis / "per_window.csv")
    if "analysis" in per_window.columns:
        per_window = per_window[per_window.analysis == "primary"]
    gaps = gap_frame(per_window)
    if gaps.empty:
        raise SystemExit(f"No teacherA/student rows found in {args.analysis / 'per_window.csv'}")

    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharex=True)
    for ax, group, title in ((axes[0], CLOSING, "(a) Gap closes"),
                             (axes[1], PERSISTING, "(b) Gap persists")):
        for condition in group:
            for start, part in gaps[gaps.condition == condition].groupby("start"):
                part = part.sort_values("weeks_since")
                ax.plot(part.weeks_since, part.gap, STYLE[start], color=COLOUR[condition], linewidth=1.3,
                        label=LABEL[condition] if start == 11 else None)
        ax.axhline(0.0, color="0.4", linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel("weeks since training", fontsize=7.5)
        ax.legend(fontsize=6.8, frameon=False)
    axes[0].set_ylabel("Teacher A $-$ student\nenergy AUROC", fontsize=7.5)
    # Each panel keeps its own vertical scale. On one shared scale the closing gaps, which never
    # exceed 0.02, collapse onto the zero line under curves six times larger; the caption says that
    # the scales differ.
    axes[1].set_ylabel("Teacher A $-$ student\nenergy AUROC", fontsize=7.5)

    students = per_window[(per_window.flows == "all") & (per_window.unknown == "all")
                          & (per_window.condition.isin(CLOSING + PERSISTING))]
    for condition, part in students.groupby("condition"):
        for start, series in part.groupby("start"):
            series = series.sort_values("weeks_since")
            axes[2].plot(series.weeks_since, series.macro_f1_mean, STYLE[start], color=COLOUR[condition],
                         linewidth=1.3)
    axes[2].set_title("(c) Student accuracy", fontsize=8, pad=4)
    axes[2].set_xlabel("weeks since training", fontsize=7.5)
    axes[2].set_ylabel("macro-F1", fontsize=7.5)
    lines = [plt.Line2D([], [], color="0.3", linestyle=STYLE[s], label=f"start wk {s}") for s in sorted(STYLE)]
    axes[2].legend(handles=lines, fontsize=6.8, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {args.out} from {args.analysis.name}: {len(gaps)} gap points, "
          f"{gaps.start.nunique()} start dates, conditions {sorted(gaps.condition.unique())}")


if __name__ == "__main__":
    main()
