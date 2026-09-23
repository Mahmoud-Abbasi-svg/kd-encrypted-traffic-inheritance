"""Redraw the inheritance figure: which teacher a student follows, and by how much.

The submitted version plotted only the raw correlations, as bars without intervals. Those bars are
the wrong quantity to read an effect from: every student, distilled or not, correlates more with the
ensemble teacher, so the visible A-over-B gap is mostly a property of Teacher A rather than of
distillation. The left panel keeps the correlations but adds the undistilled student, which is the
baseline the statistic subtracts; the right panel plots that statistic itself, with the confidence
intervals the hypothesis was decided on.

Everything is read from a finished confirmatory analysis directory: `specificity.csv` for the
correlations and `components.csv` for the shifts and their intervals. The script computes nothing.

    python scripts/18_inheritance_figure.py --analysis results/analysis/CONFIRMATORY.archive
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

A_COLOUR = "#1f77b4"
B_COLOUR = "#ff7f0e"

ORDER = ["direct", "directTS", "ls", "enddA", "kdA", "kdB", "kdA4", "kdB4"]
# (row in components.csv, what it is called in the paper, which teacher it was distilled from)
SHIFTS = [("H1", "kdA4_shift_to_A", "kdA4", A_COLOUR),
          ("H1", "kdB4_shift_to_B", "kdB4", B_COLOUR),
          ("info", "kdA_shift_to_A", "kdA", A_COLOUR),
          ("info", "kdB_shift_to_B", "kdB", B_COLOUR)]


def correlations(ax, specificity: pd.DataFrame) -> None:
    mean = specificity.groupby("condition")[["A", "B"]].mean().reindex(ORDER)
    x = np.arange(len(ORDER))
    ax.bar(x - 0.19, mean.A, 0.36, label="Teacher A", color=A_COLOUR)
    ax.bar(x + 0.19, mean.B, 0.36, label="Teacher B", color=B_COLOUR)
    # one marker per start date, so the reader sees the spread the bars average over
    for offset, column in ((-0.19, "A"), (0.19, "B")):
        for position, condition in enumerate(ORDER):
            values = specificity[specificity.condition == condition][column]
            ax.plot(np.full(len(values), position + offset), values, ".", color="0.25",
                    markersize=2.2, zorder=3)
    ax.set_xticks(x, ORDER, rotation=30, ha="right", fontsize=6.5, family="monospace")
    ax.set_ylim(0.60, 0.95)
    ax.set_ylabel("Spearman $\\rho$ of per-flow\nenergy scores", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    ax.set_title("(a) Every student follows the ensemble more", fontsize=8, pad=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def shifts(ax, components: pd.DataFrame) -> None:
    primary = components[components.analysis == "primary"]
    rows = []
    for hypothesis, component, label, colour in SHIFTS:
        match = primary[(primary.hypothesis == hypothesis) & (primary.component == component)]
        if match.empty:
            raise SystemExit(f"{component} not found in components.csv")
        rows.append((label, colour, match.iloc[0]))

    positions = np.arange(len(rows))[::-1]
    for position, (label, colour, row) in zip(positions, rows):
        ax.plot([row.ci_low, row.ci_high], [position, position], color=colour, linewidth=1.4)
        ax.plot(row.estimate, position, "o", color=colour, markersize=4.5)
    ax.axvline(0.0, color="0.4", linewidth=0.8, zorder=0)
    span = max(row.ci_high for _, _, row in rows) - min(row.ci_low for _, _, row in rows)
    ax.set_xlim(min(row.ci_low for _, _, row in rows) - 0.08 * span,
                max(row.ci_high for _, _, row in rows) + 0.08 * span)
    ax.set_yticks(positions, [label for label, _, _ in rows], fontsize=6.8, family="monospace")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("shift toward own teacher, relative to direct", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.98, 0.96, "$T=4$", transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color="0.3")
    ax.text(0.98, 0.30, "$T=1$ (accuracy-tuned)", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color="0.3")
    ax.axhline(1.5, color="0.85", linewidth=0.8, zorder=0)
    ax.set_title("(b) Only the conventional temperature transfers", fontsize=8, pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis" / "CONFIRMATORY.archive")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "paper" / "figures" / "fig_inheritance.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    specificity = pd.read_csv(args.analysis / "specificity.csv")
    components = pd.read_csv(args.analysis / "components.csv")
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    correlations(axes[0], specificity)
    shifts(axes[1], components)
    fig.tight_layout(w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}: {specificity.condition.nunique()} conditions over "
          f"{specificity.start.nunique()} start dates, {len(SHIFTS)} shifts with intervals")


if __name__ == "__main__":
    main()
