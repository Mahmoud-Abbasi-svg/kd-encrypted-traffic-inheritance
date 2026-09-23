"""Draw the study-design figure: the calendar of the three replicates, and the swap control.

Two things about this study are hard to hold in the head from prose alone: which weeks each start
date trains, validates and tests on - including which test weeks two start dates share, the reason
the 18 units are not independent - and what the teacher-swap statistic actually compares. The left
panel is the calendar; the right panel is the control.

The calendar comes from `calendar_overlap.csv`, so the figure and the dependence discussion in the
paper are drawn from the same file. Training and validation weeks are design constants, stated in
the methods section and fixed in the pre-registration.

    python scripts/17_design_figure.py --exploratory results/exploratory/REVISION_FULL
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN = {11: (11, 14), 24: (24, 27), 37: (37, 40)}
VALIDATION = {11: 15, 24: 28, 37: 41}
# Weeks 1-10 precede the exporter update; 50 and 52 carry less than half the median weekly volume.
EXCLUDED = set(range(1, 11)) | {50, 52}

TRAIN_COLOUR = "#2c4a6e"
VAL_COLOUR = "#e8a33d"
TEST_COLOUR = "#7fb2d9"
A_COLOUR = "#1f77b4"
B_COLOUR = "#ff7f0e"


def weeks_of(span: str) -> list[int]:
    """`16-19` or `51-51` as the list of weeks it covers."""
    first, _, last = span.partition("-")
    return list(range(int(first), int(last or first) + 1))


def calendar(ax, overlap: pd.DataFrame) -> None:
    covered = {start: {w for span in part.weeks for w in weeks_of(str(span))}
               for start, part in overlap.groupby("start")}
    shared = {w for w in range(1, 53) if sum(w in weeks for weeks in covered.values()) > 1}

    rows = sorted(TRAIN, reverse=True)  # start 11 on top
    for week in sorted(shared):
        ax.axvspan(week - 0.5, week + 0.5, color="0.88", zorder=0, linewidth=0)
    for week in sorted(EXCLUDED):
        ax.axvspan(week - 0.5, week + 0.5, facecolor="none", edgecolor="0.7", hatch="///",
                   linewidth=0, zorder=1)

    for row, start in enumerate(rows):
        first, last = TRAIN[start]
        ax.broken_barh([(first - 0.5, last - first + 1)], (row - 0.32, 0.64),
                       facecolor=TRAIN_COLOUR, zorder=3)
        ax.broken_barh([(VALIDATION[start] - 0.5, 1)], (row - 0.32, 0.64),
                       facecolor=VAL_COLOUR, zorder=3)
        for span in overlap[overlap.start == start].weeks:
            weeks = weeks_of(str(span))
            ax.broken_barh([(min(weeks) - 0.5, len(weeks))], (row - 0.22, 0.44),
                           facecolor=TEST_COLOUR, edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_yticks(range(len(rows)), [f"start wk {s}" for s in rows], fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.5, 52.5)
    ax.set_xticks([1, 10, 20, 30, 40, 52])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("week of 2022", fontsize=7.5, labelpad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in (TRAIN_COLOUR, VAL_COLOUR, TEST_COLOUR)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="0.88"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.7", hatch="///"))
    ax.legend(handles, ["train", "validation", "test window", "tested by $\\geq 2$ starts", "excluded"],
              fontsize=6.2, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.45), columnspacing=1.0, handlelength=1.4)
    ax.set_title("(a) Three replicates of the whole design", fontsize=8, pad=4)


def box(ax, x, y, width, height, text, colour, fontsize=6.4, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.03",
                                facecolor="white", edgecolor=colour, linewidth=1.0))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize,
            color="black", linespacing=1.25, family="monospace" if mono else None)


def arrow(ax, start, end, colour, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7, color=colour,
                                 linewidth=0.9, shrinkA=1, shrinkB=1))


def control(ax) -> None:
    """The swap: two equally accurate teachers, one student architecture, one undistilled baseline."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "distil into the same 101k student", ha="center", va="center", fontsize=6.0,
            color="0.35")
    rows = ((0.60, "Teacher A (ensemble)", "kdA4", A_COLOUR),
            (0.33, "Teacher B (wide)", "kdB4", B_COLOUR),
            (0.06, "no teacher", "direct", "0.55"))
    for y, teacher, student, colour in rows:
        box(ax, 0.00, y, 0.48, 0.20, teacher, colour, fontsize=5.9)
        box(ax, 0.66, y, 0.32, 0.20, student, colour, fontsize=7.0, mono=True)
        arrow(ax, (0.50, y + 0.10), (0.64, y + 0.10), colour)
    ax.set_title("(b) The teacher-swap control", fontsize=8, pad=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exploratory", type=Path,
                        default=PROJECT_ROOT / "results" / "exploratory" / "REVISION_FULL")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper" / "figures" / "fig_design.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    overlap = pd.read_csv(args.exploratory / "calendar_overlap.csv")
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.95), gridspec_kw={"width_ratios": [2.0, 1.0]})
    calendar(axes[0], overlap)
    control(axes[1])
    fig.tight_layout(w_pad=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}: {len(overlap)} test windows over {overlap.start.nunique()} start dates")


if __name__ == "__main__":
    main()
