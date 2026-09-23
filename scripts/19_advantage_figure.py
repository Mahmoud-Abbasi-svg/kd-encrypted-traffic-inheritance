"""Draw the unknown-detection advantage over the directly trained student, by scoring rule.

The paper's central revision result is a contrast between two kinds of score, and a table of four
numeric columns hides it: read as prose, the teacher's advantage is 0.000 in one column and 0.073 in
another, with nothing to show that the columns split into two families. This figure puts the
logit-based scores in one panel and the feature-space scores in the other, on one x-axis, so the
split is the first thing a reader sees.

Rows are the comparisons for which all four scores exist. The conditions scored only from logits
(the width sweep, Teacher C, feature distillation, the single-member teachers and the label-copying
anchor) stay in the accompanying table, which `scripts/15_exploratory_tables.py` writes.

    python scripts/19_advantage_figure.py --exploratory results/exploratory/REVISION_FULL
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

LOGIT_SCORES = [("energy", "energy", "#1f77b4"), ("msp", "MSP", "#7fb2d9")]
FEATURE_SCORES = [("maha", "Mahalanobis", "#b2182b"), ("knnfeat", "feature $k$-NN", "#ef8a62")]

# (row in detection_advantage.csv, label on the axis). Each label names both sides of the difference,
# because the blocks mix comparisons against the direct student with paired student comparisons.
BLOCKS = [[("teacherA - direct student", "Teacher A - direct"),
           ("teacherB - direct student", "Teacher B - direct")],
          [("ls - direct student", "ls - direct"),
           ("kdA4 - direct student", "kdA4 - direct"),
           ("kdB4 - direct student", "kdB4 - direct"),
           ("enddA - direct student", "enddA - direct"),
           ("kdA - direct student", "kdA - direct"),
           ("kdB - direct student", "kdB - direct")],
          [("ls - kdA4", "ls - kdA4"),
           ("kdA4 - kdA", "kdA4 - kdA"),
           ("enddA - kdA", "enddA - kdA")]]


def panel(ax, frame: pd.DataFrame, scores, title: str, rows, show_labels: bool) -> None:
    offsets = np.linspace(0.18, -0.18, len(scores))
    for key, label, colour in scores:
        part = frame[frame.score == key].set_index("comparison")
        offset = offsets[[s[0] for s in scores].index(key)]
        first = True
        for position, (comparison, _) in rows:
            if comparison not in part.index:
                continue
            row = part.loc[comparison]
            ax.plot([row.ci_low, row.ci_high], [position + offset] * 2, color=colour, linewidth=1.2)
            ax.plot(row.estimate, position + offset, "o", color=colour, markersize=3.6,
                    label=label if first else None)
            first = False
    ax.axvline(0.0, color="0.35", linewidth=0.8, zorder=0)
    ax.set_yticks([p for p, _ in rows],
                  [label for _, (_, label) in rows] if show_labels else [])
    if show_labels:
        ax.tick_params(axis="y", labelsize=6.6)
        for tick in ax.get_yticklabels():
            tick.set_family("monospace")
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("AUROC advantage", fontsize=7.5)
    ax.set_title(title, fontsize=8, pad=4)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exploratory", type=Path,
                        default=PROJECT_ROOT / "results" / "exploratory" / "REVISION_FULL")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "paper" / "figures" / "fig_advantage.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    frame = pd.read_csv(args.exploratory / "detection_advantage.csv")
    complete = {c for c, part in frame.groupby("comparison")
                if {"energy", "msp", "maha", "knnfeat"} <= set(part.score)}

    # y grows upward, so the blocks are laid out bottom-up: the first block listed ends up on top.
    rows, separators, position = [], [], 0.0
    for index, entries in enumerate(reversed(BLOCKS)):
        kept = [(comparison, label) for comparison, label in entries if comparison in complete]
        if not kept:
            continue
        if index:
            separators.append(position - 0.1)
            position += 0.7
        for entry in reversed(kept):
            rows.append((position, entry))
            position += 1.0
    if not rows:
        raise SystemExit(f"No comparison in {args.exploratory} has all four scores")

    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 0.185 * position + 1.05), sharex=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    panel(axes[0], frame, LOGIT_SCORES, "(a) Logit-based scores (pre-registered)", rows, True)
    panel(axes[1], frame, FEATURE_SCORES, "(b) Feature-space scores (exploratory)", rows, False)
    for ax in axes:
        ax.set_ylim(-0.6, position - 0.4)
        for line in separators:
            ax.axhline(line, color="0.85", linewidth=0.8, zorder=0)
    fig.tight_layout(w_pad=1.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}: {len(rows)} comparisons with all four scores, "
          f"{frame.comparison.nunique() - len(complete)} logit-only comparisons left to the table")


if __name__ == "__main__":
    main()
