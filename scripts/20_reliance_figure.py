"""Redraw the shortcut-reliance figure.

The submitted version had three problems. It labelled the curves with the internal model names; it
plotted the three seeds as one averaged line, hiding their spread; and a single linear axis let the
$\\rho=1$ point, where every model collapses, compress the range the paper's claim is about. The claim
is that below $\\rho=1$ both students rely on the planted feature about twice as much as the teacher,
which at that scale is three curves on top of each other near zero.

This version keeps the full range in the left panel and gives $\\rho \\leq 0.9$ its own panel on its
own scale, with the seed spread drawn as a band.

    python scripts/20_reliance_figure.py --shortcut results/shortcut/<run>
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

# internal name -> (label, colour). The teacher sees the feature; so do both of these students.
MODELS = {"teacher": ("teacher (2.3M)", "#2ca02c"),
          "student_both_direct": ("direct student (101k)", "#1f77b4"),
          "student_both_kd": ("KD student, $T=4$ (101k)", "#ff7f0e")}


def draw(ax, frame: pd.DataFrame, title: str) -> None:
    for name, (label, colour) in MODELS.items():
        part = frame[frame.model == name]
        if part.empty:
            continue
        stats = part.groupby("rho").reliance_f1_drop.agg(["mean", "min", "max"]).sort_index()
        ax.fill_between(stats.index, stats["min"], stats["max"], color=colour, alpha=0.18,
                        linewidth=0)
        ax.plot(stats.index, stats["mean"], "o-", color=colour, markersize=3.2, linewidth=1.3,
                label=label)
    ax.set_ylabel("flip-test reliance", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shortcut", type=Path,
                        default=PROJECT_ROOT / "results" / "shortcut" / "20260918-234156_S_train11-14")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "paper" / "figures" / "fig_reliance.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    frame = pd.read_csv(args.shortcut / "reliance.csv")
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.9), layout="constrained")
    draw(axes[0], frame, "(a) All $\\rho$")
    draw(axes[1], frame[frame.rho <= 0.9], "(b) $\\rho \\leq 0.9$")
    axes[1].set_ylabel("")  # one label for two panels; the quantity is the same
    axes[0].legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.2,
                   borderpad=0.1, labelspacing=0.3)
    # A per-panel x-label repeats a phrase too long for a half-column panel, and the two collide.
    fig.supxlabel("shortcut reliability $\\rho$ during training", fontsize=7.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}: {frame.model.nunique()} models, {frame.rho.nunique()} reliabilities, "
          f"{frame.seed.nunique()} seeds")


if __name__ == "__main__":
    main()
