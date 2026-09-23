"""Turn the exploratory outputs into LaTeX tables for the revised manuscript.

Every number in the paper reaches it from a file, never by hand (the rule `scripts/11_cpu_table.py`
already follows). This script reads the CSVs written by `scripts/12_exploratory.py` and
`scripts/13_feature_scores.py` and writes `paper/exploratory_tables.tex`, which `main.tex` inputs.

Missing inputs are skipped with a note in the output rather than silently omitted, so a table that
never got its run is visible in the PDF instead of quietly absent.

    python scripts/15_exploratory_tables.py --exploratory results/exploratory/REVISION \
        --feature-scores results/feature_scores/<run11> results/feature_scores/<run24> ...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEACHERS = ("teacherA", "teacherB")


def escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def teacher_name(key: str) -> str:
    """`teacherA_0` as `member 0`, `teacherB` as `B`: the table has one column to name both sides."""
    key = str(key)
    if key.startswith("teacherA_"):
        return "member " + key.rsplit("_", 1)[1]
    return key.removeprefix("teacher") if key.startswith("teacher") else escape(key)


def missing(title: str, path: Path) -> str:
    return (f"% {title}: input not found at {path}\n"
            f"\\draftnote{{{escape(title)}: no results yet ({escape(path.name)} missing).}}\n")


def swaps_table(directory: Path) -> str:
    path = directory / "swaps.csv"
    if not path.exists():
        return missing("Teacher-swap shifts", path)
    frame = pd.read_csv(path)
    # One column, not two: the label's first half names what the row isolates, and several rows share
    # it, so it becomes an italic group heading instead of a repeated column.
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Teacher-swap shifts, exploratory. Each row is a difference in differences "
             r"against \kd{direct}, so zero means no preference for the own teacher. Intervals are "
             r"95\% cluster bootstrap intervals over the units shown.}",
             r"\label{tab:swaps}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Comparison & Units & Shift (95\% CI) \\", r"\midrule"]
    heading = None
    for _, row in frame.iterrows():
        what, _, _ = str(row.label).partition(":")
        if what and what != heading:
            lines.append(r"\multicolumn{3}{@{}l}{\emph{" + escape(what) + r"}} \\")
            heading = what
        # The stored label spells the comparison out ("kdM0 toward member 0 vs Teacher B"), which is
        # too wide for one column; the same thing is rebuilt here from the columns it was made from.
        comparison = (r"\quad \kd{" + escape(row.condition) + "}: "
                      + teacher_name(row.own) + " vs " + teacher_name(row.other))
        lines.append(f"{comparison} & {int(row.units)} & "
                     f"${row.estimate:+.4f}$ \\tiny(${row.ci_low:+.4f}$, ${row.ci_high:+.4f}$) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def drift_table(directory: Path) -> str:
    path = directory / "drift_slopes.csv"
    if not path.exists():
        return missing("Ensemble versus member ageing", path)
    frame = pd.read_csv(path)
    wide = frame.pivot_table(index="model", columns="score", values="slope_per_week")
    keep = [m for m in ["ensemble", "member_mean"] if m in wide.index]
    keep += sorted(m for m in wide.index if m.startswith("teacherA_"))
    wide = wide.reindex(keep)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Detection ageing of the teacher ensemble and of its members (exploratory): "
             r"slope of unknown-detection AUROC per week since training, one intercept per start "
             r"date.}",
             r"\label{tab:drift}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Model & Energy & MSP \\", r"\midrule"]
    for model, row in wide.iterrows():
        name = {"ensemble": "Teacher A (ensemble)", "member_mean": "member mean"}.get(model, escape(model))
        cells = " & ".join(f"${row[s]:+.5f}$" if s in row and pd.notna(row[s]) else "---"
                           for s in ("energy", "msp"))
        lines.append(f"{name} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def overlap_table(directory: Path) -> str:
    """One row per start date. Fig.~\\ref{fig:design} draws the windows themselves, so listing all 18
    here would repeat the figure; what the text needs is how much of each series another series also
    tests."""
    path = directory / "calendar_overlap.csv"
    if not path.exists():
        return missing("Calendar overlap", path)
    frame = pd.read_csv(path)
    shared_units = int((frame.weeks_shared_with_another_start > 0).sum())
    shared_weeks = int(frame.weeks_shared_with_another_start.sum())
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Calendar overlap between the three start dates' test windows "
             r"(Fig.~\ref{fig:design}(a)): " + f"{shared_units} of the {len(frame)} units share at "
             r"least one test week with another start date, " + f"{shared_weeks}"
             + r" unit-weeks in total.}",
             r"\label{tab:overlap}", r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Start & Windows & First week & Last week & Shared unit-weeks \\", r"\midrule"]
    for start, part in frame.groupby("start"):
        weeks = [str(w) for span in part.weeks for w in str(span).split("-")]
        lines.append(f"{int(start)} & {len(part)} & {min(int(w) for w in weeks)} & "
                     f"{max(int(w) for w in weeks)} & "
                     f"{int(part.weeks_shared_with_another_start.sum())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


SCORE_HEADER = {"energy": "Energy", "msp": "MSP", "maha": "Mahalanobis", "knnfeat": "Feature $k$-NN"}
PRETTY = {"teacherA": "Teacher A", "teacherB": "Teacher B", "teacherC": "Teacher C",
          "direct student": r"\kd{direct}"}


def row_label(comparison: str) -> str:
    """`<a> - <b>` as a table row label, with condition names escaped.

    The width-sweep conditions carry a suffix (`directTS_w16`), and an unescaped underscore in text
    mode is a LaTeX error rather than a typographic blemish.
    """
    parts = [p.strip() for p in comparison.split(" - ", 1)]
    shown = [PRETTY.get(p, r"\kd{" + escape(p) + "}") for p in parts]
    return f"{shown[0]} $-$ {shown[1]}" if len(shown) == 2 else shown[0]


ALL_SCORES = ("energy", "msp", "maha", "knnfeat")
# Drawn by `scripts/19_advantage_figure.py`, and therefore left out of the table below. The two lists
# have to agree: a comparison in neither would vanish from the paper, which is why the table takes
# everything this list does not name rather than filtering on which scores exist.
FIGURE_ROWS = ("teacherA - direct student", "teacherB - direct student",
               "ls - direct student", "kdA4 - direct student", "kdB4 - direct student",
               "enddA - direct student", "kdA - direct student", "kdB - direct student",
               "ls - kdA4", "kdA4 - kdA", "enddA - kdA")
# Ordering the rest by hand keeps the width sweep together; anything unlisted follows, so a new
# condition is never dropped.
TABLE_ORDER = ["direct_w16 - direct student", "direct_w96 - direct student",
               "directTS_w16 - direct student", "directTS_w96 - direct student",
               "kdA4_w16 - direct student", "kdA4_w96 - direct student",
               "kdM0 - direct student", "kdM1 - direct student", "kdC - direct student",
               "hardA - direct student", "kdF - direct student",
               # kept beside kdF, because the paired rows are what the feature-distillation
               # comparison rests on: its own row pools over nine windows and kdM0's over eighteen.
               "kdF - kdM0", "kdF - kdA4", "hardA - kdA4"]


def remaining_table(directory: Path) -> str:
    """The detection advantage of the conditions the figure does not draw.

    Fig.~\\ref{fig:advantage} carries the comparison the section argues from, separating logit-based
    from feature-space scores. The conditions added at review --- the width sweep, the single-member
    and transformer teachers, the label-copying anchor and feature distillation --- are reported here
    instead, with whichever scores were computed for them.
    """
    path = directory / "detection_advantage.csv"
    if not path.exists():
        return missing("Open-set detection advantage", path)
    frame = pd.read_csv(path)
    rest = [c for c in frame.comparison.unique() if c not in FIGURE_ROWS]
    if not rest:
        return "% the figure draws every comparison\n"
    ordered = [c for c in TABLE_ORDER if c in rest] + [c for c in rest if c not in TABLE_ORDER]

    wide = frame.pivot_table(index="comparison", columns="score", values="estimate")
    lows = frame.pivot_table(index="comparison", columns="score", values="ci_low")
    highs = frame.pivot_table(index="comparison", columns="score", values="ci_high")
    present = wide.reindex(ordered).notna().any()
    scores = [s for s in ALL_SCORES if s in present.index and present[s]]
    # Four columns of estimate-plus-interval do not fit one column of a two-column layout.
    wide_float = len(scores) > 2
    lines = [r"\begin{table*}[t]" if wide_float else r"\begin{table}[t]", r"\centering",
             r"\caption{Detection advantage of the exploratory conditions. A row naming one condition "
             r"is that condition minus \kd{direct}; a row naming two is their paired difference. A "
             r"dash marks a score that was not computed. Intervals are 95\% cluster bootstrap "
             r"intervals over the 18 test windows, or over the nine of start date 11 for the "
             r"conditions trained there alone (the width sweep, \kd{kdC} and \kd{kdF}).}",
             r"\label{tab:logitscores}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}l" + "r" * len(scores) + "@{}}", r"\toprule",
             "Comparison & " + " & ".join(SCORE_HEADER[s] for s in scores) + r" \\", r"\midrule"]
    for comparison in ordered:
        cells = []
        for score in scores:
            estimate = wide.loc[comparison, score] if score in wide.columns else float("nan")
            if pd.isna(estimate):
                cells.append("---")
                continue
            low, high = lows.loc[comparison, score], highs.loc[comparison, score]
            cells.append(f"${estimate:+.3f}$ \\tiny(${low:+.3f}$, ${high:+.3f}$)")
        # Almost every row is against `direct`; spelling that out in each label overflows the column,
        # so the caption says it once and only the exceptions carry both names.
        left, _, right = comparison.partition(" - ")
        label = (r"\kd{" + escape(left) + "}") if right == "direct student" else row_label(comparison)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{table*}" if wide_float else r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exploratory", type=Path, default=PROJECT_ROOT / "results" / "exploratory" / "REVISION")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper" / "exploratory_tables.tex")
    args = parser.parse_args()

    parts = ["% Generated by scripts/15_exploratory_tables.py - do not edit by hand.",
             f"% Exploratory run: {args.exploratory}", ""]
    parts.append(remaining_table(args.exploratory))
    parts.append(swaps_table(args.exploratory))
    parts.append(drift_table(args.exploratory))
    parts.append(overlap_table(args.exploratory))
    args.out.write_text("\n".join(parts), encoding="utf-8")
    written = sum(1 for p in parts if p.startswith(r"\begin{table}"))
    print(f"wrote {args.out}: {written} tables, "
          f"{sum(1 for p in parts if 'draftnote' in p)} placeholders for missing inputs")


if __name__ == "__main__":
    main()
