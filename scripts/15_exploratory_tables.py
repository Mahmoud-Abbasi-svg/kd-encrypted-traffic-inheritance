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


def missing(title: str, path: Path) -> str:
    return (f"% {title}: input not found at {path}\n"
            f"\\draftnote{{{escape(title)}: no results yet ({escape(path.name)} missing).}}\n")


def swaps_table(directory: Path) -> str:
    path = directory / "swaps.csv"
    if not path.exists():
        return missing("Teacher-swap shifts", path)
    frame = pd.read_csv(path)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Teacher-swap shifts, exploratory. Each row is a difference in differences "
             r"against the directly trained student, so zero means the distilled student prefers its "
             r"own teacher no more than an undistilled one does. Intervals are 95\% day$\times$service "
             r"cluster bootstrap percentile intervals.}",
             r"\label{tab:swaps}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Comparison & Units & Shift (95\% CI) \\", r"\midrule"]
    for _, row in frame.iterrows():
        lines.append(f"{escape(row.label)} & {int(row.units)} & "
                     f"${row.estimate:+.4f}$ (${row.ci_low:+.4f}$, ${row.ci_high:+.4f}$) \\\\")
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
             r"\caption{Detection ageing of the teacher ensemble and of its individual members "
             r"(exploratory): slope of unknown-detection AUROC per week since training, fitted with "
             r"one intercept per start date. A more negative ensemble slope than its members' would "
             r"mean the faster ageing is a property of the aggregation.}",
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
    path = directory / "calendar_overlap.csv"
    if not path.exists():
        return missing("Calendar overlap", path)
    frame = pd.read_csv(path)
    shared = int((frame.weeks_shared_with_another_start > 0).sum())
    total = int(frame.weeks_shared_with_another_start.sum())
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Calendar overlap between the three start dates' test windows. The 18 units "
             r"are not independent replicates: " + f"{shared} of {len(frame)} contain at least one "
             r"week that another start date also tests (" + f"{total}" + r" unit-weeks in total).}",
             r"\label{tab:overlap}", r"\begin{tabular}{llrr}", r"\toprule",
             r"Start & Window & Weeks & Shared \\", r"\midrule"]
    for _, row in frame.iterrows():
        lines.append(f"{int(row.start)} & {escape(row.weeks)} & {int(row.n_weeks)} & "
                     f"{int(row.weeks_shared_with_another_start)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


SCORE_HEADER = {"energy": "Energy", "msp": "MSP", "maha": "Mahalanobis", "knnfeat": "Feature $k$-NN"}
ROW_LABEL = {"teacherA - direct student": r"Teacher A $-$ \kd{direct}",
             "teacherB - direct student": r"Teacher B $-$ \kd{direct}"}


def feature_table(directory: Path) -> str:
    """Detection advantage over the direct student, by scoring rule, with bootstrap intervals."""
    path = directory / "detection_advantage.csv"
    if not path.exists():
        return missing("Feature-space open-set scores", path)
    frame = pd.read_csv(path)
    scores = [s for s in ("energy", "msp", "maha", "knnfeat") if s in set(frame.score)]
    wide = frame.pivot_table(index=["kind", "comparison"], columns="score", values="estimate")
    lows = frame.pivot_table(index=["kind", "comparison"], columns="score", values="ci_low")
    highs = frame.pivot_table(index=["kind", "comparison"], columns="score", values="ci_high")
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Unknown-traffic detection advantage over the directly trained student, by "
             r"scoring rule (exploratory; 18 test windows, three start dates). The Mahalanobis and "
             r"feature $k$-NN scores are computed from each model's penultimate features, with no "
             r"model retrained. Intervals are 95\% day$\times$service cluster bootstrap percentile "
             r"intervals. Teacher rows compare models of different feature dimensionality (600 or "
             r"1{,}200 against 128) and are indicative; the student rows compare identical "
             r"architectures. \kd{directTS} is omitted because temperature scaling leaves the "
             r"penultimate features, and therefore both feature-space scores, unchanged.}",
             r"\label{tab:featurescores}",
             r"\begin{tabular}{l" + "r" * len(scores) + "}", r"\toprule",
             "Comparison & " + " & ".join(SCORE_HEADER[s] for s in scores) + r" \\", r"\midrule"]
    for kind in ("teacher", "student"):
        if kind not in wide.index.get_level_values(0):
            continue
        for comparison in wide.loc[kind].index:
            cells = []
            for score in scores:
                estimate = wide.loc[(kind, comparison), score]
                if pd.isna(estimate):
                    cells.append("---")
                    continue
                low, high = lows.loc[(kind, comparison), score], highs.loc[(kind, comparison), score]
                cells.append(f"${estimate:+.3f}$ \\tiny(${low:+.3f}$, ${high:+.3f}$)")
            label = ROW_LABEL.get(comparison, r"\kd{" + comparison.split(" - ")[0] + r"} $-$ \kd{direct}")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if kind == "teacher":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exploratory", type=Path, default=PROJECT_ROOT / "results" / "exploratory" / "REVISION")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper" / "exploratory_tables.tex")
    args = parser.parse_args()

    parts = ["% Generated by scripts/15_exploratory_tables.py - do not edit by hand.",
             f"% Exploratory run: {args.exploratory}", ""]
    parts.append(feature_table(args.exploratory))
    parts.append(swaps_table(args.exploratory))
    parts.append(drift_table(args.exploratory))
    parts.append(overlap_table(args.exploratory))
    args.out.write_text("\n".join(parts), encoding="utf-8")
    written = sum(1 for p in parts if p.startswith(r"\begin{table}"))
    print(f"wrote {args.out}: {written} tables, "
          f"{sum(1 for p in parts if 'draftnote' in p)} placeholders for missing inputs")


if __name__ == "__main__":
    main()
