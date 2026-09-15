"""Week-1 analysis: when does each CESNET-TLS-Year22 service appear, and how stable is it?

Reads the full-dataset weekly statistics extracted by 00_fetch_full_stats.py (flow counts per
service per week, covering all ~508M flows) and the service map (service -> category).

Answers the study-plan check "do any of the 180 services first appear partway through the
year?" for each training window, which decides whether unknown services can be natural or
must be simulated by holding services out. For each window a service is classified as:

    usable     average weekly flows in the window >= usable-min
    emerging   not usable in the window, never usable in any earlier study week, and usable
               later (a 4-week rolling average >= usable-min): a candidate natural unknown
    dip        not usable in the window but usable earlier (seasonal or temporary drop)
    gone       usable in the window but never again afterwards

Weeks whose total flow count is far below the median (capture gaps) are excluded.

Outputs (results/week1/):
    service_weekly_counts.csv   flow counts, services x weeks
    weekly_totals.csv           total saved flows per week, with a low-coverage flag
    service_appearance.csv      per-service appearance summary and per-window flags
    service_appearance.png      heatmap of weekly counts (log scale)
    summary.txt                 headline numbers
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(os.environ.get("KD_DATA_ROOT", "C:/datasets"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK_DIR = re.compile(r"WEEK-(\d{4})-(\d{2})$")
STUDY_FIRST_WEEK = 11  # flow exporter updated in week 10 (dataset authors' warning)
LAST_WEEK = 52
ROLLING_WEEKS = 4


def parse_windows(text: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in part.split("-")) for part in text.split(",")]


def load_weekly_counts(stats_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    records, totals = {}, {}
    for path in sorted(stats_root.glob("*/WEEK-*/stats-week.json")):
        match = WEEK_DIR.search(path.parent.name)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        stats = json.loads(path.read_text(encoding="utf-8"))
        records[key] = stats["apps"]
        totals[key] = stats["global"]["total-saved"]
    if not records:
        raise FileNotFoundError(f"No stats-week.json files under {stats_root}; run 00_fetch_full_stats.py first")
    keys = sorted(records)
    index = pd.MultiIndex.from_tuples(keys, names=["year", "week"])
    counts = pd.DataFrame({k: records[k] for k in keys}).fillna(0).astype(np.int64)
    counts.columns = index
    return counts.sort_index(), pd.Series([totals[k] for k in keys], index=index, name="total_saved")


def week_label(key: tuple[int, int]) -> str:
    return f"{key[0]}-W{key[1]:02d}"


def study_weeks(columns) -> list[tuple[int, int]]:
    return [c for c in columns if c[0] == 2022 and STUDY_FIRST_WEEK <= c[1] <= LAST_WEEK]


def first_hit(row: pd.Series, mask: pd.Series) -> float:
    hits = row.index[mask.to_numpy()]
    return float(hits[0][1]) if len(hits) else np.nan


def last_hit(row: pd.Series, mask: pd.Series) -> float:
    hits = row.index[mask.to_numpy()]
    return float(hits[-1][1]) if len(hits) else np.nan


def summarise(counts: pd.DataFrame, servicemap: pd.DataFrame, windows, usable_min: int, weeks) -> pd.DataFrame:
    study = counts[weeks]
    shares = study / study.sum(axis=0)
    tag_info = servicemap.set_index("Tag")

    rows = []
    for tag, row in study.iterrows():
        usable = row >= usable_min
        present = row > 0
        info = {
            "service": tag,
            "name": tag_info["Service"].get(tag, ""),
            "category": tag_info["Service Category"].get(tag, "UNMAPPED"),
            "total_flows_study": int(row.sum()),
            "median_weekly_share": float(shares.loc[tag].median()),
            "first_week_seen": first_hit(row, present),
            "first_usable_week": first_hit(row, usable),
            "last_usable_week": last_hit(row, usable),
            "weeks_present": int(present.sum()),
            "weeks_usable": int(usable.sum()),
            "zero_weeks": int((~present).sum()),
        }
        for start, end in windows:
            in_window = [w for w in weeks if start <= w[1] <= end]
            before = [w for w in weeks if w[1] < start]
            after = [w for w in weeks if w[1] > end]
            window_avg = float(row[in_window].mean())
            usable_in_window = window_avg >= usable_min
            usable_before = bool((row[before] >= usable_min).any()) if before else False
            later_rolling = row[after].rolling(ROLLING_WEEKS).mean()
            usable_later = bool((later_rolling >= usable_min).any())
            later_hits = later_rolling.index[(later_rolling >= usable_min).to_numpy()]
            suffix = f"w{start}_{end}"
            info[f"window_avg_{suffix}"] = round(window_avg)
            info[f"usable_{suffix}"] = usable_in_window
            info[f"emerging_{suffix}"] = (not usable_in_window) and (not usable_before) and usable_later
            info[f"dip_{suffix}"] = (not usable_in_window) and usable_before
            info[f"gone_{suffix}"] = usable_in_window and not usable_later
            info[f"first_usable_after_{suffix}"] = float(later_hits[0][1]) if len(later_hits) else np.nan
        rows.append(info)
    return pd.DataFrame(rows).sort_values(["first_usable_week", "category", "service"], na_position="last")


def plot_heatmap(counts: pd.DataFrame, summary: pd.DataFrame, windows, low_weeks, out_path: Path) -> None:
    order = summary["service"].tolist()
    matrix = np.log10(counts.loc[order].to_numpy(dtype=float) + 1)
    labels = [week_label(c) if c[0] != 2022 else f"W{c[1]:02d}" for c in counts.columns]
    col_index = {c: i for i, c in enumerate(counts.columns)}

    fig, ax = plt.subplots(figsize=(14, 24))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    if (2022, STUDY_FIRST_WEEK) in col_index:
        ax.axvline(col_index[(2022, STUDY_FIRST_WEEK)] - 0.5, color="red", linewidth=1.5)
    for start, end in windows:
        if (2022, start) in col_index and (2022, end) in col_index:
            ax.axvspan(col_index[(2022, start)] - 0.5, col_index[(2022, end)] + 0.5, color="white", alpha=0.15)
    for week in low_weeks:
        ax.get_xticklabels()[col_index[week]].set_color("red")
    ax.set_title("CESNET-TLS-Year22: weekly flows per service (log10), full dataset\n"
                 "red line = start of study period (week 11); shaded = training windows; red labels = low-coverage weeks")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="log10(flows + 1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary(summary, totals, windows, usable_min, low_weeks, out_path: Path) -> None:
    def names(frame: pd.DataFrame) -> str:
        return ", ".join(frame["service"]) if len(frame) else "none"

    lines = [
        "CESNET-TLS-Year22 service appearance (full-dataset weekly statistics)",
        f"Study period: 2022 weeks {STUDY_FIRST_WEEK}-{LAST_WEEK}; usable = >= {usable_min} flows/week "
        f"(~{usable_min * 25_000_000 / 507_739_073:.0f} flows in size S, ~{usable_min * 10_000_000 / 507_739_073:.0f} in XS)",
        f"Weekly totals: median {int(totals.median()):,}; low-coverage weeks excluded: "
        + (", ".join(f"{week_label(w)} ({totals[w]:,})" for w in low_weeks) or "none"),
        f"Services: {len(summary)}; unmapped to a category: {(summary['category'] == 'UNMAPPED').sum()}",
        f"Services with flows in every study week: {(summary['zero_weeks'] == 0).sum()}",
        f"Services with at least one zero week: {names(summary[summary['zero_weeks'] > 0])}",
        f"Services first seen after week {STUDY_FIRST_WEEK} (true mid-year appearance): "
        f"{names(summary[summary['first_week_seen'] > STUDY_FIRST_WEEK])}",
        f"Services never usable in the study period: {names(summary[summary['first_usable_week'].isna()])}",
        "",
    ]
    for start, end in windows:
        suffix = f"w{start}_{end}"
        emerging = summary[summary[f"emerging_{suffix}"]]
        dips = summary[summary[f"dip_{suffix}"]]
        gone = summary[summary[f"gone_{suffix}"]]
        lines.append(f"Training window weeks {start}-{end}: usable {int(summary[f'usable_{suffix}'].sum())}; "
                     f"emerging afterwards {len(emerging)}; seasonal/temporary dips {len(dips)}; gone afterwards {len(gone)}")
        for _, r in emerging.iterrows():
            lines.append(f"    emerging: {r['service']:<24} {r['category']:<24} window avg {r[f'window_avg_{suffix}']:>6,}/wk; "
                         f"first seen W{r['first_week_seen']:.0f}; usable (4-wk avg) from W{r[f'first_usable_after_{suffix}']:.0f}")
        lines.append(f"    dips: {names(dips)}")
        lines.append(f"    gone: {names(gone)}")
    lines += ["", "Note: counts are after the dataset's sampling, so shares are not true popularity; "
                  "appearance, growth and disappearance are still meaningful."]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    dataset_root = DEFAULT_DATA_ROOT / "CESNET-TLS-Year22"
    parser.add_argument("--stats-root", type=Path, default=dataset_root / "stats_full")
    parser.add_argument("--servicemap", type=Path, default=dataset_root / "meta" / "servicemap.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "week1")
    parser.add_argument("--windows", default="11-14,24-27,37-40", help="training windows as start-end week ranges")
    parser.add_argument("--usable-min", type=int, default=5000, help="min full-dataset flows per week to count as usable")
    parser.add_argument("--low-coverage-ratio", type=float, default=0.5,
                        help="exclude study weeks whose total flows are below this fraction of the median")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(args.windows)

    counts, totals = load_weekly_counts(args.stats_root)
    servicemap = pd.read_csv(args.servicemap)
    all_study = study_weeks(counts.columns)
    study_totals = totals[all_study]
    low_weeks = [w for w in all_study if study_totals[w] < args.low_coverage_ratio * study_totals.median()]
    weeks = [w for w in all_study if w not in low_weeks]

    flat = counts.copy()
    flat.columns = [week_label(c) for c in counts.columns]
    flat.to_csv(args.out / "service_weekly_counts.csv", index_label="service")
    pd.DataFrame({
        "week": [week_label(k) for k in totals.index],
        "total_saved": totals.to_numpy(),
        "low_coverage": [k in low_weeks for k in totals.index],
    }).to_csv(args.out / "weekly_totals.csv", index=False)

    summary = summarise(counts, servicemap, windows, args.usable_min, weeks)
    summary.to_csv(args.out / "service_appearance.csv", index=False)
    plot_heatmap(counts, summary, windows, low_weeks, args.out / "service_appearance.png")
    write_summary(summary, study_totals, windows, args.usable_min, low_weeks, args.out / "summary.txt")
    print((args.out / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
