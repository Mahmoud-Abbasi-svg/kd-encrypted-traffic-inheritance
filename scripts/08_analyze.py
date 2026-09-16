"""Confirmatory analysis of the Track A grid and the shortcut experiment (pre-registration, section 6).

Reads one scripts/06_track_a.py run per start date (and optionally one scripts/07_shortcut.py run) and
writes, to results/analysis/<name>/:
    hypotheses.csv     one row per hypothesis: intersection-union p-value, Holm p-value, decision
    components.csv     every hypothesis component: estimate, 95% bootstrap interval, one-sided p-value,
                       for the primary analysis and the two sensitivity analyses
    units.csv          per start date x window statistics behind the pooled estimates
    per_window.csv     descriptive metrics from metrics.csv (all conditions, near / far unknowns, >=5 packets)
    specificity.csv    student-teacher energy-score rank correlation, from inheritance.csv
    fig_gap.png, fig_specificity.png, fig_reliance.png
    report.md          the tables above in one page

Analyses:
    primary           all flows, exact duplicates of training flows kept (decision D5 proposal)
    drop_duplicates   test flows with an exact packet-sequence duplicate in the training window removed
    ge5_packets       flows with at least 5 packets
Only the primary analysis enters the Holm family.

--windows selects the evaluation splits that are pooled:
    test       the pre-registered test windows (needs a frozen pre-registration)
    smokewin   the windows of a --smoke --smoke-windows run (pipeline check only)
    val        the validation week (H3 is undefined: there is one time point)

Examples:
    python scripts/08_analyze.py --runs results/track_a/<run11> results/track_a/<run24> results/track_a/<run37> \
        --shortcut results/shortcut/<run> --windows test
    python scripts/08_analyze.py --runs results/track_a/<smoke run> --windows smokewin --n-boot 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.analysis import (assign_clusters, build_unit, pooled_bootstrap, shortcut_components,  # noqa: E402
                                summarise_hypotheses)
from kdtraffic.cli import RunLogger, write_json  # noqa: E402
from kdtraffic.splits import load_splits  # noqa: E402

ANALYSES = {
    "primary": lambda d: np.ones(len(d["y"]), dtype=bool),
    "drop_duplicates": lambda d: ~d["duplicate"],
    "ge5_packets": lambda d: d["ppi_len"] >= 5,
}
DESCRIPTIVE = ["macro_f1", "auroc_energy", "auroc_msp", "fpr95_energy", "oscr_energy", "ece", "ece_ts", "nll", "aurc"]


def preregistration_frozen(path: Path) -> bool:
    return path.exists() and re.search(r"^\*\*Status:\*\*\s*FROZEN", path.read_text(encoding="utf-8"),
                                       re.MULTILINE) is not None


def split_matches(split: str, kind: str) -> bool:
    return split == "val" if kind == "val" else split.startswith(f"{kind}_")


def load_runs(runs: list[Path], kind: str, log):
    """Per run: start date, number of classes, the matching score files and the run's CSV outputs."""
    loaded = []
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run / "metrics.csv")
        splits = sorted(s for s in metrics.split.unique() if split_matches(s, kind))
        if not splits:
            raise SystemExit(f"{run} has no '{kind}' splits (found: {sorted(metrics.split.unique())})")
        weeks_since = metrics.groupby("split")["weeks_since"].first().to_dict()
        files = {s: run / f"scores_{s}.npz" for s in splits}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise SystemExit(f"Missing score files: {missing}")
        start = int(config["args"]["start"])
        splits_file = Path(config["args"]["splits"])
        num_classes = len(load_splits(splits_file).known) if splits_file.exists() else None
        log(f"{run.name}: start {start}, splits {splits}")
        loaded.append({"run": run, "start": start, "files": files, "weeks_since": weeks_since,
                       "num_classes": num_classes, "metrics": metrics,
                       "inheritance": pd.read_csv(run / "inheritance.csv")})
    starts = [r["start"] for r in loaded]
    if len(set(starts)) != len(starts):
        raise SystemExit(f"Each start date may appear once; got {starts}")
    return loaded


def build_units(loaded, mask_fn):
    units, days, apps = [], [], []
    for r in loaded:
        for split, path in r["files"].items():
            with np.load(path, allow_pickle=False) as npz:
                data = {k: npz[k] for k in npz.files}
            for needed in ("day", "duplicate", "ppi_len"):
                if needed not in data:
                    raise SystemExit(f"{path} lacks '{needed}': re-run scripts/06_track_a.py with the current code")
            mask = mask_fn(data)
            num_classes = r["num_classes"] or int(data["y"].max()) + 1
            units.append(build_unit(data, r["start"], split, float(r["weeks_since"][split]), num_classes, mask))
            days.append(data["day"][mask])
            apps.append(data["app"][mask])
    n_clusters = assign_clusters(units, days, apps)
    return units, n_clusters


def descriptive_tables(loaded, kind):
    metrics = pd.concat([r["metrics"] for r in loaded], ignore_index=True)
    metrics = metrics[[split_matches(s, kind) for s in metrics.split]]
    per_window = (metrics.groupby(["start", "split", "weeks_since", "flows", "unknown", "condition"])[DESCRIPTIVE]
                  .agg(["mean", "std"]))
    per_window.columns = [f"{m}_{s}" for m, s in per_window.columns]
    inh = pd.concat([r["inheritance"] for r in loaded], ignore_index=True)
    inh = inh[[split_matches(s, kind) for s in inh.split]]
    specificity = inh.groupby(["start", "condition", "teacher"])["spearman_energy"].mean().unstack("teacher")
    return per_window.reset_index(), specificity.reset_index()


def figures(out: Path, per_window: pd.DataFrame, specificity: pd.DataFrame, reliance: pd.DataFrame | None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    view = per_window[(per_window.flows == "all") & (per_window.unknown == "all")]
    teacher = view[view.condition == "teacherA"].set_index(["start", "split"])["auroc_energy_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for condition, style in (("kdA", "-o"), ("enddA", "-s"), ("ls", "--^"), ("directTS", ":v")):
        rows = view[view.condition == condition].set_index(["start", "split"])
        gap = (teacher - rows["auroc_energy_mean"]).dropna().rename("gap").reset_index()
        gap["weeks_since"] = rows.loc[list(zip(gap.start, gap.split)), "weeks_since"].to_numpy()
        for start, g in gap.groupby("start"):
            g = g.sort_values("weeks_since")
            axes[0].plot(g.weeks_since, g.gap, style, label=f"{condition}, start {start}", alpha=0.8)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set(xlabel="weeks since end of training", ylabel="Teacher A − student energy AUROC",
                title="Unknown-detection gap over time")
    axes[0].legend(fontsize=7, ncol=2)
    for condition in ("kdA", "enddA", "ls", "directTS"):
        rows = view[view.condition == condition].sort_values(["start", "weeks_since"])
        axes[1].plot(rows.weeks_since, rows.macro_f1_mean, "o", label=condition, alpha=0.7)
    axes[1].set(xlabel="weeks since end of training", ylabel="macro-F1 (known flows)", title="Student accuracy")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_gap.png", dpi=150)
    plt.close(fig)
    made.append("fig_gap.png")

    spec = specificity[specificity.condition.isin(["kdA", "kdB", "enddA", "ls", "directTS"])]
    if {"A", "B"} <= set(spec.columns):
        mean = spec.groupby("condition")[["A", "B"]].mean()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        mean.plot.bar(ax=ax, rot=0)
        ax.set(ylabel="Spearman ρ of energy scores", title="Which teacher does the student follow?")
        ax.legend(title="teacher")
        fig.tight_layout()
        fig.savefig(out / "fig_specificity.png", dpi=150)
        plt.close(fig)
        made.append("fig_specificity.png")

    if reliance is not None:
        mean = reliance.assign(rho=reliance.rho.astype(float)).groupby(["rho", "model"])["reliance_f1_drop"].mean().unstack()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        mean.plot(ax=ax, marker="o")
        ax.set(xlabel="shortcut reliability ρ during training", ylabel="macro-F1 aligned − flipped",
               title="Flip-test reliance on the synthetic shortcut")
        fig.tight_layout()
        fig.savefig(out / "fig_reliance.png", dpi=150)
        plt.close(fig)
        made.append("fig_reliance.png")
    return made


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)
    header = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return header + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="scripts/06_track_a.py run directories")
    parser.add_argument("--shortcut", type=Path, default=None, help="scripts/07_shortcut.py run directory")
    parser.add_argument("--windows", choices=["test", "smokewin", "val"], default="test")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--analyses", default=",".join(ANALYSES))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "analysis")
    parser.add_argument("--name", default=None)
    parser.add_argument("--preregistration", type=Path, default=PROJECT_ROOT / "docs" / "preregistration.md")
    args = parser.parse_args()
    if args.windows == "test" and not preregistration_frozen(args.preregistration):
        raise SystemExit("Analysing test windows needs a frozen pre-registration")

    name = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.windows}"
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    log = RunLogger(out / "log.txt")
    write_json(out / "config.json", {"args": vars(args)})
    loaded = load_runs(args.runs, args.windows, log)

    components, unit_tables = [], []
    for analysis in args.analyses.split(","):
        started = time.time()
        units, n_clusters = build_units(loaded, ANALYSES[analysis])
        log(f"{analysis}: {len(units)} units, {sum(len(u) for u in units):,} flows, {n_clusters:,} day x service clusters")
        comp, table = pooled_bootstrap(units, n_clusters, args.n_boot, args.seed, progress=log)
        components.append(comp.assign(analysis=analysis))
        unit_tables.append(table.assign(analysis=analysis))
        log(f"{analysis}: done in {time.time() - started:.0f}s")

    reliance = None
    if args.shortcut is not None:
        reliance = pd.read_csv(args.shortcut / "reliance.csv")
        shortcut = pd.read_csv(args.shortcut / "shortcut.csv")
        components.append(shortcut_components(reliance, shortcut).assign(analysis="primary"))

    components = pd.concat(components, ignore_index=True)
    primary = components[components.analysis == "primary"]
    hypotheses = summarise_hypotheses(primary)
    if args.windows == "val":
        hypotheses.loc[hypotheses.hypothesis == "H3", "note"] = "undefined: one time point"
    components.to_csv(out / "components.csv", index=False)
    hypotheses.to_csv(out / "hypotheses.csv", index=False)
    pd.concat(unit_tables, ignore_index=True).to_csv(out / "units.csv", index=False)
    per_window, specificity = descriptive_tables(loaded, args.windows)
    per_window.to_csv(out / "per_window.csv", index=False)
    specificity.to_csv(out / "specificity.csv", index=False)
    made = figures(out, per_window, specificity, reliance)

    cols = ["analysis", "hypothesis", "component", "estimate", "ci_low", "ci_high", "p_one_sided"]
    headline = per_window[(per_window.flows == "all") & (per_window.unknown == "all")] \
        .groupby("condition")[["macro_f1_mean", "auroc_energy_mean", "ece_mean", "ece_ts_mean"]].mean().reset_index()
    report = [
        f"# Analysis: {name}",
        "",
        f"- Windows: `{args.windows}`; bootstrap resamples: {args.n_boot}; seed {args.seed}",
        "- Runs: " + ", ".join(f"`{r['run'].name}` (start {r['start']})" for r in loaded),
        f"- Shortcut run: `{args.shortcut.name}`" if args.shortcut else "- Shortcut run: none (H4 not tested)",
    ]
    if args.windows != "test":
        report.append(f"- **Not a confirmatory result:** `{args.windows}` splits are not the pre-registered test windows.")
    report += [
        "", "## Hypotheses (primary analysis, Holm-corrected)", "", markdown_table(hypotheses),
        "", "## Components", "", markdown_table(components[cols]),
        "", "## Mean over units (all flows, all unknowns)", "", markdown_table(headline),
        "", "## Figures", "",
    ] + [f"![{f}]({f})" for f in made]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log("Hypotheses:\n" + hypotheses.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    log(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
