"""Exploratory analyses added in revision, at the reviewer's request (NOT pre-registered).

The ten confirmatory hypotheses are produced by `scripts/08_analyze.py`, which the pre-registration
freezes; this script never touches it or its outputs. It answers the reviewer's questions that can be
answered from data already on disk, plus any extra per-flow scores or student conditions produced
later in the revision.

    1. Drift mechanism - does Teacher A's *ensemble* lose unknown detection faster than its own
       members? The discussion currently offers this as an untested guess.
    2. The H3 time trend fitted per start date, and a table of how much the three start dates'
       test windows overlap in calendar time.
    3. The scale on which H1's effect should be read: how much an undistilled student's preference
       between two equally good same-family teachers varies by chance.
    4. Teacher-swap shifts for any additional conditions present (kdM0/kdM1 single-teacher swaps,
       hardA label-copying anchor, kdC family swap), computed with the H1 statistic.

Outputs (results/exploratory/<name>/): drift_members.csv, drift_slopes.csv, h3_per_start.csv,
calendar_overlap.csv, teacher_preference_null.csv, swaps.csv, report.md, log.txt, config.json

Examples:
    python scripts/12_exploratory.py --runs results/track_a/20260919-113607_S_train11-14 \
        results/track_a/20260919-131820_S_train24-27 results/track_a/20260919-143354_S_train37-40
    python scripts/12_exploratory.py --runs <as above> --extra-runs results/track_a_exploratory/<run> --n-boot 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.analysis import assign_clusters, student  # noqa: E402
from kdtraffic.cli import RunLogger, write_json  # noqa: E402
from kdtraffic.exploratory import (build_unit, calendar_overlap, cluster_bootstrap_table,  # noqa: E402
                                   drift_by_member, merge_scores, seeds_of, shift, slopes_per_start,
                                   teacher_preference_null)

MEMBERS = [f"teacherA_{i}" for i in range(5)]
BASE_CONDITIONS = ("direct", "directTS", "ls", "kdA", "kdB", "kdA4", "kdB4", "enddA")

# (label, condition, own teacher, other teacher, what it answers)
SWAPS = [
    ("H1 replication: kdA4 toward A vs B", "kdA4", "teacherA", "teacherB",
     "reproduces the confirmatory H1 statistic with this code"),
    ("H1 replication: kdB4 toward B vs A", "kdB4", "teacherB", "teacherA", ""),
    ("single vs single: kdM0 toward member 0 vs Teacher B", "kdM0", "teacherA_0", "teacherB",
     "both teachers are single models, so the ensemble is no longer a confound"),
    ("identity only: kdM0 toward member 0 vs member 1", "kdM0", "teacherA_0", "teacherA_1",
     "the two teachers differ only in random seed"),
    ("identity only: kdM1 toward member 1 vs member 0", "kdM1", "teacherA_1", "teacherA_0", ""),
    ("family swap: kdC toward Teacher C vs A", "kdC", "teacherC", "teacherA",
     "the two teachers are different architectures"),
    ("anchor: hardA toward A vs B", "hardA", "teacherA", "teacherB",
     "label copying only, no soft targets - an upper bound on what argmax agreement alone produces"),
]


def load_runs(runs: list[Path], kind: str, log) -> list[dict]:
    out = []
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run / "metrics.csv")
        splits = sorted(s for s in metrics.split.unique()
                        if (s == "val" if kind == "val" else s.startswith(f"{kind}_")))
        if not splits:
            raise SystemExit(f"{run} has no '{kind}' splits (found {sorted(metrics.split.unique())})")
        out.append({"run": run, "start": int(config["args"]["start"]),
                    "weeks_since": metrics.groupby("split")["weeks_since"].first().to_dict(),
                    "files": {s: run / f"scores_{s}.npz" for s in splits}})
        log(f"{run.name}: start {out[-1]['start']}, {len(splits)} {kind} splits")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="scripts/06_track_a.py run directories")
    parser.add_argument("--extra-runs", type=Path, nargs="*", default=[],
                        help="further runs of the same start dates (new conditions or new scores)")
    parser.add_argument("--windows", choices=["test", "val"], default="test")
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "exploratory")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.windows}"
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    log = RunLogger(out / "log.txt")
    write_json(out / "config.json", {"args": vars(args)})
    log("Exploratory analyses (not pre-registered; the confirmatory outputs are not touched)")

    loaded = load_runs(list(args.runs), args.windows, log) + load_runs(list(args.extra_runs), args.windows, log)
    by_start: dict[int, list[dict]] = {}
    for r in loaded:
        by_start.setdefault(r["start"], []).append(r)

    units, days, apps, conditions_seen = [], [], [], set()
    for start, runs in sorted(by_start.items()):
        for split in runs[0]["files"]:
            paths = [r["files"][split] for r in runs if split in r["files"]]
            data = merge_scores(paths)
            present = sorted({k.split("__")[0] for k in data if k.startswith("student_")})
            conditions = sorted({m[len("student_"):].rsplit("_s", 1)[0] for m in present})
            conditions_seen.update(conditions)
            models = ["teacherA", "teacherB", "teacherC"] + MEMBERS + present
            models = [m for m in models if f"{m}__energy" in data]
            unit = build_unit(data, start, split, float(runs[0]["weeks_since"][split]), models, models)
            units.append(unit)
            days.append(data["day"])
            apps.append(data["app"])
            del data
    n_clusters = assign_clusters(units, days, apps)
    log(f"{len(units)} units, {sum(len(u) for u in units):,} flows, {n_clusters:,} day x service clusters")
    log("Student conditions found: " + ", ".join(sorted(conditions_seen)))

    # 1. Drift mechanism ------------------------------------------------------------------------
    members_table, member_slopes = drift_by_member(units, MEMBERS)
    members_table.to_csv(out / "drift_members.csv", index=False)
    member_slopes.to_csv(out / "drift_slopes.csv", index=False)
    log("Ageing of the ensemble vs its members (AUROC per week; more negative = ages faster):\n"
        + member_slopes.pivot_table(index="model", columns="score", values="slope_per_week")
        .to_string(float_format=lambda v: f"{v:+.5f}"))

    # 2. Time trend per start date, and calendar overlap ----------------------------------------
    energy = members_table[members_table.score == "energy"].copy()
    per_start = []
    for condition in ("kdA", "kdA4"):
        col = f"gap_{condition}"
        gaps = []
        for unit in units:
            ones = np.ones(len(unit))
            from kdtraffic.exploratory import auroc as _auroc
            teacher = _auroc(unit, "teacherA", "energy", ones)
            values = [_auroc(unit, student(condition, s), "energy", ones) for s in unit.seeds]
            values = [v for v in values if np.isfinite(v)]
            gaps.append({"start": unit.start, "weeks_since": unit.weeks_since,
                         col: teacher - float(np.mean(values)) if values else np.nan})
        table = pd.DataFrame(gaps)
        if table[col].notna().any():
            per_start.append(slopes_per_start(table, col).assign(condition=condition, score="energy"))
    h3 = pd.concat(per_start, ignore_index=True) if per_start else pd.DataFrame()
    h3.to_csv(out / "h3_per_start.csv", index=False)
    if len(h3):
        log("Teacher A - student energy-AUROC gap, trend per start date:\n" + h3.to_string(index=False))

    overlap = calendar_overlap(units)
    overlap.to_csv(out / "calendar_overlap.csv", index=False)
    shared = int((overlap.weeks_shared_with_another_start > 0).sum())
    log(f"Calendar overlap: {shared} of {len(overlap)} units contain at least one week that another "
        f"start date also tests; {int(overlap.weeks_shared_with_another_start.sum())} unit-weeks in total")

    # 3. How large is a teacher preference by chance? -------------------------------------------
    null = np.concatenate([teacher_preference_null(u, MEMBERS, np.ones(len(u))) for u in units])
    pd.DataFrame({"abs_difference": null}).to_csv(out / "teacher_preference_null.csv", index=False)
    log(f"Undistilled student's preference between two ensemble members: median "
        f"{np.median(null):.4f}, 95th percentile {np.quantile(null, 0.95):.4f} (n={len(null)})")

    # 4. Teacher-swap shifts --------------------------------------------------------------------
    rows = []
    for label, condition, own, other, note in SWAPS:
        if condition not in conditions_seen:
            continue
        available = [u for u in units if own in u.rank and other in u.rank
                     and any(student(condition, s) in u.rank for s in u.seeds)]
        if not available:
            continue
        log(f"  {label} ({len(available)} units)")
        stat = cluster_bootstrap_table(available, lambda u, w, c=condition, o=own, x=other: shift(u, c, o, x, w),
                                       n_clusters, args.n_boot, args.seed, progress=None)
        rows.append({"label": label, "condition": condition, "own": own, "other": other,
                     "units": len(available), **stat, "note": note})
    swaps = pd.DataFrame(rows)
    swaps.to_csv(out / "swaps.csv", index=False)
    if len(swaps):
        log("Teacher-swap shifts (difference in differences against the direct student):\n"
            + swaps[["label", "estimate", "ci_low", "ci_high", "p_one_sided"]]
            .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    report = [f"# Exploratory analyses: {name}", "",
              "Added during revision at the reviewer's request. **Not pre-registered**; the ten "
              "confirmatory hypotheses and their numbers are unchanged and were produced by "
              "`scripts/08_analyze.py`.", "",
              f"- Windows: `{args.windows}`; units: {len(units)}; bootstrap resamples: {args.n_boot}",
              "", "## Does the ensemble age faster than its members?", "",
              member_slopes.pivot_table(index="model", columns="score", values="slope_per_week")
              .to_markdown(floatfmt="+.5f"), "",
              "## Time trend per start date", "", h3.to_markdown(index=False) if len(h3) else "_not available_", "",
              "## Calendar overlap between start dates", "", overlap.to_markdown(index=False), "",
              "## Chance-level teacher preference", "",
              f"An undistilled student's |ρ difference| between two Teacher A members: median "
              f"{np.median(null):.4f}, 95th percentile {np.quantile(null, 0.95):.4f} over {len(null)} pairs.", "",
              "## Teacher-swap shifts", "",
              swaps.drop(columns=["note"]).to_markdown(index=False, floatfmt="+.4f") if len(swaps) else "_none_"]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
