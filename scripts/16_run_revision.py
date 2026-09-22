"""Run the remaining revision experiments unattended, in order, resumably.

Each step writes `logs/done/<name>.flag` when it succeeds, so re-running skips finished work: if the
laptop sleeps or a step fails, start it again and it continues where it stopped. Steps run strictly
one at a time - the 4 GB card and 15 GB of RAM do not tolerate two jobs, and a run that dies at 3 a.m.
costs a whole night.

Run directories are discovered by pattern rather than pasted in, because every run is timestamped and
the analysis needs whichever ones the earlier steps just produced.

    python scripts/16_run_revision.py                 # everything still outstanding
    python scripts/16_run_revision.py --only analysis # just the exploratory analysis and tables
    python scripts/16_run_revision.py --list          # show the plan and what is already done
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOG = PROJECT_ROOT / "logs" / "revision_laptop.log"
DONE = PROJECT_ROOT / "logs" / "done"

CONFIRMATORY = ["results/track_a/20260919-113607_S_train11-14",
                "results/track_a/20260919-131820_S_train24-27",
                "results/track_a/20260919-143354_S_train37-40"]
PILOT = "results/pilot/20260916-122627_S_train11-14"
TEACHERS = {11: PILOT,
            24: "results/track_a/20260917-121305_S_train24-27",
            37: "results/track_a/20260918-132337_S_train37-40"}
TEACHER_B = {11: "results/track_a/20260916-140053_S_train11-14/models/teacherB_wide.pt",
             24: "results/track_a/20260917-121305_S_train24-27/models/teacherB_wide.pt",
             37: "results/track_a/20260918-132337_S_train37-40/models/teacherB_wide.pt"}


def newest(pattern: str) -> Path | None:
    """The most recent run directory matching `pattern`, ignoring smoke runs."""
    found = [p for p in PROJECT_ROOT.glob(pattern) if p.is_dir() and "smoke" not in p.name]
    return max(found, key=lambda p: p.name) if found else None


def every(pattern: str) -> list[Path]:
    return sorted((p for p in PROJECT_ROOT.glob(pattern) if p.is_dir() and "smoke" not in p.name),
                  key=lambda p: p.name)


def track_a(start: int, conditions: str, extra: list[str]) -> list[str]:
    return [PYTHON, "scripts/06_track_a.py", "--size", "S", "--start", str(start), "--with-test",
            "--conditions", conditions, "--teachers-from", TEACHERS[start],
            "--teacher-b-from", TEACHER_B[start], "--out", "results/track_a_exploratory",
            "--workers", "0", *extra]


def analysis_command() -> list[str] | None:
    """The exploratory analysis over every run that exists by the time this step is reached."""
    extra = [str(p.relative_to(PROJECT_ROOT)) for p in
             every("results/track_a_exploratory/*") + every("results/feature_scores/*")]
    if not extra:
        return None
    return [PYTHON, "scripts/12_exploratory.py", "--runs", *CONFIRMATORY, "--extra-runs", *extra,
            "--n-boot", "300", "--name", "REVISION"]


def tables_command() -> list[str]:
    return [PYTHON, "scripts/15_exploratory_tables.py", "--exploratory", "results/exploratory/REVISION"]


def cpu_command() -> list[str] | None:
    run = newest("results/track_a/*_S_train11-14")
    if run is None:
        return None
    return [PYTHON, "scripts/10_cpu_cost.py", "--size", "S",
            "--student", str((run / "models" / "student_direct_s0.pt").relative_to(PROJECT_ROOT)),
            "--teachers-from", PILOT, "--teacher-b", TEACHER_B[11],
            "--xgb-estimators", "150", "--xgb-depth", "6", "--workers", "0"]


# (name, group, how to build the command). A callable is resolved when the step is reached, so it
# sees directories the earlier steps created.
STEPS = [
    ("teacherc11", "teacherc", lambda: track_a(11, "kdC", [])),
    ("width16", "widths", lambda: track_a(11, "direct,kdA4", ["--condition-suffix", "_w16",
                                                              "--student-width", "16"])),
    ("width96", "widths", lambda: track_a(11, "direct,kdA4", ["--condition-suffix", "_w96",
                                                              "--student-width", "96"])),
    ("xgb_retune", "xgboost", lambda: [PYTHON, "scripts/04_baselines.py", "--size", "S",
                                       "--xgb-estimators", "150", "--xgb-depth", "6",
                                       "--skip-knn", "--workers", "0"]),
    ("cpu_retune", "xgboost", cpu_command),
    ("exploratory", "analysis", analysis_command),
    ("tables", "analysis", tables_command),
]


def note(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="run only these groups")
    parser.add_argument("--list", action="store_true", help="show the plan and exit")
    parser.add_argument("--redo", nargs="*", default=[], help="step names to run again")
    args = parser.parse_args()

    DONE.mkdir(parents=True, exist_ok=True)
    for name in args.redo:
        (DONE / f"{name}.flag").unlink(missing_ok=True)

    planned = [s for s in STEPS if args.only is None or s[1] in args.only]
    if args.list:
        for name, group, _ in planned:
            state = "done" if (DONE / f"{name}.flag").exists() else "pending"
            print(f"  {name:14s} [{group:9s}] {state}")
        return

    for name, group, build in planned:
        flag = DONE / f"{name}.flag"
        if flag.exists():
            note(f"{name}: already done, skipped")
            continue
        command = build()
        if command is None:
            note(f"{name}: SKIPPED, its inputs do not exist yet")
            continue
        note(f"===== {name} started")
        started = time.time()
        with LOG.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            note(f"===== {name} FAILED (exit {result.returncode}); "
                 f"fix it and run this script again to resume here")
            raise SystemExit(result.returncode)
        flag.write_text("done", encoding="utf-8")
        note(f"===== {name} finished in {(time.time() - started) / 60:.0f} min")
    note("All requested revision steps are done.")


if __name__ == "__main__":
    main()
