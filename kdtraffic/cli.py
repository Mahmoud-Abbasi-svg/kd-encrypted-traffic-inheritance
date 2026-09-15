"""Command-line options and run bookkeeping shared by the experiment scripts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kdtraffic.data import DATA_ROOT, PROJECT_ROOT, DataSpec

SMOKE_SIZES = dict(train_size=20_000, val_known_size=4_000, val_unknown_size=4_000,
                   test_known_size=4_000, test_unknown_size=4_000, min_train_samples=0)


def parse_weeks(text: str) -> tuple[int, int]:
    parts = [int(p) for p in str(text).split("-")]
    return parts[0], parts[-1]


def parse_size(text: str) -> int | str:
    return "all" if text == "all" else int(text)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument("--data-root", type=Path, default=DATA_ROOT, help="folder containing CESNET-TLS-Year22/ (env KD_DATA_ROOT)")
    group.add_argument("--cache-dir", type=Path, default=None, help="array cache (default: <data-root>/cache)")
    group.add_argument("--splits", type=Path, default=PROJECT_ROOT / "configs" / "splits.json")
    group.add_argument("--size", default="S", choices=["XS", "S", "M", "L"])
    group.add_argument("--train-weeks", default="11-14")
    group.add_argument("--val-weeks", default="15")
    group.add_argument("--test-weeks", default="16-19")
    group.add_argument("--with-test", action="store_true",
                       help="also load and evaluate the test window (only after the pre-registration is frozen)")
    group.add_argument("--train-size", default="all", help="'all' or a number of flows")
    group.add_argument("--val-known-size", default="200000")
    group.add_argument("--val-unknown-size", default="all")
    group.add_argument("--test-known-size", default="400000")
    group.add_argument("--test-unknown-size", default="all")
    group.add_argument("--min-train-samples", type=int, default=100)
    group.add_argument("--data-seed", type=int, default=420)
    group.add_argument("--workers", type=int, default=4, help="DataZoo loader workers (0 on Windows laptops)")
    group.add_argument("--smoke", action="store_true", help="tiny sizes and settings for an end-to-end check")


def spec_from_args(args: argparse.Namespace) -> DataSpec:
    spec = DataSpec(
        size=args.size,
        train_weeks=parse_weeks(args.train_weeks),
        val_weeks=parse_weeks(args.val_weeks),
        test_weeks=parse_weeks(args.test_weeks),
        with_test=args.with_test,
        train_size=parse_size(args.train_size),
        val_known_size=parse_size(args.val_known_size),
        val_unknown_size=parse_size(args.val_unknown_size),
        test_known_size=parse_size(args.test_known_size),
        test_unknown_size=parse_size(args.test_unknown_size),
        min_train_samples=args.min_train_samples,
        seed=args.data_seed,
    )
    if args.smoke:
        for key, value in SMOKE_SIZES.items():
            setattr(spec, key, value)
        args.workers = 0
    return spec


def make_run_dir(out: Path, spec: DataSpec, smoke: bool) -> Path:
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.size}_train{spec.train_weeks[0]}-{spec.train_weeks[1]}"
    run_dir = Path(out) / (name + ("_smoke" if smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                          encoding="utf-8")


class RunLogger:
    """Prints messages with a timestamp and appends them to <run_dir>/log.txt."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
