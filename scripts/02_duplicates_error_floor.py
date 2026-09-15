"""Week-1 check: duplicate flows between a training window and later weeks, and the error floor.

For a random sample of flows from a training window (default weeks 11-14) and a random sample
from all later weeks, this script hashes each flow's packet sequence (PPI) and reports, per key:

    dup_rate            share of test flows whose sequence also occurs in the training sample
    label_absent_rate   share of test flows whose sequence occurs in training, but never with
                        the test flow's label: a lookup model is guaranteed to be wrong there
                        (lower bound on unavoidable error for memorisation)
    train_dup_rate      share of training flows whose sequence occurs more than once in training
    train_ambiguous     share of training flows whose sequence carries more than one label

Rates are measured against a training *sample*; with the full training set they can only be
higher, so they are lower bounds. Sample sizes are capped to fit a 16 GB laptop.

Key definitions:
    ppi30_ipt_dir_size  timing + direction + size of the first 30 packets (exact duplicate)
    ppi30_dir_size      direction + size of the first 30 packets (ignores timing jitter)
    ppi10_dir_size      direction + size of the first 10 packets (the PPI-10 fallback in the plan)

Outputs (results/week1/):
    duplicates_w{start}_{end}.csv           metrics per key x test week, plus overall rows
    duplicates_by_length_w{start}_{end}.csv overall metrics per key x PPI length bin
    duplicates_w{start}_{end}.txt           headline summary
"""

import argparse
import gc
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cesnet_datazoo.config import DatasetConfig
from cesnet_datazoo.datasets import CESNET_TLS_Year22

DEFAULT_DATA_ROOT = Path(os.environ.get("KD_DATA_ROOT", "C:/datasets"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IPT, DIR, SIZE = 0, 1, 2  # PPI channel order in cesnet-datazoo
KEYS = {
    "ppi30_ipt_dir_size": ((IPT, DIR, SIZE), 30),
    "ppi30_dir_size": ((DIR, SIZE), 30),
    "ppi10_dir_size": ((DIR, SIZE), 10),
}
LENGTH_BINS = [0, 4, 9, 29, 30]
LENGTH_LABELS = ["1-4", "5-9", "10-29", "30"]
COLUMNS = ["APP", "PPI", "PPI_LEN", "TIME_FIRST"]


def week_dates(dataset: CESNET_TLS_Year22, weeks) -> list[str]:
    dates: list[str] = []
    for week in weeks:
        dates.extend(dataset.time_periods[f"W-2022-{week}"])
    return dates


def load_frames(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = CESNET_TLS_Year22(str(args.data_root / "CESNET-TLS-Year22"), size=args.size, silent=True)
    start, end = args.window
    config = DatasetConfig(
        dataset=dataset,
        train_period_name=f"W-2022-{start}-{end}",
        train_dates=week_dates(dataset, range(start, end + 1)),
        test_period_name=f"W-2022-{end + 1}-52",
        test_dates=week_dates(dataset, range(end + 1, 53)),
        need_val_set=False,
        train_size=args.train_size,
        test_known_size=args.test_size,
        return_other_fields=True,  # TIME_FIRST assigns test flows to weeks
        use_packet_histograms=False,
        use_tcp_features=False,
        disable_label_encoding=True,
        train_workers=0,
        test_workers=0,
        val_workers=0,
    )
    dataset.set_dataset_config_and_initialize(config)
    train = dataset.get_train_df(flatten_ppi=False)[COLUMNS].reset_index(drop=True)
    gc.collect()
    test = dataset.get_test_df(flatten_ppi=False)[COLUMNS].reset_index(drop=True)
    gc.collect()
    return train, test


def sequence_keys(ppi: np.ndarray, channels, length: int) -> np.ndarray:
    block = np.ascontiguousarray(ppi[:, list(channels), :length])
    return np.array([hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest() for row in block])


def duplicate_flags(train_keys, train_apps, test_keys, test_apps) -> tuple[pd.DataFrame, float, float]:
    pairs = pd.DataFrame({"key": train_keys, "app": train_apps}).drop_duplicates()
    labels_per_key = pairs.groupby("key").size()
    test = pd.DataFrame({"key": test_keys, "app": test_apps})
    test["in_train"] = test["key"].isin(labels_per_key.index)
    pair_seen = test.merge(pairs.assign(pair_in_train=True), on=["key", "app"], how="left")["pair_in_train"]
    test["label_absent"] = test["in_train"] & pair_seen.isna().to_numpy()
    train_dup_rate = float(pd.Series(train_keys).duplicated(keep=False).mean())
    train_ambiguous = float((labels_per_key.reindex(train_keys).to_numpy() > 1).mean())
    return test, train_dup_rate, train_ambiguous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--size", default="XS", choices=["XS", "S", "M", "L"])
    parser.add_argument("--window", default="11-14", help="training weeks, start-end")
    parser.add_argument("--train-size", default="300000", help="'all' or an integer (sampled from the window)")
    parser.add_argument("--test-size", type=int, default=150_000, help="flows sampled from all later weeks")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "week1")
    args = parser.parse_args()
    args.window = tuple(int(x) for x in args.window.split("-"))
    if args.train_size != "all":
        args.train_size = int(args.train_size)
    args.out.mkdir(parents=True, exist_ok=True)
    start, end = args.window
    tag = f"w{start}_{end}"

    t0 = time.time()
    train, test = load_frames(args)
    print(f"Loaded train {len(train):,} and test {len(test):,} flows ({args.size}) in {time.time() - t0:.0f}s", flush=True)

    train_ppi = np.stack(train.pop("PPI").to_numpy())
    test_ppi = np.stack(test.pop("PPI").to_numpy())
    gc.collect()
    assert set(np.unique(train_ppi[:, DIR, :])) <= {-1.0, 0.0, 1.0}, "unexpected PPI channel order"
    test_week = (
        pd.to_datetime(test["TIME_FIRST"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Prague").dt.isocalendar().week.to_numpy()
    )
    length_bin = pd.cut(test["PPI_LEN"], bins=LENGTH_BINS, labels=LENGTH_LABELS).astype(str).to_numpy()

    weekly_rows, length_rows = [], []
    summary = [f"Duplicate check: train = random {len(train):,} flows from weeks {start}-{end}, "
               f"test = random {len(test):,} flows from weeks {end + 1}-52, size {args.size}. "
               "Rates are lower bounds (training sample, not full set)."]
    for name, (channels, length) in KEYS.items():
        t1 = time.time()
        flags, train_dup, train_amb = duplicate_flags(
            sequence_keys(train_ppi, channels, length), train["APP"].to_numpy(),
            sequence_keys(test_ppi, channels, length), test["APP"].to_numpy(),
        )
        flags["week"] = test_week
        flags["length_bin"] = length_bin
        per_week = flags.groupby("week").agg(n=("in_train", "size"), dup_rate=("in_train", "mean"),
                                             label_absent_rate=("label_absent", "mean")).reset_index()
        per_week.insert(0, "key", name)
        overall = {"key": name, "week": "all", "n": len(flags), "dup_rate": flags["in_train"].mean(),
                   "label_absent_rate": flags["label_absent"].mean()}
        weekly_rows.append(pd.concat([per_week, pd.DataFrame([overall])], ignore_index=True))
        per_len = flags.groupby("length_bin").agg(n=("in_train", "size"), dup_rate=("in_train", "mean"),
                                                  label_absent_rate=("label_absent", "mean")).reset_index()
        per_len["length_bin"] = pd.Categorical(per_len["length_bin"], categories=LENGTH_LABELS, ordered=True)
        per_len = per_len.sort_values("length_bin")
        per_len.insert(0, "key", name)
        length_rows.append(per_len)

        summary.append(
            f"\n[{name}]  test dup rate {overall['dup_rate']:.1%}; label-absent floor {overall['label_absent_rate']:.2%}; "
            f"train internal dup {train_dup:.1%}; train flows with ambiguous labels {train_amb:.2%}  ({time.time() - t1:.0f}s)"
        )
        first, last = per_week.iloc[0], per_week.iloc[-1]
        summary.append(f"    first test week {first['week']}: dup {first['dup_rate']:.1%}; "
                       f"last test week {last['week']}: dup {last['dup_rate']:.1%}")
        for _, r in per_len.iterrows():
            summary.append(f"    PPI_LEN {r['length_bin']:>5}: n={int(r['n']):>7,}  dup {r['dup_rate']:.1%}  "
                           f"label-absent {r['label_absent_rate']:.2%}")

    pd.concat(weekly_rows, ignore_index=True).to_csv(args.out / f"duplicates_{tag}.csv", index=False)
    pd.concat(length_rows, ignore_index=True).to_csv(args.out / f"duplicates_by_length_{tag}.csv", index=False)
    text = "\n".join(summary) + "\n"
    (args.out / f"duplicates_{tag}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
