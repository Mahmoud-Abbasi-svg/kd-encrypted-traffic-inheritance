"""Freeze the known / unknown service lists used by every experiment (input to the pre-registration).

Rules, fixed before any model is trained (study plan, section 4.3):
1. A service is *eligible* if it is usable (>= 5,000 full-dataset flows/week) in all three training
   windows and in at least --min-usable-share of the valid study weeks. Services that are never
   usable, that only emerge after weeks 11-14, or that are not eligible are disabled.
2. *Far* unknowns: whole categories are held out. Among categories with at least 2 eligible services
   and at most --far-max-category services, --far-val categories go to validation and --far-test to
   test. Far categories contain no known service.
3. *Near* unknowns: among the remaining categories with at least --near-min-category eligible
   services, one service per category is held out, alternating test and validation until the quotas
   are met. Each such category keeps at least three known services.
4. All other eligible services are known. Validation and test unknowns are disjoint by construction.

The output file is treated as frozen: the script refuses to overwrite it without --force.

Inputs: results/week1/service_appearance.csv and weekly_totals.csv (scripts/01_service_appearance.py).
Output: configs/splits.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kdtraffic.splits import Splits  # noqa: E402

STUDY_FIRST_WEEK = 11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    week1 = PROJECT_ROOT / "results" / "week1"
    parser.add_argument("--appearance", type=Path, default=week1 / "service_appearance.csv")
    parser.add_argument("--weekly-totals", type=Path, default=week1 / "weekly_totals.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "configs" / "splits.json")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--min-usable-share", type=float, default=0.9)
    parser.add_argument("--far-val", type=int, default=2)
    parser.add_argument("--far-test", type=int, default=2)
    parser.add_argument("--far-max-category", type=int, default=6)
    parser.add_argument("--near-val", type=int, default=5)
    parser.add_argument("--near-test", type=int, default=8)
    parser.add_argument("--near-min-category", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="overwrite an existing (frozen) splits file")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists and is treated as frozen. Use --force only before the "
                         "pre-registration is frozen.")

    table = pd.read_csv(args.appearance)
    totals = pd.read_csv(args.weekly_totals)
    parts = totals["week"].str.extract(r"(\d{4})-W(\d{2})").astype(int)
    valid_weeks = int(((parts[0] == 2022) & (parts[1] >= STUDY_FIRST_WEEK) & ~totals["low_coverage"].astype(bool)).sum())
    window_columns = [c for c in table.columns if re.fullmatch(r"usable_w\d+_\d+", c)]
    table["eligible"] = table[window_columns].all(axis=1) & (table["weeks_usable"] >= args.min_usable_share * valid_weeks)
    rng = np.random.default_rng(args.seed)

    reasons: dict[str, str] = {}
    for _, row in table.iterrows():
        if pd.isna(row["first_usable_week"]):
            reasons[row["service"]] = "never usable in the study period"
        elif row.get("emerging_w11_14", False):
            reasons[row["service"]] = "emerging after weeks 11-14 (reserved for the natural-emergence test)"
        elif not row["eligible"]:
            reasons[row["service"]] = "not usable in every training window or in enough study weeks"

    eligible = table[table["eligible"]]
    by_category = eligible.groupby("category")["service"].apply(lambda s: sorted(s))
    category_sizes = table.groupby("category")["service"].size()

    far_pool = sorted(c for c, services in by_category.items()
                      if len(services) >= 2 and category_sizes[c] <= args.far_max_category)
    needed = args.far_val + args.far_test
    if len(far_pool) < needed:
        raise SystemExit(f"Only {len(far_pool)} categories qualify for far unknowns; need {needed}")
    chosen = [str(c) for c in rng.permutation(far_pool)[:needed]]
    far_val_categories, far_test_categories = sorted(chosen[:args.far_val]), sorted(chosen[args.far_val:])
    val_far = sorted(s for c in far_val_categories for s in by_category[c])
    test_far = sorted(s for c in far_test_categories for s in by_category[c])

    near_pool = [str(c) for c in rng.permutation(sorted(set(by_category.index) - set(chosen)))
                 if len(by_category[c]) >= args.near_min_category]
    val_near: list[str] = []
    test_near: list[str] = []
    turn_test = True
    for category in near_pool:
        if len(test_near) >= args.near_test and len(val_near) >= args.near_val:
            break
        service = str(rng.choice(by_category[category]))
        if (turn_test and len(test_near) < args.near_test) or len(val_near) >= args.near_val:
            test_near.append(service)
        else:
            val_near.append(service)
        turn_test = not turn_test
    if len(test_near) < args.near_test or len(val_near) < args.near_val:
        print(f"WARNING: near-unknown quotas not met (val {len(val_near)}/{args.near_val}, "
              f"test {len(test_near)}/{args.near_test}); only {len(near_pool)} categories qualify")
    val_near, test_near = sorted(val_near), sorted(test_near)

    unknown = set(val_far + test_far + val_near + test_near)
    known = sorted(set(eligible["service"]) - unknown)
    disabled = sorted(set(table["service"]) - set(known) - unknown)
    unexplained = [s for s in disabled if s not in reasons]
    if unexplained:
        raise AssertionError(f"Disabled services without a reason: {unexplained}")
    categories = dict(zip(table["service"], table["category"]))

    splits = Splits(known, val_near, val_far, test_near, test_far, disabled, categories)
    splits.validate(available=list(table["service"]))

    flows = table.set_index("service")["total_flows_study"]
    payload = {
        "description": "Known / unknown CESNET-TLS-Year22 services for all experiments; frozen for the pre-registration.",
        "generated_by": "scripts/03_make_splits.py",
        "rules": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "force"},
        "valid_study_weeks": valid_weeks,
        "known": known,
        "val_unknown": {"near": val_near, "far": val_far},
        "test_unknown": {"near": test_near, "far": test_far},
        "far_categories": {"val": far_val_categories, "test": far_test_categories},
        "disabled": disabled,
        "disabled_reasons": {s: reasons[s] for s in disabled},
        "emerging_w11_14": sorted(table.loc[table.get("emerging_w11_14", False) == True, "service"]),  # noqa: E712
        "categories": categories,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    total = flows.sum()
    print(f"Wrote {args.out}")
    print(f"Known: {len(known)} services ({flows[known].sum() / total:.1%} of study-period flows) "
          f"in {len({categories[s] for s in known})} categories")
    print(f"Validation unknown: near {len(val_near)} {val_near}")
    print(f"                    far  {len(val_far)} {val_far} (categories {far_val_categories})")
    print(f"Test unknown:       near {len(test_near)} {test_near}")
    print(f"                    far  {len(test_far)} {test_far} (categories {far_test_categories})")
    print(f"Disabled: {len(disabled)}")
    for reason in sorted(set(reasons[s] for s in disabled)):
        members = [s for s in disabled if reasons[s] == reason]
        print(f"    {len(members):>3} {reason}")


if __name__ == "__main__":
    main()
